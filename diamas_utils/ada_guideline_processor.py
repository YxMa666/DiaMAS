#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ADA指南处理器（版本自适应）+ 通用多源指南系统 + 混合检索集成

修改历史：
  [v1-v7] 原始版本（见 git log）
  [v8]    三层改造：
    Layer-1: 药物关键词从 config.py 动态生成（单一真相）
             ADA 年份扩展至 2006-2026（全21年）
             中文指南文件级专科路由
             cn_hypoglycemia_consensus → endocrine_society（修正归属错误）
    Layer-2: BGE-M3 + BM25 + RRF 混合检索（HybridRetriever 集成）
    Layer-3: BGE-Reranker-v2-M3 cross-encoder 精排（可选）

作者：Yuxi
"""

import json
import logging
import pickle
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── 从 config.py 导入单一真相来源的药物列表 ──────────────────────────────────
from agents.config import DIABETES_MEDICATIONS, COMPLICATION_DRUGS
_CONFIG_LOADED = True

try:  # dense/hybrid retrieval was not used in the study; lexical retrieval needs no extra packages
    from diamas_utils.hybrid_retriever import HybridRetriever, build_retriever_from_chunks
    _HYBRID_AVAILABLE = True
except ImportError:
    HybridRetriever = build_retriever_from_chunks = None
    _HYBRID_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# § 0  动态药物关键词构建（Layer-1 核心：从 config.py 生成，不再硬编码）
# ═══════════════════════════════════════════════════════════════════════════════

# 药物类别 → 主题标签映射（类别名前缀 → 统一 tag）
_CATEGORY_TO_TAG: Dict[str, str] = {
    "Biguanides":              "metformin",
    "SGLT2i":                  "sglt2",
    "GLP-1RA":                 "glp1",
    "GIP_GLP1_RA":             "tirzepatide",
    "GCG_GLP1_RA":             "glp1",
    "GLP1_Multimer_RA":        "glp1",
    "DPP-4i":                  "dpp4",
    "Sulfonylureas":           "sulfonylurea",
    "Meglitinides":            "meglitinide",
    "Thiazolidinediones":      "thiazolidinedione",
    "Insulins_Basal":          "insulin",
    "Insulins_Rapid":          "insulin",
    "Insulins_Others":         "insulin",
    "Amylin_Analogs":          "insulin",
    "CKD_MRA":                 "finerenone",
    "Hypertension_ACEi":       "ckd_treatment",
    "Hypertension_ARB":        "ckd_treatment",
    "Hypertension_CCB":        "cvd_treatment",
    "Hypertension_Diuretics":  "heart_failure",
    "Hypertension_BetaBlockers":"heart_failure",
    "Lipid_Statins":           "cvd_treatment",
    "Lipid_NonStatins":        "cvd_treatment",
    "Antiplatelet":            "cvd_treatment",
    "Neuropathy_Gabapentinoids":"hypoglycemia_risk",
    "Neuropathy_SNRI":         "hypoglycemia_risk",
    "Neuropathy_TCA":          "hypoglycemia_risk",
    "Neuropathy_Others":       "hypoglycemia_risk",
    "Gastroparesis":           "hypoglycemia_risk",
    "Hypoglycemia_Rescue":     "hypoglycemia_risk",
}

# 中文药物别名词典（药物英文名 → 常见中文名列表）
_ZH_DRUG_ALIASES: Dict[str, List[str]] = {
    "Metformin":         ["二甲双胍"],
    "Empagliflozin":     ["恩格列净"],
    "Dapagliflozin":     ["达格列净"],
    "Canagliflozin":     ["卡格列净"],
    "Ertugliflozin":     ["艾托格列净"],
    "Liraglutide":       ["利拉鲁肽"],
    "Semaglutide":       ["司美格鲁肽", "索马鲁肽"],
    "Dulaglutide":       ["度拉糖肽"],
    "Exenatide":         ["艾塞那肽", "依克那肽"],
    "Tirzepatide":       ["替尔泊肽"],
    "Sitagliptin":       ["西格列汀"],
    "Saxagliptin":       ["沙格列汀"],
    "Linagliptin":       ["利格列汀"],
    "Alogliptin":        ["阿格列汀"],
    "Glipizide":         ["格列吡嗪"],
    "Glyburide":         ["格列本脲", "优降糖"],
    "Glimepiride":       ["格列美脲"],
    "Pioglitazone":      ["吡格列酮"],
    "Finerenone":        ["非奈利酮"],
    "Eplerenone":        ["依普利酮"],
    "Lisinopril":        ["赖诺普利"],
    "Enalapril":         ["依那普利"],
    "Losartan":          ["氯沙坦"],
    "Valsartan":         ["缬沙坦"],
    "Atorvastatin":      ["阿托伐他汀"],
    "Rosuvastatin":      ["瑞舒伐他汀"],
    "Aspirin":           ["阿司匹林"],
    "Furosemide":        ["呋塞米", "速尿"],
    "Spironolactone":    ["螺内酯"],
    "Metoprolol":        ["美托洛尔"],
    "Carvedilol":        ["卡维地洛"],
    "Acarbose":          ["阿卡波糖"],
    "Pregabalin":        ["普瑞巴林"],
    "Duloxetine":        ["度洛西汀"],
}


def _drug_to_regex(drug_name: str) -> str:
    """将药物名转换为宽松正则（忽略大小写，允许连字符/空格变体）。"""
    escaped = re.escape(drug_name.lower())
    # 允许连字符和空格互换（如 "insulin glargine" / "insulin-glargine"）
    return escaped.replace(r"\ ", r"[\s\-]?")


def _build_drug_kw_patterns() -> List[str]:
    """
    [Layer-1] 从 config.py 的 DIABETES_MEDICATIONS + COMPLICATION_DRUGS
    动态生成药物关键词正则列表，用于 _extract_pharmacotherapy_sections()。

    规则：每个药物名生成一条正则；同时加入中文别名。
    """
    patterns: List[str] = []
    all_drugs: List[str] = []

    for drugs in DIABETES_MEDICATIONS.values():
        all_drugs.extend(drugs)
    for drugs in COMPLICATION_DRUGS.values():
        all_drugs.extend(drugs)

    seen = set()
    for drug in all_drugs:
        key = drug.lower()
        if key not in seen:
            seen.add(key)
            patterns.append(_drug_to_regex(drug))
            # 中文别名
            for zh in _ZH_DRUG_ALIASES.get(drug, []):
                patterns.append(re.escape(zh))

    return patterns


def _build_drug_tag_rules() -> Dict[str, List[str]]:
    """
    [Layer-1] 从 config.py 动态生成 topic_tag → [regex_patterns] 映射，
    用于 _infer_topic_tags()。

    与硬编码版本相比：
      - 药物名直接来自 config.py（跟随 DIABETES_MEDICATIONS / COMPLICATION_DRUGS 变化）
      - 保留手工定义的非药物 tag（glycemic_target、ckd_treatment 等）
      - 自动补入中文别名
    """
    # 基础非药物 tag（固定）
    base_rules: Dict[str, List[str]] = {
        "glycemic_target":    [r"\ba1c\b", r"\bhba1c\b", r"\bglycemic\s+(goal|target)\b",
                               r"糖化血红蛋白", r"血糖目标", r"糖化"],
        "ckd_treatment":      [r"\bckd\b", r"\begfr\b", r"\brenal\b", r"\bnephropath\b",
                               r"慢性肾脏病", r"肾功能", r"肾小球滤过率", r"糖尿病肾病",
                               r"蛋白尿", r"albuminuria", r"proteinuria"],
        "cvd_treatment":      [r"\bascvd\b", r"\bcardiovascula\b", r"\bcvot\b",
                               r"心血管", r"动脉粥样硬化", r"atherosclerosis"],
        "heart_failure":      [r"\bheart\s+failure\b", r"\bhfref\b", r"\bhfpef\b",
                               r"心力衰竭", r"心衰"],
        "hypoglycemia_risk":  [r"\bhypoglycemi\b", r"低血糖"],
        "renal_dosing":       [r"\bdose\s+adj", r"\bcontraindica\b", r"\brenal\s+impair\b",
                               r"剂量调整", r"肾功能损害", r"禁忌"],
        "weight_management":  [r"\bweight\b", r"\bobesit\b", r"体重", r"肥胖"],
        "combination_therapy":[r"\bcombination\s+ther", r"\bdual\s+therap", r"联合治疗"],
        "treatment_algorithm":[r"\balgorithm\b", r"\binitial\s+therap\b", r"\bstep.?wise\b",
                               r"治疗路径", r"流程", r"分层"],
        "t1dm_treatment":     [r"\btype\s*1\b", r"\bt1dm\b", r"1型糖尿病"],
        "t2dm_treatment":     [r"\btype\s*2\b", r"\bt2dm\b", r"2型糖尿病"],
        "inpatient":          [r"\binpatient\b", r"\bhospitalized\b", r"\bcritical\s+ill",
                               r"住院", r"重症", r"ICU"],
    }

    # 从 config.py 动态构建药物 tag
    # 按 _CATEGORY_TO_TAG 聚合：同一 tag 的多个类别合并
    tag_patterns: Dict[str, List[str]] = {}
    for cat, drugs in {**DIABETES_MEDICATIONS, **COMPLICATION_DRUGS}.items():
        tag = _CATEGORY_TO_TAG.get(cat)
        if tag is None:
            continue
        if tag not in tag_patterns:
            tag_patterns[tag] = []
        for drug in drugs:
            tag_patterns[tag].append(rf"\b{_drug_to_regex(drug)}\b")
            for zh in _ZH_DRUG_ALIASES.get(drug, []):
                tag_patterns[tag].append(re.escape(zh))

    # 特殊类别正则补充（类别名本身作为关键词）
    class_kws: Dict[str, List[str]] = {
        "sglt2":         [r"\bsglt.?2\b", r"sglt2抑制剂", r"格列净", r"钠.葡萄糖共转运"],
        "glp1":          [r"\bglp.?1\b", r"glp.1受体激动剂", r"胰高糖素样肽"],
        "dpp4":          [r"\bdpp.?4\b", r"dpp.4抑制剂"],
        "insulin":       [r"\binsulin\b", r"胰岛素", r"\bbasal\s+insulin\b", r"\bbolus\s+insulin\b"],
        "sulfonylurea":  [r"\bsulfonylurea\b", r"磺脲类"],
        "thiazolidinedione": [r"\bthiazolidinedione\b", r"\btzd\b", r"噻唑烷二酮"],
        "finerenone":    [r"\bfigaro\b", r"\bfidelio\b"],  # 临床试验名
        "tirzepatide":   [r"\bgip.?glp.?1\b", r"\bdual.?agonist\b", r"\btwincretin\b"],
        "meglitinide":   [r"\bmeglitinide\b"],
        "acarbose":      [r"\bacarbose\b", r"阿卡波糖", r"α.葡萄糖苷酶抑制剂"],
    }
    for tag, kws in class_kws.items():
        if tag not in tag_patterns:
            tag_patterns[tag] = []
        tag_patterns[tag].extend(kws)

    # 合并
    merged = {**base_rules}
    for tag, pats in tag_patterns.items():
        if tag in merged:
            merged[tag] = merged[tag] + pats
        else:
            merged[tag] = pats

    return merged


# 模块加载时预计算一次，避免每次 parse 都重建
_DRUG_KW_PATTERNS:  List[str]              = _build_drug_kw_patterns()
_DRUG_TAG_RULES:    Dict[str, List[str]]   = _build_drug_tag_rules()


# ═══════════════════════════════════════════════════════════════════════════════
# § 1  ADA 版本元数据（2006-2026 全覆盖）
# ═══════════════════════════════════════════════════════════════════════════════

def _ada_meta_for_year(year: int) -> Dict:
    """
    [Layer-1] 为任意 ADA 年份（2006-2026）生成版本元数据。

    ADA 指南格式演变：
      ≤2014 : 全文整合式（integrated=True），药物治疗散落在多个子节
      2015-2016 : Section 7 独立章节
      2017-2018 : Section 8 独立章节（EMPA-REG/LEADER 证据首次纳入）
      ≥2019  : Section 9 独立章节，CVOT 心肾保护全面整合
    """
    if year <= 2014:
        return {
            "section_num":  None,
            "title_kws":    [r"pharmacolog", r"antihyperglycemic",
                             r"glycemic\s+treatment", r"glucose.lowering",
                             r"oral\s+agent",         r"insulin\s+therap"],
            "integrated":   True,
            "cvot_era":     False,
            "description":  f"ADA {year}: 全文整合式，药物治疗在 Pharmacologic/Management 子节",
        }
    elif year <= 2016:
        return {
            "section_num":  "7",
            "title_kws":    [r"approaches\s+to\s+glycemic\s+treatment",
                             r"pharmacolog", r"glycemic\s+management",
                             r"section\s+7"],
            "integrated":   False,
            "cvot_era":     False,
            "description":  f"ADA {year}: Section 7 - Approaches to Glycemic Treatment",
        }
    elif year <= 2018:
        return {
            "section_num":  "8",
            "title_kws":    [r"pharmacologic\s+approaches\s+to\s+glycemic",
                             r"section\s+8"],
            "integrated":   False,
            "cvot_era":     True,
            "description":  f"ADA {year}: Section 8（EMPA-REG/LEADER 证据首次纳入）",
            "landmark_trials": ["EMPA-REG", "LEADER", "SUSTAIN-6", "CANVAS"],
        }
    else:  # ≥2019
        extra_kws = []
        trials = ["CREDENCE", "DECLARE", "DAPA-HF", "EMPEROR-Reduced",
                  "EMPEROR-Preserved", "DELIVER", "FLOW", "SURMOUNT"]
        if year >= 2022:
            extra_kws.append(r"tirzepatide")
        if year >= 2024:
            extra_kws.extend([r"finerenone", r"tirzepatide"])
        if year >= 2026:
            extra_kws.extend([r"retatrutide|orforglipron|cotadutide"])
        return {
            "section_num":  "9",
            "title_kws":    [r"pharmacologic\s+approaches\s+to\s+glycemic",
                             r"section\s+9"] + extra_kws,
            "integrated":   False,
            "cvot_era":     True,
            "description":  f"ADA {year}: Section 9，心肾全面保护时代",
            "landmark_trials": [t for t in trials],
        }


# 显式元数据表（覆盖自动生成，保留向后兼容的特殊标注）
ADA_VERSION_META: Dict[int, Dict] = {
    yr: _ada_meta_for_year(yr)
    for yr in range(2006, 2027)
}


# ═══════════════════════════════════════════════════════════════════════════════
# § 2  正则模式
# ═══════════════════════════════════════════════════════════════════════════════

RECOMMENDATION_LEVEL_PATTERN = re.compile(
    r'\(([ABCE])\)'
    r'|\bLevel\s+([ABCE])\b'
    r'|\b([ABCE])\b(?=\s*[·•]|\s*$)',
    re.MULTILINE
)

SECTION_HEADER_PATTERN = re.compile(
    r'^(#{1,4})\s*((?:\d+(?:\.\d+)*\.?\s+)?[A-Z].{3,120})$',
    re.MULTILINE
)


# ═══════════════════════════════════════════════════════════════════════════════
# § 3  数据结构
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ADAChunk:
    """ADA 指南的单个文本块"""
    chunk_id:             str
    ada_year:             int
    section_number:       Optional[str]
    section_title:        str
    text:                 str
    recommendation_level: Optional[str]
    topic_tags:           List[str]
    is_recommendation:    bool
    is_whats_new:         bool
    source_page:          Optional[int]
    char_count:           int = field(init=False)

    def __post_init__(self):
        self.char_count = len(self.text)


# ═══════════════════════════════════════════════════════════════════════════════
# § 4  ADA 文本解析器
# ═══════════════════════════════════════════════════════════════════════════════

class ADATextParser:
    """
    将 MinerU 转换的 Markdown 文本解析为 ADAChunk 列表。

    版本适配策略：
    - integrated=True（≤2014）：先定位药物治疗段落，再按段落切分
    - integrated=False（≥2015）：整个章节 Markdown 按标题层级切分
    """

    MIN_CHUNK_CHARS = 80
    MAX_CHUNK_CHARS = 3000

    def __init__(self, year: int):
        self.year = year
        self.meta = ADA_VERSION_META.get(year, _ada_meta_for_year(year))

    def parse(self, markdown_text: str) -> List[ADAChunk]:
        extracted = self._extract_pharmacotherapy_sections(markdown_text)
        coverage  = len(extracted) / max(len(markdown_text), 1)
        relevant_text = markdown_text if coverage >= 0.60 else extracted
        return self._split_into_chunks(relevant_text)

    def _extract_pharmacotherapy_sections(self, text: str) -> str:
        """
        [Layer-1 改造] 从 ADA Standards 中提取药物治疗相关段落。

        关键词来源：
          ① _DRUG_KW_PATTERNS：从 config.py 动态生成的药物名正则
          ② 固定临床关键词（推荐等级格式、治疗决策动词、并发症、血糖目标等）
        """
        paragraphs = re.split(r'\n{2,}', text)

        # ── ① 动态药物关键词（来自 config.py）────────────────────────────────
        dynamic_pats = [re.compile(p, re.IGNORECASE) for p in _DRUG_KW_PATTERNS]

        # ── ② 固定临床关键词（不依赖药物列表）───────────────────────────────
        fixed_kws = [
            # 药物类别通用术语
            r"pharmacolog", r"antihyperglycemic", r"glycemic\s+treatment",
            r"glucose.lowering", r"oral\s+agent", r"insulin\s+therap",
            r"sglt.?2", r"glp.?1", r"dpp.?4",
            r"sulfonylurea", r"thiazolidinedione", r"biguanide",
            r"basal\s+insulin", r"bolus\s+insulin",
            # 推荐等级 / 指南格式
            r"\([ABCE]\)", r"\bgrade\s+[ABCE]\b", r"\blevel\s+[ABCE]\b",
            r"recommendation", r"preferred\s+agent", r"first.?line",
            r"contraindicated|contraindication",
            # 治疗决策动词
            r"\binitiat", r"\bintensif", r"add.?on\s+therap",
            r"combination\s+therap", r"treatment\s+algorithm",
            # CVOT / 心肾保护（2016+）
            r"cardiovascular\s+outcome",
            r"cvot|empa.reg|leader|canvas|credence|declare",
            r"heart\s+failure", r"renal\s+protection|kidney\s+protection",
            r"albuminuria|proteinuria",
            # 肾功能剂量调整
            r"\begfr\b", r"renal\s+impairment|renal\s+function",
            r"dose\s+adjust|dose\s+reduc", r"ckd|chronic\s+kidney",
            # 住院 / ICU
            r"inpatient|hospitalized", r"corticosteroid|steroid.induced",
            r"critical\s+illness",
            # 血糖目标
            r"\bhba1c\b|\ba1c\b", r"glycemic\s+(goal|target)",
            r"hypoglycemia\s+(risk|avoidance|prevention)",
            # 体重 / 肥胖
            r"weight\s+(loss|reduction|management)", r"obesity",
            # 中文通用
            r"药物治疗", r"降糖", r"血糖控制",
        ]
        fixed_pats = [re.compile(kw, re.IGNORECASE) for kw in fixed_kws]
        all_pats   = dynamic_pats + fixed_pats

        relevant_indices = set()
        for i, para in enumerate(paragraphs):
            if any(p.search(para) for p in all_pats):
                for j in range(max(0, i - 2), min(len(paragraphs), i + 3)):
                    relevant_indices.add(j)

        if not relevant_indices:
            logger.warning(f"ADA {self.year}: 未找到药物治疗相关段落，使用全文")
            return text

        sorted_indices = sorted(relevant_indices)
        extracted = "\n\n".join(paragraphs[i] for i in sorted_indices)
        logger.info(
            f"ADA {self.year}: 从 {len(paragraphs)} 段中提取 "
            f"{len(sorted_indices)} 段（覆盖率 {len(extracted)/max(len(text),1):.1%}）"
        )
        return extracted

    def _split_into_chunks(self, text: str) -> List[ADAChunk]:
        chunks  = []
        headers = list(SECTION_HEADER_PATTERN.finditer(text))

        if not headers:
            return self._split_by_paragraphs(text)

        for idx, header_match in enumerate(headers):
            section_title = header_match.group(2).strip()
            start         = header_match.end()
            end           = headers[idx + 1].start() if idx + 1 < len(headers) else len(text)
            section_text  = text[start:end].strip()

            if len(section_text) < self.MIN_CHUNK_CHARS:
                continue

            m           = re.match(r'^(\d+(?:\.\d+)*)', section_title)
            section_num = m.group(1) if m else self.meta.get("section_num")

            if len(section_text) > self.MAX_CHUNK_CHARS:
                chunks.extend(self._split_long_section(
                    section_text, section_title, section_num,
                    global_offset=len(chunks)
                ))
            else:
                chunk = self._make_chunk(section_text, section_title, section_num, len(chunks))
                if chunk:
                    chunks.append(chunk)

        logger.info(f"ADA {self.year}: 解析得到 {len(chunks)} 个 chunks")
        return chunks

    def _split_by_paragraphs(self, text: str) -> List[ADAChunk]:
        chunks     = []
        paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
        buffer     = ""
        for para in paragraphs:
            buffer += ("\n\n" + para if buffer else para)
            if len(buffer) >= self.MIN_CHUNK_CHARS * 3:
                chunk = self._make_chunk(buffer, "Pharmacologic Treatment",
                                         self.meta.get("section_num"), len(chunks))
                if chunk:
                    chunks.append(chunk)
                buffer = ""
        if buffer and len(buffer) >= self.MIN_CHUNK_CHARS:
            chunk = self._make_chunk(buffer, "Pharmacologic Treatment",
                                     self.meta.get("section_num"), len(chunks))
            if chunk:
                chunks.append(chunk)
        return chunks

    def _split_long_section(self, text, section_title, section_num,
                             global_offset: int = 0) -> List[ADAChunk]:
        sub_chunks = []
        paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
        buffer     = ""
        for para in paragraphs:
            if len(buffer) + len(para) > self.MAX_CHUNK_CHARS and buffer:
                chunk = self._make_chunk(buffer, section_title, section_num,
                                         global_offset + len(sub_chunks))
                if chunk:
                    sub_chunks.append(chunk)
                buffer = para
            else:
                buffer += ("\n\n" + para if buffer else para)
        if buffer and len(buffer) >= self.MIN_CHUNK_CHARS:
            chunk = self._make_chunk(buffer, section_title, section_num,
                                     global_offset + len(sub_chunks))
            if chunk:
                sub_chunks.append(chunk)
        return sub_chunks

    def _make_chunk(self, text, section_title, section_num, idx) -> Optional[ADAChunk]:
        if len(text) < self.MIN_CHUNK_CHARS:
            return None
        return ADAChunk(
            chunk_id             = f"ada{self.year}_s{section_num or 'x'}_{idx:04d}",
            ada_year             = self.year,
            section_number       = section_num,
            section_title        = section_title,
            text                 = text,
            recommendation_level = self._extract_recommendation_level(text),
            topic_tags           = self._infer_topic_tags(text, section_title),
            is_recommendation    = self._is_recommendation(text),
            is_whats_new         = self._is_whats_new(section_title),
            source_page          = None,
        )

    @staticmethod
    def _extract_recommendation_level(text: str) -> Optional[str]:
        matches = []
        for m in RECOMMENDATION_LEVEL_PATTERN.finditer(text):
            level = m.group(1) or m.group(2) or m.group(3)
            if level:
                matches.append(level)
        priority = {'A': 4, 'B': 3, 'C': 2, 'E': 1}
        return max(set(matches), key=lambda x: priority.get(x, 0)) if matches else None

    @staticmethod
    def _infer_topic_tags(text: str, title: str) -> List[str]:
        """[Layer-1 改造] 使用 _DRUG_TAG_RULES（由 config.py 动态生成）打标签。"""
        combined = (text + " " + title).lower()
        return [tag for tag, patterns in _DRUG_TAG_RULES.items()
                if any(re.search(p, combined) for p in patterns)]

    @staticmethod
    def _is_recommendation(text: str) -> bool:
        text_lower = text.lower()
        negation_signals = [
            "not recommended", "is not advised", "are not advised",
            "not be recommended", "should not be used",
            "suggest against", "do not suggest", "not suggest",
            "recommend against", "do not recommend",
            "不推荐", "不适用", "禁忌", "不建议", "反对使用", "避免使用",
        ]
        if any(s in text_lower for s in negation_signals):
            return bool(re.search(r'\([ABCE]\)', text))

        positive_patterns = [
            r'\([ABCE]\)', r'\bLevel\s+[ABCE]\b',
            r'\bGrade\s+[12][ABCD]\b', r'\bClass\s+I{1,3}[ABab]?\b',
            r'\b(?:is|are|should\s+be)\s+recommended\b', r'\brecommend\b',
            r'\bsuggest\b(?!\s+against)', r'\bshould\b.*\([ABCE]\)',
            r'\bconsider\b.*\([ABCE]\)',
            r'强[烈力]推荐', r'弱推荐|条件性推荐|有条件推荐',
            r'推荐.*(?:I类|A级|1级)', r'建议.*(?:II类|B级|2级)',
        ]
        return any(re.search(p, text, re.MULTILINE | re.IGNORECASE)
                   for p in positive_patterns)

    @staticmethod
    def _is_whats_new(title: str) -> bool:
        return bool(re.search(r"what'?s?\s+new", title, re.IGNORECASE))


# ═══════════════════════════════════════════════════════════════════════════════
# § 5  ADA 索引（集成混合检索）
# ═══════════════════════════════════════════════════════════════════════════════

class ADAGuidelineIndex:
    """
    ADA 指南多版本索引。

    Phase 1（默认）: 关键词 TF 匹配
    Phase 2（build_hybrid_index 后激活）: BGE-M3 Dense + BM25 + RRF + 可选 Reranker
    """

    def __init__(self):
        self.chunks:   List[ADAChunk]            = []
        self.by_year:  Dict[int, List[ADAChunk]] = {}
        # [Layer-2] 混合检索器（懒初始化）
        self._hybrid:     Optional[object]       = None   # HybridRetriever | None
        self._cid_to_idx: Dict[str, int]         = {}     # chunk_id → 在 _hybrid 中的位置

    def __getattr__(self, name: str):
        """
        向后兼容：旧版 pkl 反序列化后缺少新增字段时返回安全默认值，
        而非抛出 AttributeError 导致整个加载失败。
        仅当 __dict__ 中真的没有该属性时才触发（正常访问不经过此处）。
        """
        _defaults = {
            "_hybrid":     None,
            "_cid_to_idx": {},
            "chunks":      [],
            "by_year":     {},
        }
        if name in _defaults:
            # 写回 __dict__，后续访问不再走 __getattr__
            object.__setattr__(self, name, _defaults[name])
            return _defaults[name]
        raise AttributeError(f"'ADAGuidelineIndex' object has no attribute '{name}'")

    def add_chunks(self, chunks: List[ADAChunk]):
        for chunk in chunks:
            self.chunks.append(chunk)
            self.by_year.setdefault(chunk.ada_year, []).append(chunk)
        logger.info(f"ADA索引现有 {len(self.chunks)} 个 chunks，年份: {sorted(self.by_year.keys())}")

    # ── [Layer-2] 构建混合检索器 ─────────────────────────────────────────────

    def build_hybrid_index(
        self,
        embed_model:    str  = "BAAI/bge-m3",
        use_reranker:   bool = False,
        reranker_model: str  = "BAAI/bge-reranker-v2-m3",
        device:         str  = "cpu",
        use_fp16:       bool = False,
    ) -> None:
        """
        [Layer-2] 构建 BGE-M3 + BM25 混合检索索引。
        调用后，search() 自动切换到混合检索路径。

        Args:
            embed_model:    BGE-M3 模型路径或 HuggingFace ID
            use_reranker:   是否启用 cross-encoder 精排（[Layer-3]）
            reranker_model: reranker 模型路径或 HuggingFace ID
            device:         "cpu" / "cuda" / "cuda:0"
            use_fp16:       是否使用半精度（GPU 环境建议开启）
        """
        if not _HYBRID_AVAILABLE:
            logger.error("[ADAGuidelineIndex] hybrid_retriever 模块未找到，无法构建混合索引")
            return

        self._hybrid = HybridRetriever(
            embed_model=embed_model,
            use_reranker=use_reranker,
            reranker_model=reranker_model,
            device=device,
            use_fp16=use_fp16,
        )
        texts     = [c.text for c in self.chunks]
        chunk_ids = [c.chunk_id for c in self.chunks]
        self._hybrid.build(texts, chunk_ids)
        self._cid_to_idx = {cid: i for i, cid in enumerate(chunk_ids)}
        logger.info(f"[ADAGuidelineIndex] 混合索引构建完成，{len(self.chunks)} 个 chunks")

    # ── 检索主接口 ────────────────────────────────────────────────────────────

    def search(
        self,
        query:               str,
        year_filter:         Optional[int]       = None,
        min_year:            Optional[int]       = None,
        max_year:            Optional[int]       = None,
        topic_filter:        Optional[List[str]] = None,
        level_filter:        Optional[List[str]] = None,
        recommendation_only: bool                = False,
        top_k:               int                 = 5,
    ) -> List[Tuple[ADAChunk, float]]:
        """
        检索接口。
        - 若已调用 build_hybrid_index()，使用混合检索（BGE-M3 + BM25 + RRF）。
        - 否则使用关键词 TF 检索（Phase 1 兜底）。
        """
        # 元数据预过滤（两路均适用）
        candidates = self._filter_candidates(
            year_filter, min_year, max_year, topic_filter, level_filter, recommendation_only
        )
        if not candidates:
            return []

        # [Layer-2] 混合检索路径
        if self._hybrid is not None and self._hybrid.is_built:
            cand_indices = [self._cid_to_idx[c.chunk_id]
                            for c in candidates if c.chunk_id in self._cid_to_idx]
            if not cand_indices:
                return []
            hits = self._hybrid.search(
                query, candidate_indices=cand_indices, top_k=top_k,
                rerank_top_k=top_k,
            )
            cid_map = {c.chunk_id: c for c in candidates}
            results = []
            for idx, score in hits:
                cid = self._hybrid.idx_to_chunk_id(idx)
                if cid and cid in cid_map:
                    results.append((cid_map[cid], score))
            return results[:top_k]

        # Phase-1 关键词检索（兜底）
        scored = self._keyword_score(query, candidates)
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def search_for_drug_recommendation(
        self,
        patient_features: Dict,
        candidate_drugs:  Optional[List[str]] = None,
        prefer_year:      int                 = 2024,
        top_k:            int                 = 6,
    ) -> List[Tuple[ADAChunk, float]]:
        query_parts = []

        egfr = patient_features.get('egfr')
        if egfr is not None:
            if egfr < 30:
                query_parts += ["renal impairment", "eGFR", "dose adjustment", "contraindication"]
            elif egfr < 60:
                query_parts += ["CKD", "kidney", "metformin", "SGLT2"]

        if patient_features.get('has_cvd'):
            query_parts += ["cardiovascular", "ASCVD", "CVOT", "empagliflozin", "liraglutide"]
        if patient_features.get('has_hf'):
            query_parts += ["heart failure", "HFrEF", "SGLT2", "dapagliflozin"]
        if patient_features.get('has_ckd'):
            query_parts += ["chronic kidney disease", "DKD", "Finerenone"]
        if patient_features.get('hypoglycemia_risk'):
            query_parts += ["hypoglycemia", "sulfonylurea", "insulin dose"]

        if candidate_drugs:
            query_parts.extend(candidate_drugs[:4])

        dm_type = patient_features.get('dm_type', 'T2DM')
        if dm_type == 'T1DM':
            query_parts += ["type 1 diabetes", "T1DM", "insulin regimen"]
        else:
            query_parts += ["type 2 diabetes", "T2DM", "pharmacologic treatment"]

        query    = " ".join(query_parts) if query_parts else "pharmacologic treatment glycemic management"
        results: List[Tuple[ADAChunk, float]] = []

        all_years = sorted(self.by_year.keys())
        def _year_w(yr: int) -> float:
            dist = abs(yr - prefer_year)
            return max(0.5, 1.3 - dist * 0.15)

        for chunk, score in self.search(query, year_filter=prefer_year, top_k=max(top_k - 2, 3)):
            results.append((chunk, score * _year_w(chunk.ada_year)))

        seen_ids = {c.chunk_id for c, _ in results}
        if prefer_year > min(all_years, default=2010):
            for chunk, score in self.search(query, min_year=min(all_years, default=2006),
                                            max_year=prefer_year - 1, top_k=3):
                if chunk.chunk_id not in seen_ids:
                    results.append((chunk, score * _year_w(chunk.ada_year)))
                    seen_ids.add(chunk.chunk_id)

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    # ── 内部方法 ──────────────────────────────────────────────────────────────

    def _filter_candidates(self, year_filter, min_year, max_year,
                           topic_filter, level_filter, recommendation_only) -> List[ADAChunk]:
        candidates = self.chunks

        if year_filter is not None:
            candidates = [c for c in candidates if c.ada_year == year_filter]
        else:
            if min_year is not None:
                candidates = [c for c in candidates if c.ada_year >= min_year]
            if max_year is not None:
                candidates = [c for c in candidates if c.ada_year <= max_year]

        if topic_filter:
            topic_set  = set(topic_filter)
            candidates = [c for c in candidates if topic_set & set(c.topic_tags)]

        if level_filter:
            level_set  = set(level_filter)
            candidates = [c for c in candidates
                          if c.recommendation_level and c.recommendation_level in level_set]

        if recommendation_only:
            candidates = [c for c in candidates if c.is_recommendation]

        return candidates

    @staticmethod
    def _keyword_score(query: str, candidates: List[ADAChunk]) -> List[Tuple[ADAChunk, float]]:
        """Phase-1 TF 风格关键词得分 + 推荐等级奖励。"""
        query_tokens = re.findall(
            r'[\u4e00-\u9fff]{1,4}|[a-z0-9]+(?:-[a-z0-9]+)*',
            query.lower()
        )
        if not query_tokens:
            return [(c, 0.0) for c in candidates]

        scored = []
        for chunk in candidates:
            text_lower  = chunk.text.lower()
            title_lower = chunk.section_title.lower()
            tf = sum(
                text_lower.count(tok) * (1.0 + 0.5 * title_lower.count(tok))
                for tok in query_tokens
            )
            tf /= (1 + chunk.char_count / 500)
            level_bonus = {'A': 0.3, 'B': 0.2, 'C': 0.1, 'E': 0.05}.get(
                chunk.recommendation_level or '', 0
            )
            rec_bonus = 0.2 if chunk.is_recommendation else 0
            scored.append((chunk, tf + level_bonus + rec_bonus))
        return scored

    def format_for_prompt(
        self,
        chunks:       List[Tuple[ADAChunk, float]],
        max_per_year: int = 2,
        max_chars:    int = 350,
    ) -> str:
        if not chunks:
            return ""
        per_year: Dict[int, int] = {}
        selected: List[Tuple[ADAChunk, float]] = []
        for chunk, score in chunks:
            yr = chunk.ada_year
            if per_year.get(yr, 0) < max_per_year:
                selected.append((chunk, score))
                per_year[yr] = per_year.get(yr, 0) + 1
        lines = []
        for chunk, _ in selected:
            sec   = f"Section {chunk.section_number}" if chunk.section_number else chunk.section_title[:40]
            level = f"Level {chunk.recommendation_level}" if chunk.recommendation_level else ""
            header_parts = [f"ADA {chunk.ada_year}", sec]
            if level:
                header_parts.append(level)
            body = chunk.text.replace("\n", " ").strip()
            if len(body) > max_chars:
                body = body[:max_chars].rsplit(" ", 1)[0] + "..."
            lines.append(f"[{' · '.join(header_parts)}]\n{body}")
        return "\n\n".join(lines)

    def to_indexer_format(self) -> List[Dict]:
        return [
            {
                "text": chunk.text,
                "metadata": {
                    "chunk_id":             chunk.chunk_id,
                    "ada_year":             chunk.ada_year,
                    "section_number":       chunk.section_number,
                    "section_title":        chunk.section_title,
                    "topic_tags":           chunk.topic_tags,
                    "recommendation_level": chunk.recommendation_level,
                    "is_recommendation":    chunk.is_recommendation,
                    "source":               "ada_guidelines",
                }
            }
            for chunk in self.chunks
        ]

    def get_stats(self) -> Dict:
        return {
            "total_chunks":    len(self.chunks),
            "by_year":         {yr: len(cs) for yr, cs in sorted(self.by_year.items())},
            "with_level_A":    sum(1 for c in self.chunks if c.recommendation_level == 'A'),
            "recommendations": sum(1 for c in self.chunks if c.is_recommendation),
            "topic_coverage":  sorted({t for c in self.chunks for t in c.topic_tags}),
            "hybrid_built":    self._hybrid is not None and self._hybrid.is_built,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# § 6  ADA 批量处理器
# ═══════════════════════════════════════════════════════════════════════════════

class ADAGuidelineProcessor:
    """批量处理 ADA 指南 Markdown → ADAGuidelineIndex（2006-2026）。"""

    _DEFAULT_YEARS = list(range(2006, 2027))

    def __init__(self, guidelines_dir: Path, index_output: Path):
        self.guidelines_dir = Path(guidelines_dir)
        self.index_output   = Path(index_output)
        self.index          = ADAGuidelineIndex()

    @property
    def SUPPORTED_YEARS(self) -> List[int]:
        if not self.guidelines_dir.exists():
            return self._DEFAULT_YEARS
        discovered = set()
        for f in self.guidelines_dir.glob("ada_standards_*.md"):
            y = f.stem.split("_")[-1]
            if y.isdigit() and len(y) == 4:
                discovered.add(int(y))
        for d in self.guidelines_dir.iterdir():
            if d.is_dir() and d.name.startswith("ada_standards_"):
                y = d.name.split("_")[-1]
                if y.isdigit() and len(y) == 4:
                    candidate = d / f"ada_standards_{y}.md"
                    if candidate.exists():
                        discovered.add(int(y))
        return sorted(discovered) if discovered else self._DEFAULT_YEARS

    def build_index(self, force_rebuild: bool = False) -> ADAGuidelineIndex:
        if not force_rebuild and self.index_output.exists():
            logger.info(f"加载已有索引: {self.index_output}")
            return self._load_index()

        logger.info(f"开始构建 ADA 指南索引，目录: {self.guidelines_dir}")
        self.index  = ADAGuidelineIndex()
        found_years = []

        for year in self.SUPPORTED_YEARS:
            md_file = self.guidelines_dir / f"ada_standards_{year}.md"
            if not md_file.exists():
                md_file = self.guidelines_dir / f"ada_standards_{year}" / f"ada_standards_{year}.md"
            if not md_file.exists():
                logger.warning(f"未找到 {year} 年指南: {self.guidelines_dir}/ada_standards_{year}[.md]")
                continue
            try:
                chunks = self._process_single_year(md_file, year)
                self.index.add_chunks(chunks)
                found_years.append(year)
                logger.info(f"  {year}年: {len(chunks)} chunks")
            except Exception as e:
                logger.error(f"处理 {year} 年失败: {e}")
                import traceback; logger.debug(traceback.format_exc())

        if not found_years:
            raise FileNotFoundError(
                f"{self.guidelines_dir} 中未找到任何 ADA 指南文件。"
                "请按命名约定放置: ada_standards_{year}.md"
            )

        self._save_index()
        stats = self.index.get_stats()
        print(f"\nADA 指南索引构建完成")
        print(f"  覆盖年份:  {found_years}")
        print(f"  总 chunks: {stats['total_chunks']}")
        print(f"  各年份:    {stats['by_year']}")
        print(f"  推荐条目:  {stats['recommendations']}")
        print(f"  A级推荐:   {stats['with_level_A']}")
        print(f"  索引文件:  {self.index_output}\n")
        return self.index

    def _process_single_year(self, md_file: Path, year: int) -> List[ADAChunk]:
        with open(md_file, 'r', encoding='utf-8') as f:
            text = f.read()
        if not text.strip():
            raise ValueError(f"{md_file} 文件为空")
        return ADATextParser(year).parse(text)

    def _save_index(self):
        self.index_output.parent.mkdir(parents=True, exist_ok=True)
        with open(self.index_output, 'wb') as f:
            pickle.dump(self.index, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"索引已保存: {self.index_output}")

    def _load_index(self) -> ADAGuidelineIndex:
        class _Unpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module in ('__main__', 'diamas_utils.ada_guideline_processor'):
                    module = __name__
                return super().find_class(module, name)
        with open(self.index_output, 'rb') as f:
            self.index = _Unpickler(f).load()
        logger.info(f"索引加载完成: {self.index.get_stats()}")
        return self.index

    def incremental_update(self, year: int) -> ADAGuidelineIndex:
        if self.index_output.exists():
            self._load_index()
        self.index.chunks  = [c for c in self.index.chunks  if c.ada_year != year]
        self.index.by_year = {yr: cs for yr, cs in self.index.by_year.items() if yr != year}
        md_file = self.guidelines_dir / f"ada_standards_{year}.md"
        if not md_file.exists():
            md_file = self.guidelines_dir / f"ada_standards_{year}" / f"ada_standards_{year}.md"
        if not md_file.exists():
            raise FileNotFoundError(f"未找到 {year} 年指南: {self.guidelines_dir}")
        new_chunks = self._process_single_year(md_file, year)
        self.index.add_chunks(new_chunks)
        self._save_index()
        logger.info(f"增量更新完成，{year}年: {len(new_chunks)} 个新 chunks")
        return self.index


# ═══════════════════════════════════════════════════════════════════════════════
# § 7  通用多源指南系统
# ═══════════════════════════════════════════════════════════════════════════════

# ── 指南源配置 ────────────────────────────────────────────────────────────────

# [Layer-1] 中文指南文件级专科路由表
# 规则：文件名（不含扩展名）前缀 → specialty_tags
# 优先级高于 GUIDELINE_SOURCE_CONFIG 里 chinese 的全局 specialty_tags
CHINESE_FILE_SPECIALTY_MAP: Dict[str, List[str]] = {
    # 纯内分泌
    "cn_t2dm_guideline_2024":      ["endocrinology"],
    "cn_elderly_diabetes_2024":    ["endocrinology"],
    "cn_dm_classification_2022":   ["endocrinology"],
    "cn_oral_combination_2023":    ["endocrinology"],
    "cn_glp1ra_consensus":         ["endocrinology"],
    # 肾科
    "cn_ckd_diabetes_consensus":   ["nephrology"],
    "cn_dkd_2021":                 ["nephrology"],
    # 心内科 + 肾科（重叠）
    "cn_finerenone_ckd_2023":      ["cardiology", "nephrology"],
    "cn_cv_diabetes_2022":         ["cardiology"],
    "cn_lipid_management_2024":    ["cardiology"],
    "cn_hypertension_2024":        ["cardiology"],
    # 内分泌 + 肾科（SGLT2i+胰岛素联合：降糖+肾保护）
    "cn_sglt2i_insulin_2023":      ["endocrinology", "nephrology"],
    # 内分泌 + 心内科
    "cn_hypoglycemia_consensus":   ["endocrinology"],  # Endocrine Society EN guideline，误放
}

GUIDELINE_SOURCE_CONFIG: Dict[str, Dict] = {
    "kdigo": {
        "display_name":   "KDIGO CKD & Diabetes Guidelines",
        "specialty_tags": ["nephrology"],
        "dir_name":       "kdigo",
        "index_file":     "kdigo_index.pkl",
        "language":       "en",
    },
    "acc_aha": {
        "display_name":   "ACC/AHA Cardiovascular Prevention & HF Guidelines",
        "specialty_tags": ["cardiology"],
        "dir_name":       "acc_aha",
        "index_file":     "acc_aha_index.pkl",
        "language":       "en",
    },
    "esc": {
        "display_name":   "ESC Heart Failure & Diabetes Guidelines",
        "specialty_tags": ["cardiology"],
        "dir_name":       "esc",
        "index_file":     "esc_index.pkl",
        "language":       "en",
    },
    "aace": {
        "display_name":   "AACE T2DM Comprehensive Management",
        "specialty_tags": ["endocrinology"],
        "dir_name":       "aace",
        "index_file":     "aace_index.pkl",
        "language":       "en",
    },
    # [Layer-1 新增] Endocrine Society 指南（住院高血糖 + 低血糖管理）
    # 原先 cn_hypoglycemia_consensus 错误归入 chinese，实为英文 Endocrine Society 指南
    "endocrine_society": {
        "display_name":   "Endocrine Society Clinical Practice Guidelines",
        "specialty_tags": ["endocrinology"],
        "dir_name":       "endocrine_society",
        "index_file":     "endocrine_society_index.pkl",
        "language":       "en",
    },
    "chinese": {
        "display_name":   "Chinese Clinical Guidelines & Expert Consensus",
        # [Layer-1] 全局 tag 保留三专科；文件级细化由 CHINESE_FILE_SPECIALTY_MAP 覆盖
        "specialty_tags": ["endocrinology", "nephrology", "cardiology"],
        "dir_name":       "chinese",
        "index_file":     "chinese_index.pkl",
        "language":       "zh",
    },
    "ada_supplements": {
        "display_name":   "ADA Consensus Reports & Supplements",
        "specialty_tags": ["endocrinology", "nephrology"],
        "dir_name":       "ada_supplements",
        "index_file":     "ada_supplements_index.pkl",
        "language":       "en",
    },
}

# 专科 → 优先检索的源（顺序代表权重，第一位最高）
SPECIALTY_SOURCE_ROUTING: Dict[str, List[str]] = {
    # 内分泌科
    "endocrinology": ["ada", "aace", "ada_supplements", "endocrine_society"],
    # 肾科：KDIGO 为主
    "nephrology":    ["kdigo", "ada_supplements"],
    # 心内科：欧美两套 CV 指南
    "cardiology":    ["esc", "acc_aha"],
}

# ── [v18 P0-black] 非临床样板段落：检索期排除 ───────────────────────────────
# 依据 reco_multisrc(i2)：索引层样板仅占 0.8–13.9%，实调层却达 21–23% ——
# 利益冲突声明里密集出现 Dexcom / Medtronic / Novo Nordisk / Ozempic / semaglutide
# 等药名，对药物类 query 形成"关键词磁铁"，再叠加 _keyword_score 的 rec_bonus=0.2。
# 只按 section_title 排除，**不触碰 is_recommendation**
# （i2/T2 实测：改 _is_recommendation 为大小写敏感会翻转 151 chunk，其中 88% 是真推荐）。
_NONCLINICAL_SECTION_PATTERNS = [
    r"disclosur",
    r"biographic and disclosure",
    r"notes on prior panel|panel member|writing (committee|group)|task force member",
    r"^\s*references?\s*$|^\s*参\s*考\s*文\s*献\s*$|^\s*bibliograph",
    r"table of contents|list of (figures|tables|recommendations|abbreviations)",
    r"^\s*abbreviations\s*$",
    r"^\s*appendix",
    r"^\s*figures\s*$",
    r"acknowledg|copyright|permissions|author information",
]
_NONCLINICAL_RE = re.compile("|".join(_NONCLINICAL_SECTION_PATTERNS), re.IGNORECASE)


def _is_nonclinical_section(chunk) -> bool:
    """按章节标题判定非临床样板段落。标题缺失时保守放行（不排除）。"""
    title = getattr(chunk, "section_title", "") or ""
    return bool(title.strip()) and bool(_NONCLINICAL_RE.search(title))

# 中国队列消融实验：chinese 源追加路由（不修改上方全局常量）
# 通过 ClinicalGuidelineRegistry.format_evidence_for_specialty() 的 [PATCH] 逻辑触发

# 推荐等级归一化映射（多体系 → unified A/B/C/E）
_LEVEL_NORM: Dict[str, str] = {
    "A": "A", "B": "B", "C": "C", "E": "E", "D": "E",
    "1A": "A", "1B": "A", "1C": "B", "1D": "B",
    "2A": "B", "2B": "B", "2C": "C", "2D": "C",
    "I": "A", "II": "B", "IIA": "B", "IIB": "C", "III": "E",
    "IA": "A", "IB": "B", "IC": "C",
}


# ── GuidelineChunk 数据结构 ────────────────────────────────────────────────────

@dataclass
class GuidelineChunk:
    """非 ADA 来源的通用指南文本块"""
    chunk_id:             str
    source_key:           str
    source_name:          str
    specialty_tags:       List[str]
    section_title:        str
    text:                 str
    recommendation_level: Optional[str]
    topic_tags:           List[str]
    is_recommendation:    bool
    pub_year:             Optional[int]
    filename:             str
    char_count:           int = field(init=False)

    def __post_init__(self):
        self.char_count = len(self.text)


# ── GuidelineIndex（非 era-aware）+ 混合检索集成 ──────────────────────────────

class GuidelineIndex:
    """
    通用指南索引（单源）。
    Phase-2 启用后自动切换到 BGE-M3 + BM25 混合检索路径。
    """

    def __init__(self, source_key: str, source_name: str, specialty_tags: List[str]):
        self.source_key    = source_key
        self.source_name   = source_name
        self.specialty_tags = specialty_tags
        self.chunks: List[GuidelineChunk] = []
        # [Layer-2]
        self._hybrid: Optional[object] = None
        self._cid_to_idx: Dict[str, int] = {}

    def __getattr__(self, name: str):
        """向后兼容：旧 pkl 缺少新增字段时返回安全默认值。"""
        _defaults = {
            "_hybrid":     None,
            "_cid_to_idx": {},
            "chunks":      [],
        }
        if name in _defaults:
            object.__setattr__(self, name, _defaults[name])
            return _defaults[name]
        raise AttributeError(f"'GuidelineIndex' object has no attribute '{name}'")

    def add_chunks(self, chunks: List[GuidelineChunk]):
        self.chunks.extend(chunks)
        logger.debug(f"[{self.source_key}] 索引现有 {len(self.chunks)} chunks")

    # [Layer-2] 构建混合检索器
    def build_hybrid_index(
        self,
        embed_model:    str  = "BAAI/bge-m3",
        use_reranker:   bool = False,
        reranker_model: str  = "BAAI/bge-reranker-v2-m3",
        device:         str  = "cpu",
        use_fp16:       bool = False,
    ) -> None:
        if not _HYBRID_AVAILABLE:
            logger.error(f"[{self.source_key}] hybrid_retriever 模块未找到")
            return
        self._hybrid = HybridRetriever(
            embed_model=embed_model,
            use_reranker=use_reranker,
            reranker_model=reranker_model,
            device=device,
            use_fp16=use_fp16,
        )
        texts     = [c.text for c in self.chunks]
        chunk_ids = [c.chunk_id for c in self.chunks]
        self._hybrid.build(texts, chunk_ids)
        self._cid_to_idx = {cid: i for i, cid in enumerate(chunk_ids)}
        logger.info(f"[{self.source_key}] 混合索引构建完成，{len(self.chunks)} chunks")

    def search(
        self,
        query:               str,
        topic_filter:        Optional[List[str]] = None,
        recommendation_only: bool = False,
        top_k:               int  = 5,
        era_num:             Optional[int] = None,
        max_year:            Optional[int] = None,
    ) -> List[Tuple["GuidelineChunk", float]]:
        _ERA_MAX: Dict[int, int] = {1: 2013, 2: 2019, 3: 9999}
        candidates = self.chunks

        # 时序预过滤
        if era_num is not None:
            max_yr = _ERA_MAX.get(era_num, 9999)
            candidates = [c for c in candidates
                          if c.pub_year is not None and c.pub_year <= max_yr]
        if max_year is not None:
            candidates = [c for c in candidates
                          if c.pub_year is not None and c.pub_year <= max_year]

        # 主题预过滤
        if topic_filter:
            topic_set  = set(topic_filter)
            candidates = [c for c in candidates if topic_set & set(c.topic_tags)]
        if recommendation_only:
            candidates = [c for c in candidates if c.is_recommendation]
        # [v18 P0-black] 排除非临床样板段落
        _before_nc = len(candidates)
        candidates = [c for c in candidates if not _is_nonclinical_section(c)]
        if _before_nc != len(candidates):
            logger.debug(f"[{self.source_key}] 样板过滤: {_before_nc} → {len(candidates)}")

        if not candidates:
            return []

        # [Layer-2] 混合检索路径
        if self._hybrid is not None and self._hybrid.is_built:
            cand_indices = [self._cid_to_idx[c.chunk_id]
                            for c in candidates if c.chunk_id in self._cid_to_idx]
            if not cand_indices:
                return []
            hits = self._hybrid.search(query, candidate_indices=cand_indices,
                                       top_k=top_k, rerank_top_k=top_k)
            cid_map = {c.chunk_id: c for c in candidates}
            results = []
            for idx, score in hits:
                cid = self._hybrid.idx_to_chunk_id(idx)
                if cid and cid in cid_map:
                    results.append((cid_map[cid], score))
            return results[:top_k]

        # Phase-1 关键词检索（兜底）
        scored = self._keyword_score(query, candidates)
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    @staticmethod
    def _keyword_score(query: str, candidates: List["GuidelineChunk"]):
        query_tokens = re.findall(r'[\u4e00-\u9fff]{1,4}|[a-z0-9\-]+', query.lower())
        if not query_tokens:
            return [(c, 0.0) for c in candidates]
        scored = []
        for chunk in candidates:
            text_lower  = chunk.text.lower()
            title_lower = chunk.section_title.lower()
            tf = sum(
                text_lower.count(tok) * (1.0 + 0.5 * title_lower.count(tok))
                for tok in query_tokens
            )
            tf /= (1 + chunk.char_count / 500)
            level_bonus = {"A": 0.3, "B": 0.2, "C": 0.1, "E": 0.05}.get(
                chunk.recommendation_level or "", 0
            )
            rec_bonus = 0.2 if chunk.is_recommendation else 0
            scored.append((chunk, tf + level_bonus + rec_bonus))
        return scored

    def format_for_prompt(
        self,
        results:   List[Tuple["GuidelineChunk", float]],
        max_chars: int = 350,
    ) -> str:
        if not results:
            return ""
        lines = []
        for chunk, _ in results:
            level = f" · Level {chunk.recommendation_level}" if chunk.recommendation_level else ""
            year  = f" · {chunk.pub_year}" if chunk.pub_year else ""
            header = f"[{chunk.source_name}{year} · {chunk.section_title[:50]}{level}]"
            body   = chunk.text.replace("\n", " ").strip()
            if len(body) > max_chars:
                body = body[:max_chars].rsplit(" ", 1)[0] + "..."
            lines.append(f"{header}\n{body}")
        return "\n\n".join(lines)

    def get_stats(self) -> Dict:
        return {
            "source_key":      self.source_key,
            "total_chunks":    len(self.chunks),
            "recommendations": sum(1 for c in self.chunks if c.is_recommendation),
            "with_level_A":    sum(1 for c in self.chunks if c.recommendation_level == "A"),
            "pub_years":       sorted({c.pub_year for c in self.chunks if c.pub_year}),
            "hybrid_built":    self._hybrid is not None and self._hybrid.is_built,
        }


# ── GenericTextParser ────────────────────────────────────────────────────────

class GenericTextParser:
    """
    将任意 Markdown 指南文件解析为 GuidelineChunk 列表。
    支持英文（KDIGO / ACC-AHA / ESC / AACE）和中文（中国指南 / 专家共识）。
    """

    MIN_CHUNK_CHARS = 80
    MAX_CHUNK_CHARS = 3000

    _HEADER_PATTERN = re.compile(r'^(#{1,4})\s*(.{2,160})$', re.MULTILINE)

    def __init__(
        self,
        source_key:     str,
        source_name:    str,
        specialty_tags: List[str],
        pub_year:       Optional[int],
        filename:       str,
    ):
        self.source_key    = source_key
        self.source_name   = source_name
        self.specialty_tags = specialty_tags
        self.pub_year      = pub_year
        self.filename      = filename

    def parse(self, markdown_text: str) -> List[GuidelineChunk]:
        headers = list(self._HEADER_PATTERN.finditer(markdown_text))
        if not headers:
            return self._split_by_paragraphs(markdown_text)
        return self._split_by_headers(markdown_text, headers)

    def _split_by_headers(self, text: str, headers) -> List[GuidelineChunk]:
        chunks = []
        for idx, hm in enumerate(headers):
            title = hm.group(2).strip()
            start = hm.end()
            end   = headers[idx + 1].start() if idx + 1 < len(headers) else len(text)
            body  = text[start:end].strip()
            if len(body) < self.MIN_CHUNK_CHARS:
                continue
            if len(body) > self.MAX_CHUNK_CHARS:
                chunks.extend(self._split_long_section(body, title, global_offset=len(chunks)))
            else:
                c = self._make_chunk(body, title, len(chunks))
                if c:
                    chunks.append(c)
        logger.info(f"[{self.source_key}/{self.filename}] {len(chunks)} chunks")
        return chunks

    def _split_by_paragraphs(self, text: str) -> List[GuidelineChunk]:
        chunks     = []
        paragraphs = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
        buffer     = ""
        for para in paragraphs:
            buffer += ("\n\n" + para if buffer else para)
            if len(buffer) >= self.MIN_CHUNK_CHARS * 3:
                c = self._make_chunk(buffer, "General Recommendations", len(chunks))
                if c:
                    chunks.append(c)
                buffer = ""
        if buffer and len(buffer) >= self.MIN_CHUNK_CHARS:
            c = self._make_chunk(buffer, "General Recommendations", len(chunks))
            if c:
                chunks.append(c)
        return chunks

    def _split_long_section(self, text, title, global_offset=0) -> List[GuidelineChunk]:
        sub    = []
        paras  = [p.strip() for p in re.split(r'\n{2,}', text) if p.strip()]
        buffer = ""
        for para in paras:
            if len(buffer) + len(para) > self.MAX_CHUNK_CHARS and buffer:
                c = self._make_chunk(buffer, title, global_offset + len(sub))
                if c:
                    sub.append(c)
                buffer = para
            else:
                buffer += ("\n\n" + para if buffer else para)
        if buffer and len(buffer) >= self.MIN_CHUNK_CHARS:
            c = self._make_chunk(buffer, title, global_offset + len(sub))
            if c:
                sub.append(c)
        return sub

    def _make_chunk(self, text: str, section_title: str, idx: int) -> Optional[GuidelineChunk]:
        if len(text) < self.MIN_CHUNK_CHARS:
            return None
        src_abbr = self.source_key[:8]
        return GuidelineChunk(
            chunk_id             = f"{src_abbr}_{self.pub_year or 'x'}_{idx:04d}",
            source_key           = self.source_key,
            source_name          = self.source_name,
            specialty_tags       = self.specialty_tags,
            section_title        = section_title,
            text                 = text,
            recommendation_level = self._extract_level(text),
            topic_tags           = self._infer_topic_tags(text, section_title),
            is_recommendation    = self._is_recommendation(text),
            pub_year             = self.pub_year,
            filename             = self.filename,
        )

    @staticmethod
    def _extract_level(text: str) -> Optional[str]:
        """提取推荐等级并归一化为 A/B/C/E。支持 ADA/KDIGO/ACC-AHA/ESC/中文格式。"""
        if re.search(r'强[烈力]推荐', text):
            return "A"
        if re.search(r'弱推荐|条件性推荐|有条件推荐', text):
            return "C"
        patterns = [
            r'\(([ABCE])\)',
            r'\bLevel\s+([ABCE])\b',
            r'\b(1[ABCD]|2[ABCD])\b',
            r'\bGrade\s+([ABCDE1-2])\b',
            r'\bClass\s+(I{1,3}[Aa]?[Bb]?)\b',
            r'\b([ABCE])\b(?=\s*[·•]|\s*$)',
        ]
        found = []
        for pat in patterns:
            for m in re.finditer(pat, text, re.MULTILINE | re.IGNORECASE):
                raw     = m.group(1).upper().strip()
                unified = _LEVEL_NORM.get(raw)
                if unified:
                    found.append(unified)
        if not found:
            return None
        priority = {"A": 4, "B": 3, "C": 2, "E": 1}
        return max(set(found), key=lambda x: priority.get(x, 0))

    @staticmethod
    def _infer_topic_tags(text: str, title: str) -> List[str]:
        """[Layer-1 改造] 与 ADATextParser 共用 _DRUG_TAG_RULES（config.py 驱动）。"""
        combined = (text + " " + title).lower()
        return [tag for tag, patterns in _DRUG_TAG_RULES.items()
                if any(re.search(p, combined) for p in patterns)]

    @staticmethod
    def _is_recommendation(text: str) -> bool:
        text_lower = text.lower()
        negation_signals = [
            "not recommended", "is not advised", "are not advised",
            "not be recommended", "should not be used",
            "suggest against", "do not suggest", "not suggest",
            "recommend against", "do not recommend",
            "不推荐", "不适用", "禁忌", "不建议", "反对使用", "避免使用",
        ]
        if any(s in text_lower for s in negation_signals):
            return bool(re.search(r'\([ABCE]\)', text))

        positive_patterns = [
            r'\([ABCE]\)', r'\bLevel\s+[ABCE]\b',
            r'\bGrade\s+[12][ABCD]\b', r'\bClass\s+I{1,3}[ABab]?\b',
            r'\b(?:is|are|should\s+be)\s+recommended\b', r'\brecommend\b',
            r'\bsuggest\b(?!\s+against)', r'\bshould\b.*\([ABCE]\)',
            r'\bconsider\b.*\([ABCE]\)',
            r'强[烈力]推荐', r'弱推荐|条件性推荐|有条件推荐',
            r'推荐.*(?:I类|A级|1级)', r'建议.*(?:II类|B级|2级)',
        ]
        return any(re.search(p, text, re.MULTILINE | re.IGNORECASE)
                   for p in positive_patterns)


# ── GenericGuidelineProcessor ────────────────────────────────────────────────

class GenericGuidelineProcessor:
    """
    处理单个指南来源目录下所有 .md 文件 → GuidelineIndex。

    [Layer-1] 对 chinese 源：按 CHINESE_FILE_SPECIALTY_MAP 进行文件级专科路由，
    精度从"源级"提升到"文件级"。
    """

    def __init__(
        self,
        guidelines_dir: Path,
        index_dir:      Path,
        source_key:     str,
    ):
        cfg = GUIDELINE_SOURCE_CONFIG.get(source_key)
        if cfg is None:
            raise ValueError(f"未知的 source_key: {source_key!r}。"
                             f"支持: {list(GUIDELINE_SOURCE_CONFIG)}")
        self.source_key    = source_key
        self.source_name   = cfg["display_name"].split()[0]
        self.specialty_tags = cfg["specialty_tags"]
        self.source_dir    = Path(guidelines_dir) / cfg["dir_name"]
        self.index_path    = Path(index_dir) / cfg["index_file"]
        self.index         = GuidelineIndex(source_key, self.source_name, self.specialty_tags)

    def build_index(self, force_rebuild: bool = False) -> GuidelineIndex:
        if not force_rebuild and self.index_path.exists():
            logger.info(f"[{self.source_key}] 加载已有索引: {self.index_path}")
            return self._load_index()

        if not self.source_dir.exists():
            logger.warning(f"[{self.source_key}] 目录不存在: {self.source_dir}")
            return self.index

        md_files = sorted(self.source_dir.glob("*.md")) + sorted(self.source_dir.glob("*/*.md"))
        if not md_files:
            logger.warning(f"[{self.source_key}] 无 .md 文件: {self.source_dir}")
            return self.index

        self.index = GuidelineIndex(self.source_key, self.source_name, self.specialty_tags)
        for md_file in md_files:
            try:
                chunks = self._process_file(md_file)
                self.index.add_chunks(chunks)
                logger.info(f"  [{self.source_key}] {md_file.name}: {len(chunks)} chunks")
            except Exception as e:
                logger.error(f"  [{self.source_key}] {md_file.name} 处理失败: {e}")

        self._save_index()
        stats = self.index.get_stats()
        print(f"\n[{self.source_key}] 索引构建完成  chunks={stats['total_chunks']}  "
              f"recommendations={stats['recommendations']}  years={stats['pub_years']}")
        return self.index

    def _process_file(self, md_file: Path) -> List[GuidelineChunk]:
        year_match = re.search(r'(\d{4})(?=\D*$)', md_file.stem)
        pub_year   = int(year_match.group(1)) if year_match else None

        # [Layer-1] 文件级专科路由（对 chinese 源生效）
        if self.source_key == "chinese":
            stem_tags = CHINESE_FILE_SPECIALTY_MAP.get(md_file.stem)
            specialty_tags = stem_tags if stem_tags else self.specialty_tags
        else:
            specialty_tags = self.specialty_tags

        with open(md_file, encoding="utf-8") as f:
            text = f.read()
        if not text.strip():
            return []

        parser = GenericTextParser(
            source_key    = self.source_key,
            source_name   = self.source_name,
            specialty_tags = specialty_tags,
            pub_year      = pub_year,
            filename      = md_file.name,
        )
        return parser.parse(text)

    def _save_index(self):
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"[{self.source_key}] 索引已保存: {self.index_path}")

    def _load_index(self) -> GuidelineIndex:
        class _Unpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == "__main__":
                    module = __name__
                return super().find_class(module, name)
        with open(self.index_path, "rb") as f:
            self.index = _Unpickler(f).load()
        return self.index


# ═══════════════════════════════════════════════════════════════════════════════
# § 8  ClinicalGuidelineRegistry
# ═══════════════════════════════════════════════════════════════════════════════

class ClinicalGuidelineRegistry:
    """
    统一管理所有指南来源索引，按专科路由检索。

    [Layer-2] build_all_hybrid_indexes() 一键升级全库到混合检索。

    中国队列消融实验：
        registry.format_evidence_for_specialty(
            specialty="endocrinology", queries=[...],
            era_year=2020, use_chinese=True
        )
    """

    def __init__(self):
        self.ada_index: Optional[ADAGuidelineIndex] = None
        self._source_indexes: Dict[str, GuidelineIndex] = {}

    def register_ada(self, index: ADAGuidelineIndex):
        self.ada_index = index

    def register_source(self, source_key: str, index: GuidelineIndex):
        if index.chunks:
            self._source_indexes[source_key] = index

    @property
    def active_sources(self) -> List[str]:
        sources = []
        if self.ada_index and self.ada_index.chunks:
            sources.append("ada")
        sources.extend(self._source_indexes.keys())
        return sources

    # [Layer-2] 一键升级全库到混合检索
    def build_all_hybrid_indexes(
        self,
        embed_model:    str  = "BAAI/bge-m3",
        use_reranker:   bool = False,
        reranker_model: str  = "BAAI/bge-reranker-v2-m3",
        device:         str  = "cpu",
        use_fp16:       bool = False,
    ) -> None:
        """
        [Layer-2] 为所有已注册的索引构建混合检索器。
        调用后，所有 search() 自动切换到 BGE-M3 + BM25 + RRF 路径。

        建议在 build_all_indexes() 完成后调用一次。
        """
        kwargs = dict(embed_model=embed_model, use_reranker=use_reranker,
                      reranker_model=reranker_model, device=device, use_fp16=use_fp16)
        if self.ada_index:
            logger.info("[Registry] 构建 ADA 混合索引...")
            self.ada_index.build_hybrid_index(**kwargs)
        for sk, idx in self._source_indexes.items():
            logger.info(f"[Registry] 构建 {sk} 混合索引...")
            idx.build_hybrid_index(**kwargs)
        logger.info("[Registry] 所有混合索引构建完成")

    def get_all_stats(self) -> List[Dict]:
        stats = []
        if self.ada_index:
            ada_s = self.ada_index.get_stats()
            ada_s["source_key"] = "ada"
            stats.append(ada_s)
        for sk, idx in self._source_indexes.items():
            stats.append(idx.get_stats())
        return stats

    def search_specialty(
        self,
        query:     str,
        specialty: str,
        era_year:  Optional[int] = None,
        top_k:     int = 5,
        era_num:   Optional[int] = None,
        _routing_override: Optional[Dict[str, List[str]]] = None,
    ) -> List[Tuple]:
        routing    = _routing_override if _routing_override is not None else SPECIALTY_SOURCE_ROUTING
        source_keys = routing.get(specialty, ["ada"])
        results    = []
        seen_ids:  set = set()

        for sk in source_keys:
            if sk == "ada" and self.ada_index:
                max_year = era_year
                hits = self.ada_index.search(
                    query, max_year=max_year, recommendation_only=True,
                    top_k=max(2, top_k // len(source_keys) + 1)
                )
                for chunk, score in hits:
                    if chunk.chunk_id not in seen_ids:
                        results.append((chunk, score, "ada"))
                        seen_ids.add(chunk.chunk_id)

            elif sk in self._source_indexes:
                idx  = self._source_indexes[sk]
                hits = idx.search(
                    query, recommendation_only=True,
                    top_k=max(2, top_k // len(source_keys) + 1),
                    era_num=era_num,
                    max_year=era_year,
                )
                for chunk, score in hits:
                    if chunk.chunk_id not in seen_ids:
                        if era_num is None and era_year and chunk.pub_year \
                                and chunk.pub_year > era_year:
                            score *= 0.5
                        results.append((chunk, score, sk))
                        seen_ids.add(chunk.chunk_id)

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def format_evidence_for_specialty(
        self,
        specialty:   str,
        queries:     List[str],
        era_year:    Optional[int] = None,
        top_k:       int = 6,
        era_num:     Optional[int] = None,
        use_chinese: bool = False,    # [Layer-1] 中国队列消融实验开关
    ) -> str:
        # [Layer-1 PATCH] chinese opt-in：仅消融实验时追加
        _effective_routing = SPECIALTY_SOURCE_ROUTING.copy()
        if use_chinese and "chinese" in self._source_indexes:
            for spec in ["endocrinology", "nephrology", "cardiology"]:
                current = _effective_routing.get(spec, [])
                if "chinese" not in current:
                    _effective_routing[spec] = current + ["chinese"]

        all_results: List[Tuple] = []
        seen_ids: set = set()

        for query in queries[:4]:
            for chunk, score, sk in self.search_specialty(
                    query, specialty, era_year=era_year,
                    top_k=3, era_num=era_num,
                    _routing_override=_effective_routing):
                if chunk.chunk_id not in seen_ids:
                    all_results.append((chunk, score, sk))
                    seen_ids.add(chunk.chunk_id)

        all_results.sort(key=lambda x: x[1], reverse=True)
        all_results = all_results[:top_k]

        if not all_results:
            return ""

        lines = []
        for chunk, _score, _sk in all_results:
            if isinstance(chunk, ADAChunk):
                level  = f" · Level {chunk.recommendation_level}" if chunk.recommendation_level else ""
                header = f"[ADA {chunk.ada_year} · {chunk.section_title[:50]}{level}]"
            else:
                level  = f" · Level {chunk.recommendation_level}" if chunk.recommendation_level else ""
                year   = f" · {chunk.pub_year}" if chunk.pub_year else ""
                header = f"[{chunk.source_name}{year} · {chunk.section_title[:50]}{level}]"
            body = chunk.text.replace("\n", " ").strip()[:350]
            lines.append(f"{header}\n{body}")

        return "\n\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# § 9  工厂函数
# ═══════════════════════════════════════════════════════════════════════════════

def build_all_indexes(
    guidelines_dir: Path,
    index_dir:      Path,
    sources:        Optional[List[str]] = None,
    force:          bool = False,
    # [Layer-2/3] 混合检索参数
    build_hybrid:   bool = False,
    embed_model:    str  = "BAAI/bge-m3",
    use_reranker:   bool = False,
    reranker_model: str  = "BAAI/bge-reranker-v2-m3",
    device:         str  = "cpu",
    use_fp16:       bool = False,
) -> ClinicalGuidelineRegistry:
    """
    构建（或加载）所有指南来源的索引，返回 ClinicalGuidelineRegistry。

    Args:
        guidelines_dir: 指南根目录（含 ada/ kdigo/ esc/ chinese/ 等子目录）
        index_dir:      索引文件输出目录
        sources:        要构建的源列表，None 表示全部
        force:          是否强制重建
        build_hybrid:   [Layer-2] 是否构建 BGE-M3+BM25 混合索引
        embed_model:    [Layer-2] BGE-M3 模型路径
        use_reranker:   [Layer-3] 是否启用 cross-encoder 精排
        reranker_model: [Layer-3] reranker 模型路径
        device:         "cpu" / "cuda"
        use_fp16:       GPU 环境建议 True
    """
    guidelines_dir = Path(guidelines_dir)
    index_dir      = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    registry = ClinicalGuidelineRegistry()

    # ── ADA（era-aware，使用 ADAGuidelineProcessor）──────────────────────────
    if sources is None or "ada" in sources:
        ada_dir    = guidelines_dir / "ada"
        ada_output = index_dir / "ada_index.pkl"
        if ada_dir.exists():
            proc      = ADAGuidelineProcessor(guidelines_dir=ada_dir, index_output=ada_output)
            ada_index = proc.build_index(force_rebuild=force)
            registry.register_ada(ada_index)
            logger.info(f"[ada] 注册完成: {ada_index.get_stats()['total_chunks']} chunks")
        else:
            logger.warning(f"[ada] 目录不存在: {ada_dir}")

    # ── 其他来源（GenericGuidelineProcessor）────────────────────────────────
    for source_key in GUIDELINE_SOURCE_CONFIG:
        if sources is not None and source_key not in sources:
            continue
        proc  = GenericGuidelineProcessor(
            guidelines_dir = guidelines_dir,
            index_dir      = index_dir,
            source_key     = source_key,
        )
        index = proc.build_index(force_rebuild=force)
        registry.register_source(source_key, index)

    # ── [Layer-2/3] 混合检索（可选）─────────────────────────────────────────
    # 关键：混合索引构建失败 → WARNING 降级，不抛异常，不污染 registry 返回值。
    # 若此处抛异常且未捕获，调用方的 except 会将 registry 置为 None，
    # 导致所有已成功注册的 basic source_indexes 全部丢失。
    if build_hybrid:
        try:
            logger.info("[build_all_indexes] 开始构建混合检索索引（BGE-M3 + BM25）...")
            registry.build_all_hybrid_indexes(
                embed_model=embed_model,
                use_reranker=use_reranker,
                reranker_model=reranker_model,
                device=device,
                use_fp16=use_fp16,
            )
            logger.info("[build_all_indexes] 混合检索索引构建完成")
        except Exception as _hybrid_err:
            logger.warning(
                f"[build_all_indexes] 混合检索索引构建失败，降级为 BM25-only 模式: {_hybrid_err}\n"
                f"  常见原因: BGE-M3 模型未下载 / GPU 显存不足 / device={device!r} 不可用。\n"
                f"  基础索引不受影响，推理可正常继续。"
            )

    return registry


# ═══════════════════════════════════════════════════════════════════════════════
# § 10  命令行入口
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    parser = argparse.ArgumentParser(
        description="Dia-MAS 指南索引构建器\n"
                    "  --source ada       仅构建 ADA 索引\n"
                    "  --source all       构建全部指南源索引\n"
                    "  --source kdigo,esc 构建指定源索引\n"
                    "  --hybrid           启用 BGE-M3+BM25 混合检索（Layer-2）\n"
                    "  --reranker         启用 cross-encoder 精排（Layer-3）"
    )
    parser.add_argument('--guidelines_dir', '-g', default='data/guidelines_md')
    parser.add_argument('--index_dir',      '-d', default='data/indexes')
    parser.add_argument('--input',   '-i', default=None, help='[compat] ADA Markdown 目录')
    parser.add_argument('--output',  '-o', default=None, help='[compat] ADA 索引输出路径')
    parser.add_argument('--source',  '-s', default='ada')
    parser.add_argument('--force',   '-f', action='store_true')
    parser.add_argument('--update_year', type=int, default=None)
    parser.add_argument('--test',    action='store_true')
    # Layer-2/3 参数
    parser.add_argument('--hybrid',     action='store_true', help='启用混合检索')
    parser.add_argument('--reranker',   action='store_true', help='启用 cross-encoder 精排')
    parser.add_argument('--embed_model', default='BAAI/bge-m3')
    parser.add_argument('--reranker_model', default='BAAI/bge-reranker-v2-m3')
    parser.add_argument('--device',     default='cpu')
    parser.add_argument('--fp16',       action='store_true')
    args = parser.parse_args()

    guidelines_dir = Path(args.input  or args.guidelines_dir)
    index_dir      = Path(args.index_dir)

    # ── 仅 ADA 路径（向后兼容）──────────────────────────────────────────────
    if args.source == 'ada' and not args.input:
        ada_dir    = guidelines_dir / "ada"
        ada_output = Path(args.output) if args.output else index_dir / "ada_index.pkl"
        processor  = ADAGuidelineProcessor(guidelines_dir=ada_dir, index_output=ada_output)
        if args.update_year:
            index = processor.incremental_update(args.update_year)
        else:
            index = processor.build_index(force_rebuild=args.force)

        if args.hybrid:
            index.build_hybrid_index(
                embed_model=args.embed_model,
                use_reranker=args.reranker,
                reranker_model=args.reranker_model,
                device=args.device,
                use_fp16=args.fp16,
            )

        if args.test:
            print("\n=== ADA 查询测试 ===")
            test_cases = [
                ("SGLT2 inhibitor CKD eGFR <30", {"topic_filter": ["sglt2", "ckd_treatment"]}),
                ("metformin renal contraindication", {"level_filter": ["A", "B"]}),
                ("GLP-1 cardiovascular ASCVD",    {"recommendation_only": True}),
                ("钠葡萄糖共转运蛋白2抑制剂 肾脏保护",  {}),   # 中文查询测试
            ]
            for query, kwargs in test_cases:
                print(f"\n查询: {query!r}")
                for chunk, score in index.search(query, top_k=2, **kwargs):
                    print(f"  [{chunk.ada_year}] {chunk.section_title[:50]} | "
                          f"score={score:.4f} | level={chunk.recommendation_level}")
        return

    # ── 多源路径 ─────────────────────────────────────────────────────────────
    sources = None if args.source == 'all' else (
        ['ada'] if args.source == 'ada'
        else [s.strip() for s in args.source.split(',') if s.strip()]
    )

    registry = build_all_indexes(
        guidelines_dir = guidelines_dir,
        index_dir      = index_dir,
        sources        = sources,
        force          = args.force,
        build_hybrid   = args.hybrid,
        embed_model    = args.embed_model,
        use_reranker   = args.reranker,
        reranker_model = args.reranker_model,
        device         = args.device,
        use_fp16       = args.fp16,
    )
    print(f"\n构建完成，活跃源: {registry.active_sources}")
    print(f"各源统计:")
    for stats in registry.get_all_stats():
        print(f"  {stats}")


if __name__ == "__main__":
    main()
