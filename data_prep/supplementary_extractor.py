#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supplementary Feature Extractor  (supplementary_extractor.py)
==============================================================
对 patient_cache_v4/ 目录中已有的患者 JSON 做第二次扫描，
增量计算并写回新字段，**不覆盖任何现有字段**。

新增字段（直接写入原有字典结构，不新增顶层 key）：

  clinical_metrics.kidney:
    creatinine_peak     住院期间（48h 窗口内）最高肌酐值 (mg/dL)
    aki_kdigo_stage     KDIGO 2012 AKI 分期 (1/2/3 or None)

  clinical_metrics.liver  （新 section，完全新建）:
    alt                 ALT/SGPT (IU/L)
    ast                 AST/SGOT (IU/L)
    bilirubin_total     总胆红素 (mg/dL)
    data_available      bool
    liver_dysfunction   bool（ALT/AST > 3×ULN 或 bilirubin > 2.0 mg/dL）

  clinical_metrics.dm_complications:
    on_corticosteroids  bool（检测到系统性皮质类固醇用药）
    vasopressor_use     bool（检测到升压药使用，提示血流动力学不稳定）

  clinical_metrics.umls_patch（仅 --umls_patch 模式写入）:
    version             修复版本号（仅当 remaining==0 时写入，供下次跳过判断）
    remaining           本次运行后仍未映射的诊断数（>0 表示有 API 失败，下次自动重试）
    fixed_this_run      本次运行成功修复数
    last_run            最近一次运行的 ISO 时间戳

设计原则：
  - 增量写入：已有字段（egfr / creatinine / cardiovascular 等）一律不改动
  - supplementary_version 标记已处理状态；--force 强制重处理
  - umls_patch_version 独立标记，与 supplementary_version 解耦
  - 多线程并行 + 原子写（.tmp → rename），不损坏任何 JSON
  - 三库通用：MIMIC-III / MIMIC-IV 用 itemid 精确匹配；eICU 用关键词回退
  - UMLS patch：共享单一 mapper 实例（_tgt_lock 保证线程安全），只重试 umls_cui=None 的条目

使用方式：
    python supplementary_extractor.py                          # 增量处理（临床特征补充）
    python supplementary_extractor.py --force                  # 强制全量重跑
    python supplementary_extractor.py --cache_dir path/to/dir  # 自定义目录
    python supplementary_extractor.py --workers 16             # 并发线程数
    python supplementary_extractor.py --umls_patch             # UMLS 映射补丁（修复 api_errors）
    python supplementary_extractor.py --umls_patch --force     # 强制全量 UMLS 重映射

作者：Yuxi
版本：v1.1  （新增 UMLS patch 模式）
"""

from __future__ import annotations

import argparse
import json
import logging
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from agents.config import PathConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("loggings/supplementary_extractor.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

SUPPLEMENTARY_VERSION = "v1.0"
UMLS_PATCH_VERSION    = "v1.0"   # 独立版本号，与 supplementary_version 解耦

# ═══════════════════════════════════════════════════════════════
# 通用工具
# ═══════════════════════════════════════════════════════════════

def _now_iso() -> str:
    """返回 UTC ISO-8601 时间戳字符串。"""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atomic_write(filepath: Path, data: Dict) -> None:
    """原子写：先写 .tmp，再 rename，防止写入中断损坏源文件。"""
    tmp_path = filepath.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp_path.rename(filepath)


# ═══════════════════════════════════════════════════════════════
# itemid 精确映射表（来源：clinical_calculator_unified.py）
# 仅复制本脚本所需的条目，保持与原始文件完全一致
# ═══════════════════════════════════════════════════════════════

_MIMIC3_LAB_MAP: Dict[int, str] = {
    # 肾功能
    50912: "creatinine",
    # 肝功能
    50861: "alt",
    50878: "ast",
    50885: "bilirubin_total",
}

_MIMIC4_LAB_MAP: Dict[int, str] = {
    # 肾功能
    52024: "creatinine",
    50912: "creatinine",     # 兼容
    # 肝功能
    52012: "alt",
    52013: "ast",
    52014: "bilirubin_total",
}

# 关键词回退（eICU 和 itemid 匹配失败时）
_KEYWORD_FALLBACK: Dict[str, Tuple[List[str], List[str]]] = {
    # (include_keywords, exclude_keywords)
    "creatinine":      (["creatinine"],            ["urine", "clearance", "ratio"]),
    "alt":             (["alt", "sgpt", "alanine aminotransferase"],     ["urine"]),
    "ast":             (["ast", "sgot", "aspartate aminotransferase"],   ["urine"]),
    "bilirubin_total": (["bilirubin", "total bilirubin"],               ["direct", "indirect",
                                                                         "conjugated", "neonatal"]),
}

# 正常参考值上限（ULN）
_ALT_ULN  = 40.0   # IU/L
_AST_ULN  = 40.0   # IU/L
_BILI_ULN = 1.2    # mg/dL

# 系统性皮质类固醇关键词（常见 ICU/住院用药）
_CORTICOSTEROID_KW = [
    "prednisone", "prednisolone", "methylprednisolone",
    "dexamethasone", "hydrocortisone",
]

# 升压药关键词（血流动力学不稳定标志）
_VASOPRESSOR_KW = [
    "norepinephrine", "epinephrine", "vasopressin",
    "phenylephrine", "dopamine", "dobutamine", "milrinone",
]


# ═══════════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════════

def _safe_float(v) -> Optional[float]:
    """安全转 float，NaN/None/'nan' 均返回 None。"""
    import math
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    try:
        result = float(v)
        if math.isnan(result) or math.isinf(result):
            return None
        return result
    except (TypeError, ValueError):
        return None


def _get_lab_values(
    labs: List[Dict],
    semantic_name: str,
    itemid_map: Dict[int, str],
    strategy: str = "first",
):
    """
    从序列化的 lab_results 列表中提取检验值。

    Args:
        labs:         patient JSON 中 all_lab_results（list of dicts）
        semantic_name: 语义名称（如 'creatinine', 'alt'）
        itemid_map:   数据库专用 itemid → semantic_name 映射
        strategy:     'first' / 'last' / 'max' / 'min' / 'all'

    Returns:
        单个 float 或 List[float]（strategy='all'），无匹配时返回 None 或 []
    """
    matched: List[Tuple[Optional[str], float]] = []

    # Pass 1: itemid 精确匹配
    for lab in labs:
        if not isinstance(lab, dict):
            continue
        if lab.get("is_calculated"):
            continue
        raw_id = lab.get("itemid")
        if raw_id is not None:
            try:
                if itemid_map.get(int(raw_id)) == semantic_name:
                    val = _safe_float(lab.get("value"))
                    if val is not None and val > 0:
                        matched.append((lab.get("charttime"), val))
            except (ValueError, TypeError):
                pass

    # Pass 2: 关键词回退（eICU 或 itemid 匹配失败）
    if not matched and semantic_name in _KEYWORD_FALLBACK:
        inc_kws, exc_kws = _KEYWORD_FALLBACK[semantic_name]
        for lab in labs:
            if not isinstance(lab, dict):
                continue
            if lab.get("is_calculated"):
                continue
            label = str(lab.get("label", "") or "").lower()
            if not label:
                continue
            if any(kw in label for kw in inc_kws) and not any(ex in label for ex in exc_kws):
                val = _safe_float(lab.get("value"))
                if val is not None and val > 0:
                    matched.append((lab.get("charttime"), val))

    if not matched:
        return [] if strategy == "all" else None

    # 按 charttime 排序（None 时间置末）
    matched.sort(key=lambda x: (x[0] is None, str(x[0] or "")))
    values = [v for _, v in matched]

    if strategy == "all":
        return values
    elif strategy == "first":
        return values[0]
    elif strategy == "last":
        return values[-1]
    elif strategy == "max":
        return max(values)
    elif strategy == "min":
        return min(values)
    return values[0]


def _detect_medication(all_meds: List[str], keywords: List[str]) -> bool:
    """
    在用药列表中检测关键词（大小写不敏感子串匹配）。

    注意：dopamine 与 domperidone 无重叠子串，不会误匹配。
    epinephrine 是 norepinephrine 的子串，两者都是升压药，均需检测，无误报。
    """
    all_lower = " ".join(str(m).lower() for m in all_meds if m)
    return any(kw in all_lower for kw in keywords)


# ═══════════════════════════════════════════════════════════════
# 核心计算逻辑
# ═══════════════════════════════════════════════════════════════

def compute_supplementary_features(patient_json: Dict) -> Dict:
    """
    从患者 JSON 中已有字段计算补充特征。

    Returns:
        {
            "kidney": {"creatinine_peak": ..., "aki_kdigo_stage": ...},
            "liver":  {"alt": ..., "ast": ..., "bilirubin_total": ...,
                       "data_available": ..., "liver_dysfunction": ...},
            "dm_complications": {"on_corticosteroids": ..., "vasopressor_use": ...},
        }
    """
    db = str(patient_json.get("database", "")).lower()
    if any(k in db for k in ["mimic4", "mimic_iv", "mimic-iv"]):
        itemid_map = _MIMIC4_LAB_MAP
    elif any(k in db for k in ["mimic3", "mimic_iii", "mimic-iii"]):
        itemid_map = _MIMIC3_LAB_MAP
    else:
        itemid_map = {}  # eICU: 纯关键词回退

    labs = patient_json.get("all_lab_results", []) or []
    prior_meds = patient_json.get("prior_medications", []) or []
    discharge_meds = patient_json.get("discharge_medications", []) or []
    all_meds = list(prior_meds) + list(discharge_meds)

    cm = patient_json.get("clinical_metrics", {}) or {}
    kidney = cm.get("kidney", {}) or {}

    # ── 1. creatinine_peak & KDIGO AKI 分期 ──────────────────
    # baseline = kidney.creatinine（extraction pipeline 已计算的入院 24h 最低肌酐）
    # peak = 48h 窗口内最高肌酐
    # KDIGO 2012 比率阈值（使用入院最低值作为 baseline 的近似）：
    #   Stage 3: ratio ≥ 3.0
    #   Stage 2: ratio ≥ 2.0
    #   Stage 1: ratio ≥ 1.5  或  绝对升幅 ≥ 0.3 mg/dL（在48h内）
    #
    # 局限性声明（用于论文 Methods 章节）：
    #   - 真实 KDIGO 基线肌酐应来自 7 天前的稳态值，此处以入院 24h 最低值代替
    #   - 仅覆盖入院后 48h 窗口，无法检测入院后新发 AKI
    #   - 对 AKI-on-CKD 患者，分期可能被低估（baseline 已高于真实稳态）
    all_creat = _get_lab_values(labs, "creatinine", itemid_map, strategy="all")
    all_creat = [v for v in all_creat if isinstance(v, float) and v > 0]

    creat_peak = max(all_creat) if all_creat else None
    creat_min_all = min(all_creat) if all_creat else None

    # 优先用已计算的 baseline（入院24h最低值，更接近 AKI 前稳态）
    # 若不可用则用全量最低值
    baseline_creat = _safe_float(kidney.get("creatinine")) or creat_min_all

    aki_kdigo_stage: Optional[int] = None
    if (baseline_creat is not None and baseline_creat > 0
            and creat_peak is not None and creat_peak >= baseline_creat):
        ratio = creat_peak / baseline_creat
        # 48h 内绝对升幅（用全量最小值 vs 峰值近似；creat_min_all 临床上不会为 0）
        # 用 is not None 而非 or 短路，防御性保证数值语义正确
        abs_rise = (creat_peak - creat_min_all) if creat_min_all is not None else 0.0

        # KDIGO 2012/2022 AKI 分期（肌酐标准，仅覆盖住院期间数据）：
        #   Stage 1: ratio ≥ 1.5（7天内）或绝对升幅 ≥ 0.3 mg/dL（48h内）
        #   Stage 2: ratio ≥ 2.0
        #   Stage 3: ratio ≥ 3.0 或 SCr≥4.0 mg/dL 且伴急性升幅≥0.5（区分慢性高肌酐与急性恶化）
        #
        # 注意：此处 abs_rise≥0.5 额外限制仅用于 SCr≥4.0 判断，是有意更保守的阈值，
        # 目的是区分"CKD 慢性高基线（如3.8→4.2）"和"真实急性 Stage 3"，
        # 避免对 CKD G4-5 患者的假阳性分期（KDIGO 2022 Commentary 6.4）。
        # RRT 启动的 Stage 3 标准因当前 JSON 结构无 RRT 标志而未实现（方法局限性）。
        if ratio >= 3.0 or (creat_peak >= 4.0 and abs_rise >= 0.5):
            aki_kdigo_stage = 3
        elif ratio >= 2.0:
            aki_kdigo_stage = 2
        elif ratio >= 1.5 or abs_rise >= 0.3:
            aki_kdigo_stage = 1

    # ── 2. 肝功能三项 ──────────────────────────────────────────
    alt  = _get_lab_values(labs, "alt",             itemid_map, "first")
    ast  = _get_lab_values(labs, "ast",             itemid_map, "first")
    bili = _get_lab_values(labs, "bilirubin_total", itemid_map, "first")

    # 肝功能损害标志：任一项超过阈值即认为存在显著肝损
    # ADA 2024：ALT/AST > 3×ULN → metformin/TZD 相对禁忌
    # FDA：TZD 在 ALT > 2.5×ULN 时应避免
    # 此处统一使用 3×ULN 作为 "liver_dysfunction" 标志
    liver_dysfunction = (
        (alt  is not None and alt  > 3 * _ALT_ULN) or
        (ast  is not None and ast  > 3 * _AST_ULN) or
        (bili is not None and bili > 2.0)
    )

    # ── 3. 药物推断标志 ────────────────────────────────────────
    on_corticosteroids = _detect_medication(all_meds, _CORTICOSTEROID_KW)
    vasopressor_use    = _detect_medication(all_meds, _VASOPRESSOR_KW)

    return {
        "kidney": {
            "creatinine_peak":  creat_peak,
            "aki_kdigo_stage":  aki_kdigo_stage,
        },
        "liver": {
            "alt":              alt,
            "ast":              ast,
            "bilirubin_total":  bili,
            "data_available":   any(v is not None for v in [alt, ast, bili]),
            "liver_dysfunction": liver_dysfunction,
        },
        "dm_complications": {
            "on_corticosteroids": on_corticosteroids,
            "vasopressor_use":    vasopressor_use,
        },
    }


def apply_supplementary_updates(patient_json: Dict, updates: Dict) -> Dict:
    """
    将 compute_supplementary_features() 的输出增量写入 patient_json。

    规则：
      - kidney / dm_complications：仅写入本脚本新增字段，原有字段一律保留
      - liver：新建整个 section（不影响任何现有字段）
      - 写入 supplementary_version 标记
    """
    cm = patient_json.setdefault("clinical_metrics", {})
    if cm is None:
        patient_json["clinical_metrics"] = {}
        cm = patient_json["clinical_metrics"]

    # kidney section：增量写入
    kidney = cm.setdefault("kidney", {})
    if kidney is None:
        cm["kidney"] = {}
        kidney = cm["kidney"]
    for k, v in updates.get("kidney", {}).items():
        kidney[k] = v  # 覆盖本脚本负责的字段（重跑时更新到最新值）

    # liver section：整体新建（原来没有此 section）
    cm["liver"] = updates.get("liver", {})

    # dm_complications section：增量写入
    comp = cm.setdefault("dm_complications", {})
    if comp is None:
        cm["dm_complications"] = {}
        comp = cm["dm_complications"]
    for k, v in updates.get("dm_complications", {}).items():
        comp[k] = v

    cm["supplementary_version"] = SUPPLEMENTARY_VERSION
    return patient_json


# ═══════════════════════════════════════════════════════════════
# 文件级处理（原子写）
# ═══════════════════════════════════════════════════════════════

def process_file(filepath: Path, force: bool) -> str:
    """
    处理单个患者 JSON 文件。

    Returns:
        'processed' | 'skipped' | 'error'
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            patient_json = json.load(f)

        # 已处理且不强制重跑：跳过
        if not force:
            cm = patient_json.get("clinical_metrics", {}) or {}
            if cm.get("supplementary_version") == SUPPLEMENTARY_VERSION:
                return "skipped"

        updates = compute_supplementary_features(patient_json)
        patient_json = apply_supplementary_updates(patient_json, updates)

        _atomic_write(filepath, patient_json)
        return "processed"

    except json.JSONDecodeError as e:
        logger.error(f"JSON 解析失败 {filepath}: {e}")
        return "error"
    except OSError as e:
        logger.error(f"文件操作失败 {filepath}: {e}")
        return "error"
    except Exception as e:
        logger.error(f"处理失败 {filepath}: {e}", exc_info=True)
        return "error"


# ═══════════════════════════════════════════════════════════════
# UMLS 映射补丁
# ═══════════════════════════════════════════════════════════════
# 目标：对因 API 认证竞争（TGT 并发踩踏）导致 umls_cui=None 的诊断条目
#       做增量补丁，不重跑昂贵的数据库提取。
#
# 幂等策略：
#   - 只重试 umls_cui=None 且 icd_code 非空的诊断
#   - 本次运行后 remaining==0 → 写入 version 标记，下次跳过
#   - 本次仍有失败（remaining>0）→ 不写入 version，下次自动重试
#   - --force 强制忽略所有 version 标记，全量重试
#
# 线程安全：
#   - 共享单一 EnhancedUMLSMapper 实例
#   - TGT 刷新由 _tgt_lock（threading.Lock）保护，见 umls_relationship_mapper.py
# ═══════════════════════════════════════════════════════════════

def _load_json_safe(filepath: Path) -> Dict:
    """静默加载 JSON，失败时返回空 dict。"""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _load_umls_mapper():
    """
    初始化 EnhancedUMLSMapper（或带缓存版本）。
    从 config.APIConfig.UMLS_KEY 读取密钥。
    返回 mapper 实例，失败时返回 None。
    """
    try:
        from config import APIConfig
        umls_key = APIConfig.UMLS_KEY
    except Exception as e:
        logger.error(f"无法读取 UMLS_KEY（检查 config.py）: {e}")
        return None

    try:
        # 优先使用带 SQLite 缓存版本，减少重复 API 调用
        from baseline.enhanced_umls_mapper import EnhancedUMLSMapperWithCache
        mapper = EnhancedUMLSMapperWithCache(api_key=umls_key)
        logger.info("✅ UMLS mapper 初始化完成（带 SQLite 缓存）")
        return mapper
    except ImportError:
        pass

    try:
        from baseline.enhanced_umls_mapper import EnhancedUMLSMapper
        mapper = EnhancedUMLSMapper(api_key=umls_key)
        logger.info("✅ UMLS mapper 初始化完成（无缓存）")
        return mapper
    except ImportError as e:
        logger.error(f"无法导入 EnhancedUMLSMapper: {e}")
        return None


def _is_valid_icd(code) -> bool:
    """检查 ICD 编码是否值得尝试映射。"""
    if not code:
        return False
    s = str(code).strip().lower()
    return s not in ("", "nan", "none", "unknown")


def _count_patchable(patient_json: Dict) -> int:
    """统计需要 UMLS patch 的诊断数量（umls_cui=None 且 ICD 有效）。"""
    count = 0
    for diag in (patient_json.get("all_diagnoses") or []):
        if not isinstance(diag, dict):
            continue
        if diag.get("umls_cui"):
            continue
        if _is_valid_icd(diag.get("icd_code")):
            count += 1
    return count


def _patch_single_diag(diag: Dict, mapper) -> bool:
    """
    对单条诊断尝试 UMLS 映射，成功时原地更新 dict。
    返回 True 表示本次成功映射。
    """
    icd_code    = str(diag.get("icd_code", "")).strip()
    icd_version = diag.get("icd_version", 9)
    description = str(diag.get("description") or f"ICD_{icd_code}").strip()

    try:
        if icd_version == 9:
            result = mapper.map_icd9_to_standard(icd_code, description)
            # MappingResult 可能因缓存反序列化变为 dict，统一处理
            if isinstance(result, dict):
                cui = result.get("umls_cui")
                snomed_code = result.get("snomed_code")
                snomed_name = result.get("snomed_name")
            else:
                cui        = getattr(result, "umls_cui",    None) if result else None
                snomed_code = getattr(result, "snomed_code", None) if result else None
                snomed_name = getattr(result, "snomed_name", None) if result else None

            if cui:
                diag["umls_cui"]    = cui
                diag["snomed_code"] = snomed_code
                diag["snomed_name"] = snomed_name
                return True

        elif icd_version == 10:
            cui = mapper.find_cui_from_code(icd_code, "ICD10CM")
            if cui:
                diag["umls_cui"] = cui
                snomed_result = mapper.get_snomed_from_cui(cui)
                if snomed_result:
                    # get_snomed_from_cui 返回 (code, name) tuple 或缓存还原的 list
                    diag["snomed_code"] = snomed_result[0]
                    diag["snomed_name"] = snomed_result[1]
                return True

    except Exception as e:
        logger.debug(f"UMLS patch 单条失败 ({icd_code} v{icd_version}): {e}")

    return False


def patch_umls_for_patient(patient_json: Dict, mapper) -> Tuple[int, int]:
    """
    对一个患者 JSON 中所有 umls_cui=None 的诊断做 UMLS 补丁（原地修改）。

    Returns:
        (n_fixed, n_remaining)
    """
    n_fixed     = 0
    n_remaining = 0

    for diag in (patient_json.get("all_diagnoses") or []):
        if not isinstance(diag, dict):
            continue
        if diag.get("umls_cui"):          # 已有 CUI，跳过
            continue
        if not _is_valid_icd(diag.get("icd_code")):
            continue                       # 无有效 ICD，无法映射

        if _patch_single_diag(diag, mapper):
            n_fixed += 1
        else:
            n_remaining += 1

    return n_fixed, n_remaining


def process_file_umls_patch(filepath: Path, mapper, force: bool) -> Tuple[str, int, int]:
    """
    处理单个患者 JSON 文件的 UMLS patch。

    skip 条件（非 force）：
        clinical_metrics.umls_patch.version == UMLS_PATCH_VERSION
        AND remaining == 0

    Returns:
        (status, n_fixed, n_remaining)
        status ∈ {'processed', 'skipped', 'error'}
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            patient_json = json.load(f)

        # ── 快速 skip 检查 ───────────────────────────────────────
        if not force:
            patch_info = (
                (patient_json.get("clinical_metrics") or {})
                .get("umls_patch") or {}
            )
            if (patch_info.get("version") == UMLS_PATCH_VERSION
                    and patch_info.get("remaining", 1) == 0):
                return "skipped", 0, 0

        # ── 无需处理的文件（所有诊断已有 CUI）─────────────────────
        n_patchable = _count_patchable(patient_json)
        if n_patchable == 0:
            # 标记为干净，下次直接跳过
            cm = patient_json.setdefault("clinical_metrics", {}) or {}
            patient_json["clinical_metrics"] = cm
            cm["umls_patch"] = {
                "version":       UMLS_PATCH_VERSION,
                "remaining":     0,
                "fixed_this_run": 0,
                "last_run":      _now_iso(),
            }
            _atomic_write(filepath, patient_json)
            return "skipped", 0, 0

        # ── 执行 patch ────────────────────────────────────────────
        n_fixed, n_remaining = patch_umls_for_patient(patient_json, mapper)

        # 写入 patch 元信息
        cm = patient_json.setdefault("clinical_metrics", {}) or {}
        patient_json["clinical_metrics"] = cm
        cm["umls_patch"] = {
            # remaining==0 时才写 version，确保有失败时下次会重试
            "version":        UMLS_PATCH_VERSION if n_remaining == 0 else None,
            "remaining":      n_remaining,
            "fixed_this_run": n_fixed,
            "last_run":       _now_iso(),
        }

        _atomic_write(filepath, patient_json)

        if n_remaining > 0:
            logger.debug(
                f"[umls_patch] {filepath.name}: "
                f"fixed={n_fixed}, remaining={n_remaining}（将在下次重试）"
            )

        return "processed", n_fixed, n_remaining

    except json.JSONDecodeError as e:
        logger.error(f"JSON 解析失败 {filepath}: {e}")
        return "error", 0, 0
    except OSError as e:
        logger.error(f"文件操作失败 {filepath}: {e}")
        return "error", 0, 0
    except Exception as e:
        logger.error(f"UMLS patch 失败 {filepath}: {e}", exc_info=True)
        return "error", 0, 0


# ═══════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Dia-MAS 补充特征提取器 — 对已有患者 JSON 做增量补充计算"
    )
    parser.add_argument(
        "--cache_dir",
        default=str(PathConfig.PATIENT_CACHE_BASE),
        help=f"患者 JSON 缓存目录（默认：{PathConfig.PATIENT_CACHE_BASE}）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重跑（忽略已处理标记，重新计算所有补充特征）",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="并发线程数（默认：8）",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="试运行：只打印统计，不写入任何文件",
    )
    parser.add_argument(
        "--db_filter",
        default=None,
        choices=["mimic3", "mimic4", "eicu"],
        help="只处理指定数据库的患者（默认：处理全部）",
    )
    parser.add_argument(
        "--umls_patch",
        action="store_true",
        help=(
            "UMLS 映射补丁模式：对因 API 认证竞争导致 umls_cui=None 的诊断重新映射。"
            "与常规补充特征提取互斥，单独运行。"
            "配合 --force 可强制全量重映射。"
        ),
    )
    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    if not cache_dir.exists():
        logger.error(f"缓存目录不存在: {cache_dir}")
        return

    # 发现所有患者 JSON（排除进度文件和 .tmp 临时文件）
    all_files = [
        f for f in cache_dir.glob("**/*.json")
        if not f.name.startswith("_") and not f.suffix == ".tmp"
    ]

    # 按数据库过滤（依据路径包含的子目录名）
    if args.db_filter:
        all_files = [f for f in all_files if args.db_filter in str(f)]

    logger.info(
        f"发现 {len(all_files)} 个患者 JSON "
        f"{'（dry_run 模式，不写入）' if args.dry_run else ''}"
    )
    if not all_files:
        logger.warning("无文件需要处理，退出。")
        return

    # ══════════════════════════════════════════════════════════
    # 分支 A：UMLS 映射补丁模式
    # ══════════════════════════════════════════════════════════
    if args.umls_patch:
        if args.dry_run:
            # dry_run 下只统计需要修复的文件数
            n_need = sum(1 for f in all_files if _count_patchable(_load_json_safe(f)) > 0)
            logger.info(f"[dry_run] 预计需要 UMLS patch 的文件: {n_need}/{len(all_files)}")
            return

        mapper = _load_umls_mapper()
        if mapper is None:
            logger.error("UMLS mapper 初始化失败，中止 umls_patch 模式。")
            return

        stats: Dict[str, int] = {
            "processed": 0, "skipped": 0, "error": 0,
            "total_fixed": 0, "total_remaining": 0,
        }
        stats_lock = threading.Lock()

        def umls_task(fp: Path) -> None:
            status, n_fixed, n_remaining = process_file_umls_patch(fp, mapper, args.force)
            with stats_lock:
                stats[status]           += 1
                stats["total_fixed"]    += n_fixed
                stats["total_remaining"] += n_remaining

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(umls_task, f): f for f in all_files}
            for fut in tqdm(as_completed(futures), total=len(all_files),
                            desc="UMLS patch", unit="患者"):
                exc = fut.exception()
                if exc:
                    logger.error(f"未捕获异常：{exc}")

        logger.info(
            f"✅ UMLS patch 完成 | "
            f"处理={stats['processed']} | 跳过={stats['skipped']} | 错误={stats['error']} | "
            f"本次修复诊断={stats['total_fixed']} | "
            f"仍未映射={stats['total_remaining']}"
            + (f"（可再次执行 --umls_patch 重试）" if stats["total_remaining"] > 0 else " 🎉")
        )
        return

    # ══════════════════════════════════════════════════════════
    # 分支 B：常规补充特征提取（原有逻辑）
    # ══════════════════════════════════════════════════════════
    stats: Dict[str, int] = {"processed": 0, "skipped": 0, "error": 0}
    stats_lock = threading.Lock()

    def task(fp: Path) -> str:
        if args.dry_run:
            return "processed"
        result = process_file(fp, args.force)
        with stats_lock:
            stats[result] += 1
        return result

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(task, f): f for f in all_files}
        for fut in tqdm(as_completed(futures), total=len(all_files),
                        desc="补充特征提取", unit="患者"):
            exc = fut.exception()
            if exc:
                logger.error(f"未捕获异常：{exc}")

    logger.info(
        f"✅ 完成 | 处理={stats['processed']} | "
        f"跳过={stats['skipped']} | 错误={stats['error']}"
    )

    if stats["error"] > 0:
        logger.warning(
            f"有 {stats['error']} 个文件处理失败，详见日志。"
            f"可用 --force 重跑失败文件。"
        )


if __name__ == "__main__":
    main()
