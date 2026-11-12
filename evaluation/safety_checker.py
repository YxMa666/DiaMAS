#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluation/safety_checker.py
==============================
安全硬约束检查器（从 baseline_evaluation.py 抽出）

包含：
  - 辅助函数：_get_egfr, _get_dm_type, _has_complication, _get_admission_year
  - HARD_CONSTRAINTS: 15 条安全硬约束规则（HC-01 ~ HC-15）
  - _GAS_PATIENT_GATES: GAS Layer-1 患者特征门控表（对应 HC-XX）
  - _GAS_NEGATION_SIGNALS: GAS Layer-2 否定信号词
  - ConstraintViolation: 单条违反记录
  - HardConstraintChecker: 约束检查主类
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from evaluation.drug_normalizer import DrugNormalizer

logger = logging.getLogger(__name__)


def _is_trueish(val) -> bool:
    """宽松布尔解析：兼容 bool/int/str。"""
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        try:
            import math
            if isinstance(val, float) and math.isnan(val):
                return False
        except Exception:
            pass
        return val != 0
    return str(val).strip().lower() in ("true", "1", "yes", "y", "on")


def _cat_in(cat: Optional[str], *candidates: str) -> bool:
    """药物类别匹配（忽略大小写 + 连字符差异）。"""
    if not cat:
        return False
    norm = str(cat).strip().lower().replace("-", "")
    return any(norm == str(c).strip().lower().replace("-", "") for c in candidates)

# ── 严重程度权重（用于排序）────────────────────────────────────────────────
SEVERITY_LEVELS: Dict[str, int] = {
    "critical": 4,
    "major":    3,
    "moderate": 2,
    "warning":  1,
}


# ═══════════════════════════════════════════════════════════════════════════════
# 一、辅助函数（clinical_metrics 字段读取）
# ═══════════════════════════════════════════════════════════════════════════════

def _get_egfr(cm: Dict) -> Optional[float]:
    """从 clinical_metrics 中读取 eGFR 值。"""
    try:
        kidney = cm.get("kidney") or {}
        val = kidney.get("egfr") or kidney.get("eGFR") or cm.get("egfr")
        return float(val) if val is not None else None
    except (ValueError, TypeError):
        return None


def _get_dm_type(cm: Dict) -> str:
    """从 clinical_metrics 中读取 DM 分型，默认 T2DM。"""
    dm = cm.get("dm_classification") or cm.get("dm_type") or {}
    if isinstance(dm, dict):
        t = str(dm.get("type", "") or dm.get("dm_type", "") or "").upper()
    else:
        t = str(dm).upper()
    return "T1DM" if "T1DM" in t or "TYPE 1" in t or "TYPE1" in t else "T2DM"


def _has_complication(cm: Dict, key: str) -> bool:
    """
    检查并发症字段（兼容 clinical_metrics 顶层 + dm_complications 嵌套）。
    run_extraction/validator 实际会把并发症放在 cm["dm_complications"] 内。
    """
    key = str(key or "").strip().lower()
    dm_comp = cm.get("dm_complications") if isinstance(cm.get("dm_complications"), dict) else {}

    aliases = {
        "cardiovascular": ["cardiovascular", "ascvd", "cvd", "has_cardiovascular", "has_ascvd"],
        "heart_failure": ["heart_failure", "hf", "chf", "has_heart_failure"],
        "hypoglycemia": ["hypoglycemia", "hypoglycemia_history", "has_hypoglycemia", "hypo_history"],
        "nephropathy": ["nephropathy", "ckd", "diabetic_nephropathy"],
    }.get(key, [key, f"has_{key}", f"history_of_{key}", f"{key}_history"])

    # 先查 dm_complications（新版主路径）
    for k in aliases:
        if k in dm_comp and _is_trueish(dm_comp.get(k)):
            return True

    # 再查 clinical_metrics 顶层（兼容老缓存/外部扁平化特征）
    for k in aliases:
        if k in cm and _is_trueish(cm.get(k)):
            return True

    return False


def _get_admission_year(cm: Dict) -> int:
    """
    从 clinical_metrics 读取入院年份（兼容 run_extraction 缓存格式）。
    失败时返回 9999（安全默认：不触发“时代穿越”误判）。
    """
    # 1) 直接年份字段
    for key in ("admission_year", "anchor_year", "year"):
        y = cm.get(key)
        if y is None:
            continue
        try:
            yi = int(str(y)[:4])
            if 1900 <= yi <= 2100:
                return yi
        except (ValueError, TypeError):
            pass

    # 2) 从时间戳字段提取年份（如 "2014-03-01 ..."）
    for key in ("admission_time", "admittime", "discharge_time"):
        t = str(cm.get(key, "") or "")
        if len(t) >= 4 and t[:4].isdigit():
            yi = int(t[:4])
            if 1900 <= yi <= 2100:
                return yi

    # 3) MIMIC-IV anchor_year_group 可能是 "2014 - 2016"
    grp = str(cm.get("anchor_year_group", "") or "")
    if grp:
        m = re.search(r"(19\d{2}|20\d{2})", grp)
        if m:
            yi = int(m.group(1))
            if 1900 <= yi <= 2100:
                return yi

    return 9999


# ═══════════════════════════════════════════════════════════════════════════════
# 一之二、[Y1 FIX] 去标识化年份解析(单一权威来源)
# ═══════════════════════════════════════════════════════════════════════════════
# MIMIC 的 admission_time 经过脱敏偏移(MIMIC-IV ~2158, MIMIC-III ~2117)。
# 直接取 admission_time[:4] 会让全部 `yr >= N` 门控通过,使历史患者被现代规则评分。
# 本函数是【唯一】权威年份来源,任何需要"真实入院年份"的调用方都应使用它。

MAX_PLAUSIBLE_YEAR = 2023
"""超过此年份的值视为脱敏偏移产物,拒绝采纳(真实 cohort 上限 2022)。"""

MIMIC3_REPRESENTATIVE_YEAR = 2008   # = agents.config.EraConfig.MIMIC3_YEAR


def _plausible_year(y) -> "Optional[int]":
    """年份合理性检查:1990 <= y <= MAX_PLAUSIBLE_YEAR。否则 None。"""
    try:
        yi = int(str(y)[:4])
    except (ValueError, TypeError):
        return None
    return yi if 1990 <= yi <= MAX_PLAUSIBLE_YEAR else None


def resolve_admission_year(patient_data: "Dict") -> "Optional[int]":
    """真实入院年份 = agents.config.EraConfig.assigned_year(全系统唯一赋年)。无法赋年时返回 None。"""
    from agents.config import EraConfig
    return EraConfig.assigned_year(patient_data or {})


def evaluation_context(cm: "Dict", patient_data: "Optional[Dict]") -> "Dict":
    """硬约束判断用的 clinical_metrics 副本:补上赋年 admission_year 和 hemodynamic_instability
    (二者不在缓存的 clinical_metrics 里;缺了会让带年份门槛/血流动力学条件的约束永不触发)。"""
    out = dict(cm or {})
    if patient_data:
        from agents.config import hemodynamic_instability
        y = resolve_admission_year(patient_data)
        if y is not None:
            out["admission_year"] = y
        out.setdefault("hemodynamic_instability", hemodynamic_instability(patient_data))
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 二、HARD_CONSTRAINTS 规则表（HC-01 ~ HC-15）
# ═══════════════════════════════════════════════════════════════════════════════
# 结构说明：
#   id:         规则 ID（论文溯源用，Table S2）
#   name:       简短名称
#   drug_match: lambda(name_lower, category) -> bool   匹配该规则适用的药物
#   check:      lambda(name_lower, category, cm) -> bool  True = 违反
#   rule:       违反原因（临床描述）
#   severity:   critical / major / moderate / warning
#   source:     规则来源文献
# ═══════════════════════════════════════════════════════════════════════════════

HARD_CONSTRAINTS: List[Dict] = [
    # ── HC-01 ~ HC-02: Metformin ──────────────────────────────────────────────
    {
        "id": "HC-01", "name": "Metformin_eGFR<30",
        "drug_match": lambda name, cat: "metformin" in name,
        "check": lambda name, cat, cm: (
            (egfr := _get_egfr(cm)) is not None and egfr < 30
        ),
        "rule":     "Metformin 绝对禁忌：eGFR < 30 mL/min/1.73m²（乳酸酸中毒风险）",
        "severity": "critical",
        "source":   "ADA 2024 S9, FDA label",
    },
    {
        "id": "HC-02", "name": "Metformin_AKI",
        "drug_match": lambda name, cat: "metformin" in name,
        "check": lambda name, cat, cm: (
            (cm.get("kidney") or {}).get("aki_suspected", False)
        ),
        "rule":     "Metformin 禁忌：疑似 AKI（急性期乳酸酸中毒风险显著增加）",
        "severity": "critical",
        "source":   "FDA label, KDIGO AKI guidelines",
    },
    # ── HC-03 ~ HC-04: SGLT-2i ───────────────────────────────────────────────
    {
        "id": "HC-03", "name": "SGLT2i_eGFR<20",
        "drug_match": lambda name, cat: cat == "SGLT2i",
        "check": lambda name, cat, cm: (
            (egfr := _get_egfr(cm)) is not None and egfr < 20
        ),
        "rule":     "SGLT-2i 在 eGFR < 20 时降糖效果几乎消失，肾脏获益存疑",
        "severity": "major",
        "source":   "ADA 2024, 各药物说明书 eGFR cutoff",
    },
    {
        "id": "HC-04", "name": "SGLT2i_T1DM_no_DKA_plan",
        "drug_match": lambda name, cat: cat == "SGLT2i",
        "check": lambda name, cat, cm: _get_dm_type(cm) == "T1DM",
        "rule":     "SGLT-2i + T1DM：需警惕正常血糖 DKA 风险（euglycemic DKA）",
        "severity": "major",
        "source":   "FDA safety communication, ADA 2024",
    },
    # ── HC-05 ~ HC-06: 促泌剂（磺脲类/格列奈类）────────────────────────────
    {
        "id": "HC-05", "name": "Secretagogue_T1DM",
        "drug_match": lambda name, cat: cat in ("Sulfonylureas", "Meglitinides"),
        "check": lambda name, cat, cm: _get_dm_type(cm) == "T1DM",
        "rule":     "磺脲类/格列奈类 + T1DM：β细胞功能缺失，疗效极低且低血糖风险高",
        "severity": "major",
        "source":   "内分泌科共识，ADA Standards",
    },
    {
        "id": "HC-06", "name": "Secretagogue_Hypoglycemia_history",
        "drug_match": lambda name, cat: cat in ("Sulfonylureas", "Meglitinides"),
        "check": lambda name, cat, cm: _has_complication(cm, "hypoglycemia"),
        "rule":     "磺脲类/格列奈类 + 低血糖病史：显著增加反复低血糖事件风险",
        "severity": "moderate",
        "source":   "ADA 2024 hypoglycemia prevention",
    },
    # ── HC-07 ~ HC-08: TZD ────────────────────────────────────────────────────
    {
        "id": "HC-07", "name": "Rosiglitazone_CVD",
        "drug_match": lambda name, cat: "rosiglitazone" in name,
        "check": lambda name, cat, cm: _has_complication(cm, "cardiovascular"),
        "rule":     "Rosiglitazone + 心血管病：FDA 黑框警告（心肌梗死风险增加）",
        "severity": "major",  # [审计] FDA 2013 撤黑框(RECORD 再分析无显著 MI);与 HC-08(TZD+HF=major)一致,不再 critical→gas_f=0
        "source":   "FDA black box warning 2010, RECORD trial",
    },
    {
        "id": "HC-08", "name": "TZD_HF",
        "drug_match": lambda name, cat: cat == "Thiazolidinediones",
        "check": lambda name, cat, cm: _has_complication(cm, "heart_failure"),
        "rule":     "TZD + 心力衰竭/心血管病：水钠潴留加重心衰，NYHA III-IV 级禁用",
        "severity": "major",
        "source":   "ADA 2024, FDA label Pioglitazone/Rosiglitazone",
    },
    # ── HC-09: GLP-1RA ────────────────────────────────────────────────────────
    {
        "id": "HC-09", "name": "GLP1RA_renalClearance_lixi_exenatide",
        "drug_match": lambda name, cat: any(x in name for x in ("lixisenatide", "exenatide")),
        "check": lambda name, cat, cm: (
            (egfr := _get_egfr(cm)) is not None and egfr < 30
        ),
        "rule":     "利西拉肽/艾塞那肽（肾清除依赖）+ eGFR<30：避免使用；司美/利拉/度拉糖肽肾功能限制较少",
        "severity": "moderate",
        "source":   "内分泌专家审阅(FINAL-0262): GLP-1RA 不应全类 eGFR<15 一概禁用；按具体药物设肾功能限制。lixi/exenatide eGFR≤30 避免。 | orig ADA-EASD 2022",
    },
    # ── HC-10: DPP-4i 心衰风险 ──────────────────────────────────────────────
    {
        "id": "HC-10", "name": "Saxagliptin_Alogliptin_HF",
        "drug_match": lambda name, cat: any(x in name for x in ("saxagliptin", "alogliptin")),
        "check": lambda name, cat, cm: _has_complication(cm, "heart_failure"),
        "rule":     "Saxagliptin/Alogliptin + 心衰史：SAVOR-TIMI/EXAMINE 显示心衰住院率升高",
        "severity": "moderate",
        "source":   "SAVOR-TIMI 53 (2013), EXAMINE (2013)",
    },
    # ── HC-11 ~ HC-13: 时代约束 ───────────────────────────────────────────────
    {
        "id": "HC-11", "name": "Tirzepatide_pre_2022",
        "drug_match": lambda name, cat: cat == "GIP_GLP1_RA" or "tirzepatide" in name,
        "check": lambda name, cat, cm: (
            (y := _get_admission_year(cm)) < 2100 and y < 2022
        ),
        "rule":     "Tirzepatide 2022 年前尚未上市(时代错位,非安全禁忌;时代性由 era-aware 规则年份门处理)",
        "severity": "warning",  # [审计] 时代问题≠安全;critical 会误零 gas_f 且污染 A1a 禁忌构念,降 warning(仅记录,不罚分)
        "source":   "FDA approval 2022-05-13 (Mounjaro)",
    },
    {
        "id": "HC-12", "name": "Finerenone_pre_2021",
        "drug_match": lambda name, cat: "finerenone" in name,
        "check": lambda name, cat, cm: (
            (y := _get_admission_year(cm)) < 2100 and y < 2021
        ),
        "rule":     "Finerenone 2021 年前尚未上市(时代错位,非安全禁忌;由 era-aware 规则处理)",
        "severity": "warning",  # [审计] 同 HC-11:时代问题降 warning
        "source":   "FDA approval 2021-07-09 (Kerendia)",
    },
    {
        "id": "HC-13", "name": "Semaglutide_pre_2017",
        "drug_match": lambda name, cat: "semaglutide" in name,
        "check": lambda name, cat, cm: (
            (y := _get_admission_year(cm)) < 2100 and y < 2017
        ),
        "rule":     "Semaglutide 2017 年前尚未上市(时代错位,非安全禁忌;由 era-aware 规则处理)",
        "severity": "warning",  # [审计] 同 HC-11:时代问题降 warning
        "source":   "FDA approval 2017-12-05 (Ozempic)",
    },
    # ── HC-14: ACEI + ARB 联用（集合级别） ───────────────────────────────────
    {
        "id": "HC-14", "name": "ACEI_ARB_dual_blockade",
        "drug_match": lambda name, cat: cat in ("Hypertension_ACEi", "Hypertension_ARB"),
        "check": lambda name, cat, cm: False,   # 占位；实际检查在集合级别
        "rule":     "ACEI + ARB 双重 RAAS 阻断：高钾血症和肾功能恶化风险显著增加",
        "severity": "moderate",
        "source":   "ONTARGET trial, AHA/ACC guidelines",
    },
    # ── HC-15: T1DM 缺少胰岛素（集合级别反向约束） ─────────────────────────
    {
        "id": "HC-15", "name": "T1DM_missing_insulin",
        "drug_match": lambda name, cat: False,  # 占位；实际检查在集合级别
        "check": lambda name, cat, cm: False,
        "rule":     "T1DM 患者推荐方案中缺少胰岛素（T1DM 胰岛素依赖型）",
        "severity": "major",
        "source":   "ADA 2024 S9, T1DM 胰岛素绝对依赖原则",
    },
    # ── HC-16: 心衰患者应避免使用沙格列汀和阿格列汀，因其增加心衰住院风险。 ─────────
    # 来源规则 ID：DM-30  |  指南：ESC CVD in Diabetes 2023 | 5.3.2.1 DPP-4 inhibitors
    # ⚠️  未经专家确认，按置信度自动纳入（auto(high)，LLM confidence=high）
    {
        "id":         'HC-16',
        "name":       'DM_30',
        "disabled":   True,
        "disabled_reason": "DUPLICATE of HC-10 (saxagliptin/alogliptin + HF, all years); matched the whole DPP-4i class "
                           "against a saxagliptin/alogliptin rule and its year condition was inverted (<2014).",
        "drug_match": lambda name, cat: _cat_in(cat, "DPP-4i", "DPP4i"),
        "check":      lambda name, cat, cm: (
            _get_dm_type(cm) == "T2DM"
            and _has_complication(cm, "heart_failure")
            and (lambda y: y < 2100 and y < 2014)(_get_admission_year(cm))
        ),
        "rule":       '心衰患者应避免使用沙格列汀和阿格列汀，因其增加心衰住院风险。',
        "severity":   'major',
        "source":     'ESC CVD in Diabetes 2023 | 5.3.2.1 DPP-4 inhibitors',
    },
    # ── HC-17: HFrEF（LVEF≤40%）患者禁忌使用非二氢吡啶类CCB（地尔硫卓、维拉帕米），因与不良预后相关。 ─────────
    # 来源规则 ID：CPR-44  |  指南：ESC CVD in Diabetes 2023 | 12.4 Hypertension | 2021
    # ⚠️  未经专家确认，按置信度自动纳入（auto(high)，LLM confidence=high）
    {
        "id":         'HC-17',
        "name":       'CPR_44',
        "drug_match": lambda name, cat: cat == "Hypertension_CCB",
        "check":      lambda name, cat, cm: False,   # [审计-DISABLED] 见下
        "rule":       'HFrEF（LVEF≤40%）患者禁忌使用非二氢吡啶类CCB（地尔硫卓、维拉帕米），因与不良预后相关。',
        "severity":   'critical',
        "source":     'ESC CVD in Diabetes 2023 | 12.4 Hypertension | 2021',
        "disabled":   True,
        "disabled_reason": "TAXONOMY-BUG: drug_match=整个 Hypertension_CCB 类=氨氯地平/左旋氨氯地平/硝苯地平(均为 HF 安全的二氢吡啶类),规则本意针对非二氢吡啶类(地尔硫卓/维拉帕米,而后者在 DrugNormalizer→None 进不来)→只会对 HF+氨氯地平错误 critical 归零 gas_f。停用。",
    },
    # ── HC-18: T2DM合并心衰患者应避免使用沙格列汀 ─────────
    # 来源规则 ID：DM-32  |  指南：ESC CVD in Diabetes 2023 | 7.5.3. Dipeptidyl peptidase-4 inhibitors
    # ⚠️  未经专家确认，按置信度自动纳入（auto(high)，LLM confidence=high）
    # [DUPLICATE DISABLE] 2026-04-16:
    # Identical logic to HC-16 (T2DM + heart_failure + yr<2014 → DPP-4i violation).
    # Both fire on the same drug/patient combo → doubles the major violation count →
    # safety_penalty inflated by +0.15 for every correctly avoided DPP-4i scenario.
    # HC-16 is retained; HC-18 disabled to eliminate double-counting.
    {
        "id":         'HC-18',
        "name":       'DM_32',
        "drug_match": lambda name, cat: _cat_in(cat, "DPP-4i", "DPP4i"),
        "check":      lambda name, cat, cm: False,   # [DISABLED] duplicate of HC-16
        "rule":       'T2DM合并心衰患者应避免使用沙格列汀',
        "severity":   'major',
        "source":     'ESC CVD in Diabetes 2023 | 7.5.3. Dipeptidyl peptidase-4 inhibitors',
        "disabled":   True,
        "disabled_reason": "DUPLICATE: identical condition to HC-16 (T2DM+HF+yr<2014+DPP-4i); double-counting inflates safety_penalty.",
    },
    # ── HC-19: HFrEF（射血分数降低性心衰）患者禁忌使用地尔硫䓬和维拉帕米 ─────────
    # 来源规则 ID：CPR-56  |  指南：ESC CVD in Diabetes 2023 | 12.2.1 Medical therapy | 2021
    # ⚠️  未经专家确认，按置信度自动纳入（auto(high)，LLM confidence=high）
    {
        "id":         'HC-19',
        "name":       'CPR_56',
        "drug_match": lambda name, cat: cat == "Hypertension_CCB",
        "check":      lambda name, cat, cm: False,   # [审计-DISABLED] 见下
        "rule":       'HFrEF（射血分数降低性心衰）患者禁忌使用地尔硫䓬和维拉帕米',
        "severity":   'critical',
        "source":     'ESC CVD in Diabetes 2023 | 12.2.1 Medical therapy | 2021',
        "disabled":   True,
        "disabled_reason": "TAXONOMY-BUG: 同 HC-17,drug_match=整个 Hypertension_CCB 类(=HF 安全的二氢吡啶类);地尔硫卓/维拉帕米在 DrugNormalizer→None 进不来。只会误零 HF+氨氯地平的 gas_f。停用。",
    },
    # ── HC-20: 糖尿病患者应避免使用吉非贝齐，因其增加肌病风险 ─────────
    # 来源规则 ID：CPR-64  |  指南：ESC CVD in Diabetes 2023 2019 | 6.4.1.4 Fibrates
    # ⚠️  未经专家确认，按置信度自动纳入（auto(high)，LLM confidence=high）
    # [TAXONOMY FIX] 2026-04-16:
    # Original drug_match used cat == "Lipid_Fibrates" which is NOT in taxonomy.
    # Fibrates (gemfibrozil, fenofibrate) are classified as "Lipid_NonStatins" in config.py.
    # Fixed: check drug name directly for gemfibrozil (the specific drug of concern).
    # Fenofibrate has a safer myopathy profile and is not flagged here.
    {
        "id":         'HC-20',
        "name":       'CPR_64',
        "drug_match": lambda name, cat: "gemfibrozil" in name,   # [TAXONOMY FIX] was cat=="Lipid_Fibrates"
        "check":      lambda name, cat, cm: (
            _get_dm_type(cm) == "T2DM"
            and (lambda y: 2019 <= y < 2100)(_get_admission_year(cm))
        ),
        "rule":       '糖尿病患者应避免使用吉非贝齐，因其增加肌病风险（与他汀联用时尤甚）',
        "severity":   'major',
        "source":     'ESC CVD in Diabetes 2023 2019 | 6.4.1.4 Fibrates',
    },
    # ── HC-21: 射血分数降低的心衰（HFrEF）患者应使用 ACE 抑制剂 ─────────
    # 来源规则 ID：CPR-72  |  指南：ESC CVD in Diabetes 2023 | 5.3.1 Angiotensin-converting enzyme inhibitors | 2021
    # ⚠️  未经专家确认，按置信度自动纳入（auto(high)，LLM confidence=high）
    # [LOGIC-ERROR DISABLE] 2026-04-16:
    # check=True when HF + ACEi → triggers major violation when AI correctly recommends
    # ACEi for HFrEF. This is a POSITIVE OBLIGATION ("should use ACEi") misplaced in
    # HARD_CONSTRAINTS which is a NEGATIVE PROHIBITION table (check=True means "wrong drug").
    # Effect: every correct ACEi recommendation for HF patients incurs +0.15 safety_penalty,
    # and G-17 gate removes ACEi from GAS scoring entirely — both outcomes are incorrect.
    # Resolution: disable HC-21 and G-17. The positive obligation (ACEi for HFrEF) belongs
    # in CPR rules (positive scoring), not in safety constraints (negative penalties).
    {
        "id":         'HC-21',
        "name":       'CPR_72',
        "drug_match": lambda name, cat: cat == "Hypertension_ACEi",
        "check":      lambda name, cat, cm: False,   # [DISABLED] inverted logic — see comment above
        "rule":       '射血分数降低的心衰（HFrEF）患者应使用 ACE 抑制剂',
        "severity":   'major',
        "source":     'ESC CVD in Diabetes 2023 | 5.3.1 Angiotensin-converting enzyme inhibitors | 2021',
        "disabled":   True,
        "disabled_reason": "LOGIC-ERROR: positive obligation (ACEi recommended for HFrEF) placed in negative prohibition table; penalises correct ACEi recommendations for HF patients.",
    },
    # ── HC-22: 糖尿病合并症状性心衰患者不应使用噻唑烷二酮类药物 ─────────
    # 来源规则 ID：PR-08  |  指南：ESC CVD in Diabetes 2023 (2019) | 8.5.3 Thiazolidinediones
    # ⚠️  未经专家确认，按置信度自动纳入（auto(high)，LLM confidence=high）
    {
        "id":         'HC-22',
        "name":       'PR_08',
        "disabled":   True,
        "disabled_reason": "DUPLICATE of HC-08 (TZD + heart failure, all years).",
        "drug_match": lambda name, cat: cat == "Thiazolidinediones",
        "check":      lambda name, cat, cm: (
            _get_dm_type(cm) == "T2DM"
            and _has_complication(cm, "heart_failure")
            and (lambda y: 2019 <= y < 2100)(_get_admission_year(cm))
        ),
        "rule":       '糖尿病合并症状性心衰患者不应使用噻唑烷二酮类药物',
        "severity":   'moderate',
        "source":     'ESC CVD in Diabetes 2023 (2019) | 8.5.3 Thiazolidinediones',
    },
    # ── HC-23: T2DM合并心衰患者，不应使用DPP-4抑制剂或TZD ─────────
    # 来源规则 ID：PR-09  |  指南：ADA Consensus Reports & Supplements 2022 - Management of Hyperglycemia
    # ⚠️  未经专家确认，按置信度自动纳入（auto(high)，LLM confidence=high）
    {
        "id":         'HC-23',
        "name":       'PR_09',
        "disabled":   True,
        "disabled_reason": "DUPLICATE: TZD + HF is HC-08 and saxagliptin/alogliptin + HF is HC-10; the remaining "
                           "class-wide DPP-4i extension (sitagliptin, linagliptin) is not supported by their CVOTs.",
        "drug_match": lambda name, cat: (_cat_in(cat, "DPP-4i", "DPP4i") or cat == "Thiazolidinediones"),
        "check":      lambda name, cat, cm: (
            _get_dm_type(cm) == "T2DM"
            and _has_complication(cm, "heart_failure")
            and (lambda y: 2022 <= y < 2100)(_get_admission_year(cm))
        ),
        "rule":       'T2DM合并心衰患者，不应使用DPP-4抑制剂或TZD',
        "severity":   'moderate',
        "source":     'ADA Consensus Reports & Supplements 2022 - Management of Hyperglycemia',
    },
    # ── HC-24: T2DM患者若最大耐受他汀治疗仍未达标，应加用依折麦布 ─────────
    # 来源规则 ID：CPR-101
    # ⚠️  未经专家确认，按置信度自动纳入（auto(high)，LLM confidence=high）
    # [TAXONOMY+LOGIC DISABLE] 2026-04-16:
    # Issue 1 (taxonomy): cat == "Lipid_Ezetimibe" is NOT in taxonomy; ezetimibe maps to
    #   "Lipid_NonStatins" — drug_match never fires, rule is dead code.
    # Issue 2 (logic): "should add ezetimibe if statin insufficient" is a POSITIVE
    #   OBLIGATION, not a contraindication. Belongs in CPR rules, not HARD_CONSTRAINTS.
    # Also requires "on_max_statin" flag which is not populated by run_extraction.
    # Disabled pending correct implementation as a CPR rule.
    {
        "id":         'HC-24',
        "name":       'CPR_101',
        "drug_match": lambda name, cat: False,   # [DISABLED] taxonomy mismatch + logic error
        "check":      lambda name, cat, cm: False,
        "rule":       'T2DM患者若最大耐受他汀治疗仍未达标，应加用依折麦布',
        "severity":   'warning',
        "source":     '',
        "disabled":   True,
        "disabled_reason": "TAXONOMY: Lipid_Ezetimibe not in taxonomy (maps to Lipid_NonStatins); LOGIC: positive obligation misplaced in prohibition table; requires on_max_statin flag not populated.",
    },
    # ── HC-25: 合并心衰的老年T2DM患者，二甲双胍应保留在治疗方案中。 ─────────
    # 来源规则 ID：DM-47
    # ⚠️  未经专家确认，按置信度自动纳入（auto(high)，LLM confidence=high）
    # [LOGIC-ERROR DISABLE] 2026-04-16:
    # check=True when T2DM+HF+yr<2020 + Metformin → triggers violation when AI correctly
    # recommends Metformin for HF patients. "Metformin should be retained" is a POSITIVE
    # OBLIGATION, not a contraindication. Severity=warning means no GAS penalty (only
    # n_violations_critical/major count in safety_penalty), but G-21 gate will incorrectly
    # remove Metformin from GAS scoring for these patients.
    {
        "id":         'HC-25',
        "name":       'DM_47',
        "drug_match": lambda name, cat: False,   # [DISABLED] inverted logic — see comment above
        "check":      lambda name, cat, cm: False,
        "rule":       '合并心衰的老年T2DM患者，二甲双胍应保留在治疗方案中。',
        "severity":   'warning',
        "source":     '',
        "disabled":   True,
        "disabled_reason": "LOGIC-ERROR: positive obligation (Metformin should be retained for HF patients) placed in prohibition table; penalises correct Metformin recommendations.",
    },
    # ── HC-26 ~ HC-33: 新增硬约束（2026-05-16 基于 cpr_rules_final.xlsx 审查）──
    # 审查结果: 27 条候选规则 → 13 重复丢弃 | 4 重分类至 dm_rules | 2 合并 | 8 新增
    {
        "id": "HC-26", "name": "Chlorpropamide_renalImpairment",
        "drug_match": lambda name, cat: "chlorpropamide" in name,
        "check": lambda name, cat, cm: (
            (egfr := _get_egfr(cm)) is not None and egfr < 60
        ),
        "rule":     "Chlorpropamide 禁忌：eGFR < 60（活性代谢物蓄积 → 持续严重低血糖，可达 60-72h）",
        "severity": "critical",
        "source":   "ADA 2001-2008 Era-1 pharmacology; FDA label Diabinese",
        "applicable_from_year": 2001,
    },
    {
        "id": "HC-27", "name": "Troglitazone_withdrawn",
        "drug_match": lambda name, cat: "troglitazone" in name,
        "check": lambda name, cat, cm: True,
        "rule":     "Troglitazone 已于 2000-03-21 FDA 强制撤出市场（严重肝毒性/肝衰竭），绝对禁止处方",
        "severity": "critical",
        "source":   "FDA market withdrawal 2000-03-21 (Rezulin); Warner-Lambert Co.",
        "applicable_from_year": 2000,
    },
    {
        "id": "HC-28", "name": "Metformin_decompensatedHF",
        "drug_match": lambda name, cat: "metformin" in name,
        "check": lambda name, cat, cm: (
            _has_complication(cm, "heart_failure")
            and (
                _is_trueish(cm.get("hemodynamic_instability"))
                or (cm.get("kidney") or {}).get("aki_suspected", False)
            )
        ),
        "rule":     "Metformin 禁忌：失代偿性心衰伴血流动力学不稳定（组织灌注不足 → 乳酸酸中毒）",
        "severity": "critical",
        "source":   "FDA label update 2016; ADA 2024 S9; ESC HF 2021",
    },
    {
        "id": "HC-29", "name": "SGLT2i_eGFR<30_initiation",
        "drug_match": lambda name, cat: False,   # [DISABLED 2026-05-24 内分泌专家审阅 FINAL-0261]
        "check": lambda name, cat, cm: False,
        "rule":     "SGLT2i 起始禁忌：eGFR < 30（达格列净 label 禁用；恩格列净/卡格列净无显著降糖获益）",
        "severity": "major",
        "source":   "Forxiga/Farxiga US label; Jardiance label; KDIGO 2022 CKD guideline",
        "disabled": True,
        "disabled_reason": "内分泌专家审阅(FINAL-0261): eGFR<30 绝对禁用不符现行心肾保护指南；心肾保护可用至 eGFR≥20。eGFR<20 已由 HC-03/G-03 覆盖，此条过时，停用。",
    },
    {
        "id": "HC-30", "name": "SGLT2i_activeDKA",
        "drug_match": lambda name, cat: cat == "SGLT2i",
        "check": lambda name, cat, cm: (
            _has_complication(cm, "dka")        # 仅活动期 DKA/酮症；既往病史不再硬禁(专家意见 FINAL-0263)
        ),
        "rule":     "SGLT2i + 活动期 DKA/酮症：急性期停用（既往 DKA 病史改为慎用，不作绝对禁忌）",
        "severity": "major",
        "source":   "内分泌专家审阅(FINAL-0263): DKA 病史不宜作所有 SGLT2i 绝对禁忌；急性 DKA/酮症/长期禁食/围手术期/重症时停用。 | orig FDA 2015-05-15; ADA 2023+",
    },
    {
        "id": "HC-31", "name": "SGLT2i_hemodynamicInstability",
        "drug_match": lambda name, cat: cat == "SGLT2i",
        "check": lambda name, cat, cm: (
            _is_trueish(cm.get("hemodynamic_instability"))
        ),
        "rule":     "SGLT2i 急性期停用：血流动力学不稳定/休克期间暂缓（利尿加重低血容量；可诱发 euglycemic DKA）；稳定后复评，非永久禁用",
        "severity": "major",  # [审计] 与同族急性停用 HC-30(活动DKA)/HC-33(AKI)=major 一致;规则自述"临时停用非永久禁忌",不宜 critical→gas_f=0
        "source":   "ESC CVD in Diabetes 2023 Sec 5.3; ADA 2023 inpatient management | 内分泌专家审阅(FINAL-0377): 改为急性期临时停用/出院稳定后复评",
    },
    {
        "id": "HC-32", "name": "AGI_severeRenalImpairment",
        "drug_match": lambda name, cat: cat == "Alpha_Glucosidase_Inhibitors",
        "check": lambda name, cat, cm: (
            (egfr := _get_egfr(cm)) is not None and egfr < 25
        ),
        "rule":     "α糖苷酶抑制剂禁忌：eGFR < 25（阿卡波糖/米格列醇代谢物肾蓄积，FDA 说明书禁忌）",
        "severity": "major",
        "source":   "Precose (acarbose) FDA label; Glyset (miglitol) FDA label",
    },
    {
        "id": "HC-33", "name": "SGLT2i_AKI",
        "drug_match": lambda name, cat: cat == "SGLT2i",
        "check": lambda name, cat, cm: (
            (cm.get("kidney") or {}).get("aki_suspected", False)
        ),
        "rule":     "SGLT2i 急性期停用：AKI 活动期暂缓（加重肾脏低灌注；eGFR 骤降；可诱发 euglycemic DKA）；AKI 恢复后复评，非永久禁用",
        "severity": "major",
        "source":   "KDIGO AKI 2012; ADA 2024 Inpatient; ESC CVD Diabetes 2023 | 内分泌专家审阅(FINAL-1061): 改为急性期临时停用/出院稳定后复评",
    },

    # ══ HARDCODE INTERFACE：hardcode_rules.py --apply 自动追加 HC-XX 规则至此 ══
    # ── 请勿删除本注释块 ──
]


# ═══════════════════════════════════════════════════════════════════════════════
# 三、GAS Layer-1 患者特征门控表（_GAS_PATIENT_GATES）
# ═══════════════════════════════════════════════════════════════════════════════
# 镜像 HARD_CONSTRAINTS 的临床条件，用于 GAS 计算的强制清零。
# gate_id 与 HC-XX 一一对应（论文 Table S2）。

_GAS_PATIENT_GATES: List[Dict] = [
    {
        "gate_id":    "G-01",
        "drug_match": lambda name, cat: "metformin" in name,
        "condition":  lambda cm: (egfr := _get_egfr(cm)) is not None and egfr < 30,
        "reason":     "Metformin+eGFR<30: 乳酸酸中毒绝对禁忌",
    },
    {
        "gate_id":    "G-02",
        "drug_match": lambda name, cat: "metformin" in name,
        "condition":  lambda cm: (cm.get("kidney") or {}).get("aki_suspected", False),
        "reason":     "Metformin+AKI: 急性期乳酸酸中毒风险",
    },
    {
        "gate_id":    "G-03",
        "drug_match": lambda name, cat: cat == "SGLT2i",
        "condition":  lambda cm: (egfr := _get_egfr(cm)) is not None and egfr < 20,
        "reason":     "SGLT2i+eGFR<20: 降糖效果消失",
    },
    {
        "gate_id":    "G-04",
        "drug_match": lambda name, cat: "rosiglitazone" in name,
        "condition":  lambda cm: _has_complication(cm, "cardiovascular"),
        "reason":     "Rosiglitazone+CVD: FDA黑框警告",
    },
    {
        "gate_id":    "G-05",
        "drug_match": lambda name, cat: cat == "Thiazolidinediones",
        "condition":  lambda cm: _has_complication(cm, "heart_failure"),
        "reason":     "TZD+心衰/CVD: 水钠潴留",
    },
    {
        "gate_id":    "G-06",
        "drug_match": lambda name, cat: cat in ("Sulfonylureas", "Meglitinides"),
        "condition":  lambda cm: _get_dm_type(cm) == "T1DM",
        "reason":     "促泌剂+T1DM: β细胞功能缺失",
    },
    {
        "gate_id":    "G-07",
        "drug_match": lambda name, cat: any(x in name for x in ("saxagliptin", "alogliptin")),
        "condition":  lambda cm: _has_complication(cm, "heart_failure"),
        "reason":     "Saxagliptin/Alogliptin+心衰: 住院率升高",
    },
    {
        "gate_id":    "G-08",
        "drug_match": lambda name, cat: any(x in name for x in ("lixisenatide", "exenatide")),
        "condition":  lambda cm: (egfr := _get_egfr(cm)) is not None and egfr < 30,
        "reason":     "利西拉肽/艾塞那肽+eGFR<30: 肾清除依赖, 避免(专家 FINAL-0262; 原全类<15已撤)",
    },
    {
        "gate_id":    "G-09",
        "drug_match": lambda name, cat: cat == "GIP_GLP1_RA" or "tirzepatide" in name,
        "condition":  lambda cm: (
            (y := _get_admission_year(cm)) < 2100 and y < 2022
        ),
        "reason":     "Tirzepatide 2022年前未获批，时代穿越",
    },
    {
        "gate_id":    "G-10",
        "drug_match": lambda name, cat: "finerenone" in name,
        "condition":  lambda cm: (
            (y := _get_admission_year(cm)) < 2100 and y < 2021
        ),
        "reason":     "Finerenone 2021年前未获批，时代穿越",
    },
    {
        "gate_id":    "G-11",
        "drug_match": lambda name, cat: "semaglutide" in name,
        "condition":  lambda cm: (
            (y := _get_admission_year(cm)) < 2100 and y < 2017
        ),
        "reason":     "Semaglutide 2017年前未获批，时代穿越",
    },
    # ── G-12 ↔ HC-16: 心衰患者应避免使用沙格列汀和阿格列汀，因其增加心衰住院风险。 ─────────
    {
        "gate_id":    'G-12',
        "drug_match": lambda name, cat: False,   # [DISABLED] mirrors HC-16 (duplicate of G-07)
        "condition":  lambda cm: (
            _get_dm_type(cm) == "T2DM"
            and _has_complication(cm, "heart_failure")
            and (lambda y: y < 2100 and y < 2014)(_get_admission_year(cm))
        ),
        "reason":     '心衰患者应避免使用沙格列汀和阿格列汀，因其增加心衰住院风险。',
    },
    # ── G-13 ↔ HC-17: HFrEF（LVEF≤40%）患者禁忌使用非二氢吡啶类CCB（地尔硫卓、维拉帕米），因与不良预后相关 ─────────
    {
        "gate_id":    'G-13',
        "drug_match": lambda name, cat: False,   # [DISABLED] mirrors HC-17 (CCB class = HF-safe dihydropyridines)
        "condition":  lambda cm: (
            _get_dm_type(cm) == "T2DM"
            and _has_complication(cm, "heart_failure")
        ),
        "reason":     'HFrEF（LVEF≤40%）患者禁忌使用非二氢吡啶类CCB（地尔硫卓、维拉帕米），因与不良预后相关。',
    },
    # ── G-14 ↔ HC-18: T2DM合并心衰患者应避免使用沙格列汀 ─────────
    # [DUPLICATE DISABLE] 2026-04-16: identical to G-12 — double gate-out on same drug/patient.
    {
        "gate_id":    'G-14',
        "drug_match": lambda name, cat: False,   # [DISABLED] duplicate of G-12
        "condition":  lambda cm: False,
        "reason":     'T2DM合并心衰患者应避免使用沙格列汀 [DISABLED: duplicate of G-12]',
    },
    # ── G-15 ↔ HC-19: HFrEF（射血分数降低性心衰）患者禁忌使用地尔硫䓬和维拉帕米 ─────────
    {
        "gate_id":    'G-15',
        "drug_match": lambda name, cat: False,   # [DISABLED] mirrors HC-19 (CCB class = HF-safe dihydropyridines)
        "condition":  lambda cm: (
            _has_complication(cm, "heart_failure")
            and (lambda y: y < 2100 and y < 2021)(_get_admission_year(cm))
        ),
        "reason":     'HFrEF（射血分数降低性心衰）患者禁忌使用地尔硫䓬和维拉帕米',
    },
    # ── G-16 ↔ HC-20: 糖尿病患者应避免使用吉非贝齐，因其增加肌病风险 ─────────
    # [TAXONOMY FIX] 2026-04-16: cat == "Lipid_Fibrates" not in taxonomy (maps to Lipid_NonStatins).
    # Fixed: drug_match on "gemfibrozil" name directly, consistent with HC-20 fix.
    {
        "gate_id":    'G-16',
        "drug_match": lambda name, cat: "gemfibrozil" in name,   # [TAXONOMY FIX] was cat=="Lipid_Fibrates"
        "condition":  lambda cm: (
            _get_dm_type(cm) == "T2DM"
            and (lambda y: 2019 <= y < 2100)(_get_admission_year(cm))
        ),
        "reason":     '糖尿病患者应避免使用吉非贝齐，因其增加肌病风险（与他汀联用时尤甚）',
    },
    # ── G-17 ↔ HC-21: 射血分数降低的心衰（HFrEF）患者应使用 ACE 抑制剂 ─────────
    # [LOGIC-ERROR DISABLE] 2026-04-16: mirrors HC-21 inversion.
    # condition=True when HF + ACEi → gates out ACEi from GAS scoring for HF patients.
    # ACEi IS recommended for HFrEF; this gate incorrectly zeros out correct recommendations.
    {
        "gate_id":    'G-17',
        "drug_match": lambda name, cat: False,   # [DISABLED] inverted logic — see HC-21
        "condition":  lambda cm: False,
        "reason":     '射血分数降低的心衰（HFrEF）患者应使用 ACE 抑制剂 [DISABLED: logic inversion]',
    },
    # ── G-18 ↔ HC-22: 糖尿病合并症状性心衰患者不应使用噻唑烷二酮类药物 ─────────
    {
        "gate_id":    'G-18',
        "drug_match": lambda name, cat: False,   # [DISABLED] mirrors HC-22 (duplicate of G-05)
        "condition":  lambda cm: (
            _get_dm_type(cm) == "T2DM"
            and _has_complication(cm, "heart_failure")
            and (lambda y: 2019 <= y < 2100)(_get_admission_year(cm))
        ),
        "reason":     '糖尿病合并症状性心衰患者不应使用噻唑烷二酮类药物',
    },
    # ── G-19 ↔ HC-23: T2DM合并心衰患者，不应使用DPP-4抑制剂或TZD ─────────
    {
        "gate_id":    'G-19',
        "drug_match": lambda name, cat: False,   # [DISABLED] mirrors HC-23 (duplicate of G-05 / G-07)
        "condition":  lambda cm: (
            _get_dm_type(cm) == "T2DM"
            and _has_complication(cm, "heart_failure")
            and (lambda y: 2022 <= y < 2100)(_get_admission_year(cm))
        ),
        "reason":     'T2DM合并心衰患者，不应使用DPP-4抑制剂或TZD',
    },
    # ── G-20 ↔ HC-24: T2DM患者若最大耐受他汀治疗仍未达标，应加用依折麦布 ─────────
    # [TAXONOMY+LOGIC DISABLE] 2026-04-16: mirrors HC-24.
    # cat == "Lipid_Ezetimibe" not in taxonomy (ezetimibe maps to Lipid_NonStatins).
    # Also a positive obligation misplaced as a gate — would incorrectly block correct
    # ezetimibe recommendations. Disabled pending CPR rule implementation.
    {
        "gate_id":    'G-20',
        "drug_match": lambda name, cat: False,   # [DISABLED] taxonomy mismatch + logic inversion
        "condition":  lambda cm: False,
        "reason":     'T2DM患者若最大耐受他汀治疗仍未达标，应加用依折麦布 [DISABLED: taxonomy+logic]',
    },
    # ── G-21 ↔ HC-25: 合并心衰的老年T2DM患者，二甲双胍应保留在治疗方案中。 ─────────
    # [LOGIC-ERROR DISABLE] 2026-04-16: mirrors HC-25 inversion.
    # condition=True when T2DM+HF+yr<2020 + Metformin → gates out Metformin from GAS
    # scoring. Metformin IS appropriate (and recommended) for HF+T2DM patients with
    # adequate eGFR. This gate would penalise correct Metformin recommendations.
    {
        "gate_id":    'G-21',
        "drug_match": lambda name, cat: False,   # [DISABLED] inverted logic — see HC-25
        "condition":  lambda cm: False,
        "reason":     '合并心衰的老年T2DM患者，二甲双胍应保留在治疗方案中。 [DISABLED: logic inversion]',
    },
    # ══ HARDCODE INTERFACE：hardcode_rules.py --apply 自动追加 G-XX 门控至此 ══
    # ── 请勿删除本注释块 ──
]


# ═══════════════════════════════════════════════════════════════════════════════
# 四、GAS Layer-2 否定信号词
# ═══════════════════════════════════════════════════════════════════════════════

_GAS_NEGATION_SIGNALS: List[str] = [
    "should not", "do not", "must not", "cannot", "can not",
    "not recommended", "not indicated", "not be used", "not for use",
    "not be initiated", "do not initiate", "not initiate",
    "should be avoided", "avoid", "not approved", "not yet approved",
    "contraindicated", "is contraindicated", "are contraindicated",
    "absolute contraindication", "relative contraindication",
    "black box", "boxed warning", "fda warning", "fda safety",
    "increased risk of", "risk of hospitalization",
    "if egfr", "when egfr", "egfr <", "egfr<", "gfr <", "gfr<",
    "not for patients", "caution in patients",
    "caution when", "use with caution", "with caution",
    "in patients with heart failure", "in patients with hf",
    "not in patients with", "except in",
]


# ═══════════════════════════════════════════════════════════════════════════════
# 五、ConstraintViolation 和 HardConstraintChecker
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ConstraintViolation:
    """单条约束违反记录。"""
    constraint_id:   str
    constraint_name: str
    drug:            str
    category:        Optional[str]
    severity:        str
    rule:            str
    source:          str
    details:         Dict = field(default_factory=dict)


class HardConstraintChecker:
    """
    安全硬约束检查器

    设计原则：
      规则集完全独立于 LLM 推荐逻辑（避免循环论证）。
      DCEI-R 分量基于 deterministic 规则，不调用 LLM。
    """

    def __init__(self, normalizer: "DrugNormalizer"):
        self.normalizer = normalizer

    def check(
        self,
        recommended_drugs: List,        # mode_a 推荐列表（dict 或 str）
        cm:                Dict,         # clinical_metrics 字典
        patient_data:      Dict = None,  # 用于赋年和 hemodynamic_instability(见 evaluation_context)
    ) -> List[ConstraintViolation]:
        """
        对推荐药物集合检查安全硬约束（HC-01 ~ HC-25，部分已 disabled）。

        Disabled: HC-16/HC-18/HC-23 (duplicates of HC-08/HC-10), HC-22 (duplicate of HC-08), HC-17/HC-19
        (CCB taxonomy), HC-21/HC-24/HC-25 (logic), HC-29 (expert review). Year conditions use the admission's
        assigned year.

        Returns:
            违反记录列表（按严重程度降序排列）。
        """
        # HC IDs skipped in per-drug loop:
        #   HC-14/HC-15: collection-level checks (handled in _check_combination)
        #   HC-18: duplicate of HC-16 (double-counts DPP-4i+HF violation)
        #   HC-21: logic inversion (ACEi+HF is correct, not a violation)
        #   HC-24: taxonomy mismatch + positive obligation misplaced
        #   HC-25: logic inversion (Metformin+HF is correct, not a violation)
        _SKIP_IDS = {"HC-14", "HC-15", "HC-17", "HC-18", "HC-19", "HC-21", "HC-24", "HC-25"}

        cm = evaluation_context(cm, patient_data)
        violations: List[ConstraintViolation] = []

        # Step 1: 单药检查
        for drug_rec in recommended_drugs:
            if isinstance(drug_rec, dict):
                drug_name = str(drug_rec.get("drug", "") or drug_rec.get("name", "")).strip()
            elif isinstance(drug_rec, str):
                drug_name = drug_rec.strip()
            else:
                drug_name = str(drug_rec).strip()
            if not drug_name:
                continue

            mapping    = self.normalizer.normalize(drug_name)
            drug_lower = drug_name.lower()
            drug_cat   = mapping.drug_category or (
                drug_rec.get("category", "") if isinstance(drug_rec, dict) else ""
            )

            for constraint in HARD_CONSTRAINTS:
                if constraint["id"] in _SKIP_IDS or constraint.get("disabled"):
                    continue  # 集合级别 / disabled 规则，跳过单药检查

                if not constraint["drug_match"](drug_lower, drug_cat):
                    continue

                try:
                    if constraint["check"](drug_lower, drug_cat, cm):
                        violations.append(ConstraintViolation(
                            constraint_id   = constraint["id"],
                            constraint_name = constraint["name"],
                            drug            = drug_name,
                            category        = drug_cat or None,
                            severity        = constraint["severity"],
                            rule            = constraint["rule"],
                            source          = constraint["source"],
                            details         = {
                                "egfr":    _get_egfr(cm),
                                "dm_type": _get_dm_type(cm),
                                "year":    _get_admission_year(cm),
                            },
                        ))
                except Exception as e:
                    logger.debug(f"约束检查异常 {constraint['id']} / {drug_name}: {e}")

        # Step 2: 集合级别检查（HC-14, HC-15）
        violations += self._check_combination(recommended_drugs, cm)

        violations.sort(key=lambda v: SEVERITY_LEVELS.get(v.severity, 0), reverse=True)
        return violations

    def _check_combination(
        self,
        recommended_drugs: List,
        cm:                Dict,
    ) -> List[ConstraintViolation]:
        """集合级别约束（HC-14 ACEI+ARB，HC-15 T1DM 缺胰岛素）。"""
        violations: List[ConstraintViolation] = []

        drug_names: List[str] = []
        for r in recommended_drugs:
            if isinstance(r, dict):
                drug_names.append(str(r.get("drug", "") or r.get("name", "")).lower())
            elif isinstance(r, str):
                drug_names.append(r.lower())
            else:
                drug_names.append(str(r).lower())

        # HC-14: ACEI + ARB
        acei = [d for d in drug_names
                if any(x in d for x in ("lisinopril", "enalapril", "ramipril",
                                         "benazepril", "perindopril", "captopril"))]
        arb  = [d for d in drug_names
                if any(x in d for x in ("losartan", "valsartan", "irbesartan",
                                         "candesartan", "olmesartan", "telmisartan"))]
        if acei and arb:
            violations.append(ConstraintViolation(
                constraint_id   = "HC-14",
                constraint_name = "ACEI_ARB_dual_blockade",
                drug            = f"{acei[0]} + {arb[0]}",
                category        = "combination",
                severity        = "moderate",
                rule            = "ACEI + ARB 双重 RAAS 阻断：高钾血症和肾功能恶化风险",
                source          = "ONTARGET trial",
                details         = {"acei": acei, "arb": arb},
            ))

        # HC-15: T1DM 缺胰岛素
        if _get_dm_type(cm) == "T1DM":
            has_insulin = any(
                self.normalizer.normalize(d).drug_category in (
                    "Insulins_Basal", "Insulins_Rapid"
                )
                for d in drug_names
            )
            if not has_insulin:
                violations.append(ConstraintViolation(
                    constraint_id   = "HC-15",
                    constraint_name = "T1DM_missing_insulin",
                    drug            = "（整体推荐方案）",
                    category        = "combination",
                    severity        = "major",
                    rule            = "T1DM 患者推荐方案中完全缺少胰岛素覆盖",
                    source          = "ADA 2024 S9",
                    details         = {"dm_type": "T1DM", "rec_drugs": drug_names},
                ))

        return violations
    
# ═══════════════════════════════════════════════════════════════════════════════
# 六、DrugInteractionAuditor — 跨药物 DDI 阻断（Bug-3）
# ═══════════════════════════════════════════════════════════════════════════════
#
# 设计原则：
#   - Tier-1 硬编码规则（deterministic，可引用指南出处，论文 Table S3）
#   - Critical 级别：直接从 final_list 移除冲突药物（阻断性）
#   - Major 级别：写入 ddi_warnings 字段，不阻断（审计可用于 RC-4 分类）
#   - 调用方：diamas_inference_stage.recommend() 在 Bug-6 merge 之后、return 之前
#
# 调用示例：
#   auditor = DrugInteractionAuditor()
#   filtered, warnings = auditor.filter_and_warn(drug_list, flat)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DDIWarning:
    """单条 DDI 记录。"""
    ddi_id:        str
    ddi_name:      str
    severity:      str          # "critical" | "major"
    removed_drugs: List[str]    # critical 时被移除的药物名列表；major 时为空
    trigger_drugs: List[str]    # 触发 DDI 的药物名列表
    reason:        str
    source:        str


def _get_drug_cat(drug_entry, normalizer=None) -> str:
    """从 drug dict 中读取 category；优先 normalizer 映射，回退到 category 字段。"""
    if isinstance(drug_entry, dict):
        name = str(drug_entry.get("drug", "") or "").strip()
        cat  = str(drug_entry.get("category", "") or "").strip()
    else:
        name = str(drug_entry).strip()
        cat  = ""
    if normalizer and name:
        mapping = normalizer.normalize(name)
        if mapping.drug_category:
            return mapping.drug_category
    return cat


def _get_drug_name(drug_entry) -> str:
    if isinstance(drug_entry, dict):
        return str(drug_entry.get("drug", "") or "").strip()
    return str(drug_entry).strip()


# ── DDI 规则表 ─────────────────────────────────────────────────────────────────
# 每条规则：id / name / severity / detect(cats, names, flat) / resolve / source
_DDI_RULES: List[Dict] = [
    # ── DDI-01: 双重 RAAS 阻断（ACEi + ARB）──────────────────────────────────
    {
        "id":       "DDI-01",
        "name":     "Dual_RAAS_ACEi_ARB",
        "severity": "critical",
        "detect":   lambda cats, names, flat: (
            any(c == "Hypertension_ACEi" for c in cats) and
            any(c == "Hypertension_ARB"  for c in cats)
        ),
        "resolve":  "remove_arb",   # 保留 ACEi（KDIGO/ADA 一线），移除 ARB
        "reason":   "ACEi + ARB 双重RAAS阻断 → 急性肾损伤/高钾血症风险显著升高",
        "source":   "ONTARGET 2008; ADA 2024 S10; KDIGO CKD 2012 §4.1",
    },
    # ── DDI-02: SGLT2i + 活动性 AKI ──────────────────────────────────────────
    {
        "id":       "DDI-02",
        "name":     "SGLT2i_Active_AKI",
        "severity": "critical",
        "detect":   lambda cats, names, flat: (
            any(c == "SGLT2i" for c in cats) and
            bool((flat.get("kidney") or flat).get("aki_suspected", False))
        ),
        "resolve":  "remove_sglt2i",
        "reason":   "SGLT2i + 活动性AKI → 渗透性利尿加重肾小管损伤",
        "source":   "KDIGO AKI 2012; ADA 2024 S9; FDA SGLT2i label",
    },
    # ── DDI-03: TZD + 心力衰竭 ──────────────────────────────────────────────
    {
        "id":       "DDI-03",
        "name":     "TZD_Heart_Failure",
        "severity": "critical",
        "detect":   lambda cats, names, flat: (
            any(c == "Thiazolidinediones" for c in cats) and
            _has_complication(flat, "heart_failure")
        ),
        "resolve":  "remove_tzd",
        "reason":   "TZD + 心力衰竭 → 水钠潴留加重心衰（NYHA III-IV 绝对禁忌）",
        "source":   "FDA black box warning (Pioglitazone/Rosiglitazone); ADA 2024 S9",
    },
    # ── DDI-04: 重复他汀处方 ────────────────────────────────────────────────
    {
        "id":       "DDI-04",
        "name":     "Duplicate_Statin",
        "severity": "major",
        "detect":   lambda cats, names, flat: (
            sum(1 for c in cats if c == "Lipid_Statins") > 1
        ),
        "resolve":  "warn_only",    # major → 不阻断，写警告
        "reason":   "重复他汀处方（同一患者推荐了2种以上他汀），仅保留高强度他汀",
        "source":   "临床用药规范；ACC/AHA 2013 statin guidelines",
    },
    # ── DDI-05: Beta-blocker + 重度 COPD/哮喘 ───────────────────────────────
    {
        "id":       "DDI-05",
        "name":     "NonSelective_BetaBlocker_Severe_COPD",
        "severity": "major",
        "detect":   lambda cats, names, flat: (
            any(c == "Hypertension_BetaBlockers" for c in cats) and
            any(x in " ".join(names).lower()
                for x in ("propranolol", "carvedilol", "labetalol")) and
            _has_complication(flat, "copd")
        ),
        "resolve":  "warn_only",
        "reason":   "非选择性β受体阻滞剂 + COPD/哮喘 → 支气管痉挛风险（应换用选择性β1-blocker）",
        "source":   "GOLD 2023; ADA 2024 S10; ESC 2019 HF guidelines",
    },
]

_STATIN_INTENSITY = {
    "rosuvastatin": 3, "atorvastatin": 3,
    "simvastatin": 2,  "pravastatin": 2, "lovastatin": 2, "pitavastatin": 2,
}


class DrugInteractionAuditor:
    """
    跨药物 DDI 阻断审计器（Bug-3）。

    Critical 级别规则直接从推荐列表移除冲突药物；
    Major 级别只生成警告，不改变列表。
    """

    def __init__(self, normalizer=None):
        self._normalizer = normalizer

    def filter_and_warn(
        self,
        drug_list: List,
        flat:      Dict,
    ) -> tuple:  # (filtered: List, warnings: List[DDIWarning])
        """
        对 drug_list 应用 DDI 规则，返回 (filtered_list, ddi_warnings)。

        Args:
            drug_list: combined recommended_drugs（已含并发症药，Bug-6 merge 后）
            flat:      flatten_clinical_metrics 输出 / clinical_metrics dict

        Returns:
            filtered_list: 移除 critical DDI 冲突药后的列表
            ddi_warnings:  所有 DDI 记录（critical + major）
        """
        if not drug_list:
            return drug_list, []

        cats  = [_get_drug_cat(d, self._normalizer) for d in drug_list]
        names = [_get_drug_name(d).lower() for d in drug_list]
        warnings: List[DDIWarning] = []
        remove_set: set = set()   # drug name (lower) to remove

        for rule in _DDI_RULES:
            try:
                triggered = rule["detect"](cats, names, flat)
            except Exception as e:
                logger.debug(f"DDI detect error {rule['id']}: {e}")
                triggered = False
            if not triggered:
                continue

            resolve  = rule["resolve"]
            severity = rule["severity"]

            removed_names: List[str] = []

            if severity == "critical":
                if resolve == "remove_arb":
                    for i, c in enumerate(cats):
                        if c == "Hypertension_ARB":
                            n = _get_drug_name(drug_list[i])
                            removed_names.append(n)
                            remove_set.add(n.lower())

                elif resolve == "remove_sglt2i":
                    for i, c in enumerate(cats):
                        if c == "SGLT2i":
                            n = _get_drug_name(drug_list[i])
                            removed_names.append(n)
                            remove_set.add(n.lower())

                elif resolve == "remove_tzd":
                    for i, c in enumerate(cats):
                        if c == "Thiazolidinediones":
                            n = _get_drug_name(drug_list[i])
                            removed_names.append(n)
                            remove_set.add(n.lower())

            trigger_names = [
                _get_drug_name(drug_list[i])
                for i, c in enumerate(cats)
                if c in ("Hypertension_ACEi", "Hypertension_ARB",
                         "SGLT2i", "Thiazolidinediones",
                         "Hypertension_BetaBlockers", "Lipid_Statins")
            ]

            warnings.append(DDIWarning(
                ddi_id        = rule["id"],
                ddi_name      = rule["name"],
                severity      = severity,
                removed_drugs = removed_names,
                trigger_drugs = trigger_names,
                reason        = rule["reason"],
                source        = rule["source"],
            ))

        filtered = [
            d for d in drug_list
            if _get_drug_name(d).lower() not in remove_set
        ]
        return filtered, warnings
