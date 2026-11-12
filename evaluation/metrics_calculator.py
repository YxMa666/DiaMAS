#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluation/metrics_calculator.py
==================================
三层指标计算器（v3 — mode_b 完整对称修复）

变更记录
--------
v2: GT 过滤修复（filter_dm_relevant_drugs），rxf_*/ccs_* 语义明确化
v3（本版）:
  PatientMetrics 补全以下 mode_b 对称字段（原版全部缺失）：
    ccs_dm_only_category_f1  — mode_b 纯降糖药类别 F1
    ccs_atc_l4/l3/l2/l1_f1  — mode_b ATC 层次金字塔（L2 是 CII 关键指标）
    ccs_n_violations_critical/major/moderate — mode_b 安全违规计数
  MetricsCalculator.calculate() 对应补全：
    ① mode_b dm_only 类别 F1
    ② mode_b ATC 层次 F1（extensions 第二次调用，仅 mode_b 真实可用时）
    ③ mode_b 安全违规（checker.check(mode_b_raw, ...)）

设计原则
--------
  mode_a（Rx Fidelity / GAS 评估基准）与 mode_b（CCS / CII 分析基准）
  必须在所有层次上完全对称计算。
  mode_b 安全违规率 > mode_a 安全违规率 → 真实处方比指南最优解更危险
  → 这是 C5（CII）贡献的核心安全实证，之前完全缺失。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from evaluation.drug_normalizer import DrugNormalizer, DrugMapping
    from evaluation.safety_checker import HardConstraintChecker, ConstraintViolation

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 一、辅助解析函数
# ═══════════════════════════════════════════════════════════════════════════════

# ── 评估类别规范化（对称施加于 GT 与推荐两侧）────────────────────────────────
# 修正两个评分口径 artifact：
#   ① 胰岛素子类拆散：出院记录多为 Insulins_Unspecified，模型给 Rapid/Basal → 跨类不匹配。
#      对策：所有 Insulins_* 合并为单一 "Insulin"（GT 未指定时，模型任何子类都算命中）。
#   ② Hypoglycemia_Rescue：PRN/护理协议医嘱，非处方决策，模型几乎不输出 → 从评分 scope 排除。
# 论文方法学须说明此归一化；与 GAS 等其它类别级评估路径应共用同一规则以保持口径一致。
CANON_INSULIN_COLLAPSE = True
CANON_EXCLUDE_CATS = frozenset({"Hypoglycemia_Rescue"})


def canon_eval_category(cat):
    """GT/推荐两侧对称施加；幂等：canon(canon(x)) == canon(x)。"""
    if not cat:
        return None
    if cat in CANON_EXCLUDE_CATS:
        return None
    if CANON_INSULIN_COLLAPSE and str(cat).startswith("Insulins_"):
        return "Insulin"
    return cat


def _canon_scope(cats):
    """把 scope 类别集合规范化（供成员判定用）。"""
    return frozenset(c for c in (canon_eval_category(x) for x in cats) if c)


def _cat_set(mappings, scope=None):
    """从 DrugMapping 列表得到规范化后的类别集合；scope 非 None 时按 scope 过滤。"""
    out = set()
    for m in mappings:
        c = canon_eval_category(getattr(m, "drug_category", None))
        if c and (scope is None or c in scope):
            out.add(c)
    return out


def _extract_drug_list_from_rec(rec_result: Dict, mode: str = "auto") -> List[Dict]:
    """
    从推荐结果 JSON 中提取药物列表。
    
    dual_mode_raw 结构（diamas v5 双模式输出）：
      {"mode_a": {"recommended_drugs": [...], "reasoning": "..."},
       "mode_b": {"recommended_drugs": [...], "reasoning": "..."},
       "cii_divergence": [...]}
    即 dual_raw["mode_a"] 是 dict，不是 list，需要解包一层。
    """
    dual_raw = (rec_result.get("diamas_metadata", {}) or {}).get("dual_mode_raw") or {}
    if dual_raw and isinstance(dual_raw, dict):
        key = "mode_b_recommendations" if mode == "mode_b" else "mode_a_recommendations"
        alt = "mode_b" if mode == "mode_b" else "mode_a"
        recs = dual_raw.get(key) or dual_raw.get(alt) or []
        # ── FIX: LLM 输出是 {"recommended_drugs": [...]} 嵌套结构，不是平坦列表 ──
        if isinstance(recs, dict):
            recs = recs.get("recommended_drugs") or []
        if recs:
            return recs if isinstance(recs, list) else []
        # recs 为空（该 mode 确实无输出）→ fallthrough 到 top-level
    # Bug-6: DiaMAS v7 将并发症药拆到独立字段，合并后返回（与 baseline 单字段格式兼容）
    top  = rec_result.get("recommended_drugs", []) or []
    meta = rec_result.get("diamas_metadata", {}) or {}
    mdt_comp = meta.get("mdt_final_complication_drugs") or []
    comp     = mdt_comp if mdt_comp else (rec_result.get("complication_drugs", []) or [])
    combined = list(top) + [c for c in comp if c]
    # 按药名去重（baseline 格式不会有重复，DiaMAS 合并后可能有）
    seen, merged = set(), []
    for item in combined:
        name = (item.get("drug", str(item)) if isinstance(item, dict) else str(item)).strip().lower()
        if name and name not in seen:
            seen.add(name)
            merged.append(item)
    return merged


def _extract_cii_divergence_from_rec(rec_result: Dict) -> List[Dict]:
    """从推荐结果 JSON 中提取 cii_divergence 列表。"""
    dual_raw = (rec_result.get("diamas_metadata", {}) or {}).get("dual_mode_raw") or {}
    return [d for d in (dual_raw.get("cii_divergence") or []) if isinstance(d, dict)]


def _names_from_rec_list(rec_list: List) -> List[str]:
    """从推荐列表（dict 或 str 混合）中提取药物名。"""
    names: List[str] = []
    for r in rec_list:
        if isinstance(r, dict):
            name = str(r.get("drug", "") or r.get("name", "") or r.get("drug_name", "")).strip()
        elif isinstance(r, str):
            name = r.strip()
        else:
            name = str(r).strip()
        if name:
            names.append(name)
    return names


def _safe_f1(v: float) -> float:
    """NaN 安全转换（NaN → 0.0）。"""
    if v is None:
        return 0.0
    try:
        fv = float(v)
        return 0.0 if (fv != fv) else fv   # NaN check
    except (TypeError, ValueError):
        return 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 二、PatientMetrics（v3 — mode_b 完整对称）
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PatientMetrics:
    """
    单患者评测结果（admission-level）。

    字段命名约定：
      rxf_*  = mode_a vs GT（Rx Fidelity，Good Outcome 主指标）
      ccs_*  = mode_b vs GT（Clinical Concordance Score，CII 分析基准）
      （无前缀）= mode_a（与旧版兼容）
    """
    patient_id:  str
    subject_id:  int
    hadm_id:     int
    model_name:  str
    era:         str
    db_source:   str

    # ── 推荐输入/输出 ─────────────────────────────────────────────────────────
    recommended_drugs:  List[str] = field(default_factory=list)
    actual_drugs:       List[str] = field(default_factory=list)
    rec_categories:     List[str] = field(default_factory=list)
    actual_categories:  List[str] = field(default_factory=list)

    # ══════════════════════════════════════════════════════════════════════════
    # 层次一：类别级 F1
    # ══════════════════════════════════════════════════════════════════════════

    # mode_a vs GT，全 DM 相关范围
    category_precision:  float = 0.0
    category_recall:     float = 0.0
    category_f1:         float = 0.0

    # mode_a 纯降糖药类别 F1（paper 主表，EVAL_DM_ONLY_CATEGORIES）
    dm_only_category_f1: float = 0.0

    # mode_b 纯降糖药类别 F1（v3 新增，CII 对比用）
    ccs_dm_only_category_f1: float = 0.0

    # mode_a 并发症专属类别 F1（paper 补充表，EVAL_COMPLICATION_ONLY_CATEGORIES）
    # 独立评估 ACEi/ARB/statin 等并发症用药的拟合度，与 DM-only 分开报告
    complication_only_category_f1: float = 0.0

    # mode_b 并发症专属类别 F1（CII 对比用，与 mode_a 对称）
    ccs_complication_only_category_f1: float = 0.0

    # ══════════════════════════════════════════════════════════════════════════
    # 层次二：药物级 F1 + ATC 层次金字塔
    # ══════════════════════════════════════════════════════════════════════════

    # mode_a vs GT
    drug_precision:   float = 0.0
    drug_recall:      float = 0.0
    drug_f1:          float = 0.0

    # Rx Fidelity（mode_a 语义明确化别名）
    rxf_category_f1:  float = 0.0
    rxf_drug_f1:      float = 0.0

    # CCS（mode_b vs GT）— 与 mode_a 完全对称
    ccs_category_precision: float = 0.0
    ccs_category_recall:    float = 0.0
    ccs_category_f1:        float = 0.0
    ccs_drug_precision:     float = 0.0
    ccs_drug_recall:        float = 0.0
    ccs_drug_f1:            float = 0.0

    # 双模式状态
    mode_b_available:   bool  = False
    mode_ab_divergence: float = 0.0   # Jaccard(mode_a, mode_b)

    # ATC 层次金字塔 — mode_a
    atc_l4_f1:          float = 0.0
    atc_l3_f1:          float = 0.0
    atc_l2_f1:          float = 0.0   # 治疗亚组（如 A10B），临床最有意义
    atc_l1_f1:          float = 0.0
    atc_pathway_hit:    bool  = False
    atc_rec_pathways:   List[str] = field(default_factory=list)
    atc_actual_pathways: List[str] = field(default_factory=list)
    atc_rec_coverage:   float = 0.0

    # ATC 层次金字塔 — mode_b（v3 新增）
    # L2 (A10B: 口服降糖药) 是 CII 治疗路径分析的核心层次
    ccs_atc_l4_f1:      float = 0.0
    ccs_atc_l3_f1:      float = 0.0
    ccs_atc_l2_f1:      float = 0.0
    ccs_atc_l1_f1:      float = 0.0

    # ══════════════════════════════════════════════════════════════════════════
    # 层次三：安全性
    # ══════════════════════════════════════════════════════════════════════════

    # mode_a 安全违规（指南最优解的安全性基准）
    n_violations_critical:  int        = 0
    n_violations_major:     int        = 0
    n_violations_moderate:  int        = 0
    violations:             List[Dict] = field(default_factory=list)

    # mode_b 安全违规（真实预测处方的安全性，v3 新增）
    # 核心论文数据：ccs_n_violations_critical > n_violations_critical
    # → 真实处方比指南最优解更危险，量化临床惰性的安全代价
    ccs_n_violations_critical:  int = 0
    ccs_n_violations_major:     int = 0
    ccs_n_violations_moderate:  int = 0

    # ── 多药负担 ──────────────────────────────────────────────────────────────
    pb_ratio:           float = 1.0
    pb_over_prescribed: bool  = False
    pb_under_prescribed: bool = False

    # ── 附加指标 ──────────────────────────────────────────────────────────────
    n_recommendations:  int   = 0
    n_actual_drugs:     int   = 0
    dm_drug_ratio:      float = 0.0
    reasoning_has_data: bool  = False
    reasoning_length:   int   = 0
    inference_time:     float = 0.0
    estimated_cost:     float = 0.0

    # ── 元数据 ────────────────────────────────────────────────────────────────
    success:       bool          = True
    error_message: Optional[str] = None

    # ── 空 GT 分类 ────────────────────────────────────────────────────────────
    empty_gt_type:       str   = "NOT_EMPTY"
    empty_gt_confidence: float = 1.0
    empty_gt_evidence:   Dict  = field(default_factory=dict)

    # ── 用药决策（Initiation / Holding）──────────────────────────────────────
    initiation_is_tp:    bool  = False
    initiation_is_tn:    bool  = False
    initiation_is_fn:    bool  = False
    initiation_is_fp:    bool  = False
    initiation_fp_type:  str   = ""
    initiation_fp_justification: str = ""
    initiation_tn_safe:  bool  = False
    initiation_weighted_correct: float = 0.0

    # ── OSE 字段 ──────────────────────────────────────────────────────────────
    outcome_label:            str   = "Unknown"
    severity_tier:            str   = "Unknown"
    gas_score:                float = 0.0
    is_clinical_inertia:      bool  = False
    temporal_violation_count: int   = 0
    cii_evidence:             str   = ""
    prescription_divergence:  float = 0.0

    # ── CII 候选事件 ──────────────────────────────────────────────────────────
    n_cii_candidates:    int = 0
    cii_omission_count:  int = 0
    cii_commission_count: int = 0

    # ── 专家问答评分 ──────────────────────────────────────────────────────────
    expert_qa_score: Optional[float] = None


# ═══════════════════════════════════════════════════════════════════════════════
# 四、MetricsCalculator（v3）
# ═══════════════════════════════════════════════════════════════════════════════

class MetricsCalculator:
    """
    三层指标计算器（v3 — mode_a / mode_b 完整对称）。

    层次一: category_f1 / dm_only_category_f1  (mode_a)
            ccs_dm_only_category_f1            (mode_b，v3)
    层次二: drug_f1 / ATC 金字塔              (mode_a)
            ccs_drug_f1 / ccs_atc_*            (mode_b，v3)
    层次三: n_violations_*                     (mode_a)
            ccs_n_violations_*                 (mode_b，v3)
    """

    def __init__(
        self,
        normalizer:         "DrugNormalizer",
        constraint_checker: "HardConstraintChecker",
    ):
        self.normalizer = normalizer
        self.checker    = constraint_checker

        self._extensions  = None
        try:
            from evaluation.evaluation_extensions import (
                EvaluationExtensions, ATCHierarchyEvaluator,
            )
            self._extensions = EvaluationExtensions(normalizer)
        except ImportError:
            logger.warning("evaluation_extensions 未加载，ATC 指标将为默认值")

        self._dm_categories           = frozenset()
        self._dm_only_categories      = frozenset()
        self._complication_categories = frozenset()
        try:
            from agents.config import (
                DIABETES_MEDICATIONS,
                EVAL_DM_ONLY_CATEGORIES,
                EVAL_COMPLICATION_ONLY_CATEGORIES,
            )
            self._dm_categories           = frozenset(DIABETES_MEDICATIONS.keys())
            self._dm_only_categories      = _canon_scope(EVAL_DM_ONLY_CATEGORIES)
            self._complication_categories = _canon_scope(EVAL_COMPLICATION_ONLY_CATEGORIES)
        except ImportError:
            pass

    # ── 主计算接口 ────────────────────────────────────────────────────────────

    def calculate(
        self,
        rec_result:   Dict,
        patient_data: Dict,
        era:          str  = "Unknown",
        db_source:    str  = "unknown",
    ) -> PatientMetrics:
        """
        计算单患者所有指标（mode_a / mode_b 完全对称）。

        v3 核心字段：
          1. ccs_dm_only_category_f1  — mode_b 纯降糖药类别 F1
          2. ccs_atc_l*/ccs_atc_l2   — mode_b ATC 层次（extensions 第二次调用）
          3. ccs_n_violations_*       — mode_b 安全违规（vs mode_a 对比揭示 CII 安全代价）
        """
        patient_id = rec_result.get("patient_id", "")
        subject_id = rec_result.get("subject_id", 0)
        hadm_id    = rec_result.get("hadm_id",    0)
        model_name = rec_result.get("model_name", "unknown")

        # ── 双模式检测 ────────────────────────────────────────────────────────
        mode_a_raw = _extract_drug_list_from_rec(rec_result, mode="mode_a")
        mode_b_raw = _extract_drug_list_from_rec(rec_result, mode="mode_b")
        dual_raw   = (rec_result.get("diamas_metadata", {}) or {}).get("dual_mode_raw", {})
        mode_b_available = bool(dual_raw and mode_b_raw)
        if not mode_b_available:
            mode_b_raw = mode_a_raw

        # ── 药物解析 ──────────────────────────────────────────────────────────
        rec_names_a    = _names_from_rec_list(mode_a_raw)
        rec_mappings_a = [self.normalizer.normalize(n) for n in rec_names_a]
        rec_mapped_a   = [m.mapped_name for m in rec_mappings_a if m.mapped_name]
        rec_cats_a     = list(_cat_set(rec_mappings_a))

        rec_names_b    = _names_from_rec_list(mode_b_raw)
        rec_mappings_b = [self.normalizer.normalize(n) for n in rec_names_b]
        rec_mapped_b   = [m.mapped_name for m in rec_mappings_b if m.mapped_name]
        rec_cats_b     = list(_cat_set(rec_mappings_b))

        # ── GT 提取 + 过滤 ────────────────────────────────────────────────────
        actual_raw_unfiltered = self._extract_actual_drugs(patient_data, rec_result)
        actual_raw    = self.normalizer.filter_dm_relevant_drugs(actual_raw_unfiltered)
        act_mappings  = [self.normalizer.normalize(n) for n in actual_raw]
        actual_mapped = [m.mapped_name for m in act_mappings if m.mapped_name]
        actual_cats   = list(_cat_set(act_mappings))
        actual_set    = {a.lower() for a in actual_mapped if a}

        # ══════════════════════════════════════════════════════════════════════
        # 层次一：类别级 F1
        # ══════════════════════════════════════════════════════════════════════

        cat_p, cat_r, cat_f1 = self._prf(set(rec_cats_a), set(actual_cats))

        dm_only_act   = _cat_set(act_mappings, self._dm_only_categories)
        dm_only_rec_a = _cat_set(rec_mappings_a, self._dm_only_categories)
        dm_only_rec_b = _cat_set(rec_mappings_b, self._dm_only_categories)

        _, _, dm_only_cat_f1_a    = self._prf(dm_only_rec_a, dm_only_act)
        _, _, ccs_dm_only_cat_f1  = self._prf(dm_only_rec_b, dm_only_act)

        # 并发症专属类别 F1（mode_a + mode_b，与 dm_only 完全对称）
        compl_act   = _cat_set(act_mappings, self._complication_categories)
        compl_rec_a = _cat_set(rec_mappings_a, self._complication_categories)
        compl_rec_b = _cat_set(rec_mappings_b, self._complication_categories)
        _, _, compl_cat_f1_a     = self._prf(compl_rec_a, compl_act)
        _, _, ccs_compl_cat_f1   = self._prf(compl_rec_b, compl_act)

        # mode_b 类别级（完整 P/R/F1，与 mode_a 对称）
        ccs_cat_p, ccs_cat_r, ccs_cat_f1 = self._prf(set(rec_cats_b), set(actual_cats))

        # ══════════════════════════════════════════════════════════════════════
        # 层次二：药物级 F1
        # ══════════════════════════════════════════════════════════════════════

        rec_set_a = {r.lower() for r in rec_mapped_a if r}
        rec_set_b = {r.lower() for r in rec_mapped_b if r}

        drg_p, drg_r, drg_f1 = self._prf(rec_set_a, actual_set, skip_empty_gold=True)
        # mode_b 药物级（完整 P/R/F1，与 mode_a 对称）
        ccs_drg_p, ccs_drg_r, ccs_drg_f1 = self._prf(rec_set_b, actual_set, skip_empty_gold=True)

        union    = rec_set_a | rec_set_b
        sym_diff = rec_set_a.symmetric_difference(rec_set_b)
        mode_ab_div = len(sym_diff) / len(union) if union else 0.0

        cm = patient_data.get("clinical_metrics", {}) or {}

        # ══════════════════════════════════════════════════════════════════════
        # 层次三：安全性（mode_a + mode_b 对称检查）
        # ══════════════════════════════════════════════════════════════════════

        violations_a = self.checker.check(mode_a_raw, cm, patient_data)
        n_crit_a     = sum(1 for v in violations_a if v.severity == "critical")
        n_maj_a      = sum(1 for v in violations_a if v.severity == "major")
        n_mod_a      = sum(1 for v in violations_a if v.severity == "moderate")

        # mode_b 安全违规（v3 新增核心计算）
        # 旧格式（mode_b = mode_a）时 mode_b_raw == mode_a_raw，结果相同但字段存在
        violations_b = self.checker.check(mode_b_raw, cm, patient_data)
        n_crit_b     = sum(1 for v in violations_b if v.severity == "critical")
        n_maj_b      = sum(1 for v in violations_b if v.severity == "major")
        n_mod_b      = sum(1 for v in violations_b if v.severity == "moderate")

        # ══════════════════════════════════════════════════════════════════════
        # ATC 层次金字塔 — mode_a + mode_b
        # ══════════════════════════════════════════════════════════════════════

        ext_kwargs: Dict     = {}
        ccs_atc:    Dict     = {}

        if self._extensions:
            # mode_a 完整 extensions（ATC / 空GT / 多药 / Initiation）
            try:
                ext_a = self._extensions.evaluate(
                    rec_mapped_a, actual_mapped, rec_mappings_a, act_mappings,
                    cm, patient_data, era, violations_a,
                )
                ext_kwargs = {k: getattr(ext_a, k) for k in vars(ext_a)}
            except Exception as e:
                logger.debug(f"extensions (mode_a) 失败: {e}")

            # mode_b ATC（只在 mode_b 真实可用时额外调用，避免重复开销）
            if mode_b_available:
                try:
                    ext_b = self._extensions.evaluate(
                        rec_mapped_b, actual_mapped, rec_mappings_b, act_mappings,
                        cm, patient_data, era, violations_b,
                    )
                    ccs_atc = {
                        "ccs_atc_l4_f1": _safe_f1(getattr(ext_b, "atc_l4_f1", 0.0)),
                        "ccs_atc_l3_f1": _safe_f1(getattr(ext_b, "atc_l3_f1", 0.0)),
                        "ccs_atc_l2_f1": _safe_f1(getattr(ext_b, "atc_l2_f1", 0.0)),
                        "ccs_atc_l1_f1": _safe_f1(getattr(ext_b, "atc_l1_f1", 0.0)),
                    }
                except Exception as e:
                    logger.debug(f"extensions (mode_b) 失败: {e}")
                    # fallback: mode_b 失败时用 mode_a 的 ATC 结果
                    ccs_atc = {
                        "ccs_atc_l4_f1": _safe_f1(ext_kwargs.get("atc_l4_f1", 0.0)),
                        "ccs_atc_l3_f1": _safe_f1(ext_kwargs.get("atc_l3_f1", 0.0)),
                        "ccs_atc_l2_f1": _safe_f1(ext_kwargs.get("atc_l2_f1", 0.0)),
                        "ccs_atc_l1_f1": _safe_f1(ext_kwargs.get("atc_l1_f1", 0.0)),
                    }
            else:
                # 旧格式 fallback: mode_b = mode_a，ATC 结果一致
                ccs_atc = {
                    "ccs_atc_l4_f1": _safe_f1(ext_kwargs.get("atc_l4_f1", 0.0)),
                    "ccs_atc_l3_f1": _safe_f1(ext_kwargs.get("atc_l3_f1", 0.0)),
                    "ccs_atc_l2_f1": _safe_f1(ext_kwargs.get("atc_l2_f1", 0.0)),
                    "ccs_atc_l1_f1": _safe_f1(ext_kwargs.get("atc_l1_f1", 0.0)),
                }

        # ── CII 候选事件 ──────────────────────────────────────────────────────
        cii_divs     = _extract_cii_divergence_from_rec(rec_result)
        n_cii        = sum(1 for d in cii_divs if d.get("cii_candidate") is True)
        n_omission   = sum(1 for d in cii_divs
                          if d.get("divergence_type") == "omission" and d.get("cii_candidate"))
        n_commission = sum(1 for d in cii_divs
                          if d.get("divergence_type") == "commission" and d.get("cii_candidate"))

        # ── 附加 ──────────────────────────────────────────────────────────────
        dm_drugs  = [m for m in rec_mappings_a if m.drug_category in self._dm_categories]
        dm_ratio  = len(dm_drugs) / len(rec_mappings_a) if rec_mappings_a else 0.0
        reasoning = rec_result.get("reasoning", "") or ""

        def _sint(v):
            try: return int(v)
            except: return 0

        return PatientMetrics(
            patient_id = patient_id,
            subject_id = _sint(subject_id),
            hadm_id    = _sint(hadm_id),
            model_name = model_name,
            era        = era,
            db_source  = db_source,

            recommended_drugs = rec_mapped_a,
            actual_drugs      = actual_mapped,
            rec_categories    = rec_cats_a,
            actual_categories = actual_cats,

            # 层次一 mode_a
            category_precision    = cat_p,
            category_recall       = cat_r,
            category_f1           = cat_f1,
            dm_only_category_f1   = _safe_f1(dm_only_cat_f1_a),

            # 层次一 mode_b（v3 新增）
            ccs_dm_only_category_f1 = _safe_f1(ccs_dm_only_cat_f1),

            # 并发症专属类别 F1（mode_a + mode_b，v4 新增）
            complication_only_category_f1     = _safe_f1(compl_cat_f1_a),
            ccs_complication_only_category_f1 = _safe_f1(ccs_compl_cat_f1),

            # 层次二 mode_a
            drug_precision    = _safe_f1(drg_p),
            drug_recall       = _safe_f1(drg_r),
            drug_f1           = _safe_f1(drg_f1),
            rxf_category_f1   = _safe_f1(cat_f1),
            rxf_drug_f1       = _safe_f1(drg_f1),

            # 层次二 mode_b
            ccs_category_precision = _safe_f1(ccs_cat_p),
            ccs_category_recall    = _safe_f1(ccs_cat_r),
            ccs_category_f1        = _safe_f1(ccs_cat_f1),
            ccs_drug_precision     = _safe_f1(ccs_drg_p),
            ccs_drug_recall        = _safe_f1(ccs_drg_r),
            ccs_drug_f1            = _safe_f1(ccs_drg_f1),
            mode_b_available  = mode_b_available,
            mode_ab_divergence = round(mode_ab_div, 4),

            # 层次三 mode_a
            n_violations_critical = n_crit_a,
            n_violations_major    = n_maj_a,
            n_violations_moderate = n_mod_a,
            violations            = [asdict(v) for v in violations_a],

            # 层次三 mode_b（v3 新增）
            ccs_n_violations_critical  = n_crit_b,
            ccs_n_violations_major     = n_maj_b,
            ccs_n_violations_moderate  = n_mod_b,

            # ATC mode_b（v3 新增，从 ccs_atc 展开）
            **ccs_atc,

            # 附加指标
            n_recommendations  = len(rec_names_a),
            n_actual_drugs     = len(actual_raw),
            dm_drug_ratio      = dm_ratio,
            reasoning_has_data = bool(re.search(r"\d+\.?\d*", reasoning)),
            reasoning_length   = len(reasoning),
            inference_time     = rec_result.get("inference_time", 0.0),
            estimated_cost     = rec_result.get("estimated_cost", 0.0),
            success            = rec_result.get("success", True),
            error_message      = rec_result.get("error_message"),

            # CII
            n_cii_candidates    = n_cii,
            cii_omission_count  = n_omission,
            cii_commission_count = n_commission,

            # extensions mode_a（ATC / 空GT / 多药 / Initiation）
            **ext_kwargs,
        )

    # ── 内部工具 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _prf(
        pred: Set[str],
        gold: Set[str],
        skip_empty_gold: bool = False,
    ) -> Tuple[float, float, float]:
        if not pred and not gold:
            return 1.0, 1.0, 1.0
        if not pred:
            return 0.0, 0.0, 0.0
        if not gold:
            return (float("nan"), float("nan"), float("nan")) if skip_empty_gold else (0.0, 0.0, 0.0)
        tp = len(pred & gold)
        p  = tp / len(pred)
        r  = tp / len(gold)
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return p, r, f1

    def _extract_actual_drugs(
        self,
        patient_data: Dict,
        rec_result:   Optional[Dict] = None,
    ) -> List[str]:
        """
        从患者数据中提取出院用药（原始名，未过滤）。

        优先级（从高到低）：
          1. discharge_medications（旧版 extractor 字符串列表）
          2. all_medications[time_window=discharge]（新版 v4 主路径）
          3. rec_result.actual_medications.discharge（推荐文件缓存）
          4. all_medications[is_diabetes_related=True]（兜底，不完整）
          5. medications（legacy 字段）
        """
        discharge = patient_data.get("discharge_medications", [])
        if discharge and isinstance(discharge[0] if discharge else None, str):
            return [d for d in discharge if d]

        all_meds = patient_data.get("all_medications", [])
        win = [m.get("drug_name", "") for m in all_meds
               if isinstance(m, dict) and m.get("time_window") == "discharge" and m.get("drug_name")]
        if win:
            return win

        if rec_result is not None:
            ra = rec_result.get("actual_medications", {})
            if isinstance(ra, dict):
                dm = ra.get("discharge", [])
                if dm:
                    return [m if isinstance(m, str) else m.get("drug_name", "") for m in dm if m]

        if all_meds:
            dm_only = [m.get("drug_name", "") for m in all_meds
                       if m.get("drug_name") and m.get("is_diabetes_related")]
            if dm_only:
                return dm_only

        legacy = patient_data.get("medications", [])
        return [m if isinstance(m, str) else m.get("drug_name", "") for m in legacy if m]
