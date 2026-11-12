#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluation_extensions.py
========================
Dia-MAS 评测框架扩展模块（Phase 2 增强组件）

本模块是 baseline_evaluation.py 的伴生模块，提供以下四个增强组件：

  1. EmptyGTClassifier      — 空 Ground Truth 四分类器
  2. InitiationEvaluator    — 用药决策准确率（2×2 混淆矩阵 + FP 三分类）
  3. ATCHierarchyEvaluator  — ATC 四级层次评测金字塔
  5. PolypharmacyBurden     — 多药负担评分

设计原则：
  - 所有组件均可独立使用，不修改 baseline_evaluation.py 核心逻辑
  - 静态 ATC 表覆盖常见糖尿病药物；动态扩展通过 UMLS API（可选）
  - EmptyGTClassifier 基于 clinical_metrics deterministic 规则，不调用 LLM

整合方式（见 baseline_evaluation.py 修改说明）：
  在 MetricsCalculator.calculate() 末尾追加调用：
    ext = EvaluationExtensions(normalizer)
    extra = ext.evaluate(rec_result, patient_data, era, rec_mapped, actual_mapped,
                         rec_mappings, act_mappings, violations)
  将 extra 字段合并进 PatientMetrics 输出。

作者：Yuxi (由 Claude 协助重构)
版本：Phase 2 Preview
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 0. ATC 静态知识库
# ═══════════════════════════════════════════════════════════════════════════════

# 药物标准名（小写） → ATC 5级编码
# 来源：WHO ATC/DDD Index 2024
# 格式：A10 + 亚类 + 序号
# 仅覆盖糖尿病相关药物；其他药物通过 UMLS API 动态查找（ATCHierarchyEvaluator.lookup_atc）
DRUG_TO_ATC: Dict[str, str] = {
    # ── Biguanides (A10BA) ───────────────────────────────────────────────
    "metformin":                "A10BA02",

    # ── Sulfonylureas (A10BB) ────────────────────────────────────────────
    "chlorpropamide":           "A10BB02",   # 第一代，Era-1 标志性药物
    "tolbutamide":              "A10BB03",
    "glibenclamide":            "A10BB01",   # = Glyburide 国际名
    "glyburide":                "A10BB01",
    "glipizide":                "A10BB07",
    "gliclazide":               "A10BB09",
    "glimepiride":              "A10BB12",
    "gliquidone":               "A10BB08",

    # ── Alpha-glucosidase inhibitors (A10BF) ─────────────────────────────
    "acarbose":                 "A10BF01",
    "miglitol":                 "A10BF02",
    "voglibose":                "A10BF03",

    # ── Thiazolidinediones (A10BG) ───────────────────────────────────────
    "rosiglitazone":            "A10BG02",
    "pioglitazone":             "A10BG03",

    # ── DPP-4 inhibitors (A10BH) ─────────────────────────────────────────
    "sitagliptin":              "A10BH01",
    "vildagliptin":             "A10BH02",
    "saxagliptin":              "A10BH03",
    "alogliptin":               "A10BH04",
    "linagliptin":              "A10BH05",
    "teneligliptin":            "A10BH06",
    "trelagliptin":             "A10BH07",
    "omarigliptin":             "A10BH08",

    # ── GLP-1 receptor agonists (A10BJ) ─────────────────────────────────
    "exenatide":                "A10BJ01",
    "liraglutide":              "A10BJ02",
    "lixisenatide":             "A10BJ03",
    "albiglutide":              "A10BJ04",
    "dulaglutide":              "A10BJ05",
    "semaglutide":              "A10BJ06",
    "tirzepatide":              "A10BJ07",   # GIP/GLP-1 双靶点，WHO 2023 归入 A10BJ

    # ── SGLT-2 inhibitors (A10BK) ────────────────────────────────────────
    "dapagliflozin":            "A10BK01",
    "canagliflozin":            "A10BK02",
    "empagliflozin":            "A10BK03",
    "ertugliflozin":            "A10BK04",
    "ipragliflozin":            "A10BK05",
    "tofogliflozin":            "A10BK06",
    "luseogliflozin":           "A10BK07",

    # ── Meglitinides (A10BX) ─────────────────────────────────────────────
    "repaglinide":              "A10BX02",
    "nateglinide":              "A10BX03",
    "mitiglinide":              "A10BX06",

    # ── Insulins: Rapid-acting (A10AB) ───────────────────────────────────
    "regular insulin":          "A10AB01",
    "insulin aspart":           "A10AB05",
    "insulin lispro":           "A10AB04",
    "insulin glulisine":        "A10AB06",
    "faster insulin aspart":    "A10AB05",   # Fiasp = faster aspart

    # ── Insulins: Intermediate-acting (A10AC) ────────────────────────────
    "nph insulin":              "A10AC01",
    "insulin nph":              "A10AC01",

    # ── Insulins: Long-acting (A10AE) ────────────────────────────────────
    "insulin glargine":         "A10AE04",
    "insulin detemir":          "A10AE05",
    "insulin degludec":         "A10AE06",
    "insulin glargine u-300":   "A10AE04",

    # ── Glucokinase activators (A10BX, 近年新增) ──────────────────────────
    "dorzagliatin":             "A10BX13",   # 中国原研，2022 年上市
}

# ATC 3级编码 → 治疗路径（Pathway）分类
# 路径定义基于 ADA 2024 治疗范式转变节点
ATC3_TO_PATHWAY: Dict[str, str] = {
    "A10BA": "PathwayA_GlucoseCentric",     # 双胍类
    "A10BB": "PathwayA_GlucoseCentric",     # 磺脲类
    "A10BF": "PathwayA_GlucoseCentric",     # α-GI
    "A10BG": "PathwayA_GlucoseCentric",     # TZD
    "A10BX": "PathwayA_GlucoseCentric",     # 格列奈类 + 其他
    "A10BH": "PathwayC_IncrBased",          # DPP-4i（增量素基础，无心肾获益证据）
    "A10BJ": "PathwayB_CardioRenal",        # GLP-1RA（有 CVOT 证据）
    "A10BK": "PathwayB_CardioRenal",        # SGLT-2i（有 CVOT + 肾获益证据）
    "A10AB": "PathwayD_Insulin",
    "A10AC": "PathwayD_Insulin",
    "A10AD": "PathwayD_Insulin",
    "A10AE": "PathwayD_Insulin",
}

# ATC 2级编码 → 治疗意图
ATC2_TO_INTENT: Dict[str, str] = {
    "A10A": "Intent_InsulinTherapy",
    "A10B": "Intent_OralAntidiabetic",
}


def get_atc(drug_std_name: str) -> Optional[str]:
    """从静态表查找 ATC 编码（小写匹配）"""
    return DRUG_TO_ATC.get(drug_std_name.lower())


def atc_level(code: str, level: int) -> str:
    """
    截断 ATC 编码到指定层级
      level=5 → "A10BK03"  (完整)
      level=4 → "A10BK"
      level=3 → "A10B"
      level=2 → "A10"
      level=1 → "A"
    """
    cuts = {5: 7, 4: 5, 3: 4, 2: 3, 1: 1}
    return code[:cuts.get(level, 7)]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. 空 Ground Truth 分类器
# ═══════════════════════════════════════════════════════════════════════════════

class EmptyGTType(str, Enum):
    """
    空 GT 类型枚举

    评测时的处理策略：
      MODERNIZATION     → 计入 Initiation FP-Modernization
      CLINICAL_INERTIA  → AI 推荐 = TP（加分）；计入 Inertia-Correction 指标
      SAFE_DEESCALATION → AI 推荐空 = TN-Safety；违反 HC 则强惩罚
      DOCUMENTATION_GAP → 优先识别并加标记，单独分层报告；不纳入核心 F1
      NOT_EMPTY         → GT 不为空，正常计算
    """
    MODERNIZATION      = "modernization"
    CLINICAL_INERTIA   = "clinical_inertia"
    SAFE_DEESCALATION  = "safe_deescalation"
    DOCUMENTATION_GAP  = "documentation_gap"
    NOT_EMPTY          = "not_empty"


@dataclass
class EmptyGTResult:
    gt_type:    EmptyGTType
    confidence: float   # 0-1，分类置信度
    evidence:   Dict    # 支持分类的临床证据


class EmptyGTClassifier:
    """
    空 Ground Truth 四分类器

    分类逻辑（基于 clinical_metrics deterministic 规则，不调用 LLM）：
      优先级：DOCUMENTATION_GAP > SAFE_DEESCALATION > CLINICAL_INERTIA > MODERNIZATION

    参数说明：
      clinical_metrics: 来自患者 JSON 的 clinical_metrics 字典
      patient_data:     完整患者 JSON（用于入院药物记录）
      era:              患者所属时代（Era-1 / Era-2 / Era-3）
      admission_drugs:  入院时的用药列表（detect Documentation Gap）
    """

    # eGFR 阈值
    AKI_EGFR_THRESHOLD    = 30.0
    SEVERE_EGFR_THRESHOLD = 15.0

    # 高 HbA1c 阈值（clinical inertia 的标志）
    HIGH_HBCA1_THRESHOLD  = 8.0

    def classify(
        self,
        actual_drugs:     List[str],    # 标准化后的出院用药（已知为空时传 []）
        clinical_metrics: Dict,
        patient_data:     Dict,
        era:              str,
        admission_drugs:  Optional[List[str]] = None,
    ) -> EmptyGTResult:
        """
        对单患者的 GT 状态进行分类

        Returns:
            EmptyGTResult
        """
        # 如果 GT 不为空，直接返回
        if actual_drugs:
            return EmptyGTResult(EmptyGTType.NOT_EMPTY, 1.0, {})

        cm      = clinical_metrics or {}
        egfr    = self._egfr(cm)
        hba1c   = self._hba1c(cm)
        has_aki = self._has_aki(cm)

        # ── 优先级 1：Documentation Gap ───────────────────────────────────
        # 信号：入院时有糖尿病用药记录，出院时无 → 很可能是续用药未记录
        if admission_drugs:
            admission_dm = self._filter_dm_drugs(admission_drugs)
            if admission_dm:
                return EmptyGTResult(
                    EmptyGTType.DOCUMENTATION_GAP,
                    confidence=0.80,
                    evidence={
                        "admission_dm_drugs": admission_dm,
                        "signal": "入院时有降糖药，出院时为空：疑似续用药未录入出院医嘱"
                    }
                )

        # ── 优先级 2：Safe De-escalation ──────────────────────────────────
        # 信号：eGFR < 30，或急性肾损伤，或严重低血糖合并高龄
        if has_aki or (egfr is not None and egfr < self.AKI_EGFR_THRESHOLD):
            conf = 0.95 if (egfr is not None and egfr < self.SEVERE_EGFR_THRESHOLD) else 0.80
            return EmptyGTResult(
                EmptyGTType.SAFE_DEESCALATION,
                confidence=conf,
                evidence={
                    "egfr":    egfr,
                    "has_aki": has_aki,
                    "signal":  "急性肾损伤/eGFR<30：安全停药（HC-01/02）"
                }
            )

        # ── 优先级 3：Clinical Inertia ─────────────────────────────────────
        # 信号：HbA1c > 8.0%，有明确降糖需求，但本次入院为非代谢性主诊断
        if hba1c is not None and hba1c > self.HIGH_HBCA1_THRESHOLD:
            # 通过入院诊断判断是否为非代谢科入院（骨折、创伤、外科等）
            non_dm_admit = self._is_non_dm_admission(patient_data)
            if non_dm_admit:
                return EmptyGTResult(
                    EmptyGTType.CLINICAL_INERTIA,
                    confidence=0.75,
                    evidence={
                        "hba1c":         hba1c,
                        "non_dm_admit":  non_dm_admit,
                        "signal":        f"HbA1c {hba1c:.1f}% 超标但非代谢科入院，医生未处理血糖"
                    }
                )

        # ── 优先级 4：Modernization（默认，Era-1 最多见）──────────────────
        # 信号：ERA-1 + 轻度 DM + 无其他明确停药原因
        conf = 0.85 if era == "Era-1" else 0.50
        return EmptyGTResult(
            EmptyGTType.MODERNIZATION,
            confidence=conf,
            evidence={
                "era":    era,
                "signal": "Era-1 时代局限或当时指南尚未推荐现代药物（合理超前）"
            }
        )

    # ── 内部工具 ─────────────────────────────────────────────────────────

    @staticmethod
    def _egfr(cm: Dict) -> Optional[float]:
        return (cm.get("kidney") or {}).get("egfr")

    @staticmethod
    def _hba1c(cm: Dict) -> Optional[float]:
        return (cm.get("glycemic") or {}).get("hba1c")

    @staticmethod
    def _has_aki(cm: Dict) -> bool:
        return bool((cm.get("kidney") or {}).get("has_aki", False))

    @staticmethod
    def _filter_dm_drugs(drug_list: List[str]) -> List[str]:
        """简单过滤出降糖药（基于关键词）"""
        dm_keywords = {
            "metformin", "insulin", "glipizide", "glyburide", "glimepiride",
            "sitagliptin", "linagliptin", "empagliflozin", "dapagliflozin",
            "liraglutide", "semaglutide", "pioglitazone", "acarbose",
        }
        return [d for d in drug_list if any(k in d.lower() for k in dm_keywords)]

    @staticmethod
    def _is_non_dm_admission(patient_data: Dict) -> bool:
        """判断是否为非代谢性主诊断入院"""
        non_dm_icd_prefixes = (
            "S",    # 损伤
            "M",    # 肌肉骨骼
            "K",    # 消化
            "J",    # 呼吸
            "I6",   # 卒中
            "820",  # 股骨骨折 ICD-9
            "821",  # 其他骨折 ICD-9
        )
        # 尝试从多个字段获取主诊断 ICD 编码
        primary_icd = (
            patient_data.get("primary_diagnosis_icd")
            or patient_data.get("admission_diagnosis", "")
            or ""
        )
        for prefix in non_dm_icd_prefixes:
            if str(primary_icd).startswith(prefix):
                return True
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# 2. 用药决策评测（Initiation/Holding Accuracy + FP 三分类）
# ═══════════════════════════════════════════════════════════════════════════════

class FPType(str, Enum):
    """
    False Positive 三分类

    Modernization:      AI 开了有 ADA Level A 证据支持的药，但当时指南未推荐
    Inertia_Correction: 患者有明确适应症但医生漏开，AI 正确指出
    Hallucination:      无临床依据的推荐（真正的 FP）
    """
    MODERNIZATION      = "fp_modernization"
    INERTIA_CORRECTION = "fp_inertia_correction"
    HALLUCINATION      = "fp_hallucination"


@dataclass
class InitiationMetrics:
    """
    2×2 混淆矩阵 + FP 三分类

    定义：
      - "有用药决策" = 患者出院时有≥1个降糖药（GT 不为空）
      - TP: AI 推荐有药 + GT 有药
      - TN: AI 推荐无药 + GT 无药（含 Safe Deescalation）
      - FN: AI 推荐无药 + GT 有药（危险！漏药）
      - FP: AI 推荐有药 + GT 无药（需细分）
    """
    # 4格混淆矩阵（患者级别，每格 0 或 1）
    is_tp: bool = False
    is_tn: bool = False
    is_fn: bool = False
    is_fp: bool = False

    # FP 细分（仅当 is_fp=True 时有意义）
    fp_type:            Optional[FPType]    = None
    fp_justification:   str                 = ""

    # 安全权重（FN 比 FP 更危险，用于加权 Accuracy 计算）
    fn_weight: float = 3.0   # FN 惩罚权重（论文中需做敏感性分析）

    # TN 细分
    is_tn_safe_deescalation: bool = False   # TN 中属于安全停药（AI 正确不开药）

    @property
    def weighted_correct(self) -> float:
        """加权正确率（单患者贡献，用于聚合）"""
        if self.is_tp or self.is_tn:
            return 1.0
        if self.is_fn:
            return 0.0   # FN 完全不得分（危险）
        if self.is_fp:
            # FP-Modernization / FP-Inertia 有部分功劳
            if self.fp_type == FPType.MODERNIZATION:
                return 0.5
            elif self.fp_type == FPType.INERTIA_CORRECTION:
                return 0.6
            else:
                return 0.0   # FP-Hallucination
        return 0.0


class InitiationEvaluator:
    """
    用药决策准确率评测器

    同时依赖：
      - EmptyGTClassifier：判断空 GT 类型
      - ATC 知识库：辅助判断 FP 类型
      - 患者临床指标：判断是否有明确适应症
    """

    def __init__(self, empty_gt_classifier: EmptyGTClassifier):
        self.gt_classifier = empty_gt_classifier

    def evaluate(
        self,
        rec_mapped:       List[str],     # 推荐的标准化药名
        actual_mapped:    List[str],     # 实际出院用药标准化药名
        gt_type:          EmptyGTType,   # 已由 EmptyGTClassifier 判断
        clinical_metrics: Dict,
        era:              str,
    ) -> InitiationMetrics:
        """计算单患者的 Initiation/Holding 指标"""

        ai_has_drugs = len([d for d in rec_mapped if d]) > 0
        gt_has_drugs = len([d for d in actual_mapped if d]) > 0

        # ── 四格分类 ──────────────────────────────────────────────────────
        if gt_has_drugs and ai_has_drugs:
            return InitiationMetrics(is_tp=True)

        if not gt_has_drugs and not ai_has_drugs:
            is_safe = (gt_type == EmptyGTType.SAFE_DEESCALATION)
            return InitiationMetrics(is_tn=True, is_tn_safe_deescalation=is_safe)

        if gt_has_drugs and not ai_has_drugs:
            return InitiationMetrics(is_fn=True)

        # gt_has_drugs=False, ai_has_drugs=True → FP，需细分
        fp_type, justification = self._classify_fp(
            rec_mapped, clinical_metrics, gt_type, era
        )
        return InitiationMetrics(
            is_fp=True,
            fp_type=fp_type,
            fp_justification=justification,
        )

    def _classify_fp(
        self,
        rec_mapped:       List[str],
        clinical_metrics: Dict,
        gt_type:          EmptyGTType,
        era:              str,
    ) -> Tuple[FPType, str]:
        """FP 三分类逻辑"""

        # FP-Inertia-Correction：GT 本身就是临床惰性
        if gt_type == EmptyGTType.CLINICAL_INERTIA:
            return (
                FPType.INERTIA_CORRECTION,
                "患者 HbA1c 超标但医生未处理；AI 推荐有证据支持"
            )

        # FP-Modernization：GT 为 Era-1 时代局限
        if gt_type == EmptyGTType.MODERNIZATION:
            # 进一步确认：AI 推荐的药物中是否有 SGLT-2i / GLP-1RA（PathwayB）
            modern_drugs = {
                "empagliflozin", "dapagliflozin", "canagliflozin",
                "liraglutide", "semaglutide", "dulaglutide", "tirzepatide",
            }
            has_modern = any(d.lower() in modern_drugs for d in rec_mapped if d)
            if has_modern:
                return (
                    FPType.MODERNIZATION,
                    f"Era={era}：AI 推荐了当时未有的现代心肾保护药物（ADA Level A 证据）"
                )

        # FP-Hallucination：其余情况（真假阳性）
        return (
            FPType.HALLUCINATION,
            "无明确临床适应证支撑的推荐"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. ATC 四级层次评测金字塔
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ATCMetrics:
    """
    四层 ATC 评测结果

    L4 (精确药物):    RxNorm CUI 级别，= 现有 Drug-level F1
    L3 (药物亚类):    ATC 4级（如 SGLT-2i = SGLT-2i）
    L2 (治疗路径):    ATC 3级（PathwayA/B/C/D）
    L1 (靶点意图):    ATC 2级（口服降糖 / 胰岛素治疗）
    """
    # L4: 精确药物匹配（与 Drug-level F1 一致）
    l4_precision: float = 0.0
    l4_recall:    float = 0.0
    l4_f1:        float = 0.0

    # L3: ATC 4级亚类匹配
    l3_precision: float = 0.0
    l3_recall:    float = 0.0
    l3_f1:        float = 0.0

    # L2: 治疗路径匹配
    l2_precision: float = 0.0
    l2_recall:    float = 0.0
    l2_f1:        float = 0.0

    # L1: 靶点意图匹配
    l1_precision: float = 0.0
    l1_recall:    float = 0.0
    l1_f1:        float = 0.0

    # 路径级命中明细（用于 Era 分析）
    rec_pathways:    List[str] = field(default_factory=list)
    actual_pathways: List[str] = field(default_factory=list)
    pathway_hit:     bool      = False   # L2 至少有 1 个路径命中

    # ATC 覆盖率（有多少药物找到了 ATC 编码）
    rec_atc_coverage:    float = 0.0   # 推荐药物的 ATC 覆盖率
    actual_atc_coverage: float = 0.0


class ATCHierarchyEvaluator:
    """
    ATC 四级层次评测金字塔

    实现方式：
      1. 静态 ATC 表（DRUG_TO_ATC）：覆盖常见糖尿病药物，离线可用
      2. UMLS API 扩展（可选）：通过 enhanced_umls_mapper 动态查找未覆盖的药物

    使用方法：
      evaluator = ATCHierarchyEvaluator(use_umls=False)
      metrics = evaluator.evaluate(rec_mapped, actual_mapped)
    """

    def __init__(self, use_umls: bool = False, umls_mapper=None):
        """
        Args:
            use_umls:    是否启用 UMLS API 查找（需要联网和 TGT 认证）
            umls_mapper: enhanced_umls_mapper.EnhancedUMLSMapper 实例（use_umls=True 时必传）
        """
        self.use_umls    = use_umls
        self.umls_mapper = umls_mapper
        self._atc_cache: Dict[str, Optional[str]] = {}

    def evaluate(
        self,
        rec_mapped:    List[str],
        actual_mapped: List[str],
    ) -> ATCMetrics:
        """
        计算全部四级 ATC 评测指标

        Args:
            rec_mapped:    推荐药物标准名列表（来自 DrugNormalizer）
            actual_mapped: 实际用药标准名列表

        Returns:
            ATCMetrics
        """
        # 获取 ATC 编码
        rec_atc    = {d: self._get_atc(d) for d in rec_mapped if d}
        actual_atc = {d: self._get_atc(d) for d in actual_mapped if d}

        # 覆盖率
        rec_cov    = sum(1 for v in rec_atc.values() if v) / len(rec_atc)    if rec_atc    else 0.0
        actual_cov = sum(1 for v in actual_atc.values() if v) / len(actual_atc) if actual_atc else 0.0

        # L4: 精确药物（与 Drug-level 相同，ATC 5级 = 药物本身）
        l4_p, l4_r, l4_f1 = self._prf_set(
            {d.lower() for d in rec_mapped if d},
            {d.lower() for d in actual_mapped if d}
        )

        # L3: ATC 4级亚类
        rec_l3    = {atc_level(v, 4) for v in rec_atc.values()    if v}
        actual_l3 = {atc_level(v, 4) for v in actual_atc.values() if v}
        l3_p, l3_r, l3_f1 = self._prf_set(rec_l3, actual_l3)

        # L2: 治疗路径（ATC 3级 → Pathway）
        rec_pathways    = {
            ATC3_TO_PATHWAY.get(atc_level(v, 3), f"Other_{atc_level(v, 3)}")
            for v in rec_atc.values() if v
        }
        actual_pathways = {
            ATC3_TO_PATHWAY.get(atc_level(v, 3), f"Other_{atc_level(v, 3)}")
            for v in actual_atc.values() if v
        }
        l2_p, l2_r, l2_f1 = self._prf_set(rec_pathways, actual_pathways)
        pathway_hit = bool(rec_pathways & actual_pathways)

        # L1: 靶点意图（ATC 2级）
        rec_l1    = {atc_level(v, 2) for v in rec_atc.values()    if v}
        actual_l1 = {atc_level(v, 2) for v in actual_atc.values() if v}
        l1_p, l1_r, l1_f1 = self._prf_set(rec_l1, actual_l1)

        return ATCMetrics(
            l4_precision=l4_p,  l4_recall=l4_r,  l4_f1=l4_f1,
            l3_precision=l3_p,  l3_recall=l3_r,  l3_f1=l3_f1,
            l2_precision=l2_p,  l2_recall=l2_r,  l2_f1=l2_f1,
            l1_precision=l1_p,  l1_recall=l1_r,  l1_f1=l1_f1,
            rec_pathways=sorted(rec_pathways),
            actual_pathways=sorted(actual_pathways),
            pathway_hit=pathway_hit,
            rec_atc_coverage=rec_cov,
            actual_atc_coverage=actual_cov,
        )

    def _get_atc(self, drug_std_name: str) -> Optional[str]:
        """查找 ATC 编码（静态表 + 可选 UMLS）"""
        key = drug_std_name.lower()
        if key in self._atc_cache:
            return self._atc_cache[key]

        # 1. 静态表
        code = DRUG_TO_ATC.get(key)

        # 2. UMLS 动态查找（未覆盖时）
        if code is None and self.use_umls and self.umls_mapper:
            try:
                # enhanced_umls_mapper 接口：get_atc_codes(drug_name) → List[str]
                codes = self.umls_mapper.get_atc_codes(drug_std_name)
                code  = codes[0] if codes else None
            except Exception as e:
                logger.debug(f"UMLS ATC 查找失败 {drug_std_name}: {e}")

        self._atc_cache[key] = code
        return code

    @staticmethod
    def _prf_set(pred: Set[str], gold: Set[str]) -> Tuple[float, float, float]:
        if not pred and not gold:
            return 1.0, 1.0, 1.0
        if not pred or not gold:
            return 0.0, 0.0, 0.0
        tp = len(pred & gold)
        p  = tp / len(pred)
        r  = tp / len(gold)
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return p, r, f1


# ═══════════════════════════════════════════════════════════════════════════════
# 5. 多药负担评分
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PolypharmacyBurden:
    """
    多药负担评分（Polypharmacy Burden，PB）

    临床背景：
      在老年糖尿病和 ICU 患者中，过度处方本身是质量问题。
      即使每种药物都有 Level A 证据，7种降糖药同时开具也是不合理的。

    指标：
      precision_burden_ratio: len(predicted) / max(len(actual), 1)
        < 0.8  → 可能漏药（偏少）
        0.8–2.0 → 合理范围
        > 2.0  → 过度处方风险
    """
    n_predicted:           int   = 0
    n_actual:              int   = 0
    precision_burden_ratio: float = 1.0   # 推荐药数 / 实际药数
    is_over_prescribed:    bool  = False  # ratio > 2.0
    is_under_prescribed:   bool  = False  # ratio < 0.5 且 actual > 0

    OVER_THRESHOLD  = 2.0
    UNDER_THRESHOLD = 0.5

    @classmethod
    def calculate(cls, n_predicted: int, n_actual: int) -> "PolypharmacyBurden":
        ratio = n_predicted / max(n_actual, 1)
        return cls(
            n_predicted=n_predicted,
            n_actual=n_actual,
            precision_burden_ratio=ratio,
            is_over_prescribed=(ratio > cls.OVER_THRESHOLD),
            is_under_prescribed=(ratio < cls.UNDER_THRESHOLD and n_actual > 0),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 6. 总入口：EvaluationExtensions（供 MetricsCalculator.calculate() 调用）
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExtendedEvalResult:
    """
    所有扩展指标的汇总容器

    设计为 dict 可序列化（通过 asdict()），直接合并进 PatientMetrics 输出。
    """
    # 空 GT 分类
    empty_gt_type:       str   = EmptyGTType.NOT_EMPTY.value
    empty_gt_confidence: float = 1.0
    empty_gt_evidence:   Dict  = field(default_factory=dict)

    # 用药决策（Initiation/Holding）
    initiation_is_tp:    bool  = False
    initiation_is_tn:    bool  = False
    initiation_is_fn:    bool  = False
    initiation_is_fp:    bool  = False
    initiation_fp_type:  str   = ""
    initiation_fp_justification: str = ""
    initiation_tn_safe:  bool  = False
    initiation_weighted_correct: float = 0.0

    # ATC 四级评测
    atc_l4_f1:           float = 0.0
    atc_l3_f1:           float = 0.0
    atc_l2_f1:           float = 0.0
    atc_l1_f1:           float = 0.0
    atc_pathway_hit:     bool  = False
    atc_rec_pathways:    List[str] = field(default_factory=list)
    atc_actual_pathways: List[str] = field(default_factory=list)
    atc_rec_coverage:    float = 0.0

    # 多药负担
    pb_ratio:            float = 1.0
    pb_over_prescribed:  bool  = False
    pb_under_prescribed: bool  = False


class EvaluationExtensions:
    """
    评测扩展主入口

    在 MetricsCalculator.calculate() 末尾调用：

        from evaluation import EvaluationExtensions
        ext = EvaluationExtensions(normalizer=self.normalizer)
        extra = ext.evaluate(
            rec_mapped, actual_mapped, rec_mappings, act_mappings,
            clinical_metrics, patient_data, era, violations
        )
        # 将 asdict(extra) 合并进 PatientMetrics 输出

    不修改 baseline_evaluation.py 现有逻辑，纯追加扩展。
    """

    def __init__(
        self,
        normalizer,               # DrugNormalizer 实例（来自 baseline_evaluation.py）
        use_umls:    bool = False,
        umls_mapper           = None,
    ):
        self.gt_classifier    = EmptyGTClassifier()
        self.atc_evaluator    = ATCHierarchyEvaluator(use_umls=use_umls, umls_mapper=umls_mapper)
        self.init_evaluator   = InitiationEvaluator(self.gt_classifier)
        self._normalizer      = normalizer

    def evaluate(
        self,
        rec_mapped:       List[str],      # 推荐药物（标准化后）
        actual_mapped:    List[str],      # 实际用药（标准化后）
        rec_mappings,                     # List[DrugMapping]
        act_mappings,                     # List[DrugMapping]
        clinical_metrics: Dict,
        patient_data:     Dict,
        era:              str,
        violations:       List,           # List[ConstraintViolation]
    ) -> ExtendedEvalResult:
        """计算所有扩展指标"""

        # ── 0. 提取入院用药（for Documentation Gap 检测）────────────────
        admission_drugs = self._extract_admission_drugs(patient_data)

        # ── 1. 空 GT 分类 ────────────────────────────────────────────────
        gt_result = self.gt_classifier.classify(
            actual_drugs=actual_mapped,
            clinical_metrics=clinical_metrics,
            patient_data=patient_data,
            era=era,
            admission_drugs=admission_drugs,
        )

        # ── 2. 用药决策指标 ──────────────────────────────────────────────
        init_metrics = self.init_evaluator.evaluate(
            rec_mapped=rec_mapped,
            actual_mapped=actual_mapped,
            gt_type=gt_result.gt_type,
            clinical_metrics=clinical_metrics,
            era=era,
        )

        # ── 3. ATC 四级评测 ──────────────────────────────────────────────
        atc_metrics = self.atc_evaluator.evaluate(rec_mapped, actual_mapped)

        # ── 4. 多药负担 ──────────────────────────────────────────────────
        pb = PolypharmacyBurden.calculate(
            n_predicted=len([d for d in rec_mapped if d]),
            n_actual=len([d for d in actual_mapped if d]),
        )

        # ── 汇总 ─────────────────────────────────────────────────────────
        return ExtendedEvalResult(
            # 空 GT
            empty_gt_type=gt_result.gt_type.value,
            empty_gt_confidence=gt_result.confidence,
            empty_gt_evidence=gt_result.evidence,

            # Initiation
            initiation_is_tp=init_metrics.is_tp,
            initiation_is_tn=init_metrics.is_tn,
            initiation_is_fn=init_metrics.is_fn,
            initiation_is_fp=init_metrics.is_fp,
            initiation_fp_type=init_metrics.fp_type.value if init_metrics.fp_type else "",
            initiation_fp_justification=init_metrics.fp_justification,
            initiation_tn_safe=init_metrics.is_tn_safe_deescalation,
            initiation_weighted_correct=init_metrics.weighted_correct,

            # ATC
            atc_l4_f1=atc_metrics.l4_f1,
            atc_l3_f1=atc_metrics.l3_f1,
            atc_l2_f1=atc_metrics.l2_f1,
            atc_l1_f1=atc_metrics.l1_f1,
            atc_pathway_hit=atc_metrics.pathway_hit,
            atc_rec_pathways=atc_metrics.rec_pathways,
            atc_actual_pathways=atc_metrics.actual_pathways,
            atc_rec_coverage=atc_metrics.rec_atc_coverage,

            # PB
            pb_ratio=pb.precision_burden_ratio,
            pb_over_prescribed=pb.is_over_prescribed,
            pb_under_prescribed=pb.is_under_prescribed,
        )

    @staticmethod
    def _extract_admission_drugs(patient_data: Dict) -> List[str]:
        """从患者数据提取入院时用药列表"""
        all_meds = patient_data.get("all_medications", [])
        admission = [
            m.get("drug_name", "") for m in all_meds
            if isinstance(m, dict)
            and m.get("time_window") in ("admission", "pre_admission", "prior")
            and m.get("drug_name")
        ]
        # 回退到 medications 字段
        if not admission:
            legacy = patient_data.get("medications", [])
            admission = [
                m if isinstance(m, str) else m.get("drug_name", "")
                for m in legacy if m
            ]
        return admission
