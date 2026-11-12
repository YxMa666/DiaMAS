#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diamas_utils/clinical_phenotyper.py
=====================================
临床表型推断器（从 diamas_inference_stage.py 抽出，L128–L1178）

包含：
  ClinicalPhenotypeState  — 单个临床表型状态数据结构
  ClinicalPhenotyper      — 三阶段推断器（Stage1确定性/Stage2概率性/Stage3 LLM辅助）

设计原则：
  Stage 1 规则 cite 到具体指南条款（KDIGO/ACC-AHA/ESC），可审计
  Stage 2 多特征组合推断，confidence 由可用特征数加权
  Stage 3 任务严格限定为"补充结构化字段遗漏的硬/软禁忌"，不做一般状态识别
"""

from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from agents.llm_client import DeepseekClient

logger = logging.getLogger(__name__)

# ── 从 config 导入临床编码集合 ───────────────────────────────────────────────
from agents.config import ClinicalCodeConfig  # noqa: F401
# ClinicalPhenotypeState 定义在本文件（@dataclass），不从 config 导入

# [Solution A] 声明式安全状态注册表
from diamas_utils.clinical_safety_states import (
    CRITICAL_SAFETY_STATE_SPECS,
    match_safety_specs,
)

# ── NaN 安全工具 ──────────────────────────────────────────────────────────────

def _safe_bool(v, default: bool = False) -> bool:
    if isinstance(v, bool): return v
    if v is None: return default
    if isinstance(v, float) and math.isnan(v): return default
    return bool(v)

def _safe_float(v, default=None):
    if v is None: return default
    if isinstance(v, float) and math.isnan(v): return default
    try: return float(v)
    except (TypeError, ValueError): return default

def _safe_str(v, default: str = "") -> str:
    if v is None: return default
    if isinstance(v, float) and math.isnan(v): return default
    return str(v)

def _is_potassium_extreme(k: Optional[float]) -> bool:
    """True when K⁺ is critically extreme (>7.0 or <2.0) — warrants highest urgency/confidence."""
    if k is None: return False
    return k > 7.0 or k < 2.0

# ── 从 config 读取常量 ────────────────────────────────────────────────────────
ASCVD_ICD9  = ClinicalCodeConfig.ASCVD_ICD9
ASCVD_ICD10 = ClinicalCodeConfig.ASCVD_ICD10
HF_ICD9     = ClinicalCodeConfig.HF_ICD9
HF_ICD10    = ClinicalCodeConfig.HF_ICD10


@dataclass
class ClinicalPhenotypeState:
    """单个临床表型状态（重新导出，保证向后兼容）。"""
    name:                str
    confidence:          float
    evidence_type:       str        # deterministic / probabilistic / llm_inferred
    supporting_features: Dict
    implication:         str
    specialty_tags:      List[str]
    era_sensitivity:     bool
    guideline_queries:   List[str]
    contraindication_hints: List[str] = field(default_factory=list)
    icd_anchor:          Optional[str] = None
    era_implications:    Optional[Dict[int, str]] = None


class ClinicalPhenotyper:
    """
    三阶段临床表型推断器。

    Stage 1 — Deterministic：国际分类标准硬性规则（KDIGO/ACC-AHA/ESC）
    Stage 2 — Probabilistic：多特征组合推断（EHR 信息不完整场景）
    Stage 3 — LLM-fallback：从自由文本补充结构化字段遗漏的禁忌信息
    """

    def __init__(self, llm_client: Optional["DeepseekClient"] = None):
        self._llm = llm_client

    def infer(self, flat: Dict, patient_data: Dict) -> List[ClinicalPhenotypeState]:
        states: List[ClinicalPhenotypeState] = []
        states.extend(self._stage0_dm_baseline(flat, patient_data))
        states.extend(self._stage1_deterministic(flat, patient_data))
        states.extend(self._stage2_probabilistic(flat, patient_data, existing=states))
        if self._llm:
            # [Solution B] 使用 critical_safety_states scope（8类结构化扫描，覆盖原 contraindications_only）
            states.extend(self._stage3_llm_fallback(
                flat, patient_data, existing=states, scope="critical_safety_states",
            ))
        states = self._deduplicate(states)
        states.sort(key=lambda s: (-len(s.specialty_tags), -s.confidence))
        return states

    # ── Stage 0: DM-type-aware 基础用药 state ────────────────────────────────

    def _stage0_dm_baseline(self, flat: Dict, patient_data: Dict) -> List[ClinicalPhenotypeState]:
        """
        Stage 0: 每个 DM 患者生成一个病情驱动的基础用药 state。

        设计原则（防御审稿人）：
          - 不 hardcode，根据 dm_type / 肾功能 / 血糖严重度 / 病房 vs ICU 动态分支
          - T1DM 跳过：Stage 1 已生成 "T1DM (insulin-dependent)" + 对应 queries
          - 4 个互斥分支，覆盖全部 T2DM 住院场景
          - queries 指向 ADA 指南中住院药物管理章节，
            与 Stage 1 的并发症 queries 互补，避免 RAG 盲区

        论文叙事：
          "The phenotyper generates a mandatory diabetes pharmacotherapy state for every
           T2DM patient, stratified by DM type, renal function, and glycemic severity,
           ensuring baseline guideline retrieval regardless of complication burden."
        """
        # ── T1DM 跳过（Stage 1 完整覆盖）────────────────────────────────────
        if _safe_bool(flat.get("is_t1dm")):
            return []

        # ── 基础字段读取 ──────────────────────────────────────────────────────
        dm_type = _safe_str(flat.get("dm_type", "T2DM")).upper() or "T2DM"
        dq      = patient_data.get("data_quality") or {}
        dm_conf = _safe_str(dq.get("dm_confidence_cached", "low")).lower()
        base_conf = 0.90 if dm_conf in ("high", "medium") else 0.75

        egfr    = _safe_float(flat.get("egfr"))
        aki     = _safe_bool(flat.get("aki_suspected"))
        hemo    = _safe_bool(flat.get("hemodynamic_instability"))

        # mean_glucose 不在 flat，从 patient_data 取
        gluc    = (patient_data.get("clinical_metrics") or {}).get("glucose") or {}
        mean_bg = _safe_float(gluc.get("mean_glucose"))

        # ICU-level: severity_tier="High" 或血流动力学不稳定
        severity = _safe_str(patient_data.get("severity_tier", "")).strip()
        is_icu   = (severity == "High") or hemo

        # 肾功能受限（eGFR<45 或 AKI）
        renal_limited = (egfr is not None and egfr < 45) or aki

        # 急性高血糖（均值 >250 或 ICU）
        acute_hyper = (mean_bg is not None and mean_bg > 250) or is_icu

        # ── 分支 1：ICU / 危重症 ─────────────────────────────────────────────
        if is_icu or hemo:
            return [ClinicalPhenotypeState(
                name="T2DM critical illness glycemic management",
                confidence=0.92,
                evidence_type="deterministic",
                supporting_features={
                    "dm_type": dm_type, "is_icu": is_icu,
                    "hemo": hemo, "mean_bg": mean_bg,
                },
                implication=(
                    "危重症 T2DM：暂停所有口服降糖药及 SGLT2i；"
                    "静脉胰岛素泵持续输注；目标血糖 140–180 mg/dL（ADA 住院指南 Level A）"
                ),
                specialty_tags=["endocrinology"],
                era_sensitivity=True,
                guideline_queries=[
                    "T2DM critical illness insulin infusion protocol ICU",
                    "hospitalized diabetes insulin therapy blood glucose target 140 180",
                    "inpatient hyperglycemia oral agent hold critical illness",
                    "ADA inpatient glycemic management ICU insulin targets",
                ],
                contraindication_hints=["oral_agents_icu", "sglt2i_critical_illness"],
                era_implications={
                    2010: "危重症：静脉胰岛素；目标 140-180（NICE-SUGAR 后放宽强化目标）",
                    2016: "危重症：胰岛素泵 140-180；所有口服药暂停",
                    2020: "危重症：胰岛素泵 140-180；SGLT2i 暂停（正常血糖 DKA 风险）",
                },
            )]

        # ── 分支 2：肾功能受限（eGFR<45 or AKI）────────────────────────────
        if renal_limited:
            egfr_str = f"eGFR {egfr:.0f}" if egfr is not None else ("AKI" if aki else "受损")
            return [ClinicalPhenotypeState(
                name="T2DM renal-limited inpatient pharmacotherapy",
                confidence=min(base_conf + 0.05, 0.90),
                evidence_type="deterministic",
                supporting_features={
                    "dm_type": dm_type, "egfr": egfr, "aki": aki,
                },
                implication=(
                    f"T2DM 合并肾功能受损（{egfr_str}）住院：Metformin 减量或停用；"
                    "SGLT2i 降糖效弱且 AKI 风险；DPP-4i（剂量调整）/ 胰岛素相对安全"
                ),
                specialty_tags=["endocrinology"],
                era_sensitivity=True,
                guideline_queries=[
                    "T2DM CKD inpatient pharmacotherapy drug selection renal impairment",
                    "metformin hold renal impairment hospitalized diabetes",
                    "insulin preferred T2DM reduced eGFR inpatient guideline",
                    "DPP4 inhibitor dose adjustment CKD diabetes inpatient",
                    "ADA diabetes CKD pharmacotherapy oral hypoglycemic agents",
                ],
                contraindication_hints=["metformin_aki"],
                era_implications={
                    2010: "肾功能受损：Metformin eGFR<45 慎用/停用；SU 减量；胰岛素优选",
                    2016: "肾功能受损：SGLT2i 降糖效弱（eGFR<45）；DPP-4i 需调整剂量",
                    2020: "肾功能受损：SGLT2i 肾保护（CREDENCE/DAPA-CKD）；Metformin 可至 eGFR≥30",
                },
            )]

        # ── 分支 3：急性高血糖（>250 mg/dL，非 ICU）────────────────────────
        if acute_hyper:
            return [ClinicalPhenotypeState(
                name="T2DM acute hyperglycemia inpatient management",
                confidence=0.88,
                evidence_type="deterministic",
                supporting_features={
                    "dm_type": dm_type, "mean_bg": mean_bg,
                },
                implication=(
                    f"T2DM 急性高血糖（均值 {mean_bg:.0f} mg/dL >250）住院："
                    "胰岛素为主（basal-bolus）；口服药暂停条件评估；目标血糖 140–180 mg/dL"
                ),
                specialty_tags=["endocrinology"],
                era_sensitivity=True,
                guideline_queries=[
                    "T2DM acute hyperglycemia inpatient basal bolus insulin initiation",
                    "hospitalized T2DM insulin start oral agent hold criteria",
                    "ADA inpatient diabetes pharmacotherapy glycemic target ward",
                    "insulin dose titration inpatient hyperglycemia protocol",
                ],
                contraindication_hints=[],
                era_implications={
                    2010: "急性高血糖：basal-bolus 胰岛素；Metformin 需评估；目标 140-180",
                    2016: "急性高血糖：basal-bolus 优于 sliding scale（ADA 2016 Level A）",
                    2020: "急性高血糖：胰岛素首选；SGLT2i 谨慎（DKA 风险）",
                },
            )]

        # ── 分支 4：稳定期 T2DM（标准住院，肾功能正常，血糖 ≤250）────────
        return [ClinicalPhenotypeState(
            name="T2DM stable inpatient pharmacotherapy",
            confidence=base_conf,
            evidence_type="deterministic",
            supporting_features={
                "dm_type": dm_type, "egfr": egfr, "mean_bg": mean_bg,
            },
            implication=(
                "T2DM 稳定期住院：急性稳定期可继续安全口服药（评估 AKI/造影禁忌后）；"
                "Metformin 评估后可维持；胰岛素作为覆盖调整工具；目标血糖 140–180 mg/dL"
            ),
            specialty_tags=["endocrinology"],
            era_sensitivity=True,
            guideline_queries=[
                "T2DM inpatient stable pharmacotherapy oral agents continuation criteria",
                "metformin continuation hospitalized stable T2DM no AKI",
                "basal insulin initiation T2DM general ward inpatient",
                "ADA standards inpatient diabetes management pharmacotherapy",
                "oral hypoglycemic agents inpatient acute stable phase policy",
            ],
            contraindication_hints=[],
            era_implications={
                2010: "稳定期：Metformin 可继续（无 AKI/造影）；需要时加基础胰岛素（NPH/glargine）",
                2016: "稳定期：Metformin 继续；SGLT2i/DPP-4i 安全；根据 LOS 决定新增药物",
                2020: "稳定期：个体化方案；SGLT2i/GLP-1RA 优先（有心肾获益）；避免低血糖",
            },
        )]

    # ── Stage 1: 确定性表型 ───────────────────────────────────────────────────

    def _stage1_deterministic(self, flat: Dict, patient_data: Dict) -> List[ClinicalPhenotypeState]:
        states: List[ClinicalPhenotypeState] = []
        icd = self._extract_icd_prefixes(patient_data)

        # ── KDIGO 2012 CKD 分期 ─────────────────────────────────────────────
        egfr = _safe_float(flat.get("egfr"))
        acr  = _safe_float(flat.get("urine_albumin_creatinine_ratio"))

        # [FIX-AKI-FALLBACK] aki_suspected 来自上游提取管道，有时因字段缺失而漏判。
        # 增加 ICD 码 fallback：若 all_diagnoses 包含 584.x（ICD-9 AKI）或
        # N17.x（ICD-10 AKI），也触发 AKI 检测——即使 flat["aki_suspected"]=False。
        # P15 案例验证：患者有 ICD 584.9 但 aki_suspected 未被提取管道正确设置。
        _AKI_ICD9_PFX  = {"584"}   # 584.x = acute renal failure (ICD-9)
        _AKI_ICD10_PFX = {"N17"}   # N17.x = acute kidney failure (ICD-10)
        _aki_icd = any(
            str((_d or {}).get("icd_code", "") or "").strip().upper()[:3]
            in (_AKI_ICD9_PFX | _AKI_ICD10_PFX)
            for _d in patient_data.get("all_diagnoses", [])
            if isinstance(_d, dict)
        )
        aki = _safe_bool(flat.get("aki_suspected")) or _aki_icd

        if egfr is not None:
            if egfr < 15:
                states.append(ClinicalPhenotypeState(
                    name="CKD G5 (ESRD)",          confidence=0.95, evidence_type="deterministic",
                    supporting_features={"egfr": egfr},
                    implication="eGFR<15: 禁用 Metformin/SGLT2i/GLP-1RA（脱水风险）；需透析评估；仅胰岛素安全",
                    specialty_tags=["nephrology"], era_sensitivity=True,
                    guideline_queries=["CKD G5 diabetes management", "ESRD insulin therapy"],
                    contraindication_hints=["metformin_egfr<30", "sglt2i_egfr<20", "glp1ra_egfr<15"],
                    era_implications={2010: "CKD G5: 胰岛素为唯一安全选择，口服药均禁忌", 2020: "CKD G5: 胰岛素首选；部分 SGLT2i 有肾保护证据但需 eGFR≥20"},
                ))
            elif egfr < 30:
                states.append(ClinicalPhenotypeState(
                    name="CKD G4",                 confidence=0.92, evidence_type="deterministic",
                    supporting_features={"egfr": egfr},
                    implication="eGFR 15-30: Metformin 绝对禁忌；SGLT2i 降糖效果消失；GLP-1RA 慎用",
                    specialty_tags=["nephrology"], era_sensitivity=True,
                    guideline_queries=["CKD G4 diabetes pharmacotherapy"],
                    contraindication_hints=["metformin_egfr<30"],
                    era_implications={2010: "CKD G4: 胰岛素/格列奈类/部分 DPP-4i（剂量调整）", 2020: "CKD G4: DPP-4i（需剂量调整）；SGLT2i 肾保护证据但降糖效弱"},
                ))
            elif egfr < 45:
                states.append(ClinicalPhenotypeState(
                    name="CKD G3b",                confidence=0.90, evidence_type="deterministic",
                    supporting_features={"egfr": egfr},
                    # [FIX-CKD-G3b] Base implication must be era-neutral.
                    # "SGLT2i 降糖减弱但仍有心肾获益" references post-2015 CVOT evidence.
                    # Moved to era_implications[2020]; base is now era-agnostic.
                    implication="eGFR 30-45（CKD G3b）：Metformin 减量或停用（eGFR<45 慎用）；SU 减量；避免 NSAIDs；胰岛素优选",
                    specialty_tags=["nephrology"], era_sensitivity=True,
                    guideline_queries=["CKD G3b metformin dose reduction diabetes", "CKD G3b diabetes drug selection ADA KDIGO"],
                    era_implications={
                        2010: "CKD G3b（≤2010）：Metformin 减量；SU减量；避免 NSAIDs；胰岛素相对安全",
                        2014: "CKD G3b（2014）：Metformin eGFR<45 慎用（ADA 2014/KDIGO 2012）；SGLT2i 降糖效力减弱；ACEi/ARB 肾保护",
                        2020: "CKD G3b（2020）：SGLT2i 优先（CREDENCE/DAPA-CKD 肾保护 Level A）；Metformin 可维持至 eGFR≥30",
                    },
                ))
            elif egfr < 60:
                states.append(ClinicalPhenotypeState(
                    name="CKD G3a",                confidence=0.85, evidence_type="deterministic",
                    supporting_features={"egfr": egfr},
                    # [FIX-CKD-G3a] "SGLT2i 心肾获益存在" references EMPA-REG 2015 / CREDENCE 2019.
                    # Base implication now era-neutral; SGLT2i benefit moved to era_implications[2020].
                    implication="eGFR 45-60（CKD G3a）：所有口服降糖药可用但需监测；Metformin 可继续（eGFR≥45）；SGLT2i 降糖有效",
                    specialty_tags=["nephrology"], era_sensitivity=True,
                    guideline_queries=["CKD G3a diabetes drug selection metformin", "CKD G3a ACEi ARB renoprotection"],
                    era_implications={
                        2010: "CKD G3a（≤2010）：Metformin 可用；SU 需监测；ACEi/ARB 肾保护（ADA 2010）",
                        2014: "CKD G3a（2014）：Metformin 可用（ADA 2014）；SGLT2i 降糖有效；ACEi/ARB 微量蛋白尿时启动",
                        2020: "CKD G3a（2020）：SGLT2i 心肾获益（EMPA-REG/CREDENCE Level A）；RAAS + SGLT2i 联合方案",
                    },
                ))

        if aki:
            states.append(ClinicalPhenotypeState(
                name="AKI suspected",              confidence=0.85, evidence_type="deterministic",
                supporting_features={"aki_suspected": True, "egfr": egfr},
                implication="疑似 AKI：立即停用 Metformin/SGLT2i/NSAIDs；胰岛素安全",
                specialty_tags=["nephrology"], era_sensitivity=False,
                guideline_queries=["AKI diabetes drug hold"],
                contraindication_hints=["metformin_aki", "sglt2i_aki"],
            ))

        if acr is not None and acr >= 300:
            states.append(ClinicalPhenotypeState(
                name="Macroalbuminuria (A3)",       confidence=0.88, evidence_type="deterministic",
                supporting_features={"acr": acr},
                # [FIX-MACRO] Base implication was "SGLT2i + RAAS 阻断是 ADA 2020+ 核心方案"
                # which references post-2015/2019 evidence. Now era-neutral.
                implication="大量蛋白尿（UACR≥300）：ACEi/ARB 一线肾保护（减少蛋白尿进展）；血压目标 <130/80 mmHg",
                specialty_tags=["nephrology"], era_sensitivity=True,
                guideline_queries=["macroalbuminuria ACEi ARB renoprotection diabetes CKD",
                                   "diabetic kidney disease proteinuria RAAS inhibitor guidelines"],
                era_implications={
                    2010: "A3 蛋白尿（≤2010）：ACEi/ARB 一线（ADA 2010/KDIGO 2012）；目标血压 <130/80",
                    2014: "A3 蛋白尿（2014）：ACEi/ARB 一线（KDIGO 2012/ADA 2014）；SGLT2i 未获批肾保护适应症",
                    2020: "A3 蛋白尿（2020）：SGLT2i（CREDENCE/DAPA-CKD Level A）；Finerenone（FIDELIO 2021）",
                },
            ))

        # ── ACC/AHA 2013 / ESC 2013 ASCVD ────────────────────────────────────
        has_ascvd_icd  = bool(icd & (ASCVD_ICD9 | ASCVD_ICD10))
        has_ascvd_flag = _safe_bool(flat.get("has_cardiovascular")) or _safe_bool(flat.get("has_ascvd"))

        # ── Text-based fallback：ICD 字段为空时从诊断描述文字匹配 ACS/ASHD 关键词 ──
        # 场景：MIMIC 中少量记录 icd_code 字段缺失，但 description/snomed_name 含有
        # "acute coronary syndrome"、"myocardial infarction" 等明确 ASCVD 文字
        _ACS_KW = {
            "acute coronary syndrome", "myocardial infarction", "stemi", "nstemi",
            "unstable angina", "ashd", "coronary artery disease", "ischemic heart",
            "stroke", "cerebral infarction", "tia", "transient ischemic",
            "peripheral arterial disease", "peripheral vascular disease",
        }
        has_ascvd_text = False
        _ascvd_text_source = ""
        if not has_ascvd_icd:
            for _diag in patient_data.get("all_diagnoses", []):
                if not isinstance(_diag, dict):
                    continue
                _desc = (
                    str(_diag.get("description", "") or "")
                    + " "
                    + str(_diag.get("snomed_name", "") or "")
                ).lower()
                if any(kw in _desc for kw in _ACS_KW):
                    has_ascvd_text = True
                    _ascvd_text_source = _desc[:80]
                    break

        if has_ascvd_icd or has_ascvd_flag or has_ascvd_text:
            conf = 0.93 if has_ascvd_icd else (0.85 if has_ascvd_text else 0.80)
            ev_type = "deterministic" if has_ascvd_icd else "probabilistic"
            states.append(ClinicalPhenotypeState(
                name="ASCVD confirmed",
                confidence=conf,
                evidence_type=ev_type,
                supporting_features={
                    "icd_based":   has_ascvd_icd,
                    "flag_based":  has_ascvd_flag,
                    "text_based":  has_ascvd_text,
                    "text_source": _ascvd_text_source,
                },
                implication="确诊 ASCVD：高强度他汀强制（ADA/ACC-AHA Level A）；抗血小板（阿司匹林±P2Y12，依据 ACS/卒中适应症）；RAAS 抑制剂（ACEi/ARB，尤其合并 HF/CKD/蛋白尿）；β-blocker（心梗后/HFrEF）",
                specialty_tags=["cardiology"],
                era_sensitivity=True,
                guideline_queries=[
                    "ASCVD statin antiplatelet therapy diabetes secondary prevention",
                    "acute coronary syndrome dual antiplatelet aspirin clopidogrel diabetes",
                    "ACEi ARB heart failure ASCVD diabetes cardiovascular outcomes",
                    "ASCVD SGLT2i GLP-1 cardiovascular outcomes diabetes",
                ],
                era_implications={
                    2010: (
                        "ASCVD（≤2010）：他汀（ADA 2010 Level A）；阿司匹林 75-162mg；"
                        "ACS 后双联抗板（阿司匹林+氯吡格雷）；ACEi/ARB；β-blocker（心梗后）"
                    ),
                    2014: (
                        "ASCVD（2014）：高强度他汀（ACC/AHA 2013 Class I Level A）；"
                        "ACS 后双联抗板≥12月（阿司匹林+氯吡格雷）；ACEi/ARB+β-blocker（HFrEF）；"
                        "抗血小板一级预防：70岁男性+DM+高危因素 → 阿司匹林 81mg（ADA 2013 Level C）"
                    ),
                    2020: "ASCVD: SGLT2i/GLP-1RA Level A；高强度他汀；ACS 后双联抗板",
                },
            ))

        # ── 心力衰竭 ─────────────────────────────────────────────────────────
        has_hf_icd  = bool(icd & (HF_ICD9 | HF_ICD10))
        has_hf_flag = _safe_bool(flat.get("has_heart_failure"))
        if has_hf_icd or has_hf_flag:
            conf = 0.93 if has_hf_icd else 0.82
            states.append(ClinicalPhenotypeState(
                name="Heart failure (HF)",         confidence=conf, evidence_type="deterministic",
                supporting_features={"icd_based": has_hf_icd, "flag_based": has_hf_flag},
                implication=(
                    "心力衰竭：禁用 TZD（体液潴留）；Saxagliptin/Alogliptin 慎用（心衰住院率可能升高）；"
                    "HFrEF 标准治疗：ACEi/ARB + β-blocker + MRA + 袋利尿剂；"
                    "血糖控制优先选用不加重心衰的药物（胰岛素、Metformin 如 HF 稳定可用）"
                ),
                specialty_tags=["cardiology"], era_sensitivity=True,
                guideline_queries=["heart failure diabetes drug selection inpatient", "TZD heart failure contraindication fluid retention", "saxagliptin alogliptin heart failure hospitalization risk", "heart failure ACEi beta-blocker MRA standard therapy diabetes"],
                contraindication_hints=["tzd_hf", "saxagliptin_hf", "alogliptin_hf"],
                era_implications={
                    2010: "HF（≤2010）：TZD 禁用；Metformin 可用（稳定 HF）；胰岛素注意鈢水潴留；ACEi+β-blocker+利尿剂",
                    2014: "HF（2014）：TZD 禁用；Saxagliptin 慎用（SAVOR-TIMI 2013）；Alogliptin 慎用（EXAMINE 2013）；Metformin 稳定 HF 可用；胰岛素优选",
                    2016: "HF（2016）：TZD 禁用；Saxagliptin/Alogliptin 慎用；EMPA-REG HF 亚组（2015）；SGLT2i 开始进入指南",
                    2020: "HF: SGLT2i（EMPEROR-Reduced/DAPA-HF）Level A；HFrEF/HFpEF 均有证据",
                },
            ))

        # ── DM 分型 ───────────────────────────────────────────────────────────
        dm_type = str((flat.get("dm_classification") or {}).get("type", "") or flat.get("dm_type", "")).upper()
        if "T1DM" in dm_type or "TYPE 1" in dm_type:
            states.append(ClinicalPhenotypeState(
                name="T1DM (insulin-dependent)",   confidence=0.92, evidence_type="deterministic",
                supporting_features={"dm_type": dm_type},
                implication="T1DM：胰岛素绝对必需（基础+餐时）；禁用磺脲类/格列奈类；SGLT2i 需 DKA 预防计划",
                specialty_tags=["endocrinology"], era_sensitivity=False,
                guideline_queries=["T1DM insulin management"],
                contraindication_hints=["sulfonylurea_t1dm", "meglitinide_t1dm"],
            ))

        # ── 高钾血症 (3-tier) ────────────────────────────────────────────────
        # [FIX-K] ≥5.5 → confirmed; 5.0–5.4 + AKI/sCKD → borderline; >7.0 → critical
        k = _safe_float(flat.get("potassium"))
        _aki_for_k = _safe_bool(flat.get("aki_suspected"))
        _egfr_for_k = _safe_float(flat.get("egfr"))
        _ckd_severe_for_k = (_egfr_for_k is not None and _egfr_for_k < 30)
        if k is not None and k >= 5.5:
            is_extreme = _is_potassium_extreme(k)
            _k_conf = min(0.97, 0.85 + 0.04 * (k - 5.5))
            if is_extreme: _k_conf = 0.97
            states.append(ClinicalPhenotypeState(
                name="Hyperkalemia" + (" (critical)" if is_extreme else ""),
                confidence=_k_conf, evidence_type="deterministic",
                supporting_features={"potassium": k, "critical": is_extreme,
                                     "aki": _aki_for_k, "egfr": _egfr_for_k},
                implication=(
                    f"高钾血症（K⁺={k} mEq/L）：住院急性期 ACEi/ARB 暂缓启动（非调整剂量）；"
                    "绝对禁用 Finerenone/Eplerenone/Spironolactone；监测心电图；Patiromer/SPS 考虑；"
                    + ("AKI 同时存在：RAAS 抑制剂为相对禁忌" if _aki_for_k else "")
                ),
                specialty_tags=["nephrology"], era_sensitivity=False,
                guideline_queries=["hyperkalemia ACEi ARB hold contraindication CKD diabetes",
                                   "hyperkalemia management inpatient potassium CKD AKI"],
                contraindication_hints=["acei_arb_hyperkalemia", "mra_hyperkalemia",
                                        "finerenone_hyperkalemia", "spironolactone_hyperkalemia"],
            ))
        elif k is not None and k >= 5.0 and (_aki_for_k or _ckd_severe_for_k):
            states.append(ClinicalPhenotypeState(
                name="Borderline hyperkalemia with ACEi-ARB caution",
                confidence=0.82, evidence_type="deterministic",
                supporting_features={"potassium": k, "aki": _aki_for_k, "egfr": _egfr_for_k},
                implication=(
                    f"边界高钾（K⁺={k} mEq/L）合并 AKI/严重 CKD："
                    "启动 ACEi/ARB 需极度谨慎；禁用 Finerenone/Spironolactone；"
                    "AKI 缓解、K⁺<5.0 后方可评估 RAAS 启动"
                ),
                specialty_tags=["nephrology"], era_sensitivity=False,
                guideline_queries=["borderline hyperkalemia ACEi ARB initiation risk CKD AKI",
                                   "potassium monitoring RAAS inhibitor CKD AKI caution"],
                contraindication_hints=["acei_arb_borderline_hyperkalemia", "mra_borderline_hyperkalemia"],
            ))

        # ── 糖尿病神经病变 ────────────────────────────────────────────────────
        # [FIX-NEURO] ICD-9: 357.2/250.6x | ICD-10: E11.40x/G63 | 文字关键词三级检测
        _NEURO_ICD9_EXACT  = {"357.2","250.6","250.60","250.61","250.62","250.63"}
        _NEURO_ICD10_EXACT = {"E10.40","E10.41","E10.42","E10.43","E10.44","E10.49",
                              "E11.40","E11.41","E11.42","E11.43","E11.44","E11.49"}
        _NEURO_ICD10_PFX = {"G63"}
        _NEURO_KW = {"neuropathy","peripheral neuropathy","diabetic neuropathy",
                     "polyneuropathy","numbness","burning feet","neuropathic pain"}
        _has_neuro_icd = False; _has_neuro_text = False
        for _d in patient_data.get("all_diagnoses", []):
            if not isinstance(_d, dict): continue
            _code = str(_d.get("icd_code", "") or "").strip().upper()
            _desc_low = (str(_d.get("description","") or "") + " " +
                         str(_d.get("snomed_name","") or "")).lower()
            if _code in _NEURO_ICD9_EXACT or _code in _NEURO_ICD10_EXACT:
                _has_neuro_icd = True; break
            if _code[:3] in _NEURO_ICD10_PFX:
                _has_neuro_icd = True; break
            if not _has_neuro_icd and any(kw in _desc_low for kw in _NEURO_KW):
                _has_neuro_text = True
        if _has_neuro_icd or _has_neuro_text:
            states.append(ClinicalPhenotypeState(
                name="Diabetic peripheral neuropathy",
                confidence=0.85 if _has_neuro_icd else 0.65,
                evidence_type="deterministic" if _has_neuro_icd else "probabilistic",
                supporting_features={"icd_based": _has_neuro_icd, "text_based": _has_neuro_text},
                implication="糖尿病周围神经病变：症状性神经痛首选 Duloxetine 或 Pregabalin/Gabapentin；TCA 老年人慎用",
                specialty_tags=["endocrinology","neurology"], era_sensitivity=True,
                guideline_queries=["diabetic peripheral neuropathy gabapentin pregabalin ADA",
                                   "neuropathic pain duloxetine gabapentin diabetes","ADA diabetic neuropathy 2013 2014"],
                era_implications={
                    2010: "神经病变（≤2010）：TCA、Gabapentin、Pregabalin、Duloxetine（ADA 2010 Level B）",
                    2014: "神经病变（2014）：Duloxetine/Pregabalin 一线；Gabapentin B级（ADA 2014）",
                    2020: "神经病变（2020）：Duloxetine/Pregabalin/Gabapentin 一线；外用辛辣素可选",
                },
            ))

        # ── 肝功能损害 ────────────────────────────────────────────────────────
        alt = _safe_float(flat.get("alt"))
        if alt is not None and alt > 120:
            states.append(ClinicalPhenotypeState(
                name="Hepatic impairment suspected", confidence=0.72, evidence_type="deterministic",
                supporting_features={"alt": alt},
                implication="肝功能损害：避免 Pioglitazone/Rosiglitazone；Metformin 需监测乳酸；胰岛素相对安全",
                specialty_tags=["endocrinology"], era_sensitivity=False,
                guideline_queries=["diabetes liver impairment drug selection"],
            ))

        # ── 类固醇诱导高血糖 ──────────────────────────────────────────────────
        steroid = _safe_bool(flat.get("on_systemic_steroids"))
        if steroid:
            states.append(ClinicalPhenotypeState(
                name="Steroid-induced hyperglycemia", confidence=0.80, evidence_type="deterministic",
                supporting_features={"on_systemic_steroids": True},
                implication="类固醇高血糖：餐后血糖主导 → 速效/NPH 胰岛素；餐时覆盖优先于基础",
                specialty_tags=["endocrinology"], era_sensitivity=False,
                guideline_queries=["steroid-induced hyperglycemia insulin management"],
            ))

        # ── 血流动力学不稳定 ──────────────────────────────────────────────────
        hemo = _safe_bool(flat.get("hemodynamic_instability"))
        if hemo:
            states.append(ClinicalPhenotypeState(
                name="Hemodynamic instability",    confidence=0.88, evidence_type="deterministic",
                supporting_features={"hemodynamic_instability": True},
                implication="血流动力学不稳定：ICU 急性期暂停所有口服降糖药；静脉胰岛素泵控制血糖",
                specialty_tags=["endocrinology"], era_sensitivity=False,
                guideline_queries=["ICU hyperglycemia insulin infusion hemodynamic instability"],
                contraindication_hints=["oral_agents_hemodynamic_instability"],
            ))

        # ── [Solution A] 表驱动安全状态扫描 ─────────────────────────────────
        # 遍历 CRITICAL_SAFETY_STATE_SPECS，命中的 ICD 码自动生成对应 state。
        # 新增场景只需在 clinical_safety_states.py 的表中追加 dict，此处无需修改。
        _existing_names = {s.name for s in states}
        for spec in match_safety_specs(patient_data, flat, _existing_names):
            states.append(ClinicalPhenotypeState(
                name=spec["name"],
                confidence=spec["confidence"],
                evidence_type=spec.get("evidence_type", "deterministic"),
                supporting_features={"source": "safety_table", "icd_based": True},
                implication=spec["implication"],
                specialty_tags=spec.get("specialty_tags", ["endocrinology"]),
                era_sensitivity=spec.get("era_sensitivity", False),
                guideline_queries=spec.get("guideline_queries", [spec["name"]]),
                contraindication_hints=spec.get("contraindication_hints", []),
                era_implications=spec.get("era_implications"),
            ))

        return states

    # ── Stage 2: 概率性表型 ───────────────────────────────────────────────────

    def _stage2_probabilistic(
        self, flat: Dict, patient_data: Dict,
        existing: List[ClinicalPhenotypeState],
    ) -> List[ClinicalPhenotypeState]:
        states: List[ClinicalPhenotypeState] = []
        existing_names = {s.name for s in existing}

        # HFrEF 高度怀疑（无 EF 值，但有 BNP/NT-proBNP/ICD/利尿剂）
        if "Heart failure (HF)" not in existing_names:
            bnp = _safe_float(flat.get("bnp") or flat.get("ntprobnp"))
            furosemide = _safe_bool(flat.get("on_loop_diuretic"))
            icd = self._extract_icd_prefixes(patient_data)
            has_hf_icd = bool(icd & (HF_ICD9 | HF_ICD10))
            if (bnp and bnp > 400) or (furosemide and has_hf_icd):
                features = []
                if bnp and bnp > 400: features.append(f"BNP/NT-proBNP={bnp}")
                if furosemide: features.append("on_loop_diuretic")
                conf = min(0.80, 0.55 + 0.10 * len(features))
                states.append(ClinicalPhenotypeState(
                    name="HFrEF suspected",        confidence=conf, evidence_type="probabilistic",
                    supporting_features={"features": features},
                    implication=(
                    "疑似 HFrEF：TZD 禁用（体液潴留）；利尿剂是容量管理基础；"
                    "ACEi/ARB + β-blocker 为标准 HFrEF 改善预后治疗"
                ),
                    specialty_tags=["cardiology"], era_sensitivity=True,
                    guideline_queries=["HFrEF ACEi beta-blocker standard therapy diabetes", "TZD heart failure fluid retention contraindication"],
                ))

        # 肥胖相关胰岛素抵抗
        bmi = _safe_float(flat.get("bmi"))
        if bmi and bmi >= 35:
            states.append(ClinicalPhenotypeState(
                name="Severe obesity + IR",        confidence=0.78, evidence_type="probabilistic",
                supporting_features={"bmi": bmi},
                implication=f"BMI={bmi:.1f}（重度肥胖）：GLP-1RA/SGLT2i 优先（减重获益）；避免促进体重增加的药物（SU/TZD/胰岛素）",
                specialty_tags=["endocrinology"], era_sensitivity=True,
                guideline_queries=["obesity T2DM GLP-1RA weight loss SGLT2i"],
            ))

        # 低血糖高风险
        age = _safe_float(flat.get("age"))
        hx_hypo = _safe_bool(flat.get("hypoglycemia_history") or flat.get("has_hypoglycemia"))
        ckd_flag = any(s.name.startswith("CKD G") for s in existing)
        _egfr_hypo = _safe_float(flat.get("egfr"))
        renal_hypo_risk = ckd_flag or (_egfr_hypo is not None and _egfr_hypo < 45)  # [FIX] dedup
        hypo_risk_count = sum([bool(age and age >= 75), hx_hypo, renal_hypo_risk])
        # [FIX-HYPO-SEVERE] eGFR<30 单独触发——肾清除显著延长，胰岛素/SU 蓄积风险极高，
        # 即使只有"肾损"这一个风险因素也应生成 state（不依赖总分≥2）。
        renal_severe = (_egfr_hypo is not None and _egfr_hypo < 30)
        if renal_severe or hypo_risk_count >= 2:
            conf = 0.70 + 0.05 * hypo_risk_count
            states.append(ClinicalPhenotypeState(
                name="High hypoglycemia risk",     confidence=min(conf, 0.88), evidence_type="probabilistic",
                supporting_features={"age": age, "hx_hypo": hx_hypo, "ckd": ckd_flag},
                implication="高低血糖风险：避免磺脲类/格列奈类；首选 DPP-4i/GLP-1RA/SGLT2i；胰岛素需谨慎调量",
                specialty_tags=["endocrinology"], era_sensitivity=False,
                guideline_queries=["hypoglycemia risk T2DM elderly CKD safe agents"],
            ))

        # ── 隐式 ASCVD 极高危：CKD G4/G5 + T2DM ────────────────────────────────
        # ESC 2013 将 eGFR<30 合并 T2DM 定义为 ASCVD 极高危等效状态（Class I），
        # 即使无明确 ASCVD ICD，也强制他汀 + 考虑抗血小板，触发心内科 MDT 证据路由。
        if "ASCVD confirmed" not in existing_names:
            has_severe_ckd = any(
                s.name.startswith("CKD G4") or s.name.startswith("CKD G5")
                for s in existing
            )
            egfr_v = _safe_float(flat.get("egfr"))
            ckd_severe_egfr = (egfr_v is not None and egfr_v < 30)
            if has_severe_ckd or ckd_severe_egfr:
                states.append(ClinicalPhenotypeState(
                    name="ASCVD very high risk (CKD G4+T2DM)",
                    confidence=0.82,
                    evidence_type="probabilistic",
                    supporting_features={
                        "egfr": egfr_v,
                        "basis": "CKD G4/G5 + T2DM = ESC 2013 very high CV risk equivalent",
                    },
                    implication=(
                        "CKD G4/G5 合并 T2DM：ESC 2013 定义为心血管极高危等效状态。"
                        "强制他汀治疗（ADA 2012/ESC 2013 Class I Level A）；"
                        "阿司匹林 81mg（无出血禁忌时用于心血管一级预防）；"
                        "ACEi/ARB 肾保护（待 AKI 缓解、K⁺ 正常后启动）"
                    ),
                    specialty_tags=["cardiology"],
                    era_sensitivity=True,
                    guideline_queries=[
                        "statin therapy T2DM CKD very high cardiovascular risk prevention",
                        "antiplatelet aspirin diabetes CKD primary prevention cardiovascular",
                        "ACEi ARB diabetic nephropathy CKD G4 renoprotection cardiovascular",
                        "ESC 2013 cardiovascular risk stratification diabetes CKD very high risk statin",
                        "ACC AHA 2013 statin diabetes CKD primary prevention high intensity",
                    ],
                    era_implications={
                        2010: "CKD G4+DM 极高危：他汀（ADA 2010/ESC 2007 B级）；阿司匹林 75-162mg；RAAS 抑制剂",
                        2014: "CKD G4+DM：他汀 A 级（ACC/AHA 2013 Pooled Cohort）；阿司匹林 81mg；ACEi/ARB 一线",
                        2020: "CKD G4+DM：他汀 A 级 + SGLT2i 心肾保护（CREDENCE/DAPA-CKD）；Finerenone CKD-MRA",
                    },
                ))

        # ── 急性呼吸衰竭 + 容量过载可疑（袢利尿剂适应症） ──────────────────────
        # ICD-9: 518.x（呼吸衰竭）；ICD-10: J96.x（急性呼衰）
        # 合并 AKI/高肌酐时：临床高度怀疑容量过载/急性肺水肿，袢利尿剂有适应症
        _RESP_ICD9_PFX  = {"518"}
        _RESP_ICD10_PFX = {"J96"}
        has_resp_failure = False
        for _d in (patient_data.get("all_diagnoses") or []):
            if not isinstance(_d, dict):
                continue
            _code = str(_d.get("icd_code", "") or "").strip().upper()
            if _code[:3] in _RESP_ICD9_PFX or _code[:3] in _RESP_ICD10_PFX:
                has_resp_failure = True
                break

        if has_resp_failure:
            _aki_v = _safe_bool(flat.get("aki_suspected"))
            _cr_v  = _safe_float(flat.get("creatinine"))
            _hf_present = any(
                s.name.startswith("Heart failure") or s.name.startswith("HFrEF")
                for s in (existing + states)
            )
            # 仅在 HF 未确诊时添加（HF 状态已内含 Furosemide 适应症路径）
            if not _hf_present and (_aki_v or (_cr_v and _cr_v > 1.5)):
                states.append(ClinicalPhenotypeState(
                    name="Acute respiratory failure + volume overload suspected",
                    confidence=0.68,
                    evidence_type="probabilistic",
                    supporting_features={
                        "resp_failure": True, "aki": _aki_v, "creatinine": _cr_v,
                    },
                    implication=(
                        "急性呼吸衰竭合并肾功能损害：需评估容量状态（急性肺水肿/液体过载）。"
                        "若容量过载明确，袢利尿剂（Furosemide/Bumetanide）为一线干预；"
                        "AKI 期间需同步监测尿量、肌酐、电解质"
                    ),
                    specialty_tags=["cardiology", "nephrology"],
                    era_sensitivity=False,
                    guideline_queries=[
                        "loop diuretic acute pulmonary edema volume overload management",
                        "furosemide bumetanide acute respiratory failure fluid overload",
                        "diuretic therapy AKI volume overload acute decompensation",
                    ],
                ))

        return states

    # ── Stage 3: LLM fallback（结构化8类安全扫描）────────────────────────────
    _STAGE3_CATEGORIES = (
        "A. 出血风险（颅内/GI/手术后活动性出血）→ 抗血小板/抗凝禁忌\n"
        "B. 急性器官衰竭（急性肝衰/急性胰腺炎/脓毒症）→ 口服降糖药暂停\n"
        "C. DM代谢急症（DKA/HHS/严重低血糖）→ 静脉胰岛素必需\n"
        "D. 血流动力学不稳定（休克/低血压/升压药依赖）→ 口服药全停\n"
        "E. 妊娠（妊娠期DM/GDM）→ SU/TZD/SGLT2i/GLP-1RA禁用\n"
        "F. 血液系统（严重血小板减少<50×10⁹/L）→ 抗血小板相对禁忌\n"
        "G. 围手术期（心脏大手术/腹部大手术<72h）→ 口服药围手术期暂停\n"
        "H. 药物不耐受/过敏（有明确文本记录）→ 具体药物禁用"
    )

    def _stage3_llm_fallback(
        self, flat: Dict, patient_data: Dict,
        existing: List[ClinicalPhenotypeState],
        max_tokens: int = 8000,
        scope: str = "critical_safety_states",
    ) -> List[ClinicalPhenotypeState]:
        if self._llm is None:
            return []
        existing_names_str = "; ".join(s.name for s in existing) or "无"
        diag_text = "; ".join(
            str(d.get("description", ""))
            for d in patient_data.get("all_diagnoses", [])[:12]
            if isinstance(d.get("description"), str)
        )
        notes_text = " | ".join(
            str(n)[:200] for n in (
                patient_data.get("discharge_notes", []) or
                patient_data.get("clinical_notes", []) or []
            )[:3] if n
        )

        if scope == "critical_safety_states":
            prompt = (
                f"患者诊断列表：{diag_text}\n"
                f"病程/手术记录摘要：{notes_text or '（无）'}\n"
                f"已识别状态（勿重复）：{existing_names_str}\n\n"
                "你是临床安全状态检测器。请扫描以下 8 个类别，"
                "仅对有明确诊断依据的类别输出状态（不得推断）：\n\n"
                f"{self._STAGE3_CATEGORIES}\n\n"
                "输出规则：有明确依据→JSON对象；无依据→跳过；confidence=0.7-0.95\n"
                "输出格式（严格JSON数组，无多余文字）：\n"
                '[{"name":"状态名","confidence":0.88,"implication":"对DM处方含义（中文）",'
                '"specialty":"endocrinology","contraindication_hints":["hint1"],"category":"A"}]\n'
                "无命中则输出 []\n\n"
                "⚠ 硬性约束（违反则不输出该条目）：\n"
                "1. [A类出血风险] 仅当诊断列表中有以下任一内容时才可输出：\n"
                "   (a) ICD码前缀 430-432/578/I60-I62/K92，或\n"
                "   (b) 诊断描述中有明确词汇：hemorrhage/bleeding/haemorrhage/\n"
                "       subarachnoid/intracranial bleed/GI bleed。\n"
                "   禁止：不得从'腹水+血小板减少+AST升高'等组合推断出血性诊断。\n"
                "2. [所有类别] 每条输出的 confidence 最高 0.75（llm_inferred 类型，\n"
                "   低于 deterministic 的 ICD 匹配结果，下游将以更低权重处理）。\n"
                "3. 多步推断禁止：不允许 A→B→C 式逻辑推断（如：腹水→肝硬化→静脉曲张→出血）。\n"
                "   只接受直接诊断证据（一步）。"
            )
        elif scope == "contraindications_only":
            prompt = (
                f"患者诊断列表：{diag_text}\n已识别状态：{existing_names_str}\n\n"
                "仅识别结构化字段遗漏的明确禁忌（手术/应激/过敏）。无明确依据则输出 []。\n"
                '[{"name":"状态名","confidence":0.85,"implication":"含义","specialty":"endocrinology"}]'
            )
        else:
            prompt = (
                f"患者诊断：{diag_text}\n已识别状态：{existing_names_str}\n\n"
                "识别尚未发现的重要病理生理状态。输出JSON数组（无则[]）：\n"
                '[{"name":"状态名","confidence":0.6,"implication":"含义","specialty":"endocrinology"}]'
            )
        try:
            response = self._llm.chat(
                messages=[{"role": "user", "content": prompt}],
                model=None, temperature=0.1, max_tokens=max_tokens,
            )
            text = self._llm.get_response_text(response) or ""
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
            match = re.search(r"\[.*\]", text, re.DOTALL)
            if not match:
                return []
            items = json.loads(match.group(0))
            return [
                ClinicalPhenotypeState(
                    name=item["name"],
                    confidence=float(item.get("confidence", 0.5)),
                    evidence_type="llm_inferred",
                    supporting_features={"source": f"llm_stage3_{scope}",
                                         "category": item.get("category", "?")},
                    implication=item.get("implication", ""),
                    specialty_tags=[item.get("specialty", "endocrinology")],
                    era_sensitivity=False,
                    guideline_queries=[item["name"]],
                    contraindication_hints=item.get("contraindication_hints", []),
                )
                for item in items
                if isinstance(item, dict) and item.get("name")
            ]
        except Exception as e:
            # 不 raise:Stage-3 是规则表型之上的补全,单例失败不应废掉整个患者。
            # 但必须可见 + 可计数,否则重演"100% 失败但无人知晓"。
            type(self)._stage3_fail_count = getattr(type(self), "_stage3_fail_count", 0) + 1
            logger.error("[Stage3-FAIL][%s] #%d %s: %s",
                         scope, type(self)._stage3_fail_count, type(e).__name__, e)
            return []

    # ── 工具方法 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_icd_prefixes(patient_data: Dict) -> Set[str]:
        codes: Set[str] = set()
        for diag in patient_data.get("all_diagnoses", []):
            if not isinstance(diag, dict): continue
            code = str(diag.get("icd_code", "") or "").strip().upper()
            if code:
                codes.add(code[:3])
                codes.add(code[:4])
        return codes

    @staticmethod
    def _deduplicate(states: List[ClinicalPhenotypeState]) -> List[ClinicalPhenotypeState]:
        seen: Dict[str, ClinicalPhenotypeState] = {}
        for s in states:
            if s.name not in seen:
                seen[s.name] = s
            else:
                existing = seen[s.name]
                winner = s if s.confidence > existing.confidence else existing
                loser  = existing if s.confidence > existing.confidence else s
                winner.specialty_tags = list(dict.fromkeys(winner.specialty_tags + loser.specialty_tags))
                merged = list(dict.fromkeys(h.lower() for h in winner.contraindication_hints))
                for h in loser.contraindication_hints:
                    if h.lower() not in {x.lower() for x in winner.contraindication_hints}:
                        merged.append(h.lower())
                winner.contraindication_hints = merged
                seen[s.name] = winner
        return list(seen.values())
