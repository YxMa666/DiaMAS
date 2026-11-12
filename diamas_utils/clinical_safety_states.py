#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clinical_safety_states.py
==========================
声明式临床安全状态注册表 (Solution A)

设计原则：
  - 将分散在 clinical_phenotyper.py Stage 1 中的"简单 ICD→state"逻辑集中到此表
  - 保留 phenotyper Stage 1 中需要复杂条件逻辑的状态（CKD分期/高钾分级/神经病变三层匹配等）
  - 此表只处理"ICD码确定→state"的直接映射，无复杂判断分支
  - 每个 spec 有两种 ICD 匹配模式：
      icd_prefixes:   前3字符匹配（快速，适合大类如 578.x）
      icd_substrings: startswith 匹配（精确，适合子分类如 250.1x=DKA）
  - modifier: 可选 callable(flat, patient_data) -> dict，用于覆盖默认字段
    （如 GI出血+低血压→升高 confidence）

新增状态时只需在 CRITICAL_SAFETY_STATE_SPECS 追加一条 dict，
phenotyper 的循环会自动驱动。无需修改 phenotyper 主逻辑。
"""

from typing import Any, Callable, Dict, List, Optional, Set

# ── 辅助工具 ─────────────────────────────────────────────────────────────────

import math


def _sf(v, default=None):
    if v is None: return default
    if isinstance(v, float) and math.isnan(v): return default
    try: return float(v)
    except: return default


def _sb(v, default: bool = False) -> bool:
    if isinstance(v, bool): return v
    if v is None: return default
    if isinstance(v, float) and math.isnan(v): return default
    return bool(v)


# ── Modifier 函数（GI出血合并低血压→升高confidence）─────────────────────────

def _gi_bleed_modifier(flat: Dict, patient_data: Dict) -> Dict:
    """活动性GI出血合并血流动力学不稳定时升高 confidence 并补充 implication。"""
    if _sb(flat.get("hemodynamic_instability")):
        return {
            "confidence": 0.93,
            "implication": (
                "活动性消化道出血合并低血压/血流动力学不稳定：抗血小板/NSAIDs明确禁忌；"
                "所有非必要口服降糖药立即暂停；静脉胰岛素控糖；PPI静脉输注；"
                "优先止血和容量复苏，血糖目标放宽至140-180 mg/dL"
            ),
        }
    return {}


def _pancreatitis_modifier(flat: Dict, patient_data: Dict) -> Dict:
    """急性胰腺炎合并器官衰竭时升高 confidence。"""
    egfr = _sf(flat.get("egfr"))
    if egfr is not None and egfr < 30:
        return {
            "confidence": 0.92,
            "implication": (
                "急性胰腺炎合并肾功能损害：口服降糖药全部暂停；胰岛素静脉输注；"
                "GLP-1RA/DPP-4i急性期暂停；目标血糖140-180 mg/dL（重症胰腺炎标准）"
            ),
        }
    return {}


# ── 声明式安全状态规范表 ─────────────────────────────────────────────────────
#
# 字段说明：
#   name            : ClinicalPhenotypeState.name（唯一标识，用于去重）
#   icd_prefixes    : set[str]  — 前3字符精确匹配（ICD-9 和 ICD-10 混合）
#   icd_substrings  : set[str]  — startswith 匹配（用于子分类精确限定）
#   confidence      : float     — 默认置信度
#   evidence_type   : str       — "deterministic" | "probabilistic"
#   implication     : str       — 药物含义说明（注入 Section 2）
#   specialty_tags  : list[str] — 触发 MDT 专科路由
#   era_sensitivity : bool
#   guideline_queries: list[str] — RAG 检索查询
#   contraindication_hints: list[str] — 传递给 specialist prompt 的禁忌信号
#   era_implications: dict[int, str] — 可选，时代匹配说明
#   modifier        : callable  — 可选，(flat, patient_data) -> dict 覆盖字段
#   exclude_if      : set[str]  — 若已存在这些 state name 则跳过（防重复）
#
CRITICAL_SAFETY_STATE_SPECS: List[Dict[str, Any]] = [

    # ════════════════════════════════════════════════════════════════════
    # 类别 1：出血 / 凝血风险
    # ════════════════════════════════════════════════════════════════════

    {
        "name": "Hemorrhagic stroke / SAH",
        "icd_prefixes": {"430", "431", "432",          # ICD-9: SAH, 脑内出血, 其他颅内出血
                         "I60", "I61", "I62"},          # ICD-10: 对应分类
        "icd_substrings": set(),
        "confidence": 0.92,
        "evidence_type": "deterministic",
        "implication": (
            "出血性卒中/蛛网膜下腔出血：急性期抗血小板（阿司匹林/氯吡格雷）相对禁忌，"
            "需神经外科/神经内科会诊评估出血稳定性后方可权衡启动；"
            "tPA溶栓绝对禁忌；抗凝需极度谨慎；"
            "同时存在ASCVD抗板适应症时须多科会诊，权衡出血 vs 缺血风险"
        ),
        "specialty_tags": ["neurology", "cardiology"],
        "era_sensitivity": False,
        "guideline_queries": [
            "subarachnoid hemorrhage antiplatelet contraindication restart timing",
            "intracranial hemorrhage aspirin clopidogrel risk benefit",
            "hemorrhagic stroke secondary prevention antiplatelet guidelines",
        ],
        "contraindication_hints": [
            "antiplatelet_hemorrhagic_stroke",
            "thrombolytic_hemorrhagic_stroke",
            "anticoagulant_hemorrhagic_stroke_caution",
        ],
        "modifier": None,
        "exclude_if": set(),
    },

    {
        "name": "Active GI bleeding",
        "icd_prefixes": {"578",                        # ICD-9: 消化道出血
                         "K92"},                       # ICD-10: K92.0呕血 K92.1黑便 K92.2NOS
        "icd_substrings": set(),
        "confidence": 0.88,
        "evidence_type": "deterministic",
        "implication": (
            "活动性消化道出血：抗血小板（阿司匹林/氯吡格雷）明确禁忌；"
            "NSAIDs绝对禁忌（直接加重黏膜损伤）；"
            "抗凝（Warfarin/肝素）需权衡后极度谨慎；"
            "PPI静脉输注为标准辅助治疗；胰岛素为首选血糖控制手段"
        ),
        "specialty_tags": ["gastroenterology", "endocrinology"],
        "era_sensitivity": False,
        "guideline_queries": [
            "GI bleeding antiplatelet aspirin clopidogrel contraindication acute",
            "gastrointestinal hemorrhage NSAID hold diabetes oral agent safety",
            "acute GI bleed oral hypoglycemic agent hold insulin inpatient",
        ],
        "contraindication_hints": [
            "antiplatelet_gi_bleed",
            "nsaid_gi_bleed",
            "anticoagulant_gi_bleed_caution",
        ],
        "modifier": _gi_bleed_modifier,
        "exclude_if": set(),
    },

    {
        "name": "Active peptic ulcer disease",
        "icd_prefixes": {"531", "532", "533", "534",   # ICD-9: 胃/十二指肠/空肠溃疡
                         "K25", "K26", "K27", "K28"},  # ICD-10
        "icd_substrings": set(),
        "confidence": 0.80,
        "evidence_type": "deterministic",
        "implication": (
            "活动性消化性溃疡（无活动出血）：阿司匹林需联合PPI胃保护后评估；"
            "NSAIDs加重溃疡应避免；Metformin可致GI不适但不直接加重溃疡；"
            "若有活动出血表现，参考 Active GI bleeding state"
        ),
        "specialty_tags": ["gastroenterology"],
        "era_sensitivity": False,
        "guideline_queries": [
            "peptic ulcer disease aspirin PPI gastric protection",
            "NSAID peptic ulcer contraindication diabetes",
        ],
        "contraindication_hints": [
            "nsaid_peptic_ulcer",
            "aspirin_pud_ppi_required",
        ],
        "modifier": None,
        "exclude_if": {"Active GI bleeding"},  # 已有出血则不再重复
    },

    # ════════════════════════════════════════════════════════════════════
    # 类别 2：急性器官功能障碍
    # ════════════════════════════════════════════════════════════════════

    {
        "name": "Acute pancreatitis",
        "icd_prefixes": set(),
        "icd_substrings": {"577.0",                    # ICD-9: 急性胰腺炎（577=慢性，577.0=急性）
                           "K85"},                     # ICD-10: K85.x 急性胰腺炎
        "confidence": 0.87,
        "evidence_type": "deterministic",
        "implication": (
            "急性胰腺炎：暂停所有口服降糖药；"
            "GLP-1RA/DPP-4i急性期禁用（FDA 2013胰腺炎安全警告）；"
            "胰岛素为唯一安全血糖控制手段；目标血糖140-180 mg/dL"
        ),
        "specialty_tags": ["gastroenterology", "endocrinology"],
        "era_sensitivity": True,
        "guideline_queries": [
            "acute pancreatitis diabetes oral agent hold metformin safety",
            "GLP-1RA DPP4 inhibitor pancreatitis risk FDA warning hold",
            "insulin therapy acute pancreatitis hyperglycemia inpatient",
        ],
        "contraindication_hints": [
            "oral_agents_acute_pancreatitis",
            "glp1ra_pancreatitis_fda_warning",
            "dpp4i_pancreatitis_caution",
        ],
        "era_implications": {
            2010: "急性胰腺炎：口服药暂停，胰岛素控糖；GLP-1RA胰腺炎风险数据不足",
            2014: "急性胰腺炎：FDA 2013警告DPP-4i/GLP-1RA胰腺炎信号；急性期全部暂停",
            2020: "急性胰腺炎：因果关系未证实但FDA维持警告；急性期暂停GLP-1RA/DPP-4i",
        },
        "modifier": _pancreatitis_modifier,
        "exclude_if": set(),
    },

    {
        "name": "Acute liver failure",
        "icd_prefixes": set(),
        "icd_substrings": {"570",                      # ICD-9: 急性/亚急性肝坏死
                           "K72.0", "K72.00", "K72.01"},  # ICD-10: 急性/亚急性肝衰竭
        "confidence": 0.90,
        "evidence_type": "deterministic",
        "implication": (
            "急性肝衰竭：Metformin绝对禁忌（乳酸酸中毒风险极高）；"
            "TZD禁用（肝脏代谢）；SU/格列奈类禁用（低血糖+肝代谢受损）；"
            "胰岛素为唯一安全选择，需减量防低血糖；密切监测血糖"
        ),
        "specialty_tags": ["hepatology", "endocrinology"],
        "era_sensitivity": False,
        "guideline_queries": [
            "acute liver failure diabetes metformin contraindication lactic acidosis",
            "hepatic failure insulin glucose management TZD contraindication",
        ],
        "contraindication_hints": [
            "metformin_liver_failure",
            "tzd_liver_failure",
            "su_liver_failure",
            "meglitinide_liver_failure",
        ],
        "modifier": None,
        "exclude_if": {"Hepatic impairment suspected"},  # Stage 1 已有 ALT>120 的轻度肝损state
    },

    {
        "name": "Sepsis / bacteremia",
        "icd_prefixes": {"038",                        # ICD-9: 败血症/菌血症
                         "A40", "A41"},                # ICD-10: 链球菌/其他细菌败血症
        "icd_substrings": {"995.9",                    # ICD-9: 995.91=sepsis, 995.92=severe
                           "R65.2"},                   # ICD-10: 严重脓毒症
        "confidence": 0.85,
        "evidence_type": "deterministic",
        "implication": (
            "脓毒症/败血症：血流动力学不稳定+肾功能波动，暂停所有口服降糖药；"
            "Metformin乳酸酸中毒风险极高；"
            "静脉胰岛素持续输注为标准治疗；目标血糖140-180 mg/dL（ADA住院指南）"
        ),
        "specialty_tags": ["infectious_disease", "endocrinology"],
        "era_sensitivity": False,
        "guideline_queries": [
            "sepsis diabetes insulin protocol glucose target inpatient",
            "bacteremia metformin hold oral agent safety acute infection",
        ],
        "contraindication_hints": [
            "oral_agents_sepsis",
            "metformin_sepsis",
        ],
        "modifier": None,
        "exclude_if": set(),
    },

    # ════════════════════════════════════════════════════════════════════
    # 类别 3：DM代谢急症
    # ════════════════════════════════════════════════════════════════════

    {
        "name": "Diabetic ketoacidosis (DKA)",
        "icd_prefixes": set(),
        "icd_substrings": {
            "250.1",                                   # ICD-9: 250.10-250.13 DKA
            "E10.1", "E11.1", "E13.1",                # ICD-10: T1DM/T2DM/其他DM DKA
        },
        "confidence": 0.93,
        "evidence_type": "deterministic",
        "implication": (
            "糖尿病酮症酸中毒：立即静脉胰岛素+补液（ADA DKA方案）；"
            "暂停所有口服降糖药；SGLT2i相关正常血糖DKA需特别注意；"
            "血糖<250 mg/dL时加用含糖液，防止血糖过快下降"
        ),
        "specialty_tags": ["endocrinology"],
        "era_sensitivity": False,
        "guideline_queries": [
            "DKA treatment protocol insulin infusion hydration ADA guideline",
            "diabetic ketoacidosis management inpatient glucose target",
        ],
        "contraindication_hints": [
            "oral_agents_dka",
            "sglt2i_dka",
        ],
        "modifier": None,
        "exclude_if": set(),
    },

    {
        "name": "Hyperosmolar hyperglycemic state (HHS)",
        "icd_prefixes": set(),
        "icd_substrings": {
            "250.2",                                   # ICD-9: 250.20-250.23 高渗性昏迷
            "E11.0", "E13.0",                          # ICD-10: T2DM/其他DM高渗
        },
        "confidence": 0.92,
        "evidence_type": "deterministic",
        "implication": (
            "高渗高血糖状态：缓慢静脉补液+胰岛素（速度较DKA更慢，防止脑水肿）；"
            "所有口服降糖药暂停；目标血糖250-300 mg/dL（前12h），然后逐步降低"
        ),
        "specialty_tags": ["endocrinology"],
        "era_sensitivity": False,
        "guideline_queries": [
            "hyperosmolar hyperglycemic state treatment insulin fluid replacement",
            "HHS management inpatient protocol ADA guideline",
        ],
        "contraindication_hints": [
            "oral_agents_hhs",
        ],
        "modifier": None,
        "exclude_if": set(),
    },

    # ════════════════════════════════════════════════════════════════════
    # 类别 4：妊娠
    # ════════════════════════════════════════════════════════════════════

    {
        "name": "Pregnancy with diabetes",
        "icd_prefixes": {"V22", "V23",                 # ICD-9: 正常妊娠/高危妊娠监护
                         "Z34", "O24"},                # ICD-10: 妊娠监护/妊娠期DM
        "icd_substrings": {"648.0", "648.8"},          # ICD-9: 妊娠合并DM/妊娠期DM
        "confidence": 0.92,
        "evidence_type": "deterministic",
        "implication": (
            "妊娠期糖尿病：胰岛素为首选（最安全）；"
            "禁用SU/TZD/SGLT2i/GLP-1RA（胎儿安全性证据不足）；"
            "DPP-4i妊娠期数据极有限，不推荐；"
            "Metformin有争议（可透过胎盘），部分指南允许但非首选"
        ),
        "specialty_tags": ["obstetrics", "endocrinology"],
        "era_sensitivity": True,
        "guideline_queries": [
            "gestational diabetes insulin therapy pregnancy safety",
            "diabetes pregnancy oral agent safety metformin ADA guideline",
        ],
        "contraindication_hints": [
            "su_pregnancy",
            "tzd_pregnancy",
            "sglt2i_pregnancy",
            "glp1ra_pregnancy",
            "dpp4i_pregnancy_limited_data",
        ],
        "era_implications": {
            2010: "妊娠期：胰岛素首选；Metformin数据有限但部分指南允许",
            2014: "妊娠期：胰岛素首选（ADA 2014）；Glyburide安全性存疑；所有新型药物禁用",
            2020: "妊娠期：胰岛素/Metformin；SGLT2i/GLP-1RA均缺乏充分安全性证据",
        },
        "modifier": None,
        "exclude_if": set(),
    },

    # ════════════════════════════════════════════════════════════════════
    # 类别 5：血液系统
    # ════════════════════════════════════════════════════════════════════

    {
        "name": "Severe thrombocytopenia",
        "icd_prefixes": {"287",                        # ICD-9: 287.3-287.5 血小板减少
                         "D69"},                       # ICD-10: D69.6 血小板减少NOS
        "icd_substrings": set(),
        "confidence": 0.78,
        "evidence_type": "deterministic",
        "implication": (
            "血小板减少（<50×10⁹/L时）：抗血小板药物相对禁忌，出血风险显著升高；"
            "阿司匹林/氯吡格雷需血液科会诊评估抗板收益后方可决定"
        ),
        "specialty_tags": ["hematology", "cardiology"],
        "era_sensitivity": False,
        "guideline_queries": [
            "thrombocytopenia antiplatelet aspirin safety bleeding risk",
        ],
        "contraindication_hints": [
            "antiplatelet_thrombocytopenia_caution",
        ],
        "modifier": None,
        "exclude_if": set(),
    },

    # ════════════════════════════════════════════════════════════════════
    # 类别 6：手术/操作后
    # ════════════════════════════════════════════════════════════════════

    {
        "name": "Post-major-cardiac-surgery",
        "icd_prefixes": set(),
        "icd_substrings": {
            "V43.3",                                   # ICD-9: 心脏瓣膜置换术后
            "Z95.1", "Z95.2", "Z95.3", "Z95.4",       # ICD-10: 冠状动脉/瓣膜假体
        },
        "confidence": 0.75,
        "evidence_type": "deterministic",
        "implication": (
            "心脏大手术后（CABG/瓣膜置换）：围手术期口服降糖药暂停；"
            "胰岛素静脉输注为首选；"
            "抗血小板方案依手术类型（CABG后阿司匹林恢复；机械瓣膜Warfarin+桥接）"
        ),
        "specialty_tags": ["cardiology", "endocrinology"],
        "era_sensitivity": False,
        "guideline_queries": [
            "cardiac surgery perioperative diabetes insulin glucose management",
            "CABG valve replacement diabetes oral agent hold",
        ],
        "contraindication_hints": [
            "oral_agents_perioperative_cardiac",
        ],
        "modifier": None,
        "exclude_if": set(),
    },
]


def match_safety_specs(
    patient_data: Dict,
    flat: Dict,
    existing_state_names: Set[str],
) -> List[Dict[str, Any]]:
    """
    对 CRITICAL_SAFETY_STATE_SPECS 表做 ICD 匹配，返回命中的 spec 列表（已应用 modifier）。

    Args:
        patient_data:          原始患者数据（含 all_diagnoses）
        flat:                  扁平化临床字段（来自 flatten_clinical_metrics）
        existing_state_names:  已存在的 state name 集合（用于 exclude_if）

    Returns:
        命中 spec 的列表，每个 spec 已通过 modifier 更新字段
    """
    matched: List[Dict[str, Any]] = []
    all_diags = patient_data.get("all_diagnoses", []) or []

    # 预提取所有 ICD 码（大写，去首尾空格）
    raw_codes: List[str] = []
    for d in all_diags:
        if not isinstance(d, dict): continue
        c = str(d.get("icd_code", "") or "").strip().upper()
        if c:
            raw_codes.append(c)

    for spec in CRITICAL_SAFETY_STATE_SPECS:
        name = spec["name"]

        # exclude_if 检查
        if spec.get("exclude_if", set()) & existing_state_names:
            continue

        prefixes: Set[str]    = spec.get("icd_prefixes", set())
        substrings: Set[str]  = spec.get("icd_substrings", set())

        hit = False
        for code in raw_codes:
            if prefixes and code[:3] in prefixes:
                hit = True; break
            if substrings and any(code.startswith(s) for s in substrings):
                hit = True; break

        if not hit:
            continue

        # 应用 modifier（可覆盖 confidence / implication 等字段）
        effective_spec = dict(spec)
        modifier = spec.get("modifier")
        if callable(modifier):
            overrides = modifier(flat, patient_data)
            effective_spec.update(overrides)

        matched.append(effective_spec)

    return matched
