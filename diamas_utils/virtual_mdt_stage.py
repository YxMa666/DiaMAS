#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
虚拟多学科会诊（Virtual MDT）- Phase B

核心思路：
  三个虚拟专家（内分泌科 / 肾科 / 心内科）对同一患者独立给出推荐，
  通过冲突检测器识别分歧，再由裁决 LLM 综合给出最终方案，
  并生成体现多科博弈过程的推理链。

Phase 1（当前实现）：
  - 串行调用三个专家（同一 LLM，不同系统提示）
  - 规则检测器识别冲突（关键词匹配）
  - 单次裁决 LLM 整合

Phase 2 升级接口（已预留，标注 TODO）：
  - 多专家并行调用
  - 迭代辩论（专家看到彼此意见后再次发言）
  - 专家权重动态调整（基于患者主诉域）
  - 更多专家角色（临床药师 / 老年科 / 营养科）

与 diamas_inference_stage.py 的关系：
  diamas_inference_stage 输出 Phase A 推荐结果（含 diamas_metadata）
  virtual_mdt_stage 接收 Phase A 结果 + 原始患者数据 → 输出 MDT 增强结果
  两者可以串联，也可以独立运行

作者：Yuxi
"""

import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Config imports (single source of truth for drug taxonomy and era mapping) ──
from agents.config import DrugEraConfig, DIABETES_MEDICATIONS, COMPLICATION_DRUGS
JUDGE_MODEL_DEFAULT = "deepseek-r1-250528"
# MDT 迭代辩论最大轮次 —— 唯一定义处（§63-P9）。
# 必须 ≥2：conduct_mdt 的触发条件是 max_debate_rounds >= 2，
# 默认 1 会让辩论永不执行（历史：与 judge_model 同类断链）。
MAX_DEBATE_ROUNDS_DEFAULT = 2

# ═══════════════════════════════════════════════════════════
# 数据结构
# ═══════════════════════════════════════════════════════════

@dataclass
class ExpertOpinion:
    """单个专家的推荐意见"""
    role:               str           # "内分泌科" / "肾科" / "心内科"
    recommended_drugs:  List[str]     # DM 降糖药名称（用于冲突检测，向后兼容）
    complication_drugs: List[Dict]    # [v5] 专科并发症药物（他汀/ACEi/β-blocker/MRA 等）
                                      # 格式: {drug, category, indication, dose}
    reasoning:          str           # 专科推理过程
    key_concerns:       List[str]     # 本科视角的核心顾虑（用于冲突检测）
    raw_output:         str           # LLM 原始输出
    inference_time:     float = 0.0


@dataclass
class ConflictItem:
    """检测到的冲突条目"""
    drug:              str           # 冲突药物
    conflict_type:     str           # "contraindication" / "dosing" / "priority"
    expert_a:          str           # 支持方专家
    expert_b:          str           # 反对方专家
    reason_a:          str           # 支持理由
    reason_b:          str           # 反对理由
    severity:          str           # "critical" / "major" / "minor"


@dataclass
class MDTResult:
    """MDT 会诊完整结果"""
    patient_id:              str
    expert_opinions:         List[ExpertOpinion]
    conflicts:               List[ConflictItem]
    final_drugs:             List[Dict]    # DM 降糖药（与 RecommendationResult 格式兼容）
    final_complication_drugs: List[Dict]  # [v5] 并发症药物（心内科/肾科专科输出）
    final_reasoning:         str           # 完整的多科推理链
    consensus_level:         str           # "full" / "partial" / "disputed"
    total_time:              float = 0.0

    # Phase 2 新增
    debate_rounds:     int   = 1
    expert_weights:    Dict[str, float] = field(default_factory=dict)
    conflicts_pre_debate: List = field(default_factory=list)


# ═══════════════════════════════════════════════════════════
# 专家角色配置
# ═══════════════════════════════════════════════════════════

EXPERT_CONFIGS = {
    # Keys are kept as Chinese role identifiers for backward compat with all
    # role-based dispatch, logging, and conflict-detection code downstream.
    # All prompt CONTENT is in English to match the main pipeline.
    "内分泌科": {
        "role_display": "Endocrinology",
        "system_prompt": (
            "You are an attending endocrinologist specializing in inpatient diabetes management.\n"
            "Core focus areas:\n"
            "  - Glycemic targets: HbA1c, TIR, inpatient glucose management\n"
            "  - Correct DM subtype identification (T1DM vs T2DM vs LADA)\n"
            "  - Antidiabetic drug efficacy and selection\n"
            "  - Hypoglycemia risk stratification and mitigation\n"
            "  - Long-term metabolic outcome optimization"
        ),
        "task": (
            "Recommend the optimal DM glycemic regimen (populate recommended_drugs). "
            "For microvascular complications within your scope (neuropathy, retinopathy, "
            "gastroparesis), populate complication_drugs. "
            "For cardiovascular/renal complication drugs, note only key interaction concerns "
            "in key_concerns — do not recommend them."
        ),
        "weight": 1.0,
        "dm_drugs_only": True,
    },
    "肾科": {
        "role_display": "Nephrology",
        "system_prompt": (
            "You are an attending nephrologist specializing in CKD and drug safety "
            "in kidney disease.\n"
            "Core focus areas:\n"
            "  - Drug nephrotoxicity and eGFR-based contraindications (KDIGO staging)\n"
            "  - Renoprotective strategy: SGLT2i, RAAS blockade (ACEi/ARB), MRA\n"
            "  - AKI risk: contrast exposure, NSAIDs, Metformin hold criteria "
            "(hold if eGFR < 30, or < 45 peri-procedurally)\n"
            "  - Volume management: loop diuretics in CKD with fluid overload\n"
            "  - Electrolyte safety: hyperkalemia with RAASi, metabolic acidosis"
        ),
        "task": (
            "Two responsibilities:\n"
            "  1. DM drug safety evaluation: identify DM drugs requiring dose reduction or hold "
            "due to renal function — list in dm_drugs_safety_notes (dict) and key_concerns. "
            "Leave recommended_drugs as [].\n"
            "  2. Renoprotective complication drugs: recommend ACEi/ARB (CKD G1–G4 + proteinuria), "
            "MRA (Finerenone/Eplerenone if era-appropriate), and loop diuretics if fluid-overloaded, "
            "in complication_drugs."
        ),
        "weight": 1.2,
        "dm_drugs_only": False,
    },
    "心内科": {
        "role_display": "Cardiology",
        "system_prompt": (
            "You are an attending cardiologist specializing in cardiovascular risk management "
            "in diabetes.\n"
            "Core focus areas:\n"
            "  - ASCVD risk stratification and CV-protective drug selection\n"
            # [FIX-TEMPORAL-CARDIO] Removed post-2014 trial references (DAPA-HF 2019,
            # EMPEROR-Reduced 2020, EMPA-REG 2015, LEADER 2016, SUSTAIN-6 2016, CREDENCE 2019).
            # Era-specific evidence is injected via specialty_evidence at call time.
            # The system prompt must be era-neutral to avoid contaminating the LLM persona
            # with modern knowledge when evaluating historical Era-1/Era-2 patients.
            "  - HFrEF management: ACEi/ARB, beta-blocker, MRA, loop diuretics (standard "
            "guideline-based neurohormonal blockade); avoid TZD and Saxagliptin/Alogliptin\n"
            "  - Statin therapy for ASCVD prevention; antiplatelet agents per indication\n"
            "  - QT prolongation risk with antidiabetic agents\n"
            "  - Blood pressure and heart rate management interactions with DM drugs\n"
            # [v18 P4] 移除 DISCHARGE PLANNING 段：该指令只存在于 A2 侧，而 A1 的共享
            # prompt(prompt_builder.py R3/R7) 方向相反（phase 未知 → 按 acute_unstable 处理）。
            # arm_asymmetry(b) AUDIT-4 两队列均判 A1-ORAL-OK（口服类 A1−GT = −0.6/+2.7pp）
            # ⇒ 无证据支持把出院口径上移共享层（那会改 A1 的 prompt，作废 34,673 例）。
            # 最小改动原则：撤掉单边指令即恢复对称。
        ),
        "task": (
            "Two responsibilities:\n"
            "  1. DM drug CV safety assessment: flag CV-relevant concerns for SGLT2i/GLP-1RA/"
            "TZD/DPP-4i in dm_drugs_safety_notes and key_concerns. "
            "Leave recommended_drugs as [].\n"
            "  2. Cardiovascular complication drugs: recommend statins, beta-blockers, ACEi/ARB, "
            "and antiplatelet agents based on the patient's CV diagnoses, in complication_drugs."
        ),
        "weight": 1.1,
        "dm_drugs_only": False,
    },

    # ══ Phase 2 reserved expert roles ══
    # "clinical_pharmacist": { "weight": 0.9, ... },
    # "geriatrics":          { "weight": 1.0, ... },
}

# ── Specialty → COMPLICATION_DRUGS categories (the closed set each specialty owns) ──
# ACEi/ARB appear in both nephrology and cardiology intentionally:
#   nephrology → CKD+proteinuria indication; cardiology → HF/post-MI indication.
# Adjudicator handles any overlap.
_SPECIALTY_COMPLICATION_CATS: Dict[str, List[str]] = {
    "endocrinology": [
        "Neuropathy_Gabapentinoids", "Neuropathy_SNRI", "Neuropathy_TCA", "Neuropathy_Others",
        "Gastroparesis", "Hypoglycemia_Rescue",
        "Retinopathy_AntiVEGF", "Retinopathy_Oral",
        "Microcirculation_Vasodilators",
        # [v18c P3a] 他汀与抗血小板本属内分泌科自身职责（ADA Standards §10
        # Cardiovascular Disease and Risk Management），独占给 cardiology 是原设计错误。
        # 内分泌科恒激活 ⇒ 补入后，心内科未激活的患者不再结构性丢失这两类。
        # 实测依据：v18b 冒烟 AUDIT-3，Lipid_Statins 心内科缺席层 Δ=−46.5pp
        # [−62.8,−32.6]，在场层 +0.0pp [−6.0,+6.0]，两层分离干净。
        # 不加 Hypertension_BetaBlockers：两层均 +2.0/+0.0pp（未丢失），
        # 且其 HF/CAD 适应症确属心内科。
        "Lipid_Statins", "Antiplatelet",
    ],
    "nephrology": [
        "Hypertension_ACEi", "Hypertension_ARB",
        "CKD_MRA",
        "Hypertension_Diuretics",
        "Hypertension_CCB",
    ],
    "cardiology": [
        "Lipid_Statins", "Lipid_NonStatins",
        "Hypertension_ACEi", "Hypertension_ARB",
        "Hypertension_BetaBlockers",
        "Antiplatelet",
        "Hypertension_Diuretics",
    ],
}


# ═══════════════════════════════════════════════════════════
# 冲突检测器
# ═══════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════
# 冲突检测器（Phase 1 规则版 → Phase 2 LLM Judge 版）
# ═══════════════════════════════════════════════════════════

class ConflictDetector:
    """
    Phase 1 冲突检测器（规则 + 关键词匹配）

    保留作为：
      1. judge_model=None 时的降级兜底
      2. LLM Judge 的预筛选器（过滤明显假阳性）
    """

    CONTRAINDICATION_SIGNALS = {
        "肾科":   ["contraindicated", "avoid", "hold", "do not use",
                   "not recommended", "withhold", "discontinue", "stop"],
        "心内科": ["contraindicated", "avoid", "not recommended",
                   "heart failure contraindication", "do not use"],
        "内分泌科": ["contraindicated", "avoid", "not recommended",
                    "do not use", "withhold", "discontinue", "should stop"],
    }

    CRITICAL_DRUG_SIGNALS = {
        "metformin":           ["egfr", "renal", "lactic acidosis", "kidney"],
        "sglt2":               ["dka", "euglycemic", "volume", "aki", "contrast", "surgery",
                                "hemodynamic"],
        "sulfonylurea":        ["hypoglycemia", "renal", "kidney"],
        "thiazolidinedione":   ["heart failure", "fluid retention", "edema"],
        "pioglitazone":        ["heart failure", "bladder"],
        "saxagliptin":         ["heart failure"],
        "alogliptin":          ["heart failure"],
        "insulin":             ["hypoglycemia"],
    }

    def _assess_severity(self, drug: str, context: str) -> str:
        drug_lower = drug.lower()
        for drug_key, safety_signals in self.CRITICAL_DRUG_SIGNALS.items():
            if drug_key in drug_lower or drug_lower in drug_key:
                if any(sig in context for sig in safety_signals):
                    return "critical"
                return "major"
        return "major"

    def detect(self, opinions: Dict[str, "ExpertOpinion"]) -> List["ConflictItem"]:
        conflicts = []
        roles = list(opinions.keys())

        recommend_sets = {
            role: set(d.lower() for d in op.recommended_drugs)
            for role, op in opinions.items()
        }
        concern_texts = {
            role: " ".join(op.key_concerns + [op.reasoning]).lower()
            for role, op in opinions.items()
        }

        for i, role_a in enumerate(roles):
            for role_b in roles[i + 1:]:
                for drug in recommend_sets[role_a]:
                    concern_b = concern_texts[role_b]
                    if drug in concern_b:
                        for signal in self.CONTRAINDICATION_SIGNALS.get(role_b, []):
                            idx = concern_b.find(drug)
                            if idx >= 0:
                                context = concern_b[max(0, idx - 50): idx + 100]
                                if signal in context:
                                    conflicts.append(ConflictItem(
                                        drug          = drug,
                                        conflict_type = "contraindication",
                                        expert_a      = role_a,
                                        expert_b      = role_b,
                                        reason_a      = f"{role_a} 推荐: {drug}",
                                        reason_b      = f"{role_b} 关注: ...{context}...",
                                        severity      = self._assess_severity(drug, context),
                                    ))
                                    break

                for drug in recommend_sets[role_b]:
                    concern_a = concern_texts[role_a]
                    if drug in concern_a:
                        for signal in self.CONTRAINDICATION_SIGNALS.get(role_a, []):
                            idx = concern_a.find(drug)
                            if idx >= 0:
                                context = concern_a[max(0, idx - 50): idx + 100]
                                if signal in context:
                                    conflicts.append(ConflictItem(
                                        drug          = drug,
                                        conflict_type = "contraindication",
                                        expert_a      = role_b,
                                        expert_b      = role_a,
                                        reason_a      = f"{role_b} 推荐: {drug}",
                                        reason_b      = f"{role_a} 关注: ...{context}...",
                                        severity      = self._assess_severity(drug, context),
                                    ))
                                    break

        seen = set()
        unique = []
        for c in conflicts:
            key = (c.drug, frozenset([c.expert_a, c.expert_b]), c.conflict_type)
            if key not in seen:
                seen.add(key)
                unique.append(c)

        unique.sort(key=lambda x: {"critical": 0, "major": 1, "minor": 2}[x.severity])
        return unique


class LLMConflictJudge:
    """
    Phase 2 LLM Judge 冲突检测器。

    使用与推荐 LLM 不同的模型家族进行语义冲突分析，
    避免自评偏差（self-scoring bias）。

    设计：
    - 默认模型：qwen-plus（与 DeepSeek 推荐模型不同家族）
    - judge_model=None → 自动降级到 ConflictDetector（向后兼容）
    - 输出 ConflictItem 列表（与 ConflictDetector 接口完全相同）
    - temperature=0.0（确定性判断）
    - 独立 LLM 调用，在专家意见收集完毕后、裁决前运行

    局限性（用于论文 Limitations 章节）：
    - 中文专家意见 → 英文 Judge 模型可能存在理解偏差，建议一致语言
    - 单轮 Judge 调用，不做迭代验证
    - 药物别名处理依赖 LLM 内置知识（如 empagliflozin = Jardiance）
    """

    # Judge prompt severity classification guide
    _SEVERITY_GUIDE = """
Severity classification:
- critical: Absolute safety contraindication (e.g., Metformin with eGFR < 30, SGLT2i during AKI/shock, TZD with active HF)
- major:    Efficacy conflict or relative contraindication (e.g., priority disagreement, dose adjustment required)
- minor:    Secondary concern (monitoring frequency difference, patient preference)
"""

    def __init__(
        self,
        llm_client,
        judge_model:       str   = JUDGE_MODEL_DEFAULT,
        judge_temperature: float = 0.0,
        max_tokens:        int   = 2048,
        fallback_detector: Optional[ConflictDetector] = None,
        inference_model: str = "",
    ):
        """
        Args:
            llm_client:        DeepseekClient 实例（同一 API 接口，不同模型）
            judge_model:       Judge 使用的模型名；None → 降级到规则检测器
            judge_temperature: 确定性输出（推荐 0.0）
            max_tokens:        Judge 输出最大 token 数
            fallback_detector: judge_model=None 或 LLM 调用失败时的降级检测器
        """
        self.llm          = llm_client
        self.judge_model  = judge_model
        self.temperature  = judge_temperature
        self.max_tokens   = max_tokens
        self.fallback     = fallback_detector or ConflictDetector()
        # if inference_model and judge_model == inference_model:
        #     raise ValueError(
        #         f"judge_model 与推理模型相同({judge_model}) —— MDT 退化为自评。"
        #         f"请显式指定不同模型。")
        # # ② 启动探针:一次 1-token 调用确认该模型名端点真的提供
        # #    根因防复发:qwen-plus 恒 404 之所以活了 6,951 例,
        # #    就是因为没有任何启动期校验。
        probe = self.llm.chat(
            messages=[{"role": "user", "content": "ping"}],
            model=judge_model, temperature=0.0, max_tokens=1,
        )
        if probe.get("error"):
            raise RuntimeError(
                f"judge_model={judge_model} 不可用:{probe.get('message')}。"
                f"启动即失败,避免整批跑完才发现零冲突。")
        logger.info("[MDT] Judge 模型自检通过: %s", judge_model)

    def detect(
        self,
        opinions:        Dict[str, "ExpertOpinion"],
        patient_summary: str = "",
    ) -> List["ConflictItem"]:
        """
        语义冲突检测。

        如果 judge_model 为 None 或 LLM 调用失败，自动降级到规则检测器。

        Args:
            opinions:        {role: ExpertOpinion} 字典
            patient_summary: 患者摘要（提供给 Judge 的临床背景）

        Returns:
            ConflictItem 列表，按 severity 排序
        """
        # 不做降级:静默 fallback 会让结果集混合两种冲突检测机制,
        # 事后无法区分哪个患者走了哪条路 —— Methods 里写不出来。
        # ConflictDetector 保留在代码中,仅供显式消融实验使用,不接生产路径。
        return self._llm_detect(opinions, patient_summary)

    def _build_judge_prompt(
        self,
        opinions:        Dict[str, "ExpertOpinion"],
        patient_summary: str,
    ) -> str:
        """Build Judge prompt."""
        opinions_text = "\n\n".join([
            f"[{role} Opinion]\n"
            f"Recommended DM drugs: {', '.join(op.recommended_drugs) or '(none)'}\n"
            f"Reasoning summary: {op.reasoning[:300]}\n"
            f"Key concerns: {'; '.join(op.key_concerns[:4]) or '(none)'}"
            for role, op in opinions.items()
        ])

        return f"""You are an independent clinical safety advisor reviewing drug conflicts in a Virtual MDT for a hospitalized diabetes patient.

[Patient Background]
{patient_summary or "(not provided)"}

[Expert Opinions by Specialty]
{opinions_text}

{self._SEVERITY_GUIDE}

[Task]
Analyze the above specialty opinions and identify all genuine drug conflicts
(one specialty recommends a drug while another opposes it or raises a safety concern).
Report only conflicts with clear clinical significance — do not flag surface-level
differences in phrasing or emphasis.

Output strictly as a JSON array (empty array [] if no conflicts; no extra text):
[
  {{
    "drug": "standard INN name (e.g., metformin)",
    "conflict_type": "contraindication|dosing|priority",
    "expert_a": "recommending specialty (e.g., 内分泌科)",
    "expert_b": "opposing specialty (e.g., 肾科)",
    "reason_a": "recommendation rationale (one sentence)",
    "reason_b": "opposition/safety concern (with specific clinical basis)",
    "severity": "critical|major|minor"
  }}
]"""

    def _llm_detect(
        self,
        opinions:        Dict[str, "ExpertOpinion"],
        patient_summary: str,
    ) -> List["ConflictItem"]:
        """调用 Judge LLM 并解析结果。"""
        prompt = self._build_judge_prompt(opinions, patient_summary)

        response = self.llm.chat(
            messages      = [{"role": "user", "content": prompt}],
            model         = self.judge_model,
            temperature   = self.temperature,
            max_tokens    = self.max_tokens,
            deep_thinking = False,  # Judge 要求确定性，不开 thinking
        )
        raw_text = self.llm.get_response_text(response)

        # 解析 JSON 数组（兼容 <think> 块和 markdown 代码块）
        text_clean = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()

        # 提取 JSON 数组
        json_str = None
        for pat in [r"```json\s*(.*?)\s*```", r"```\s*(.*?)\s*```"]:
            m = re.search(pat, text_clean, re.DOTALL)
            if m:
                json_str = m.group(1).strip()
                break
        if not json_str:
            # 直接找 [ ... ] 数组
            m = re.search(r"\[.*\]", text_clean, re.DOTALL)
            if m:
                json_str = m.group(0)

        if not json_str:
            logger.warning("LLM Judge 返回的文本中未找到 JSON 数组，返回空冲突列表")
            return []

        items = []
        try:
            items = json.loads(json_str)
        except json.JSONDecodeError:
            try:
                import json_repair
                items = json_repair.repair_json(json_str, return_objects=True)
            except Exception:
                logger.warning("LLM Judge JSON 解析失败，返回空冲突列表")
                return []

        if not isinstance(items, list):
            logger.warning(f"LLM Judge 返回非列表结构：{type(items)}，返回空冲突列表")
            return []

        # 转换为 ConflictItem 列表
        valid_severities = {"critical", "major", "minor"}
        valid_types      = {"contraindication", "dosing", "priority"}
        known_roles      = set(opinions.keys())

        conflicts: List[ConflictItem] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            drug = str(item.get("drug", "")).strip()
            if not drug:
                continue

            severity     = item.get("severity", "major")
            conflict_type = item.get("conflict_type", "contraindication")
            expert_a     = item.get("expert_a", "")
            expert_b     = item.get("expert_b", "")

            # 规范化：severity / type 不在枚举内时降级为 major / contraindication
            if severity not in valid_severities:
                severity = "major"
            if conflict_type not in valid_types:
                conflict_type = "contraindication"

            # 专家角色校验：允许 LLM 返回已知角色名的子集或中文
            # 若无法识别则用 "unknown" 但保留条目（信息仍有价值）
            if expert_a not in known_roles:
                expert_a = next((r for r in known_roles if r in expert_a), expert_a)
            if expert_b not in known_roles:
                expert_b = next((r for r in known_roles if r in expert_b), expert_b)

            conflicts.append(ConflictItem(
                drug          = drug.lower(),
                conflict_type = conflict_type,
                expert_a      = expert_a,
                expert_b      = expert_b,
                reason_a      = str(item.get("reason_a", f"{expert_a} 推荐: {drug}"))[:300],
                reason_b      = str(item.get("reason_b", f"{expert_b} 顾虑: {drug}"))[:300],
                severity      = severity,
            ))

        # 去重（同药物 + 同专家对 + 同冲突类型）
        seen = set()
        unique = []
        for c in conflicts:
            key = (c.drug, frozenset([c.expert_a, c.expert_b]), c.conflict_type)
            if key not in seen:
                seen.add(key)
                unique.append(c)

        unique.sort(key=lambda x: {"critical": 0, "major": 1, "minor": 2}[x.severity])
        logger.info(
            f"  LLM Judge 检测到 {len(unique)} 条冲突 "
            f"(critical: {sum(1 for c in unique if c.severity == 'critical')})"
        )
        return unique


# ═══════════════════════════════════════════════════════════
# 虚拟 MDT 主体
# ═══════════════════════════════════════════════════════════

class VirtualMDT:
    """
    虚拟多学科会诊系统

    Phase 1 工作流：
    ┌─ 1. 各专家独立阅读患者资料 → 给出推荐 + 核心顾虑
    ├─ 2. 冲突检测器分析分歧
    ├─ 3. 裁决 LLM 综合三方意见 → 最终方案
    └─ 4. 输出 MDTResult（含完整推理链）

    Phase 2 升级接口：
    - conduct_mdt() 中的串行专家调用 → 并行（ThreadPoolExecutor）
    - 裁决轮次：固定 1 次 → 可迭代（迭代辩论）
    - 专家选择：固定三科 → 根据患者主诉域动态选择（EXPERT_CONFIGS 可配置）
    """

    def __init__(
        self,
        llm_client,
        model_name:         str   = "deepseek-r1-250528",
        temperature:        float = 0.3,
        max_tokens:         int   = 4096,
        active_experts:     Optional[List[str]] = None,
        registry:           Optional[object]    = None,
        judge_model:        Optional[str]       = JUDGE_MODEL_DEFAULT,
        judge_temperature:  float               = 0.0,
        max_debate_rounds:  int                 = MAX_DEBATE_ROUNDS_DEFAULT,
        use_dynamic_weights: bool               = True,
    ):
        """
        Args:
            llm_client:          DeepseekClient 实例
            model_name:          推荐使用的模型（建议推理能力较强的）
            temperature:         专家发言 temperature（稍高于0以产生多样视角）
            max_tokens:          最大 token 数
            active_experts:      参与会诊的专家列表（默认全部三科）
            registry:            ClinicalGuidelineRegistry 实例
            judge_model:         冲突检测使用的 Judge 模型名；
                                 须与 model_name 属于不同模型家族以避免自评偏差。
                                 None → 降级到规则检测器（向后兼容）
            judge_temperature:   Judge 模型 temperature（推荐 0.0 保证确定性）
            max_debate_rounds:   MDT 迭代辩论最大轮次（Phase 2）
                                 1  = Phase 1 行为（单轮，无辩论，默认）
                                 2+ = Phase 2：冲突数 > 2 时触发追加辩论轮次
            use_dynamic_weights: 是否根据患者表型动态调整专科权重（Phase 2）
                                 True  = 按 specialty_tags 提升主诊专科权重（默认）
                                 False = 等权（Phase 1 行为）
        """
        self.llm               = llm_client
        self.model_name        = model_name
        self.temperature       = temperature
        self.max_tokens        = max_tokens
        self.active_experts    = active_experts or list(EXPERT_CONFIGS.keys())
        self.registry          = registry
        self.max_debate_rounds = max_debate_rounds
        self.use_dynamic_weights = use_dynamic_weights

        # 冲突检测器：优先使用 LLM Judge，judge_model=None 时自动降级到规则版
        self.detector = LLMConflictJudge(
            llm_client        = llm_client,
            judge_model       = judge_model,
            judge_temperature = judge_temperature,
            max_tokens        = 2048,
            fallback_detector = ConflictDetector(),
            inference_model=model_name, 
        )

    # ── 专科 → SPECIALTY_SOURCE_ROUTING key 的映射 ──────────────
    _ROLE_TO_SPECIALTY = {
        "内分泌科": "endocrinology",
        "肾科":     "nephrology",
        "心内科":   "cardiology",
    }
    # ── 表型关键词 → 专科角色的权重加成映射 ──────────────────────
    _PHENOTYPE_TO_ROLE_BOOST: Dict[str, str] = {
        "nephrology":    "肾科",
        "cardiology":    "心内科",
        "endocrinology": "内分泌科",
        "heart_failure": "心内科",
        "ckd":           "肾科",
        "aki":           "肾科",
        "ascvd":         "心内科",
        "cardiovascular":"心内科",
        "hyperkalemia":  "肾科",
    }

    def _compute_expert_weights(
        self,
        specialty_tags: List[str],
    ) -> Dict[str, float]:
        """
        根据患者表型的专科关联度分配裁决权重（Phase 2 动态权重）。

        算法：
          1. 所有专科基础权重 = 1.0
          2. specialty_tags 中每命中一个 → 对应专科权重 += 0.5（上限 2.0）
          3. 归一化到和为 1.0

        示例：
          specialty_tags = ["nephrology", "ckd"]
          → 肾科 weight = (1.0 + 0.5 + 0.5) / (3.0 + 1.0) ≈ 0.50
          → 内分泌科 ≈ 0.25，心内科 ≈ 0.25

        Args:
            specialty_tags: 来自 ClinicalPhenotypeState.specialty_tags

        Returns:
            {角色名: 归一化权重}，所有激活专家都有权重
        """
        if not self.use_dynamic_weights or not specialty_tags:
            # 等权（Phase 1 行为）
            n = len(self.active_experts)
            return {role: 1.0 / n for role in self.active_experts}

        weights: Dict[str, float] = {role: 1.0 for role in self.active_experts}
        for tag in specialty_tags:
            role = self._PHENOTYPE_TO_ROLE_BOOST.get(tag.lower())
            if role and role in weights:
                weights[role] = min(weights[role] + 0.5, 2.0)

        total = sum(weights.values())
        if total <= 0:
            n = len(self.active_experts)
            return {role: 1.0 / n for role in self.active_experts}
        return {role: w / total for role, w in weights.items()}

    def conduct_mdt(
        self,
        phase_a_result:  Dict,
        flat:            Optional[Dict] = None,
        patient_data:    Optional[Dict] = None,
        active_experts:  Optional[List[str]] = None,
    ) -> MDTResult:
        """
        Execute Virtual MDT consultation.

        Args:
            phase_a_result:  Output from diamas_inference_stage (contains patient_summary
                             and diamas_metadata with patho_states, era_label, ada_year_used).
            flat:            Pre-computed standardized clinical features from
                             flatten_clinical_metrics(). If None and patient_data is
                             provided, computed inline. Enables specialty-specific
                             clinical panels (information routing).
            patient_data:    Raw patient JSON. Used to access clinical_metrics sub-dicts
                             (lipids, full electrolytes, dm_complications) not yet in flat.
                             Required for complete specialty panels; gracefully degraded if absent.
            active_experts:  Expert roles for this consultation (overrides __init__ setting).
        """
        start_time = time.time()
        patient_id = phase_a_result.get("patient_id", "unknown")

        experts_this_call = active_experts if active_experts is not None else self.active_experts
        experts_this_call = [r for r in experts_this_call if r in EXPERT_CONFIGS]
        if not experts_this_call:
            logger.warning("active_experts empty after filtering — falling back to all experts")
            experts_this_call = list(EXPERT_CONFIGS.keys())

        # patient_summary always sourced from phase_a_result (single source of truth)
        patient_summary = phase_a_result.get("patient_summary", "")

        # Compute flat inline if not pre-provided (batch runner path)
        flat_local = flat
        if flat_local is None and patient_data is not None:
            try:
                from diamas_utils.era_retriever import flatten_clinical_metrics
                flat_local = flatten_clinical_metrics(patient_data)
            except Exception as _fe:
                logger.warning(f"  flatten_clinical_metrics failed inline: {_fe}")
                flat_local = {}

        patho_text   = self._format_patho_states(phase_a_result)
        diamas_meta  = phase_a_result.get("diamas_metadata", {})
        if isinstance(diamas_meta, str):
            try:
                diamas_meta = json.loads(diamas_meta)
            except Exception:
                diamas_meta = {}
        if not isinstance(diamas_meta, dict):
            diamas_meta = {}
        ada_evidence = diamas_meta.get("ada_evidence", "")

        era_label = diamas_meta.get("era_label", "Unknown")
        ada_year  = diamas_meta.get("ada_year_used", 2020)

        # Derive era_num from config (SSOT) — no hardcoded map
        _era_num_map = {
            year: int(label.split("-")[1])
            for label, year in DrugEraConfig.ERA_TO_RAG_YEAR.items()
            if label != "Unknown"
        }
        _era_num = _era_num_map.get(
            ada_year,
            3 if ada_year >= 2020 else (2 if ada_year >= 2014 else 1)
        )

        patho_states = diamas_meta.get("phenotype_states") or diamas_meta.get("patho_states", [])
        _all_queries: List[str] = []
        for ps in patho_states:
            _all_queries.extend(ps.get("guideline_queries", []))

        logger.info(f"  MDT: patient={patient_id}, specialties={experts_this_call}, era={era_label}, "
                    f"flat={'yes' if flat_local else 'no'}, patient_data={'yes' if patient_data else 'no'}")

        expert_opinions: Dict[str, ExpertOpinion] = {}

        # Fix R2-M2: three expert calls are fully independent — run in parallel
        from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed

        # Derive era_num from config (already done above, _era_num available)
        def _get_specialty_evidence(role: str) -> str:
            """Retrieve specialty-aligned guideline evidence; fall back to global ada_evidence."""
            if self.registry is None or not _all_queries:
                return ada_evidence
            specialty = self._ROLE_TO_SPECIALTY.get(role, "endocrinology")
            try:
                return self.registry.format_evidence_for_specialty(
                    specialty = specialty,
                    queries   = _all_queries[:5],
                    era_year  = ada_year if era_label != "Unknown" else None,
                    era_num   = _era_num if era_label != "Unknown" else None,
                    top_k     = 5,
                )
            except Exception as _e:
                logger.warning(f"  [{role}] specialty evidence retrieval failed, using global: {_e}")
                return ada_evidence

        def _expert_task(role: str):
            """Single expert full task: evidence retrieval + specialty panel + LLM call."""
            evidence       = _get_specialty_evidence(role)
            specialty_panel = self._build_specialty_panel(role, flat_local, patient_data, ada_year)
            drug_scope      = self._build_specialty_drug_scope(role, ada_year)
            return role, self._get_expert_opinion(
                role               = role,
                patient_summary    = patient_summary,
                patho_text         = patho_text,
                specialty_evidence = evidence,
                specialty_panel    = specialty_panel,
                drug_scope         = drug_scope,
                era_label          = era_label,
                ada_year           = ada_year,
            )

        with ThreadPoolExecutor(max_workers=len(experts_this_call)) as _pool:
            _futs = {_pool.submit(_expert_task, role): role for role in experts_this_call}
            for _fut in _as_completed(_futs):
                _role = _futs[_fut]
                try:
                    _, opinion = _fut.result()
                    expert_opinions[_role] = opinion
                    logger.debug(f"  {_role}: 推荐 {opinion.recommended_drugs}")
                except Exception as _e:
                    logger.warning(f"  {_role} 专家调用失败，使用空意见: {_e}")
                    expert_opinions[_role] = ExpertOpinion(
                        role=_role, recommended_drugs=[], complication_drugs=[],
                        reasoning="", key_concerns=[], raw_output="", inference_time=0.0
                    )

        # ── 提取 specialty_tags（动态权重依据）────────────────────
        all_specialty_tags: List[str] = []
        for ps in patho_states:
            all_specialty_tags.extend(ps.get("specialty_tags", []))
        expert_weights = self._compute_expert_weights(all_specialty_tags)
        logger.info(f"  专科权重: { {r: round(w, 2) for r, w in expert_weights.items()} }")

        # ── Step 2: 冲突检测 ─────────────────────────────────────
        # LLMConflictJudge 接受 patient_summary 作为临床背景上下文；
        # 规则版 ConflictDetector 忽略此参数（向后兼容）
        conflicts = self.detector.detect(expert_opinions, patient_summary=patient_summary)
        conflicts_pre_debate = list(conflicts)
        logger.info(f"  MDT 冲突检测: {len(conflicts)} 条冲突 "
                    f"(critical: {sum(1 for c in conflicts if c.severity == 'critical')})")

        # ── Step 2.5: 迭代辩论（Phase 2，冲突数 >= 1 时激活）─────────────────
        # 触发条件：max_debate_rounds >= 2 且当前有 >= 1 条冲突
        actual_debate_rounds = 1
        if self.max_debate_rounds >= 2 and len(conflicts) >= 1:
            logger.info(
                f"  [Phase 2] 检出冲突数={len(conflicts)}，触发迭代辩论 "
                f"(max_rounds={self.max_debate_rounds})"
            )
            for debate_round in range(2, self.max_debate_rounds + 1):
                actual_debate_rounds = debate_round
                logger.info(f"  [辩论 Round {debate_round}] 各专家审阅对方顾虑...")

                # 并行获取修订意见
                def _rebuttal_task(role: str):
                    other_concerns = {
                        r: op.key_concerns
                        for r, op in expert_opinions.items() if r != role
                    }
                    return role, self._get_expert_rebuttal(
                        role            = role,
                        original_opinion= expert_opinions[role],
                        other_concerns  = other_concerns,
                        conflicts       = conflicts,
                        patient_summary = patient_summary,
                        era_label       = era_label,
                        ada_year        = ada_year,
                    )

                with ThreadPoolExecutor(max_workers=len(experts_this_call)) as _pool:
                    _futs = {_pool.submit(_rebuttal_task, role): role
                             for role in experts_this_call}
                    for _fut in _as_completed(_futs):
                        _role = _futs[_fut]
                        try:
                            _, revised = _fut.result()
                            expert_opinions[_role] = revised
                        except Exception as _e:
                            logger.warning(f"  [{_role}] 辩论修订失败，保留原意见: {_e}")

                # 重新检测冲突
                new_conflicts = self.detector.detect(expert_opinions,
                                                     patient_summary=patient_summary)
                resolved = len(conflicts) - len(new_conflicts)
                logger.info(
                    f"  [辩论 Round {debate_round}] 冲突: {len(conflicts)} → {len(new_conflicts)} "
                    f"（解决 {resolved} 条）"
                )
                conflicts = new_conflicts
                if len(conflicts) <= 1:
                    logger.info("  [辩论] 冲突已基本解决，提前终止辩论")
                    break

        # ── Step 3: 裁决整合 ─────────────────────────────────────
        final_dm_drugs, final_comp_drugs, final_reasoning, consensus = self._adjudicate(
            expert_opinions = expert_opinions,
            conflicts       = conflicts,
            patient_summary = patient_summary,
            ada_evidence    = ada_evidence,
            expert_weights  = expert_weights,
            era_label       = era_label,
            ada_year        = ada_year,
        )

        total_time = time.time() - start_time

        return MDTResult(
            patient_id               = patient_id,
            expert_opinions          = list(expert_opinions.values()),
            conflicts                = conflicts,
            final_drugs              = final_dm_drugs,
            final_complication_drugs = final_comp_drugs,
            final_reasoning          = final_reasoning,
            consensus_level          = consensus,
            total_time               = time.time() - start_time,
            debate_rounds            = actual_debate_rounds,
            expert_weights           = expert_weights,
            conflicts_pre_debate = conflicts_pre_debate,
        )

    @staticmethod
    def _build_specialty_panel(
        role:         str,
        flat:         Optional[Dict],
        patient_data: Optional[Dict],
        ada_year:     int,
    ) -> str:
        """
        Build a specialty-specific clinical data panel to append after patient_summary.

        Design rationale (paper motivation):
          Real MDTs route specialty-relevant data subsets to each expert — nephrologists
          review KDIGO labs, cardiologists review cardiac biomarkers and lipids.
          Providing each agent with a domain-complete panel rather than a generic text
          summary enables reasoning from domain-precise data, which single-agent systems
          cannot replicate by prompt-engineering alone.

        Args:
            role:         Chinese role key ("内分泌科" / "肾科" / "心内科")
            flat:         Standardized dict from flatten_clinical_metrics() — primary source.
            patient_data: Raw patient JSON — provides clinical_metrics sub-dicts (lipids,
                          full electrolytes, dm_complications) not yet in flat.
            ada_year:     For era-contextual clinical notes.
        """
        f  = flat or {}
        cm: Dict = {}
        if patient_data:
            cm = patient_data.get("clinical_metrics", {}) or {}

        def _v(val, unit: str = "", prec: int = 1) -> str:
            """Format numeric value with unit, or 'N/A'."""
            if val is None:
                return "N/A"
            try:
                fval = float(val)
                return f"{round(fval, prec)} {unit}".strip()
            except (ValueError, TypeError):
                return f"{val} {unit}".strip()

        def _b(val) -> str:
            if val is True:  return "Yes"
            if val is False: return "No"
            return "N/A"

        specialty = {
            "内分泌科": "endocrinology",
            "肾科":     "nephrology",
            "心内科":   "cardiology",
        }.get(role, "endocrinology")

        cfg_display = EXPERT_CONFIGS.get(role, {}).get("role_display", role)
        lines = [f"[Specialty-Relevant Clinical Panel — {cfg_display}]"]

        # ── Nephrology panel ──────────────────────────────────────────────────
        if specialty == "nephrology":
            kidney = cm.get("kidney", {}) or {}
            elec   = cm.get("electrolytes", {}) or {}
            comp   = cm.get("dm_complications", {}) or {}

            egfr       = f.get("egfr")
            creatinine = f.get("creatinine")
            uacr       = f.get("urine_albumin_creatinine_ratio")
            ckd_full   = kidney.get("ckd_full_stage") or f.get("ckd_stage")
            ckd_a      = kidney.get("ckd_a_stage")
            aki_stage  = f.get("aki_kdigo_stage")
            k          = f.get("potassium")

            egfr_flag = ""
            if egfr is not None:
                if egfr < 15:   egfr_flag = " ⚠ KDIGO G5 (Kidney Failure/ESRD)"
                elif egfr < 30: egfr_flag = " ⚠ KDIGO G4 — Severely Decreased (Metformin ABS.CONTRAINDICATED; SGLT2i generally avoid)"
                elif egfr < 45: egfr_flag = " ⚠ KDIGO G3b — Moderately-Severely Decreased (Metformin: hold or ≤1000mg/d; SGLT2i consider stopping)"
                elif egfr < 60: egfr_flag = " KDIGO G3a — Moderately Decreased"

            lines += [
                f"  eGFR:              {_v(egfr, 'mL/min/1.73m²')}{egfr_flag}",
                f"  Serum Creatinine:  {_v(creatinine, 'mg/dL')}",
                f"  CKD Stage (G+A):   {ckd_full or 'N/A'}"
                + (f"  (Albuminuria: {ckd_a})" if ckd_a else ""),
                f"  Urine ACR:         {_v(uacr, 'mg/g')}"
                + (" ⚠ Severely Increased (A3) — ACEi/ARB strongly indicated"
                   if uacr is not None and uacr >= 300 else
                   " Moderately Increased (A2)" if uacr is not None and uacr >= 30 else ""),
                f"  BUN:               {_v(kidney.get('bun'), 'mg/dL')}",
                f"  Serum Albumin:     {_v(kidney.get('albumin'), 'g/dL')}",
                f"  AKI Suspected:     {_b(f.get('aki_suspected'))}"
                + (f"  KDIGO AKI Stage: {aki_stage}" if aki_stage else ""),
                f"  Peak Creatinine:   {_v(f.get('creatinine_peak'), 'mg/dL')}",
                f"  Potassium (K⁺):    {_v(k, 'mEq/L')}"
                + (" ⚠ HYPERKALEMIA — RAASi use with caution / dose-reduce"
                   if f.get("potassium_suspect") else ""),
                f"  Sodium (Na⁺):      {_v(elec.get('sodium'), 'mEq/L')}",
                f"  Bicarbonate (HCO₃⁻): {_v(elec.get('bicarbonate'), 'mEq/L')}"
                + (" ⚠ Metabolic acidosis — check lactic acidosis / RTA"
                   if (elec.get("bicarbonate") is not None
                       and isinstance(elec.get("bicarbonate"), (int, float))
                       and float(elec["bicarbonate"]) < 18) else ""),
                f"  Anion Gap:         {_v(elec.get('anion_gap'), 'mEq/L')}",
                f"  Nephropathy (ICD): {_b(comp.get('nephropathy'))}",
                f"  Active Loop Diuretic: {_b(f.get('on_loop_diuretic'))}",
                f"  Hemodynamic Instability / Vasopressor: {_b(f.get('vasopressor_use'))}",
            ]

        # ── Cardiology panel ──────────────────────────────────────────────────
        elif specialty == "cardiology":
            comp   = cm.get("dm_complications", {}) or {}
            lipids = cm.get("lipids", {}) or {}

            bnp      = f.get("bnp")
            ntprobnp = f.get("ntprobnp")
            lvef     = f.get("lvef")
            has_cv   = comp.get("cardiovascular") or f.get("has_cardiovascular")
            has_hf   = comp.get("heart_failure")  or f.get("has_heart_failure")
            hf_src   = comp.get("heart_failure_source")
            ldl      = lipids.get("ldl")

            lvef_flag = ""
            if lvef is not None:
                if lvef < 40:   lvef_flag = " ⚠ HFrEF — SGLT2i+beta-blocker+ACEi/ARB indicated; avoid TZD"
                elif lvef < 50: lvef_flag = " HFmrEF"

            lines += [
                f"  BNP:               {_v(bnp, 'pg/mL')}"
                + (" ⚠ Elevated (>100) — Active HF likely" if bnp is not None and float(bnp) > 100 else ""),
                f"  NT-proBNP:         {_v(ntprobnp, 'pg/mL')}",
                f"  LVEF:              {_v(lvef, '%')}{lvef_flag}",
                f"  Established ASCVD (ICD): {_b(has_cv)}"
                + (" ⚠ High-intensity statin mandatory (ADA)" if has_cv else ""),
                f"  Heart Failure:     {_b(has_hf)}"
                + (f" [Source: {hf_src}]" if hf_src else ""),
                f"  Peripheral Vascular Disease: {_b(comp.get('peripheral_vascular'))}",
                f"  LDL-C:             {_v(ldl, 'mg/dL')}"
                + (" ⚠ Statin indicated — LDL target < 70 mg/dL for ASCVD"
                   if has_cv and ldl is not None and float(ldl) >= 70 else ""),
                f"  HDL-C:             {_v(lipids.get('hdl'), 'mg/dL')}",
                f"  Total Cholesterol: {_v(lipids.get('total_cholesterol'), 'mg/dL')}",
                f"  Triglycerides:     {_v(lipids.get('triglycerides'), 'mg/dL')}",
                f"  Potassium (K⁺):    {_v(f.get('potassium'), 'mEq/L')}",
                f"  Hemodynamic Instability / Vasopressor: {_b(f.get('vasopressor_use'))}",
                "  Note: Blood pressure, heart rate, and atrial fibrillation status "
                "not available in structured fields — infer from diagnoses if documented.",
            ]

        # ── Endocrinology panel ───────────────────────────────────────────────
        else:
            glucose = cm.get("glucose", {}) or {}
            comp    = cm.get("dm_complications", {}) or {}

            hba1c    = f.get("hba1c")
            fglu     = f.get("fasting_glucose")
            bmi      = f.get("bmi")
            on_ster  = f.get("on_corticosteroids")
            vaso     = f.get("vasopressor_use")

            lines += [
                f"  DM Type:           {f.get('dm_type', 'N/A')}",
                f"  HbA1c:             {_v(hba1c, '%')}",
                f"  Estimated Avg Glucose (eAG): {_v(cm.get('eag_from_hba1c'), 'mg/dL')}",
                f"  Fasting Glucose:   {_v(fglu, 'mg/dL')}",
            ]

            if glucose.get("data_available"):
                lines += [
                    f"  Mean Inpatient Glucose:     {_v(glucose.get('mean_glucose'), 'mg/dL')}",
                    f"  Glucose-in-Range (70-180):  {_v(glucose.get('glucose_in_range_rate'), '%')}",
                    f"  Hypoglycemia Events (<70):  {glucose.get('hypoglycemia_events', 'N/A')}",
                    f"  Severe Hypo Events (<54):   {glucose.get('severe_hypoglycemia_events', 'N/A')}",
                    f"  Glucose Measurements/day:   {glucose.get('glucose_sampling_frequency_per_day', 'N/A')}",
                ]
            else:
                lines.append("  Inpatient glucose dynamics: not available")

            lines += [
                f"  BMI:               {_v(bmi, 'kg/m²')}",
                f"  Hypoglycemia History: {_b(f.get('hypoglycemia_history'))}",
                f"  On Systemic Corticosteroids: {_b(on_ster)}"
                + (" ⚠ Steroid-induced hyperglycemia — raise targets; prefer insulin"
                   if on_ster else ""),
                f"  Hemodynamic Instability / Vasopressor: {_b(vaso)}"
                + (" ⚠ ICU — IV insulin drip preferred; hold oral agents and SC non-insulin"
                   if vaso else ""),
                f"  Retinopathy (ICD): {_b(comp.get('retinopathy'))}",
                f"  Neuropathy (ICD):  {_b(comp.get('neuropathy'))}",
                f"  Liver Dysfunction: {_b(f.get('liver_dysfunction'))}",
                f"  eGFR (renal context for DM drugs): {_v(f.get('egfr'), 'mL/min/1.73m²')}",
            ]

        return "\n".join(lines)

    @staticmethod
    def _build_specialty_drug_scope(role: str, ada_year: int) -> str:
        """
        Build drug taxonomy scope text from config constants, filtered by era (ada_year).

        The LLM must use EXACT category names and standard INN drug names from this
        taxonomy in its JSON output — enabling downstream DrugNormalizer and OSE/GAS
        scoring to match without additional normalization.

        Args:
            role:     Chinese role key
            ada_year: Filters out drugs not yet FDA-approved by this year.
        """
        specialty = {
            "内分泌科": "endocrinology",
            "肾科":     "nephrology",
            "心内科":   "cardiology",
        }.get(role, "endocrinology")

        fda = DrugEraConfig.FDA_APPROVAL_YEAR

        def era_ok(drug_name: str) -> bool:
            yr = fda.get(drug_name.lower())
            return yr is None or yr <= ada_year

        def fmt_cat(cat: str, drugs: List[str]) -> Optional[str]:
            eligible = [d for d in drugs if era_ok(d)]
            return f"    {cat}: {', '.join(eligible)}" if eligible else None

        lines: List[str] = []

        # ── Section A: DM glycemic drugs ──────────────────────────────────────
        if specialty == "endocrinology":
            lines.append(
                "SECTION A — DM Glycemic Drugs "
                "(populate recommended_drugs from these categories ONLY):"
            )
        else:
            lines.append(
                "SECTION A — DM Glycemic Drugs "
                "(SAFETY EVALUATION only — flag unsafe drugs in dm_drugs_safety_notes; "
                "leave recommended_drugs as []):"
            )
        for cat, drugs in DIABETES_MEDICATIONS.items():
            entry = fmt_cat(cat, drugs)
            if entry:
                lines.append(entry)

        lines.append("")

        # ── Section B: Specialty complication drugs ────────────────────────────
        cats = _SPECIALTY_COMPLICATION_CATS.get(specialty, [])
        if cats:
            lines.append(
                "SECTION B — Specialty Complication Drugs "
                "(populate complication_drugs from these categories ONLY):"
            )
            for cat in cats:
                if cat in COMPLICATION_DRUGS:
                    entry = fmt_cat(cat, COMPLICATION_DRUGS[cat])
                    if entry:
                        lines.append(entry)
        else:
            lines.append("SECTION B — Complication Drugs: (none in scope for this specialty)")

        lines += [
            "",
            "⚠ HARD CONSTRAINTS:",
            "  1. Use EXACT category names (case-sensitive, e.g., 'Biguanides', 'Lipid_Statins').",
            "  2. Use standard INN drug names as listed above.",
            f" 3. Do NOT recommend any drug not approved by {ada_year}.",
            "  4. Leave recommended_drugs/complication_drugs as [] if no indication — do not fabricate.",
        ]

        return "\n".join(lines)

    def _get_expert_opinion(
        self,
        role:               str,
        patient_summary:    str,
        patho_text:         str,
        specialty_evidence: str,
        specialty_panel:    str = "",
        drug_scope:         str = "",
        era_label:          str = "Unknown",
        ada_year:           int = 2020,
    ) -> ExpertOpinion:
        """Call LLM to get a single expert opinion."""
        start  = time.time()
        config = EXPERT_CONFIGS[role]
        role_display = config.get("role_display", role)

        era_constraint = (
            f"\n⚠ ERA CONSTRAINT: This patient was admitted in the {era_label} era (≤{ada_year}). "
            f"Cite ONLY guidelines and trial evidence published by {ada_year}. "
            f"Do NOT reference studies or drugs approved/published after {ada_year}."
        )

        evidence_header = (
            f"[{role_display} Specialty Guideline Evidence (≤{ada_year})]"
            if specialty_evidence else
            f"[ADA Guideline Evidence (≤{ada_year})] — specialty-specific evidence unavailable"
        )

        prompt = f"""You are a {role_display} consultant in a Virtual Multidisciplinary Team (MDT) \
consultation for a hospitalized diabetes patient.
{era_constraint}

[Patient Summary]
{patient_summary}

{specialty_panel}

[Pathophysiological State Analysis]
{patho_text}

{evidence_header}
{specialty_evidence if specialty_evidence else "(none retrieved)"}

[Drug Taxonomy — You MUST use these exact names in your JSON output]
{drug_scope}

[Your Consultation Task]
{config['task']}

Output strictly as JSON (no extra text, no markdown fences):
{{
  "recommended_drugs": [
    {{"drug": "INN name", "category": "exact taxonomy category", \
"reason": "indication with guideline citation and year", "dose": "suggested dose"}}
  ],
  "complication_drugs": [
    {{"drug": "INN name", "category": "exact taxonomy category", \
"indication": "specific indication", "dose": "suggested dose"}}
  ],
  "dm_drugs_safety_notes": {{"drug_name": "hold/reduce to X mg/day/safe — clinical reason (cite eGFR/LVEF value)"}},
  "reasoning": "Detailed specialty reasoning citing specific lab values, guideline names, \
and trial data (include publication year for each citation)",
  "key_concerns": [
    "Concern 1 — be specific (e.g., 'Metformin: eGFR=28 mL/min — ABSOLUTE CONTRAINDICATION per KDIGO 2012, hold immediately')"
  ]
}}

Output rules:
- recommended_drugs: DM glycemic drugs only; endocrinology fills this; \
cardiology/nephrology leave []
- complication_drugs: specialty complication drugs only; \
use EXACT category and INN names from taxonomy
- dm_drugs_safety_notes: cardiology/nephrology use this dict for DM drug flags; \
endocrinology leaves {{}}
- Leave [] or {{}} for any section with no indication — do not fabricate recommendations
- Do NOT recommend drugs outside the taxonomy or not yet approved by {ada_year}"""

        response = self.llm.chat(
            messages    = [{"role": "user", "content": prompt}],
            model       = self.model_name,
            temperature = self.temperature,
            max_tokens  = self.max_tokens,
            deep_thinking = False,
        )
        raw_text = self.llm.get_response_text(response)
        drugs, reasoning, concerns = self._parse_expert_response(raw_text)

        # [v5] 额外解析 complication_drugs
        comp_drugs: List[Dict] = []
        try:
            import re as _re
            _clean = _re.sub(r'<think>.*?</think>', '', raw_text, flags=_re.DOTALL).strip()
            for pat in [r"```json\s*(.*?)\s*```", r"```\s*(.*?)\s*```"]:
                _m = _re.search(pat, _clean, _re.DOTALL)
                if _m:
                    _clean = _m.group(1).strip(); break
            _data = json.loads(_clean)
            raw_comp = _data.get("complication_drugs", [])
            comp_drugs = [c for c in raw_comp if isinstance(c, dict) and c.get("drug")]
        except Exception:
            pass  # 解析失败时留空，不影响主流程

        return ExpertOpinion(
            role               = role,
            recommended_drugs  = drugs,
            complication_drugs = comp_drugs,
            reasoning          = reasoning,
            key_concerns       = concerns,
            raw_output         = raw_text,
            inference_time     = time.time() - start,
        )

    def _adjudicate(
        self,
        expert_opinions:    Dict[str, ExpertOpinion],
        conflicts:          List[ConflictItem],
        patient_summary:    str,
        ada_evidence:       str,        # 向后兼容参数名，实际接受任何来源的指南证据
        expert_weights:     Optional[Dict[str, float]] = None,   # Phase 2 动态权重
        era_label:          str = "Unknown",
        ada_year:           int = 2020,
    ) -> Tuple[List[Dict], List[Dict], str, str]:
        """
        裁决整合各专家意见（Phase 2 支持专科权重注入）

        Args:
            expert_weights: {角色名: 权重}，来自 _compute_expert_weights()。
                            None 或空 dict → 等权（Phase 1 行为）

        返回 (final_dm_drugs, final_complication_drugs, final_reasoning, consensus_level)
        [v5] 分离降糖药 / 并发症药输出，分别对应 JSON schema 的两个字段。
        consensus_level 当前未在仲裁 prompt 中定义，恒返回 "unreported"；如需启用须先补 JSON schema，属行为变更
        """
        era_constraint = (
            f"\n⚠ ERA CONSTRAINT: This patient was admitted in the {era_label} era (≤{ada_year}). "
            f"Cite ONLY guidelines and trial evidence published by {ada_year}.\n"
        )

        # Retrieve consolidated evidence for adjudication
        adjudication_evidence = ada_evidence
        if self.registry:
            _queries: List[str] = []
            for op in expert_opinions.values():
                _queries.extend(op.key_concerns[:2])
            if _queries:
                try:
                    all_ev = []
                    for specialty in ("endocrinology", "nephrology", "cardiology"):
                        ev = self.registry.format_evidence_for_specialty(
                            specialty = specialty,
                            queries   = _queries[:3],
                            era_year  = ada_year,
                            top_k     = 3,
                        )
                        if ev:
                            all_ev.append(ev)
                    if all_ev:
                        adjudication_evidence = "\n\n".join(all_ev)
                except Exception as _e:
                    logger.warning(f"Adjudication evidence retrieval failed, using ada_evidence: {_e}")

        # Specialty weight header (shown only when weight asymmetry is meaningful)
        _weights = expert_weights or {}
        _show_weights = bool(_weights) and max(_weights.values(), default=0) > 0.45
        if _show_weights:
            weight_header = "[Specialty Weight Distribution (based on patient's primary domain)]\n"
            weight_header += "\n".join(
                f"  {role}: {w:.0%} ({'dominant specialty' if w >= 0.4 else 'contributing specialty'})"
                for role, w in sorted(_weights.items(), key=lambda x: -x[1])
            )
            weight_header += "\n"
        else:
            weight_header = ""

        opinions_text = "\n\n".join([
            f"[{role} Opinion]\n"
            f"DM drug recommendation: {', '.join(op.recommended_drugs) or '(see concerns)'}\n"
            f"Complication drugs: {', '.join(d.get('drug', '') for d in op.complication_drugs) or '(none)'}\n"
            f"Reasoning: {op.reasoning[:400]}\n"
            f"Key concerns: {'; '.join(op.key_concerns[:3])}"
            for role, op in expert_opinions.items()
        ])

        conflicts_text = ""
        if conflicts:
            conflicts_text = "\n".join([
                f"⚠ [{c.severity.upper()}] {c.drug}: {c.expert_a} recommends vs {c.expert_b} opposes"
                for c in conflicts[:5]
            ])

        adjudication_prompt = f"""You are a senior clinical endocrinologist chairing a Virtual MDT \
for inpatient diabetes management.
{era_constraint}
[Patient Summary]
{patient_summary}

{weight_header}[Specialty Expert Opinions]
{opinions_text}

[Detected Conflicts]
{conflicts_text if conflicts_text else "No major conflicts — expert opinions are broadly consistent."}

[Consolidated Guideline Evidence (Multi-source)]
{adjudication_evidence if adjudication_evidence else "(none)"}

[Adjudication Task]
Synthesize the expert opinions into a final, complete drug management plan:
1. CRITICAL conflicts: explicitly state which recommendation you accept/reject and why
2. Safety priority: always defer to nephrology/cardiology contraindications
3. Specialty weights: when evidence is equal, the dominant specialty's view takes priority
4. Cite guideline evidence for key decisions (source: ADA/KDIGO/ESC/ACC-AHA + year)
5. Taxonomy compliance: use EXACT category names and standard INN drug names

Output strictly as JSON (no extra text, no markdown fences):
{{
  "dm_drugs": [
    {{
      "drug":     "standard INN name",
      "category": "exact taxonomy category (e.g., Biguanides)",
      "reason":   "rationale including multi-specialty consensus or conflict resolution \
with guideline citation",
      "dosage":   "suggested dose"
    }}
  ],
  "complication_drugs": [
    {{
      "drug":     "standard INN name",
      "category": "exact taxonomy category (e.g., Lipid_Statins)",
      "reason":   "specialty recommendation rationale with guideline citation",
      "dosage":   "suggested dose"
    }}
  ],
  "reasoning": "Complete MDT reasoning chain including all conflict resolution steps",
  "consensus_level": "full/partial/disputed"
}}"""

        response = self.llm.chat(
            messages      = [{"role": "user", "content": adjudication_prompt}],
            model         = self.model_name,
            temperature   = 0.1,     # 裁决用低温度，保证确定性
            max_tokens    = self.max_tokens,
            deep_thinking = True,    # 裁决开启深度思考
        )
        raw_text = self.llm.get_response_text(response)

        # Fix V1: 裁决结果解析同样使用"从末尾向前找最后一个完整 JSON 对象"策略
        text_clean = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()
        data = {}
        json_str = None
        for pat in [r"```json\s*(.*?)\s*```", r"```\s*(.*?)\s*```"]:
            m_pat = re.search(pat, text_clean, re.DOTALL)
            if m_pat:
                json_str = m_pat.group(1).strip()
                break
        if not json_str:
            brace_positions = [i for i, c in enumerate(text_clean) if c == "{"]
            for pos in reversed(brace_positions):
                candidate = text_clean[pos:]
                depth, end = 0, -1
                for j, ch in enumerate(candidate):
                    if ch == "{": depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            end = j
                            break
                if end != -1:
                    json_str = candidate[:end + 1]
                    break
        if json_str:
            try:
                data = json.loads(json_str)
            except Exception:
                try:
                    import json_repair
                    data = json_repair.repair_json(json_str, return_objects=True)
                except Exception:
                    data = {}
        else:
            try:
                import json_repair
                data = json_repair.repair_json(text_clean, return_objects=True)
            except Exception:
                data = {}

        final_dm_drugs     = [r for r in data.get("dm_drugs", [])
                               if isinstance(r, dict) and r.get("drug")]
        final_comp_drugs   = [r for r in data.get("complication_drugs", [])
                               if isinstance(r, dict) and r.get("drug")]
        # 向后兼容：若模型输出了旧格式 recommended_drugs，归入 dm_drugs
        if not final_dm_drugs:
            final_dm_drugs = [r for r in data.get("recommended_drugs", [])
                              if isinstance(r, dict) and r.get("drug")]
        # [v6 bugfix] 若裁决未能输出 complication_drugs（LLM 格式退化），
        # 从各专科意见中直接聚合（去重，以药物 INN 名为 key）
        if not final_comp_drugs:
            seen_drugs: set = set()
            for op in expert_opinions.values():
                for cd in getattr(op, "complication_drugs", []):
                    drug_key = cd.get("drug", "").lower().strip()
                    if drug_key and drug_key not in seen_drugs:
                        final_comp_drugs.append(cd)
                        seen_drugs.add(drug_key)
            if final_comp_drugs:
                logger.debug(
                    f"[v6 fallback] adjudication complication_drugs empty, "
                    f"aggregated {len(final_comp_drugs)} drugs from expert opinions"
                )

        final_reasoning    = data.get("reasoning", raw_text[:500])
        consensus = data.get("consensus_level") or "unreported"

        return final_dm_drugs, final_comp_drugs, final_reasoning, consensus

    # ── 工具方法 ─────────────────────────────────────────────

    @staticmethod
    def _format_patho_states(phase_a_result: Dict) -> str:
        """
        格式化病理生理状态文本，供专家 prompt 使用。

        兼容两种格式：
        - 新格式（PCAR v2）：diamas_metadata.phenotype_states，字段为 name/confidence/implication/specialty_tags
        - 旧格式（Phase A）：diamas_metadata.patho_states，字段为 state/confidence/implication
        """
        meta   = phase_a_result.get("diamas_metadata", {})

        # 优先使用新格式
        states = meta.get("phenotype_states") or meta.get("patho_states", [])
        if not states:
            return "（无病理生理状态分析）"

        lines = []
        for s in states:
            if not isinstance(s, dict):
                continue
            name       = s.get("name") or s.get("state", "Unknown state")
            conf       = s.get("confidence", 0.0)
            implication = s.get("implication", "")
            specialty  = "/".join(s.get("specialty_tags", [])) if s.get("specialty_tags") else ""
            era_flag   = "Era-sensitive" if s.get("era_sensitivity") else ""
            extra      = "  ".join(filter(None, [specialty, era_flag]))

            line = f"- {name} (confidence {conf:.0%})"
            if extra:
                line += f"  [{extra}]"
            if implication:
                line += f": {implication}"
            lines.append(line)

        return "\n".join(lines)

    def _get_expert_rebuttal(
        self,
        role:             str,
        original_opinion: ExpertOpinion,
        other_concerns:   Dict[str, List[str]],   # {其他角色: key_concerns}
        conflicts:        List[ConflictItem],
        patient_summary:  str,
        era_label:        str = "Unknown",
        ada_year:         int = 2020,
    ) -> ExpertOpinion:
        """
        迭代辩论：当前专家阅读其他专科的顾虑后，决定是否修正推荐。

        此方法在 conduct_mdt 的辩论循环中被调用。
        若对方顾虑涉及自己推荐的药物，专家可以：
          a) 坚持原推荐（并给出更充分的理由）
          b) 撤回或更换推荐（并说明原因）

        返回更新后的 ExpertOpinion（字段同 _get_expert_opinion）。
        """
        start = time.time()
        config = EXPERT_CONFIGS[role]
        role_display = config.get("role_display", role)

        # Build other-specialty concerns summary
        concern_lines = []
        for other_role, concerns in other_concerns.items():
            other_display = EXPERT_CONFIGS.get(other_role, {}).get("role_display", other_role)
            if concerns:
                concern_lines.append(f"[{other_display}] " + "; ".join(concerns[:3]))
        other_concerns_text = "\n".join(concern_lines) or "(none)"

        conflict_drugs = list({c.drug for c in conflicts
                               if c.expert_a == role or c.expert_b == role})
        conflict_drug_text = ", ".join(conflict_drugs) if conflict_drugs else "(no direct conflicts)"

        era_constraint = (
            f"⚠ ERA CONSTRAINT: This patient was admitted in the {era_label} era (≤{ada_year}). "
            f"Cite only evidence published by {ada_year}."
        )

        prompt = f"""You are a {role_display} consultant in a Virtual MDT rebuttal round.
{era_constraint}

[Patient Summary]
{patient_summary}

[Your Original Recommendation]
DM drugs: {', '.join(original_opinion.recommended_drugs) or '(none)'}
Reasoning: {original_opinion.reasoning[:300]}
Your concerns: {'; '.join(original_opinion.key_concerns[:3])}

[Other Specialties' Concerns — Review Carefully]
{other_concerns_text}

[Drugs Involved in Active Conflicts]
{conflict_drug_text}

[Rebuttal Task]
Review your original recommendations in light of the other specialties' concerns:
1. If their concern is clinically valid, withdraw or replace the conflicting drug and explain
2. If you maintain your recommendation, provide specific counter-evidence against their concern
3. Either way, re-output your complete updated recommendation

Output strictly as JSON (no extra text, no markdown fences):
{{
  "recommended_drugs": ["drug1", "drug2", ...],
  "reasoning": "Updated reasoning — explicitly state whether you accepted the concern and why \
(e.g., 'Withdrew Metformin per nephrology: eGFR=28 confirms absolute contraindication')",
  "key_concerns": ["Updated concern 1", ...]
}}"""

        response = self.llm.chat(
            messages    = [{"role": "user", "content": prompt}],
            model       = self.model_name,
            temperature = self.temperature,
            max_tokens  = self.max_tokens,
            deep_thinking = False,
        )
        raw_text = self.llm.get_response_text(response)
        drugs, reasoning, concerns = self._parse_expert_response(raw_text)

        return ExpertOpinion(
            role               = role,
            recommended_drugs  = drugs if drugs else original_opinion.recommended_drugs,
            complication_drugs = original_opinion.complication_drugs,  # [FIX] preserve from round 1
            reasoning          = reasoning or original_opinion.reasoning,
            key_concerns       = concerns or original_opinion.key_concerns,
            raw_output         = raw_text,
            inference_time     = time.time() - start,
        )

    @staticmethod
    def _parse_expert_response(text: str) -> Tuple[List[str], str, List[str]]:
        """解析专家意见的 JSON 输出"""
        text_clean = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

        # Fix V1: 同 diamas_inference_stage._parse_response，从末尾向前找最后一个完整 JSON 对象，
        # 避免贪婪 \{.*\} 在含多个 JSON 块时匹配过长的无效字符串。
        data = {}
        json_str = None
        for pat in [r"```json\s*(.*?)\s*```", r"```\s*(.*?)\s*```"]:
            m = re.search(pat, text_clean, re.DOTALL)
            if m:
                json_str = m.group(1).strip()
                break

        if not json_str:
            brace_positions = [i for i, c in enumerate(text_clean) if c == "{"]
            for pos in reversed(brace_positions):
                candidate = text_clean[pos:]
                depth, end = 0, -1
                for j, ch in enumerate(candidate):
                    if ch == "{": depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            end = j
                            break
                if end != -1:
                    json_str = candidate[:end + 1]
                    break

        if json_str:
            try:
                data = json.loads(json_str)
            except Exception:
                try:
                    import json_repair
                    data = json_repair.repair_json(json_str, return_objects=True)
                except Exception:
                    data = {}
        else:
            try:
                import json_repair
                data = json_repair.repair_json(text_clean, return_objects=True)
            except Exception:
                data = {}

        drugs = data.get("recommended_drugs", [])
        if isinstance(drugs, list):
            # Fix R2-M3: LLM 有时返回 [{"drug": "empagliflozin", "dose": "10mg"}]
            # str(dict) 会产生 "{'drug': 'empagliflozin'...}" 无法匹配任何药物名
            def _extract_drug_name(d) -> str:
                if isinstance(d, str): return d.strip()
                if isinstance(d, dict):
                    return str(d.get("drug") or d.get("name") or
                               d.get("drug_name") or "").strip()
                return str(d).strip()
            drugs = [_extract_drug_name(d) for d in drugs]
            drugs = [d for d in drugs if d]  # 过滤空字符串
        reasoning = data.get("reasoning", "")
        concerns  = data.get("key_concerns", [])
        if isinstance(concerns, list):
            concerns = [c if isinstance(c, str) else str(c) for c in concerns]

        return drugs, reasoning, concerns


# ═══════════════════════════════════════════════════════════
# MDTResult → baseline_evaluation.py 兼容格式转换
# ═══════════════════════════════════════════════════════════

def mdt_to_inference_format(
    mdt_result:     MDTResult,
    phase_a_result: Dict,
) -> Dict:
    """
    将 MDTResult 转换为与 llm_inference_stage.RecommendationResult 兼容的格式

    Phase B 最终输出可直接送入 baseline_evaluation.py 评测，
    无需修改评测代码。
    """
    base = phase_a_result.copy()  # 保留 Phase A 的所有字段

    # 覆盖推荐和推理（升级为 MDT 结果）
    base["recommended_drugs"] = mdt_result.final_drugs
    base["reasoning"]         = mdt_result.final_reasoning
    base["success"]           = len(mdt_result.final_drugs) > 0
    base["error_message"]     = None if mdt_result.final_drugs else "MDT 无输出"

    # 追加 MDT 专属元数据
    base["mdt_metadata"] = {
        "expert_opinions": [
            {
                "role":              op.role,
                "recommended_drugs": op.recommended_drugs,
                "key_concerns":      op.key_concerns,
                "reasoning":         op.reasoning[:200],
            }
            for op in mdt_result.expert_opinions
        ],
        "conflicts":      [asdict(c) for c in mdt_result.conflicts],
        "consensus_level": mdt_result.consensus_level,
        "total_mdt_time":  mdt_result.total_time,
        # Phase 2 新增字段
        "debate_rounds":   mdt_result.debate_rounds,
        "expert_weights":  mdt_result.expert_weights,
    }

    return base


# ═══════════════════════════════════════════════════════════
# 独立运行（读取 Phase A 结果，输出 MDT 增强结果）
# ═══════════════════════════════════════════════════════════

def run_mdt_on_directory(
    phase_a_dir:          Path,
    patient_dir:          Path,
    output_dir:           Path,
    model_name:           str   = "deepseek-r1-250528",
    temperature:          float = 0.3,
    max_patients:         Optional[int] = None,
    guidelines_dir:       Optional[Path] = None,
    index_dir:            Optional[Path] = None,
    guideline_sources:    Optional[List[str]] = None,
    judge_model:          Optional[str] = JUDGE_MODEL_DEFAULT,
    max_debate_rounds:    int  = MAX_DEBATE_ROUNDS_DEFAULT,
    use_dynamic_weights:  bool = True,
):
    """
    批量对 Phase A 结果执行 MDT 会诊

    Args:
        phase_a_dir:       diamas_inference_stage 的输出目录
        patient_dir:       原始患者 JSON 目录
        output_dir:        MDT 结果输出目录
        model_name:        使用的 LLM
        temperature:       temperature
        max_patients:      最大处理患者数
        guidelines_dir:    多源指南根目录；None 时降级为只用 phase_a ada_evidence
        index_dir:         索引文件目录；默认为 guidelines_dir/../indexes
        guideline_sources: 参与检索的源列表；None 表示全部
    """
    from agents.llm_client import DeepseekClient
    import concurrent.futures
    from threading import Lock

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fix V4: 断点续传 —— 记录已完成的 patient_id，崩溃重启后跳过
    progress_file = output_dir / "mdt_progress.json"
    completed_pids: set = set()
    if progress_file.exists():
        try:
            completed_pids = set(json.loads(progress_file.read_text()).get("completed", []))
            logger.info(f"断点续传：已跳过 {len(completed_pids)} 例")
        except Exception:
            pass
    progress_lock = threading.Lock()

    def _mark_done(pid: str):
        with progress_lock:
            completed_pids.add(pid)
            progress_file.write_text(json.dumps({"completed": list(completed_pids)}))

    llm    = DeepseekClient()

    # ── 构建多源指南注册表（可选）────────────────────────────────────────────
    registry = None
    if guidelines_dir and Path(guidelines_dir).exists():
        try:
            from diamas_utils.ada_guideline_processor import build_all_indexes
            _idx_dir = Path(index_dir) if index_dir else Path(guidelines_dir).parent / "indexes"
            registry = build_all_indexes(
                guidelines_dir = Path(guidelines_dir),
                index_dir      = _idx_dir,
                sources        = guideline_sources,
                force          = False,
            )
            active = registry.active_sources
            logger.info(f"MDT 注册表构建完成：sources={active}")
        except Exception as _e:
            logger.warning(f"指南注册表构建失败，MDT 将使用 phase_a ada_evidence: {_e}")
            registry = None
    else:
        logger.info("未指定 guidelines_dir，MDT 将使用 phase_a ada_evidence（降级模式）")

    mdt = VirtualMDT(
        llm_client           = llm,
        model_name           = model_name,
        temperature          = temperature,
        registry             = registry,
        judge_model          = judge_model,
        max_debate_rounds    = max_debate_rounds,
        use_dynamic_weights  = use_dynamic_weights,
    )

    phase_a_files = sorted(Path(phase_a_dir).glob("patient_*.json"))
    if max_patients:
        phase_a_files = phase_a_files[:max_patients]

    # 预加载患者 JSON
    # Fix: 同时支持 "patient_12345_67890.json" 和 "12345_67890.json" 两种命名格式
    patient_cache: Dict[str, Dict] = {}
    for pf in Path(patient_dir).glob("*.json"):
        try:
            with open(pf, encoding="utf-8") as fh:
                key = pf.stem
                if key.startswith("patient_"):
                    key = key[len("patient_"):]   # 去掉 "patient_" 前缀以匹配 f"{sid}_{hid}"
                patient_cache[key] = json.load(fh)
        except Exception:
            pass

    logger.info(f"Phase B MDT: {len(phase_a_files)} 例，patient_cache={len(patient_cache)} 条")

    success = failed = 0
    from tqdm import tqdm
    from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed

    # Fix R2-M1: 原来是串行 for 循环（concurrent.futures 只被导入但未使用），
    # 1500 例 × ~46s = ~19h。改为 ThreadPoolExecutor，3–5 workers 可降至 ~5h。
    prog_lock = threading.Lock()  # 保护 success/failed 计数和进度写入

    def _process_one_mdt(pa_file: Path) -> str:
        """返回 'success' / 'skipped' / 'failed'"""
        try:
            with open(pa_file, encoding="utf-8") as f:
                phase_a = json.load(f)

            pid = phase_a.get("patient_id", "")

            with prog_lock:
                if pid in completed_pids:
                    return "skipped"

            sid = phase_a.get("subject_id", 0)
            hid = phase_a.get("hadm_id", 0)
            patient_data = patient_cache.get(f"{sid}_{hid}", {})

            mdt_res = mdt.conduct_mdt(
                phase_a_result  = phase_a,
                flat            = None,          # computed inline from patient_data
                patient_data    = patient_data,
            )
            merged = mdt_to_inference_format(mdt_res, phase_a)

            with prog_lock:
                out_path = output_dir / f"patient_{pid}.json"
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(merged, f, indent=2, ensure_ascii=False)
                _mark_done(pid)

            return "success"
        except Exception as e:
            logger.error(f"MDT 失败 {pa_file.name}: {e}")
            return "failed"

    num_workers = min(5, max(2, len(phase_a_files) // 100 + 1))
    logger.info(f"Phase B MDT 并行 workers={num_workers}")

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futs = {pool.submit(_process_one_mdt, f): f for f in phase_a_files}
        for fut in tqdm(_as_completed(futs), total=len(phase_a_files), desc="Virtual MDT"):
            status = fut.result()
            if status == "success":
                success += 1
            elif status == "failed":
                failed += 1
            # skipped 不计入

    logger.info(f"Phase B 完成: success={success}, failed={failed}")


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Virtual MDT - Phase B")
    parser.add_argument("--phase_a_dir",      required=True)
    parser.add_argument("--patient_dir",      required=True)
    parser.add_argument("--output_dir",       required=True)
    parser.add_argument("--model",            default="deepseek-r1-250528")
    parser.add_argument("--temperature",      type=float, default=0.3)
    parser.add_argument("--max_patients",     type=int, default=None)
    parser.add_argument("--guidelines_dir",   default=None,
                        help="多源指南根目录（如 data/guidelines）；省略则降级到 phase_a ada_evidence")
    parser.add_argument("--index_dir",        default=None,
                        help="索引缓存目录（默认为 guidelines_dir/../indexes）")
    parser.add_argument("--sources",          nargs="*", default=None,
                        help="参与检索的指南源列表（如 ada kdigo esc）；省略=全部")
    parser.add_argument("--judge_model",      default=JUDGE_MODEL_DEFAULT,
                        help="冲突检测 Judge 模型（须与推荐模型不同家族）；"
                             "'none'/'None' → 降级为规则检测器")
    parser.add_argument("--debate_rounds",    type=int, default=1,
                        help="MDT 迭代辩论轮次（1=Phase 1 行为；2=Phase 2 辩论，"
                             "冲突数>2 时自动触发追加轮次）")
    parser.add_argument("--no_dynamic_weights", action="store_true",
                        help="禁用动态专家权重（等权，Phase 1 行为）")
    args = parser.parse_args()

    run_mdt_on_directory(
        phase_a_dir          = Path(args.phase_a_dir),
        patient_dir          = Path(args.patient_dir),
        output_dir           = Path(args.output_dir),
        model_name           = args.model,
        temperature          = args.temperature,
        max_patients         = args.max_patients,
        guidelines_dir       = Path(args.guidelines_dir) if args.guidelines_dir else None,
        index_dir            = Path(args.index_dir)      if args.index_dir      else None,
        guideline_sources    = args.sources,
        judge_model          = None if args.judge_model.lower() == "none" else args.judge_model,
        max_debate_rounds    = args.debate_rounds,
        use_dynamic_weights  = not args.no_dynamic_weights,
    )
