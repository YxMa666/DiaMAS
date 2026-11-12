#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diamas_utils/prompt_builder.py  (v7 — single-mode + per-source evidence)
========================================================================
5段式 Prompt 构建器

v7 关键变更（P0 修复）:
  1. 删除 mode_a / mode_b / cii_divergence 双模式 schema
     原因：让 LLM 同时预测"指南最优"和"医生实际处方"会引入方法学污染。
     CII (Clinical Inertia Index, C5 contribution) 改为后处理：
       CII = compare(系统推荐, 物理处方 GT)，由 gas_physician_validation 计算。
  2. _format_evidence() 改为接受 evidence_by_source: Dict[str, str]
     按指南源分组展示证据：
       ■ ADA Standards of Medical Care  [Endocrinology]
       ■ KDIGO CKD Guidelines           [Nephrology]
       ■ ESC CVD-Diabetes Guidelines    [Cardiology]
       ■ ACC/AHA Guidelines             [Cardiology]
     缺失时显式声明（"no chunks retrieved — verify --sources includes esc/acc_aha"），
     避免静默失败。保留 ada_evidence: str 参数作为向后兼容（自动构造 dict）。
  3. JSON schema 新增 complication_drugs 字段（与 MDT v5 schema 对齐）
  4. R8/R9 去掉 [Mode B] 标签，改为通用临床原则（保留语义，符合单模式语境）

包含：
  DiamasPromptBuilder      — 生成 (system_content, user_content) 元组
  era_label_for_year()     — 委托 DrugEraConfig.era_label_for_year（向后兼容）
  _get_era_drug_classes()  — 委托 DrugEraConfig.get_era_drug_classes（向后兼容）

Prompt 结构：
  Section 1: Patient Clinical Snapshot
  Section 2: Pathophysiological State Map（era-implication 感知）
  Section 3: Era-Matched Evidence（按指南源分组，严格 ≤ era_year）
  Section 4: Virtual MDT Summary（条件性，含各专科 DM + 并发症药 + 关键顾虑）
  Section 5: Contrastive Reasoning Task（R1–R9 规则集）

"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from diamas_utils.clinical_phenotyper import ClinicalPhenotypeState


from agents.config import (
    DrugEraConfig, DIABETES_MEDICATIONS, COMPLICATION_DRUGS
)


# 指南源 → (显示名称, 主导专科) 映射，用于 Section 3 按源分组的 header
SOURCE_DISPLAY: Dict[str, Tuple[str, str]] = {
    "ada":             ("ADA Standards of Medical Care",   "Endocrinology"),
    "aace":            ("AACE Diabetes Guidelines",        "Endocrinology"),
    "ada_supplements": ("ADA Supplements / Consensus",     "Endocrinology + Nephrology"),
    "kdigo":           ("KDIGO CKD Guidelines",            "Nephrology"),
    "esc":             ("ESC CVD-Diabetes Guidelines",     "Cardiology"),
    "acc_aha":         ("ACC/AHA Guidelines",              "Cardiology"),
    "chinese":         ("Chinese Diabetes Guidelines",     "Endocrinology (CN cohort)"),
}

# 指南源 → 主要服务专科（用于"缺失时哪个专科会受影响"的提示）
SOURCE_TO_PRIMARY_SPECIALTY: Dict[str, str] = {
    "ada":             "endocrinology",
    "aace":            "endocrinology",
    "ada_supplements": "endocrinology",
    "kdigo":           "nephrology",
    "esc":             "cardiology",
    "acc_aha":         "cardiology",
    "chinese":         "endocrinology",
}


# ── 并发症药物评估范围声明（与 config.COMPLICATION_DRUGS 保持同步）──────────────
# 用于 Section 5 中告知 LLM 哪些药物类别在评估分类表内、哪些已被显式排除，
# 避免 LLM 推荐无法通过 DrugNormalizer 映射的药物（造成 FP）。
# ⛔ 排除项：LLM 审计一致判定为 NOT_RELEVANT 或无 DM 特异性适应症的药物。
COMPLICATION_DRUG_SCOPE: Dict[str, Dict] = {
    "Hypertension_ACEi": {
        "drugs": ["Lisinopril", "Enalapril", "Ramipril", "Benazepril",
                  "Perindopril", "Fosinopril", "Captopril", "Quinapril"],
        "indication": "DM nephropathy, hypertension (ADA Grade A)",
    },
    "Hypertension_ARB": {
        "drugs": ["Losartan", "Valsartan", "Telmisartan", "Irbesartan",
                  "Candesartan", "Olmesartan"],
        "indication": "DM nephropathy, hypertension — use when ACEi not tolerated",
    },
    "Hypertension_CCB": {
        "drugs": ["Levamlodipine", "Nitrendipine"],
        "indication": "DM hypertension adjunct (guideline-endorsed CCBs only)",
        "excluded": "Amlodipine, Nifedipine, Verapamil — NOT in evaluation taxonomy",
    },
    "Hypertension_Diuretics": {
        "drugs": ["Furosemide", "Bumetanide", "Torsemide", "Spironolactone"],
        "indication": "HF volume overload, CKD fluid management",
        "excluded": "HCTZ, Chlorthalidone — not DM-first-line",
    },
    "Hypertension_BetaBlockers": {
        "drugs": ["Metoprolol", "Bisoprolol", "Carvedilol"],
        "indication": "HF (HFrEF), post-MI with DM",
        "excluded": "Labetalol, Atenolol, Propranolol — NOT in evaluation taxonomy",
    },
    "Lipid_Statins": {
        "drugs": ["Atorvastatin", "Rosuvastatin", "Simvastatin",
                  "Pravastatin", "Lovastatin", "Pitavastatin"],
        "indication": "ASCVD prevention — mandatory for DM+ASCVD or DM+very-high-CV-risk",
    },
    "Lipid_NonStatins": {
        "drugs": ["Ezetimibe", "Fenofibrate"],
        "indication": "Add-on when statin target not met, or statin intolerance",
    },
    "Antiplatelet": {
        "drugs": ["Aspirin", "Clopidogrel", "Ticagrelor", "Prasugrel"],
        "indication": "ASCVD secondary prevention; primary prevention in very-high-risk DM",
        "excluded": "Heparin, Enoxaparin — anticoagulants, NOT in Antiplatelet taxonomy",
    },
    "CKD_MRA": {
        "drugs": ["Finerenone", "Eplerenone"],
        "indication": "Diabetic CKD with residual albuminuria on RAAS (post-2021 evidence)",
    },
    "Neuropathy_Gabapentinoids": {
        "drugs": ["Pregabalin", "Gabapentin"],
        "indication": "Diabetic peripheral neuropathy pain",
    },
    "Neuropathy_SNRI": {
        "drugs": ["Duloxetine"],
        "indication": "Diabetic peripheral neuropathy — ADA first-line",
    },
}

# 显式排除药物（常见但不在分类表内，需告知 LLM 以防 FP）
COMPLICATION_DRUGS_EXCLUDED_NOTABLE: List[str] = [
    "Amlodipine (→ use Levamlodipine/Nitrendipine instead if CCB needed)",
    "Labetalol (IV hypertensive urgency only; not in DM complication taxonomy)",
    "Heparin / Enoxaparin (anticoagulants, not antiplatelet; excluded from taxonomy)",
    "HCTZ / Chlorthalidone (worsen insulin resistance; not DM first-line)",
    "Warfarin / NOACs (no DM-specific indication; excluded from taxonomy)",
]


def _get_era_drug_classes(ada_year: int) -> List[Tuple[str, str]]:
    """委托给 DrugEraConfig.get_era_drug_classes（向后兼容包装）。"""
    return DrugEraConfig.get_era_drug_classes(ada_year)


def era_label_for_year(ada_year: int) -> str:
    """委托给 DrugEraConfig.era_label_for_year（向后兼容包装）。"""
    return DrugEraConfig.era_label_for_year(ada_year)


class DiamasPromptBuilder:
    """
    Dia-MAS 5段式 Prompt 构建器。

    返回 (system_content, user_content) 元组，
    调用方将 system_content 放入 API messages 的 system 字段。
    """

    def build(
        self,
        patient_summary:   str,
        states:            List["ClinicalPhenotypeState"],
        ada_evidence:      str,
        pubmed_evidence:   str,
        era_label:         str,
        ada_year:          int,
        mdt_summary:       Optional[str],
        active_roles:      List[str],
        flat:              Dict,
        reflexion_text:    Optional[str] = None,                # [v6 deprecated, ignored]
        guideline_sources: Optional[Dict] = None,
        evidence_by_source: Optional[Dict[str, str]] = None,    # [v7 新增]
    ) -> Tuple[str, str]:
        """
        构建最终 prompt 对。

        Args:
            patient_summary:    患者摘要文本（Section 1）
            states:             ClinicalPhenotypeState 列表
            ada_evidence:       [v7 deprecated, kept for backward compat]
                                合并后的指南证据字符串。当 evidence_by_source 为 None
                                时，作为单一 ADA 源构造 dict。
            pubmed_evidence:    PubMed 检索结果文本
            era_label:          如 "Era-2"
            ada_year:           精确 ADA 年份（get_ada_year_ceiling 输出）
            mdt_summary:        MDT 共识文本（None = 跳过 MDT）
            active_roles:       激活的专科角色列表
            flat:               flatten_clinical_metrics 输出
            reflexion_text:     反思注入文本（v6 起忽略）
            guideline_sources:  {"ada_years": [...], "other": [...]}（动态指南引用声明）
            evidence_by_source: {source_key: formatted_text} 按指南源分组的证据
                                source_key ∈ {"ada", "aace", "ada_supplements",
                                              "kdigo", "esc", "acc_aha", "chinese"}
                                None 时退化使用 ada_evidence 作为单一 ADA 块。

        Returns:
            (system_content, user_content)
        """
        is_t1 = flat.get("is_t1dm", False)
        guideline_ref_str = self._build_guideline_ref(guideline_sources, ada_year)

        # ── 系统角色（v7：单模式） ────────────────────────────────────────────
        # ── KNOWLEDGE PRIORITY clause（共享文本，两个分支都追加）──────────────
        _knowledge_priority = (
            "\n\nKNOWLEDGE PRIORITY (mandatory): Base your reasoning on the evidence "
            "explicitly provided in Section 3. When your parametric knowledge conflicts "
            "with Section 3 content, defer to Section 3. Within Section 3, apply a "
            "recency-first hierarchy: prioritize the most recently published guideline "
            "(≤ admission year) that directly addresses the clinical question; use earlier "
            "guidelines only as supplementary support where the most recent source is "
            "silent or ambiguous. If Section 3 provides no evidence for a decision point, "
            "note the gap explicitly in 'reasoning' — do not silently fill it from "
            "parametric memory, as guideline retrieval gaps are tracked in this study "
            "as RC-5 failures."
        )

        if is_t1:
            system_content = (
                "You are a senior endocrinologist with expertise in diabetes management "
                "and inpatient glycemic safety. "
                f"You apply evidence-based guidelines from {guideline_ref_str} "
                "strictly matched to the patient's clinical era, and you never recommend "
                "agents not yet approved or guideline-endorsed at the time of admission."
                + _knowledge_priority
            )
        else:
            system_content = (
                "You are a senior endocrinologist with expertise in diabetes management "
                "and inpatient glycemic care. You integrate patient pathophysiology, "
                "comorbidities, multi-specialty MDT consensus, and RCT evidence into "
                "your recommendations.\n\n"

                f"You apply {guideline_ref_str} strictly matched to the patient's "
                "clinical era. The patient's diabetes type and clinical details are "
                "described in Section 1 — use those to determine appropriate therapy.\n\n"

                "Your task: produce a single guideline-optimal prescription plan covering "
                "(a) glycemic-control drugs (DM drugs) and "
                "(b) complication-management drugs (statins, ACEi/ARB, β-blockers, "
                "MRAs, loop diuretics, antiplatelets) consistent with the patient's "
                "comorbidities and the MDT consensus shown in Section 4.\n\n"

                f"TEMPORAL FIDELITY (non-negotiable): Admission year ≈ {ada_year}. "
                f"Never cite or recommend any drug, trial, or guideline published/approved after {ada_year}."
                + _knowledge_priority
            )

        # ── Section 内容构建 ────────────────────────────────────────────────
        # v7: 优先使用按源分组的 dict；缺省时从 ada_evidence 构造单 ADA 字典
        if evidence_by_source is None:
            _by_source: Dict[str, str] = {}
            if ada_evidence:
                _by_source["ada"] = ada_evidence
        else:
            _by_source = dict(evidence_by_source)   # 防御性 copy

        s2 = self._format_states(states, ada_year)
        s3 = self._format_evidence(_by_source, pubmed_evidence, era_label, ada_year,
                                    guideline_sources=guideline_sources)
        s4 = self._format_mdt(mdt_summary, active_roles)
        s5 = self._format_contrastive_task(states, is_t1, ada_year, flat,
            temporal_drug_filter=getattr(self, "_temporal_drug_filter", False),)

        # ── JSON schema — category key 从 config 动态生成（Bug-1）───────────
        _era_classes = DrugEraConfig.get_era_drug_classes(ada_year)
        _dm_keys_str = " | ".join(
            ck for ck, _ in _era_classes
        ) + " | Insulins_Rapid"   # Insulins_Rapid 始终并列展示

        _comp_keys_str = " | ".join(COMPLICATION_DRUGS.keys())

        _dm_schema_line = (
            f'{{"drug": "INN name", '
            f'"category": "EXACT key, one of: {_dm_keys_str}", '
            f'"reason": "...", "dose_note": "..."}}'
        )
        _comp_schema_line = (
            f'{{"drug": "INN name", '
            f'"category": "EXACT key, one of: {_comp_keys_str}", '
            f'"reason": "guideline citation", "dose_note": "...", '
            f'"continued_from_home": false}}'
        )

        if is_t1:
            json_schema = (
                '{\n'
                f'  "recommended_drugs": [{_dm_schema_line}],\n'
                '  "complication_drugs": [],\n'
                '  "reasoning": "Full chain: phenotype -> safety -> era -> evidence -> plan",\n'
                '  "excluded_classes": [{"class": "...", "reason": "..."}]\n'
                '}'
            )
        else:
            json_schema = (
                '{\n'
                f'  "recommended_drugs": [{_dm_schema_line}],\n'
                f'  "complication_drugs": [{_comp_schema_line}],\n'
                '  "reasoning": "Full chain: phenotype -> safety -> era -> evidence -> plan",\n'
                '  "excluded_classes": [{"class": "...", "reason": "..."}]\n'
                '}'
            )

        user_content = f"""[Section 1: Patient Clinical Snapshot]
{patient_summary}

[Section 2: Pathophysiological State Map]
(Era: {era_label} — implication text is calibrated to guidelines ≤ {ada_year})
(⚠ Visibility note: Section 1 shows a representative subset of diagnoses. \
Section 2 states are derived from the COMPLETE clinical record — \
all states have verified ICD-code or structured-field evidence, \
even if the source diagnosis is not visible in Section 1 above.)
{s2}

[Section 3: Era-Matched Evidence]
(Only guidelines/RCTs published ≤ {ada_year} are cited below, grouped by guideline source)
{s3}

[Section 4: Virtual MDT Summary]
{s4}

[Section 5: Contrastive Reasoning Task]
{s5}
Output (strict JSON only, no surrounding text):
{json_schema}"""

        return system_content, user_content

    # ── 内部格式化方法 ────────────────────────────────────────────────────────

    @staticmethod
    def _build_guideline_ref(guideline_sources: Optional[Dict], ada_year: int) -> str:
        if not guideline_sources:
            return f"ADA Standards of Medical Care in Diabetes (≤ {ada_year}) and EASD guidelines"
        ada_years    = sorted(guideline_sources.get("ada_years", []))
        other_guides = guideline_sources.get("other", [])
        parts = []
        avail = [y for y in ada_years if y <= ada_year]
        if avail:
            yrs = "/".join(str(y) for y in avail)
            parts.append(f"ADA Standards of Medical Care ({yrs})" if len(avail) > 1
                         else f"ADA Standards of Medical Care {avail[0]}")
        else:
            parts.append(f"ADA Standards of Medical Care (≤ {ada_year})")
        for g in other_guides:
            m = re.search(r"(\d{4})$", g)
            if not m or int(m.group(1)) <= ada_year:
                parts.append(g)
        return ", ".join(parts) if parts else f"ADA Standards of Medical Care (≤ {ada_year})"

    @staticmethod
    def _format_states(states: List["ClinicalPhenotypeState"], ada_year: int) -> str:
        if not states:
            return "(No specific pathophysiological states identified — apply standard T2DM protocol)"
        lines = []
        etype_map = {"deterministic": "deterministic", "probabilistic": "probabilistic",
                     "llm_inferred": "LLM-inferred"}
        for i, s in enumerate(states, 1):
            # [FIX-VISIBILITY] safety_table 来源的 state 可能引用 Section 1 未显示的 ICD 码，
            # 在 evidence_type 中注明，让推理 LLM 和 CAPE auditor 知道有完整依据。
            raw_etype = s.evidence_type
            src = (s.supporting_features or {}).get("source", "")
            if raw_etype == "deterministic" and src == "safety_table":
                etype = "deterministic (ICD-verified, may be from non-displayed diagnosis)"
            else:
                etype = etype_map.get(raw_etype, raw_etype)
            spec_str = "/".join(s.specialty_tags)
            era_flag = "era-sensitive" if s.era_sensitivity else "cross-era stable"
            impl = s.implication
            if s.era_implications:
                avail = sorted(k for k in s.era_implications if k <= ada_year)
                if avail:
                    impl = s.era_implications[max(avail)]
            lines.append(
                f"{i}. {s.name} [{etype}, confidence {s.confidence:.0%}, "
                f"specialty: {spec_str}, {era_flag}]\n"
                f"   Drug implication: {impl}"
            )
        return "\n".join(lines)

    @staticmethod
    def _format_evidence(
        evidence_by_source: Dict[str, str],
        pubmed_evidence:    str,
        era_label:          str,
        ada_year:           int,
        guideline_sources:  Optional[Dict] = None,
    ) -> str:
        """
        v7: 按指南源分组渲染证据。

        显示顺序遵循 SOURCE_DISPLAY 字典（Endocrinology → Nephrology → Cardiology），
        每个源独立 header，缺失时显式声明（不静默退化）。

        guideline_sources["other"] 为系统启动时声明的 active sources，
        用于在某个源完全无证据时区分"未启用"vs"启用但无命中"。
        """
        # 确定哪些源是"系统已启用"的（用于诊断 missing 信息）
        active_source_keys: set = set()
        if guideline_sources:
            other_decls = guideline_sources.get("other", []) or []
            # other_decls 是显示名形式，反向匹配
            _name_to_key = {v[0].lower(): k for k, v in SOURCE_DISPLAY.items()}
            for decl in other_decls:
                low = decl.lower()
                # ADA 不在 other 中，单独处理
                for name_low, sk in _name_to_key.items():
                    if name_low in low or low in name_low:
                        active_source_keys.add(sk)
                        break
            if guideline_sources.get("ada_years"):
                active_source_keys.add("ada")
        # 同时把已经检索到的源加入 active（即使 guideline_sources 为空）
        active_source_keys.update(k for k, v in evidence_by_source.items() if v)

        # ── Section 3 开头：证据层级声明（与 system_content 的 KNOWLEDGE PRIORITY 呼应）──
        parts: List[str] = [
            f"[Evidence Hierarchy]\n"
            f"Apply retrieved evidence in this priority order:\n"
            f"  1. Most recent guideline ≤ {ada_year} that directly addresses the clinical question "
            f"→ PRIMARY recommendation basis.\n"
            f"  2. Earlier guidelines (same source, prior years) → SUPPLEMENTARY support only "
            f"where the most recent source is silent.\n"
            f"  3. Cross-specialty guidelines (e.g., KDIGO for nephrology, ESC/ACC-AHA for cardiology) "
            f"→ apply for their respective comorbidity domains; do not override ADA DM drug decisions.\n"
            f"  4. If no retrieved chunk addresses a decision point → flag as evidence gap in 'reasoning'; "
            f"do not fill from parametric memory."
        ]
        # 按 SOURCE_DISPLAY 顺序渲染（保证心内科证据放在最后但不被遗漏）
        for sk, (display_name, specialty_label) in SOURCE_DISPLAY.items():
            text = evidence_by_source.get(sk, "")
            header = f"■ {display_name}  [{specialty_label}]  (≤ {ada_year})"

            if text and text.strip():
                parts.append(f"{header}\n{text.strip()}")
            elif sk in active_source_keys:
                parts.append(
                    f"{header}\n(no relevant chunks retrieved for {era_label}; "
                    f"index is built but no recommendation matched the phenotype queries)"
                )
            # 未启用的源完全跳过，避免噪音；但若所有源都未启用，下方 fallback

        # Fallback：如果完全没渲染任何指南证据
        if not parts:
            parts.append(
                f"■ Guideline Evidence: (no guideline indexes available for this run — "
                f"verify --sources includes ada,kdigo,esc,acc_aha and the index_dir is populated)"
            )

        # PubMed 单独一段（不与指南源混合）
        if pubmed_evidence and pubmed_evidence.strip():
            parts.append(f"■ PubMed RCT Evidence:\n{pubmed_evidence.strip()}")
        else:
            parts.append("■ PubMed RCT Evidence: (none retrieved)")

        return "\n\n".join(parts)

    @staticmethod
    def _format_mdt(mdt_summary: Optional[str], active_roles: List[str]) -> str:
        if mdt_summary is None:
            roles_str = "/".join(active_roles) if active_roles else "Endocrinology"
            return (f"(Single-specialty case [{roles_str}] — no inter-specialty conflicts "
                    f"detected; MDT skipped. Recommend complication drugs as appropriate "
                    f"based on Section 1 comorbidities.)")
        return mdt_summary

    @staticmethod
    def _build_drug_safety_notes(flat: Dict, ada_year: int) -> Dict[str, str]:
        notes: Dict[str, str] = {}
        egfr        = flat.get("egfr")
        has_aki     = flat.get("aki_suspected", False)
        vasopressor = flat.get("vasopressor_use", False)
        liver_bad   = flat.get("liver_dysfunction", False)
        has_hf      = flat.get("has_heart_failure", False)
        age         = flat.get("age") or 0

        def _n(*msgs): return "; ".join(m for m in msgs if m)

        m_biguanides = []
        if vasopressor:   m_biguanides.append("HOLD — hemodynamic instability/shock (lactic acidosis CI)")
        elif has_aki:     m_biguanides.append("HOLD — active AKI (resume after renal recovery)")
        elif egfr is not None:
            if egfr < 30: m_biguanides.append(f"CONTRAINDICATED — eGFR {egfr:.0f} <30")
            elif egfr < 45: m_biguanides.append(f"CAUTION — eGFR {egfr:.0f} (30-45): max 50% dose")
        if liver_bad: m_biguanides.append("AVOID — hepatic dysfunction >3×ULN")
        notes["Biguanides"] = _n(*m_biguanides)

        if ada_year >= 2016:
            m_sglt2 = []
            if vasopressor: m_sglt2.append("HOLD — hemodynamic instability (volume depletion, DKA risk)")
            elif has_aki:   m_sglt2.append("HOLD — active AKI (osmotic diuresis worsens tubular injury)")
            elif egfr is not None:
                if egfr < 20:  m_sglt2.append(f"AVOID for glycemic use — eGFR {egfr:.0f} <20")
                elif egfr < 45: m_sglt2.append(f"NOTE — eGFR {egfr:.0f}: glycosuric efficacy reduced; cardiorenal protection may persist")
            notes["SGLT2 inhibitors"] = _n(*m_sglt2)

        m_glp1 = []
        if vasopressor: m_glp1.append("HOLD — critical illness (GI side-effects impair enteral nutrition)")
        if egfr is not None and egfr < 15: m_glp1.append(f"CAUTION — eGFR {egfr:.0f} <15: limited PK data")
        notes["GLP-1RA"] = _n(*m_glp1)

        m_dpp4 = []
        if egfr is not None and egfr < 45:
            m_dpp4.append(f"eGFR {egfr:.0f}: prefer linagliptin (biliary excretion); sitagliptin requires dose reduction")
        notes["DPP-4 inhibitors"] = _n(*m_dpp4)

        m_su = []
        if egfr is not None and egfr < 45:
            m_su.append(f"eGFR {egfr:.0f}: glyburide CONTRAINDICATED; glipizide with dose reduction only")
        if age >= 75: m_su.append(f"Age {age:.0f}y: elevated hypoglycemia risk")
        notes["Sulfonylureas"] = _n(*m_su)

        m_tzd = []
        if has_hf:    m_tzd.append("CONTRAINDICATED — heart failure (FDA Black Box Warning)")
        if liver_bad: m_tzd.append("CONTRAINDICATED — hepatic dysfunction >2.5×ULN")
        notes["TZD"] = _n(*m_tzd)

        m_ins = []
        if egfr is not None and egfr < 30: m_ins.append(f"eGFR {egfr:.0f}: REDUCE dose (prolonged renal clearance)")
        if age >= 75: m_ins.append("Elderly: conservative dosing; frequent monitoring required")
        notes["Insulin"] = _n(*m_ins)

        if ada_year >= 2024:
            m_tirzep = []
            if egfr is not None and egfr < 15: m_tirzep.append(f"eGFR {egfr:.0f}: limited data in ESRD")
            notes["GIP/GLP-1 RA"] = _n(*m_tirzep)

        return notes

    def _format_contrastive_task(
        self,
        states:   List["ClinicalPhenotypeState"],
        is_t1:    bool,
        ada_year: int,
        flat:     Dict,
        temporal_drug_filter: bool = False,
    ) -> str:
        all_hints: set = set()
        for s in states:
            all_hints.update(h.lower() for h in s.contraindication_hints)

        if is_t1:
            era_classes: List[Tuple[str, str]] = [("Insulin", "Insulin (basal / bolus)")]
            safety_notes: Dict[str, str] = {}
        else:
            era_classes  = _get_era_drug_classes(ada_year)
            safety_notes = self._build_drug_safety_notes(flat, ada_year)

        era_lbl = era_label_for_year(ada_year)
        lines = [
            f"For each drug class available in {era_lbl} (≤ {ada_year}), decide: RECOMMEND / DEFER / EXCLUDE.",
            "Justify every EXCLUDE by citing the Section 2 state name or Section 3 evidence.", "",
        ]

        for class_key, display_label in era_classes:
            hint_flag = ""
            cls_tokens = set(re.split(r"[\s/_\-]+", class_key.lower())) - {""}
            for h in all_hints:
                if cls_tokens & (set(re.split(r"[\s/_\-]+", h)) - {""}):
                    hint_flag = "  [Section 2: contraindication flagged]"
                    break
            lines.append(f"  {display_label}{hint_flag}")
            note = safety_notes.get(class_key, "")
            if note:
                lines.append(f"    Patient safety: {note}")

        if ada_year <= 2010 and not is_t1:
            lines.append(
                "\n  Note (GLP-1RA at Era-1): only exenatide available before 2010; "
                "liraglutide Jan 2010 with minimal guideline evidence — flag if recommending."
            )

        # ── 并发症药评估范围声明（让 LLM 知道 taxonomy 边界，避免推荐无效药物）────
        if not is_t1:
            lines.append("")
            lines.append("Complication drug scope (use the category key EXACTLY as shown):")
            if temporal_drug_filter:
                _comp_src = DrugEraConfig.get_available_comp_drugs(ada_year)
            else:
                _comp_src = COMPLICATION_DRUGS
            for cat_key, drug_list in _comp_src.items():
                drug_str = ", ".join(drug_list)
                lines.append(f"  {cat_key}: {drug_str}")
            lines.append("Notable out-of-scope drugs (do NOT recommend — will not be scored):")
            for excl in COMPLICATION_DRUGS_EXCLUDED_NOTABLE:
                lines.append(f"  ⛔ {excl}")

        # v7: R8/R9 改为通用临床原则（去掉 [Mode B] 标签，因为 mode_b 已废弃）
        lines.extend([
            "",
            "Mandatory rules:",
            "  R1 [Exclusion]        Section 2 state confidence ≥0.90 with CI → must appear in excluded_classes.",
            "  R2 [Temporal]         Do NOT recommend any drug or cite any RCT postdating this patient's era.",
            "  R3 [Inpatient phase]  Apply phase-stratified oral agent policy:",
            "                          • acute_unstable → hold ALL oral agents; insulin only.",
            "                          • acute_stable → insulin primary; oral agents may be CONTINUED if no acute CI.",
            "                          • pre_discharge → resume home regimen per outpatient guidelines.",
            "  R4 [AKI]              Active AKI → hold metformin and SGLT2i; resume after renal recovery.",
            "  R5 [Contrast]         Iodinated contrast within 48h → hold metformin peri-procedurally.",
            "  R6 [Continuity-Output]  Drugs listed in Section 1 'Home DM/complication medications' that remain "
            "clinically appropriate MUST still appear in complication_drugs with "
            "\"continued_from_home\": true — this marks continuity rather than new initiation. "
            "Do NOT omit them from the JSON output; omission causes evaluation scoring errors.",
            "  R7 [Uncertainty]      If care unit or phase unknown → apply acute_unstable restrictions and flag.",
            "  R8 [Continuity]       Continue safe home medications unless acutely contraindicated.",
            "  R9 [Simplification]   For LOS ≤ 5 days, avoid introducing new drug classes that are not "
            "guideline-mandatory; prefer rapid-acting insulin for acute glycemic control.",
        ])


        return "\n".join(lines)
