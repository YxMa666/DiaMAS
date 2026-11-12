#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_diamas.py — DiaMAS recommendation programs (Single and Collaborative).

For every admission JSON in --patient_dir the engine
  1. flattens the structured record and infers clinical states (deterministic/probabilistic rules),
  2. assigns the admission year and the matching guideline year,
  3. decides which specialist agents are eligible (nephrology, cardiology; endocrinology always),
  4. retrieves year-bounded guideline evidence routed by specialty,
  5. (Collaborative only) runs specialist opinions, semantic conflict detection, revision and adjudication,
  6. builds the prompt and calls the shared LLM for the final synthesis,
  7. parses the plan and applies the critical drug-interaction audit.

One JSON result per admission is written to <output_dir>/<model>_<temperature>/patient_<id>.json.
Recorded prescriptions and post-index outcomes are never passed to the programs.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Tuple

from tqdm import tqdm

# ── 日志 ──────────────────────────────────────────────────────────────────────
Path("loggings").mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler("loggings/diamas_inference.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ── 内部模块 ──────────────────────────────────────────────────────────────────
from agents.llm_client import DeepseekClient
from agents.config import ModelConfig, DrugEraConfig

from diamas_utils.clinical_phenotyper import ClinicalPhenotypeState, ClinicalPhenotyper
from diamas_utils.era_retriever import (
    flatten_clinical_metrics, infer_era_year,
    EraAwareEvidenceRetriever, decide_mdt_activation,
    _safe_str, _safe_float, _safe_bool,
)
from diamas_utils.prompt_builder import DiamasPromptBuilder, era_label_for_year
from diamas_utils.ada_guideline_processor import (
    ADAGuidelineIndex, ADAGuidelineProcessor,
    ClinicalGuidelineRegistry, build_all_indexes,   # [v8] 多源指南 + 混合检索
)


# ═══════════════════════════════════════════════════════════════════════════════
# DiamasRecommendationEngine — 核心推荐引擎
# ═══════════════════════════════════════════════════════════════════════════════

class DiamasRecommendationEngine:
    """
    Dia-MAS 核心推荐引擎（PCAR 框架，8步流程）

    Step 1: flatten_clinical_metrics()    → 特征扁平化
    Step 2: ClinicalPhenotyper.infer()    → 三阶段表型推断
    Step 3: infer_era_year()              → Era 标签 + 精确 ADA 年份
    Step 4: EraAwareEvidenceRetriever     → 时序保真证据检索
    Step 5: decide_mdt_activation() + VirtualMDT.conduct_mdt()  → 条件性 MDT
    Step 6: DiamasPromptBuilder.build()   → 5段式 Prompt
    Step 7: LLM call → _parse_response()  → 结构化推荐结果 + DDI 审计
    """

    def __init__(
        self,
        model_name:         str,
        temperature:        float  = 0.0,
        max_tokens:         int    = 4096,
        ada_index_path:     Path   = Path("data/indexes/ada_index.pkl"),
        ada_guidelines_dir: Path   = Path("data/guidelines/ada"),
        use_mdt:            bool   = True,
        mdt_temperature:    float  = 0.3,
        guidelines_dir:     Optional[Path] = None,
        index_dir:          Optional[Path] = None,
        guideline_sources:  Optional[List[str]] = None,
        judge_model:          str  = "qwen3-32b",
        max_debate_rounds:    int  = 2,
    ):
        self.model_name  = model_name
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self.use_mdt     = use_mdt
        self._guideline_sources = guideline_sources or []   # [v6] 用于动态构建指南来源声明

        self.llm_client    = DeepseekClient()
        self.phenotyper    = ClinicalPhenotyper(llm_client=None)
        self.prompt_builder = DiamasPromptBuilder()

        # ── 指南索引加载 ──────────────────────────────────────────────────────
        registry  = None
        ada_index = None

        if guidelines_dir and Path(guidelines_dir).exists():
            # try:
            from diamas_utils.ada_guideline_processor import build_all_indexes
            _idx_dir = Path(index_dir) if index_dir else Path(guidelines_dir).parent / "indexes"
            registry = build_all_indexes(
                guidelines_dir=Path(guidelines_dir),
                index_dir=_idx_dir,
                sources=guideline_sources,
                force=False,
            )
            if registry.ada_index and registry.ada_index.chunks:
                ada_index = registry.ada_index
                logger.info(f"多源指南注册表: sources={registry.active_sources}, ada_chunks={len(ada_index.chunks)}")
            else:
                logger.warning("注册表 ADA 索引为空，降级到独立加载")
            # except Exception as e:
            #     logger.warning(f"多源注册表构建失败，降级到 ADA-only: {e}")
            #     registry = None

        if ada_index is None:
            ada_index = self._load_ada_index(ada_index_path, ada_guidelines_dir)

        self._registry = registry

        self.retriever = EraAwareEvidenceRetriever(ada_index=ada_index, pubmed=None, registry=registry)

        # ── MDT 懒加载 ────────────────────────────────────────────────────────
        self._judge_model = judge_model
        self._max_debate_rounds = max_debate_rounds
        self._mdt = None
        if use_mdt:
            try:
                from diamas_utils.virtual_mdt_stage import VirtualMDT
                self._mdt = VirtualMDT(
                    llm_client=self.llm_client, model_name=model_name,
                    temperature=mdt_temperature, registry=registry, judge_model=judge_model,max_debate_rounds=max_debate_rounds,
                )
                logger.info(f"VirtualMDT 初始化成功 (registry={'多源' if registry else 'ADA-only'})")
            except ImportError:
                logger.warning("virtual_mdt_stage 导入失败，MDT 不可用")

        logger.info(
            f"DiamasRecommendationEngine: model={model_name}, "
            f"ada_chunks={len(ada_index.chunks) if ada_index else 0}, "
            f"mdt={self._mdt is not None}, sources={guideline_sources}"
        )

    # ── 主推荐接口 ────────────────────────────────────────────────────────────

    def recommend(self, patient_data: Dict) -> Dict:
        """为单个患者生成推荐（8步 PCAR 流程）。"""
        t0         = time.time()
        subject_id = patient_data.get("subject_id", 0)
        hadm_id    = patient_data.get("hadm_id", 0)
        patient_id = f"{subject_id}_{hadm_id}"

        try:
            # Step 1: 特征扁平化
            flat = flatten_clinical_metrics(patient_data)
            patient_summary = self._build_patient_summary(patient_data, flat)

            # Step 2: 表型推断
            states = self.phenotyper.infer(flat, patient_data)
            logger.debug(f"{patient_id}: {len(states)} states: {[s.name for s in states]}")

            # Step 3: Era 推断
            era_label, ada_year = infer_era_year(patient_data, flat)

            # Step 4: 条件性 MDT 激活决策（先于证据检索）
            # ─────────────────────────────────────────────────────────────────
            # 必须先决定激活哪些专科，再精准路由专科指南检索：
            #   旧流程：retrieve(threshold=0.6) → 白费 token 检索未激活专科
            #            → EvidenceGap 事后补丁（Step 4b）
            #   新流程：先 decide_mdt_activation(threshold=0.70) 确定 active_roles
            #            → retrieve(active_roles=...) 按激活专科精准路由
            #            → EvidenceGap fallback 仍保留，应对"激活但无时序匹配"场景
            should_mdt, active_roles = decide_mdt_activation(states)

            # Step 5: 证据检索 (v8: 按激活专科精准路由 + PubMed 文本)
            evidence_by_source, pubmed_evidence = self.retriever.retrieve(
                states, ada_year, flat, active_roles=active_roles
            )

            # Step 5b: EvidenceGap fallback — 激活专科未被主检索命中时的补充检索
            # 场景：retrieve() 路由到激活专科，但 evidence_by_source 里没有该专科
            # owned 的任何源（可能是路由/检索侧的覆盖缺口）。
            # [v18b P6] **era 天花板始终保留**（era_year=ada_year）。原实现用
            # era_year=None 放开天花板，是全系统唯一会主动破坏时间防火墙的路径；
            # v18 冒烟实测 Era-1 触发率 39%，注入了 KDIGO 2012+ 内容。
            # 现在：某时代若确无该专科指南（Era-1 肾科即如此，KDIGO 首版 2012），
            # fallback 返回空 → 不注入 → 忠实表现"该时代无证据"。
            _SPECIALTY_OWNED_SOURCES = {
                "nephrology": ["kdigo", "ada_supplements"],
                "cardiology": ["esc", "acc_aha"],
            }
            if should_mdt and self._registry is not None:
                _all_queries = [
                    q for s in states
                    for q in s.guideline_queries[:2]
                    if s.confidence >= 0.6
                ]
                for role in active_roles:
                    sp = self.retriever._ROLE_TO_SPECIALTY.get(role)
                    if not sp or sp == "endocrinology":
                        continue
                    owned = _SPECIALTY_OWNED_SOURCES.get(sp, [])
                    already_present = any(k in evidence_by_source for k in owned)
                    if already_present:
                        continue
                    # Specialty is activated but has no era-appropriate evidence.
                    # Try without era_num filter so the MDT agent receives at least
                    # some guideline context (labeled with temporal warning in prompt).
                    logger.warning(
                        f"[EvidenceGap] {role} ({sp}) activated but no "
                        f"era-appropriate guideline found for {era_label} "
                        f"(ada_year={ada_year}). Attempting specialty-routed fallback "
                        f"(era ceiling retained)."
                    )
                    try:
                        fallback_ev = self._registry.format_evidence_for_specialty(
                            specialty    = sp,
                            queries      = _all_queries[:5],
                            # [v18b P6] 保留 era 天花板（原为 None = 取消天花板）。
                            # v18 冒烟实测：P0-year 之后本 fallback 从"从未触发"变成
                            # Era-1 触发率 65/168=39%，而 era_year=None 是全系统唯一
                            # 会主动破坏时间防火墙的路径。
                            # Era-1 无肾科指南是历史事实（KDIGO 首版 2012、ESC 首版 2007），
                            # 正确表现是"该时代无证据"，而非拿 2012 年内容凑数。
                            era_year     = ada_year,
                            top_k        = 4,
                            use_chinese  = False,
                        )
                        if fallback_ev:
                            evidence_by_source[f"{sp}_fallback"] = (
                                f"[Specialty-routed retrieval for {sp} "
                                f"(era ceiling {ada_year} enforced).]\n\n"
                                + fallback_ev
                            )
                            logger.info(
                                f"[EvidenceGap] Fallback evidence injected for {sp} "
                                f"({len(fallback_ev)} chars)"
                            )
                    except Exception as _eg:
                        logger.debug(f"[EvidenceGap] fallback retrieval failed [{sp}]: {_eg}")

            # legacy 字符串供 MDT 内部 fallback 与 metadata 字段使用
            ada_evidence_legacy = "\n\n".join(v for v in evidence_by_source.values() if v.strip())
            mdt_result = mdt_summary = None
            if should_mdt and self._mdt is not None:
                try:
                    phase_a_stub = {
                        "patient_id": patient_id,
                        "patient_summary": patient_summary,
                        "diamas_metadata": {
                            "patho_states":  [asdict(s) for s in states],
                            "ada_evidence":  ada_evidence_legacy,
                            "era_label":     era_label,
                            "ada_year_used": ada_year,
                        },
                    }
                    mdt_result  = self._mdt.conduct_mdt(
                        phase_a_result  = phase_a_stub,
                        flat            = flat,           # pre-computed at Step 1
                        patient_data    = patient_data,   # for lipids, dm_complications, etc.
                        active_experts  = active_roles,
                    )
                    mdt_summary = self._format_mdt_for_prompt(mdt_result)
                    logger.debug(f"{patient_id}: MDT consensus={mdt_result.consensus_level}")
                except Exception as e:
                    logger.warning(f"{patient_id}: MDT 失败，退化为单路径: {e}")

            # Step 7: Prompt 构建 + LLM 推理 (v7: 传入 evidence_by_source dict)
            system_content, user_content = self.prompt_builder.build(
                patient_summary=patient_summary, states=states,
                ada_evidence=ada_evidence_legacy,                # 向后兼容字段
                pubmed_evidence=pubmed_evidence,
                era_label=era_label, ada_year=ada_year,
                mdt_summary=mdt_summary, active_roles=active_roles,
                flat=flat,
                guideline_sources=self._collect_guideline_sources(evidence_by_source),
                evidence_by_source=evidence_by_source,           # v7: 主路径
            )

            response = self.llm_client.chat(
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user",   "content": user_content},
                ],
                model=self.model_name, temperature=self.temperature,
                max_tokens=self.max_tokens, stream=False, deep_thinking=False,
            )
            response_text = self.llm_client.get_response_text(response) or ""
            usage = response.get("usage", {})

            # Step 8: 解析输出 (v7: 单模式 + complication_drugs)
            recommended_drugs, complication_drugs, reasoning, excluded = self._parse_response(response_text)

            # Bug-6: 合并并发症药到 recommended_drugs，与旧版 baseline 格式兼容（评估流水线只读该字段）
            # MDT 激活时用仲裁后的 mdt_final_complication_drugs，否则用单模型 complication_drugs
            # [v18 P2] 取消"MDT 激活时整体替换"。原逻辑令 A2 的并发症药直接取自专科仲裁，
            # 而 _SPECIALTY_COMPLICATION_CATS 把 Antiplatelet / Lipid_Statins /
            # BetaBlockers 独占挂在 cardiology 名下 ⇒ 心内科未激活的患者结构性开不出这些药。
            # arm_asymmetry(b)：心内科缺席层 Lipid_Statins Δ=−75.4pp(test)/−75.9pp(seed)，
            # BetaBlockers −24.4/−18.7；Antiplatelet 两层均 ≈−20pp。
            # 改为两臂统一由共享合成 LLM 产出；MDT 结论经 mdt_summary 注入 prompt 仍然生效，
            # mdt_final_complication_drugs 继续落盘，可事后算「MDT 建议采纳率」。
            _comp_source = complication_drugs or []
            _seen_names = {
                str(d.get("drug", "") if isinstance(d, dict) else d).strip().lower()
                for d in recommended_drugs
            }
            for _cd in _comp_source:
                _name = str(_cd.get("drug", "") if isinstance(_cd, dict) else _cd).strip().lower()
                if _name and _name not in _seen_names:
                    recommended_drugs.append(_cd)
                    _seen_names.add(_name)

            # Bug-3: DDI 阻断（critical 级别直接移除冲突药）
            try:
                from evaluation.safety_checker import DrugInteractionAuditor, DDIWarning
                from dataclasses import asdict as _asdict
                _ddi_auditor = DrugInteractionAuditor(normalizer=getattr(self, "_normalizer", None))
                recommended_drugs, _ddi_warns = _ddi_auditor.filter_and_warn(recommended_drugs, flat)
                _ddi_warn_dicts = [_asdict(w) for w in _ddi_warns]
            except Exception as _ddi_e:
                logger.warning(f"DDI audit error (non-fatal): {_ddi_e}")
                _ddi_warn_dicts = []

            return {
                "patient_id":    patient_id,
                "subject_id":    subject_id,
                "hadm_id":       hadm_id,
                "timestamp":     datetime.now().isoformat(),
                "model_name":    self.model_name,
                "patient_summary": patient_summary,
                "actual_medications": {
                    "prior":    patient_data.get("prior_medications", []),
                    "discharge": patient_data.get("discharge_medications", []),
                },
                "recommended_drugs":  recommended_drugs,
                "complication_drugs": complication_drugs,        # v7 新增
                "reasoning":     reasoning,
                "raw_output":    response_text,
                "inference_time":     time.time() - t0,
                "prompt_tokens":      usage.get("prompt_tokens", 0),
                "completion_tokens":  usage.get("completion_tokens", 0),
                "total_tokens":       usage.get("total_tokens", 0),
                "estimated_cost":     self._estimate_cost(usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)),
                "success":       len(recommended_drugs) > 0 or len(complication_drugs) > 0,
                "error_message": None if (recommended_drugs or complication_drugs) else "parse failed",
                "diamas_metadata": {
                    "phenotype_states":    [asdict(s) for s in states],
                    "ddi_warnings": _ddi_warn_dicts,
                    "era_label":           era_label,
                    "ada_year_used":       ada_year,
                    "ada_evidence":        ada_evidence_legacy,
                    "evidence_by_source":  evidence_by_source,
                    "pubmed_evidence":     pubmed_evidence,
                    "mdt_activated":       mdt_result is not None,
                    "activated_specialties": active_roles,
                    "mdt_consensus_level": mdt_result.consensus_level if mdt_result else "n/a",
                    "mdt_conflicts":       ([c.__dict__ for c in mdt_result.conflicts] if mdt_result else []),
                    "mdt_conflicts_pre_debate": (
                        [c.__dict__ for c in mdt_result.conflicts_pre_debate] if mdt_result else []),
                    "mdt_conflicts_resolved": (
                        len(mdt_result.conflicts_pre_debate) - len(mdt_result.conflicts)
                        if mdt_result else 0),
                    "mdt_final_complication_drugs": (mdt_result.final_complication_drugs if mdt_result else []),
                    "excluded_classes":    excluded,
                    "patho_states":        [{"state": s.name, "confidence": s.confidence, "implication": s.implication} for s in states],

                    # ── §63-P6 新增：可审计性与配置可复原性 ──────────────────
                    "mdt_expert_opinions": (
                        [
                            {
                                "role":               op.role,
                                "recommended_drugs":  op.recommended_drugs,
                                "complication_drugs": op.complication_drugs,
                                "key_concerns":       op.key_concerns,
                                "reasoning":          op.reasoning,
                                # raw_output 不落盘：体积大且 §66 已证自由文本不可复现
                            }
                            for op in mdt_result.expert_opinions
                        ] if mdt_result else []
                    ),
                    "mdt_debate_rounds":   (mdt_result.debate_rounds if mdt_result else 0),
                    "mdt_expert_weights":  (mdt_result.expert_weights if mdt_result else {}),
                    "judge_model":         (self._judge_model if self._mdt else None),
                    "mdt_max_debate_rounds": (self._max_debate_rounds if self._mdt else None),
                    "evidence_sources_effective": sorted((evidence_by_source or {}).keys()),
                    "phenotyper_stage3_enabled":  self.phenotyper._llm is not None,
                },
            }

        except Exception as e:
            import traceback
            logger.error(f"推荐失败 {patient_id}: {e}\n{traceback.format_exc()}")
            return {
                "patient_id": patient_id, "subject_id": subject_id,
                "hadm_id": hadm_id, "timestamp": datetime.now().isoformat(),
                "model_name": self.model_name, "patient_summary": "",
                "actual_medications": {}, "recommended_drugs": [],
                "reasoning": "", "raw_output": "",
                "inference_time": time.time() - t0,
                "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                "estimated_cost": 0.0, "success": False, "error_message": str(e),
                "diamas_metadata": {"dual_mode_raw": None},
            }

    # ── 辅助方法 ──────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_response(response_text: str) -> Tuple[List[Dict], List[Dict], str, List[Dict]]:
        """
        v7: 解析 LLM 单模式输出 → (recommended_drugs, complication_drugs, reasoning, excluded_classes)。

        兼容旧版 mode_a 嵌套格式（自动平铺到顶层）。
        complication_drugs 字段缺失时返回空列表。
        """
        if not response_text:
            return [], [], "", []

        clean = re.sub(r"<think>.*?</think>", "", response_text, flags=re.DOTALL).strip()

        last_close = clean.rfind("}")
        if last_close == -1:
            return [], [], clean, []
        depth, start = 0, -1
        for i in range(last_close, -1, -1):
            if clean[i] == "}": depth += 1
            elif clean[i] == "{":
                depth -= 1
                if depth == 0: start = i; break
        if start == -1:
            return [], [], clean, []

        try:
            data = json.loads(clean[start: last_close + 1])
        except json.JSONDecodeError:
            try:
                import json_repair
                data = json_repair.repair_json(clean[start: last_close + 1], return_objects=True)
            except Exception:
                return [], [], clean, []

        if not isinstance(data, dict):
            return [], [], clean, []

        # 兼容老 mode_a 包裹结构
        mode_a = data.get("mode_a") if isinstance(data.get("mode_a"), dict) else {}

        raw_dm    = data.get("recommended_drugs")  or mode_a.get("recommended_drugs")  or []
        raw_comp  = data.get("complication_drugs") or mode_a.get("complication_drugs") or []
        excluded  = data.get("excluded_classes")   or mode_a.get("excluded_classes")   or []
        reasoning = str(data.get("reasoning") or mode_a.get("reasoning") or "")

        def _normalize_drug_list(items) -> List[Dict]:
            out: List[Dict] = []
            if not isinstance(items, list):
                return out
            for item in items:
                if isinstance(item, dict):
                    drug = (item.get("drug") or item.get("drug_name") or item.get("name") or "").strip()
                    if drug:
                        out.append({
                            "drug":     drug,
                            "category": item.get("category", ""),
                            "reason":   item.get("reason", ""),
                            "dosage":   item.get("dosage") or item.get("dose_note") or "",
                        })
                elif isinstance(item, str) and item.strip():
                    out.append({"drug": item.strip(), "category": "", "reason": "", "dosage": ""})
            return out

        dm_drugs   = _normalize_drug_list(raw_dm)
        comp_drugs = _normalize_drug_list(raw_comp)
        excluded_list = excluded if isinstance(excluded, list) else []

        return dm_drugs, comp_drugs, reasoning, excluded_list

    @staticmethod
    def _extract_dual_mode_raw(response_text: str) -> Optional[Dict]:
        """
        [v7 deprecated] 双模式解析器。
        v7 移除 mode_b/cii_divergence 后保留为兼容性 stub（始终返回 None）。
        """
        return None

    @staticmethod
    def _estimate_cost(prompt_tokens: int, completion_tokens: int) -> float:
        return round((prompt_tokens * 0.27 + completion_tokens * 1.10) / 1_000_000, 6)

    def _format_mdt_for_prompt(self, mdt_result) -> str:
        """
        v7: 完整注入各专科 DM + 并发症药 + 关键顾虑 + 最终并发症方案。

        旧版 (v6) bug: 仅显示 op.recommended_drugs (DM 降糖药)，导致心内科/肾科的
        并发症药 (op.complication_drugs) 被静默丢弃，让主推断 LLM 看到的 MDT 摘要
        变成 "[心内科] → (no specific drugs)"。本方法修复该缺陷。
        """
        if mdt_result is None:
            return ""

        lines: List[str] = [
            f"[Virtual MDT — consensus: {mdt_result.consensus_level}]",
            f"  Active specialties: {', '.join(op.role for op in mdt_result.expert_opinions)}",
            "",
        ]

        # ── 各专家详细意见 ────────────────────────────────────────────────────
        for op in mdt_result.expert_opinions:
            lines.append(f"  [{op.role}]")

            # DM 降糖药（内分泌科主导，心内科/肾科可能为空，标"(none)"以示故意而非疏漏）
            if op.recommended_drugs:
                dm_str = ', '.join(op.recommended_drugs[:8])
            else:
                dm_str = "(no DM-drug recommendation from this specialty)"
            lines.append(f"    DM drugs:           {dm_str}")

            # 并发症药（心内科/肾科主导）
            comp_drugs = getattr(op, "complication_drugs", []) or []
            if comp_drugs:
                comp_items = []
                for cd in comp_drugs[:12]:
                    if not isinstance(cd, dict):
                        continue
                    name = str(cd.get("drug", "")).strip()
                    if not name:
                        continue
                    cat  = str(cd.get("category", "")).strip()
                    indi = str(cd.get("indication", "")).strip()
                    dose = str(cd.get("dose") or cd.get("dosage", "")).strip()
                    parts = [name]
                    if cat:  parts.append(f"[{cat}]")
                    detail_bits = []
                    if indi: detail_bits.append(indi)
                    if dose: detail_bits.append(dose)
                    if detail_bits:
                        parts.append(f"({'; '.join(detail_bits)})")
                    comp_items.append(" ".join(parts))
                comp_str = "; ".join(comp_items) if comp_items else "(no complication drugs)"
            else:
                comp_str = "(no complication drugs)"
            lines.append(f"    Complication drugs: {comp_str}")

            # 关键顾虑
            if op.key_concerns:
                concerns_str = '; '.join(op.key_concerns[:5])
                lines.append(f"    Key concerns:       {concerns_str}")
            lines.append("")

        # ── 检测到的冲突 ────────────────────────────────────────────────────
        if mdt_result.conflicts:
            lines.append(f"  Conflicts detected ({len(mdt_result.conflicts)}):")
            for c in mdt_result.conflicts[:6]:
                lines.append(
                    f"    [{c.severity.upper()}] {c.drug}: {c.expert_a} vs {c.expert_b} "
                    f"— {c.reason_b[:80]}"
                )
            lines.append("")

        # ── MDT 裁决最终方案 ────────────────────────────────────────────────
        final_dm = mdt_result.final_drugs or []
        final_dm_str = ', '.join(d.get('drug', '') for d in final_dm[:6]) or "(see individual opinions)"
        lines.append(f"  Final MDT plan — DM drugs:           {final_dm_str}")

        final_comp = getattr(mdt_result, "final_complication_drugs", []) or []
        if final_comp:
            final_comp_items = []
            for cd in final_comp[:12]:
                if not isinstance(cd, dict): continue
                name = str(cd.get("drug", "")).strip()
                if not name: continue
                cat  = str(cd.get("category", "")).strip()
                dose = str(cd.get("dose") or cd.get("dosage", "")).strip()
                bits = [name]
                if cat:  bits.append(f"[{cat}]")
                if dose: bits.append(f"({dose})")
                final_comp_items.append(" ".join(bits))
            final_comp_str = "; ".join(final_comp_items) if final_comp_items else "(none)"
        else:
            final_comp_str = "(none — verify if comorbidities require complication drugs)"
        lines.append(f"  Final MDT plan — Complication drugs: {final_comp_str}")

        lines.append("")
        lines.append(
            "(For any CRITICAL conflict, the final plan must explicitly state which "
            "position is adopted and why. The complication drugs above represent the "
            "MDT's specialty-driven recommendations; integrate them into your final "
            "prescription unless you identify a contraindication not yet considered. "
            # [v18c P3a] MDT 方案只覆盖**已激活专科**的 scope；未激活专科对应的
            # 药物类别不会出现在上面的方案里。v18b 实测合成阶段对该方案强锚定
            # （MDT 采纳率均值 96%），导致未覆盖类别被连带遗漏。
            "NOTE: the MDT plan above reflects only the ACTIVATED specialties' scope — "
            "categories outside that scope are simply absent from it, not ruled out. "
            "Independently verify that every guideline-indicated complication category "
            "for this patient is covered in your final prescription.)"
        )
        return "\n".join(lines)

    def _collect_guideline_sources(self, evidence_by_source: Optional[Dict] = None) -> Dict:
        """
        动态构建指南来源声明，与实际 --sources 参数保持一致。

        [v6 修复] 原代码在 registry 无索引时 fallback 到硬编码全量列表，
        导致系统提示声明的来源与实际 RAG 检索来源不一致（消融实验失效）。
        现在 fallback 使用 self._guideline_sources（来自 --sources 参数）。
        """
        ada_years: List[int] = []
        if self.retriever.ada_index and self.retriever.ada_index.by_year:
            ada_years = sorted(self.retriever.ada_index.by_year.keys())

        # ── [v18 P1a] 按患者生成来源声明 ──────────────────────────────────
        # 原实现按**启动配置**生成，对每个患者恒定：A1 声明 1 源、A2 声明 5 源，
        # 构成一处覆盖 100% 患者（含未激活者）的臂间非对称。
        # arm_asymmetry(b) 实测未激活组 ACEi Δ=−9.7pp [−13.3,−5.9]，下行合计 −19.4pp；
        # prompt_render_probe(p1) 逐字符证实这是未激活组两臂唯一的 build() 输入差异，
        # 且按患者生成后 25/25 例逐字符相同、A1 侧渲染 0/40 例改变。
        # 副作用（预期）：Section 3 不再为"已声明但本患者无命中"的源渲染占位块；
        # 该信息可由 activated_specialties × SPECIALTY_SOURCE_ROUTING 与
        # evidence_sources_effective 的差集事后完整还原，无需进入 prompt。
        _SRC_DISPLAY_V18 = {
            "kdigo":             "KDIGO CKD Guidelines",
            "esc":               "ESC CVD in Diabetes Guidelines",
            "acc_aha":           "ACC/AHA Cholesterol & HF Guidelines",
            "aace":              "AACE Diabetes Guidelines",
            "ada_supplements":   "ADA Supplement Guidelines",
            "endocrine_society": "Endocrine Society Clinical Practice Guidelines",
            "chinese":           "Chinese Clinical Guidelines & Expert Consensus",
        }
        if evidence_by_source is not None:
            other = [
                _SRC_DISPLAY_V18.get(str(k).strip().lower(), str(k).strip().upper())
                for k, v in sorted(evidence_by_source.items())
                if v and str(k).strip().lower() != "ada"
            ]
            return {"ada_years": ada_years, "other": other}

        registry = getattr(self, "_registry", None)
        if registry and getattr(registry, "_source_indexes", None):
            # 从真实索引中读取来源名称
            # [v18] 过滤 ada：原代码未过滤，导致 A2 的声明里 ADA 出现两次
            other = [getattr(idx, "source_name", sk.upper())
                     for sk, idx in registry._source_indexes.items()
                     if str(sk).strip().lower() != "ada"]
        elif self._guideline_sources:
            # 用实际传入的 --sources 参数生成声明（无索引时的准确 fallback）
            _source_display = {
                "kdigo":             "KDIGO CKD Guidelines",
                "esc":               "ESC CVD in Diabetes Guidelines",
                "acc_aha":           "ACC/AHA Cholesterol & HF Guidelines",
                "aace":              "AACE Diabetes Guidelines",
                "ada_supplements":   "ADA Supplement Guidelines",
                "endocrine_society": "Endocrine Society Clinical Practice Guidelines",  # [v8]
                "chinese":           "Chinese Clinical Guidelines & Expert Consensus",   # [v8]
            }
            other = [_source_display.get(s.strip().lower(), s.strip().upper())
                     for s in self._guideline_sources
                     if s.strip().lower() not in ("ada",)]
        else:
            other = []   # 纯 ADA-only 模式，不声明其他来源

        return {"ada_years": ada_years, "other": other}

    def _load_ada_index(self, index_path: Path, guidelines_dir: Path) -> ADAGuidelineIndex:
        import pickle
        index_path = Path(index_path)
        if index_path.exists():
            try:
                class _SafeUnpickler(pickle.Unpickler):
                    def find_class(self, module, name):
                        if module in ("__main__", "diamas_utils.ada_guideline_processor"):
                            module = "ada_guideline_processor"
                        return super().find_class(module, name)
                with open(index_path, "rb") as f:
                    idx = _SafeUnpickler(f).load()
                logger.info(f"ADA 索引加载: {idx.get_stats()}")
                return idx
            except Exception as e:
                logger.warning(f"ADA 索引加载失败: {e}")
        guidelines_dir = Path(guidelines_dir)
        if guidelines_dir.exists() and list(guidelines_dir.glob("*.md")):
            logger.info("从 ADA Markdown 重建索引...")
            proc = ADAGuidelineProcessor(guidelines_dir=guidelines_dir, index_output=index_path)
            return proc.build_index(force_rebuild=True)
        logger.warning(f"未找到 ADA 指南文件（{guidelines_dir}），无 ADA 证据运行")
        return ADAGuidelineIndex()

    def _build_patient_summary(self, patient_data: Dict, flat: Dict) -> str:
        """构建 Section 1 患者摘要（含 demographics / 检验 / 用药 / 住院背景）。"""
        lines: List[str] = []
        cm    = patient_data.get("clinical_metrics", {}) or {}
        dq    = patient_data.get("data_quality",     {}) or {}
        comp  = cm.get("dm_complications", {}) or {}
        elec  = cm.get("electrolytes",     {}) or {}
        liver = cm.get("liver",            {}) or {}
        gluc  = cm.get("glucose",          {}) or {}

        age    = patient_data.get("age", "?")
        gender = {"M": "Male", "F": "Female"}.get(str(patient_data.get("gender", "")).upper(), "Unknown")
        bmi    = flat.get("bmi");  los = _safe_float(patient_data.get("los_days"))
        demo   = f"Age: {age} y  Sex: {gender}"
        if bmi is not None: demo += f"  BMI: {bmi:.1f} kg/m²"
        if los is not None: demo += f"  LOS: {los:.1f} d"
        lines.append(demo)

        dm_cls  = cm.get("dm_classification", {}) or {}
        dm_type = dq.get("dm_type_cached") or dm_cls.get("dm_type", "T2DM")
        lines.append(f"DM type: {dm_type}  (confidence: {dq.get('dm_confidence_cached', 'low')})")

        severity = _safe_str(patient_data.get("severity_tier", "")).strip()
        if severity == "High":   setting, basis = "ICU-level care", "severity_tier=High"
        elif severity in ("Low", "Medium"): setting, basis = "general ward care", f"severity_tier={severity}"
        elif flat.get("vasopressor_use"):   setting, basis = "ICU-level care", "vasopressor detected"
        else:
            sofa = _safe_float(patient_data.get("lab_sofa_proxy"))
            setting = "ICU-level care" if (sofa and sofa >= 4) else ("general ward care" if sofa else "inpatient (care unit unknown)")
            basis   = f"Lab-SOFA={sofa:.1f}" if sofa else "no severity signal; applying conservative protocol"
        lines.append(f"Care setting: {setting}  [{basis}]")

        diags = patient_data.get("all_diagnoses", [])[:6]
        if diags:
            lines.append("Diagnoses:")
            for d in diags:
                if not isinstance(d, dict): continue
                def _s(v):
                    return v if isinstance(v, str) else ""

                icd_code = _s(d.get("icd_code")).strip()
                desc     = (_s(d.get("snomed_name")) or _s(d.get("description"))).strip()
                if "|" in desc:
                    parts = [p.strip() for p in desc.split("|") if p.strip()]
                    desc = " — ".join(parts[-2:]) if len(parts) >= 2 else parts[-1]
                prim = " [primary]" if d.get("is_primary") else ""
                lines.append(f"  [ICD-{d.get('icd_version', '?')}: {icd_code}]{prim} {desc}")

        g_parts = []
        if flat.get("hba1c"):           g_parts.append(f"HbA1c {flat['hba1c']:.1f}%")
        fg = flat.get("fasting_glucose") or _safe_float(gluc.get("fasting_glucose"))
        if fg:                          g_parts.append(f"FBG {fg:.0f} mg/dL")
        gm = _safe_float(gluc.get("mean_glucose"))
        if gm:                          g_parts.append(f"Mean BG {gm:.0f} mg/dL")
        if g_parts:                     lines.append("Glycemic: " + "  |  ".join(g_parts))

        egfr = flat.get("egfr"); cr = flat.get("creatinine"); aki = flat.get("aki_suspected")
        if egfr is not None:
            r = f"Renal: eGFR {egfr:.0f}"
            if flat.get("ckd_stage"):           r += f"  CKD {flat['ckd_stage']}"
            if cr:                              r += f"  SCr {cr:.2f}"
            if flat.get("urine_albumin_creatinine_ratio"): r += f"  UACR {flat['urine_albumin_creatinine_ratio']:.0f}"
            if aki:                             r += "  ⚠ AKI"
            lines.append(r)

        k = flat.get("potassium"); na = _safe_float(elec.get("sodium"))
        if k or na:
            e = "Electrolytes:"
            if k:  flag = " ⚠HIGH" if k > 5.5 else (" ⚠LOW" if k < 3.5 else ""); e += f"  K⁺ {k:.1f}{flag}"
            if na: e += f"  Na⁺ {na:.0f}"
            lines.append(e)

        alt = flat.get("alt"); ast = flat.get("ast")
        if alt or ast:
            l = "Liver:";
            if alt: l += f"  ALT {alt:.0f}"
            if ast: l += f"  AST {ast:.0f}"
            if flat.get("liver_dysfunction"): l += "  ⚠ hepatic dysfunction"
            lines.append(l)

        bnp = flat.get("bnp") or flat.get("ntprobnp")
        if bnp: lines.append(f"BNP/NT-proBNP: {bnp:.0f} pg/mL")

        # ── Home medications：区分 DM/并发症相关 vs 其他 ──────────────────────
        # 问题：prior_medications 包含所有入院前用药（如 Labetalol、镇静剂），
        # 统一标注为 "Home DM medications" 会误导 LLM 将非 DM 药当作 DM 用药继续。
        # 修复：按关键词分组，DM/并发症药走第一行（R8/R6 评估），其他走第二行（仅临床背景）。
        _DM_COMP_KW = {
            # DM 药
            "insulin", "metformin", "glipizide", "glimepiride", "glyburide", "glibenclamide",
            "sitagliptin", "linagliptin", "saxagliptin", "alogliptin", "vildagliptin",
            "empagliflozin", "dapagliflozin", "canagliflozin", "ertugliflozin",
            "liraglutide", "semaglutide", "exenatide", "dulaglutide", "tirzepatide",
            "pioglitazone", "rosiglitazone", "repaglinide", "nateglinide",
            "acarbose", "miglitol", "colesevelam", "bromocriptine",
            # 并发症药（在 COMPLICATION_DRUGS 分类表内）
            "atorvastatin", "rosuvastatin", "simvastatin", "pravastatin", "lovastatin",
            "fluvastatin", "pitavastatin", "ezetimibe", "fenofibrate",
            "lisinopril", "enalapril", "ramipril", "benazepril", "captopril", "fosinopril",
            "losartan", "valsartan", "telmisartan", "irbesartan", "candesartan",
            "metoprolol", "bisoprolol", "carvedilol",
            "furosemide", "bumetanide", "torsemide", "spironolactone",
            "aspirin", "clopidogrel", "ticagrelor", "prasugrel",
            "finerenone", "eplerenone",
            "pregabalin", "gabapentin", "duloxetine", "amitriptyline",
        }

        def _is_dm_or_comp(name: str) -> bool:
            n = name.lower()
            return any(kw in n for kw in _DM_COMP_KW)

        home_dm_meds = patient_data.get("home_dm_medications") or []
        prior_meds   = patient_data.get("prior_medications")   or []

        if home_dm_meds:
            # 显式 home_dm_medications 字段（一般由提取器精确过滤）
            meds_str = "; ".join(str(m) for m in home_dm_meds[:6] if m)
            lines.append(f"Home DM medications: {meds_str or '(not available)'}")
        elif prior_meds:
            dm_comp = [m for m in prior_meds if _is_dm_or_comp(str(m))][:6]
            other   = [m for m in prior_meds if not _is_dm_or_comp(str(m))][:4]
            if dm_comp:
                lines.append(
                    "Home DM/complication medications: "
                    + "; ".join(str(m) for m in dm_comp)
                )
            if other:
                lines.append(
                    "Other home medications (continuity reference, not scored): "
                    + "; ".join(str(m) for m in other)
                )
            if not dm_comp and not other:
                lines.append("Home DM medications: (not available)")

        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# DiamasInferencePipeline — 多线程批量推理流水线
# ═══════════════════════════════════════════════════════════════════════════════

class DiamasInferencePipeline:
    """
    Dia-MAS 批量推理流水线（每个入院一个结果文件，支持断点续跑）。
    """

    def __init__(
        self,
        patient_json_dir:    Path,
        output_dir:          Path,
        model_name:          str,
        temperature:         float = 0.0,
        max_tokens:          int   = 8000,
        resume:              bool  = True,
        num_workers:         int   = 2,
        ada_index_path:      Path  = Path("data/indexes/ada_index.pkl"),
        ada_guidelines_dir:  Path  = Path("data/guidelines/ada"),
        use_mdt:             bool  = True,
        mdt_temperature:     float = 0.3,
        guidelines_dir:      Optional[Path]      = None,
        index_dir:           Optional[Path]      = None,
        guideline_sources:   Optional[List[str]] = None,
        cohort_manifest_csv: Optional[Path] = None,
        judge_model:         str  = "qwen3-32b",
        max_debate_rounds:   int  = 2,
    ):
        self.patient_dir       = Path(patient_json_dir)
        self.output_dir        = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.resume            = resume
        self.num_workers       = num_workers
        self.max_tokens        = max_tokens

        # cohort manifest
        self._manifest: Dict[str, Dict] = {}
        if cohort_manifest_csv and Path(cohort_manifest_csv).exists():
            try:
                import pandas as pd
                df = pd.read_csv(cohort_manifest_csv)
                for _, row in df.iterrows():
                    pid = str(row.get("patient_id", ""))
                    if pid: self._manifest[pid] = row.to_dict()
                logger.info(f"cohort_manifest 加载: {len(self._manifest)} 条")
            except Exception as e:
                logger.warning(f"cohort_manifest 加载失败: {e}")

        # 引擎初始化
        self.engine = DiamasRecommendationEngine(
            model_name=model_name, temperature=temperature, max_tokens=max_tokens,
            ada_index_path=ada_index_path, ada_guidelines_dir=ada_guidelines_dir,
            use_mdt=use_mdt, mdt_temperature=mdt_temperature,
            guidelines_dir=guidelines_dir, index_dir=index_dir, guideline_sources=guideline_sources,
            judge_model=judge_model, max_debate_rounds=max_debate_rounds,
        )

    def _preflight_model_check(self) -> None:
        """[E3-FAILSAFE] 启动前确认目标模型可用；不可用则立即停机。

        动机：2026-08-26 API 下线 deepseek-v3.2，整批任务跑了数小时全部失败，
        并写出 8,612 个残缺结果文件。预检把这个代价降到一次请求。

        故意不吞异常：模型不可用是致命配置错误，必须响亮失败（PREREG §0）。
        """
        from agents.llm_client import DeepseekClient
        from agents.config import BASELINE_MODELS

        # [E3-PREFLIGHT-FIX] model_name / judge 不在 pipeline 上，而在 engine：
        #   engine.model_name (126) / engine._judge_model (178) / engine._mdt (None ⇒ 不用 judge)
        eng = getattr(self, "engine", None)
        if eng is None:
            raise RuntimeError(f"[E3-FAILSAFE] pipeline 无 engine，无法预检")
        primary = getattr(eng, "model_name", None)
        if not primary:
            raise RuntimeError(f"[E3-FAILSAFE] engine 无 model_name 属性，无法预检")
        names = [primary]
        if getattr(eng, "_mdt", None) is not None:
            jm = getattr(eng, "_judge_model", None)
            if jm and jm != primary:
                names.append(jm)

        client = DeepseekClient()
        for name in names:
            cfg = BASELINE_MODELS.get(name, {"model_str": name})   # models outside the study list: name as given
            resp = client.chat(
                messages=[{"role": "user", "content": "ping"}],
                model=cfg["model_str"], temperature=0.0, max_tokens=8,
                stream=False, deep_thinking=False,
            )
            if isinstance(resp, dict) and resp.get("error"):
                raise RuntimeError(
                    f"[E3-FAILSAFE] 模型 {name!r} 预检失败，拒绝启动（避免整队列失败 + "
                    f"污染结果目录）。status={resp.get('status_code')} "
                    f"msg={str(resp.get('message'))[:300]}")
            logger.info(f"[E3-FAILSAFE] 模型预检通过: {name}")

    def run(self) -> Dict[str, int]:
        """运行流水线，返回统计信息。"""
        self._preflight_model_check()                       # [E3-FAILSAFE]

        patient_files = sorted(self.patient_dir.glob("*.json"))
        if not patient_files:
            logger.warning(f"无患者文件: {self.patient_dir}")
            return {"total": 0, "success": 0, "skipped": 0, "failed": 0}

        total    = len(patient_files)
        success  = skipped = failed = 0
        lock     = Lock()

        # canonical_id 去重集合：防止同一病人以不同文件名被双重处理
        # （符号链接、命名变体：12345_67890.json vs patient_12345_67890.json 等）
        _seen_canonical: set = set()

        def process_one(pf: Path):
            nonlocal success, skipped, failed

            patient_id = pf.stem
            out_path   = self.output_dir / f"patient_{patient_id}.json"

            # ── Resume logic（文件级）────────────────────────────────────────
            if self.resume and out_path.exists():
                with lock:
                    skipped += 1
                return

            # ── 读取患者数据 ──────────────────────────────────────────────────
            try:
                with open(pf, encoding="utf-8") as f:
                    patient_data = json.load(f)
            except Exception as e:
                logger.error(f"读取失败 {pf.name}: {e}")
                with lock:
                    failed += 1
                return

            # ── Canonical-ID 去重（内容级）───────────────────────────────────
            sid = str(patient_data.get("subject_id", "")).strip()
            hid = str(patient_data.get("hadm_id",   "")).strip()
            canonical = f"{sid}_{hid}" if (sid and hid) else pf.stem
            with lock:
                if canonical in _seen_canonical:
                    logger.warning(
                        f"[Dedup] 跳过重复病人 canonical_id={canonical} "
                        f"file={pf.name}，该 subject_id+hadm_id 已被处理"
                    )
                    skipped += 1
                    return
                _seen_canonical.add(canonical)

            # ── 推理 ─────────────────────────────────────────────────────────
            try:
                manifest_entry = self._manifest.get(patient_id, {})
                if manifest_entry:
                    for k, v in manifest_entry.items():
                        if k not in patient_data:
                            patient_data[k] = v

                result = self.engine.recommend(patient_data)
                result["patient_id"] = patient_id

                # Bug-2 / [E3-FAILSAFE]: 任何失败都不写文件，保持"未处理"状态，
                # 下次 --resume 自动重跑。
                #
                # 原判据只覆盖 error_message == "parse failed"，而 API 层失败
                # （404 Invalid model / 401 / 限流耗尽重试）走的是 recommend() 的
                # 兜底分支，返回 success=False 且 error_message 为异常文本
                # ⇒ 会被写成残缺结果文件，并被 resume 当作已完成永久顶替。
                # 2026-08-26 事故实测污染 8,612 个文件。
                _failed_msg = str(result.get("error_message") or "")
                if (result.get("success") is False) or (_failed_msg == "parse failed"):
                    _kind = ("parse" if _failed_msg == "parse failed"
                             else "api" if _failed_msg else "unknown")
                    logger.warning(
                        f"[skip-save] {_kind} failure for {patient_id} — "
                        f"file NOT written, will retry on next run | "
                        f"{_failed_msg[:160]}"
                    )
                    with lock:
                        failed += 1
                    return

                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2, default=str)

                with lock:
                    success += 1

            except Exception as e:
                logger.error(f"处理失败 {pf.name}: {e}")
                with lock:
                    failed += 1

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {executor.submit(process_one, pf): pf for pf in patient_files}
            with tqdm(total=total, desc="Dia-MAS inference") as pbar:
                for future in as_completed(futures):
                    future.result()
                    pbar.update(1)

        logger.info(f"完成: total={total}, success={success}, skipped={skipped}, failed={failed}")
        return {"total": total, "success": success, "skipped": skipped, "failed": failed}


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

PROGRAMS = {
    # Single: endocrinology agent only, ADA evidence only
    "single":        {"use_mdt": False, "sources": ["ada"]},
    # Collaborative: adds eligible nephrology and cardiology agents with their own guideline sources
    "collaborative": {"use_mdt": True,  "sources": ["ada", "ada_supplements", "kdigo", "esc", "acc_aha"]},
}


def main():
    import argparse

    ap = argparse.ArgumentParser(description="DiaMAS Single / Collaborative recommendation programs")
    ap.add_argument("--program", required=True, choices=sorted(PROGRAMS))
    ap.add_argument("--patient_dir", required=True, help="directory of admission JSON files")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--model", default="qwen3-32b")
    ap.add_argument("--judge_model", default="qwen3-32b", help="conflict-detection model (Collaborative)")
    ap.add_argument("--temperature", type=float, default=0.0, help="final synthesis temperature")
    ap.add_argument("--max_tokens", type=int, default=8000)
    ap.add_argument("--mdt_temperature", type=float, default=0.3, help="specialist opinion temperature")
    ap.add_argument("--debate_rounds", type=int, default=2)
    ap.add_argument("--guidelines_dir", default="data/guidelines_md")
    ap.add_argument("--index_dir", default="data/guidelines_indexes")
    ap.add_argument("--cohort_manifest", default=None, help="optional CSV whose fields fill keys missing from patient JSON")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--no_resume", action="store_true")
    args = ap.parse_args()

    prog = PROGRAMS[args.program]
    index_dir = Path(args.index_dir)
    pipeline = DiamasInferencePipeline(
        patient_json_dir    = Path(args.patient_dir),
        output_dir          = Path(args.output_dir) / f"{args.model}_{args.temperature}",
        model_name          = args.model,
        temperature         = args.temperature,
        max_tokens          = args.max_tokens,
        resume              = not args.no_resume,
        num_workers         = args.num_workers,
        ada_index_path      = index_dir / "ada_index.pkl",
        ada_guidelines_dir  = Path(args.guidelines_dir) / "ada",
        use_mdt             = prog["use_mdt"],
        mdt_temperature     = args.mdt_temperature,
        guidelines_dir      = Path(args.guidelines_dir),
        index_dir           = index_dir,
        guideline_sources   = prog["sources"],
        cohort_manifest_csv = Path(args.cohort_manifest) if args.cohort_manifest else None,
        judge_model         = args.judge_model,
        max_debate_rounds   = args.debate_rounds,
    )
    print(pipeline.run())


if __name__ == "__main__":
    main()
