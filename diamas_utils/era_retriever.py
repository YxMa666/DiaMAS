#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diamas_utils/era_retriever.py  (v7 — per-source evidence dict)
================================================================
时序感知证据检索器、患者特征扁平化、Era 推断

v7 关键变更（P0 修复）:
  - EraAwareEvidenceRetriever.retrieve() 返回 (Dict[str, str], str)
    第一个返回值是按指南源分组的字典：{"ada": ..., "kdigo": ..., "esc": ..., ...}
    供 prompt_builder._format_evidence() 按源独立 header 渲染，
    解决"心内科指南被静默合并到 ADA 块"的展示问题。

  - 新方法 retrieve_legacy_text() 返回扁平字符串（向后兼容老调用方，
    如 MDT 内部 fallback 与 diamas_metadata.ada_evidence 字段）。

  - _retrieve_guidelines_by_source() 复用 ada_index.search() (ADA) +
    registry.search_specialty() (其他源)，按 chunk 的 source_key 分组聚合。

包含：
  flatten_clinical_metrics()   — 嵌套患者 JSON → 扁平 dict
  infer_era_year()             — Era 标签 + 精确 ADA 年份推断
  EraAwareEvidenceRetriever    — 时序保真的多源指南 + PubMed 检索器
  decide_mdt_activation()      — MDT 激活决策
"""

from __future__ import annotations

import logging
import math
import re
from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from diamas_utils.clinical_phenotyper import ClinicalPhenotypeState
    from diamas_utils.pubmed_retriever import PubMedRetriever
    from diamas_utils.ada_guideline_processor import ADAGuidelineIndex

logger = logging.getLogger(__name__)

# ── 从 config 导入 ─────────────────────────────────────────────────────────────
from agents.config import DrugEraConfig, EraConfig, ClinicalCodeConfig, hemodynamic_instability

ERA_TO_ADA_YEAR = DrugEraConfig.ERA_TO_RAG_YEAR   # local alias (RAG year per era)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. NaN 安全工具（局部副本，避免跨模块循环）
# ═══════════════════════════════════════════════════════════════════════════════

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

def _is_potassium_suspect(k) -> bool:
    if k is None: return False
    return k > 7.0 or k < 2.0


# ═══════════════════════════════════════════════════════════════════════════════
# 2. flatten_clinical_metrics
# ═══════════════════════════════════════════════════════════════════════════════

def flatten_clinical_metrics(patient_data: Dict) -> Dict:
    """
    将嵌套患者 JSON 转换为 ClinicalPhenotyper 期望的扁平 dict。

    eGFR=0（ESRD）不能被 `or 90` 短路，所有字段均使用 _safe_float / _safe_bool。
    LVEF 在 dict literal 外预计算（避免 0.0 被 or 链短路）。
    """
    cm     = patient_data.get("clinical_metrics", {}) or {}
    dq     = patient_data.get("data_quality",     {}) or {}
    kidney = cm.get("kidney",            {}) or {}
    glucose= cm.get("glucose",           {}) or {}
    comp   = cm.get("dm_complications",  {}) or {}
    dm_cls = cm.get("dm_classification", {}) or {}
    elec   = cm.get("electrolytes",      {}) or {}
    cbc    = cm.get("cbc",               {}) or {}
    liver  = cm.get("liver",             {}) or {}

    _egfr_val = _safe_float(kidney.get("egfr"))
    _bmi_cm   = _safe_float(cm.get("bmi"))
    _bmi_val  = _bmi_cm if _bmi_cm is not None else _safe_float(patient_data.get("bmi"))
    _dm_type_raw = (
        _safe_str(dq.get("dm_type_cached"))
        or _safe_str(dm_cls.get("dm_type"))
        or "T2DM"
    )
    _lvef_raw = patient_data.get("lvef") or cm.get("lvef") or kidney.get("lvef")

    _vasopressor_flag = _safe_bool(comp.get("vasopressor_use"))
    _hemo = hemodynamic_instability(patient_data)   # 与评价端硬约束同一定义

    flat = {
        "age":    patient_data.get("age"),
        "gender": patient_data.get("gender", "unknown"),
        "bmi":    _bmi_val,

        "egfr":             _egfr_val,
        "creatinine":       _safe_float(kidney.get("creatinine")),
        "ckd_stage":        kidney.get("ckd_g_stage") or kidney.get("ckd_stage"),
        "urine_albumin_creatinine_ratio": _safe_float(kidney.get("uacr")),
        "aki_suspected":    (_safe_bool(kidney.get("aki_suspected")) or _safe_bool(dq.get("aki_suspected"))),
        "creatinine_peak":  _safe_float(kidney.get("creatinine_peak")),
        "aki_kdigo_stage":  kidney.get("aki_kdigo_stage"),

        "hba1c":            _safe_float(cm.get("hba1c")),
        "fasting_glucose":  _safe_float(glucose.get("fasting_glucose")),

        "bnp":              _safe_float(cm.get("bnp")),
        "ntprobnp":         _safe_float(cm.get("nt_probnp")),

        "has_cardiovascular":       _safe_bool(comp.get("cardiovascular")),
        "has_heart_failure":        _safe_bool(comp.get("heart_failure")),
        "has_ckd":                  (_safe_bool(comp.get("nephropathy")) or (_egfr_val is not None and _egfr_val < 60)),
        "hypoglycemia_history":     _safe_bool(comp.get("hypoglycemia")),

        "dm_type":   _dm_type_raw,
        "is_t1dm":   _dm_type_raw == "T1DM",

        "potassium":         _safe_float(elec.get("potassium")),
        "potassium_suspect": _is_potassium_suspect(_safe_float(elec.get("potassium"))),
        "hemoglobin":        _safe_float(cbc.get("hemoglobin")),

        "alt":             _safe_float(liver.get("alt")),
        "ast":             _safe_float(liver.get("ast")),
        "bilirubin_total": _safe_float(liver.get("bilirubin_total")),
        "liver_dysfunction": _safe_bool(liver.get("liver_dysfunction")),

        "on_corticosteroids":  _safe_bool(comp.get("on_corticosteroids")),
        "on_systemic_steroids": _safe_bool(comp.get("on_corticosteroids")),
        "vasopressor_use":     _vasopressor_flag,
        "hemodynamic_instability": _hemo,
        "on_loop_diuretic":    _safe_bool(comp.get("on_loop_diuretic")),

        "lvef":            _safe_float(_lvef_raw),
        "admission_year":  EraConfig.assigned_year(patient_data),   # 赋年（非移位后的 admission_time 年份）
    }
    return flat


# ═══════════════════════════════════════════════════════════════════════════════
# 3. infer_era_year
# ═══════════════════════════════════════════════════════════════════════════════

def infer_era_year(patient_data: dict, flat: dict) -> Tuple[str, int]:
    """
    推断 Era 标签和对应的 ADA 指南年份（精确到就诊年份，v3 时序保真核心）。

    Returns:
        (era_label: str, ada_year: int)
    """
    year = EraConfig.assigned_year(patient_data)   # 全局唯一赋年（MIMIC-III 2008 / eICU 出院年 / MIMIC-IV 按入院推算）
    if year is not None:
        return EraConfig.year_to_era(year), DrugEraConfig.get_ada_year_ceiling(year)
    if "eicu" in str(patient_data.get("database", "")).lower():
        return "Era-2", ERA_TO_ADA_YEAR["Era-2"]
    logger.warning(f"[infer_era_year] 无法确定 Era，默认 Era-3")
    return "Era-3", ERA_TO_ADA_YEAR["Era-3"]


# ═══════════════════════════════════════════════════════════════════════════════
# 4. EraAwareEvidenceRetriever
# ═══════════════════════════════════════════════════════════════════════════════

class EraAwareEvidenceRetriever:
    """
    时序保真的证据检索器（v7：按指南源分组返回）。

    C2 核心不变量：
      ADA 检索 max_year 始终 = era_year（绝对上限，不因 era_sensitivity 放宽）。
      非 ADA 来源通过 era_num 做硬性 pub_year 过滤。
    """

    def __init__(
        self,
        ada_index: "ADAGuidelineIndex",
        pubmed:    Optional["PubMedRetriever"],
        registry=None,   # Optional[ClinicalGuidelineRegistry]
    ):
        self.ada_index = ada_index
        self.pubmed    = pubmed
        self._registry = registry

    # ── 主接口 ────────────────────────────────────────────────────────────────

    def retrieve(
        self,
        states:       List["ClinicalPhenotypeState"],
        era_year:     int,
        flat:         Dict,
        active_roles: Optional[List[str]] = None,
    ) -> Tuple[Dict[str, str], str]:
        """
        v8: 返回按指南源分组的证据字典 + PubMed 文本。

        Args:
            active_roles: decide_mdt_activation() 返回的专科角色列表（中文，如 ["内分泌科","肾科"]）。
                          传入后检索仅覆盖激活专科，与 MDT 阈值(0.70)对齐，
                          避免在未激活专科上浪费 token。
                          None 时退化到原 states 路由（向后兼容）。

        Returns:
            (evidence_by_source, pubmed_evidence_text)
            evidence_by_source: {source_key: formatted_text}
              source_key ∈ {"ada", "aace", "ada_supplements",
                            "kdigo", "esc", "acc_aha", "chinese"}
              空源不包含在 dict 中。
        """
        evidence_by_source = self._retrieve_guidelines_by_source(states, era_year, active_roles)
        pubmed_text        = self._retrieve_pubmed(states, era_year, flat)

        # ── 整体诊断日志（INFO，always visible）──────────────────────────────
        logger.info(
            f"[retrieve] era_year={era_year}, "
            f"evidence_sources={list(evidence_by_source.keys())}, "
            f"pubmed_chars={len(pubmed_text)}, "
            f"active_roles={active_roles}"
        )
        if not evidence_by_source:
            _ada_n = len(self.ada_index.chunks) if self.ada_index else 0
            _reg_s = list(self._registry._source_indexes.keys()) if self._registry else []
            logger.warning(
                f"[retrieve] evidence_by_source is EMPTY. "
                f"Diagnostics: ada_chunks={_ada_n}, registry_sources={_reg_s}, era_year={era_year}. "
                "Possible causes: (1) index pkl not found / empty, "
                "(2) all chunks filtered by max_year ceiling, "
                "(3) --guidelines_dir not passed → registry=None."
            )

        return evidence_by_source, pubmed_text

    def retrieve_legacy_text(
        self,
        states:   List["ClinicalPhenotypeState"],
        era_year: int,
        flat:     Dict,
    ) -> Tuple[str, str]:
        """
        v6 兼容接口：返回扁平字符串 (guidelines_text, pubmed_text)。

        供老调用方（如 diamas_metadata.ada_evidence 字段、MDT 内部 fallback）使用。
        新代码应优先使用 retrieve()。
        """
        by_source, pubmed_text = self.retrieve(states, era_year, flat)
        guidelines_text = "\n\n".join(v for v in by_source.values() if v.strip())
        return guidelines_text, pubmed_text

    # ── 内部检索方法 ──────────────────────────────────────────────────────────

    # 角色名 → 专科标签（与 decide_mdt_activation / _ROLE_TO_SPECIALTY 保持一致）
    _ROLE_TO_SPECIALTY: Dict[str, str] = {
        "内分泌科": "endocrinology",
        "肾科":     "nephrology",
        "心内科":   "cardiology",
    }

    def _retrieve_guidelines_by_source(
        self,
        states:       List["ClinicalPhenotypeState"],
        era_year:     int,
        active_roles: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """
        按指南源 (source_key) 分组检索证据并格式化。（v8：接受 active_roles）

        逻辑：
          1. ADA：始终独立处理（沿用 _retrieve_ada_only 逻辑）
          2. 非 ADA 专科：
             - 若传入 active_roles，只检索激活专科（与 MDT 阈值 0.70 对齐）
             - 否则按 states.specialty_tags + confidence≥0.6 路由（向后兼容）
          3. 通过 registry.search_specialty() 拿 (chunk, score, source_key) 三元组，
             按 source_key 分组聚合，每组独立格式化。
        """
        by_source: Dict[str, str] = {}

        # ── 诊断：索引状态 INFO ────────────────────────────────────────────────
        _ada_n = len(self.ada_index.chunks) if self.ada_index else 0
        _reg_s = list(self._registry._source_indexes.keys()) if self._registry else []
        logger.info(
            f"[_retrieve_by_source] ada_chunks={_ada_n}, "
            f"registry_sources={_reg_s}, era_year={era_year}, "
            f"active_roles={active_roles}"
        )

        # ── 1. ADA：始终独立处理（保持原 _retrieve_ada_only 逻辑） ──────────
        ada_text = self._retrieve_ada_only(states, era_year)
        if ada_text:
            by_source["ada"] = ada_text
        else:
            logger.warning(
                f"[_retrieve_by_source] ADA evidence EMPTY "
                f"(ada_chunks={_ada_n}, era_year ceiling={era_year}). "
                "Check: index pkl exists & has chunks with pub_year<=era_year."
            )

        # ── 2. 其他源：通过 registry.search_specialty 拿到 source_key ───────
        if self._registry is None or not getattr(self._registry, "_source_indexes", None):
            logger.warning(
                "[_retrieve_by_source] registry=None or no source_indexes → ADA-only mode. "
                "Pass --guidelines_dir and --sources to enable specialty evidence."
            )
            return by_source

        # ── 确定目标专科集合 ──────────────────────────────────────────────────
        # 若 active_roles 已知，只检索 MDT 激活的专科（threshold=0.70 对齐）；
        # 否则退化到 states confidence≥0.6 的宽松路由（向后兼容）。
        if active_roles is not None:
            target_specialties: Optional[Set[str]] = (
                {self._ROLE_TO_SPECIALTY[r] for r in active_roles
                 if r in self._ROLE_TO_SPECIALTY}
                - {"endocrinology"}   # ADA 路径已覆盖内分泌科，不重复
            )
            logger.info(
                f"[_retrieve_by_source] active_roles → target_specialties={target_specialties}"
            )
        else:
            target_specialties = None  # 不限制，按 states 路由

        # Collect specialty → queries mapping for non-ADA sources.
        # Endocrinology is intentionally excluded: ADA (handled above) plus
        # aace/ada_supplements routed via registry already cover endocrinology.
        specialty_queries: Dict[str, List[str]] = {}
        for state in states:
            if state.confidence < 0.6:
                continue
            for tag in state.specialty_tags:
                if tag == "endocrinology":
                    continue   # ADA path handles endocrinology
                # 若传入 active_roles，跳过未激活的专科（精准路由）
                if target_specialties is not None and tag not in target_specialties:
                    continue
                specialty_queries.setdefault(tag, []).extend(state.guideline_queries[:3])

        if not specialty_queries:
            logger.info(
                f"[_retrieve_by_source] No non-endo specialty queries "
                f"(target_specialties={target_specialties}) → ADA-only."
            )
            return by_source

        # era_year → era_num 映射，传给非 ADA 来源做硬性 pub_year 过滤
        _era_num_val = 3 if era_year >= 2020 else (2 if era_year >= 2014 else 1)

        # 每个 source_key 累积 (chunk, score) 列表
        chunks_by_source: Dict[str, List[Tuple]] = {}
        seen_chunk_ids: Set[str] = set()

        for specialty, queries in specialty_queries.items():
            # 去重 query
            uniq_queries = list(dict.fromkeys(queries))[:5]
            for query in uniq_queries:
                expanded = self._expand_query(query)   # ← 同义词扩展
                try:
                    hits = self._registry.search_specialty(
                        query     = expanded,
                        specialty = specialty,
                        era_year  = era_year,
                        era_num   = _era_num_val,
                        top_k     = 5,
                    )
                    for chunk, score, sk in hits:
                        cid = getattr(chunk, "chunk_id", None) or id(chunk)
                        if cid in seen_chunk_ids:
                            continue
                        seen_chunk_ids.add(cid)
                        chunks_by_source.setdefault(sk, []).append((chunk, score))
                except Exception as e:
                    logger.debug(f"按专科检索失败 [{specialty} / {query[:40]}]: {e}")

        # ── 3. 格式化每个 source 的 top chunks ──────────────────────────────
        for sk, chunks in chunks_by_source.items():
            if sk == "ada":
                # ADA 已通过独立路径处理，跳过避免重复
                continue
            if not chunks:
                continue
            chunks.sort(key=lambda x: x[1], reverse=True)
            lines: List[str] = []
            for chunk, _score in chunks[:6]:
                source_name = getattr(chunk, "source_name", sk.upper())
                level       = getattr(chunk, "recommendation_level", None)
                pub_year    = getattr(chunk, "pub_year", None)
                section     = getattr(chunk, "section_title", "") or ""
                text_body   = getattr(chunk, "text", "") or ""

                level_str  = f" · Level {level}" if level else ""
                year_str   = f" · {pub_year}" if pub_year else ""
                header     = f"[{source_name}{year_str} · {section[:50]}{level_str}]"
                body_str   = text_body.replace("\n", " ").strip()[:350]
                lines.append(f"{header}\n{body_str}")
            by_source[sk] = "\n\n".join(lines)

        return by_source

    # ── 同义词扩展（替代 BGE-M3 的跨表述匹配，零依赖、完全可复现）──────────────
    # 覆盖场景：eGFR vs "estimated glomerular filtration rate"、
    #           kidney vs renal、HbA1c vs A1C 等指南常见表述差异。
    _SYNONYM_MAP: Dict[str, List[str]] = {
        # 肾脏
        "kidney":                   ["renal", "nephro"],
        "renal":                    ["kidney"],
        "eGFR":                     ["estimated glomerular filtration rate", "GFR"],
        "CKD":                      ["chronic kidney disease", "diabetic kidney disease", "DKD"],
        "proteinuria":              ["albuminuria", "UACR", "microalbuminuria"],
        # 心血管
        "heart failure":            ["HF", "cardiac failure", "HFrEF", "HFpEF"],
        "myocardial infarction":    ["MI", "heart attack", "ACS", "acute coronary syndrome"],
        "blood pressure":           ["BP", "hypertension", "systolic"],
        "cardiovascular":           ["CV", "ASCVD", "CVD", "cardio"],
        # 血糖 / 用药
        "HbA1c":                    ["glycated hemoglobin", "A1C", "hemoglobin A1c"],
        "SGLT2":                    ["sodium-glucose cotransporter", "gliflozin",
                                     "empagliflozin", "dapagliflozin", "canagliflozin"],
        "GLP-1":                    ["glucagon-like peptide", "liraglutide",
                                     "semaglutide", "dulaglutide"],
        "insulin":                  ["basal insulin", "bolus insulin", "insulin regimen"],
        "metformin":                ["biguanide"],
        "sulfonylurea":             ["glipizide", "glimepiride", "glyburide", "SU"],
        # 通用
        "hypoglycemia":             ["low blood glucose", "hypoglycaemia", "glucose <70"],
        "hyperglycemia":            ["high blood glucose", "hyperglycaemia"],
        "type 2 diabetes":          ["T2DM", "T2D"],
        "type 1 diabetes":          ["T1DM", "T1D"],
    }

    @classmethod
    def _expand_query(cls, query: str) -> str:
        """
        对单条查询做同义词扩展，返回扩展后的合并字符串（BM25 TF-IDF 友好格式）。

        设计原则：
          - 只做追加，不替换原 query，保证原始词命中不受影响
          - 不去重（BM25 对词频敏感，重复出现的词权重更高是合理的）
          - 大小写不敏感匹配，追加时保留原始大小写（指南术语区分大小写）
        """
        parts = [query]
        q_lower = query.lower()
        for term, syns in cls._SYNONYM_MAP.items():
            if term.lower() in q_lower:
                parts.extend(syns)
        return " ".join(parts)

    def _retrieve_ada_only(self, states, era_year: int) -> str:
        """ADA 索引独立检索（沿用 v6 实现，保证 ADA 块的格式与原版一致）。"""
        if not self.ada_index or not self.ada_index.chunks:
            return ""
        all_chunks, seen_ids = [], set()
        for state in states:
            for query in state.guideline_queries:
                expanded = self._expand_query(query)   # ← 同义词扩展
                chunks = self.ada_index.search(
                    query=expanded, top_k=3,
                    max_year=era_year,           # C2 时序保真：始终严格上限
                    recommendation_only=True,
                )
                for chunk, score in chunks:
                    if chunk.chunk_id not in seen_ids:
                        # Bug-4: 近度权重，与 search_for_drug_recommendation._year_w 一致
                        # dist=0 → weight=1.3，每差1年衰减0.15，最低0.5
                        dist   = abs(chunk.ada_year - era_year)
                        year_w = max(0.5, 1.3 - dist * 0.15)
                        all_chunks.append((chunk, score * year_w))
                        seen_ids.add(chunk.chunk_id)
        all_chunks.sort(key=lambda x: x[1], reverse=True)
        return self.ada_index.format_for_prompt(all_chunks[:10], max_per_year=3, max_chars=400)

    def _retrieve_pubmed(self, states, era_year: int, flat: Dict) -> str:
        """PubMed 检索（沿用 v6 实现）。"""
        if self.pubmed is None:
            return ""
        queries = []
        for state in states:
            if state.era_sensitivity:
                queries.extend(state.guideline_queries[:1])
        if not queries:
            egfr = flat.get("egfr", 90)
            if egfr and egfr < 45:
                queries.append("SGLT2 inhibitor CKD renal protection RCT")
            if flat.get("has_cardiovascular"):
                queries.append("GLP-1 SGLT2 cardiovascular outcomes trial")
            queries.append("T2DM pharmacotherapy RCT meta-analysis")
        try:
            results = []
            for q in queries[:4]:
                try:
                    arts = self.pubmed.search_by_query(query=q, year_end=era_year, top_k=1)
                    results.extend(arts)
                except Exception as eq:
                    logger.debug(f"PubMed 单条查询失败 [{q}]: {eq}")
            return self.pubmed.format_articles(results[:4], max_abstract_chars=200)
        except Exception as e:
            logger.warning(f"PubMed 检索失败: {e}")
            return ""

    # ── v6 向后兼容包装 ──────────────────────────────────────────────────────

    def _retrieve_guidelines(self, states, era_year: int) -> str:
        """[v6 deprecated, kept for backward compat] 扁平字符串形式的指南证据。"""
        by_source = self._retrieve_guidelines_by_source(states, era_year)
        return "\n\n".join(v for v in by_source.values() if v.strip())

    def _retrieve_supplementary(self, states, era_year: int) -> str:
        """[v6 deprecated, kept for backward compat] 仅返回非 ADA 部分。"""
        by_source = self._retrieve_guidelines_by_source(states, era_year)
        return "\n\n".join(v for k, v in by_source.items() if k != "ada" and v.strip())


# ═══════════════════════════════════════════════════════════════════════════════
# 5. decide_mdt_activation
# ═══════════════════════════════════════════════════════════════════════════════

def decide_mdt_activation(
    states: List["ClinicalPhenotypeState"],
) -> Tuple[bool, List[str]]:
    """
    根据病理生理状态决定是否激活 MDT 以及激活哪些专科。

    激活逻辑（冲突检测门槛，[FIX-P3]）：
      - 始终包含内分泌科
      - nephrology 或 cardiology 状态 confidence ≥ 0.70 → 候选激活
      - 候选激活后：检查跨专科 contraindication_hints 是否有交集
        或 implication 中是否存在对立语义词
      - 无真实冲突 → 不触发 MDT（避免浪费 token）
      - 有真实冲突 → 触发 MDT 仲裁
    """
    SPECIALTY_TO_ROLE = {
        "endocrinology": "内分泌科",
        "nephrology":    "肾科",
        "cardiology":    "心内科",
    }
    CONFLICT_DRUGS = ClinicalCodeConfig.CONFLICT_DRUGS

    active: Dict[str, List] = {"endocrinology": []}
    for state in states:
        for tag in state.specialty_tags:
            if tag == "nephrology" and state.confidence >= 0.70:
                active.setdefault("nephrology", []).append(state)
            elif tag == "cardiology" and state.confidence >= 0.70:
                active.setdefault("cardiology", []).append(state)
            elif tag == "endocrinology":
                active["endocrinology"].append(state)

    active_roles = [SPECIALTY_TO_ROLE[s] for s in active if s in SPECIALTY_TO_ROLE]
    non_endo = [s for s in active if s != "endocrinology"]

    if not non_endo:
        return False, active_roles

    # [v5] 并发症用药需求：只要心内科或肾科 state 存在（confidence ≥ 0.70），
    # 即激活 MDT 以获取专科并发症药物推荐（他汀/β-blocker/ACEi/MRA 等）。
    return True, active_roles
