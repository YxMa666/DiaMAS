#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gas_v5_domain_eval.py  (GAS v5 — Stage 3 计算引擎)
====================================================
域级双轴 GAS。把"规则级 recall"升级为"决策域级 Concordance × 处方 Appropriateness"。

两轴(均为 per-patient [0,1],可直接套用现有 PSM / Cliff's d / E-value 验证):

  Concordance(指南覆盖 / recall / START 型)
      = Σ(域权重 · [域被满足]) / Σ(激活域权重)
      域权重 = 该域内适用 RECOMMEND 规则的最高 grade
      域被满足 = 处方含该域任一推荐药物类别(域内/跨规则替代在此自动化解)
      → 病的越重激活域越多,但分母是"域数(≤十几个)"而非"规则数(数百)",
        严重度膨胀被结构性压制。

  Appropriateness(处方适当性 / precision / STOPP 型)
      = 1 − |commission 药物| / max(1, |在范围内的处方药物|)
      commission = 处方里命中了某条适用 AVOID 规则点名的禁忌药
      → 与短期不良结局(低血糖/AKI/禁忌药致害)机制直连。

  ALLOW(条件许可)第三极性:
      不进 Concordance 分子,也不算 commission;且【覆盖】同药同条件的 AVOID
      (解决 metformin-in-HF 这类新旧指南冲突:新指南 ALLOW 压过旧指南 AVOID)。

  GAS_v5_F = 2·C·A/(C+A)  (需单标量时;论文建议两轴并列报)

接口与 ose_calculator._family_dedup_eval 对齐,可直接替换:
  规则 dict 需含:id, domain, polarity, condition(cm,pd,yr)->bool,
                  any_of_cats[], exclude_drugs[], grade, (可选)scorable/disabled
  cat_to_drug_names: {category: [drug_name_lower, ...]}  (来自 DrugNormalizer)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple
from agents.config import (
    GAS_EQUAL_WEIGHTS, DrugEraConfig, DIABETES_MEDICATIONS, COMPLICATION_DRUGS,
)  # [TEMPORAL-CONSISTENCY-V1]
logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# [GAS-CANON] 评估类别规范化 —— 与 evaluation/metrics_calculator.canon_eval_category
# 同源口径,**对称**施加于规则侧(any_of_cats)与处方侧(cat_to_drug_names)。
#
# 修的是什么:规则库只含 Insulins_Basal(108 条)/ Insulins_Rapid(95 条),没有
# Insulins_Unspecified;_domain_satisfied 的 Unspecified 桥接是单向的 —— 开
# "insulin(未指明)"者两类规则全命中,开具体子类者只命中自己那一类。该分辨率
# 因此只作用于处方更精细的一方,从未对称施加,测的是处方措辞而非临床质量。
#
# 开关:
#   GAS_CANON=0             → 全关。行为与打补丁前逐字节等价(复现旧 gas_f 用)。
#   GAS_CANON_DROP_RESCUE=1 → 额外把 Hypoglycemia_Rescue 移出 GAS scope(sensitivity)。
#     默认 **不** 排除:F1 侧排除是因为出院 GT 含 PRN 协议医嘱(Type-B);GAS 侧
#     rescue 规则是 ADA 指南推荐(Type-A)。per-rule 消融尚未判此题,不预先下结论。
# ═══════════════════════════════════════════════════════════════════════════════
import os as _os

GAS_CANON_ENABLED = True
GAS_CANON_INSULIN_COLLAPSE = True
GAS_CANON_INSULIN_LABEL = "Insulin"
GAS_CANON_DROP_CATS = frozenset()


def gas_canon_category(cat):
    """幂等;GAS_CANON=0 时为恒等映射(仅过滤空值)。"""
    if not cat:
        return None
    if not GAS_CANON_ENABLED:
        return cat
    if cat in GAS_CANON_DROP_CATS:
        return None
    if GAS_CANON_INSULIN_COLLAPSE and str(cat).startswith("Insulins_"):
        return GAS_CANON_INSULIN_LABEL
    return cat


def _canon_cat_set(cats):
    """规则侧:any_of_cats -> 规范化类别集合(不修改原规则 dict)。"""
    return {c for c in (gas_canon_category(x) for x in (cats or [])) if c}


def _canon_cat_to_drug_names(c2d):
    """处方侧:{category: [drug]} -> 规范化后合并同类(保序去重)。"""
    if not GAS_CANON_ENABLED or not c2d:
        return c2d
    out = {}
    for cat, names in c2d.items():
        c = gas_canon_category(cat)
        if not c:
            continue
        bucket = out.setdefault(c, [])
        for n in (names or []):
            if n not in bucket:
                bucket.append(n)
    return out


# [TEMPORAL-CONSISTENCY-V1] Encounter-year availability helpers.
_FORMULARY_BY_CATEGORY = {
    **DIABETES_MEDICATIONS,
    **COMPLICATION_DRUGS,
}


def _filter_prescription_by_availability(c2d, admission_year):
    """Best-effort defensive filter for direct domain_eval callers."""
    if not c2d:
        return c2d
    out = {}
    for cat, names in c2d.items():
        keep = [
            n for n in (names or [])
            if DrugEraConfig.drug_available_at(str(n), admission_year)
        ]
        if keep:
            out[cat] = keep
    return out


def _available_rule_categories(cats, admission_year):
    """Keep only categories with >=1 encounter-era-available formulary member.

    This is a drug-availability mask, not an indication-specific evidence-year
    adjudicator. Indication-specific timing remains in rule condition(..., yr).
    """
    out = set()
    for cat in (cats or []):
        formulary = _FORMULARY_BY_CATEGORY.get(cat)
        if formulary is None:
            out.add(cat)  # fail-open for unknown taxonomy; separate audit catches it
            continue
        if any(DrugEraConfig.drug_available_at(d, admission_year) for d in formulary):
            out.add(cat)
    return out


_CANON_CHECKED = False


def _canon_selfcheck_once():
    """漂移哨兵:与 metrics_calculator 在**共享子集**上必须一致,否则 F1/GAS 口径分叉。

    有意分歧(不告警):Hypoglycemia_Rescue —— metrics 排除,GAS 默认保留。
    """
    global _CANON_CHECKED
    if _CANON_CHECKED:
        return
    _CANON_CHECKED = True
    if not GAS_CANON_ENABLED:
        logger.warning("[GAS-CANON] 已通过 GAS_CANON=0 关闭 —— gas_f 回到打补丁前口径。")
        return
    try:
        from evaluation.metrics_calculator import canon_eval_category as _m_canon
    except Exception:
        return   # 独立单测/裸跑:不阻塞
    probes = ["Insulins_Basal", "Insulins_Rapid", "Insulins_Unspecified",
              "SGLT2i", "GLP-1RA", "Biguanides", "Lipid_Statins"]
    drift = [p for p in probes if _m_canon(p) != gas_canon_category(p)]
    if drift:
        logger.error("[GAS-CANON] 与 metrics_calculator 口径漂移:%s "
                     "(F1 与 GAS 将用不同类别定义,论文口径不一致,请先对齐)", drift)
    else:
        logger.info("[GAS-CANON] 口径一致性 OK(共享子集);rescue 分歧为有意设计:"
                    "metrics=排除 / GAS=%s",
                    "排除" if GAS_CANON_DROP_CATS else "保留")

# 域分组(用于可选的 glycemic vs complication 拆分报告,向后兼容 Path-DM/Path-C)
DOMAIN_GROUP: Dict[str, str] = {
    "glycemic_first_line": "glycemic", "glycemic_intensification": "glycemic",
    "glycemic_insulin": "glycemic",
    "cardiorenal_protection": "complication", "ascvd_antiplatelet": "complication",
    "lipid_management": "complication", "bp_raas": "complication", "bp_other": "complication",
    "hf_management": "complication", "ckd_mra": "complication", "neuropathy": "complication",
    "microvascular": "complication", "retinopathy": "complication",
    "gastroparesis": "complication", "hypoglycemia_rescue": "complication",
}


@dataclass
class DecisionItem:
    """逐项决策记录(κ 评审的最小单元)。评分逻辑不产出此对象时不影响任何分值。

    kind / axis 映射到 gas_f 的三重选择性:
      contraindication → commission 罚分 → 验证 A1a → κ_禁忌
      omission         → 激活域未覆盖   → 验证 A1b → κ_漏项
      neutral          → in-scope 无罚分 → 验证 A1c → κ_中性
      coverage_met     → 激活域被覆盖   → RECOMMEND 条件的正向确认(非 κ 三轴,可选报)
    """
    item_id:          str
    kind:             str                       # contraindication|omission|neutral|coverage_met
    axis:             str                       # kappa_contra|kappa_omission|kappa_neutral|coverage_positive
    gas_decision:     str                       # penalize_commission|penalize_omission|no_penalty|credit_indicated
    domain:           Optional[str] = None
    rule_id:          Optional[str] = None
    drug:             Optional[str] = None       # 具体药名(禁忌/中性项)
    category:         Optional[str] = None
    recommended_cats: List[str] = field(default_factory=list)   # 漏项:本该出现的类别
    grade:            float = 0.0
    sat_frac:         Optional[float] = None      # 覆盖/漏项:域内连续满足度


@dataclass
class DomainEvalResult:
    # ── 双轴主指标 ──
    concordance:     float = 0.0          # recall / 域覆盖
    appropriateness: float = 1.0          # precision / 1 − commission 率
    gas_f:           float = 0.0          # 两轴调和均值
    # ── 逐项明细(仅 collect_items=True 时填充;不影响任何分值)──
    items:           List["DecisionItem"] = field(default_factory=list)
    # ── 域分解 ──
    n_activated_domains:  int = 0
    n_satisfied_domains:  int = 0
    domain_status:  Dict[str, str] = field(default_factory=dict)   # domain → SATISFIED/OMITTED
    # ── precision 分解 ──
    n_prescribed_in_scope: int = 0
    commission_violations: List[Tuple[str, str]] = field(default_factory=list)  # (rule_id, drug)
    allow_overrides:       List[Tuple[str, str]] = field(default_factory=list)  # (rule_id, drug)
    # ── 可选:分组并列(glycemic vs complication)──
    concordance_glycemic:     Optional[float] = None
    concordance_complication: Optional[float] = None
    # ── 专科分层(RECOMMEND-only,与 gas_f 同源)──
    concordance_by_specialty: Dict[str, float] = field(default_factory=dict)
    gas_macro: Optional[float] = None   # 各专科 concordance 等权 macro(MIN_RULES 门槛)


# 域 → "核心达成类别":该域的满足只能由这些类别达成,联合方案里的附带类别不算。
# 解决类别污染:cardiorenal 域内联合规则(如"metformin + SGLT2i")的 any_of_cats 混入
# Biguanides/ACEi 等,导致开附带药即误判整个域满足、掩盖 SGLT2i/GLP-1 核心漏项。
# 仅对 cardiorenal_protection 启用;其余域(降压/降糖等同质可替代)维持 OR 语义不变。
# domain → 专科(单一事实源,零外部文件)。定案:cardiorenal→Renal, bp_raas→Cardiovascular, micro→Renal。
DOMAIN_SPECIALTY: Dict[str, str] = {
    "glycemic_first_line": "Endocrine", "glycemic_intensification": "Endocrine",
    "glycemic_insulin": "Endocrine", "hypoglycemia_rescue": "Endocrine",
    "ckd_mra": "Renal", "cardiorenal_protection": "Renal",
    "microvascular": "Renal", "retinopathy": "Renal",
    "bp_raas": "Cardiovascular", "ascvd_antiplatelet": "Cardiovascular",
    "lipid_management": "Cardiovascular", "bp_other": "Cardiovascular",
    "hf_management": "Cardiovascular",
    "neuropathy": "Neuro", "gastroparesis": "Neuro",
}
_SPEC_MIN_RULES = 3   # 某专科激活 RECOMMEND 域数 < 此值 → 该患者不计入 macro(避免小专科噪声)


_DOMAIN_CORE_CATS: Dict[str, Set[str]] = {
    "cardiorenal_protection": {"SGLT2i", "GLP-1RA", "GIP_GLP1_RA", "CKD_MRA"},
}


def _domain_satisfied(domain_cats: Set[str], cat_to_drug_names: Dict[str, List[str]],
                      domain: Optional[str] = None) -> bool:
    """处方是否含该域任一推荐类别(含 Insulins_Unspecified 桥接)。
    若该域在 _DOMAIN_CORE_CATS 中,仅核心类别可判定满足(附带类别不算)。"""
    prescribed = {c for c, names in cat_to_drug_names.items() if names}
    # [类别污染修复] 有核心白名单的域,只认核心类别达成满足
    core = _DOMAIN_CORE_CATS.get(domain)
    effective_cats = (domain_cats & core) if core else domain_cats
    if effective_cats & prescribed:
        return True
    # 胰岛素未指明剂型桥接:剂型无关的胰岛素域可被 Insulins_Unspecified 满足
    if ({"Insulins_Basal", "Insulins_Rapid"} & effective_cats
            and cat_to_drug_names.get("Insulins_Unspecified")):
        return True
    return False


def domain_eval(
    rules:             List[Dict],
    cat_to_drug_names: Dict[str, List[str]],
    cm:                Dict,
    patient_data:      Dict,
    admission_year:    int,
    taxonomy_scope:    Optional[Set[str]] = None,
    collect_items:     bool = False,
    item_prefix:       str = "",
) -> DomainEvalResult:
    """
    域级双轴评分核心。

    Args:
        rules:             含 domain/polarity 的规则列表(三个 destination 全纳入)。
        cat_to_drug_names: 处方 {category: [drug_name_lower]}。
        taxonomy_scope:    in-scope 类别集合(precision 分母用);None=用 cat_to_drug_names 全部。
    """
    res = DomainEvalResult()

    # [TEMPORAL-CONSISTENCY-V1] Future therapies are outside the GAS
    # scoring universe for this encounter year.
    cat_to_drug_names = _filter_prescription_by_availability(
        cat_to_drug_names, admission_year
    )

    # [GAS-CANON] 类别规范化:处方侧 + scope。规则侧在下方读 any_of_cats 处对称施加。
    _canon_selfcheck_once()
    cat_to_drug_names = _canon_cat_to_drug_names(cat_to_drug_names)
    if taxonomy_scope is not None:
        taxonomy_scope = _canon_cat_set(taxonomy_scope)

    # 处方内类别/药名集合
    prescribed_cats  = {c for c, names in cat_to_drug_names.items() if names}
    prescribed_drugs = {d.lower() for names in cat_to_drug_names.values() for d in names}

    # ── 1. 遍历,按极性收集适用规则 ──────────────────────────────────────────
    domain_recommend: Dict[str, Dict] = {}     # domain → {weight, cats:set, satisfied:bool}
    avoid_rules:  List[Dict] = []
    # [PATCH-CC] RECOMMEND 满足判定延后到 commission 算完之后,
    # 元素为 (domain, cats);见本文件 §1.5。
    _pending_recommend: List[Tuple[str, Set[str]]] = []
    allow_targets_drugs: Set[str] = set()       # ALLOW 许可的药名
    allow_targets_cats:  Set[str] = set()       # ALLOW 许可的类别

    crash = defaultdict(int)
    for rule in rules:
        if rule.get("disabled"):
            continue
        if rule.get("scorable") is False:        # taxonomy 外:仅标注不评分
            continue
        try:
            if not rule["condition"](cm, patient_data, admission_year):
                continue
        except Exception:
            crash[rule.get("id", "?")] += 1
            continue

        polarity = rule.get("polarity", "RECOMMEND")
        _raw_cats = _available_rule_categories(
            rule.get("any_of_cats"), admission_year
        )
        cats = _canon_cat_set(_raw_cats)    # [GAS-CANON] 规则侧

        # [TEMPORAL-CONSISTENCY-V1] A RECOMMEND rule with no encounter-era
        # available target category must not enter the omission denominator.
        if polarity == "RECOMMEND" and not cats:
            continue

        # [GAS-CANON] 规范化后类别为空(整类被移出 scope)且无 exclude_drugs 兜底
        # → 该规则不可能被满足。若仍计入 RECOMMEND 分母,会给全体患者凭空加一个
        # 永久漏项域(18 条 rescue 规则全在 domain='hypoglycemia_rescue')。
        # 跳过 = 把该域整体移出分母,才是"移出评分 scope"的正确语义。
        if GAS_CANON_ENABLED and not cats and not (rule.get("exclude_drugs") or []):
            continue
        grade    = float(rule.get("grade") or 0.0)
        domain   = rule.get("domain") or "UNCLASSIFIED"

        if polarity == "RECOMMEND":
            d = domain_recommend.setdefault(
                domain, {"weight": 0.0, "cats": set(), "satisfied": False,
                         "n_rules": 0, "n_hit": 0})   # [SCHEME-2] 域内连续满足度:逐规则计数
            _g = 1.0 if GAS_EQUAL_WEIGHTS else grade        # ← 等权开关
            d["weight"] = max(d["weight"], _g)
            d["cats"] |= cats
            # [SCHEME-2] 该规则自身是否被命中(保留 core-cat 白名单语义:传入本规则 domain)
            d["n_rules"] += 1
            # [PATCH-CC] 原为就地 _domain_satisfied(...) → n_hit += 1。
            # 此刻 commission 尚未计算,禁忌药会被计入 concordance 分子。
            # 改为登记,统一在 §1.5 用扣除 commission 后的处方回放判定。
            _pending_recommend.append((domain, cats))

        elif polarity == "AVOID":
            avoid_rules.append(rule)

        elif polarity == "ALLOW":
            # 许可:该域药物在此条件下安全 → 记入许可集,后续抑制 AVOID
            allow_targets_cats |= cats
            for c in cats:
                allow_targets_drugs |= {n.lower() for n in cat_to_drug_names.get(c, [])}

    if crash:
        logger.debug("domain_eval condition crashes: %s", dict(crash))

    # ── 1.5 [PATCH-CC] commission 前置 + concordance 分子过滤 ────────────────
    # 原第 3 节的 commission 计算整体前移至此(逻辑逐字不变),
    # 目的:让 concordance 的满足判定能看到哪些药是 commission。
    commission_drugs: Set[str] = set()
    for rule in avoid_rules:
        cats = _canon_cat_set(rule.get("any_of_cats"))        # [GAS-CANON] 规则侧
        excl = {d.lower() for d in (rule.get("exclude_drugs") or [])}
        # avoid 目标:有 exclude_drugs 用之(精确匹配,药名已在 _compute_gas 中归一化);否则避开整个类别
        if excl:
            hit = prescribed_drugs & excl
        else:
            hit = {n.lower() for c in cats for n in cat_to_drug_names.get(c, [])}
        for drug in hit:
            # ALLOW 覆盖:同药/同类别被许可 → 不算 commission
            drug_cat_allowed = any(drug in {n.lower() for n in cat_to_drug_names.get(c, [])}
                                   for c in (cats & allow_targets_cats))
            if drug in allow_targets_drugs or drug_cat_allowed:
                res.allow_overrides.append((rule.get("id", "?"), drug))
                continue
            commission_drugs.add(drug)
            res.commission_violations.append((rule.get("id", "?"), drug))

    # [PATCH-CC] 扣除 commission 药后的处方副本:禁忌药不得为任何 RECOMMEND 域
    # 提供满足证据。类别下全部药物被扣光 → 该类别自然退出 `prescribed`,
    # _domain_satisfied 的语义无需改动。
    _clean_cat_to_drug_names = {
        c: [n for n in names if n.lower() not in commission_drugs]
        for c, names in cat_to_drug_names.items()
    }
    for _dom, _cats in _pending_recommend:
        if _domain_satisfied(_cats, _clean_cat_to_drug_names, _dom):
            domain_recommend[_dom]["n_hit"] += 1

    # ── 2. Concordance(域覆盖)────────────────────────────────────────────────
    tot_w = sat_w = 0.0
    grp_tot = defaultdict(float); grp_sat = defaultdict(float)
    spec_tot = defaultdict(float); spec_sat = defaultdict(float); spec_cnt = defaultdict(int)
    for domain, d in domain_recommend.items():
        # [SCHEME-2] 域满足度 = 域内被命中的 RECOMMEND 规则数 / 域内激活规则数 ∈ [0,1]
        #   稀疏域(n_rules==1)-> ∈{0,1} 退化回旧二值(保护 mimic3);
        #   多规则域 -> 连续,保留域内部分合规(mimic4 增益)。
        n_rules = d.get("n_rules", 0)
        sat_frac = (d.get("n_hit", 0) / n_rules) if n_rules > 0 else 0.0
        d["satisfied"] = sat_frac                        # 现为 [0,1] 连续值
        # 展示/下游兼容:>=0.5 记 SATISFIED
        res.domain_status[domain] = "SATISFIED" if sat_frac >= 0.5 else "OMITTED"
        w = d["weight"]
        tot_w += w
        grp = DOMAIN_GROUP.get(domain, "complication")
        grp_tot[grp] += w
        _sp = DOMAIN_SPECIALTY.get(domain)
        if _sp:
            spec_tot[_sp] += w; spec_cnt[_sp] += 1
        # [SCHEME-2] 用满足度加权而非 0/1
        sat_w += w * sat_frac
        grp_sat[grp] += w * sat_frac
        if _sp:
            spec_sat[_sp] += w * sat_frac

    res.n_activated_domains = len(domain_recommend)
    # n_satisfied_domains 语义近似:满足度 >=0.5 的域数
    res.n_satisfied_domains = sum(1 for v in domain_recommend.values()
                                  if v.get("satisfied", 0.0) >= 0.5)
    res.concordance = round(sat_w / tot_w, 4) if tot_w > 1e-9 else 0.0
    if grp_tot.get("glycemic", 0) > 1e-9:
        res.concordance_glycemic = round(grp_sat["glycemic"] / grp_tot["glycemic"], 4)
    if grp_tot.get("complication", 0) > 1e-9:
        res.concordance_complication = round(grp_sat["complication"] / grp_tot["complication"], 4)
    # 专科分层 concordance + 等权 macro(仅计入激活域数≥_SPEC_MIN_RULES 的专科)
    _spec_rates = {}
    for _sp, _tw in spec_tot.items():
        if _tw > 1e-9 and spec_cnt[_sp] >= _SPEC_MIN_RULES:
            _spec_rates[_sp] = round(spec_sat[_sp] / _tw, 4)
    res.concordance_by_specialty = _spec_rates
    if _spec_rates:
        res.gas_macro = round(sum(_spec_rates.values()) / len(_spec_rates), 4)

    # ── 3. Appropriateness(commission 罚分,含 ALLOW 覆盖)─────────────────────
    # [PATCH-CC] commission_drugs 与 res.commission_violations / res.allow_overrides
    # 已在 §1.5 计算完毕(原本在此处的循环整块前移)。唯一真源 = §1.5;
    # 若要改动 commission 判定逻辑,只改 §1.5,不要在此重建。

    # precision 分母:在范围内的处方类别数
    scope = taxonomy_scope if taxonomy_scope is not None else prescribed_cats
    in_scope_drugs = {d.lower() for c in (prescribed_cats & scope) for d in cat_to_drug_names.get(c, [])} \
        if taxonomy_scope is not None else prescribed_drugs
    res.n_prescribed_in_scope = len(in_scope_drugs)
    if in_scope_drugs:
        res.appropriateness = round(max(0.0, 1.0 - len(commission_drugs) / len(in_scope_drugs)), 4)
    else:
        res.appropriateness = 1.0   # 无在范围内处方 → 无不当可言

    # ── 3.5 逐项决策明细(κ 评审单元;不改变任何分值)────────────────────────
    if collect_items:
        # 药名 → 类别(反查,给禁忌/中性项标注)
        drug_to_cat: Dict[str, str] = {}
        for c, names in cat_to_drug_names.items():
            for n in names:
                drug_to_cat.setdefault(n.lower(), c)

        # 满足任一激活 RECOMMEND 域的药名集合(用于判定"中性")
        satisfying_drugs: Set[str] = set()
        for domain, d in domain_recommend.items():
            core = _DOMAIN_CORE_CATS.get(domain)
            eff_cats = (d["cats"] & core) if core else d["cats"]
            for c in eff_cats:
                satisfying_drugs |= {n.lower() for n in cat_to_drug_names.get(c, [])}
            if ({"Insulins_Basal", "Insulins_Rapid"} & eff_cats
                    and cat_to_drug_names.get("Insulins_Unspecified")):
                satisfying_drugs |= {n.lower() for n in cat_to_drug_names["Insulins_Unspecified"]}

        pfx = item_prefix or "P"

        # (a) 每个激活 RECOMMEND 域 → coverage_met(已覆盖) / omission(漏项)
        for domain, d in domain_recommend.items():
            sat_frac = float(d.get("satisfied", 0.0))
            core = _DOMAIN_CORE_CATS.get(domain)
            eff_cats = sorted((d["cats"] & core) if core else d["cats"])
            covering = sorted({n.lower() for c in eff_cats
                               for n in cat_to_drug_names.get(c, [])} & prescribed_drugs)
            omitted = sat_frac < 0.5
            res.items.append(DecisionItem(
                item_id=f"{pfx}::dom::{domain}",
                kind="omission" if omitted else "coverage_met",
                axis="kappa_omission" if omitted else "coverage_positive",
                gas_decision="penalize_omission" if omitted else "credit_indicated",
                domain=domain,
                recommended_cats=eff_cats,
                drug=(", ".join(covering) if covering else None),
                grade=float(d.get("weight", 0.0)),
                sat_frac=round(sat_frac, 4),
            ))

        # (b) 每个 commission → contraindication(禁忌命中)
        for rid, drug in res.commission_violations:
            res.items.append(DecisionItem(
                item_id=f"{pfx}::contra::{drug}",
                kind="contraindication",
                axis="kappa_contra",
                gas_decision="penalize_commission",
                rule_id=rid,
                drug=drug,
                category=drug_to_cat.get(drug.lower()),
                grade=1.0,
            ))

        # (c) in-scope 处方药:commission→已在(b);满足激活域→indicated;其余→neutral
        for drug in sorted(in_scope_drugs):
            if drug in commission_drugs:
                continue
            if drug in satisfying_drugs:
                # 满足某个激活 RECOMMEND 域 → GAS 判「该用/恰当」(Task A 标签 1)
                res.items.append(DecisionItem(
                    item_id=f"{pfx}::indicated::{drug}",
                    kind="indicated",
                    axis="kappa_indicated",
                    gas_decision="credit_indicated",
                    drug=drug,
                    category=drug_to_cat.get(drug.lower()),
                    grade=1.0,
                ))
                continue
            res.items.append(DecisionItem(
                item_id=f"{pfx}::neutral::{drug}",
                kind="neutral",
                axis="kappa_neutral",
                gas_decision="no_penalty",
                drug=drug,
                category=drug_to_cat.get(drug.lower()),
                grade=0.0,
            ))

    # ── 4. F ──
    C, A = res.concordance, res.appropriateness
    res.gas_f = round(2 * C * A / (C + A), 4) if (C + A) > 1e-9 else 0.0
    return res


# ═══════════════════════════════════════════════════════════════════════════════
# 自测:mock 规则 + 患者,覆盖 SATISFIED / OMITTED / COMMISSION / ALLOW-override
# ═══════════════════════════════════════════════════════════════════════════════
def _selftest():
    T = lambda cm, pd, yr: True
    HF = lambda cm, pd, yr: bool(cm.get("heart_failure"))

    rules = [
        # RECOMMEND:降糖一线(metformin)
        {"id": "R1", "domain": "glycemic_first_line", "polarity": "RECOMMEND",
         "condition": T, "any_of_cats": ["Biguanides"], "exclude_drugs": [], "grade": 1.0, "scorable": True},
        # RECOMMEND:HF → SGLT2i 心肾保护
        {"id": "R2", "domain": "cardiorenal_protection", "polarity": "RECOMMEND",
         "condition": HF, "any_of_cats": ["SGLT2i"], "exclude_drugs": [], "grade": 1.0, "scorable": True},
        # RECOMMEND:调脂(他汀)
        {"id": "R3", "domain": "lipid_management", "polarity": "RECOMMEND",
         "condition": T, "any_of_cats": ["Lipid_Statins"], "exclude_drugs": [], "grade": 1.0, "scorable": True},
        # AVOID:HF 禁用 metformin(旧指南)
        {"id": "A1", "domain": "glycemic_first_line", "polarity": "AVOID",
         "condition": HF, "any_of_cats": ["Biguanides"], "exclude_drugs": ["metformin"], "grade": 1.0, "scorable": True},
        # ALLOW:HF+eGFR>30 metformin 安全(新指南,覆盖 A1)
        {"id": "L1", "domain": "glycemic_first_line", "polarity": "ALLOW",
         "condition": HF, "any_of_cats": ["Biguanides"], "exclude_drugs": [], "grade": 0.75, "scorable": True},
    ]

    print("="*70)
    # 病例1:无 HF。开 metformin+statin。glycemic_first_line✓ lipid✓,无心肾域激活。
    cm1 = {}
    rx1 = {"Biguanides": ["metformin"], "Lipid_Statins": ["atorvastatin"]}
    r1 = domain_eval([x for x in rules], rx1, cm1, {}, 2020)
    print(f"病例1 (无HF, metformin+statin):")
    print(f"  Concordance={r1.concordance} (期望1.0: first_line+lipid 两域全满足)")
    print(f"  Appropriateness={r1.appropriateness} (期望1.0: 无commission)")
    print(f"  域: {r1.domain_status}")
    assert r1.concordance == 1.0 and r1.appropriateness == 1.0

    print("-"*70)
    # 病例2:HF。只开 metformin(无 SGLT2i,无 statin)。
    #   激活域:first_line(满足), cardiorenal(遗漏), lipid(遗漏) → C=1/3
    #   AVOID A1 命中 metformin,但 ALLOW L1 覆盖 → 无 commission,A=1.0
    cm2 = {"heart_failure": True}
    rx2 = {"Biguanides": ["metformin"]}
    r2 = domain_eval([x for x in rules], rx2, cm2, {}, 2020)
    print(f"病例2 (HF, 仅metformin):")
    print(f"  Concordance={r2.concordance} (期望0.333: 3域激活仅first_line满足)")
    print(f"  Appropriateness={r2.appropriateness} (期望1.0: A1禁忌被L1 ALLOW覆盖)")
    print(f"  ALLOW覆盖: {r2.allow_overrides}")
    print(f"  域: {r2.domain_status}")
    assert abs(r2.concordance - 0.3333) < 0.01
    assert r2.appropriateness == 1.0 and r2.allow_overrides

    print("-"*70)
    # 病例3:HF,但把 ALLOW 规则去掉(模拟旧 era,无新指南许可)。开 metformin。
    #   AVOID A1 命中 metformin 且无覆盖 → commission, A = 1 − 1/1 = 0
    rules_no_allow = [x for x in rules if x["id"] != "L1"]
    r3 = domain_eval(rules_no_allow, rx2, cm2, {}, 2020)
    print(f"病例3 (HF, 仅metformin, 无ALLOW):")
    print(f"  Appropriateness={r3.appropriateness} (期望0.0: metformin commission未被覆盖)")
    print(f"  commission: {r3.commission_violations}")
    assert r3.appropriateness == 0.0 and r3.commission_violations

    print("-"*70)
    # 病例4:HF。开 metformin + empagliflozin(SGLT2i)+ statin → 全域满足,ALLOW 覆盖禁忌
    rx4 = {"Biguanides": ["metformin"], "SGLT2i": ["empagliflozin"], "Lipid_Statins": ["atorvastatin"]}
    r4 = domain_eval([x for x in rules], rx4, cm2, {}, 2020)
    print(f"病例4 (HF, metformin+SGLT2i+statin):")
    print(f"  Concordance={r4.concordance} (期望1.0)  Appropriateness={r4.appropriateness} (期望1.0)")
    print(f"  GAS_F={r4.gas_f}  glycemic域C={r4.concordance_glycemic} 并发症域C={r4.concordance_complication}")
    assert r4.concordance == 1.0 and r4.appropriateness == 1.0

    print("="*70)
    print("✅ 全部断言通过:SATISFIED / OMITTED / COMMISSION / ALLOW-override 四分支正确")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _selftest()
