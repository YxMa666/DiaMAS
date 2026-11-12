#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluation/ose_calculator.py  v4 (dual-path GAS + complication management)
=========================================================================

变更记录 v4(在 v3 基础上):
  [REMOVED] Path-T(文本近邻)已移除,解决复现性问题。
  [UPDATED] 公式简化为双路径:
          n_dm   = 适用 DM 规则数量(Path-DM 的评估对象)
          n_comp = 适用并发症规则数量(Path-C 的评估对象)
          GAS_dm   = Path-DM_raw
          GAS_comp = Path-C_raw
          GAS_raw  = (n_dm × GAS_dm + n_comp × GAS_comp) / (n_dm + n_comp)
          → 无并发症患者 n_comp=0,退化为 Path-DM,与旧版语义一致

变更记录 v3(在 v2 基础上):
  [NEW-1] Path-C:并发症药物条件性评分路径(Complication-conditional path)
          覆盖 5 条并发症管理规则(CPR-01~CPR-05),对应 6 个指南推荐药物类别:
            CPR-01: Lipid_Statins         - T2DM + ASCVD / 年龄>=40 (ADA+ACC-AHA)
            CPR-02: Hypertension_ACEi/ARB - T2DM + CKD/蛋白尿 (ADA+KDIGO)
            CPR-03: Antiplatelet           - T2DM + 确诊 ASCVD (ADA)
            CPR-04: Hypertension_BetaBlockers - T2DM + 心衰 (ESC 2023)
            CPR-05: CKD_MRA (Finerenone)  - T2DM + DKD eGFR 25-75 (KDIGO 2022)

  [NEW-2] OSEMetrics 新增路径分解字段:
          gas_comp_score     - Path-C 得分
          gas_comp_rules     - 触发的 CPR 规则 ID 列表
          gas_comp_satisfied - 满足的 CPR 规则 ID 列表
          gas_n_dm_drugs     - 参与评分的降糖药数量
          gas_n_comp_rules   - 适用的并发症规则数量

方法论依据(来自 A3 诊断分析):
  PSM 配对患者处方 Jaccard=0.17(接近0),平均药物差异 6.68 种,
  差异主要在并发症用药而非降糖药。现有 GAS 只评估降糖药,
  无法捕捉处方质量的核心区分维度,导致 PSM 配对后 p=0.17 不显著。
  扩展到并发症药物后,GAS 能直接测量多智能体 VDT 设计的核心价值:
  跨指南的综合并发症管理能力(ADA + KDIGO + ESC + ACC-AHA)。

继承自 v2:
  [FIX-1] 移除 Grade A 关键词里的硬编码药物名
  [FIX-2] 确定性药物类别 x 入院年份查表(Path-D)
  [FIX-3] 入院年份统一由 safety_checker.resolve_admission_year 给出
  [FIX-4] 扩展 Grade B 关键词
"""

from __future__ import annotations
import json as _json
import logging
import math
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from evaluation.drug_normalizer import DrugNormalizer
    from evaluation.metrics_calculator import PatientMetrics

from evaluation.safety_checker import (
    _GAS_PATIENT_GATES, _GAS_NEGATION_SIGNALS,
    _get_egfr, _get_dm_type, _has_complication, _get_admission_year,
    resolve_admission_year,   # [Y1 FIX] 去标识化年份的唯一权威来源
)
from evaluation.metrics_calculator import _extract_drug_list_from_rec

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 一、[v4] Path-T(文本近邻)已移除,仅保留 DRUG_APPROVAL_YEAR(时代穿越检测)
# ═══════════════════════════════════════════════════════════════════════════════


_DISABLED_RULES_REGISTRY: Dict[str, Dict] = {}


# 二、DM_RULES:规则化降糖药评分(v4 Path-DM,替代 Path-T + Path-D)
# ═══════════════════════════════════════════════════════════════════════════════
#
# 与 COMPLICATION_DRUG_GRADES 完全同构:
#   condition   - lambda(cm, patient_data, year) -> bool
#   any_of_cats - OR 逻辑,满足任意一类即得分
#   grade       - 1.0=Grade A 等效,0.5=Grade B 等效
#
# 评分:GAS_dm = Σ(satisfied weights) / Σ(applicable weights)
#
# ══ HARDCODE INTERFACE ═════════════════════════════════════════════════════════
# 新规则通过 hardcode_rules.py --apply 追加到「待添加区」
# ══════════════════════════════════════════════════════════════════════════════

DM_RULES: List[Dict] = [
    {
        "id":          'FINAL-0001',
        "description": 'For T2DM with CKD and eGFR ≤30 mL/min/1.73m², avoid lixisenatide and exenatide (GLP-1RAs requiring renal elimination).',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2013
            and (_has_ckd_g3plus(cm) or _get_dm_comp(cm, "nephropathy"))
            and (lambda egfr: egfr is not None and egfr <= 30)(_get_egfr(cm))
        ),
        "any_of_cats": ['GLP-1RA'],
        "exclude_drugs": ['lixisenatide', 'exenatide'],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2026, Section 9: Glucose-Lowering Therapy for People With Cardiovascular Disease',
        "domain":      'cardiorenal_protection',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0002',
        "description": 'Avoid thiazolidinedione (TZD) treatment in patients with symptomatic heart failure',
        "condition":   lambda cm, pd, yr: (
            yr >= 2010
            and _get_dm_comp(cm, "heart_failure")
        ),
        "any_of_cats": ['Thiazolidinediones'],
        "exclude_drugs": ['pioglitazone', 'rosiglitazone'],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2010 - Heart Failure',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0003',
        "description": 'Metformin contraindicated in patients with treated CHF',
        "condition":   lambda cm, pd, yr: (
            yr >= 2006
            and _get_dm_comp(cm, "heart_failure")
        ),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": ['metformin'],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2006 - Treatment recommendations and goals',
        "domain":      'glycemic_first_line',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0004',
        "description": 'Caution in prescribing TZDs in setting of known CHF or other heart diseases',
        "condition":   lambda cm, pd, yr: (
            yr >= 2006
            and _get_dm_comp(cm, "heart_failure")
        ),
        "any_of_cats": ['Thiazolidinediones'],
        "exclude_drugs": ['pioglitazone', 'rosiglitazone'],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2006 - Recommendations',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0005',
        "description": 'In patients with treated CHF, metformin and thiazolidinedione (TZD) are contraindicated and should be avoided at discharge.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2008
            and _get_dm_comp(cm, "heart_failure")
        ),
        "any_of_cats": ['Biguanides', 'Thiazolidinediones'],
        "exclude_drugs": ['metformin', 'Pioglitazone', 'Rosiglitazone'],
        "grade":       0.5,
        "source":      'ADA Standards of Care 2008, Section on CVD',
        "domain":      'glycemic_first_line',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0006',
        "description": 'Metformin is contraindicated in diabetic patients with renal insufficiency or significant heart failure',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2008
            and _get_dm_comp(cm, "heart_failure")
            and (lambda egfr: egfr is not None and egfr <= 60)(_get_egfr(cm))
        ),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": ['metformin'],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2008, Older Adults / Special Considerations',
        "domain":      'glycemic_first_line',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0007',
        "description": 'Thiazolidinediones should not be used in diabetic patients with CHF (NYHA class III/IV)',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2008
            and _get_dm_comp(cm, "heart_failure")
        ),
        "any_of_cats": ['Thiazolidinediones'],
        "exclude_drugs": ['pioglitazone', 'rosiglitazone'],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2008, Older Adults / Special Considerations',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'audit:avoid_uncheckable',
    },
    {
        "id":          'FINAL-0008',
        "description": 'In T2DM patients with CKD (eGFR<30) or heart failure, metformin should be avoided at discharge.',
        "condition":   lambda cm, pd, yr: (_has_ckd_g3plus(cm)
            and 
            _get_dm_type(cm) == "T2DM"
            and yr >= 2013
            and _get_dm_comp(cm, "heart_failure")
            and (lambda egfr: egfr is not None and egfr < 30)(_get_egfr(cm))
        ),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": ['metformin'],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2013, Recommendations on Older Adults',
        "domain":      'glycemic_first_line',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0009',
        "description": 'Avoid TZD treatment in patients with symptomatic heart failure.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2011
            and _get_dm_comp(cm, "heart_failure")
        ),
        "any_of_cats": ['Thiazolidinediones'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2011, Treatment',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0010',
        "description": 'For T1DM patients, must use intensive insulin therapy (multiple daily injections or CSII)',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T1DM"
            and yr >= 2010
        ),
        "any_of_cats": ['Insulins_Basal', 'Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2010, Therapy for type 1 diabetes',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0011',
        "description": 'SGLT2 inhibitors should not be used in individuals with type 1 diabetes (not approved)',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T1DM"
            and yr >= 2013
        ),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2026, SGLT Inhibition and Risk of Ketosis',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0012',
        "description": 'Discontinue oral hypoglycemic agents and initiate insulin therapy for the majority of patients with type 2 diabetes at the time of hospital admission for an acute illness.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2012
        ),
        "any_of_cats": ['Biguanides', 'Sulfonylureas', 'Meglitinides', 'Thiazolidinediones', 'Alpha_Glucosidase_Inhibitors', 'DPP-4i'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Consensus Reports & Supplements 2012, Section 4.2.2',
        "domain":      'glycemic_first_line',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'audit:composite_deprescribe+initiate; unquantifiable scope (majority/acute illness); RECOMMEND encoding inverted vs "discontinue orals"',
    },
    {
        "id":          'FINAL-0014',
        "description": 'Discontinuation of the DPP-4 inhibitor is recommended when intensifying from a DPP-4 inhibitor to a GLP-1 RA.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2006
        ),
        "any_of_cats": ['DPP-4i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2022 Section 9 Fig 9.3 Principle of Incorporation',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'audit:avoid_uncheckable',
    },
    {
        "id":          'FINAL-0015',
        "description": 'Metformin is contraindicated because of renal insufficiency in diabetic patients',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2008 and (lambda egfr: egfr is not None and egfr < 30)(_get_egfr(cm))),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2008 - Older Adults Recommendations',
        "domain":      'glycemic_first_line',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0016',
        "description": 'Avoid routine use of sulfonylureas in hospitalized patients due to long action and hypoglycemia risk with altered nutrition',
        "condition":   lambda cm, pd, yr: yr >= 2008,
        "any_of_cats": ['Sulfonylureas'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2008 - Hospitalized Patients Treatment Options',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'audit:routine_use_unquantifiable + no phenotype gate (blanket anti-SU too absolute for inpatient)',
    },
    {
        "id":          'FINAL-0017',
        "description": 'Specific caution is required with metformin due to renal insufficiency or unstable hemodynamic status',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2010 and ((lambda egfr: egfr is not None and egfr < 30)(_get_egfr(cm)) or _is_icu_approx(pd))),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": ['metformin'],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2010 - Treatment options in hospitalized patients',
        "domain":      'glycemic_first_line',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0018',
        "description": 'Sliding-scale insulin (SSI) as the sole regimen is potentially dangerous and ineffective in type 1 diabetes.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T1DM"
            and yr >= 2013
        ),
        "any_of_cats": ['Insulins_Rapid'],
        "exclude_drugs": ['sliding_scale_insulin_monotherapy'],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2013 | 2. Antihyperglycemic agents in hospitalized patients',
        "domain":      'glycemic_insulin',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'P2-A: exclude 指向策展中有意移除的药(噻嗪/非DHP-CCB/gemfibrozil/保钾/跨类别),commission 永不触发,孤儿规则',
    },
    {
        "id":          'FINAL-0019',
        "description": 'Avoid routine use of sulfonylureas in hospitalized patients due to long action and hypoglycemia risk',
        "condition":   lambda cm, pd, yr: yr >= 2009,
        "any_of_cats": ['Sulfonylureas'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2009 - 3. Treatment options in hospitalized patients',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'audit:duplicate_of_FINAL-0016 (routine-use SU, no phenotype gate)',
    },
    {
        "id":          'FINAL-0020',
        "description": 'Meglitinides are precluded from use in hospital due to lack of data and prandial effect',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2009
        ),
        "any_of_cats": ['Meglitinides'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2009 - 3. Treatment options in hospitalized patients',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'judge:scope_out_of_discharge',
    },
    {
        "id":          'FINAL-0021',
        "description": 'Avoid metformin in hospitalized patients with hemodynamic instability or renal insufficiency due to lactic acidosis risk',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2009
        ),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2009 - 3. Treatment options in hospitalized patients',
        "domain":      'glycemic_first_line',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'judge:scope_out_of_discharge',
    },
    {
        "id":          'FINAL-0022',
        "description": 'Omit metformin in the presence of unstable hemodynamic status during hospitalization',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2011
        ),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": ['metformin'],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2011 - Anti-hyperglycemic agents in hospitalized patients',
        "domain":      'glycemic_first_line',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'judge:scope_out_of_discharge',
    },
    {
        "id":          'FINAL-0023',
        "description": 'Avoid or use metformin with caution in patients with unstable hemodynamic status',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2012
        ),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2012 - Antihyperglycemic agents in hospitalized patients',
        "domain":      'glycemic_first_line',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'judge:scope_out_of_discharge',
    },
    {
        "id":          'FINAL-0024',
        "description": 'DPP-4 inhibitors are typically weaned or discontinued when initiating combination injectable therapy',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2007
        ),
        "any_of_cats": ['DPP-4i'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2022/2023 Section 9 Combination Injectable Therapy',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'audit:avoid_uncheckable',
    },
    {
        "id":          'FINAL-0025',
        "description": 'Avoid routine use of sulfonylureas and meglitinides in hospitalized patients due to long action and hypoglycemia risk.',
        "condition":   lambda cm, pd, yr: yr >= 2006,
        "any_of_cats": ['Sulfonylureas', 'Meglitinides'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2006, 2. Treatment options',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'audit:routine_use_unquantifiable + no phenotype gate',
    },
    {
        "id":          'FINAL-0028',
        "description": 'Sliding scale insulin as the sole regimen is potentially dangerous in type 1 diabetes',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T1DM"
            and yr >= 2010
        ),
        "any_of_cats": ['Insulins_Rapid'],
        "exclude_drugs": ['sliding_scale_insulin_sole_regimen'],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2010 - 2. Treatment options in hospitalized patients',
        "domain":      'glycemic_insulin',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'audit:avoid_uncheckable',
    },
    {
        "id":          'FINAL-0029',
        "description": 'Initiation of SGLT2i is not recommended for glycemic control if eGFR < 30 mL/min/1.73m²',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2013
            and (lambda egfr: egfr is not None and egfr <= 29)(_get_egfr(cm))
        ),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Consensus Reports & Supplements 2022 - SGLT2i eGFR thresholds',
        "domain":      'cardiorenal_protection',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0030',
        "description": 'Avoid sulfonylureas in patients with impaired renal function due to risk of severe and prolonged hypoglycemia',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2012 and (lambda egfr: egfr is not None and egfr < 45)(_get_egfr(cm))),
        "any_of_cats": ['Sulfonylureas'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Consensus Reports & Supplements 2012 - 4.2.1–4.2.3 Evidence',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0031',
        "description": 'Avoid metformin in patients with renal failure at discharge',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2012
            and (lambda egfr: egfr is not None and egfr <= 30)(_get_egfr(cm))
        ),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Consensus Reports & Supplements 2012 | 4.4.1–4.4.3 Evidence',
        "domain":      'glycemic_first_line',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0033',
        "description": 'Discontinue oral diabetes drugs and non-insulin injectable diabetes medications upon hospital admission.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2012
        ),
        "any_of_cats": ['Biguanides', 'SGLT2i', 'GLP-1RA', 'DPP-4i', 'Sulfonylureas', 'Meglitinides', 'Thiazolidinediones', 'Alpha_Glucosidase_Inhibitors'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Consensus Reports & Supplements 2012 | Table 1',
        "domain":      'glycemic_first_line',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'judge:scope_out_of_discharge',
    },
    {
        "id":          'FINAL-0034',
        "description": 'Discontinuation of oral and noninsulin injectable antidiabetic agents before surgery',
        "condition":   lambda cm, pd, yr: yr >= 2012,
        "any_of_cats": ['Biguanides', 'SGLT2i', 'DPP-4i', 'Sulfonylureas', 'Meglitinides', 'Thiazolidinediones', 'Alpha_Glucosidase_Inhibitors', 'Glucokinase_Activators', 'Bile_Acid_Sequestrants_DM', 'Dopamine_Agonists_DM', 'GLP-1RA', 'GIP_GLP1_RA', 'GCG_GLP1_RA', 'GLP1_Multimer_RA', 'Amylin_Analogues'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Consensus Reports & Supplements 2012 5.3.2',
        "domain":      'glycemic_first_line',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'judge:scope_out_of_discharge',
    },
    {
        "id":          'FINAL-0035',
        "description": 'Discontinue oral and noninsulin injectable antidiabetic agents before surgery; initiate insulin therapy if hyperglycemia occurs',
        "condition":   lambda cm, pd, yr: yr >= 2012,
        "any_of_cats": ['Biguanides', 'SGLT2i', 'GLP-1RA', 'DPP-4i', 'Sulfonylureas', 'Meglitinides', 'Thiazolidinediones', 'Alpha_Glucosidase_Inhibitors', 'GIP_GLP1_RA', 'GCG_GLP1_RA', 'GLP1_Multimer_RA'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Consensus Reports & Supplements 2012 - 5.3.2',
        "domain":      'glycemic_first_line',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'judge:scope_out_of_discharge',
    },
    {
        "id":          'FINAL-0036',
        "description": 'Avoid metformin in patients with heart failure',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2006
            and _get_dm_comp(cm, "heart_failure")
        ),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": ['metformin'],
        "grade":       1.0,
        "source":      'ADA 2006 Recommendations',
        "domain":      'glycemic_first_line',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0037',
        "description": 'Avoid metformin in patients with heart failure, hemodynamic instability, or CKD due to lactic acidosis risk',
        "condition":   lambda cm, pd, yr: (_has_ckd_g3plus(cm)
            and 
            _get_dm_type(cm) == "T2DM"
            and yr >= 2008
            and _get_dm_comp(cm, "heart_failure")
        ),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2008 | 3. Treatment options',
        "domain":      'glycemic_first_line',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'audit:avoid_uncheckable',
    },
    {
        "id":          'FINAL-0039',
        "description": 'Tolbutamide: avoid in renal impairment (eGFR<60) due to increased hypoglycemia risk',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and (lambda egfr: egfr is not None and egfr <= 60)(_get_egfr(cm))
        ),
        "any_of_cats": ['Sulfonylureas'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2001 (historical Era-1)',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0146',
        "description": 'For T2DM patients with heart failure, avoid thiazolidinediones due to increased HF risk.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and _get_dm_comp(cm, "heart_failure")
        ),
        "any_of_cats": ['Thiazolidinediones'],
        "exclude_drugs": ['pioglitazone', 'rosiglitazone'],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2024, Table 9.2',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0150',
        "description": 'Discontinue or taper sulfonylureas and DPP-4 inhibitors when initiating combination injectable therapy in T2DM.',
        "condition":   lambda cm, pd, yr: _get_dm_type(cm) == "T2DM",
        "any_of_cats": ['Sulfonylureas', 'DPP-4i'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2022/2023, Combination Injectable Therapy',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'audit:avoid_uncheckable',
    },
    {
        "id":          'FINAL-0151',
        "description": 'Insulin is the mainstay of therapy for adults with type 1 diabetes at discharge',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T1DM" and yr >= 2001),
        "any_of_cats": ['Insulins_Basal', 'Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2016, Insulin Therapy section',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0152',
        "description": 'T1DM patients should be prescribed basal insulin at discharge (via MDI or CSII)',
        "condition":   lambda cm, pd, yr: (yr >= 2001 and _get_dm_type(cm) == "T1DM"),
        "any_of_cats": ['Insulins_Basal'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2022, Section 9, Recommendation 9.1',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0153',
        "description": 'T1DM patients should be prescribed rapid-acting insulin analogs at discharge to reduce hypoglycemia risk',
        "condition":   lambda cm, pd, yr: (yr >= 2001 and _get_dm_type(cm) == "T1DM"),
        "any_of_cats": ['Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2022, Section 9, Recommendation 9.2',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0158',
        "description": 'Sulfonylureas should be used with caution in elderly patients due to increased risk of severe hypoglycemia',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and (_get_age(cm, pd) or 0) >= 65
        ),
        "any_of_cats": ['Sulfonylureas'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Consensus Reports & Supplements 2012, 4.2.1–4.2.3',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0159',
        "description": 'Sulfonylureas should be avoided in patients with impaired renal function due to increased hypoglycemia risk',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and (lambda egfr: egfr is not None and egfr <= 30)(_get_egfr(cm))
        ),
        "any_of_cats": ['Sulfonylureas'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Consensus Reports & Supplements 2012, 4.2.1–4.2.3',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0160',
        "description": 'In T1DM, use of GLP-1 RAs or dual GIP/GLP-1 RA should be avoided due to higher rates of DKA and limited evidence.',
        "condition":   lambda cm, pd, yr: _get_dm_type(cm) == "T1DM",
        "any_of_cats": ['GLP-1RA', 'GIP_GLP1_RA'],
        "exclude_drugs": ['liraglutide', 'semaglutide', 'tirzepatide', 'exenatide'],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2026 Section 9 (Noninsulin Treatments for Type 1 Diabetes)',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0161',
        "description": 'Metformin should be avoided in unstable or hospitalized patients with CHF.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and _get_dm_comp(cm, "heart_failure")
        ),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2009 - Treatment',
        "domain":      'glycemic_first_line',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'judge:scope_out_of_discharge',
    },
    {
        "id":          'FINAL-0162',
        "description": 'In patients with treated CHF, metformin and thiazolidinedione (TZD) use are contraindicated.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and _get_dm_comp(cm, "heart_failure")
        ),
        "any_of_cats": ['Biguanides', 'Thiazolidinediones'],
        "exclude_drugs": ['metformin', 'Pioglitazone', 'Rosiglitazone'],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2008 - Treatment',
        "domain":      'glycemic_first_line',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0163',
        "description": 'In adults with T2DM starting insulin therapy, reassess the need for or omit glucose-lowering agents with higher hypoglycemia risk (i.e., sulfonylureas and meglitinides)',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and _requires_insulin(cm, pd)),
        "any_of_cats": ['Sulfonylureas', 'Meglitinides'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2024 Section 9 Pharmacologic Approaches',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0164',
        "description": 'In older adults with diabetes who are frail or have multiple comorbidities, consider deprescribing medications associated with harm, such as hypoglycemia.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and (_get_age(cm, pd) or 0) >= 65
        ),
        "any_of_cats": ['Sulfonylureas', 'Meglitinides', 'Insulins_Basal', 'Insulins_Rapid'],
        "exclude_drugs": ['Glipizide', 'Glyburide', 'Glimepiride', 'Gliclazide', 'Chlorpropamide', 'Tolbutamide', 'Tolazamide', 'Acetohexamide', 'Repaglinide', 'Nateglinide', 'Mitiglinide', 'NPH insulin', 'Insulin human', 'Insulin glargine', 'Insulin detemir', 'Insulin degludec', 'Insulin icodec', 'Regular insulin', 'Insulin lispro', 'Insulin aspart', 'Insulin glulisine', 'Faster insulin aspart'],
        "grade":       0.75,
        "source":      'ADA Consensus Reports & Supplements 2022 - Age: Older People With Diabetes',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0227',
        "description": 'For T1DM, SGLT2i is not recommended due to increased risk of DKA.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T1DM"
            and yr >= 2017
        ),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2025, Noninsulin Treatments for Type 1 Diabetes',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0228',
        "description": 'SGLT2 inhibitors are not FDA-approved for T1DM; risk of euglycemic DKA',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T1DM"
            and yr >= 2018
        ),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2018 - Incretin-Based Therapies',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0229',
        "description": 'GLP-1 receptor agonists and DPP-4 inhibitors are not FDA-approved for T1DM',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T1DM"
            and yr >= 2018
        ),
        "any_of_cats": ['GLP-1RA', 'DPP-4i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2018 - Incretin-Based Therapies',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0230',
        "description": 'Metformin should be omitted in T2DM patients with eGFR < 30 mL/min/1.73 m²',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2018
            and (lambda egfr: egfr is not None and egfr < 30)(_get_egfr(cm))
        ),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Consensus Reports & Supplements 2018, Medications for Lowering Glucose Metformin',
        "domain":      'glycemic_first_line',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0231',
        "description": 'Metformin should be omitted in T2DM patients with acute kidney injury',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2018
            and (cm.get("kidney") or {}).get("aki_suspected", False)
        ),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Consensus Reports & Supplements 2018, Medications for Lowering Glucose Metformin',
        "domain":      'glycemic_first_line',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0233',
        "description": 'Do not initiate Metformin if eGFR 30-45',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2018
            and (lambda egfr: egfr is not None and 30 <= egfr and egfr <= 45)(_get_egfr(cm))
        ),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2018 Table 8.2',
        "domain":      'glycemic_first_line',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'audit:avoid_uncheckable',
    },
    {
        "id":          'FINAL-0234',
        "description": 'Avoid SGLT2i if eGFR < 30',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2018
            and (lambda egfr: egfr is not None and egfr < 30)(_get_egfr(cm))
        ),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2018 Table 8.2',
        "domain":      'cardiorenal_protection',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0235',
        "description": 'Avoid Alpha-Glucosidase Inhibitors if eGFR < 25',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2018
            and (lambda egfr: egfr is not None and egfr <= 25)(_get_egfr(cm))
        ),
        "any_of_cats": ['Alpha_Glucosidase_Inhibitors'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2018 Table 8.2',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0236',
        "description": 'Metformin should be avoided in unstable or hospitalized patients with CHF',
        "condition":   lambda cm, pd, yr: (
            yr >= 2014
            and _get_dm_comp(cm, "heart_failure")
        ),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": ['metformin'],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2014 Cardiovascular Disease and Risk Management',
        "domain":      'glycemic_first_line',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'judge:scope_out_of_discharge',
    },
    {
        "id":          'FINAL-0238',
        "description": 'Avoid metformin in significant heart failure',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2014
            and _get_dm_comp(cm, "heart_failure")
        ),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": ['metformin'],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2014 - Older Adults',
        "domain":      'glycemic_first_line',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0239',
        "description": 'Use TZDs very cautiously in those with or at risk for CHF',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2014
            and _get_dm_comp(cm, "heart_failure")
        ),
        "any_of_cats": ['Thiazolidinediones'],
        "exclude_drugs": ['pioglitazone', 'rosiglitazone'],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2014 - Older Adults',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0240',
        "description": 'Concurrent use of DPP-4 inhibitors with GLP-1 RA or dual GIP/GLP-1 RA is not recommended',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2015
            and any(
                any(kw in str(m.get('drug_name', '') if isinstance(m, dict) else m).lower()
                    for kw in ('glp-1', 'glp1', 'semaglutide', 'liraglutide', 'dulaglutide',
                               'exenatide', 'lixisenatide', 'tirzepatide', 'gip_glp', 'gip-glp'))
                for m in (pd.get('all_medications', []) or [])
            )
        ),
        "any_of_cats": ['DPP-4i'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2025 Section 9',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0241',
        "description": 'When initiating intensification of insulin therapy, use of sulfonylureas, meglitinides, and DPP-4 inhibitors should be limited or discontinued',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2018
        ),
        "any_of_cats": ['Sulfonylureas', 'Meglitinides', 'DPP-4i'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2026 - Combination Injectable Therapy',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'audit:superseded_by_FINAL-0163 (insulin + SU/meglitinide); re-enabling would double-count SU',
    },
    {
        "id":          'FINAL-0242',
        "description": 'Avoid lixisenatide if eGFR < 15 mL/min',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2018
            and (lambda egfr: egfr is not None and egfr <= 14)(_get_egfr(cm))
        ),
        "any_of_cats": ['GLP-1RA'],
        "exclude_drugs": ['lixisenatide'],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2018 - Combination Injectable Therapy',
        "domain":      'cardiorenal_protection',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0243',
        "description": 'Adjunctive agents (SGLT2i, GLP-1RA) are not approved in the context of type 1 diabetes due to DKA risk.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T1DM"
            and yr >= 2019
        ),
        "any_of_cats": ['SGLT2i', 'GLP-1RA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2019 Section 9 Noninsulin Treatments for Type 1 Diabetes',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0244',
        "description": 'Avoid GLP-1 receptor agonists in type 1 diabetes as they are currently approved only for type 2 diabetes.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T1DM"
            and yr >= 2017
        ),
        "any_of_cats": ['GLP-1RA'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2023 - Noninsulin Treatments for Type 1 Diabetes',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0245',
        "description": 'SGLT2i, GLP-1RA, and DPP-4i are not FDA-approved for use in patients with type 1 diabetes',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T1DM"
            and yr >= 2018
        ),
        "any_of_cats": ['SGLT2i', 'GLP-1RA', 'DPP-4i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2018 - Pharmacologic Approaches to Glycemic Treatment',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0246',
        "description": 'Metformin is not FDA-approved for use in patients with type 1 diabetes; addition to insulin did not improve glycemic control and increased risk for gastrointestinal adverse events.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T1DM"
            and yr >= 2018
        ),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": ['metformin'],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2018 - Pharmacologic Approaches to Glycemic Treatment',
        "domain":      'glycemic_first_line',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0247',
        "description": 'Metformin is not FDA-approved for use in patients with type 1 diabetes',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T1DM"
            and yr >= 2017
        ),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": ['metformin'],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2017 - Investigational Agents Metformin',
        "domain":      'glycemic_first_line',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0248',
        "description": 'DPP-4 inhibitors are not currently FDA approved for use in patients with type 1 diabetes',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T1DM"
            and yr >= 2017
        ),
        "any_of_cats": ['DPP-4i'],
        "exclude_drugs": ['Sitagliptin', 'Vildagliptin', 'Saxagliptin', 'Linagliptin', 'Alogliptin', 'Teneligliptin', 'Gemigliptin', 'Trelagliptin', 'Omarigliptin', 'Anagliptin', 'Evogliptin', 'Gosogliptin', 'Retagliptin'],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2017 - Incretin-Based Therapies',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0249',
        "description": 'Avoid saxagliptin in patients with heart failure',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2018
            and _get_dm_comp(cm, "heart_failure")
        ),
        "any_of_cats": ['DPP-4i'],
        "exclude_drugs": ['saxagliptin'],
        "grade":       1.0,
        "source":      'ADA Consensus Reports & Supplements 2018 - Figure 3',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0250',
        "description": 'Consider discontinuing saxagliptin and alogliptin in people who develop heart failure.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2014
            and _get_dm_comp(cm, "heart_failure")
        ),
        "any_of_cats": ['DPP-4i'],
        "exclude_drugs": ['saxagliptin', 'alogliptin'],
        "grade":       0.75,
        "source":      'ADA Standards of Medical Care in Diabetes 2024 Section 16.11',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0251',
        "description": 'Avoid adding DPP-4i to sulfonylurea in patients with hypoglycemia history due to 50% increased risk',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2018 and _has_hypoglycemia_hx(pd) and _is_on_sulfonylurea(pd)),
        "any_of_cats": ['DPP-4i'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Consensus Reports & Supplements 2018 - DPP-4 Inhibitors',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0252',
        "description": 'Stop or reduce dose of sulfonylureas by 50% when basal insulin is initiated',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2018
        ),
        "any_of_cats": ['Sulfonylureas'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Consensus Reports & Supplements 2018 - SULFONYLUREA',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'audit:dose_titration_uncheckable (reduce SU 50%); presence-only proxy would over-penalise legitimate SU+insulin',
    },
    {
        "id":          'FINAL-0253',
        "description": 'Stop thiazolidinediones when beginning insulin therapy',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and _requires_insulin(cm, pd)),
        "any_of_cats": ['Thiazolidinediones'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Consensus Reports & Supplements 2018 - SULFONYLUREA Figure',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    False,
        "disabled_reason": 'REENABLED (audit): AVOID SU/TZD when insulin required; presence-checkable',
    },
    {
        "id":          'FINAL-0254',
        "description": 'Discontinue DPP-4 inhibitors if GLP-1 receptor agonists are introduced',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2018
        ),
        "any_of_cats": ['DPP-4i'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Consensus Reports & Supplements 2018 - SULFONYLUREA Figure',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'audit:duplicate_of_FINAL-0240 (DPP-4i + GLP-1RA concurrency already covered)',
    },
    {
        "id":          'FINAL-0257',
        "description": 'DPP-4 inhibitors or TZDs are not recommended for patients with diabetes with stage B, C, and D HF',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2014
            and _get_dm_comp(cm, "heart_failure")
        ),
        "any_of_cats": ['DPP-4i', 'Thiazolidinediones'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Consensus Reports & Supplements 2022 - Key Points',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0259',
        "description": 'If HbA1c is below 6.5% or substantially below the individualized glycemic target, consider stopping or reducing the dose of medications with risk of hypoglycemia.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2018 and (lambda h: h is not None and h < 6.5)(cm.get("hba1c"))),
        "any_of_cats": ['Sulfonylureas', 'Insulins_Basal', 'Insulins_Rapid', 'Meglitinides'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Consensus Reports & Supplements 2018 - Consensus recommendation',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0360',
        "description": 'Do not combine GLP-1 RA with DPP-4 inhibitor due to lack of added glucose-lowering benefit',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2020
        ),
        "any_of_cats": ['GLP-1RA', 'DPP-4i'],
        "exclude_drugs": ['sitagliptin', 'saxagliptin', 'vildagliptin', 'linagliptin', 'alogliptin'],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2025 - Section 9',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'audit:avoid_uncheckable',
    },
    {
        "id":          'FINAL-0361',
        "description": 'When initiating a new glucose-lowering medication, reassess and potentially reduce dose of sulfonylureas, meglitinides, or insulin to minimize hypoglycemia risk',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2025
        ),
        "any_of_cats": ['Sulfonylureas', 'Meglitinides', 'Insulins_Basal', 'Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2025, Recommendation 9.20',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'audit:avoid_uncheckable',
    },
    {
        "id":          'FINAL-0362',
        "description": 'Concurrent use of DPP-4 inhibitors with GLP-1 RA or dual GIP/GLP-1 RA is not recommended',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2014
        ),
        "any_of_cats": ['DPP-4i', 'GLP-1RA', 'GIP_GLP1_RA', 'GCG_GLP1_RA', 'GLP1_Multimer_RA'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2025, Recommendation 9.21',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'audit:avoid_uncheckable',
    },
    {
        "id":          'FINAL-0363',
        "description": 'SGLT2 inhibitors (including sotagliflozin) are contraindicated in type 1 diabetes due to increased DKA risk.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T1DM"
            and yr >= 2026
        ),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": ['sotagliflozin', 'empagliflozin', 'dapagliflozin', 'canagliflozin', 'ertugliflozin'],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2026, Noninsulin Treatments for Type 1 Diabetes',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0364',
        "description": 'Avoid incretin medications (GLP-1 RAs, dual GIP/GLP-1 RA, DPP-4 inhibitors) in individuals with pancreatic diabetes and a history of pancreatitis',
        "condition":   lambda cm, pd, yr: (yr >= 2026 and _icd_starts_with(pd, {"K85", "K860", "K861"})),
        "any_of_cats": ['GLP-1RA', 'GIP_GLP1_RA', 'DPP-4i'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2026, Pancreatic Diabetes and Cystic Fibrosis–Related Diabetes',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0365',
        "description": 'Individuals who have experienced DKA should not be treated with SGLT inhibition',
        "condition":   lambda cm, pd, yr: (yr >= 2026 and _has_dka_hhs(pd)),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2026, SGLT Inhibition and Risk of Ketosis',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0366',
        "description": 'Avoid sulfonylureas in hospitalized patients with T2DM due to increased hypoglycemia risk and common interrupted nutrition',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2022
        ),
        "any_of_cats": ['Sulfonylureas'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Consensus Reports & Supplements 2022, Benefits and harms',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0367',
        "description": 'Avoid Thiazolidinediones in heart failure due to increased risk of fluid retention',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2023
            and _get_dm_comp(cm, "heart_failure")
        ),
        "any_of_cats": ['Thiazolidinediones'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2023 Table 9.1',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0368',
        "description": 'When intensifying to insulin therapy, sulfonylureas and DPP-4 inhibitors are typically weaned or discontinued',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2024
        ),
        "any_of_cats": ['Sulfonylureas', 'DPP-4i'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2024 Section 9',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'audit:avoid_uncheckable',
    },
    {
        "id":          'FINAL-0369',
        "description": 'Do not use GLP-1 RAs or dual GIP/GLP-1 RA together with a DPP-4 inhibitor due to no added benefit',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2025
        ),
        "any_of_cats": ['DPP-4i'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2025 Section 9',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'audit:avoid_uncheckable',
    },
    {
        "id":          'FINAL-0370',
        "description": 'Concurrent use of DPP-4 inhibitors with a GLP-1 RA or a dual GIP and GLP-1 RA is not recommended',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2026
        ),
        "any_of_cats": ['DPP-4i'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2026 Section 9.18',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'audit:avoid_uncheckable',
    },
    {
        "id":          'FINAL-0371',
        "description": 'Avoid SGLT2i (increased ketoacidosis risk) and GLP-1RA (not approved) in type 1 diabetes.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T1DM"
            and yr >= 2021
        ),
        "any_of_cats": ['SGLT2i', 'GLP-1RA'],
        "exclude_drugs": ['Empagliflozin', 'Dapagliflozin', 'Canagliflozin', 'Ertugliflozin', 'Sotagliflozin', 'Ipragliflozin', 'Tofogliflozin', 'Bexagliflozin', 'Henagliflozin', 'Rongliflozin', 'Janagliflozin', 'Remogliflozin', 'Exenatide', 'Lixisenatide', 'Liraglutide', 'Semaglutide', 'Dulaglutide', 'Albiglutide', 'Beinaglutide', 'Loxenatide', 'Polyethylene glycol loxenatide'],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2021 - Noninsulin Treatments for Type 1 Diabetes',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0372',
        "description": 'Preferential de-escalation of therapies causing hypoglycemia without cardiovascular, kidney, or metabolic benefits (e.g., SU, meglitinides).',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2026
        ),
        "any_of_cats": ['Sulfonylureas', 'Meglitinides'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2026 - Section 9 (Combination Therapy)',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'audit:context-dependent de-escalation priority, not an absolute AVOID; intent-to-deescalate unobservable',
    },
    {
        "id":          'FINAL-0373',
        "description": 'Use of non-FDA-approved compounded incretin products is not recommended due to safety, quality, and effectiveness concerns.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2025
        ),
        "any_of_cats": ['GLP-1RA', 'GIP_GLP1_RA'],
        "exclude_drugs": ['compounded_incretins'],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2025 - Therapeutic Strategies With Medication Unavailability',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'audit:avoid_uncheckable',
    },
    {
        "id":          'FINAL-0374',
        "description": 'Initiation of SGLT2i is not recommended for HF if eGFR < 20 mL/min/1.73m²',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2020
            and _get_dm_comp(cm, "heart_failure")
            and (lambda egfr: egfr is not None and egfr <= 19)(_get_egfr(cm))
        ),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Consensus Reports & Supplements 2022 - SGLT2i eGFR thresholds',
        "domain":      'cardiorenal_protection',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0375',
        "description": 'Careful use and titration of insulin and sulfonylurea agents is recommended to avoid hypoglycemia in T2D and CKD',
        "condition":   lambda cm, pd, yr: (_has_ckd_g3plus(cm)
            and 
            _get_dm_type(cm) == "T2DM"
            and yr >= 2022
        ),
        "any_of_cats": ['Sulfonylureas', 'Insulins_Basal', 'Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Consensus Reports & Supplements 2022 - Glucose-Lowering Agents in T2D and CKD',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'audit:careful-titration directive (dose management), not a presence rule',
    },
    {
        "id":          'FINAL-0376',
        "description": 'Stop or reduce dose of medications with hypoglycemia risk if HbA1c<6.5% or hypoglycemia history',
        "condition":   lambda cm, pd, yr: yr >= 2022,
        "any_of_cats": ['Sulfonylureas', 'Meglitinides', 'Insulins_Basal', 'Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Consensus Reports & Supplements 2022 - Proactive Care: Avoiding Inertia',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'P3: 减停药却编成 RECOMMEND,极性错,停用(二期作剂量滴定重构)',
    },
    {
        "id":          'FINAL-0464',
        "description": 'All GLP-1 RAs are contraindicated in T2DM patients with eGFR < 30 mL/min/1.73m² (CKD stage 4-5)',
        "condition":   lambda cm, pd, yr: (_has_ckd_g3plus(cm)
            and 
            _get_dm_type(cm) == "T2DM"
            and yr >= 2010
            and (lambda egfr: egfr is not None and egfr < 30)(_get_egfr(cm))
        ),
        "any_of_cats": ['GLP-1RA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      '',
        "domain":      'cardiorenal_protection',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0465',
        "description": 'Sulfonylureas and glinides should be discontinued when prandial insulin is introduced',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2009
        ),
        "any_of_cats": ['Sulfonylureas', 'Meglitinides'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      '',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'audit:avoid_uncheckable',
    },
    {
        "id":          'FINAL-0466',
        "description": 'Avoid metformin in patients with contraindications: eGFR<60 mL/min, CHF requiring pharmacologic management, active liver disease, chronic alcohol abuse, sepsis, acute illness with risk of tissue hypoperfusion or hypoxemia, or IV contrast administration.',
        "condition":   lambda cm, pd, yr: (
            yr >= 2009
            and _get_dm_comp(cm, "heart_failure")
            and (lambda egfr: egfr is not None and egfr <= 60)(_get_egfr(cm))
        ),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": ['metformin'],
        "grade":       1.0,
        "source":      '',
        "domain":      'glycemic_first_line',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0467',
        "description": 'Exenatide and DPP-4 inhibitors have not been approved by the FDA for concomitant use with insulin',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2009
        ),
        "any_of_cats": ['GLP-1RA', 'DPP-4i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      '',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'audit:avoid_uncheckable',
    },
    {
        "id":          'FINAL-0468',
        "description": 'SGLT2i is not indicated for eGFR < 45 mL/min/1.73 m²',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2013
            and (lambda egfr: egfr is not None and egfr <= 44)(_get_egfr(cm))
        ),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      '',
        "domain":      'cardiorenal_protection',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0469',
        "description": 'Avoid saxagliptin and alogliptin in heart failure due to increased hospitalization risk',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2013
            and _get_dm_comp(cm, "heart_failure")
        ),
        "any_of_cats": ['DPP-4i'],
        "exclude_drugs": ['saxagliptin', 'alogliptin'],
        "grade":       1.0,
        "source":      '',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0472',
        "description": 'DPP-4 inhibitors were not approved for use with insulin in 2009 guideline',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2009
        ),
        "any_of_cats": ['DPP-4i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      '',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'audit:avoid_uncheckable',
    },
    {
        "id":          'FINAL-0508',
        "description": 'Metformin should not be used in T2DM patients with eGFR < 30 mL/min/1.73m²',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2016
            and (lambda egfr: egfr is not None and egfr < 30)(_get_egfr(cm))
        ),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": ['metformin'],
        "grade":       1.0,
        "source":      '',
        "domain":      'glycemic_first_line',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0509',
        "description": 'Metformin should not be used in patients with stage 3B, 4, or 5 CKD (eGFR <45 mL/min or CKD G3b+)',
        "condition":   lambda cm, pd, yr: (_has_ckd_g3plus(cm)
            and 
            _get_dm_type(cm) == "T2DM"
            and yr >= 2016
            and (lambda egfr: egfr is not None and egfr <= 45)(_get_egfr(cm))
        ),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      '',
        "domain":      'glycemic_first_line',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0510',
        "description": 'Exenatide should not be used if creatinine clearance is <30 mL/min (approximated by eGFR <30)',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2016
            and (lambda egfr: egfr is not None and egfr < 30)(_get_egfr(cm))
        ),
        "any_of_cats": ['GLP-1RA'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      '',
        "domain":      'cardiorenal_protection',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0511',
        "description": 'For T2DM patients with severe renal impairment (eGFR<30) or ESRD, exenatide is contraindicated; other GLP-1 RAs should be used with caution',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2018
            and (lambda egfr: egfr is not None and egfr < 30)(_get_egfr(cm))
        ),
        "any_of_cats": ['GLP-1RA'],
        "exclude_drugs": ['exenatide'],
        "grade":       0.75,
        "source":      '',
        "domain":      'cardiorenal_protection',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'audit:avoid_uncheckable',
    },
    {
        "id":          'FINAL-0512',
        "description": 'Avoid SGLT2i when eGFR < 45 mL/min/1.73m² due to lack of glycemic efficacy',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2016
            and (lambda egfr: egfr is not None and egfr <= 44)(_get_egfr(cm))
        ),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      '',
        "domain":      'cardiorenal_protection',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0514',
        "description": 'SGLT2i have limited glycemic efficacy in patients with eGFR <45 mL/min/1.73m2',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2015
            and (lambda egfr: egfr is not None and egfr <= 44)(_get_egfr(cm))
        ),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      '',
        "domain":      'cardiorenal_protection',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0515',
        "description": 'SGLT2i have limited efficacy and should be avoided in T2DM patients with eGFR < 45 mL/min/1.73m2',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2016
            and (lambda egfr: egfr is not None and egfr <= 44)(_get_egfr(cm))
        ),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      '',
        "domain":      'cardiorenal_protection',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0516',
        "description": 'Avoid concomitant use of SFUs with insulin due to increased risks of hypoglycemia and weight gain.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2016
        ),
        "any_of_cats": ['Sulfonylureas'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      '',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0663',
        "description": 'Thiazolidinediones are contraindicated in patients with heart failure due to sodium and water retention worsening HF',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2007
            and _get_dm_comp(cm, "heart_failure")
        ),
        "any_of_cats": ['Thiazolidinediones'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ESC CVD in Diabetes 2023 2021 | 13.1 Diabetes',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0664',
        "description": 'Metformin should be avoided in diabetic patients with CKD stage 3-4.',
        "condition":   lambda cm, pd, yr: (_has_ckd_g3plus(cm)
            and 
            _get_dm_type(cm) == "T2DM"
            and yr >= 2013
            and (lambda egfr: egfr is not None and 15 <= egfr and egfr <= 59)(_get_egfr(cm))
        ),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": ['metformin'],
        "grade":       0.75,
        "source":      'ESC CVD in Diabetes 2023 2013, 6.2.7 Special considerations',
        "domain":      'glycemic_first_line',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0666',
        "description": 'Avoid metformin if eGFR <50 mL/min due to risk of lactic acidosis',
        "condition":   lambda cm, pd, yr: (
            yr >= 2013
            and (lambda egfr: egfr is not None and egfr <= 49)(_get_egfr(cm))
        ),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": ['metformin'],
        "grade":       0.75,
        "source":      'ESC CVD in Diabetes 2023 (2013 version), Section 6.2.6 Glucose-lowering agents',
        "domain":      'glycemic_first_line',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0667',
        "description": 'Avoid thiazolidinediones in diabetic patients with HFrEF',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2013
            and _has_hfref(cm, pd)
        ),
        "any_of_cats": ['Thiazolidinediones'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ACC-AHA Cholesterol/Cardiovascular 2022, 7.3.7 Drugs of Unproven Value or That May Worsen HF',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0668',
        "description": 'Thiazolidinediones should be avoided in patients with HF due to increased HF events',
        "condition":   lambda cm, pd, yr: (
            yr >= 2013
            and _get_dm_comp(cm, "heart_failure")
        ),
        "any_of_cats": ['Thiazolidinediones'],
        "exclude_drugs": ['pioglitazone', 'rosiglitazone'],
        "grade":       0.75,
        "source":      'ACC-AHA Cholesterol / Cardiovascular 2013, Section 7.3.2.9.5',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0669',
        "description": 'Metformin, acarbose and most sulphonylureas should be avoided in stage 3-4 CKD (eGFR < 50)',
        "condition":   lambda cm, pd, yr: (_has_ckd_g3plus(cm)
            and 
            yr >= 2013
            and (lambda egfr: egfr is not None and egfr < 50)(_get_egfr(cm))
        ),
        "any_of_cats": ['Biguanides', 'Sulfonylureas', 'Alpha_Glucosidase_Inhibitors'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 2013 | 6.2.7 Special considerations',
        "domain":      'glycemic_first_line',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0670',
        "description": 'Metformin is not recommended if the estimated eGFR is < 50 mL/min',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2013
            and (lambda egfr: egfr is not None and egfr <= 49)(_get_egfr(cm))
        ),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": ['metformin'],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2013 | 6.2.6 Glucose-lowering agents',
        "domain":      'glycemic_first_line',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0671',
        "description": 'FDA recommends discontinuation specifically of saxagliptin and alogliptin in patients who develop HF',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2013
            and _get_dm_comp(cm, "heart_failure")
        ),
        "any_of_cats": ['DPP-4i'],
        "exclude_drugs": ['saxagliptin', 'alogliptin'],
        "grade":       0.75,
        "source":      'ACC-AHA Cardiovascular 2022 - Section 7.3.7',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0798',
        "description": 'For diabetic patients with CKD undergoing coronary angiography/PCI, metformin should be withheld before the procedure and renal function monitored.',
        "condition":   lambda cm, pd, yr: (_has_ckd_g3plus(cm) and yr >= 2019),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ESC CVD in Diabetes 2023 2019, 7.2.2 Adjunctive pharmacotherapy, 2019',
        "domain":      'glycemic_first_line',
        "polarity":    'RECOMMEND',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'judge:scope_out_of_discharge',
    },
    {
        "id":          'FINAL-0799',
        "description": 'Saxagliptin is not recommended in diabetic patients with heart failure due to increased risk of HF hospitalization.',
        "condition":   lambda cm, pd, yr: (_has_ckd_g3plus(cm)
            and 
            yr >= 2019
            and _get_dm_comp(cm, "heart_failure")
        ),
        "any_of_cats": ['DPP-4i'],
        "exclude_drugs": ['saxagliptin'],
        "grade":       0.75,
        "source":      'ESC CVD in Diabetes 2023 2019, 8.5.4 Dipeptidyl peptidase-4 inhibitors, 2019',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0800',
        "description": 'Thiazolidinediones are not recommended in patients with heart failure',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2016
            and _get_dm_comp(cm, "heart_failure")
        ),
        "any_of_cats": ['Thiazolidinediones'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ESC CVD in Diabetes 2023 (2016 version), Chunk 2',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0801',
        "description": 'Avoid saxagliptin and alogliptin in T2DM patients with heart failure',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2014
            and _get_dm_comp(cm, "heart_failure")
        ),
        "any_of_cats": ['DPP-4i'],
        "exclude_drugs": ['saxagliptin', 'alogliptin'],
        "grade":       0.75,
        "source":      'ACC-AHA Cholesterol/Cardiovascular 2022, 7.3.7 Drugs of Unproven Value or That May Worsen HF',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0802',
        "description": 'Thiazolidinediones are not recommended in patients with HF due to sodium/water retention and increased risk of worsening HF',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2016
            and _get_dm_comp(cm, "heart_failure")
        ),
        "any_of_cats": ['Thiazolidinediones'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 2016 | 11.6 Diabetes',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0803',
        "description": 'Thiazolidinediones should be avoided in patients with heart failure due to increased risk of HF occurrence',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2019
            and _get_dm_comp(cm, "heart_failure")
        ),
        "any_of_cats": ['Thiazolidinediones'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 | 7.1.2.1.4 Thiazolidinediones | 2019',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0804',
        "description": 'SGLT2i or GLP-1RA are not indicated for T1DM due to lack of CVOT data and risk of ketoacidosis',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T1DM"
            and yr >= 2015
        ),
        "any_of_cats": ['SGLT2i', 'GLP-1RA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 | 11.3 Glucose-lowering agents beyond insulin',
        "domain":      'cardiorenal_protection',
        # [P0 FIX] desc 为否定式("not indicated for T1DM"),原 RECOMMEND 会强制
        #          T1DM 患者激活 cardiorenal_protection 域并要求 SGLT2i/GLP-1RA。
        #          纠正为 AVOID。commission_drugs 为 set,与 FINAL-0011/0227/... 不双计。
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0927',
        "description": 'Avoid thiazolidinediones in patients with reduced LVEF (HFrEF) due to fluid retention and increased HF hospitalization risk',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2022
            and _has_hfref(cm, pd)
        ),
        "any_of_cats": ['Thiazolidinediones'],
        "exclude_drugs": ['pioglitazone', 'rosiglitazone'],
        "grade":       1.0,
        "source":      'ACC-AHA Cholesterol / Cardiovascular 2022, Chunk 1, Level A',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0928',
        "description": 'Avoid saxagliptin and alogliptin in patients with HF; FDA recommends discontinuation in patients who develop HF',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2022
            and _get_dm_comp(cm, "heart_failure")
        ),
        "any_of_cats": ['DPP-4i'],
        "exclude_drugs": ['saxagliptin', 'alogliptin'],
        "grade":       0.75,
        "source":      'ACC-AHA Cholesterol / Cardiovascular 2022, Chunk 2, Level B',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0929',
        "description": 'Saxagliptin and alogliptin should be avoided in T2DM patients with HF due to increased risk of HF hospitalization',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2022
            and _get_dm_comp(cm, "heart_failure")
        ),
        "any_of_cats": ['DPP-4i'],
        "exclude_drugs": ['saxagliptin', 'alogliptin'],
        "grade":       0.75,
        "source":      'ACC-AHA Cardiovascular 2022 | 7.3.7. Drugs of Unproven Value or That May Worsen HF',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0930',
        "description": 'Avoid thiazolidinediones in diabetic patients with HF due to potential exacerbation of HF',
        "condition":   lambda cm, pd, yr: (
            yr >= 2022
            and _get_dm_comp(cm, "heart_failure")
        ),
        "any_of_cats": ['Thiazolidinediones'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ACC-AHA Cardiovascular 2022, Section 7.3.5 / Synopsis / Table 13',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0931',
        "description": 'Avoid saxagliptin and alogliptin (DPP-4i) in diabetic patients with HF due to potential exacerbation of HF',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2022
            and _get_dm_comp(cm, "heart_failure")
        ),
        "any_of_cats": ['DPP-4i'],
        "exclude_drugs": ['saxagliptin', 'alogliptin'],
        "grade":       1.0,
        "source":      'ACC-AHA Cardiovascular 2022, Section 7.3.5 / Synopsis / Table 13',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0987',
        "description": 'Avoid thiazolidinediones (TZDs) in diabetic patients with HF due to increased risk of worsening HF and hospitalization',
        "condition":   lambda cm, pd, yr: (
            yr >= 2016
            and _get_dm_comp(cm, "heart_failure")
        ),
        "any_of_cats": ['Thiazolidinediones'],
        "exclude_drugs": ['pioglitazone', 'rosiglitazone'],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 (引用2016 HF指南) - Section 16',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0988',
        "description": 'DPP-4 inhibitors are not recommended to reduce CV events in diabetic patients with HF.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2016
            and _get_dm_comp(cm, "heart_failure")
        ),
        "any_of_cats": ['DPP-4i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 / 2021 ESC HF Guideline - 13.1 Diabetes',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0989',
        "description": 'GLP-1 receptor agonists are not recommended for the prevention of HF events.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2019
            and _get_dm_comp(cm, "heart_failure")
        ),
        "any_of_cats": ['GLP-1RA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 / 2021 ESC HF Guideline - 13.1 Diabetes',
        "domain":      'cardiorenal_protection',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'P1-3: GLP-1RA@HF 用途维度不可表达,与 0994 冲突,停用',
    },
    {
        "id":          'FINAL-0990',
        "description": 'DPP-4 inhibitor saxagliptin is not recommended in patients with HF',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2014
            and _get_dm_comp(cm, "heart_failure")
        ),
        "any_of_cats": ['DPP-4i'],
        "exclude_drugs": ['saxagliptin'],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 2021 | Recommendations for management of patients with HF and diabetes',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0991',
        "description": 'Thiazolidinediones (glitazones) are not recommended in patients with HF, as they increase the risk of HF worsening and HF hospitalization.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2016
            and _get_dm_comp(cm, "heart_failure")
        ),
        "any_of_cats": ['Thiazolidinediones'],
        "exclude_drugs": ['pioglitazone', 'rosiglitazone'],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 (2016) | 7.5.3 Treatments that may cause harm in HFrEF',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1006',
        "description": 'The DPP-4 inhibitor saxagliptin is not recommended in patients with HF.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2021
            and _get_dm_comp(cm, "heart_failure")
        ),
        "any_of_cats": ['DPP-4i'],
        "exclude_drugs": ['saxagliptin'],
        "grade":       0.75,
        "source":      'ESC CVD in Diabetes 2023 (2021 version) - Recommendations for treatment of patients with HF and diabetes',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1022',
        "description": 'Avoid metformin when eGFR < 30 mL/min/1.73m²',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2012
            and (lambda egfr: egfr is not None and egfr <= 29)(_get_egfr(cm))
        ),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'KDIGO CKD & Diabetes-CKD Guidelines 2012 Chapter 2',
        "domain":      'glycemic_first_line',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1023',
        "description": 'Avoid mainly renally excreted sulfonylureas (e.g., glyburide) when eGFR < 30',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2012
            and (lambda egfr: egfr is not None and egfr <= 29)(_get_egfr(cm))
        ),
        "any_of_cats": ['Sulfonylureas'],
        "exclude_drugs": ['glyburide', 'Glyburide'],
        "grade":       1.0,
        "source":      'KDIGO CKD & Diabetes-CKD Guidelines 2012 Chapter 2',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1024',
        "description": 'Discontinue metformin in patients with CKD and eGFR < 30 mL/min/1.73m2',
        "condition":   lambda cm, pd, yr: (_has_ckd_g3plus(cm)
            and 
            _get_dm_type(cm) == "T2DM"
            and yr >= 2012
            and (lambda egfr: egfr is not None and egfr <= 29)(_get_egfr(cm))
        ),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'KDIGO CKD 2012 4.4.6',
        "domain":      'glycemic_first_line',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1049',
        "description": 'GLP-1 RA should not be used in combination with DPP-4 inhibitors',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2015
            and any(
                any(kw in str(m.get('drug_name', '') if isinstance(m, dict) else m).lower()
                    for kw in ('glp-1', 'glp1', 'semaglutide', 'liraglutide', 'dulaglutide',
                               'exenatide', 'lixisenatide', 'tirzepatide', 'gip_glp', 'gip-glp'))
                for m in (pd.get('all_medications', []) or [])
            )
        ),
        "any_of_cats": ['DPP-4i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'KDIGO CKD & Diabetes-CKD Guidelines 2022 | Practice Point 4.2.3',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1055',
        "description": 'Discontinue or do not prescribe metformin when eGFR <30.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2022
            and (lambda egfr: egfr is not None and egfr < 30)(_get_egfr(cm))
        ),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": ['metformin'],
        "grade":       0.75,
        "source":      'KDIGO CKD & Diabetes-CKD Guidelines 2022, Chapter 4, Figure 23',
        "domain":      'glycemic_first_line',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1056',
        "description": 'Do not initiate SGLT2i when eGFR <20.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2022
            and (lambda egfr: egfr is not None and egfr <= 20)(_get_egfr(cm))
        ),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": ['sglt2i'],
        "grade":       0.75,
        "source":      'KDIGO CKD & Diabetes-CKD Guidelines 2022, Chapter 4, Figure 23',
        "domain":      'cardiorenal_protection',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'audit:avoid_uncheckable',
    },
    {
        "id":          'FINAL-1057',
        "description": 'GLP-1 RA should not be used in combination with DPP-4 inhibitors',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2015
            and any(
                any(kw in str(m.get('drug_name', '') if isinstance(m, dict) else m).lower()
                    for kw in ('glp-1', 'glp1', 'semaglutide', 'liraglutide', 'dulaglutide',
                               'exenatide', 'lixisenatide', 'tirzepatide', 'gip_glp', 'gip-glp'))
                for m in (pd.get('all_medications', []) or [])
            )
        ),
        "any_of_cats": ['DPP-4i'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'KDIGO 2020 CKD & Diabetes-CKD Guidelines, Practice Point 4.3.3',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1058',
        "description": 'GLP-1 RA should not be used in combination with DPP-4 inhibitors',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2017
        ),
        "any_of_cats": ['DPP-4i'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'KDIGO CKD & Diabetes-CKD Guidelines 2022 | Practice Point 4.2.3',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'audit:avoid_uncheckable',
    },
    {
        "id":          'FINAL-1059',
        "description": 'Avoid exenatide when eGFR <= 30 mL/min',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2020
            and (lambda egfr: egfr is not None and egfr <= 30)(_get_egfr(cm))
        ),
        "any_of_cats": ['GLP-1RA'],
        "exclude_drugs": ['exenatide'],
        "grade":       1.0,
        "source":      'KDIGO CKD & Diabetes-CKD Guidelines 2020 Chapter 3',
        "domain":      'cardiorenal_protection',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1060',
        "description": 'Withhold Metformin if AKI is suspected to prevent lactic acidosis',
        "condition":   lambda cm, pd, yr: (yr >= 2024 and (cm.get("kidney") or {}).get("aki_suspected", False)),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'KDIGO CKD & Diabetes-CKD Guidelines 2024 | 4.3 Polypharmacy and drug stewardship',
        "domain":      'glycemic_first_line',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'DR-01',
        "description": 'T1DM: must have basal insulin (ADA all eras, Grade A)',
        "condition":   lambda cm, pd, yr: _get_dm_type(cm) == "T1DM",
        "any_of_cats": ['Insulins_Basal'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards all years',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'DR-02',
        "description": 'T1DM: must have rapid-acting insulin (ADA all eras, Grade A)',
        "condition":   lambda cm, pd, yr: _get_dm_type(cm) == "T1DM",
        "any_of_cats": ['Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       0.8,
        "source":      'ADA Standards all years',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'DR-03',
        "description": 'T2DM + CVD + yr>=2016: SGLT2i or GLP-1RA (ADA Grade A)',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and _get_dm_comp(cm, "cardiovascular")
            and yr >= 2016
            and not _is_hospice(pd)
        ),
        "any_of_cats": ['SGLT2i', 'GLP-1RA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA 2016+ Sec 9 + EMPA-REG + LEADER',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'DR-04',
        "description": 'T2DM + CKD(eGFR 20-60) + yr>=2020: SGLT2i (ADA Grade A)',
        "condition":   lambda cm, pd, yr: (_has_ckd_g3plus(cm)
            and 
            _get_dm_type(cm) == "T2DM"
            and yr >= 2020
            and (lambda egfr: egfr is not None and 20 <= egfr <= 60)(_get_egfr(cm))
            and not _is_hospice(pd)
        ),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA 2020/KDIGO 2022 + CREDENCE + DAPA-CKD',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'DR-05',
        "description": 'T2DM + HF + yr>=2020: SGLT2i (ADA Grade A)',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and _get_dm_comp(cm, "heart_failure")
            and yr >= 2020
            and not _is_hospice(pd)
        ),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA 2021 Sec 10 + DAPA-HF + EMPEROR-Reduced',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'DR-06',
        "description": 'T2DM + eGFR>=30 (no AKI): Metformin first-line (ADA, Grade B inpatient)',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and (lambda egfr: egfr is None or egfr >= 30)(_get_egfr(cm))
            and not (cm.get("kidney") or {}).get("aki_suspected", False)
            and not _is_hospice(pd)
        ),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": [],
        "grade":       0.5,
        "source":      'ADA Standards all years first-line',
        "domain":      'glycemic_first_line',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'DR-07',
        "description": 'T2DM: requires ≥1 glucose-lowering agent (ADA all eras). Excludes appropriate-omission cases: ESRD (eGFR<15), AKI+eGFR<30, hospice/palliative/expired discharge.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            # --- Exclusion 1: ESRD (eGFR<15) ---
            and not (lambda egfr: egfr is not None and egfr < 15)(_get_egfr(cm))
            # --- Exclusion 2: AKI + severe CKD (safety hold) ---
            and not (
                (cm.get("kidney") or {}).get("aki_suspected", False)
                and (lambda egfr: egfr is not None and egfr < 30)(_get_egfr(cm))
            )
            # --- Exclusion 3: Hospice / palliative / expired ---
            and str(pd.get("discharge_type", "") or "").lower()
                not in ("hospice", "comfort care", "palliative", "expired")
        ),
        "any_of_cats": ['Biguanides', 'SGLT2i', 'GLP-1RA', 'DPP-4i', 'Insulins_Basal', 'Insulins_Rapid', 'Sulfonylureas', 'Thiazolidinediones', 'Meglitinides', 'Alpha_Glucosidase_Inhibitors', 'GIP_GLP1_RA'],
        "exclude_drugs": [],
        "grade":       0.5,
        "source":      'ADA Standards all years',
        "domain":      'glycemic_first_line',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'PR-06',
        "description": '[P2 REVIEWED] T2DM patients: presence of sulfonylureas or rapid insulin at discharge (proxy for appropriate intensification). Note: original description mentioned de-intensification — mismatch with OR-reward logic; description corrected to reflect actual scoring behavior.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and not _is_appropriate_dm_omission(cm, pd)
        ),
        "any_of_cats": ['Sulfonylureas', 'Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       0.5,
        "source":      'ADA Standards of Care 2026 - Section 9 (Combination Therapy)',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'audit:avoid_uncheckable',
    },
    {
        "id":          'PR-07',
        "description": 'Recommend metformin for patients with T2D, CKD, and eGFR >= 30 mL/min/1.73m^2.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and _has_ckd_g3plus(cm) and (lambda egfr: egfr is not None and egfr >= 30)(_get_egfr(cm)) and not _is_hospice(pd)),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'KDIGO CKD & Diabetes-CKD Guidelines 2022 | Recommendation 4.1.1',
        "domain":      'glycemic_first_line',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'DR-09',
        "description": 'Glucocorticoid-induced hyperglycemia: insulin is the preferred agent (all eras, DM-21)',
        "condition":   lambda cm, pd, yr: (
            bool(cm.get("on_systemic_steroids") or pd.get("on_systemic_steroids"))
            and not _is_appropriate_dm_omission(cm, pd)
        ),
        "any_of_cats": ['Insulins_Basal', 'Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       0.5,
        "source":      'ADA Standards of Care 2026 - Cancer/Steroid hyperglycemia; DM-21',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'DR-10',
        "description": 'T2DM + eGFR 15-30 + yr>=2006: selected DPP-4 inhibitors safe and effective (DM-36)',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2006
            and (lambda egfr: egfr is not None and 15 <= float(egfr) < 30)(_get_egfr(cm))
            and not _is_hospice(pd)
        ),
        "any_of_cats": ['DPP-4i'],
        "exclude_drugs": [],
        "grade":       0.5,
        "source":      'ADA Consensus Reports & Supplements 2022 - Advanced CKD; DM-36',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'DR-11',
        "description": 'T1DM: insulin analogs preferred over NPH/human insulins (ADA 2026 Rec 9.2, DM-14)',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T1DM"
            and not _is_appropriate_dm_omission(cm, pd)
        ),
        "any_of_cats": ['Insulins_Basal', 'Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2026 Sec 9 Recommendation 9.2; DM-14',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'DR-12',
        "description": 'T2DM + BMI>=30 + yr>=2022: GLP-1RA or dual GIP/GLP-1RA preferred (obesity, DM-03/41)',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2022
            and (cm.get("bmi", 0) >= 30 or pd.get("bmi", 0) >= 30)
            and not _is_appropriate_dm_omission(cm, pd)
        ),
        "any_of_cats": ['GLP-1RA', 'GIP_GLP1_RA'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2026 Sec 8/9; DM-03/DM-41',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'DR-13',
        "description": 'DM + hypoglycemia history + yr>=2017: prefer SGLT2i/GLP-1RA/DPP-4i (DM-25/42)',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) in ("T1DM", "T2DM") and yr >= 2017 and (_has_complication(cm, 'hypoglycemia') or _has_hypoglycemia_hx(pd)) and not (_is_hospice(pd) or _has_pregnancy(pd) or _has_esrd_dialysis(pd))),
        "any_of_cats": ['SGLT2i', 'GLP-1RA', 'DPP-4i', 'GIP_GLP1_RA'],
        "exclude_drugs": [],
        "grade":       0.5,
        "source":      'KDIGO 2022 PP 2.2.1; DM-25/DM-42',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'DR-14',
        "description": 'T2DM + age>=65: prefer low-hypoglycemia-risk agents (all eras, CN guideline 2024, DM-54/55)',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and (lambda age: age is not None and age >= 65)(_get_age(cm, pd))
            and not _is_appropriate_dm_omission(cm, pd)
        ),
        "any_of_cats": ['Biguanides', 'SGLT2i', 'GLP-1RA', 'DPP-4i', 'Alpha_Glucosidase_Inhibitors', 'GIP_GLP1_RA'],
        "exclude_drugs": [],
        "grade":       0.5,
        "source":      'ADA all eras elderly principles; DM-54/DM-55',
        "domain":      'glycemic_first_line',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'DR-15',
        "description": 'T2DM + CKD G3+ + eGFR>15 + yr>=2017: GLP-1RA for glycaemic control (ESC Class I, DM-26)',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2017
            and _has_ckd_g3plus(cm)
            and (lambda egfr: egfr is None or float(egfr) > 15)(_get_egfr(cm))
            and not _is_hospice(pd)
        ),
        "any_of_cats": ['GLP-1RA', 'GIP_GLP1_RA'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ESC CVD in Diabetes 2023 Sec 9 (Class I, Level A); DM-26',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'DR-16',
        "description": 'T2DM + ASCVD + yr>=2017: GLP-1RA explicitly first-line (supplements DR-03, DM-39)',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2017
            and _get_dm_comp(cm, "cardiovascular")
            and not _is_appropriate_dm_omission(cm, pd)
        ),
        "any_of_cats": ['GLP-1RA', 'GIP_GLP1_RA'],
        "exclude_drugs": [],
        "grade":       0.5,
        "source":      'ESC CVD 2023 5.3.1.2; DM-39',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'DR-17',
        "description": 'T2DM + HF + eGFR>30 + yr>=2016: Metformin should be retained (FDA 2016, ESC 2023, DM-33/34)',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2016
            and _get_dm_comp(cm, "heart_failure")
            and (lambda egfr: egfr is None or float(egfr) > 30)(_get_egfr(cm))
            and not (cm.get("kidney") or {}).get("aki_suspected", False)
            and not _is_hospice(pd)
        ),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ESC CVD in Diabetes 2023 7.5.5 (Class IIa); ADA 2016 label update; DM-33/DM-34',
        "domain":      'glycemic_first_line',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'DR-18',
        "description": 'T2DM + HF + yr>=2019: SGLT2i (ESC 2019/DAPA-HF, extends DR-05 by 1yr, DM-27/40)',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2019
            and _get_dm_comp(cm, "heart_failure")
            and not pd.get("in_dka_risk_state", False)
            and not _is_hospice(pd)
        ),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2019 update; DAPA-HF NEJM 2019; DM-27/DM-40',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'DR-20',
        "description": 'T1DM + BMI>=25 + eGFR>=30 + yr>=2016: Metformin adjunct (REMOVAL 2017, DM-13/18)',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T1DM"
            and yr >= 2016
            and (cm.get("bmi", 0) >= 25 or pd.get("bmi", 0) >= 25)
            and (lambda egfr: egfr is None or float(egfr) >= 30)(_get_egfr(cm))
            and not (cm.get("kidney") or {}).get("aki_suspected", False)
            and not _is_hospice(pd)
        ),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": [],
        "grade":       0.5,
        "source":      'ADA Standards of Care 2016 Sec 5; REMOVAL trial LANCET 2017; DM-13/DM-18',
        "domain":      'glycemic_first_line',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'DR-21',
        "description": 'T2DM + eGFR 15-30 + no HF: TZD viable option in advanced CKD (ADA Consensus 2022, DM-37)',
        "condition":   lambda cm, pd, yr: (_has_ckd_g3plus(cm)
            and 
            _get_dm_type(cm) == "T2DM"
            and (lambda egfr: egfr is not None and 15 <= float(egfr) < 30)(_get_egfr(cm))
            and not _get_dm_comp(cm, "heart_failure")
            and not _is_hospice(pd)
        ),
        "any_of_cats": ['Thiazolidinediones'],
        "exclude_drugs": [],
        "grade":       0.5,
        "source":      'ADA Consensus Reports & Supplements 2022 - Advanced CKD; DM-37',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
]


# ── 药物上市年份表(时代穿越检测)──────────────────────────────────────────
# try:
from agents.config import DrugEraConfig  # [TEMPORAL-CONSISTENCY-V1] single temporal registry
# except ImportError:
#     DRUG_APPROVAL_YEAR: Dict[str, int] = {
#         "dapagliflozin": 2012, "canagliflozin": 2013, "empagliflozin": 2014,
#         "ertugliflozin": 2017, "semaglutide": 2017, "liraglutide": 2010,
#         "dulaglutide": 2014, "albiglutide": 2014, "exenatide": 2005,
#         "sitagliptin": 2006, "saxagliptin": 2009, "linagliptin": 2011,
#         "alogliptin": 2013, "pioglitazone": 1999, "rosiglitazone": 1999,
#         "metformin": 1994, "tirzepatide": 2022, "finerenone": 2021,
#     }


# ═══════════════════════════════════════════════════════════════════════════════
# 三、Path-C 辅助函数(读取真实 JSON 字段结构)
# ═══════════════════════════════════════════════════════════════════════════════
#
# 基于实测 JSON 结构(来自 clinical_metrics 字段审查):
#   cm["dm_complications"]["cardiovascular"] = True/False
#   cm["dm_complications"]["heart_failure"]  = True/False
#   cm["kidney"]["ckd_g_stage"]              = "G1"/"G2"/"G3a"/"G3b"/"G4"/"G5"
#   cm["kidney"]["uacr"]                     = float (mg/g)
#   cm["kidney"]["ckd_a_stage"]              = "A1"/"A2"/"A3"
#   cm["kidney"]["egfr"]                     = float
# ─────────────────────────────────────────────────────────────────────────────

def _lab_get(lab, key: str, default=None):
    """兼容 lab 为 dict 或 dataclass/object。"""
    if isinstance(lab, dict):
        return lab.get(key, default)
    return getattr(lab, key, default)


def _to_float(val) -> Optional[float]:
    try:
        if val is None:
            return None
        return float(val)
    except (ValueError, TypeError):
        return None


def _is_hospice(patient_data: Dict) -> bool:
    """
    True if this patient was discharged to hospice / palliative / comfort care / expired.
    In end-of-life scenarios strict glycemic control is not clinically indicated;
    absence of glucose-lowering agents is appropriate and must not be penalised by GAS.
    """
    candidates = [
        patient_data.get("discharge_type"),
        patient_data.get("discharge_disposition"),
        patient_data.get("discharge_location"),
        (patient_data.get("admission_info") or {}).get("discharge_location") if isinstance(patient_data.get("admission_info"), dict) else None,
    ]
    text = " | ".join(str(x) for x in candidates if x is not None).lower().strip()
    if not text:
        return False
    return any(k in text for k in ("hospice", "comfort care", "palliative", "expired", "comfort"))


def _is_appropriate_dm_omission(cm: Dict, patient_data: Dict) -> bool:
    """
    True when omitting glucose-lowering pharmacotherapy is clinically appropriate.
    Used in DM_RULES to prevent penalising patients for whom no drug is the right answer.
    Three criteria (consistent with EmptyGTClassifier.SAFE_DEESCALATION and §2.67 CII):
      1. ESRD (eGFR < 15): most oral DM agents contraindicated; insulin complex at end-stage.
      2. AKI + eGFR < 30: safety hold indicated (HC-01/02 in safety_checker.py).
      3. Hospice / palliative / expired: strict glycemic targets not appropriate at end-of-life.
    """
    # Criterion 1: ESRD
    egfr = _get_egfr(cm)
    if egfr is not None and egfr < 15:
        return True
    # Criterion 2: AKI + severe CKD
    kidney = cm.get("kidney") or {}
    if kidney.get("aki_suspected", False) and egfr is not None and egfr < 30:
        return True
    # Criterion 3: End-of-life
    if _is_hospice(patient_data):
        return True
    return False

def _get_lvef(pd: Dict) -> Optional[float]:
    cm = pd.get("clinical_metrics", {}) or {}
    # 优先 cm 顶层
    v = _to_float(cm.get("lvef"))
    if v is not None:
        return v
    # 次选 cm.cardiac 子结构
    cardiac = cm.get("cardiac") if isinstance(cm.get("cardiac"), dict) else {}
    v = _to_float(cardiac.get("lvef") or cardiac.get("ejection_fraction"))
    if v is not None:
        return v
    # 最后从化验/检查文本中回退
    for lab in pd.get("all_lab_results", []) or []:
        label = str(_lab_get(lab, "label", "") or "").lower()
        if "lvef" in label or "ejection fraction" in label:
            v = _to_float(_lab_get(lab, "value"))
            if v is not None:
                return v
    return None

def _get_potassium(pd: Dict) -> Optional[float]:
    # 优先从化验单提取 (MIMIC itemid: 50971, 52032)
    for lab in pd.get("all_lab_results", []) or []:
        itemid = _lab_get(lab, "itemid")
        label = str(_lab_get(lab, "label", "") or "").lower()
        if itemid in [50971, 52032] or "potassium" in label:
            v = _to_float(_lab_get(lab, "value"))
            if v is not None:
                return v

    cm = pd.get("clinical_metrics", {}) or {}
    electrolytes = cm.get("electrolytes") if isinstance(cm.get("electrolytes"), dict) else {}
    return _to_float(cm.get("potassium") or electrolytes.get("potassium"))

def _get_bmi_val(pd: Dict) -> Optional[float]:
    return _to_float(pd.get("bmi") or (pd.get("clinical_metrics", {}) or {}).get("bmi"))

def _get_fbg_val(pd: Dict) -> Optional[float]:
    cm = pd.get("clinical_metrics", {}) or {}
    # 兼容两种可能的键名
    val = _to_float(cm.get("fasting_glucose") or cm.get("fasting_glucose_mg_dl"))
    if val is not None:
        return val

    gluc = cm.get("glucose") if isinstance(cm.get("glucose"), dict) else {}
    val = _to_float(gluc.get("fasting_glucose"))
    if val is not None:
        return val

    for lab in pd.get("all_lab_results", []) or []:
        label = str(_lab_get(lab, "label", "") or "").lower()
        if "glucose" in label and "fasting" in label:
            v = _to_float(_lab_get(lab, "value"))
            if v is not None:
                return v
    return None


def _build_rule_eval_context(cm: Dict, patient_data: Dict, admission_year: int) -> Dict:
    """
    构建规则评估上下文:把 run_extraction 的真实存储结构映射到规则常用键。
    目的:避免规则直接读 cm 顶层时因字段嵌套而静默失效。

    [Bug-A fix] on_systemic_steroids: run_extraction 没有专门提取该字段。
      原实现 `out.get("on_systemic_steroids")` 在 if-None 分支里永远返回 None,
      逻辑上等价于恒 False。修复:从 all_medications(discharge + prior)中
      检测常用全身性糖皮质激素名称来推断。这使 DR-09(激素→胰岛素优先)可以正常触发。

    [Bug-B fix] has_hypertension: CPR-71 用 pd.get("has_hypertension") 但该字段
      从未被提取管线设置。修复:从 all_diagnoses ICD 编码推断
      (ICD-9: 401.x/402.x, ICD-10: I10/I11/I12/I13)。
    """
    out = dict(cm or {})
    glucose = out.get("glucose") if isinstance(out.get("glucose"), dict) else {}
    dm_comp = out.get("dm_complications") if isinstance(out.get("dm_complications"), dict) else {}

    out["admission_year"] = admission_year if admission_year else out.get("admission_year")

    # ── 高血糖相关别名(规则常用 fasting/random)────────────────────────────────
    if out.get("fasting_glucose") is None:
        out["fasting_glucose"] = _get_fbg_val({"clinical_metrics": out, "all_lab_results": patient_data.get("all_lab_results", [])})
    if out.get("fasting_glucose_mg_dl") is None:
        out["fasting_glucose_mg_dl"] = out.get("fasting_glucose")
    if out.get("random_glucose_mg_dl") is None and isinstance(glucose, dict):
        out["random_glucose_mg_dl"] = glucose.get("max_glucose") or glucose.get("mean_glucose")

    # ── 低血糖史别名(validator 主路径在 dm_complications / glucose 内)────────
    if out.get("hypo_history") is None:
        out["hypo_history"] = bool(
            dm_comp.get("hypoglycemia")
            or dm_comp.get("hypoglycemia_history")
            or (isinstance(glucose, dict) and (glucose.get("hypoglycemia_events") or 0) > 0)
        )
    if out.get("hypoglycemia_history") is None:
        out["hypoglycemia_history"] = out.get("hypo_history")

    # ── [Bug-A fix] on_systemic_steroids:从用药列表推断 ──────────────────────
    # run_extraction 没有专门提取此字段;clinical_phenotyper.py 的 flat["on_systemic_steroids"]
    # 也只在推断表型时用一次,不会回写到 patient_data["clinical_metrics"]。
    # 因此 DR-09(激素→胰岛素优先)从未触发过。修复:在 context 构建时检测用药列表。
    _STEROID_KEYWORDS = (
        "prednisone", "prednisolone", "methylprednisolone", "dexamethasone",
        "hydrocortisone", "triamcinolone", "budesonide systemic", "betamethasone",
        "cortisone", "fludrocortisone",
    )
    if not out.get("on_systemic_steroids"):
        _steroid_detected = False
        # 检查 all_medications(含 discharge + prior + admission_order)
        for med in patient_data.get("all_medications", []) or []:
            drug_name = ""
            if isinstance(med, dict):
                drug_name = str(med.get("drug_name", "") or "").lower()
            elif isinstance(med, str):
                drug_name = med.lower()
            if any(kw in drug_name for kw in _STEROID_KEYWORDS):
                _steroid_detected = True
                break
        # 也检查旧版字符串列表(prior_medications)
        if not _steroid_detected:
            for drug_name in patient_data.get("prior_medications", []) or []:
                if any(kw in str(drug_name).lower() for kw in _STEROID_KEYWORDS):
                    _steroid_detected = True
                    break
        # 最后检查 dm_complications.on_corticosteroids(若未来提取管线填充)
        if not _steroid_detected:
            _steroid_detected = bool(dm_comp.get("on_corticosteroids"))
        out["on_systemic_steroids"] = _steroid_detected
        # 同步到 patient_data 顶层,供 DR-09 的 pd.get("on_systemic_steroids") 读取
        if _steroid_detected and not patient_data.get("on_systemic_steroids"):
            patient_data["on_systemic_steroids"] = True

    # ── [Bug-B fix] has_hypertension:从 ICD 诊断推断 ────────────────────────
    # CPR-71(DM+HTN→ACEi/ARB)用 pd.get("has_hypertension"),但提取管线从未设置此字段。
    # 修复:从 all_diagnoses 中检测高血压 ICD 编码(ICD-9: 401/402, ICD-10: I10-I13/R03)。
    _HTN_ICD9_PREFIXES  = ("401", "402", "403", "404", "405")
    _HTN_ICD10_PREFIXES = ("I10", "I11", "I12", "I13", "I15", "R03")
    if not patient_data.get("has_hypertension"):
        _htn_detected = False
        for diag in patient_data.get("all_diagnoses", []) or []:
            if not isinstance(diag, dict):
                continue
            code    = str(diag.get("icd_code", "") or "").strip().upper()
            version = diag.get("icd_version", 9)
            if not code:
                continue
            if version == 10:
                if any(code.startswith(p) for p in _HTN_ICD10_PREFIXES):
                    _htn_detected = True
                    break
            else:
                if any(code.startswith(p) for p in _HTN_ICD9_PREFIXES):
                    _htn_detected = True
                    break
        if _htn_detected:
            # 写入 patient_data 顶层,供 CPR-71 的 pd.get("has_hypertension") 读取
            patient_data["has_hypertension"] = True
        out["has_hypertension"] = _htn_detected

    # ── [Tier-1 fix] inpatient 恒真 ─────────────────────────────────────────
    # 本队列全部为住院患者(MIMIC-III/IV + eICU inpatient cohort)。
    # "因急性疾病住院""住院患者"等条件恒为True,无需额外门控。
    # 写入 cm 以供规则条件直接引用 cm.get("is_inpatient")。
    if out.get("is_inpatient") is None:
        out["is_inpatient"] = True

    # ── [Tier-1 fix] IV 胰岛素推断 ─────────────────────────────────────────
    # 从 all_medications 中检测 route ∈ {IV, Intravenous} 且药名含 insulin。
    # medication_extractor.py 三个库都填了 route(MIMIC 用 route,eICU 用 routeadmin)。
    if not out.get("on_iv_insulin"):
        _iv_insulin_detected = False
        _IV_INSULIN_KEYWORDS = ("insulin",)
        _IV_ROUTE_NAMES = {"iv", "intravenous", "iv drip", "iv infusion", "continuous infusion"}
        for med in patient_data.get("all_medications", []) or []:
            if isinstance(med, dict):
                _dn = str(med.get("drug_name", "") or "").lower()
                _rt = str(med.get("route", "") or "").lower().strip()
            else:
                continue
            if any(kw in _dn for kw in _IV_INSULIN_KEYWORDS) and _rt in _IV_ROUTE_NAMES:
                _iv_insulin_detected = True
                break
        # 也检查 prior_medications(字符串列表,无 route 信息,跳过)
        if not _iv_insulin_detected:
            _iv_insulin_detected = bool(out.get("iv_insulin_active"))
        out["on_iv_insulin"] = _iv_insulin_detected
        if _iv_insulin_detected and not patient_data.get("on_iv_insulin"):
            patient_data["on_iv_insulin"] = True

    # ── [Tier-1 fix] ICU/危重状态近似 ─────────────────────────────────────
    # 精确 ICU 判定需要 careunit(不在 pd 中),短期用已有 cm 信号近似:
    #   vasopressor_use=True → 几乎必然 ICU
    #   aki_kdigo_stage>=2 → 常需 ICU 监护
    #   mean_glucose>250 + on_iv_insulin → ICU-level 胰岛素输注
    # 写入 cm.get("is_icu_approx"),规则条件可引用。
    if out.get("is_icu_approx") is None:
        _icu_approx = False
        if dm_comp.get("vasopressor_use"):
            _icu_approx = True
        elif isinstance(out.get("kidney"), dict) and (out["kidney"].get("aki_kdigo_stage") or 0) >= 2:
            _icu_approx = True
        elif isinstance(glucose, dict) and (glucose.get("mean_glucose") or 0) > 250 and out.get("on_iv_insulin"):
            _icu_approx = True
        out["is_icu_approx"] = _icu_approx

    # ── BMI / K / LVEF 别名 ───────────────────────────────────────────────────
    if out.get("bmi") is None:
        out["bmi"] = _get_bmi_val(patient_data)
    if out.get("potassium") is None:
        out["potassium"] = _get_potassium(patient_data)
    if out.get("lvef") is None:
        out["lvef"] = _get_lvef(patient_data)

    # ── HbA1c 与随机血糖别名 ─────────────────────────────────────────────────
    if out.get("hba1c") is None:
        out["hba1c"] = _to_float(out.get("hba1c") or (out.get("glucose") or {}).get("hba1c"))
    if out.get("random_glucose_mg_dl") is None and isinstance(glucose, dict):
        out["random_glucose_mg_dl"] = glucose.get("max_glucose") or glucose.get("mean_glucose")

    # ── 年龄别名 ──────────────────────────────────────────────────────────────
    if out.get("age") is None and patient_data.get("age") is not None:
        out["age"] = patient_data.get("age")

    return out


def _get_dm_comp(cm: Dict, key: str) -> bool:
    """
    从 cm["dm_complications"] 读取并发症标志。
    支持 cardiovascular / heart_failure / nephropathy / hypoglycemia 等 key。
    """
    dm_comp = cm.get("dm_complications") or {}
    if not isinstance(dm_comp, dict):
        return False

    # 直接匹配
    val = dm_comp.get(key)
    if val is not None:
        if isinstance(val, bool):
            return val
        if str(val).lower() in ("true", "1", "yes"):
            return True

    # 字段别名映射
    _ALIASES: Dict[str, List[str]] = {
        "cardiovascular": ["cardiovascular", "has_cvd", "ascvd", "cv_disease"],
        "heart_failure":  ["heart_failure", "hf", "has_hf", "chf"],
        "nephropathy":    ["nephropathy", "ckd", "diabetic_nephropathy"],
        "hypoglycemia":   ["hypoglycemia"],
    }
    for alias in _ALIASES.get(key, []):
        val = dm_comp.get(alias)
        if val is None:
            continue
        if isinstance(val, bool):
            return val
        if str(val).lower() in ("true", "1", "yes"):
            return True
    return False


def _get_age(cm: Dict, patient_data: Dict) -> Optional[int]:
    """读取患者年龄。优先 clinical_metrics,兜底 patient_data 顶层字段。"""
    for src in [cm, patient_data]:
        for key in ("age", "age_at_admission", "patient_age"):
            v = src.get(key)
            if v is not None:
                try:
                    return int(float(v))
                except (ValueError, TypeError):
                    pass
    return None


def _has_ckd_g3plus(cm: Dict) -> bool:
    """
    CKD G3+(eGFR < 60 或 ckd_g_stage 为 G3a/G3b/G4/G5)。
    G3+ 意味着有 KDIGO 推荐 RAAS 阻断的基础。
    """
    kidney = cm.get("kidney") or {}
    stage  = str(kidney.get("ckd_g_stage") or "").strip()
    if stage and stage not in ("", "G1", "G2"):
        return True
    egfr = _get_egfr(cm)
    if egfr is not None and egfr < 60:
        return True
    # fallback: dm_complications.nephropathy
    return _get_dm_comp(cm, "nephropathy")


def _has_albuminuria(cm: Dict) -> bool:
    """
    微量蛋白尿(UACR > 30 mg/g)或更严重的蛋白尿。
    CKD A2/A3 阶段同样视为有蛋白尿。
    """
    kidney = cm.get("kidney") or {}
    uacr   = kidney.get("uacr")
    if uacr is not None:
        try:
            return float(uacr) > 30.0
        except (ValueError, TypeError):
            pass
    ckd_a = str(kidney.get("ckd_a_stage") or "").strip()
    if ckd_a in ("A2", "A3"):
        return True
    return False


def _has_ckd_with_proteinuria(cm: Dict) -> bool:
    """CKD + 蛋白尿:ACEi/ARB / Finerenone 的核心适应证。

    注意这里必须是 AND:仅有 CKD 或仅有蛋白尿都不应触发该组合适应证,
    否则会大幅抬高规则适用分母,系统性拉低 GAS。
    """
    return _has_ckd_g3plus(cm) and _has_albuminuria(cm)


# ═══════════════════════════════════════════════════════════════════════════════
# 三·五、ICD-based 表型辅助函数(Day 2 新增)
# ═══════════════════════════════════════════════════════════════════════════════
#
# 设计原则:
#   - lambda 签名 (cm, pd, yr) 中 pd 已包含 all_diagnoses,可直接 ICD 匹配
#   - 单 patient 缓存避免重复遍历(pd["_icd_codes_cache"])
#   - 同时支持 ICD-9(数字前缀)和 ICD-10(字母+数字前缀),去除小数点
#   - 临床保守原则:LVEF 缺失时 HFrEF/HFmrEF/HFpEF gate 不触发
#   - "history of" 语义为liberal:任何过去或当前诊断均计入
# ─────────────────────────────────────────────────────────────────────────────

def _get_icd_codes(pd: Dict) -> Set[str]:
    """提取该患者所有诊断的 ICD 代码(去除小数点+大写化),单 patient 缓存。"""
    cache_key = "_icd_codes_cache"
    if cache_key in pd:
        return pd[cache_key]
    codes: Set[str] = set()
    for diag in pd.get("all_diagnoses", []) or []:
        if not isinstance(diag, dict):
            continue
        code = str(diag.get("icd_code", "") or "").strip().upper().replace(".", "")
        if code:
            codes.add(code)
    pd[cache_key] = codes
    return codes


def _icd_starts_with(pd: Dict, prefixes: Set[str]) -> bool:
    """检查患者ICD codes中是否有任何code以指定前缀开头。"""
    codes = _get_icd_codes(pd)
    return any(c.startswith(p) for c in codes for p in prefixes)


# ── 心血管亚型 ──────────────────────────────────────────────────────────────────

def _has_prior_mi(pd: Dict) -> bool:
    """
    既往心肌梗死(任何时点的MI诊断均计入)。
    ICD-9: 410.x (急性MI,会保留为 dx 即使已转慢性), 412 (陈旧性MI)
    ICD-10: I21.x (急性MI), I22.x (后续MI), I25.2 (陈旧性MI)
    用于:所有 "prior MI" / "history of MI" / "post-MI" 规则
    """
    codes = _get_icd_codes(pd)
    if any(c.startswith("410") for c in codes): return True
    if any(c.startswith("412") for c in codes): return True
    if any(c.startswith("I21") for c in codes): return True
    if any(c.startswith("I22") for c in codes): return True
    if any(c.startswith("I252") for c in codes): return True
    return False


def _has_acute_mi(pd: Dict) -> bool:
    """
    急性心肌梗死(本次入院/近期)。
    与 _has_prior_mi 区别:仅 410.x / I21.x(明确 acute coding),不含陈旧性 412 / I25.2。
    注:MIMIC 编码不区分 primary vs secondary dx,无法精确隔离 "本次发作",
    故此函数返回 "any acute MI coding present",应配合时间相关条件使用。
    """
    codes = _get_icd_codes(pd)
    if any(c.startswith("410") for c in codes): return True
    if any(c.startswith("I21") for c in codes): return True
    return False


def _has_acs(pd: Dict) -> bool:
    """
    急性冠脉综合征(ACS)= 急性MI + 不稳定心绞痛。
    ICD-9: 410.x (AMI), 411.0 (postinfarction syndrome), 411.1 (intermediate coronary syndrome / unstable angina)
    ICD-10: I20.0 (unstable angina), I21.x (acute MI), I24.x (other acute IHD)
    """
    codes = _get_icd_codes(pd)
    if any(c.startswith("410") for c in codes): return True
    if any(c.startswith("411") for c in codes): return True
    if any(c.startswith("I20.0") or c.startswith("I200") for c in codes): return True
    if any(c.startswith("I21") for c in codes): return True
    if any(c.startswith("I24") for c in codes): return True
    return False


def _has_cad(pd: Dict) -> bool:
    """
    冠状动脉疾病(CAD/CHD),含稳定性心绞痛、慢性 IHD、PCI/CABG史。
    ICD-9: 413.x (angina pectoris), 414.x (chronic ischemic heart disease, incl. atherosclerosis)
    ICD-10: I20.x (angina pectoris), I25.x (chronic ischemic heart disease)
    """
    codes = _get_icd_codes(pd)
    if any(c.startswith("413") for c in codes): return True
    if any(c.startswith("414") for c in codes): return True
    if any(c.startswith("I20") for c in codes): return True
    if any(c.startswith("I25") for c in codes): return True
    return False


# ── 脑血管亚型 ──────────────────────────────────────────────────────────────────

def _has_stroke_hx(pd: Dict) -> bool:
    """
    缺血性脑卒中或卒中后遗症。
    ICD-9: 433.x (precerebral artery occlusion), 434.x (cerebral artery occlusion),
           436 (acute, but ill-defined CVD), 438 (late effects of CVD)
    ICD-10: I63.x (cerebral infarction), I65.x (precerebral occlusion without infarction),
            I69.x (sequelae of CVD)
    """
    codes = _get_icd_codes(pd)
    for prefix in ("433","434","436","438"):
        if any(c.startswith(prefix) for c in codes): return True
    if any(c.startswith("I63") for c in codes): return True
    if any(c.startswith("I65") for c in codes): return True
    if any(c.startswith("I69") for c in codes): return True
    return False


def _has_tia(pd: Dict) -> bool:
    """
    短暂性脑缺血发作(TIA)。
    ICD-9: 435.x (transient cerebral ischemia)
    ICD-10: G45.x (transient cerebral ischemic attacks)
    """
    codes = _get_icd_codes(pd)
    if any(c.startswith("435") for c in codes): return True
    if any(c.startswith("G45") for c in codes): return True
    return False


# ── 心律失常 ────────────────────────────────────────────────────────────────────

def _has_af(pd: Dict) -> bool:
    """
    心房颤动(含房扑)。
    ICD-9: 427.31 (atrial fibrillation), 427.32 (atrial flutter)
    ICD-10: I48.x (atrial fibrillation and flutter)
    """
    codes = _get_icd_codes(pd)
    if any(c.startswith("4273") for c in codes): return True   # covers 42731, 42732
    if any(c.startswith("I48") for c in codes): return True
    return False


def _has_ventricular_arrhythmia(pd: Dict) -> bool:
    """
    室性心律失常(VT/VF/复杂室性早搏)。
    ICD-9: 427.1 (paroxysmal VT), 427.4 (VF/V flutter), 427.5 (cardiac arrest)
    ICD-10: I47.0 (re-entry VA), I47.2 (VT), I49.0 (VF/V flutter), I49.3 (premature ventricular depolarization)
    """
    codes = _get_icd_codes(pd)
    for prefix in ("4271","4274","4275"):
        if any(c.startswith(prefix) for c in codes): return True
    if any(c.startswith("I47") for c in codes): return True
    if any(c.startswith("I490") for c in codes): return True
    if any(c.startswith("I493") for c in codes): return True
    return False


# ── 外周血管 ────────────────────────────────────────────────────────────────────

def _has_pad(pd: Dict) -> bool:
    """
    外周动脉疾病(PAD/LEAD/PVD),含下肢动脉硬化、跛行、栓塞、坏疽。
    ICD-9: 440.2x (atherosclerosis of arteries of extremities), 440.3 (bypass grafts),
           443.x (other peripheral vascular disease), 444.2x (arterial embolism extremities)
    ICD-10: I70.2 (atherosclerosis of native arteries of extremities), I70.3-9 (bypass grafts),
            I73.x (other peripheral vascular diseases), I74.3-4 (arterial embolism extremities)
    """
    codes = _get_icd_codes(pd)
    for prefix in ("4402","4403","443","4442"):
        if any(c.startswith(prefix) for c in codes): return True
    if any(c.startswith("I702") for c in codes): return True
    if any(c.startswith("I703") for c in codes): return True
    if any(c.startswith("I709") for c in codes): return True
    if any(c.startswith("I73") for c in codes): return True
    if any(c.startswith("I743") for c in codes): return True
    if any(c.startswith("I744") for c in codes): return True
    return False


# ── 肾脏 ────────────────────────────────────────────────────────────────────────

def _has_esrd_dialysis(pd: Dict) -> bool:
    """
    终末期肾病(ESRD)或正在透析。
    ICD-9: 585.6 (ESRD), V45.11 (dialysis status), V56.x (encounter for dialysis)
    ICD-10: N18.6 (ESRD), Z99.2 (dialysis dependence), Z49.x (encounter for dialysis care)
    亦检查 procedures 含 "dialysis" 字样。
    """
    codes = _get_icd_codes(pd)
    if any(c.startswith("5856") for c in codes): return True
    if any(c.startswith("V4511") for c in codes): return True
    if any(c.startswith("V56") for c in codes): return True
    if any(c.startswith("N186") for c in codes): return True
    if any(c.startswith("Z992") for c in codes): return True
    if any(c.startswith("Z49") for c in codes): return True
    # Fallback: 检查 procedures 中是否有 dialysis 关键字
    for proc in pd.get("all_procedures", []) or []:
        if not isinstance(proc, dict): continue
        name = str(proc.get("procedure_name", "") or proc.get("name", "") or "").lower()
        if "dialysis" in name or "hemodialy" in name or "renal replacement" in name:
            return True
    return False


# ── DM 相关历史 ─────────────────────────────────────────────────────────────────

def _has_hypoglycemia_hx(pd: Dict) -> bool:
    """
    既往严重低血糖事件。
    ICD-9: 251.0 (hypoglycemic coma), 251.1 (other specified hypoglycemia),
           251.2 (hypoglycemia, unspecified), 250.8x (DM with other manifestations 含 hypoglycemia)
    ICD-10: E16.0 (drug-induced hypoglycemia without coma), E16.1 (other hypoglycemia),
            E16.2 (hypoglycemia, unspecified), E10/11.64x (DM with hypoglycemia)
    """
    codes = _get_icd_codes(pd)
    for prefix in ("2510","2511","2512"):
        if any(c.startswith(prefix) for c in codes): return True
    if any(c.startswith("2508") for c in codes): return True
    for prefix in ("E160","E161","E162"):
        if any(c.startswith(prefix) for c in codes): return True
    # E10.64x / E11.64x - DM with hypoglycemia
    if any(c.startswith("E1064") or c.startswith("E1164") for c in codes): return True
    # Fallback: existing dm_comp flag
    cm = pd.get("clinical_metrics", {}) or {}
    return _get_dm_comp(cm, "hypoglycemia")


def _has_gi_bleed_hx(pd: Dict) -> bool:
    """
    消化道出血史(用于aspirin/抗血小板禁忌评估)。
    ICD-9: 531.x-534.x (peptic ulcer with hemorrhage), 578.x (GI bleeding NOS)
    ICD-10: K25.x-K28.x (gastric/duodenal ulcer with hemorrhage), K92.x (other GI hemorrhage)
    """
    codes = _get_icd_codes(pd)
    for prefix in ("531","532","533","534","578"):
        if any(c.startswith(prefix) for c in codes): return True
    for prefix in ("K25","K26","K27","K28","K92"):
        if any(c.startswith(prefix) for c in codes): return True
    return False


def _has_pregnancy(pd: Dict) -> bool:
    """
    妊娠状态(含妊娠糖尿病及孕产期诊断)。
    ICD-9: 630-679 (complications of pregnancy), V22-V24, V27 (normal pregnancy outcomes)
    ICD-10: O00-O9A (pregnancy, childbirth, puerperium), Z33-Z34 (pregnancy state),
            Z36-Z37 (encounter for antenatal screening / outcome of delivery)
    """
    codes = _get_icd_codes(pd)
    # ICD-9: numeric prefix
    for prefix in ("63","64","65","66","67"):
        if any(c.startswith(prefix) and len(c) >= 3 and c[2].isdigit() for c in codes): return True
    for prefix in ("V22","V23","V24","V27"):
        if any(c.startswith(prefix) for c in codes): return True
    # ICD-10: O-block, Z33-Z37
    if any(c.startswith("O") for c in codes): return True
    for prefix in ("Z33","Z34","Z36","Z37"):
        if any(c.startswith(prefix) for c in codes): return True
    return False


def _is_on_iv_insulin(pd: Dict) -> bool:
    """判断患者是否正在使用静脉胰岛素输注。
    从 all_medications 中检测 route ∈ {IV, Intravenous} 且药名含 insulin。
    也检查 cm.on_iv_insulin(由 _build_rule_eval_context 注入)。"""
    cm = pd.get("clinical_metrics", {}) or {}
    if cm.get("on_iv_insulin"):
        return True
    _IV_ROUTE_NAMES = {"iv", "intravenous", "iv drip", "iv infusion", "continuous infusion"}
    for med in pd.get("all_medications", []) or []:
        if not isinstance(med, dict):
            continue
        _dn = str(med.get("drug_name", "") or "").lower()
        _rt = str(med.get("route", "") or "").lower().strip()
        if "insulin" in _dn and _rt in _IV_ROUTE_NAMES:
            return True
    return False


def _is_icu_approx(pd: Dict) -> bool:
    """ICU/危重状态近似判断(精确 careunit 不在 pd 中)。
    用已有 cm 信号近似:vasopressor_use / aki_kdigo_stage>=2 / IV胰岛素+高血糖。"""
    cm = pd.get("clinical_metrics", {}) or {}
    if cm.get("is_icu_approx"):
        return True
    dm_comp = cm.get("dm_complications", {}) or {}
    if dm_comp.get("vasopressor_use"):
        return True
    kidney = cm.get("kidney", {}) or {}
    if (kidney.get("aki_kdigo_stage") or 0) >= 2:
        return True
    glucose = cm.get("glucose", {}) or {}
    if (glucose.get("mean_glucose") or 0) > 250 and _is_on_iv_insulin(pd):
        return True
    return False


def _requires_insulin(cm: Dict, pd: Dict) -> bool:
    """判断患者是否需要胰岛素治疗。
    T1DM 始终需要；T2DM 在以下情况需要：
    - 已处方胰岛素（任意途径）
    - 严重高血糖（max_glucose > 300 或 hba1c >= 10）
    - 正在使用 IV 胰岛素
    """
    if _get_dm_type(cm) == "T1DM":
        return True
    if _is_on_iv_insulin(pd):
        return True
    patient_drugs = _get_patient_drug_names(pd)
    if any("insulin" in d for d in patient_drugs):
        return True
    glucose = cm.get("glucose") or {}
    if (glucose.get("max_glucose") or 0) > 300:
        return True
    hba1c = cm.get("hba1c")
    if hba1c is not None and hba1c >= 10:
        return True
    return False


# ── HF 亚型(LVEF threshold gates) ────────────────────────────────────────────
# 临床保守原则(per memory):LVEF 缺失 → HFrEF/HFmrEF/HFpEF gate 不触发。
# 现有 _get_dm_comp(cm, "heart_failure") 用于通用 HF 检测,本组用于 LVEF-分层规则。

def _has_hfref(cm: Dict, pd: Dict) -> bool:
    """
    HFrEF:LVEF ≤ 40% 且患者有HF诊断。
    LVEF 缺失 → 不触发(避免误标 HFrEF 致严重药物推荐)。
    """
    if not _get_dm_comp(cm, "heart_failure"):
        return False
    lvef = _get_lvef(pd)
    return lvef is not None and lvef <= 40.0


def _has_hfmref(cm: Dict, pd: Dict) -> bool:
    """
    HFmrEF:LVEF 41-49% 且患者有HF诊断。
    LVEF 缺失 → 不触发。
    """
    if not _get_dm_comp(cm, "heart_failure"):
        return False
    lvef = _get_lvef(pd)
    return lvef is not None and 40.0 < lvef < 50.0


def _has_hfpef(cm: Dict, pd: Dict) -> bool:
    """
    HFpEF:LVEF ≥ 50% 且患者有HF诊断。
    LVEF 缺失 → 不触发。
    """
    if not _get_dm_comp(cm, "heart_failure"):
        return False
    lvef = _get_lvef(pd)
    return lvef is not None and lvef >= 50.0


# ───────────────────────────────────────────────────────────────────────────────
# 药物类 helper（_is_on_med_class 系列 + _has_discharge_med + _has_dka_hhs）
# 供 LLM 翻译后的 raw_condition 引用，以及 FEATURE_REGISTRY 中的 getter。
# 匹配逻辑：从 discharge_medications（药名字符串列表）中做大小写不敏感的
# 子串匹配，对照 config.py 的 DIABETES_MEDICATIONS 类别药名表。
# ───────────────────────────────────────────────────────────────────────────────

# 延迟加载药物分类表（避免循环 import）
_DM_DRUG_TABLE: Optional[Dict[str, List[str]]] = None

def _load_dm_drug_table() -> Dict[str, List[str]]:
    global _DM_DRUG_TABLE
    if _DM_DRUG_TABLE is not None:
        return _DM_DRUG_TABLE
    try:
        from agents.config import DIABETES_MEDICATIONS
        _DM_DRUG_TABLE = {
            k: [str(v).lower() for v in (vals if isinstance(vals, list) else [vals])]
            for k, vals in DIABETES_MEDICATIONS.items()
        }
    except Exception:
        _DM_DRUG_TABLE = {}
    return _DM_DRUG_TABLE


def _get_patient_drug_names(pd: Dict) -> List[str]:
    """从 patient_data 收集所有药名（小写）。"""
    names: List[str] = []
    for key in ("discharge_medications", "all_medications"):
        meds = pd.get(key, []) or []
        for m in meds:
            if isinstance(m, dict):
                dn = m.get("drug_name", "") or ""
            else:
                dn = str(m or "")
            dn = dn.strip().lower()
            if dn:
                names.append(dn)
    return names


def _is_on_med_class(pd: Dict, class_name: str) -> bool:
    """判断患者是否正在使用某类药物。
    class_name 对齐 config.py 的 DIABETES_MEDICATIONS 键名
    （如 'Sulfonylureas', 'GLP-1RA', 'DPP-4i', 'SGLT2i' 等）。
    """
    table = _load_dm_drug_table()
    class_lower = class_name.lower()
    # 模糊匹配类别键
    candidates = None
    for k, v in table.items():
        if k.lower() == class_lower:
            candidates = v
            break
    if candidates is None:
        return False
    patient_drugs = _get_patient_drug_names(pd)
    for drug in patient_drugs:
        for ref in candidates:
            # 双向子串匹配：药名含参考名 或 参考名含药名
            if ref in drug or drug in ref:
                return True
    return False


def _has_discharge_med(pd: Dict, class_name: str) -> bool:
    """检查出院带药中是否有某类药物（_is_on_med_class 的语义别名）。"""
    return _is_on_med_class(pd, class_name)


def _is_on_sulfonylurea(pd: Dict) -> bool:
    """是否正在使用磺脲类药物。"""
    return _is_on_med_class(pd, "Sulfonylureas")


def _is_on_glp1_ra(pd: Dict) -> bool:
    """是否正在使用 GLP-1RA 类药物。"""
    return _is_on_med_class(pd, "GLP-1RA")


# 别名：代码库里存在 _is_on_glp1ra 和 _is_on_glp1_ra 两种命名
_is_on_glp1ra = _is_on_glp1_ra


def _is_on_gip_glp1ra(pd: Dict) -> bool:
    """是否正在使用 GIP/GLP-1 双重激动剂（如 Tirzepatide）。"""
    return _is_on_med_class(pd, "GIP_GLP1_RA")


def _is_on_dual_gip_glp1_ra(pd: Dict) -> bool:
    """是否同时使用 GIP/GLP-1 RA 和传统 GLP-1RA（联合用药）。"""
    return _is_on_gip_glp1ra(pd) and _is_on_glp1_ra(pd)


def _has_dka_hhs(pd: Dict) -> bool:
    """糖尿病酮症酸中毒/高渗高血糖状态 判断。
    从 ICD 编码（E1x.x0/E1x.x1 DKA, E1x.x5 HHS）或 clinical_metrics 判断。
    """
    # ICD 编码路径
    icds = _get_icd_codes(pd)
    for code in icds:
        code_upper = code.upper()
        # DKA: E10.10, E10.11, E11.10, E11.11, E13.10, E13.11 等
        if any(code_upper.startswith(p) for p in ("E10.1", "E11.1", "E13.1")):
            return True
        # HHS: E10.5, E11.5, E13.5
        if any(code_upper.startswith(p) for p in ("E10.5", "E11.5", "E13.5")):
            return True
    # clinical_metrics 路径
    cm = pd.get("clinical_metrics", {}) or {}
    dm_comp = cm.get("dm_complications", {}) or {}
    if dm_comp.get("dka") or dm_comp.get("dka_hhs") or dm_comp.get("hhs"):
        return True
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# 三·五(END)-- 16 个新 helpers 完成
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# 四、Path-C:并发症药物条件性评分规则表(CPR-01~CPR-05)
# ═══════════════════════════════════════════════════════════════════════════════
#
# 每条规则格式:
#   id              - 规则 ID(论文 Table S4 溯源)
#   description     - 临床含义
#   condition       - lambda(cm, patient_data, year) -> bool
#                     该规则是否适用于该患者(触发评估)
#   any_of_cats     - 满足条件的药物类别集合(OR 逻辑,有任意一类即得满分)
#   exclude_drugs   - 需要排除的具体药物名(小写),防止类别混用误判
#                     例如 Antiplatelet 类别中的 Heparin(抗凝而非抗板)
#   grade           - 满足时的得分(1.0=Grade A 等效,0.5=Grade B 等效)
#   source          - 规则来源指南
# ─────────────────────────────────────────────────────────────────────────────

# ═══════════════════════════════════════════════════════════════════════════════
# DM_DRUG_CONDITIONS:降糖药「条件优先」规则
# 特定患者表型下,某类降糖药的推荐等级提升(影响 Path-D 评分)
# 来源:guideline_rule_extractor.py 提取,临床专家审核后硬编码
# ═══════════════════════════════════════════════════════════════════════════════

DM_DRUG_CONDITIONS: List[Dict] = [
    {
        "id":          'FINAL-0041',
        "description": 'Metformin may be used in patients with stable CHF if renal function is normal. It should be avoided in unstable or hospitalized patients with CHF.',
        "condition":   lambda cm, pd, yr: (
            yr >= 2010
            and _get_dm_comp(cm, "heart_failure")
            and (lambda egfr: egfr is not None and egfr >= 30)(_get_egfr(cm))
        ),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2010 - Heart Failure',
        "domain":      'glycemic_first_line',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0042',
        "description": 'Prefer metformin as the initial pharmacologic agent for type 2 diabetes if no contraindications and tolerated',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2013 and (lambda egfr: egfr is None or egfr >= 30)(_get_egfr(cm)) and not (cm.get("kidney") or {}).get("aki_suspected", False)),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2013, Pharmacological therapy for hyperglycemia in type 2 diabetes',
        "domain":      'glycemic_first_line',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0043',
        "description": 'Metformin may be used in patients with stable CHF if renal function is normal. It should be avoided in unstable or hospitalized patients with CHF.',
        "condition":   lambda cm, pd, yr: (
            yr >= 2010
            and _get_dm_comp(cm, "heart_failure")
            and (lambda egfr: egfr is not None and 60 <= egfr)(_get_egfr(cm))
        ),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2011, Treatment',
        "domain":      'glycemic_first_line',
        "polarity":    'RECOMMEND',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'audit:avoid_uncheckable',
    },
    {
        "id":          'FINAL-0044',
        "description": 'Initiate insulin therapy for critically ill inpatients with diabetes when persistent hyperglycemia >=180 mg/dL, target glucose 140-180 mg/dL',
        "condition":   lambda cm, pd, yr: (yr >= 2011 and _get_dm_type(cm) in ("T1DM", "T2DM") and _is_icu_approx(pd) and (lambda g: g is not None and g > 180)((cm.get("glucose") or {}).get("mean_glucose"))),
        "any_of_cats": ['Insulins_Basal', 'Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2011 - Inpatient Glycemic Management',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0045',
        "description": 'Initiate metformin plus lifestyle modification for newly diagnosed T2DM patients',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2008),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Medical Care 2008 - Therapy for type 2 diabetes section',
        "domain":      'glycemic_first_line',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0046',
        "description": 'T1DM patients must be treated with multiple daily insulin injections (basal+prandial) or continuous subcutaneous insulin infusion',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T1DM"
            and yr >= 2008
        ),
        "any_of_cats": ['Insulins_Basal', 'Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Medical Care 2008 - Recommended therapy for type 1 diabetes section',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0047',
        "description": 'Initiate insulin at diagnosis for T2DM patients presenting with weight loss or severe hyperglycemic symptoms',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2008 and ((cm.get("hba1c") is not None and cm.get("hba1c") > 10) or (cm.get("random_glucose_mg_dl") is not None and cm.get("random_glucose_mg_dl") > 300))),
        "any_of_cats": ['Insulins_Basal', 'Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Medical Care 2008 - Therapy for type 2 diabetes section',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0048',
        "description": 'Insulin is preferred for glycemic control in hospitalized patients with diabetes',
        "condition":   lambda cm, pd, yr: (yr >= 2008 and _get_dm_type(cm) in ("T1DM", "T2DM")),
        "any_of_cats": ['Insulins_Basal', 'Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2008, Chapter: Hospital Care',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0049',
        "description": 'Most people with type 1 diabetes should be treated with MDI injections (basal and prandial) or CSII',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T1DM"
            and yr >= 2013
        ),
        "any_of_cats": ['Insulins_Basal', 'Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2013, Insulin therapy for type 1 diabetes',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0050',
        "description": 'For newly diagnosed T2DM patients, initiate metformin in combination with lifestyle changes',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2010),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2010, Approach to treatment',
        "domain":      'glycemic_first_line',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0051',
        "description": 'Patients with cystic fibrosis-related diabetes (CFRD) should be treated with insulin to attain individualized glycemic goals',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2013
        ),
        "any_of_cats": ['Insulins_Basal', 'Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2013, Section D. Cystic fibrosis–related diabetes Recommendations',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0052',
        "description": 'Consider adding SGLT2i or TZD for T2DM patients with suboptimal glycemic control requiring large insulin doses, to improve control and reduce insulin requirement.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2013 and ((cm.get("hba1c") is not None and cm.get("hba1c") > 8) or ((cm.get("glucose") or {}).get("max_glucose") is not None and (cm.get("glucose") or {}).get("max_glucose") > 250))),
        "any_of_cats": ['SGLT2i', 'Thiazolidinediones'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2022/2023, Combination Injectable Therapy',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0053',
        "description": 'For T1DM with hypoglycemia history, prefer insulin analogs over human insulins',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T1DM" and yr >= 2011 and _has_hypoglycemia_hx(pd)),
        "any_of_cats": ['Insulins_Basal', 'Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2011, Therapy for type 1 diabetes',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0054',
        "description": 'Pramlintide (amylin analog) may be considered as adjunct therapy in T1DM, with prandial insulin dose reduction to mitigate hypoglycemia risk',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T1DM"
            and yr >= 2005
        ),
        "any_of_cats": ['Amylin_Analogues'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2017, Section 8: Pharmacologic Approaches to Glycemic Treatment',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0055',
        "description": 'When transitioning from intravenous to subcutaneous insulin, initiate a basal-bolus insulin regimen at 80% of total daily intravenous insulin requirement over preceding 24 hours, divided into basal and bolus components.',
        "condition":   lambda cm, pd, yr: (yr >= 2009 and _is_on_iv_insulin(pd)),
        "any_of_cats": ['Insulins_Basal', 'Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2009, Inpatient Section 3. Transition from intravenous to subcutaneous insulin therapy',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0056',
        "description": 'Initiate basal-bolus insulin regimen in patients on systemic steroids at discharge',
        "condition":   lambda cm, pd, yr: yr >= 2012,
        "any_of_cats": ['Insulins_Basal', 'Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Consensus Reports & Supplements 2012, 5.4.1–5.4.4',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'judge:uncheckable:?',
    },
    {
        "id":          'FINAL-0057',
        "description": 'For patients with history of severe hypoglycemia, impaired awareness of hypoglycemia, or renal/hepatic dysfunction, prescribe long-acting insulin analogs (e.g., glargine, detemir, degludec) over NPH insulin at discharge to reduce risk of severe hypoglycemia.',
        "condition":   lambda cm, pd, yr: (yr >= 2005 and _get_dm_type(cm) in ("T1DM", "T2DM") and (_has_hypoglycemia_hx(pd) or _has_ckd_g3plus(cm) or (cm.get("liver") or {}).get("liver_dysfunction"))),
        "any_of_cats": ['Insulins_Basal'],
        "exclude_drugs": ['nph insulin'],
        "grade":       0.75,
        "source":      'ADA Consensus Reports & Supplements - Remarks & Benefits and Harms',
        "domain":      'glycemic_insulin',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0058',
        "description": 'Insulin therapy is recommended as the preferred method for glycemic control in hospitalized patients with hyperglycemia.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2012 and ((cm.get("random_glucose_mg_dl") is not None and cm.get("random_glucose_mg_dl") > 180) or ((cm.get("glucose") or {}).get("mean_glucose") is not None and (cm.get("glucose") or {}).get("mean_glucose") > 180))),
        "any_of_cats": ['Insulins_Basal', 'Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Consensus Reports & Supplements 2012, Section 4.2.1',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0059',
        "description": 'T1DM undergoing surgical procedures must receive CII or subcutaneous basal-bolus insulin to prevent DKA',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T1DM" and yr >= 2012),
        "any_of_cats": ['Insulins_Basal', 'Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Consensus Reports & Supplements 2012, 5.3.1',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0060',
        "description": 'Aggressive glycemic management with insulin for severe acute illness, perioperatively, or following MI',
        "condition":   lambda cm, pd, yr: (
            yr >= 2006
            and _get_dm_comp(cm, "cardiovascular")
        ),
        "any_of_cats": ['Insulins_Rapid', 'Insulins_Basal'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2006 - Treatment',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0061',
        "description": 'Initiate or resume oral agents and non-insulin injectables in clinically stable patients anticipating discharge who can consume meals at regular intervals.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2013
        ),
        "any_of_cats": ['Biguanides', 'SGLT2i', 'GLP-1RA', 'DPP-4i'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2013 | 2. Antihyperglycemic agents in hospitalized patients',
        "domain":      'glycemic_first_line',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0062',
        "description": 'Use intravenous insulin infusion for DKA, critical illness, NPO, or high-dose glucocorticoid therapy',
        "condition":   lambda cm, pd, yr: (yr >= 2006 and _is_icu_approx(pd)),
        "any_of_cats": ['Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2006 - 2. Treatment options',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0063',
        "description": 'Critically ill inpatients usually require intravenous insulin to maintain blood glucose 110-180 mg/dL.',
        "condition":   lambda cm, pd, yr: (yr >= 2006 and _is_icu_approx(pd)),
        "any_of_cats": ['Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2006 - Inpatient Management',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0064',
        "description": 'Initiate insulin therapy for treatment of persistent hyperglycemia starting at a threshold of no greater than 180 mg/dL in critically ill patients.',
        "condition":   lambda cm, pd, yr: (yr >= 2013 and _is_icu_approx(pd)),
        "any_of_cats": ['Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2013 - Diabetes care in the hospital',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0065',
        "description": 'Transition from IV to subcutaneous insulin should include short- or rapid-acting bolus insulin',
        "condition":   lambda cm, pd, yr: (yr >= 2008 and _is_on_iv_insulin(pd)),
        "any_of_cats": ['Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2008 - Treatment options in hospitalized patients',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0066',
        "description": 'T1DM patients require continuous exogenous basal insulin even when fasting to suppress gluconeogenesis and ketone production',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T1DM"
            and yr >= 2012
        ),
        "any_of_cats": ['Insulins_Basal'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Consensus Reports & Supplements 2012 - 4.3.1–4.3.4 Remarks',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0067',
        "description": 'Prefer rapid-acting insulin analogs over human regular insulin to reduce hypoglycemia risk in patients at high risk for hypoglycemia',
        "condition":   lambda cm, pd, yr: (yr >= 2012 and _get_dm_type(cm) in ("T1DM", "T2DM") and _has_hypoglycemia_hx(pd)),
        "any_of_cats": ['Insulins_Rapid'],
        "exclude_drugs": ['Regular insulin'],
        "grade":       0.75,
        "source":      'ADA Consensus Reports & Supplements / Endocrine Society Inpatient Guideline 2022',
        "domain":      'glycemic_insulin',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0148',
        "description": 'Metformin is the preferred initial pharmacologic agent for T2DM',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2000),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2019, Section 9, Recommendation 9.5',
        "domain":      'glycemic_first_line',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0165',
        "description": 'For T2DM patients without contraindications, recommend metformin as first-line therapy (eGFR≥30).',
        "condition":   lambda cm, pd, yr: (yr >= 2000 and _get_dm_type(cm) == "T2DM" and (lambda egfr: egfr is not None and 30 <= egfr)(_get_egfr(cm))),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2024, Table 9.2',
        "domain":      'glycemic_first_line',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0166',
        "description": 'For T2DM with CKD, metformin can be started if eGFR ≥45 mL/min/1.73m²; reduce dose if eGFR <45; discontinue if eGFR <30.',
        "condition":   lambda cm, pd, yr: (yr >= 2000 and _get_dm_type(cm) == "T2DM" and _has_ckd_g3plus(cm) and (lambda egfr: egfr is not None and egfr >= 45)(_get_egfr(cm))),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2025, Section 9: Glucose-Lowering Therapy for People With Chronic Kidney Disease',
        "domain":      'glycemic_first_line',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0167',
        "description": 'Metformin as first-line therapy for T2DM',
        "condition":   lambda cm, pd, yr: (yr >= 2000 and _get_dm_type(cm) == "T2DM"),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2022, Section 9.4a',
        "domain":      'glycemic_first_line',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0168',
        "description": 'Early insulin initiation for T2DM with catabolism, hyperglycemia symptoms, A1C>10% or BG≥300mg/dL',
        "condition":   lambda cm, pd, yr: (yr >= 2001 and _get_dm_type(cm) == "T2DM" and ((lambda h: h is not None and h > 10)(cm.get("hba1c")) or (lambda g: g is not None and g >= 300)(cm.get("random_glucose_mg_dl")))),
        "any_of_cats": ['Insulins_Basal', 'Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2022, Section 9.7',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0169',
        "description": 'In adults with T2DM, glucose-lowering agents may be continued upon insulin initiation unless contraindicated or not tolerated, for ongoing metabolic benefits.',
        "condition":   lambda cm, pd, yr: (yr >= 2000 and _get_dm_type(cm) == "T2DM"),
        "any_of_cats": ['Biguanides', 'SGLT2i', 'GLP-1RA', 'DPP-4i', 'Sulfonylureas', 'Meglitinides', 'Thiazolidinediones', 'Alpha_Glucosidase_Inhibitors', 'Glucokinase_Activators', 'GIP_GLP1_RA', 'GCG_GLP1_RA', 'GLP1_Multimer_RA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2024, Section 9, Recommendation 9.25',
        "domain":      'glycemic_first_line',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'audit:AVOID+empty-exclude flags every oral as commission for all T2DM(yr>=2000); desc permissive("may be continued"); duplicate of 0188',
    },
    {
        "id":          'FINAL-0170',
        "description": 'Metformin is the preferred initial agent for T2DM and should be continued at discharge if tolerated and not contraindicated.',
        "condition":   lambda cm, pd, yr: (yr >= 2001 and _get_dm_type(cm) == "T2DM" and (lambda egfr: egfr is None or egfr >= 30)(_get_egfr(cm))),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2021, Section 9, Recommendation 9.4 and 9.5',
        "domain":      'glycemic_first_line',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0171',
        "description": 'In T2DM patients on basal insulin >0.5 IU/kg/day with hypoglycemia or high variability, evaluate for overbasalization and adjust regimen.',
        "condition":   lambda cm, pd, yr: _get_dm_type(cm) == "T2DM",
        "any_of_cats": ['Insulins_Basal'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2021, Section 9, Recommendation 9.13',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'judge:uncheckable:?',
    },
    {
        "id":          'FINAL-0172',
        "description": 'Prescribe metformin for T2DM at discharge unless contraindicated (e.g., eGFR<30, liver dysfunction)',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2000 and (lambda egfr: egfr is not None and egfr >= 30)(_get_egfr(cm)) and (cm.get("liver") or {}).get("liver_dysfunction") is not True),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2024 - Section 9',
        "domain":      'glycemic_first_line',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0173',
        "description": 'Metformin contraindicated in T2DM patients with eGFR <30 mL/min/1.73 m²',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and (lambda egfr: egfr is not None and egfr < 30)(_get_egfr(cm))
        ),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2026, Table 9.2',
        "domain":      'glycemic_first_line',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0174',
        "description": 'For patients with severe hyperglycemia (A1C>10% or blood glucose ≥300mg/dL), symptoms of hyperglycemia, catabolic features, or possible T1DM, insulin should be considered as the first injectable therapy.',
        "condition":   lambda cm, pd, yr: (yr >= 2001 and _get_dm_type(cm) in ("T1DM", "T2DM") and ((lambda h: h is not None and h > 10)(cm.get("hba1c")) or (lambda g: g is not None and g >= 300)((cm.get("glucose") or {}).get("max_glucose")))),
        "any_of_cats": ['Insulins_Basal', 'Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2021, Injectable Therapies (Figure 9.2); 2023, Simplified overview',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0175',
        "description": 'For T1DM, discharge regimen should use basal-bolus insulin (MDI or pump) rather than twice-daily premixed insulin.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T1DM" and yr >= 2001),
        "any_of_cats": ['Insulins_Basal', 'Insulins_Rapid'],
        "exclude_drugs": ['premixed_insulin'],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2023, Figure 9.1',
        "domain":      'glycemic_insulin',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'P2-A: exclude 指向策展中有意移除的药(噻嗪/非DHP-CCB/gemfibrozil/保钾/跨类别),commission 永不触发,孤儿规则',
    },
    {
        "id":          'FINAL-0176',
        "description": 'Initiate insulin therapy when A1C >10% or blood glucose ≥300 mg/dL or with hyperglycemia symptoms/catabolism',
        "condition":   lambda cm, pd, yr: (yr >= 2000 and _get_dm_type(cm) == "T2DM" and ((lambda h: h is not None and h > 10)(cm.get("hba1c")) or (lambda g: g is not None and g >= 300)(cm.get("random_glucose_mg_dl")))),
        "any_of_cats": ['Insulins_Basal', 'Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2024, Section 9: Pharmacologic Approaches to Glycemic Treatment',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0177',
        "description": 'For T2DM patients with history of hypoglycemia, prefer long-acting basal analogs (U-300 glargine or degludec) for basal insulin at discharge to reduce hypoglycemia risk',
        "condition":   lambda cm, pd, yr: (yr >= 2000 and _get_dm_type(cm) == "T2DM" and _has_hypoglycemia_hx(pd)),
        "any_of_cats": ['Insulins_Basal'],
        "exclude_drugs": ['NPH insulin'],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2022, Section 9 (Basal Insulin)',
        "domain":      'glycemic_insulin',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0178',
        "description": 'Insulin should be the first injectable therapy when A1C>10% (or blood glucose >300mg/dL) or type 1 diabetes is suspected.',
        "condition":   lambda cm, pd, yr: (yr >= 2001 and _get_dm_type(cm) in ("T1DM", "T2DM") and ((lambda h: h is not None and h > 10)(cm.get("hba1c")) or (lambda g: g is not None and g > 300)((cm.get("glucose") or {}).get("max_glucose")))),
        "any_of_cats": ['Insulins_Basal', 'Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2023, Combination Therapy',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0179',
        "description": 'Type 1 diabetes discharge regimen should include multiple daily insulin injections or CSII',
        "condition":   lambda cm, pd, yr: yr >= 2000 and _get_dm_type(cm) == "T1DM",
        "any_of_cats": ['Insulins_Basal', 'Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2011, Therapy for type 1 diabetes',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0180',
        "description": 'If noninsulin monotherapy at maximal tolerated dose does not achieve or maintain the A1C target over 3-6 months, add a second oral agent, a GLP-1 receptor agonist, or insulin.',
        "condition":   lambda cm, pd, yr: (yr >= 2000 and _get_dm_type(cm) == "T2DM" and (lambda h: h is not None and h > 7.5)(cm.get("hba1c"))),
        "any_of_cats": ['Biguanides', 'Sulfonylureas', 'Meglitinides', 'Thiazolidinediones', 'DPP-4i', 'GLP-1RA', 'Insulins_Basal', 'Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2012, Recommendations Level E',
        "domain":      'glycemic_first_line',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0181',
        "description": 'Insulin (basal and rapid) is mandatory for all T1DM patients upon discharge as mainstay therapy',
        "condition":   lambda cm, pd, yr: (yr >= 1993 and _get_dm_type(cm) == "T1DM"),
        "any_of_cats": ['Insulins_Basal', 'Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2017, Section 8: Pharmacologic Approaches to Glycemic Treatment',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0182',
        "description": 'Most individuals with type 1 diabetes should use rapid-acting insulin analogs to reduce hypoglycemia risk',
        "condition":   lambda cm, pd, yr: _get_dm_type(cm) == "T1DM",
        "any_of_cats": ['Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2019 Section 9, Recommendation 9.2',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0183',
        "description": 'For T2DM patients requiring additional postprandial glucose control (already on basal insulin), add prandial insulin. Starting dose: 4 units or 10% of basal insulin dose at largest meal.',
        "condition":   lambda cm, pd, yr: (yr >= 2000 and _get_dm_type(cm) == "T2DM"),
        "any_of_cats": ['Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Medical Care in Diabetes 2020, Section 9: Pharmacologic Approaches to Glycemic Treatment - Prandial Insulin',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0184',
        "description": 'Most individuals with type 1 diabetes should use rapid-acting insulin analogs to reduce hypoglycemia risk.',
        "condition":   lambda cm, pd, yr: _get_dm_type(cm) == "T1DM",
        "any_of_cats": ['Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2021, Section 9, Recommendation 9.2',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0185',
        "description": 'Patients with history of hypoglycemia should prioritize glucose-lowering agents without risk of hypoglycemia',
        "condition":   lambda cm, pd, yr: (yr >= 2001 and _get_dm_type(cm) in ("T1DM", "T2DM") and (_has_hypoglycemia_hx(pd) or cm.get("hypo_history") or (lambda x: x is not None and x > 0)((cm.get("glucose") or {}).get("severe_hypoglycemia_events")))),
        "any_of_cats": ['Biguanides', 'SGLT2i', 'GLP-1RA', 'DPP-4i', 'Thiazolidinediones', 'Alpha_Glucosidase_Inhibitors'],
        "exclude_drugs": ['Glipizide', 'Glyburide', 'Glimepiride', 'Gliclazide', 'Chlorpropamide', 'Tolbutamide', 'Tolazamide', 'Acetohexamide', 'Repaglinide', 'Nateglinide', 'Mitiglinide'],
        "grade":       0.75,
        "source":      'ADA Consensus Reports & Supplements 2022 - Choice of Glucose-Lowering Medication',
        "domain":      'glycemic_first_line',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0186',
        "description": 'T1DM patients must be discharged on basal-bolus insulin regimen',
        "condition":   lambda cm, pd, yr: (yr >= 2001 and _get_dm_type(cm) == "T1DM"),
        "any_of_cats": ['Insulins_Basal', 'Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Consensus Reports & Supplements 2012, 4.2.1–4.2.3',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0187',
        "description": 'If noninsulin monotherapy at maximum tolerated dose does not achieve or maintain the A1C target over 3 months, add a second oral agent, GLP-1RA, or insulin.',
        "condition":   lambda cm, pd, yr: (yr >= 2000 and _get_dm_type(cm) == "T2DM" and (cm.get("hba1c") is not None and cm.get("hba1c") > 7.0)),
        "any_of_cats": ['SGLT2i', 'GLP-1RA', 'DPP-4i', 'Sulfonylureas', 'Thiazolidinediones', 'Insulins_Basal'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2014 - Pharmacological Therapy for Hyperglycemia in Type 2 Diabetes',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0188',
        "description": 'When initiating insulin, continue glucose-lowering agents for ongoing glycemic and metabolic benefits',
        "condition":   lambda cm, pd, yr: (yr >= 2000 and _get_dm_type(cm) == "T2DM" and _requires_insulin(cm, pd)),
        "any_of_cats": ['Biguanides', 'SGLT2i', 'GLP-1RA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2026 - Use of glucose-lowering medications in T2DM',
        "domain":      'glycemic_first_line',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0189',
        "description": 'Add second agent to metformin if HbA1c target not met after 3 months',
        "condition":   lambda cm, pd, yr: (yr >= 2000 and _get_dm_type(cm) == "T2DM" and (cm.get("hba1c") is not None and cm.get("hba1c") > 7.0)),
        "any_of_cats": ['Sulfonylureas', 'Thiazolidinediones', 'DPP-4i', 'GLP-1RA', 'Insulins_Basal'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2014 - Antihyperglycemic Therapy',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0190',
        "description": 'Treat most adults with type 1 diabetes with basal insulin',
        "condition":   lambda cm, pd, yr: (yr >= 2000 and _get_dm_type(cm) == "T1DM"),
        "any_of_cats": ['Insulins_Basal'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2024 Section 9 Rec 9.1',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0191',
        "description": 'Most people with type 1 diabetes should be treated with basal insulin (as part of MDI or CSII regimen)',
        "condition":   lambda cm, pd, yr: (yr >= 2001 and _get_dm_type(cm) == "T1DM"),
        "any_of_cats": ['Insulins_Basal'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2021 - 9. Pharmacologic Approaches to Glycemic Treatment',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0192',
        "description": 'If cost is a major issue and no ASCVD/CKD, prefer SU or TZDs as second line.',
        "condition":   lambda cm, pd, yr: (yr >= 2000 and _get_dm_type(cm) == "T2DM" and not _get_dm_comp(cm, "cardiovascular") and not _has_ckd_g3plus(cm)),
        "any_of_cats": ['Sulfonylureas', 'Thiazolidinediones'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Consensus Reports & Supplements 2018 - Figure 6',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0193',
        "description": 'For T1D with advanced CKD, insulin remains the only approved therapy',
        "condition":   lambda cm, pd, yr: (_has_ckd_g3plus(cm)
            and yr >= 2000 and _get_dm_type(cm) == "T1DM" and (lambda egfr: egfr is not None and egfr <= 29)(_get_egfr(cm))),
        "any_of_cats": ['Insulins_Basal'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Consensus Reports & Supplements 2022 | Glycemic Management in Advanced CKD',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0194',
        "description": 'Insulin treatment could be added if additional glycemic control is indicated in individuals with HF',
        "condition":   lambda cm, pd, yr: (yr >= 2001 and _get_dm_type(cm) == "T2DM" and _get_dm_comp(cm, "heart_failure")),
        "any_of_cats": ['Insulins_Basal'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Consensus Reports & Supplements 2022 - Key Points',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0195',
        "description": 'Prefer long-acting basal analog insulins over NPH to reduce severe hypoglycemia risk.',
        "condition":   lambda cm, pd, yr: (yr >= 2001 and _get_dm_type(cm) in ("T1DM", "T2DM") and _requires_insulin(cm, pd)),
        "any_of_cats": ['Insulins_Basal'],
        "exclude_drugs": ['nph insulin'],
        "grade":       0.75,
        "source":      'ADA Consensus Reports & Supplements | Justification for the Recommendation',
        "domain":      'glycemic_insulin',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0196',
        "description": 'When discontinuing IV insulin, subcutaneous basal insulin should be given 2 h before IV infusion is discontinued to minimize rebound hyperglycemia',
        "condition":   lambda cm, pd, yr: (yr >= 2001 and _get_dm_type(cm) in ("T1DM", "T2DM") and _is_on_iv_insulin(pd)),
        "any_of_cats": ['Insulins_Basal'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2024 - Subcutaneous Insulin',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0197',
        "description": 'Use rapid-acting insulin analogs rather than regular insulin for adult and pediatric individuals on basal-bolus insulin therapy who are at high risk for hypoglycemia.',
        "condition":   lambda cm, pd, yr: (yr >= 2001 and _get_dm_type(cm) in ("T1DM", "T2DM") and _requires_insulin(cm, pd) and (_has_hypoglycemia_hx(pd) or cm.get("hypo_history") or (lambda x: x is not None and x > 0)((cm.get("glucose") or {}).get("severe_hypoglycemia_events")))),
        "any_of_cats": ['Insulins_Rapid'],
        "exclude_drugs": ['regular insulin', 'Regular insulin'],
        "grade":       0.75,
        "source":      'ADA Consensus Reports & Supplements - Justification for the Recommendation',
        "domain":      'glycemic_insulin',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0264',
        "description": 'For T2DM patients with ASCVD, recommend SGLT2i with proven cardiovascular benefit to reduce MACE.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2016
            and _get_dm_comp(cm, "cardiovascular")
        ),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2024, Table 9.2 (Drug-Specific and Class Effects)',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0266',
        "description": 'For T2DM patients without contraindications, metformin is recommended as first-line glucose-lowering therapy.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2019
            and (lambda egfr: egfr is not None and 30 <= egfr)(_get_egfr(cm))
        ),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2019, Initial Therapy (Figure 9.1)',
        "domain":      'glycemic_first_line',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0267',
        "description": 'In T2DM with established ASCVD, regimen should include agents with cardiovascular benefit (SGLT2i or GLP-1RA)',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2017
            and _get_dm_comp(cm, "cardiovascular")
        ),
        "any_of_cats": ['SGLT2i', 'GLP-1RA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2023 - Pharmacologic Approaches to Glycemic Treatment',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0268',
        "description": 'For T2DM with established ASCVD, recommend SGLT2i or GLP-1RA with cardiovascular benefit as part of glycemic plan, independent of A1C and metformin.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2016
            and _get_dm_comp(cm, "cardiovascular")
        ),
        "any_of_cats": ['SGLT2i', 'GLP-1RA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2024, Combination Therapy',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0269',
        "description": 'For T2DM with A1C 1.5-2% above goal, consider initial combination therapy (e.g., metformin + DPP-4i).',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2019 and (lambda h: h is not None and h >= 8.5)(cm.get("hba1c"))),
        "any_of_cats": ['Biguanides', 'DPP-4i'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2024, Combination Therapy',
        "domain":      'glycemic_first_line',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0270',
        "description": 'If insulin is used, combination with GLP-1RA is recommended',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2017
        ),
        "any_of_cats": ['GLP-1RA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2022, Section 9.11',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0271',
        "description": 'For adults with T2DM and advanced CKD (eGFR <30), a GLP-1 RA is preferred for glycemic management due to lower hypoglycemia risk and CV event reduction.',
        "condition":   lambda cm, pd, yr: (_has_ckd_g3plus(cm)
            and _get_dm_type(cm) == "T2DM" and yr >= 2017 and (lambda egfr: egfr is not None and egfr < 30)(_get_egfr(cm))),
        "any_of_cats": ['GLP-1RA'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2024, Section 9, Recommendation 9.21',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0272',
        "description": 'For adults with T2DM, a GLP-1 RA (including dual GIP/GLP-1 RA) is preferred over insulin.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2017
        ),
        "any_of_cats": ['GLP-1RA', 'GIP_GLP1_RA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2024, Section 9, Recommendation 9.23',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0273',
        "description": 'Consider initiating insulin therapy (with or without additional agents) in patients with newly diagnosed type 2 diabetes who are symptomatic and/or have A1C ≥10% and/or blood glucose levels ≥300 mg/dL.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2016 and ((cm.get("hba1c") is not None and cm.get("hba1c") >= 10) or (cm.get("random_glucose_mg_dl") is not None and cm.get("random_glucose_mg_dl") >= 300) or ((cm.get("glucose") or {}).get("max_glucose") is not None and (cm.get("glucose") or {}).get("max_glucose") >= 300))),
        "any_of_cats": ['Insulins_Basal', 'Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2018, Section 8',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0274',
        "description": 'In most patients who need injectable medication, GLP-1 RAs are preferred to insulin',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2019),
        "any_of_cats": ['GLP-1RA'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2019, Section 9, Recommendation 9.14',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0275',
        "description": 'In T2DM patients with established ASCVD, heart failure, or CKD, SGLT2i or GLP-1RA with demonstrated CV benefit is recommended as part of the glucose-lowering regimen independent of A1C.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2017 and (_get_dm_comp(cm, "cardiovascular") or _get_dm_comp(cm, "heart_failure") or _has_ckd_g3plus(cm))),
        "any_of_cats": ['SGLT2i', 'GLP-1RA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2021, Section 9, Recommendation 9.9',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0276',
        "description": 'If insulin is used, combination therapy with a GLP-1 receptor agonist is recommended for greater efficacy, durability of treatment effect, and weight and hypoglycemia benefit.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2017),
        "any_of_cats": ['GLP-1RA', 'Insulins_Basal', 'Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2023, Recommendation 9.11',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0277',
        "description": 'DPP-4 inhibitors should not be used concurrently with GLP-1 RA or dual GIP/GLP-1 RA due to lack of additional glucose lowering',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2016
        ),
        "any_of_cats": ['DPP-4i', 'GLP-1RA', 'GIP_GLP1_RA'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2026, 9.18',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'audit:avoid_uncheckable',
    },
    {
        "id":          'FINAL-0279',
        "description": 'For T2DM patients with ASCVD, HF, or CKD, recommend GLP-1RA with demonstrated cardiovascular benefit as part of the glucose-lowering regimen independent of A1C',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2017 and (_get_dm_comp(cm, "cardiovascular") or _get_dm_comp(cm, "heart_failure") or _has_ckd_g3plus(cm))),
        "any_of_cats": ['GLP-1RA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2020, Section 9, Pharmacologic Approaches to Glycemic Treatment',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0280',
        "description": 'For T2DM patients with A1C ≥1.5% above target, consider initiating dual combination therapy or a more potent glucose-lowering agent.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2015 and (lambda h: h is not None and h >= 8.5)(cm.get("hba1c"))),
        "any_of_cats": ['Biguanides', 'SGLT2i', 'GLP-1RA', 'DPP-4i', 'Sulfonylureas', 'Thiazolidinediones', 'Insulins_Basal', 'Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2023, Simplified overview of beta-cell replacement',
        "domain":      'glycemic_first_line',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0281',
        "description": 'For T2DM patients with eGFR ≥30 mL/min/1.73m², metformin can be safely used.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2016
            and (lambda egfr: egfr is not None and 30 <= egfr)(_get_egfr(cm))
        ),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2023, Simplified overview',
        "domain":      'glycemic_first_line',
        "polarity":    'ALLOW',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0282',
        "description": 'For T2DM adults without severe hyperglycemia, prefer GLP-1-based therapy (including GLP-1 RA, dual GIP/GLP-1 RA) over insulin for initial or add-on therapy.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2017 and (lambda h: h is not None and h < 10)(cm.get("hba1c"))),
        "any_of_cats": ['GLP-1RA', 'GIP_GLP1_RA', 'GCG_GLP1_RA', 'GLP1_Multimer_RA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2026, Recommendation 9.21',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0283',
        "description": 'Do not prescribe GLP-1RA and DPP-4i in combination',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2018
        ),
        "any_of_cats": ['GLP-1RA', 'DPP-4i'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2018, Combination Injectable Therapy section, Figure 8.1 note',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'audit:avoid_uncheckable',
    },
    {
        "id":          'FINAL-0284',
        "description": 'If noninsulin monotherapy at max tolerated dose fails to achieve A1C target in 3 months, add second oral agent, GLP-1 RA, or basal insulin',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2015),
        "any_of_cats": ['Biguanides', 'Sulfonylureas', 'Thiazolidinediones', 'DPP-4i', 'SGLT2i', 'GLP-1RA', 'Insulins_Basal'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2015, Section: Pharmacological Glycemic Management',
        "domain":      'glycemic_first_line',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0285',
        "description": 'Metformin monotherapy should be started at diagnosis of type 2 diabetes unless contraindicated or intolerant',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2018 and (lambda egfr: egfr is None or egfr >= 30)(_get_egfr(cm))),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2018 - Initial Therapy',
        "domain":      'glycemic_first_line',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0286',
        "description": 'Metformin can be safely used in T2DM patients with eGFR ≥30 mL/min/1.73m²',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2018
            and (lambda egfr: egfr is not None and 30 <= egfr)(_get_egfr(cm))
        ),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2018 - Initial Therapy',
        "domain":      'glycemic_first_line',
        "polarity":    'ALLOW',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0287',
        "description": 'When blood glucose ≥300 mg/dL or A1C ≥10% or hyperglycemia symptoms, consider initiating insulin therapy in T2DM',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2018 and ((cm.get("glucose") or {}).get("max_glucose") is not None and (cm.get("glucose") or {}).get("max_glucose") >= 300 or (cm.get("hba1c") is not None and cm.get("hba1c") >= 10))),
        "any_of_cats": ['Insulins_Basal', 'Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2018 - Initial Therapy',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0288',
        "description": 'Meglitinides may replace sulfonylureas in T2DM patients with sulfa allergy, irregular meal schedules, or late postprandial hypoglycemia on sulfonylurea',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2017 and _has_hypoglycemia_hx(pd)),
        "any_of_cats": ['Meglitinides'],
        "exclude_drugs": ['Glipizide', 'Glyburide', 'Glimepiride', 'Gliclazide', 'Chlorpropamide', 'Tolbutamide', 'Tolazamide', 'Acetohexamide'],
        "grade":       0.5,
        "source":      'ADA Standards of Care 2017 - Combination Therapy',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0289',
        "description": 'For non-critically ill hospitalized diabetes patients, scheduled subcutaneous insulin with basal, nutritional, and correctional components (basal + rapid insulin) should be preferred at discharge over sliding scale alone',
        "condition":   lambda cm, pd, yr: (yr >= 2014 and _get_dm_type(cm) in ("T1DM", "T2DM") and not _is_icu_approx(pd)),
        "any_of_cats": ['Insulins_Basal', 'Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2014, Hospital Diabetes Care',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0290',
        "description": 'Consider initiating combination insulin injectable therapy (basal + prandial or premixed) in T2DM patients with HbA1c≥10% or blood glucose≥300 mg/dL with hyperglycemia symptoms.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2017 and ((lambda h: h is not None and h >= 10)(cm.get("hba1c")) or (lambda g: g is not None and g >= 300)((cm.get("glucose") or {}).get("max_glucose")))),
        "any_of_cats": ['Insulins_Basal', 'Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Medical Care in Diabetes 2017, Section 8: Pharmacologic Approaches to Glycemic Treatment',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0291',
        "description": 'For T2DM patients with heart failure or CKD, metformin can be prescribed at discharge (avoid during hospitalization) as it is as safe as other glucose-lowering treatments.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2014 and (_get_dm_comp(cm, "heart_failure") or _has_ckd_g3plus(cm)) and (lambda e: e is None or e > 30)(_get_egfr(cm))),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2014, Treatment of Diabetes in Hospitalized Patients',
        "domain":      'glycemic_first_line',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0292',
        "description": 'Consider adding pramlintide at discharge for adult type 1 diabetes patients with inadequate postprandial glycemic control on insulin',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T1DM" and yr >= 2016 and (_get_age(cm, pd) or 0) >= 18 and (lambda h: h is not None and h > 7)(cm.get("hba1c"))),
        "any_of_cats": ['Amylin_Analogues'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2016, Pramlintide section',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0293',
        "description": 'Most people with type 1 diabetes should be treated with multiple-dose insulin (MDI) injections or continuous subcutaneous insulin infusion (CSII), and use insulin analogs to reduce hypoglycemia risk.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T1DM"
            and yr >= 2015
        ),
        "any_of_cats": ['Insulins_Basal', 'Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2015 - Recommendations (Section C)',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0294',
        "description": 'Most individuals with type 1 diabetes should use insulin analogs (basal or rapid) to reduce hypoglycemia risk',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T1DM"
            and yr >= 2016
        ),
        "any_of_cats": ['Insulins_Basal', 'Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2016, Section: Recommendations',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0295',
        "description": 'Most individuals with type 1 diabetes should be treated with multiple daily injections of prandial insulin and basal insulin or continuous subcutaneous insulin infusion.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T1DM"
            and yr >= 2017
        ),
        "any_of_cats": ['Insulins_Basal', 'Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2017, Chapter on Insulin Therapy',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0296',
        "description": 'Most individuals with type 1 diabetes should use rapid-acting insulin analogs to reduce hypoglycemia risk.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T1DM" and yr >= 2017),
        "any_of_cats": ['Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2017, Chapter on Insulin Therapy',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0297',
        "description": 'Most people with type 1 diabetes should be treated with multiple daily injections of prandial insulin and basal insulin or continuous subcutaneous insulin infusion at discharge',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T1DM"
            and yr >= 2018
        ),
        "any_of_cats": ['Insulins_Basal', 'Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2018, Section 8 - Pharmacologic Approaches to Glycemic Treatment',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0298',
        "description": 'Most individuals with type 1 diabetes should use rapid-acting insulin analogs to reduce hypoglycemia risk at discharge',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T1DM"
            and yr >= 2018
        ),
        "any_of_cats": ['Insulins_Rapid'],
        "exclude_drugs": ['Regular insulin'],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2018, Section 8 - Pharmacologic Approaches to Glycemic Treatment',
        "domain":      'glycemic_insulin',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0299',
        "description": 'For T2DM patients with established ASCVD, prioritize prescribing antihyperglycemic agents proven to reduce major adverse cardiovascular events and/or cardiovascular mortality (e.g., SGLT2i or GLP-1RA) at discharge.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2018
            and _get_dm_comp(cm, "cardiovascular")
        ),
        "any_of_cats": ['SGLT2i', 'GLP-1RA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2018, Chapter 8: Pharmacologic Approaches to Glycemic Treatment, ASCVD algorithm',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0300',
        "description": 'For most patients, especially those at elevated risk of hypoglycemia, use insulin analogs',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2016
        ),
        "any_of_cats": ['Insulins_Basal', 'Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2016, Insulin Therapy section',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0301',
        "description": 'In T2DM without ASCVD or CKD, when there is compelling need to minimize weight gain or promote weight loss, prefer GLP-1 RA or SGLT2i in addition to metformin',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2018 and not _get_dm_comp(cm, "cardiovascular") and not _get_dm_comp(cm, "heart_failure") and not _has_ckd_g3plus(cm)),
        "any_of_cats": ['GLP-1RA', 'SGLT2i'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Consensus Report 2018, Choosing Glucose-Lowering Medication if Compelling Need to Minimize Weight Gain or Promote Weight Loss, Figure 4',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0302',
        "description": 'For T2DM with HF or CKD, prefer SGLT2i if eGFR adequate; otherwise use GLP-1 RA',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2018 and (_get_dm_comp(cm, "heart_failure") or _has_ckd_g3plus(cm))),
        "any_of_cats": ['SGLT2i', 'GLP-1RA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Consensus Reports 2018, Figure 3',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0303',
        "description": 'For T2DM patients with compelling need to minimize hypoglycemia, prioritize glucose-lowering medications with low hypoglycemia risk: DPP-4i, GLP-1 RA, SGLT2i, or TZD',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2018 and _has_hypoglycemia_hx(pd)),
        "any_of_cats": ['DPP-4i', 'GLP-1RA', 'SGLT2i', 'Thiazolidinediones'],
        "exclude_drugs": ['Glipizide', 'Glyburide', 'Glimepiride', 'Gliclazide', 'Chlorpropamide', 'Tolbutamide', 'Tolazamide', 'Acetohexamide', 'NPH insulin', 'Insulin human', 'Insulin glargine', 'Insulin detemir', 'Insulin degludec', 'Insulin icodec', 'Regular insulin', 'Insulin lispro', 'Insulin aspart', 'Insulin glulisine', 'Faster insulin aspart'],
        "grade":       0.75,
        "source":      'ADA Consensus Reports & Supplements 2018 | Figure 5: Choosing glucose-lowering medication if compelling need to minimize hypoglycemia',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0304',
        "description": 'For T2D patients with ASCVD or CKD (eGFR≥15), GLP-1 RA should be prioritized at discharge (reduces MACE and composite kidney outcomes).',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2017 and (_get_dm_comp(cm, "cardiovascular") or ((_has_ckd_g3plus(cm) or _has_albuminuria(cm)) and (lambda egfr: egfr is not None and egfr >= 15)(_get_egfr(cm))))),
        "any_of_cats": ['GLP-1RA'],
        "exclude_drugs": ['Exenatide'],
        "grade":       1.0,
        "source":      'ADA Consensus Report 2022, GLP-1 Receptor Agonists section',
        "domain":      'cardiorenal_protection',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0305',
        "description": 'For T2DM with CKD not achieving glycemic targets on metformin and/or SGLT2i, or unable to take them, prefer addition of a long-acting GLP-1 RA.',
        "condition":   lambda cm, pd, yr: (_has_ckd_g3plus(cm)
            and _get_dm_type(cm) == "T2DM" and yr >= 2017 and (lambda egfr: egfr is not None and egfr <= 60)(_get_egfr(cm)) and (lambda h: h is not None and h > 7.0)(cm.get("hba1c"))),
        "any_of_cats": ['GLP-1RA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Consensus Reports & Supplements 2022, Use of Additional Glucose-Lowering Agents',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0306',
        "description": 'Prefer SGLT2i or GLP-1RA in patients prioritizing weight loss or weight maintenance',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2018 and (lambda b: b is not None and b >= 27)(_get_bmi_val(pd))),
        "any_of_cats": ['SGLT2i', 'GLP-1RA'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Consensus Reports & Supplements 2018 Consensus recommendation',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0308',
        "description": 'Reduce sulfonylurea dose by 50% when starting basal insulin to avoid hypoglycemia',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2018
        ),
        "any_of_cats": ['Sulfonylureas'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Consensus Reports & Supplements 2018, Figure 8',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'P1-4: SU 减量规则误编成 AVOID,与 0326 重复,停用',
    },
    {
        "id":          'FINAL-0309',
        "description": 'Maintain metformin, SGLT2i, and GLP-1 RA when using insulin to avoid weight gain, limit insulin dose, and reduce hypoglycemia risk',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2017 and _requires_insulin(cm, pd)),
        "any_of_cats": ['Biguanides', 'SGLT2i', 'GLP-1RA'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Consensus Reports & Supplements 2022, If not already on GLP-1 RA...',
        "domain":      'glycemic_first_line',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0310',
        "description": 'For T2DM patients requiring injectable therapy, prefer GLP-1 receptor agonists over insulin',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2018),
        "any_of_cats": ['GLP-1RA'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Consensus Reports & Supplements 2018, Choosing Glucose-Lowering Medication in Those with Established ASCVD or CKD, Chunk 2',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0311',
        "description": 'For patients with extreme and symptomatic hyperglycemia, insulin is recommended',
        "condition":   lambda cm, pd, yr: (yr >= 2018 and _get_dm_type(cm) in ("T1DM", "T2DM") and (cm.get("glucose") or {}).get("max_glucose") is not None and (cm.get("glucose") or {}).get("max_glucose") > 300),
        "any_of_cats": ['Insulins_Basal', 'Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Consensus Reports & Supplements 2018, Choosing Glucose-Lowering Medication in Those with Established ASCVD or CKD, Chunk 2',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0312',
        "description": 'For hospitalized patients with diabetes requiring insulin, a basal-plus-bolus correction insulin regimen should be continued at discharge; avoid sliding scale insulin alone.',
        "condition":   lambda cm, pd, yr: (yr >= 2016 and _get_dm_type(cm) in ("T1DM", "T2DM") and _requires_insulin(cm, pd)),
        "any_of_cats": ['Insulins_Basal', 'Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Consensus Reports & Supplements 2016, Inpatient Diabetes Management',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0313',
        "description": 'For noncritically ill hospitalized patients with poor oral intake or NPO, prefer basal insulin or basal plus correction insulin regimen at discharge',
        "condition":   lambda cm, pd, yr: yr >= 2019,
        "any_of_cats": ['Insulins_Basal', 'Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Consensus Reports 2019, Recommendation 15.6, Level A',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'phase3: NPO/poor-oral-intake 无对应 cm 特征,无法门控 → 停用',
    },
    {
        "id":          'FINAL-0314',
        "description": 'Consider initiating dual therapy in patients with newly diagnosed type 2 diabetes who have A1C >= 9%.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2018 and (lambda h: h is not None and h >= 9)(cm.get("hba1c"))),
        "any_of_cats": ['Sulfonylureas', 'Thiazolidinediones', 'DPP-4i', 'SGLT2i', 'GLP-1RA', 'Insulins_Basal'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2018 Section 8 Pharmacologic Approaches',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0315',
        "description": 'Metformin may be safely used in patients with reduced eGFR down to 30 mL/min/1.73m^2.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2016
            and (lambda egfr: egfr is not None and 30 <= egfr)(_get_egfr(cm))
        ),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2019 Section 9',
        "domain":      'glycemic_first_line',
        "polarity":    'ALLOW',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0316',
        "description": 'Initiate or resume oral agents or injectable noninsulin therapies in stable patients anticipating discharge',
        "condition":   lambda cm, pd, yr: (yr >= 2014 and _get_dm_type(cm) in ("T1DM", "T2DM")),
        "any_of_cats": ['Biguanides', 'SGLT2i', 'GLP-1RA', 'DPP-4i'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2014 - 2. Antihyperglycemic Agents in Hospitalized Patients',
        "domain":      'glycemic_first_line',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0317',
        "description": 'T1DM patients require basal insulin regimen; sole use of sliding scale insulin is discouraged due to DKA risk',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T1DM"
            and yr >= 2014
        ),
        "any_of_cats": ['Insulins_Basal'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2014 - 2. Antihyperglycemic Agents in Hospitalized Patients',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0318',
        "description": 'If A1C target not achieved after ~3 months, consider combination of metformin and one of: SU, TZD, DPP-4i, SGLT2i, GLP-1 RA, or basal insulin.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2017 and (lambda h: h is not None and h > 7)(cm.get("hba1c"))),
        "any_of_cats": ['Sulfonylureas', 'Thiazolidinediones', 'DPP-4i', 'SGLT2i', 'GLP-1RA', 'Insulins_Basal'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2017 Section 8 Combination Therapy',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0320',
        "description": 'Critically ill patients require an intravenous insulin protocol',
        "condition":   lambda cm, pd, yr: (yr >= 2014 and _is_icu_approx(pd)),
        "any_of_cats": ['Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2014 - Diabetes Care in the Hospital',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0321',
        "description": 'Longer-acting basal analogs (U-300 glargine or degludec) may convey lower hypoglycemia risk compared with U-100 glargine in T1DM',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T1DM"
            and yr >= 2018
        ),
        "any_of_cats": ['Insulins_Basal'],
        "exclude_drugs": ['Insulin glargine'],
        "grade":       0.75,
        "source":      'ADA Standards of Medical Care in Diabetes 2018 Sec. 8 Pharmacologic approaches',
        "domain":      'glycemic_insulin',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'audit:U-300 vs U-100 glargine不可分(DrugNormalizer将toujeo/lantus均归一为"Insulin glargine",浓度标记被剥离)->exclude会误罚推荐的U-300;taxonomy无法忠实实现',
    },
    {
        "id":          'FINAL-0322',
        "description": 'Rapid-acting analogs are preferred for mealtime bolus insulin in T2DM due to their prompt onset of action after dosing.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2018
        ),
        "any_of_cats": ['Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2018 - Pharmacologic Approaches to Glycemic Treatment',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0323',
        "description": 'In T1DM with hypoglycemia history, prefer next-generation long-acting basal analogs (U-300 glargine or degludec) to confer lower hypoglycemia risk.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T1DM" and yr >= 2015 and _has_hypoglycemia_hx(pd)),
        "any_of_cats": ['Insulins_Basal'],
        "exclude_drugs": ['Insulin glargine'],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2022 - Pharmacologic Therapy for Type 1 Diabetes',
        "domain":      'glycemic_insulin',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'audit:同0321,U-300/U-100 glargine不可分->exclude会误罚推荐的U-300;taxonomy无法忠实实现',
    },
    {
        "id":          'FINAL-0324',
        "description": 'If compelling need to minimize hypoglycemia and no ASCVD/CKD, prefer DPP-4i, GLP-1 RA, SGLT2i, or TZDs as second line.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2018 and not _get_dm_comp(cm, "cardiovascular") and not _has_ckd_g3plus(cm) and _has_hypoglycemia_hx(pd)),
        "any_of_cats": ['DPP-4i', 'GLP-1RA', 'SGLT2i', 'Thiazolidinediones'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Consensus Reports & Supplements 2018 - Figure 5',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0325',
        "description": 'For T2DM inadequately controlled on basal insulin or GLP-1RA, fixed-ratio combination is recommended to reduce hypo and weight',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2018 and (cm.get("hba1c") is not None and cm.get("hba1c") > 7.0)),
        "any_of_cats": ['GLP-1RA', 'Insulins_Basal'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Consensus Reports & Supplements 2022 - Medications For Lowering Glucose',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0326',
        "description": 'Reduce sulfonylurea dose by 50% when starting basal insulin to avoid hypoglycemia.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2018
        ),
        "any_of_cats": ['Sulfonylureas'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Consensus Reports & Supplements 2018 - Figure 8',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'audit:dose-titration("reduce SU 50%")uncheckable; RECOMMEND gave concordance credit for prescribing SU',
    },
    {
        "id":          'FINAL-0327',
        "description": 'If additional glycemic control is needed for T2D at high risk for or with established HF, use of GLP-1RA, metformin, or both should be favored over sulfonylureas',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2017 and (_get_dm_comp(cm, "heart_failure") or _get_dm_comp(cm, "cardiovascular"))),
        "any_of_cats": ['GLP-1RA', 'Biguanides'],
        "exclude_drugs": ['Glipizide', 'Glyburide', 'Glimepiride', 'Gliclazide', 'Chlorpropamide', 'Tolbutamide', 'Tolazamide', 'Acetohexamide'],
        "grade":       0.75,
        "source":      'ADA Consensus Reports & Supplements 2022 - Key Points',
        "domain":      'cardiorenal_protection',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0328',
        "description": 'For T2DM intensification beyond dual therapy, combinations including insulin and GLP-1RA demonstrate superiority due to broad ranges of glycemic efficacy.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2018 and (cm.get("hba1c") is not None and cm.get("hba1c") > 7.0)),
        "any_of_cats": ['Insulins_Basal', 'GLP-1RA'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Consensus Reports & Supplements 2018 - Consensus recommendation',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0329',
        "description": 'Intensify treatment with GLP-1RA, SGLT2i, or prandial insulin for T2DM patients unable to maintain glycemic targets on basal insulin plus oral medications.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2018 and (cm.get("hba1c") is not None and cm.get("hba1c") > 7.0)),
        "any_of_cats": ['GLP-1RA', 'SGLT2i', 'Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Consensus Reports & Supplements 2018 - Pharmacologic approaches',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0330',
        "description": 'Transition from IV to subcutaneous insulin requires administration of basal insulin 2-4h prior to stopping IV insulin to prevent recurrence of ketoacidosis.',
        "condition":   lambda cm, pd, yr: (yr >= 2019 and _is_on_iv_insulin(pd)),
        "any_of_cats": ['Insulins_Basal'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Consensus Reports & Supplements 2019 - Diabetic Ketoacidosis and Hyperosmolar Hyperglycemic State',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0331',
        "description": 'If oral medications are held in the hospital, there should be a protocol for resuming them 1–2 days before discharge.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2016),
        "any_of_cats": ['Biguanides', 'SGLT2i', 'DPP-4i', 'Sulfonylureas', 'Thiazolidinediones', 'Alpha_Glucosidase_Inhibitors'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Consensus Reports & Supplements 2016 - ANTIHYPERGLYCEMIC AGENTS IN HOSPITALIZED PATIENTS',
        "domain":      'glycemic_first_line',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0332',
        "description": 'T1DM patients must be prescribed basal insulin; rapid insulin alone increases DKA risk',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T1DM"
            and yr >= 2016
        ),
        "any_of_cats": ['Insulins_Basal'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Consensus Reports & Supplements 2016 - Type 1 Diabetes',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0378',
        "description": 'For patients with T2DM and CKD (eGFR >20 mL/min/1.73m²), GLP-1 receptor agonists (including semaglutide) should be preferred as glucose-lowering agents.',
        "condition":   lambda cm, pd, yr: (_has_ckd_g3plus(cm)
            and _get_dm_type(cm) == "T2DM" and yr >= 2025 and (lambda egfr: egfr is not None and egfr > 20)(_get_egfr(cm))),
        "any_of_cats": ['GLP-1RA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2025, Section 9: Glucose-Lowering Therapy for People With Chronic Kidney Disease',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0379',
        "description": 'For patients with T2DM and CKD (eGFR >20 mL/min/1.73m²), SGLT2 inhibitors should be preferred as glucose-lowering agents.',
        "condition":   lambda cm, pd, yr: (_has_ckd_g3plus(cm)
            and _get_dm_type(cm) == "T2DM" and yr >= 2020 and (lambda egfr: egfr is not None and egfr > 20)(_get_egfr(cm))),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2025, Section 9: Glucose-Lowering Therapy for People With Chronic Kidney Disease',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0380',
        "description": 'For T2DM with established ASCVD, prioritize SGLT2i or GLP-1RA with CV benefit.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2022
            and _get_dm_comp(cm, "cardiovascular")
        ),
        "any_of_cats": ['SGLT2i', 'GLP-1RA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2022, Section 9, Cardiovascular Outcomes Trials',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0381',
        "description": 'For T2DM with heart failure, prioritize SGLT2i or GLP-1RA.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2025
            and _get_dm_comp(cm, "heart_failure")
        ),
        "any_of_cats": ['SGLT2i', 'GLP-1RA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2025, Figure 9.3, Use of Glucose-Lowering Medications',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0382',
        "description": 'For T2DM with CKD and eGFR≥20, prioritize SGLT2i.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2025 and (_has_ckd_g3plus(cm) or _has_albuminuria(cm)) and (lambda egfr: egfr is not None and 20 <= egfr)(_get_egfr(cm))),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2025, Figure 9.3, Use of Glucose-Lowering Medications',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0383',
        "description": 'If glycemic target not met, consider combination therapy with SGLT2i and GLP-1RA.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2025 and (lambda h: h is not None and h > 7.0)(cm.get("hba1c"))),
        "any_of_cats": ['SGLT2i', 'GLP-1RA'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2025, Figure 9.3',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0384',
        "description": 'In T2DM with heart failure, use SGLT2i for HF benefit',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2020
            and _get_dm_comp(cm, "heart_failure")
        ),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2023 - Pharmacologic Approaches to Glycemic Treatment',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0385',
        "description": 'In T2DM with CKD, use SGLT2i for nephroprotection',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2020 and (_has_ckd_g3plus(cm) or _has_albuminuria(cm))),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2023 - Pharmacologic Approaches to Glycemic Treatment',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0386',
        "description": 'For T2DM with heart failure or CKD, recommend SGLT2i or GLP-1RA with cardiovascular benefit as part of glycemic plan, independent of A1C and metformin.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2020 and (_get_dm_comp(cm, "heart_failure") or _has_ckd_g3plus(cm) or _has_albuminuria(cm))),
        "any_of_cats": ['SGLT2i', 'GLP-1RA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2024, Combination Therapy',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0387',
        "description": 'For adults with T2DM and CKD (eGFR 20-60 and/or albuminuria), SGLT2 inhibitor should be used to minimize CKD progression, CV events, and HF hospitalization.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2020 and ((lambda egfr: egfr is not None and 20 <= egfr <= 60)(_get_egfr(cm)) or _has_albuminuria(cm))),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2024, Section 9, Recommendation 9.20',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0388',
        "description": 'For T2DM patients with ASCVD, HF, or CKD, prescribe SGLT2i or GLP-1 RA with proven cardiovascular/kidney benefit at discharge',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2022 and (_get_dm_comp(cm, "cardiovascular") or _get_dm_comp(cm, "heart_failure") or _has_ckd_g3plus(cm))),
        "any_of_cats": ['SGLT2i', 'GLP-1RA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2024 - Section 9',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0389',
        "description": 'Initiate insulin in adults with T2DM if symptoms of hyperglycemia present or A1C>10% or blood glucose>300 mg/dL',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2025 and ((lambda h: h is not None and h > 10)(cm.get("hba1c")) or (lambda g: g is not None and g > 300)(cm.get("random_glucose_mg_dl")))),
        "any_of_cats": ['Insulins_Basal', 'Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2025, Recommendation 9.23',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0390',
        "description": 'In adults with T2DM without evidence of insulin deficiency, GLP-1 RA (including dual GIP/GLP-1 RA) is preferred to insulin',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2025),
        "any_of_cats": ['GLP-1RA', 'GIP_GLP1_RA', 'GCG_GLP1_RA', 'GLP1_Multimer_RA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2025, Recommendation 9.24',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0391',
        "description": 'In adults with T2DM initiating insulin, continue other glucose-lowering agents (unless contraindicated) for ongoing glycemic and metabolic benefits',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2025),
        "any_of_cats": ['Biguanides', 'SGLT2i', 'GLP-1RA', 'DPP-4i', 'GIP_GLP1_RA', 'GCG_GLP1_RA', 'GLP1_Multimer_RA', 'Sulfonylureas', 'Meglitinides', 'Thiazolidinediones', 'Alpha_Glucosidase_Inhibitors', 'Glucokinase_Activators', 'Amylin_Analogues', 'Bile_Acid_Sequestrants_DM', 'Dopamine_Agonists_DM'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2025, Recommendation 9.26',
        "domain":      'glycemic_first_line',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0392',
        "description": 'GLP-1 RA preferred for glycemic management in T2DM with MASH or high risk of liver fibrosis',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2021 and (cm.get("liver") or {}).get("liver_dysfunction")),
        "any_of_cats": ['GLP-1RA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2026, 9.13a',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0393',
        "description": 'For T2DM patients with established ASCVD, initial combination therapy with an SGLT2 inhibitor and a GLP-1 receptor agonist is recommended irrespective of A1C level.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2020
            and _get_dm_comp(cm, "cardiovascular")
        ),
        "any_of_cats": ['SGLT2i', 'GLP-1RA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2026, Chapter 9 (Combination Therapy)',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0394',
        "description": 'For T2DM patients intensifying insulin therapy, maintain metformin, SGLT2i, and GLP-1RA (or dual GIP/GLP-1RA) unless contraindicated; limit or discontinue sulfonylureas, meglitinides, and DPP-4i.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2020
        ),
        "any_of_cats": ['Biguanides', 'SGLT2i', 'GLP-1RA'],
        "exclude_drugs": ['sulfonylureas', 'meglitinides', 'dpp-4 inhibitors'],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2026, Chapter 9 (Combination Injectable Therapy and Prandial Insulin)',
        "domain":      'glycemic_first_line',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'audit:avoid_uncheckable',
    },
    {
        "id":          'FINAL-0395',
        "description": 'Concomitant use of GLP-1RA and DPP-4i is not recommended due to no added glucose-lowering benefit; if initiating GLP-1RA, discontinue DPP-4i.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2026
        ),
        "any_of_cats": ['GLP-1RA'],
        "exclude_drugs": ['dpp-4 inhibitors'],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2026, Chapter 9 (Combination Therapy)',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'audit:avoid_uncheckable',
    },
    {
        "id":          'FINAL-0396',
        "description": 'In adults with T2DM, MASLD, and overweight/obesity, consider GLP-1 RA or dual GIP/GLP-1 RA for glycemic and weight management',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2021 and (cm.get("liver") or {}).get("liver_dysfunction") and (lambda b: b is not None and b >= 25)(_get_bmi_val(pd))),
        "any_of_cats": ['GLP-1RA', 'GIP_GLP1_RA'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2025, Section 9, Recommendation 9.15',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0397',
        "description": 'In adults with T2DM and biopsy-proven MASH or high risk for liver fibrosis, prefer pioglitazone, GLP-1 RA, or dual GIP/GLP-1 RA for glycemic management',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2021 and (cm.get("liver") or {}).get("liver_dysfunction")),
        "any_of_cats": ['Thiazolidinediones', 'GLP-1RA', 'GIP_GLP1_RA'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2025, Section 9, Recommendation 9.16a',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0398',
        "description": 'For patients receiving alpelisib (any DM type), metformin is first-line; SGLT2i or pioglitazone as second/third-line; insulin and sulfonylureas should be last-line.',
        "condition":   lambda cm, pd, yr: (yr >= 2025 and _get_dm_type(cm) in ("T1DM", "T2DM")),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": ['NPH insulin', 'Insulin human', 'Insulin glargine', 'Insulin detemir', 'Insulin degludec', 'Insulin icodec', 'Regular insulin', 'Insulin lispro', 'Insulin aspart', 'Insulin glulisine', 'Faster insulin aspart', 'Glipizide', 'Glyburide', 'Glimepiride', 'Gliclazide', 'Chlorpropamide', 'Tolbutamide', 'Tolazamide', 'Acetohexamide'],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2025, Therapeutic Strategies for Individuals Receiving Cancer Treatment',
        "domain":      'glycemic_first_line',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'audit:missing alpelisib gate -> fires for all yr>=2025; alpelisib exposure undetectable; desc is metformin-first-line not insulin/SU-avoid',
    },
    {
        "id":          'FINAL-0399',
        "description": 'In T2DM with established ASCVD, include SGLT2i or GLP-1RA for cardiovascular/kidney risk reduction',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2025
            and _get_dm_comp(cm, "cardiovascular")
        ),
        "any_of_cats": ['SGLT2i', 'GLP-1RA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2025 Section 9, Recommendation 9.2 (Chunk 1)',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0400',
        "description": 'In T2DM with HF, include SGLT2i for heart failure risk reduction',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2025
            and _get_dm_comp(cm, "heart_failure")
        ),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2025 Section 9, Recommendation 9.2 (Chunk 1)',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0401',
        "description": 'In T2DM without ASCVD, HF, or CKD, metformin is the first-line glucose-lowering agent',
        "condition":   lambda cm, pd, yr: (not _has_ckd_g3plus(cm)
            and _get_dm_type(cm) == "T2DM" and yr >= 2025 and not _get_dm_comp(cm, "cardiovascular") and not _get_dm_comp(cm, "heart_failure") and (lambda egfr: egfr is not None and egfr >= 60)(_get_egfr(cm))),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2025 Section 9 (Chunk 2)',
        "domain":      'glycemic_first_line',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0402',
        "description": 'For T2DM patients with inadequate glycemic control after basal insulin titration, consider adding a GLP-1 RA',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2020 and (lambda h: h is not None and h > 7.0)(cm.get("hba1c"))),
        "any_of_cats": ['GLP-1RA'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2022, Section 9: Combination Injectable Therapy',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0403',
        "description": 'Treat most adults with type 1 diabetes with continuous subcutaneous insulin infusion or multiple daily doses of prandial and basal insulin.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T1DM"
            and yr >= 2025
        ),
        "any_of_cats": ['Insulins_Basal', 'Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care, 2025, Recommendation 9.1',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0404',
        "description": 'For individuals developing hyperglycemia during immunotherapy (anti-PD-1/PD-L1), assess immediate need for initiation of insulin therapy due to potential risk of DKA.',
        "condition":   lambda cm, pd, yr: (yr >= 2026 and (lambda g: g is not None and g > 200)(cm.get("random_glucose_mg_dl"))),
        "any_of_cats": ['Insulins_Basal'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2026, Section 9, Recommendation 9.33',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0405',
        "description": 'Consider metformin as first-line treatment of hyperglycemia due to mTOR inhibitors.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2026 and (lambda h: h is not None and h > 7.0)(cm.get("hba1c"))),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2026, Section 9, Recommendation 9.34',
        "domain":      'glycemic_first_line',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0406',
        "description": 'DPP-4 inhibitor can be considered for mild hyperglycemia in the postoperative setting',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2026 and (lambda g: g is not None and 140 < g <= 180)(cm.get("random_glucose_mg_dl"))),
        "any_of_cats": ['DPP-4i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2026, Chunk 1',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0407',
        "description": 'For T2DM patients with inadequate glycemic control on basal insulin (elevated HbA1c or postprandial hyperglycemia), if not already on a GLP-1 RA or dual GIP/GLP-1 RA, consider adding one before initiating prandial insulin',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2026 and (cm.get("hba1c") is not None and cm.get("hba1c") > 7.5)),
        "any_of_cats": ['GLP-1RA', 'GIP_GLP1_RA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2026, Combination Injectable Therapy and Prandial Insulin',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0408',
        "description": 'Individuals with cystic fibrosis–related diabetes should be treated with insulin therapy',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2026
        ),
        "any_of_cats": ['Insulins_Basal', 'Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2026, Pancreatic Diabetes and Cystic Fibrosis–Related Diabetes',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0409',
        "description": 'Inhaled insulin is contraindicated in patients with chronic lung disease (asthma, COPD) and in those who smoke or recently stopped smoking. Spirometry (FEV1) testing is required before and after starting therapy.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2021
        ),
        "any_of_cats": ['Insulins_Rapid'],
        "exclude_drugs": ['inhaled insulin'],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2021, Inhaled Insulin section',
        "domain":      'glycemic_insulin',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'audit:avoid_uncheckable',
    },
    {
        "id":          'FINAL-0410',
        "description": 'For T2DM patients requiring prandial insulin, consider GLP-1RA first if not already on it',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2023 and (cm.get("hba1c") is not None and cm.get("hba1c") > 7.5)),
        "any_of_cats": ['GLP-1RA'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2023, Prandial Insulin section',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0411',
        "description": 'For T2DM inpatients with ASCVD, discharge with SGLT2i with demonstrated CV benefit (Grade A).',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2022
            and _get_dm_comp(cm, "cardiovascular")
        ),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA 2022 Standards of Medical Care, Recommendation 10.42a',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0412',
        "description": 'T2DM with CKD, prioritize SGLT2i for renoprotection',
        "condition":   lambda cm, pd, yr: (_has_ckd_g3plus(cm)
            and _get_dm_type(cm) == "T2DM" and yr >= 2020 and (lambda egfr: egfr is not None and egfr >= 25)(_get_egfr(cm)) and (lambda uacr: uacr is not None and uacr > 300)((cm.get("kidney") or {}).get("uacr"))),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Consensus Reports & Supplements 2022, #5',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0413',
        "description": 'For T2DM patients with heart failure or ASCVD, prioritize SGLT2i or GLP-1 RA upon discharge',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2022 and (_get_dm_comp(cm, "heart_failure") or _get_dm_comp(cm, "cardiovascular"))),
        "any_of_cats": ['SGLT2i', 'GLP-1RA'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Consensus Reports & Supplements 2022, Management of Hyperglycemia in Individuals With Diabetes at',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0414',
        "description": 'For T2DM with HF or high risk for HF requiring additional glycemic control, favor metformin, GLP-1RA, or insulin',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2022 and _get_dm_comp(cm, "heart_failure")),
        "any_of_cats": ['Biguanides', 'GLP-1RA', 'Insulins_Basal', 'Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Consensus Reports & Supplements 2022 - Summary and Conclusions (HF section)',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0415',
        "description": 'For T2DM patients with ASCVD, HF, or CKD, discharge prescription should prioritize SGLT2i or GLP-1RA for glucose lowering.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2020 and (_get_dm_comp(cm, "cardiovascular") or _get_dm_comp(cm, "heart_failure") or _has_ckd_g3plus(cm) or _has_albuminuria(cm))),
        "any_of_cats": ['SGLT2i', 'GLP-1RA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Consensus Reports & Supplements 2022; Section: The Rationale, Importance, and Context of Glucose-Lowering T',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0416',
        "description": 'In type 2 diabetes (T2DM) patients without contraindications, consider adding GLP-1 RA before initiating insulin therapy to achieve glycemic targets with lower injection burden and hypoglycemia risk.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2022
        ),
        "any_of_cats": ['GLP-1RA'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Consensus Reports & Supplements 2022, Figure 5 – Place of insulin',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0418',
        "description": 'For patients with T2DM and CKD (eGFR 30-89 and UACR ≥300 mg/g), discharge prescription should prioritize SGLT2i (e.g., canagliflozin) to reduce cardiorenal risk.',
        "condition":   lambda cm, pd, yr: (_has_ckd_g3plus(cm)
            and _get_dm_type(cm) == "T2DM" and yr >= 2022 and (lambda egfr: egfr is not None and 30 <= egfr <= 89)(_get_egfr(cm)) and (lambda u: u is not None and u >= 300)((cm.get("kidney") or {}).get("uacr"))),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Consensus Reports & Supplements 2022, New Evidence From Cardiorenal Outcomes Studies',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0419',
        "description": 'Continue organ-protective glucose-lowering medications and metformin when initiating insulin',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2022 and _requires_insulin(cm, pd)),
        "any_of_cats": ['SGLT2i', 'GLP-1RA', 'Biguanides'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Consensus Reports & Supplements 2022, Practical Tips for Clinicians',
        "domain":      'glycemic_first_line',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0420',
        "description": 'Early combination therapy when HbA1c >1.5% above target (e.g., >8.5%)',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2022 and (lambda h: h is not None and h > 8.5)(cm.get("hba1c"))),
        "any_of_cats": ['Biguanides', 'SGLT2i', 'GLP-1RA', 'DPP-4i', 'Insulins_Basal', 'Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Consensus Reports & Supplements 2022, Proactive Care: Avoiding Inertia',
        "domain":      'glycemic_first_line',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0421',
        "description": 'For hospitalized T2DM patients with well-controlled glycemia (HbA1c <7.5%, BG <180 mg/dL, and if on insulin before admission total daily dose <0.6 units/kg/day), DPP-4i can be continued or initiated; at discharge, discuss cost and acceptability if planned for continuation',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2022 and (lambda h: h is not None and h < 7.5)(cm.get("hba1c")) and (lambda g: g is not None and g < 180)(cm.get("random_glucose_mg_dl"))),
        "any_of_cats": ['DPP-4i'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Consensus Reports & Supplements 2022, Question 8 (Remarks section)',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0422',
        "description": 'When intensifying to insulin therapy, maintain metformin, SGLT2i, and GLP-1 RA or dual GIP/GLP-1 RA',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2024),
        "any_of_cats": ['Biguanides', 'SGLT2i', 'GLP-1RA', 'GIP_GLP1_RA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2024 Section 9',
        "domain":      'glycemic_first_line',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0423',
        "description": 'In T2DM with obesity, weight-neutral medications can be used as add-on if more effective weight-loss meds are not tolerated/contraindicated',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2026 and (lambda b: b is not None and b >= 30)(_get_bmi_val(pd))),
        "any_of_cats": ['Biguanides', 'SGLT2i', 'DPP-4i', 'Dopamine_Agonists_DM', 'Bile_Acid_Sequestrants_DM', 'Alpha_Glucosidase_Inhibitors'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2026 - Section 9 Pharmacologic Approaches',
        "domain":      'glycemic_first_line',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'audit:permissive conditional("can be used as add-on if...")miscoded AVOID -> flags weight-neutral agents as commission; not checkable',
    },
    {
        "id":          'FINAL-0424',
        "description": 'Insulins, sulfonylureas, and thiazolidinediones promote weight gain; use judiciously at lowest dose in T2DM with obesity',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2026
        ),
        "any_of_cats": ['Insulins_Basal', 'Insulins_Rapid', 'Sulfonylureas', 'Thiazolidinediones'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2026 - Section 9 Pharmacologic Approaches',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'audit:AVOID+empty-exclude flags all insulin/SU/TZD as commission; desc is dose-caution("use judiciously at lowest dose"), not prohibition; also missing obesity gate',
    },
    {
        "id":          'FINAL-0425',
        "description": 'When initiating intensification of insulin therapy, metformin, SGLT2 inhibitors, and GLP-1 RAs (or dual GIP/GLP-1 RA) should be maintained',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2022
        ),
        "any_of_cats": ['Biguanides', 'SGLT2i', 'GLP-1RA', 'GIP_GLP1_RA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2026 - Combination Injectable Therapy',
        "domain":      'glycemic_first_line',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0426',
        "description": 'For adults with type 1 diabetes, only pramlintide is approved as an adjunct to insulin treatment.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T1DM"
            and yr >= 2021
        ),
        "any_of_cats": ['Amylin_Analogues'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2021 - Noninsulin Treatments for Type 1 Diabetes',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0427',
        "description": 'Consider DPP-4 inhibitor for mild hyperglycemia in the postoperative setting.',
        "condition":   lambda cm, pd, yr: (yr >= 2026 and (lambda g: g is not None and 140 <= g <= 180)((cm.get("glucose") or {}).get("mean_glucose"))),
        "any_of_cats": ['DPP-4i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2026, Inpatient Glycemic Management',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0428',
        "description": 'Most people with type 1 diabetes should be treated with basal insulin',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T1DM"
            and yr >= 2020
        ),
        "any_of_cats": ['Insulins_Basal'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2020 Section 9 Pharmacologic Approaches to Glycemic Treatment',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0429',
        "description": 'Most individuals with type 1 diabetes should use rapid-acting insulin analogs to reduce hypoglycemia risk',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T1DM" and yr >= 2020),
        "any_of_cats": ['Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2020 Section 9 Pharmacologic Approaches to Glycemic Treatment',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0430',
        "description": 'Prefer longer-acting basal analogs (U-300 glargine or degludec) in T1DM with hypoglycemia history',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T1DM" and _has_hypoglycemia_hx(pd) and yr >= 2025),
        "any_of_cats": ['Insulins_Basal'],
        "exclude_drugs": ['Insulin glargine'],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2025 - Insulin Therapy',
        "domain":      'glycemic_insulin',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'audit:同0321,U-300/U-100 glargine不可分->exclude会误罚推荐的U-300;taxonomy无法忠实实现',
    },
    {
        "id":          'FINAL-0431',
        "description": 'Early initiation of metformin is recommended in most patients with T2D and CKD',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and _has_ckd_g3plus(cm) and yr >= 2022),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Consensus Reports & Supplements 2022 - Glucose-Lowering Agents in T2D and CKD',
        "domain":      'glycemic_first_line',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0432',
        "description": 'Essential to continue basal insulin in T1DM even if feedings are discontinued',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T1DM"
            and yr >= 2024
        ),
        "any_of_cats": ['Insulins_Basal'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Consensus Reports & Supplements 2024 | Enteral/Parenteral Feedings',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0433',
        "description": 'Insulin and short-acting sulfonylureas are often necessary to control glucose when medications with less propensity to cause hypoglycemia are contraindicated, not tolerated, unavailable, or insufficient.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2022
            and (lambda egfr: egfr is not None and egfr <= 30)(_get_egfr(cm))
        ),
        "any_of_cats": ['Insulins_Basal', 'Insulins_Rapid', 'Sulfonylureas'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Consensus Reports & Supplements 2022 - Glycemic Management in Advanced CKD',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0434',
        "description": 'Basal insulin is necessary and must not be held for hospitalized T1DM patients, especially during care transitions',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T1DM"
            and yr >= 2024
        ),
        "any_of_cats": ['Insulins_Basal'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Consensus Reports & Supplements 2024 - Type 1 Diabetes',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0476',
        "description": 'Basal insulin (long-acting analogs preferred over NPH) should be the initial choice for insulin therapy in T2DM',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2009 and (cm.get("hba1c") is not None and cm.get("hba1c") >= 9.0)),
        "any_of_cats": ['Insulins_Basal'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      '',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0477',
        "description": 'Metformin is the safest and most commonly used medication to combine with insulin in T2DM',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2009 and (lambda egfr: egfr is not None and 30 <= egfr)(_get_egfr(cm))),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      '',
        "domain":      'glycemic_first_line',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0478',
        "description": 'SGLT2i may be used for T2DM when eGFR >= 45 mL/min/1.73m²',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2013
            and (lambda egfr: egfr is not None and 45 <= egfr)(_get_egfr(cm))
        ),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      '',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0479',
        "description": 'For T2DM, if metformin contraindicated, use TZD as base; second agent priority: GLP-1RA > DPP-4i > glinide > sulfonylurea (dual therapy)',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2009 and ((lambda egfr: egfr is not None and egfr < 30)(_get_egfr(cm)) or _get_dm_comp(cm, "heart_failure"))),
        "any_of_cats": ['Thiazolidinediones', 'GLP-1RA', 'DPP-4i', 'Meglitinides', 'Sulfonylureas'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      '',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0480',
        "description": 'Prefer GLP-1 agonists and DPP-4 inhibitors over sulfonylureas in T2DM due to better safety profile',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2009
        ),
        "any_of_cats": ['GLP-1RA', 'DPP-4i'],
        "exclude_drugs": ['Glipizide', 'Glyburide', 'Glimepiride', 'Gliclazide', 'Chlorpropamide', 'Tolbutamide', 'Tolazamide', 'Acetohexamide'],
        "grade":       0.75,
        "source":      '',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0481',
        "description": 'In T2DM with overweight/obesity and weight loss goal, prefer SGLT2i.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2023 and (lambda b: b is not None and b >= 25)(_get_bmi_val(pd))),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      '',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0482',
        "description": 'Metformin as first-line monotherapy for T2DM',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2009
        ),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      '',
        "domain":      'glycemic_first_line',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0483',
        "description": 'Thiazolidinediones (TZDs) are contraindicated in patients with NYHA class III or IV congestive heart failure',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2009
            and _get_dm_comp(cm, "heart_failure")
        ),
        "any_of_cats": ['Thiazolidinediones'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      '',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'audit:avoid_uncheckable',
    },
    {
        "id":          'FINAL-0484',
        "description": 'For T2DM patients requiring intensification of glycemic control who are already on 3-4 oral therapies, a GLP-1 RA should be the initial choice if not already in use.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2010 and (cm.get("hba1c") is not None and cm.get("hba1c") > 7.0)),
        "any_of_cats": ['GLP-1RA'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      '',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0485',
        "description": 'For T2DM patients requiring insulin therapy, discharge prescription should prioritize basal-bolus regimen (long-acting analogue + rapid-acting analogue) over premixed insulins (including premixed analogues and premixed human insulin/NPH).',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2009 and (cm.get("hba1c") is not None and cm.get("hba1c") > 9.0)),
        "any_of_cats": ['Insulins_Basal', 'Insulins_Rapid'],
        "exclude_drugs": ['premixed insulin', 'novolog mix', 'humalog mix', 'nph', 'regular human insulin'],
        "grade":       0.75,
        "source":      '',
        "domain":      'glycemic_insulin',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'P2-A: 预混胰岛素无法干净分类 + nph token 误报,停用',
    },
    {
        "id":          'FINAL-0486',
        "description": 'For obese T2DM patients using pre-meal insulin, consider adding pramlintide at discharge to reduce postprandial hyperglycemia and promote weight loss.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2009 and (_get_bmi_val(pd) is not None and _get_bmi_val(pd) >= 30)),
        "any_of_cats": ['Amylin_Analogues'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      '',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0487',
        "description": 'For T2DM patients initiating basal insulin at discharge, start TDD 0.1-0.2 U/kg if A1c>8%, or 0.2-0.3 U/kg if A1c<8%. Titrate every 2-3 days based on FBG.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2013
        ),
        "any_of_cats": ['Insulins_Basal'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      '',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0488',
        "description": 'For T2DM patients requiring intensification but reluctant to inject insulin or at risk of hypoglycemia, consider GLP-1 RA or DPP-4i as alternatives at discharge.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2013 and (cm.get("hba1c") is not None and cm.get("hba1c") > 7.5) and _has_hypoglycemia_hx(pd)),
        "any_of_cats": ['GLP-1RA', 'DPP-4i'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      '',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0489',
        "description": 'For T2DM patients inadequately controlled with metformin, sulfonylurea, or insulin, consider adding colesevelam (bile acid sequestrant) for glycemic control',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2009 and cm.get("hba1c") is not None and cm.get("hba1c") > 7.0),
        "any_of_cats": ['Bile_Acid_Sequestrants_DM'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      '',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0490',
        "description": 'Metformin contraindicated when eGFR < 45 mL/min (Stage 3B,4,5)',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2013
            and (lambda egfr: egfr is not None and egfr <= 44)(_get_egfr(cm))
        ),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": ['metformin'],
        "grade":       1.0,
        "source":      '',
        "domain":      'glycemic_first_line',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0491',
        "description": 'Exenatide contraindicated when CrCl < 30 mL/min',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2013
            and (lambda egfr: egfr is not None and egfr <= 29)(_get_egfr(cm))
        ),
        "any_of_cats": ['GLP-1RA'],
        "exclude_drugs": ['exenatide'],
        "grade":       0.75,
        "source":      '',
        "domain":      'cardiorenal_protection',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0492',
        "description": 'Dosage of DPP-4 inhibitors must be reduced for T2DM patients with moderate or severe renal insufficiency',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2009 and _has_ckd_g3plus(cm)),
        "any_of_cats": ['DPP-4i'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      '',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0493',
        "description": 'In T2DM patients requiring prandial insulin, rapid-acting insulin analogues are preferred over regular human insulin.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2009 and cm.get("hba1c") is not None and cm.get("hba1c") > 7.0),
        "any_of_cats": ['Insulins_Rapid'],
        "exclude_drugs": ['Regular insulin', 'regular insulin'],
        "grade":       1.0,
        "source":      '',
        "domain":      'glycemic_insulin',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0494',
        "description": 'Intravenous insulin infusions are the preferred method for achieving and maintaining glycemic control in critically ill patients',
        "condition":   lambda cm, pd, yr: (yr >= 2009 and _is_icu_approx(pd)),
        "any_of_cats": ['Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      '',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0495',
        "description": 'Insulin infusion should be used to control hyperglycemia in critically ill patients in the ICU setting',
        "condition":   lambda cm, pd, yr: (yr >= 2009 and _is_icu_approx(pd)),
        "any_of_cats": ['Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      '',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0496',
        "description": 'Triple therapy initiation for HbA1c>9% with MET+TZD+SU/Glinide',
        "condition":   lambda cm, pd, yr: _get_dm_type(cm) == "T2DM" and yr >= 2009 and (lambda h: h is not None and h > 9)(cm.get("hba1c")),
        "any_of_cats": ['Biguanides', 'Thiazolidinediones', 'Sulfonylureas'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      '',
        "domain":      'glycemic_first_line',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0497',
        "description": 'Add GLP-1RA/SGLT2i/DPP-4i when basal insulin fails',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2013 and _requires_insulin(cm, pd) and (lambda h: h is not None and h > 7)(cm.get("hba1c"))),
        "any_of_cats": ['GLP-1RA', 'SGLT2i', 'DPP-4i'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      '',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0498',
        "description": 'Prioritize Metformin as foundational therapy for T2DM unless contraindicated',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2009
            and (lambda egfr: egfr is not None and 60 <= egfr)(_get_egfr(cm))
        ),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      '',
        "domain":      'glycemic_first_line',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0500',
        "description": 'Initiate metformin in T2DM without ASCVD, HF, stroke, or CKD, if no contraindication (GFR <30).',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2000 and not _get_dm_comp(cm, "cardiovascular") and not _get_dm_comp(cm, "heart_failure") and not _has_stroke_hx(pd) and not _get_dm_comp(cm, "nephropathy") and (lambda egfr: egfr is not None and egfr >= 30)(_get_egfr(cm))),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      '',
        "domain":      'glycemic_first_line',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0501',
        "description": 'In T2DM with history of or high risk for hypoglycemia, prefer agents with low hypoglycemia risk: GLP-1RA, SGLT2i, dual GIP/GLP-1 RA, TZD, DPP-4i.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2000 and _has_hypoglycemia_hx(pd)),
        "any_of_cats": ['GLP-1RA', 'SGLT2i', 'GIP_GLP1_RA', 'Thiazolidinediones', 'DPP-4i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      '',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0517',
        "description": 'For T2DM patients with DKD (eGFR ≥15 mL/min/1.73m²), GLP-1RA is an option to reduce progression of albuminuria, eGFR decline, and ASCVD risk',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2023 and _get_dm_comp(cm, "nephropathy") and (lambda egfr: egfr is not None and egfr >= 15)(_get_egfr(cm))),
        "any_of_cats": ['GLP-1RA'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      '',
        "domain":      'cardiorenal_protection',
        "polarity":    'ALLOW',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0518',
        "description": 'For T2DM patients with established CVD, consider SGLT2i (empagliflozin, canagliflozin) to reduce cardiovascular death and heart failure hospitalization',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2016
            and _get_dm_comp(cm, "cardiovascular")
            and (lambda egfr: egfr is not None and 45 <= egfr)(_get_egfr(cm))
        ),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      '',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0519',
        "description": 'In T2DM with overweight/obesity and weight loss goal, prefer GLP-1 RA (e.g., semaglutide, liraglutide).',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2014 and (lambda bmi: bmi is not None and bmi >= 25)(_get_bmi_val(pd))),
        "any_of_cats": ['GLP-1RA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      '',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0520',
        "description": 'For overweight/obese T2DM patients (BMI>=25), prefer SGLT2i, GLP-1RA, or GIP/GLP-1RA at discharge',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2016 and (lambda b: b is not None and b >= 25)(_get_bmi_val(pd))),
        "any_of_cats": ['SGLT2i', 'GLP-1RA', 'GIP_GLP1_RA', 'GCG_GLP1_RA', 'GLP1_Multimer_RA'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      '',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0521',
        "description": 'For T2DM with severe hyperglycemia (HbA1c>9% or >1.5% above goal), discharge prescription should include basal insulin plus prandial insulin or GLP-1RA/GIP/GLP-1RA',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2016 and (lambda h: h is not None and h > 9)(cm.get("hba1c"))),
        "any_of_cats": ['Insulins_Basal', 'Insulins_Rapid', 'GLP-1RA', 'GIP_GLP1_RA', 'GCG_GLP1_RA', 'GLP1_Multimer_RA'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      '',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0522',
        "description": 'If A1C >7.5%, start two agents; if >9% or >1.5% above goal, start two to three agents at discharge',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2016 and (lambda h: h is not None and h > 7.5)(cm.get("hba1c"))),
        "any_of_cats": ['Biguanides', 'SGLT2i', 'GLP-1RA', 'DPP-4i', 'Insulins_Basal', 'Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      '',
        "domain":      'glycemic_first_line',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0523',
        "description": 'In T2DM patients with established ASCVD or heart failure, prioritize SGLT2 inhibitors at discharge to reduce cardiovascular death and HF hospitalization',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2016 and (_get_dm_comp(cm, "cardiovascular") or _get_dm_comp(cm, "heart_failure")) and (lambda egfr: egfr is not None and egfr >= 45)(_get_egfr(cm))),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      '',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0524',
        "description": 'For T2DM patients requiring insulin therapy with a history of hypoglycemia, prefer basal insulin analogs (e.g., glargine U300, degludec) over NPH insulin to reduce risk of hypoglycemia.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2018 and _has_hypoglycemia_hx(pd)),
        "any_of_cats": ['Insulins_Basal'],
        "exclude_drugs": ['NPH insulin'],
        "grade":       1.0,
        "source":      '',
        "domain":      'glycemic_insulin',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0525',
        "description": 'Add GLP-1RA, SGLT2i, or DPP-4i for T2DM patients with uncontrolled basal insulin to minimize weight gain and hypoglycemia',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2018 and (cm.get("hba1c") is not None and cm.get("hba1c") > 7.5)),
        "any_of_cats": ['GLP-1RA', 'SGLT2i', 'DPP-4i'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      '',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0526',
        "description": 'In T2DM requiring prandial insulin, rapid-acting insulin analogs are preferred over regular human insulin to reduce hypoglycemia.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2016 and (cm.get("hba1c") is not None and cm.get("hba1c") > 9.0)),
        "any_of_cats": ['Insulins_Rapid'],
        "exclude_drugs": ['Regular insulin'],
        "grade":       1.0,
        "source":      '',
        "domain":      'glycemic_insulin',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0527',
        "description": 'Pramlintide is indicated for use with basal-bolus insulin regimens in T2DM.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2016
        ),
        "any_of_cats": ['Amylin_Analogues'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      '',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0537',
        "description": 'To reduce hypoglycemia risk, prioritize DPP-4i, GLP-1 RA, or SGLT2i over insulin or sulfonylureas in T2DM at discharge.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2023 and _has_hypoglycemia_hx(pd)),
        "any_of_cats": ['DPP-4i', 'GLP-1RA', 'SGLT2i'],
        "exclude_drugs": ['NPH insulin', 'Insulin human', 'Insulin glargine', 'Insulin detemir', 'Insulin degludec', 'Insulin icodec', 'Regular insulin', 'Insulin lispro', 'Insulin aspart', 'Insulin glulisine', 'Faster insulin aspart', 'Glipizide', 'Glyburide', 'Glimepiride', 'Gliclazide', 'Chlorpropamide', 'Tolbutamide', 'Tolazamide', 'Acetohexamide'],
        "grade":       0.75,
        "source":      '',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0538',
        "description": 'In T2DM with overweight/obesity and weight loss goal, prefer dual GIP/GLP-1 RA (tirzepatide).',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2022 and (lambda b: b is not None and b >= 25)(_get_bmi_val(pd))),
        "any_of_cats": ['GIP_GLP1_RA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      '',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0539',
        "description": 'For T2DM patients with severe hyperglycemia (HbA1c>10%), advise basal insulin ± GLP-1RA',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2023 and (lambda h: h is not None and h > 10)(cm.get("hba1c"))),
        "any_of_cats": ['Insulins_Basal', 'GLP-1RA'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      '',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0540',
        "description": 'For T2DM patients initiating basal insulin, prefer long-acting analogs (detemir, glargine, degludec) over NPH to reduce hypoglycemia.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2023
        ),
        "any_of_cats": ['Insulins_Basal'],
        "exclude_drugs": ['NPH insulin'],
        "grade":       0.75,
        "source":      '',
        "domain":      'glycemic_insulin',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0541',
        "description": 'Discontinue sulfonylureas after initiating basal insulin in T2DM patients to avoid hypoglycemia from overlapping therapy.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2023
        ),
        "any_of_cats": ['Sulfonylureas'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      '',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0542',
        "description": 'Prefer dual GIP/GLP-1 RA, GLP-1 RA, or SGLT2i for weight loss in overweight/obese T2DM without ASCVD/HF/CKD',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2023 and not _get_dm_comp(cm, "cardiovascular") and not _get_dm_comp(cm, "heart_failure") and not _has_ckd_g3plus(cm) and (lambda b: b is not None and b >= 25)(_get_bmi_val(pd))),
        "any_of_cats": ['GIP_GLP1_RA', 'GLP-1RA', 'SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      '',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0543',
        "description": 'Prefer agents with low hypoglycemia risk (GLP-1 RA, SGLT2i, GIP/GLP-1 RA, TZD, DPP-4i) in T2DM with hypoglycemia history/risk',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2023 and _has_hypoglycemia_hx(pd)),
        "any_of_cats": ['GLP-1RA', 'SGLT2i', 'GIP_GLP1_RA', 'Thiazolidinediones', 'DPP-4i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      '',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0544',
        "description": 'For overweight or obese T2DM patients, GLP-1 RAs, GIP/GLP-1 RAs, or SGLT2i are preferred',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2023 and (lambda b: b is not None and b >= 25)(_get_bmi_val(pd))),
        "any_of_cats": ['GLP-1RA', 'GIP_GLP1_RA', 'SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      '',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0672',
        "description": 'Pioglitazone may be considered in T2DM with ASCVD without HF if additional glucose control needed',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2005
            and _get_dm_comp(cm, "cardiovascular")
            and not _get_dm_comp(cm, "heart_failure")
        ),
        "any_of_cats": ['Thiazolidinediones'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ESC CVD in Diabetes 2023, Section 2.2',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0673',
        "description": 'Insulin-based glycaemic control considered in ACS with significant hyperglycaemia',
        "condition":   lambda cm, pd, yr: (yr >= 2013 and _get_dm_type(cm) in ("T1DM", "T2DM") and _has_acs(pd) and (lambda g: g is not None and g > 180)((cm.get("glucose") or {}).get("max_glucose"))),
        "any_of_cats": ['Insulins_Basal'],
        "exclude_drugs": [],
        "grade":       0.5,
        "source":      'ESC CVD in Diabetes 2023 (2013 version) 7.1.9',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0674',
        "description": 'Thiazolidinediones should not be used in heart failure and T2DM',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2013
            and _get_dm_comp(cm, "heart_failure")
        ),
        "any_of_cats": ['Thiazolidinediones'],
        "exclude_drugs": ['Pioglitazone', 'Rosiglitazone'],
        "grade":       0.75,
        "source":      'ESC CVD in Diabetes 2023 (2013 version) 8.7',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0675',
        "description": 'In T2DM with CKD, DPP-4i (e.g., linagliptin) is an alternative to GLP-1RA, safely lowering HbA1c without CVD/kidney benefit.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2012 and _has_ckd_g3plus(cm)),
        "any_of_cats": ['DPP-4i'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ESC CVD in Diabetes 2023 | 9.3. Blood pressure and glycaemic control',
        "domain":      'glycemic_intensification',
        "polarity":    'ALLOW',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0676',
        "description": 'Avoid Saxagliptin in T2DM patients at risk of or with prior HF',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2013
            and _get_dm_comp(cm, "heart_failure")
        ),
        "any_of_cats": ['DPP-4i'],
        "exclude_drugs": ['saxagliptin'],
        "grade":       0.75,
        "source":      'ESC CVD in Diabetes 2023 | 17. ‘What to do’ and ‘What not to do’ messages | 2023',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0677',
        "description": 'For T2DM patients with eGFR ≥50 mL/min, metformin should be prescribed as first-line glucose-lowering therapy',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2013
            and (lambda egfr: egfr is not None and 50 <= egfr)(_get_egfr(cm))
        ),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ESC CVD in Diabetes 2013 | 6.2.6 Glucose-lowering agents',
        "domain":      'glycemic_first_line',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0751',
        "description": 'Consider metformin in T2DM with ASCVD if additional glucose control needed',
        "condition":   lambda cm, pd, yr: (yr >= 2001 and _get_dm_type(cm) == "T2DM" and _get_dm_comp(cm, "cardiovascular")),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ESC CVD in Diabetes 2023, Section 2.2',
        "domain":      'glycemic_first_line',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0752',
        "description": 'In T2DM patients with ASCVD who are already on metformin, SGLT2i and/or GLP-1RA should be added regardless of glycemic control need',
        "condition":   lambda cm, pd, yr: (yr >= 2001 and _get_dm_type(cm) == "T2DM" and _get_dm_comp(cm, "cardiovascular")),
        "any_of_cats": ['SGLT2i', 'GLP-1RA'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ESC CVD in Diabetes 2023, 5.3.3.1. Metformin, Class IIa',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0753',
        "description": 'For T2DM patients without ASCVD or severe TOD, consider metformin based on CV risk level (should be considered for low/moderate risk; may be considered for high/very high risk)',
        "condition":   lambda cm, pd, yr: (yr >= 2000 and _get_dm_type(cm) == "T2DM" and not _get_dm_comp(cm, "cardiovascular")),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ESC CVD in Diabetes 2023, 5.3.3.1. Metformin',
        "domain":      'glycemic_first_line',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0754',
        "description": 'Insulin therapy for diabetic patients with ACS and significant hyperglycemia (glucose >10 mmol/L or >180 mg/dL)',
        "condition":   lambda cm, pd, yr: (yr >= 1997 and _get_dm_type(cm) in ("T1DM", "T2DM") and _has_acs(pd) and (cm.get("random_glucose_mg_dl") is not None and cm.get("random_glucose_mg_dl") > 180)),
        "any_of_cats": ['Insulins_Rapid', 'Insulins_Basal'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ESC CVD in Diabetes 2023 2013, Section 7.1.7',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0755',
        "description": 'First-line treatment of DM in HF should include metformin',
        "condition":   lambda cm, pd, yr: (yr >= 2000 and _get_dm_type(cm) == "T2DM" and _get_dm_comp(cm, "heart_failure")),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2019 | Key messages & Section 8',
        "domain":      'glycemic_first_line',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0805',
        "description": 'When T2DM patients are on basal-bolus insulin regimen, discontinue sulfonylureas, DPP-4 inhibitors, and GLP-1 receptor agonists',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2015
        ),
        "any_of_cats": ['Insulins_Basal', 'Insulins_Rapid'],
        "exclude_drugs": ['sulfonylureas', 'dpp-4 inhibitors', 'glp-1 receptor agonists'],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2015, Insulin Therapy section, Figure 7.2 and accompanying text',
        "domain":      'glycemic_insulin',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'audit:avoid_uncheckable',
    },
    {
        "id":          'FINAL-0806',
        "description": 'SGLT2i recommended for T2DM patients with ASCVD or high/very high CV risk to reduce CV events',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2019 and _get_dm_comp(cm, "cardiovascular")),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 2019, Section 7.1.2.2',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0807',
        "description": 'GLP-1RA recommended for T2DM patients with ASCVD or high/very high CV risk to reduce CV events',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2019 and _get_dm_comp(cm, "cardiovascular")),
        "any_of_cats": ['GLP-1RA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 2019, Section 7.1.2.2',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0808',
        "description": 'Metformin may be considered in overweight T2DM patients without CVD and at moderate CV risk',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2019 and not _get_dm_comp(cm, "cardiovascular") and (lambda b: b is not None and b >= 25)(_get_bmi_val(pd))),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ESC CVD in Diabetes 2023 2019, Section 7.1.2.2',
        "domain":      'glycemic_first_line',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0809',
        "description": 'In patients with T2DM and CKD, initiate SGLT2 inhibitor alongside ACEi/ARB when eGFR ≥20 mL/min/1.73m²',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2019 and (_has_ckd_g3plus(cm) or _has_albuminuria(cm)) and (lambda egfr: egfr is not None and 20 <= egfr)(_get_egfr(cm))),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023, 9.2 Management of cardiovascular disease risk and kidney function',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0810',
        "description": 'For T2DM patients with HFpEF/HFmrEF, metformin should be the first-line oral hypoglycaemic agent.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2016
            and (_has_hfmref(cm, pd) or _has_hfpef(cm, pd))
        ),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 2016, Section 9.4',
        "domain":      'glycemic_first_line',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0812',
        "description": 'In T2DM patients with eGFR >30 mL/min/1.73m², GLP-1 RA (liraglutide, semaglutide) should be considered for diabetes treatment.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2017 and (lambda egfr: egfr is not None and egfr > 30)(_get_egfr(cm))),
        "any_of_cats": ['GLP-1RA'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ESC CVD in Diabetes 2023, Section 9, Chronic kidney disease and diabetes',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0813',
        "description": 'First-line treatment for DM with HF should include metformin and SGLT2 inhibitors',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2019
            and _get_dm_comp(cm, "heart_failure")
        ),
        "any_of_cats": ['Biguanides', 'SGLT2i'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ESC CVD in Diabetes 2023 2019, Key messages',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0814',
        "description": 'SGLT2i recommended in T2DM patients with CV disease, CV risk factors, or CKD to reduce HF hospitalization, MACE, ESRD, and CV death',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2016 and (_get_dm_comp(cm, "cardiovascular") or _has_ckd_g3plus(cm) or _has_albuminuria(cm))),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023, section 13.1 Diabetes, 2021 version (based on EMPA-REG 2015, CANVAS 2017, DECLARE 2019, CREDENCE 2019)',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0815',
        "description": 'Metformin should be considered as a first-line treatment of glycaemic control in patients with diabetes and HF, unless contra-indicated.',
        "condition":   lambda cm, pd, yr: (yr >= 2016 and _get_dm_type(cm) in ("T1DM", "T2DM") and _get_dm_comp(cm, "heart_failure")),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ESC CVD in Diabetes 2023 (2016 Section) | Recommendations for the treatment of other co-morbidities',
        "domain":      'glycemic_first_line',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0816',
        "description": 'For T2DM patients, after metformin as first-line therapy, consider SGLT2i or GLP-1RA',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2019),
        "any_of_cats": ['SGLT2i', 'GLP-1RA'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ACC-AHA Cholesterol/Cardiovascular 2019, Top 10 Take-Home Messages #6',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0817',
        "description": 'In T2DM patients with established ASCVD, prefer SGLT2i or GLP-1RA as glucose-lowering therapy at discharge to reduce CVD events.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2019
            and _get_dm_comp(cm, "cardiovascular")
        ),
        "any_of_cats": ['SGLT2i', 'GLP-1RA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ACC-AHA Cholesterol / Cardiovascular 2019, Section 4.2-4.3, Recommendation-Specific Supportive Text',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0818',
        "description": 'Continue SGLT2i in patients with established cardiovascular disease if already prescribed',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2018 and _get_dm_comp(cm, "cardiovascular")),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Consensus Reports & Supplements 2018 (figure description)',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0819',
        "description": 'SGLT2 inhibitors are recommended for patients with DM at high risk of HF.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2016
            and _get_dm_comp(cm, "heart_failure")
        ),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 | 8.5.6 | 2019',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0932',
        "description": 'In T2DM patients with eGFR >15 mL/min/1.73m², GLP-1 RA is recommended to achieve adequate glycaemic control due to low risk of hypoglycaemia and beneficial effects on weight, CV risk, and albuminuria.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2023 and (lambda egfr: egfr is not None and egfr > 15)(_get_egfr(cm))),
        "any_of_cats": ['GLP-1RA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023, Section 9, Chronic kidney disease and diabetes',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0933',
        "description": 'SGLT2 inhibitors recommended for all patients with type 2 diabetes',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2021
        ),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 (2021 version), Key message 19',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0934',
        "description": 'SGLT2i recommended for glycemic control in T2DM patients with heart failure',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2022
            and _get_dm_comp(cm, "heart_failure")
        ),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ACC-AHA Cholesterol/Cardiovascular 2022',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0979',
        "description": 'Metformin is the recommended first-line glucose lowering treatment in T2DM patients with heart failure.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2013
            and _get_dm_comp(cm, "heart_failure")
        ),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ESC CVD in Diabetes 2013 | 8.5 Glucose-lowering treatment in patients with heart failure',
        "domain":      'glycemic_first_line',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0993',
        "description": 'For T2DM patients at high CV risk (ASCVD, HF, or CKD), discharge prescription should include SGLT2i to prevent HF hospitalization and CV events.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2021 and (_get_dm_comp(cm, "cardiovascular") or _get_dm_comp(cm, "heart_failure") or _has_ckd_g3plus(cm)) and (lambda egfr: egfr is not None and egfr >= 20)(_get_egfr(cm))),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023, Chunk3 (2021) HF and diabetes section',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0994',
        "description": 'In T2DM patients with HF, GLP-1RA with neutral effects on HF is recommended for glycemic control without worsening HF (Class IIa)',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2019
            and _get_dm_comp(cm, "heart_failure")
        ),
        "any_of_cats": ['GLP-1RA'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ESC CVD in Diabetes 2023 | 7.1. Other glucose-lowering agents with neutral effects on HF',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0995',
        "description": 'In T2DM patients with HF, DPP-4i (sitagliptin, linagliptin) with neutral effects on HF is recommended (Class IIa), excluding those that increase HF risk',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2019
            and _get_dm_comp(cm, "heart_failure")
        ),
        "any_of_cats": ['DPP-4i'],
        "exclude_drugs": ['saxagliptin', 'alogliptin', 'vildagliptin'],
        "grade":       0.75,
        "source":      'ESC CVD in Diabetes 2023 | 7.1. Other glucose-lowering agents with neutral effects on HF',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0996',
        "description": 'In T2DM patients with HF, Basal Insulin (glargine, degludec) with neutral effects on HF is recommended (Class IIa)',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2019
            and _get_dm_comp(cm, "heart_failure")
        ),
        "any_of_cats": ['Insulins_Basal'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ESC CVD in Diabetes 2023 | 7.1. Other glucose-lowering agents with neutral effects on HF',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1007',
        "description": 'GLP-1 RAs should be considered for glucose-lowering treatment in T2DM patients at risk of or with HF (neutral HF hospitalization risk)',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2023 and _get_dm_comp(cm, "heart_failure")),
        "any_of_cats": ['GLP-1RA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 2023, Section 7 Heart failure and diabetes',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1008',
        "description": 'For T2DM patients with HFrEF, discharge prescription should prioritize SGLT2i (dapagliflozin or empagliflozin).',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2020 and _has_hfref(cm, pd) and (lambda egfr: egfr is not None and 20 <= egfr)(_get_egfr(cm))),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023, Chunk3 (2021) HF and diabetes section',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1025',
        "description": 'For T2DM with CKD and eGFR ≥45, recommend continuing or initiating metformin',
        "condition":   lambda cm, pd, yr: (_has_ckd_g3plus(cm)
            and 
            _get_dm_type(cm) == "T2DM"
            and yr >= 2012
            and (lambda egfr: egfr is not None and 45 <= egfr)(_get_egfr(cm))
        ),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'KDIGO 2012 CKD & Diabetes-CKD Guidelines, 4.4.6',
        "domain":      'glycemic_first_line',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1026',
        "description": 'Continue metformin in people with eGFR >= 45 mL/min/1.73m²',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) in ("T1DM", "T2DM") and yr >= 2012 and (lambda egfr: egfr is not None and 45 <= egfr)(_get_egfr(cm))),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'KDIGO CKD 2012 4.4.6',
        "domain":      'glycemic_first_line',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1027',
        "description": 'Continue metformin in patients with CKD and eGFR >= 45 mL/min/1.73m2',
        "condition":   lambda cm, pd, yr: (_has_ckd_g3plus(cm)
            and 
            _get_dm_type(cm) == "T2DM"
            and yr >= 2012
            and (lambda egfr: egfr is not None and 45 <= egfr)(_get_egfr(cm))
        ),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'KDIGO CKD 2012 4.4.6',
        "domain":      'glycemic_first_line',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1028',
        "description": 'Initiate metformin at half the dose and monitor kidney function more frequently for patients with eGFR 30-44 mL/min/1.73m²',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) in ("T1DM", "T2DM") and yr >= 2020 and (lambda egfr: egfr is not None and 30 <= egfr and egfr <= 44)(_get_egfr(cm))),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'KDIGO CKD & Diabetes-CKD Guidelines 2020, Figure 1',
        "domain":      'glycemic_first_line',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1038',
        "description": 'Use metformin as first-line glucose-lowering therapy in T2DM patients with CKD and eGFR >=30.',
        "condition":   lambda cm, pd, yr: (_has_ckd_g3plus(cm)
            and _get_dm_type(cm) == "T2DM" and yr >= 2001 and (lambda egfr: egfr is not None and egfr >= 30)(_get_egfr(cm))),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'KDIGO CKD & Diabetes-CKD Guidelines 2022, Section 1.1, Figure 1',
        "domain":      'glycemic_first_line',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1050',
        "description": 'SGLT2i preferred as glucose-lowering agent in T2DM with CKD (eGFR≥25) due to cardiorenal benefits.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2019 and (_has_ckd_g3plus(cm) or _has_albuminuria(cm)) and (lambda egfr: egfr is not None and 25 <= egfr)(_get_egfr(cm)) and not (cm.get("kidney") or {}).get("aki_suspected", False)),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'KDIGO CKD & Diabetes-CKD Guidelines 2022 References (chunk2)',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1051',
        "description": 'First-line SGLT2i for T2DM and CKD patients with eGFR >= 30 to delay CKD progression.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2019 and (_has_ckd_g3plus(cm) or _has_albuminuria(cm)) and (lambda egfr: egfr is not None and 30 <= egfr)(_get_egfr(cm))),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'KDIGO CKD & Diabetes-CKD Guidelines 2022 Chapter 4',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1062',
        "description": 'For T2D with CKD and eGFR≥30, recommend metformin + SGLT2i as first-line antihyperglycemic therapy',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2020 and (lambda egfr: egfr is not None and 30 <= egfr)(_get_egfr(cm)) and (_has_ckd_g3plus(cm) or _has_albuminuria(cm))),
        "any_of_cats": ['Biguanides', 'SGLT2i'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'KDIGO 2020 CKD & Diabetes Guidelines, Chapter 4, Practice Point 4.1',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1063',
        "description": 'For patients with T2D, CKD, and eGFR ≥30 ml/min/1.73m², SGLT2i are preferred as initial antihyperglycemic and organ-protective agents with metformin',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2020 and (lambda egfr: egfr is not None and 30 <= egfr)(_get_egfr(cm)) and (_has_ckd_g3plus(cm) or _has_albuminuria(cm))),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'KDIGO 2020 CKD & Diabetes-CKD Guidelines, Chunk 2',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1064',
        "description": 'For T2D with CKD and eGFR ≥30, GLP-1 RA (liraglutide, semaglutide, dulaglutide) as addition if glycemic target not achieved or unable to tolerate metformin/SGLT2i',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2020 and _get_dm_comp(cm, "nephropathy") and (lambda egfr: egfr is not None and egfr >= 30)(_get_egfr(cm))),
        "any_of_cats": ['GLP-1RA'],
        "exclude_drugs": ['exenatide', 'lixisenatide'],
        "grade":       0.75,
        "source":      'KDIGO 2020 CKD & Diabetes-CKD Guidelines, Chunk 2',
        "domain":      'cardiorenal_protection',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1065',
        "description": 'In hospitalized T2D patients with CKD who do not achieve glycemic targets despite metformin and SGLT2i, or who cannot use those medications, add a long-acting GLP-1RA with documented cardiovascular benefit.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2022 and _get_dm_comp(cm, "nephropathy")),
        "any_of_cats": ['GLP-1RA'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'KDIGO 2024 Clinical Practice Guideline for Diabetes Management in Chronic Kidney Disease (Recommendation 3.9.1, referencing KDIGO 2022)',
        "domain":      'cardiorenal_protection',
        # [P0 FIX] desc 为肯定式("add a long-acting GLP-1RA"),原 AVOID + exclude_drugs=[]
        #          会罚掉 T2DM+CKD 患者的全部 GLP-1RA,与同域 FINAL-0820 直接矛盾。
        #          纠正为 RECOMMEND。
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1066',
        "description": 'For T1D and CKD, insulin is the foundation of glycemic control',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T1DM" and yr >= 2020 and _get_dm_comp(cm, "nephropathy")),
        "any_of_cats": ['Insulins_Basal', 'Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'KDIGO CKD & Diabetes-CKD Guidelines 2020, Figure 2',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1068',
        "description": 'For T2D patients with CKD and eGFR ≥20 ml/min, SGLT2i should be used as first-line therapy for glucose lowering and for kidney and cardiovascular protection.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2022 and _get_dm_comp(cm, "nephropathy") and (lambda egfr: egfr is not None and egfr >= 20)(_get_egfr(cm))),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'KDIGO 2022 Clinical Practice Guideline for Diabetes Management in CKD, Chapter 4, Practice Point 4.1 and Section 1.3',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1069',
        "description": 'Recommend combination of metformin and SGLT2i in patients with T2D and CKD and eGFR ≥30.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2022 and _get_dm_comp(cm, "nephropathy") and (lambda egfr: egfr is not None and egfr >= 30)(_get_egfr(cm))),
        "any_of_cats": ['Biguanides', 'SGLT2i'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'KDIGO CKD & Diabetes-CKD Guidelines 2022, Chapter 4, Figure 23',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1070',
        "description": 'Metformin should be dose-reduced when eGFR is 30-45 in patients with T2D and CKD.',
        "condition":   lambda cm, pd, yr: (_has_ckd_g3plus(cm)
            and 
            _get_dm_type(cm) == "T2DM"
            and yr >= 2022
            and (lambda egfr: egfr is not None and 30 <= egfr and egfr <= 45)(_get_egfr(cm))
        ),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'KDIGO CKD & Diabetes-CKD Guidelines 2022, Chapter 4, Figure 23',
        "domain":      'glycemic_first_line',
        "polarity":    'RECOMMEND',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'audit:avoid_uncheckable',
    },
    {
        "id":          'FINAL-1071',
        "description": 'In T2D with ASCVD, GLP-1 RA is the preferred additional glucose-lowering agent.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2022
            and _get_dm_comp(cm, "cardiovascular")
        ),
        "any_of_cats": ['GLP-1RA'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'KDIGO CKD & Diabetes-CKD Guidelines 2022, Practice Point 4.3',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1072',
        "description": 'In T2D with eGFR <60, GLP-1 RA is the preferred additional agent due to kidney benefits.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2022
            and (lambda egfr: egfr is not None and egfr <= 60)(_get_egfr(cm))
        ),
        "any_of_cats": ['GLP-1RA'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'KDIGO CKD & Diabetes-CKD Guidelines 2022, Practice Point 4.3',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1073',
        "description": 'For patients with T2DM and CKD, SGLT2i is recommended as first-line glucose-lowering therapy, regardless of glycemic control, to prevent CKD progression and cardiovascular events.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2022 and _get_dm_comp(cm, "nephropathy")),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'KDIGO CKD & Diabetes-CKD Guidelines 2022, Practice Point 1.4.2',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1074',
        "description": 'For patients with T2D, CKD, and eGFR ≥ 30 ml/min/1.73m², recommend metformin',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2020 and _get_dm_comp(cm, "nephropathy") and (lambda egfr: egfr is not None and egfr >= 30)(_get_egfr(cm))),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'KDIGO CKD & Diabetes-CKD Guidelines 2020, Recommendation 1',
        "domain":      'glycemic_first_line',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1075',
        "description": 'For T2DM patients with CKD who cannot use SGLT2i (e.g., eGFR <30 or intolerance), recommend GLP-1 RA, particularly those with ASCVD or high risk',
        "condition":   lambda cm, pd, yr: (_has_ckd_g3plus(cm)
            and 
            _get_dm_type(cm) == "T2DM"
            and yr >= 2020
            and (lambda egfr: egfr is not None and egfr < 30)(_get_egfr(cm))
        ),
        "any_of_cats": ['GLP-1RA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'KDIGO CKD & Diabetes-CKD Guidelines 2020, Values and preferences section on GLP-1 RA',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1076',
        "description": 'Metformin dose reduction may be considered when eGFR 45-59 ml/min per 1.73 m² and conditions predisposing to hypoperfusion/hypoxemia are present',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2022 and (lambda egfr: egfr is not None and 45 <= egfr and egfr <= 59)(_get_egfr(cm))),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'KDIGO 2022 CKD & Diabetes-CKD Guidelines, Practice Point 4.1.3',
        "domain":      'glycemic_first_line',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1077',
        "description": 'In T2D and CKD with eGFR < 15 or dialysis, DPP4i, insulin, and TZD are more suitable',
        "condition":   lambda cm, pd, yr: (_has_ckd_g3plus(cm)
            and _get_dm_type(cm) == "T2DM" and yr >= 2020 and ((lambda egfr: egfr is not None and egfr < 15)(_get_egfr(cm)) or _has_esrd_dialysis(pd))),
        "any_of_cats": ['DPP-4i', 'Insulins_Basal', 'Thiazolidinediones'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'KDIGO CKD & Diabetes-CKD Guidelines 2020 Chapter 4 Figure 20',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1078',
        "description": 'Prioritize metformin as initial antiglycemic medication in eligible patients with T2D and CKD',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2020 and _has_ckd_g3plus(cm)),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'KDIGO CKD & Diabetes-CKD Guidelines 2020 Chapter 3',
        "domain":      'glycemic_first_line',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1079',
        "description": 'Prefer glucose-lowering agents with lower risk of hypoglycemia for T2D and CKD patients not doing daily monitoring',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2022 and _has_ckd_g3plus(cm)),
        "any_of_cats": ['Biguanides', 'SGLT2i', 'GLP-1RA', 'DPP-4i'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'KDIGO CKD & Diabetes-CKD Guidelines 2022 | Practice Point 2.1.5',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'DM-02',
        "description": 'In T2DM with MASH or high risk of liver fibrosis, Pioglitazone or a dual GIP and GLP-1 RA can be considered.',
        "condition":   lambda cm, pd, yr: (
    _get_dm_type(cm) == "T2DM"
    and yr >= 2024
    and (cm.get("has_mash", False) or cm.get("has_nash", False) or cm.get("has_liver_fibrosis_risk", False))
),
        "any_of_cats": ['Thiazolidinediones', 'GIP_GLP1_RA'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2026 Section 9 (9.13a)',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'judge:empty_any_of_cats',
    },
    {
        "id":          'DM-03',
        "description": 'In T2DM patients needing weight loss, glucose-lowering medications that promote weight loss (Tirzepatide, Semaglutide) should be prioritized.',
        "condition":   lambda cm, pd, yr: (
    _get_dm_type(cm) == "T2DM"
    and yr >= 2022
    and (cm.get("bmi", 0) >= 30 or (cm.get("bmi", 0) >= 27 and cm.get("has_weight_related_complication", False)))
),
        "any_of_cats": ['GIP_GLP1_RA', 'GLP-1RA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2026 Section 8/9 (Weight Management)',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'judge:empty_any_of_cats',
    },
    {
        "id":          'DM-04',
        "description": 'In T2DM, initiate insulin therapy if A1C > 10% or symptomatic hyperglycemia/catabolism is present.',
        "condition":   lambda cm, pd, yr: (
    _get_dm_type(cm) == "T2DM"
    and (cm.get("hba1c", 0) > 10 or cm.get("has_symptomatic_hyperglycemia", False) or cm.get("has_catabolism", False))
),
        "any_of_cats": ['Insulins_Basal', 'Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2024/2026 - Section 9 (Pharmacologic Approaches)',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'judge:empty_any_of_cats',
    },
    {
        "id":          'DM-05',
        "description": 'In T2DM with severe hyperglycemia (A1C > 10%), GLP-1 RA or Tirzepatide are effective alternatives to insulin.',
        "condition":   lambda cm, pd, yr: (
    _get_dm_type(cm) == "T2DM"
    and yr >= 2023
    and cm.get("hba1c", 0) > 10
),
        "any_of_cats": ['GLP-1RA', 'GIP_GLP1_RA'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2026 - Section 9 (Combination Therapy)',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'judge:empty_any_of_cats',
    },
    {
        "id":          'DM-06',
        "description": 'In T2DM adults with A1C >= 10% or significant hyperglycemia, consider initiating insulin therapy.',
        "condition":   lambda cm, pd, yr: (
    _get_dm_type(cm) == "T2DM"
    and yr >= 2016
    and (cm.get("hba1c", 0) >= 10 or cm.get("fasting_glucose_mg_dl", 0) >= 300 or cm.get("random_glucose_mg_dl", 0) >= 300)
),
        "any_of_cats": ['Insulins_Basal', 'Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2016, Section 7 - Combination Therapy',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'judge:empty_any_of_cats',
    },
    {
        "id":          'DM-08',
        "description": 'In T2DM, consider insulin as the first injectable therapy if A1C >10% or blood glucose is very high.',
        "condition":   lambda cm, pd, yr: _get_dm_type(cm) == "T2DM" and (cm.get('hba1c', 0) > 10 or cm.get('fasting_glucose', 0) >= 300),
        "any_of_cats": ['Insulins_Basal', 'Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2024, Section 9 (Figure 9.4 & Text)',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'judge:empty_any_of_cats',
    },
    {
        "id":          'DM-09',
        "description": 'In T2DM, recommend GLP-1 RA or dual GIP/GLP-1 RA before initiating insulin therapy.',
        "condition":   lambda cm, pd, yr: _get_dm_type(cm) == "T2DM" and yr >= 2016 and not (cm.get('hba1c', 0) > 10 or cm.get('fasting_glucose', 0) >= 300),
        "any_of_cats": ['GLP-1RA', 'GIP_GLP1_RA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2024, Section 9 (Figure 9.4)',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'judge:empty_any_of_cats',
    },
    {
        "id":          'DM-10',
        "description": 'If basal insulin titrated but A1C remains above target, consider adding a GLP-1 receptor agonist or mealtime insulin.',
        "condition":   lambda cm, pd, yr: _get_dm_type(cm) == "T2DM" and yr >= 2016 and "Insulins_Basal" in pd.get('current_med_classes', []),
        "any_of_cats": ['GLP-1RA', 'Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2016, Section 7. Basal Insulin',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'judge:empty_any_of_cats',
    },
    {
        "id":          'DM-11',
        "description": 'In patients with history of hypoglycemia, insulin analogs are preferred over NPH/Regular to reduce hypoglycemia risk.',
        "condition":   lambda cm, pd, yr: _get_dm_type(cm) == "T2DM" and yr >= 2016 and cm.get('hypo_history', False),
        "any_of_cats": ['Insulins_Basal', 'Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2016/2020, Section 7. Insulin Therapy',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'judge:empty_any_of_cats',
    },
    {
        "id":          'DM-14',
        "description": 'For T1DM, insulin analogs (or inhaled insulin) are preferred over injectable human insulins to minimize hypoglycemia risk.',
        "condition":   lambda cm, pd, yr: _get_dm_type(cm) == "T1DM",
        "any_of_cats": ['Insulins_Basal', 'Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2026 Section 9 Recommendation 9.2',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'judge:empty_any_of_cats',
    },
    {
        "id":          'DM-15',
        "description": 'Initiate insulin therapy for patients who present with blood glucose levels >= 300 mg/dL or A1C > 10% or symptoms of catabolism.',
        "condition":   lambda cm, pd, yr: _get_dm_type(cm) == "T2DM" and (cm.get('hba1c', 0) > 10 or cm.get('fasting_glucose', 0) >= 300 or cm.get('catabolism_symptoms', False)),
        "any_of_cats": ['Insulins_Basal', 'Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2020 Section 9 Initial Therapy',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'judge:empty_any_of_cats',
    },
    {
        "id":          'DM-16',
        "description": 'A DPP-4 inhibitor can be considered for mild hyperglycemia in the postoperative setting.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2026
        ),
        "any_of_cats": ['DPP-4i'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2026, Inpatient Management (Chunk 1)',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'judge:empty_any_of_cats',
    },
    {
        "id":          'DM-17',
        "description": 'In adults with PTDM or preexisting type 2 diabetes, a GLP-1 RA can be considered for cardiometabolic benefits.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2026
        ),
        "any_of_cats": ['GLP-1RA'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2026, Recommendation 9.38b',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'judge:empty_any_of_cats',
    },
    {
        "id":          'DM-18',
        "description": 'Consider adding metformin to insulin therapy in overweight/obese patients with poorly controlled type 1 diabetes.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T1DM"
            and yr >= 2016
        ),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2016, Section 5 - Pharmacologic Therapy for Type 1 Diabetes',
        "domain":      'glycemic_first_line',
        "polarity":    'RECOMMEND',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'judge:empty_any_of_cats',
    },
    {
        "id":          'DM-19',
        "description": 'In T2DM patients requiring bolus insulin, rapid-acting insulin analogs are preferred.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2016
        ),
        "any_of_cats": ['Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2016, Pharmacologic Approaches to Glycemic Treatment',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'judge:empty_any_of_cats',
    },
    {
        "id":          'DM-20',
        "description": 'In T1DM, insulin analogs are preferred over human insulins to reduce hypoglycemia risk.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T1DM"
            and yr >= 2020
        ),
        "any_of_cats": ['Insulins_Basal', 'Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2020, Pharmacologic Approaches to Glycemic Treatment',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'judge:empty_any_of_cats',
    },
    {
        "id":          'DM-21',
        "description": 'Insulin is the preferred agent for hyperglycemia management during glucocorticoid therapy.',
        "condition":   lambda cm, pd, yr: _get_dm_type(cm) == "T2DM",
        "any_of_cats": ['Insulins_Basal', 'Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2026 - Therapeutic Strategies for Individuals Receiving Cancer Treatment',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'judge:empty_any_of_cats',
    },
    {
        "id":          'DM-23',
        "description": 'Insulin therapy is the cornerstone of management for hyperglycemia resulting from immune checkpoint inhibitors (ICIs).',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2018
        ),
        "any_of_cats": ['Insulins_Basal', 'Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2026 Section 4 (Therapeutic Strategies for Cancer)',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'judge:empty_any_of_cats',
    },
    {
        "id":          'DM-24',
        "description": 'In T2D and CKD, recommend long-acting GLP-1 RA if glycemic targets not achieved with metformin and SGLT2i, or if unable to use them.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2022 and _has_ckd_with_proteinuria(cm)),
        "any_of_cats": ['GLP-1RA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'KDIGO 2024 (citing KDIGO 2022) | Recommendation 3.9.1',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'judge:empty_any_of_cats',
    },
    {
        "id":          'DM-26',
        "description": 'In T2DM patients with eGFR > 15 mL/min/1.73m², a GLP-1 RA is recommended to achieve adequate glycaemic control.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2017 and (lambda egfr: egfr is not None and egfr > 15)(_get_egfr(cm))),
        "any_of_cats": ['GLP-1RA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 | Section 9 Chronic kidney disease and diabetes',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'judge:empty_any_of_cats',
    },
    {
        "id":          'DM-27',
        "description": 'In T2DM patients with heart failure, first-line treatment should include SGLT2 inhibitors.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2019
            and _get_dm_comp(cm, "heart_failure")
        ),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 (2019 update) | Key messages',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'judge:empty_any_of_cats',
    },
    {
        "id":          'DM-29',
        "description": 'In T2DM patients with overweight or obesity, glucose-lowering medications with effects on weight loss (e.g. GLP-1 RAs) should be considered.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2023 and pd.get('bmi', 0) >= 25),
        "any_of_cats": ['GLP-1RA'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ESC CVD in Diabetes 2023 | 5.1.1 Weight reduction',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'judge:empty_any_of_cats',
    },
    {
        "id":          'DM-31',
        "description": 'In T2DM with prevalent ASCVD, it is reasonable to consider pioglitazone to mitigate ASCVD risk, provided no heart failure.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2005
            and _get_dm_comp(cm, "cardiovascular")
            and not _get_dm_comp(cm, "heart_failure")
        ),
        "any_of_cats": ['Thiazolidinediones'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ESC CVD in Diabetes 2023 | 5.3.1.3. Pioglitazone',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'judge:empty_any_of_cats',
    },
    {
        "id":          'DM-34',
        "description": 'In T2DM patients with heart failure, Metformin is suggested to be safe if eGFR > 30 mL/min/1.73m^2 and kidney function is stable.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2016
            and _get_dm_comp(cm, "heart_failure")
            and (lambda egfr: egfr is not None and egfr > 30)(_get_egfr(cm))

        ),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ESC CVD in Diabetes 2023 | 7.5.5. Metformin',
        "domain":      'glycemic_first_line',
        "polarity":    'RECOMMEND',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'judge:empty_any_of_cats',
    },
    {
        "id":          'DM-35',
        "description": 'For T2DM adults with additional ASCVD risk factors requiring additional glucose-lowering therapy, SGLT2i or GLP-1RA may be reasonable.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2016
        ),
        "any_of_cats": ['SGLT2i', 'GLP-1RA'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ACC-AHA Cholesterol / Cardiovascular 2019 | Recommendations for Adults With Type 2 Diabetes Mellitus',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'judge:empty_any_of_cats',
    },
    {
        "id":          'DM-36',
        "description": 'Selected DPP4 inhibitors can be used with eGFR < 30 mL/min/1.73m² and provide a safe and effective option.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2006
            and (lambda egfr: egfr is not None and egfr < 30)(_get_egfr(cm))
        ),
        "any_of_cats": ['DPP-4i'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Consensus Reports & Supplements 2022 | Glycemic Management in Advanced CKD',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'judge:empty_any_of_cats',
    },
    {
        "id":          'DM-39',
        "description": 'In T2D with established ASCVD, initiate GLP-1 RA with proven CV benefit as first-line therapy.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2017
            and _get_dm_comp(cm, "cardiovascular")
        ),
        "any_of_cats": ['GLP-1RA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      '',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'judge:empty_any_of_cats',
    },
    {
        "id":          'DM-40',
        "description": 'In persons with T2D and HF, SGLT2i should be initiated as first-line therapy.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2020
            and _get_dm_comp(cm, "heart_failure")
        ),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      '',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'judge:empty_any_of_cats',
    },
    {
        "id":          'DM-41',
        "description": 'In patients with overweight/obesity and the additional goal of weight loss, dual GIP/GLP-1 RA, GLP-1 RA, or SGLT2i are preferred options.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2023
        ),
        "any_of_cats": ['GIP_GLP1_RA', 'GLP-1RA', 'SGLT2i'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      '',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'judge:empty_any_of_cats',
    },
    {
        "id":          'DM-42',
        "description": 'Persons with a history of hypoglycemia... should preferentially be initiated with an agent associated with low risk for hypoglycemia.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2017 and (pd.get("history_hypoglycemia") or pd.get("high_hypo_risk"))),
        "any_of_cats": ['Biguanides', 'DPP-4i', 'SGLT2i', 'GLP-1RA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      '',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'judge:empty_any_of_cats',
    },
    {
        "id":          'DM-43',
        "description": 'For T2DM with severe hyperglycemia and symptoms, basal plus prandial insulin is recommended.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2023 and (cm.get("hba1c", 0) > 10.0 or cm.get("bg_random", 0) > 300 or pd.get("symptomatic_hyperglycemia"))),
        "any_of_cats": ['Insulins_Basal', 'Insulins_Rapid'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      '',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'judge:empty_any_of_cats',
    },
    {
        "id":          'DM-44',
        "description": 'When initiating basal insulin in T2DM, analogs are preferred over NPH to reduce hypoglycemia.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2023 and pd.get("initiating_basal_insulin")),
        "any_of_cats": ['Insulins_Basal'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      '',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'judge:empty_any_of_cats',
    },
    {
        "id":          'DM-45',
        "description": 'In T2DM, if A1C > 10% or symptomatic hyperglycemia (BG > 300 mg/dL), basal insulin should be initiated.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2000 and (cm.get("hba1c", 0) > 10.0 or cm.get("bg_fasting", 0) > 300)),
        "any_of_cats": ['Insulins_Basal'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      '',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'judge:empty_any_of_cats',
    },
    {
        "id":          'DM-12',
        "description": 'In T2DM, consider GLP-1 RA or dual GIP/GLP-1 RA prior to prandial insulin to minimize hypoglycemia and weight gain.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2017
        ),
        "any_of_cats": ['GLP-1RA', 'GIP_GLP1_RA'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2024 - Section 9.4 Prandial Insulin',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'judge:empty_any_of_cats',
    },
    {
        "id":          'DM-33',
        "description": 'In patients with HF and eGFR > 30 mL/min, Metformin is safe and results in lower risk of death and HF hospitalization compared with insulin and sulfonylureas.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2019 and _get_dm_comp(cm, "heart_failure") and (lambda egfr: egfr is not None and egfr > 30)(_get_egfr(cm))),
        "any_of_cats": ['Biguanides'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ESC CVD in Diabetes 2023 2019 | 8.5.1 Metformin',
        "domain":      'glycemic_first_line',
        "polarity":    'RECOMMEND',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'judge:empty_any_of_cats',
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# ██████████████████  HARDCODE INTERFACE  ██████████████████████████████████████
#
# COMPLICATION_DRUG_GRADES 是并发症管理规则的唯一硬编码入口。
#
# 规则来源:指南文本 → guideline_rule_extractor.py 提取草稿
#           → 临床专家审核确认 → 硬编码到本列表
# 严禁在运行时动态添加(防止循环验证)。
#
# 修改方式:
#   1. 运行 guideline_rule_extractor.py → 生成 cpr_rules_draft.md
#   2. 临床专家审核填写确认意见
#   3. 在列表末尾「待添加区」追加新规则
#   4. grade 仅取 1.0 (Grade A) 或 0.5 (Grade B)
#   5. condition 仅使用以下辅助函数:
#        _get_dm_type / _get_dm_comp / _has_ckd_g3plus
#        _has_ckd_with_proteinuria / _has_albuminuria / _get_egfr / _get_age
#
# 当前规则 CPR-01~CPR-05:已通过临床专家审核
# 待添加    CPR-06+    :由 guideline_rule_extractor.py 生成草稿,待审核
# ══════════════════════════════════════════════════════════════════════════════

COMPLICATION_DRUG_GRADES: List[Dict] = [
    {
        "id":          'FINAL-0072',
        "description": 'In patients with a prior MI, β-blockers should be continued for at least 2 years after the event',
        "condition":   lambda cm, pd, yr: (yr >= 2010 and _get_dm_type(cm) in ("T1DM", "T2DM") and _has_prior_mi(pd)),
        "any_of_cats": ['Hypertension_BetaBlockers'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2010 - CVD Risk Management',
        "domain":      'bp_other',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0074',
        "description": 'For patients with diabetes and a history of CVD, aspirin (75–162 mg/day) is recommended as secondary prevention; for those with aspirin allergy, clopidogrel (75 mg/day) should be used.',
        "condition":   lambda cm, pd, yr: (yr >= 2011 and _get_dm_type(cm) in ("T1DM", "T2DM") and _get_dm_comp(cm, "cardiovascular")),
        "any_of_cats": ['Antiplatelet'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care Chunk 4 2011, Recommendation',
        "domain":      'ascvd_antiplatelet',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0075',
        "description": 'In nonpregnant patients with micro- or macroalbuminuria, either ACE inhibitors or ARBs should be used',
        "condition":   lambda cm, pd, yr: (yr >= 2011 and _get_dm_type(cm) in ("T1DM", "T2DM") and _has_ckd_with_proteinuria(cm) and not _has_pregnancy(pd)),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2011 - Treatment of Diabetic Nephropathy',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0076',
        "description": 'Statin therapy recommended for T2DM patients >40 years without overt CVD',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2006
            and not _get_dm_comp(cm, "cardiovascular")
            and (_get_age(cm, pd) or 0) >= 40
        ),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2006 - Treatment recommendations and goals',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0077',
        "description": 'All diabetic patients with overt CVD should be treated with a statin',
        "condition":   lambda cm, pd, yr: (yr >= 2006 and _get_dm_type(cm) in ("T1DM", "T2DM") and _get_dm_comp(cm, "cardiovascular")),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2006 - Treatment recommendations and goals',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0078',
        "description": 'ACE inhibitor should be considered in patients >55 with CVD or another risk factor',
        "condition":   lambda cm, pd, yr: (yr >= 2006 and _get_dm_type(cm) in ("T1DM", "T2DM") and _get_dm_comp(cm, "cardiovascular") and (_get_age(cm, pd) or 0) > 55),
        "any_of_cats": ['Hypertension_ACEi'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2006 - Recommendations',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0080',
        "description": 'For diabetic patients aged 40 years or older, statin therapy is recommended for primary prevention of cardiovascular events.',
        "condition":   lambda cm, pd, yr: (yr >= 2006 and _get_dm_type(cm) in ("T1DM", "T2DM") and not _get_dm_comp(cm, "cardiovascular") and (_get_age(cm, pd) or 0) >= 40),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Medical Care 2006 - Treatment recommendations and goals',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0081',
        "description": 'For diabetic patients with albuminuria (micro- or macroalbuminuria), ACE inhibitors or ARBs should be used to delay progression of nephropathy.',
        "condition":   lambda cm, pd, yr: (yr >= 2006 and _get_dm_type(cm) in ("T1DM", "T2DM") and _has_albuminuria(cm)),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Medical Care 2006 - Treatment | Level=A',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0082',
        "description": 'For diabetic patients with established CVD, aspirin therapy is recommended for secondary prevention of cardiovascular events.',
        "condition":   lambda cm, pd, yr: (yr >= 2008 and _get_dm_type(cm) in ("T1DM", "T2DM") and _get_dm_comp(cm, "cardiovascular")),
        "any_of_cats": ['Antiplatelet'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Medical Care 2008 - Standards of Medical Care | Level=A',
        "domain":      'ascvd_antiplatelet',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0087',
        "description": 'Prescribe statin therapy for diabetic patients with overt cardiovascular disease',
        "condition":   lambda cm, pd, yr: (yr >= 2012 and _get_dm_type(cm) in ("T1DM", "T2DM") and _get_dm_comp(cm, "cardiovascular")),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2012, Treatment recommendations and goals',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0089',
        "description": 'ACE inhibitor or ARB therapy should be used in diabetic patients with hypertension',
        "condition":   lambda cm, pd, yr: (yr >= 2013 and _get_dm_type(cm) in ("T1DM", "T2DM") and cm.get("has_hypertension")),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2013, Treatment',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0090',
        "description": 'Statin therapy should be added for diabetic patients with overt CVD',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2008
            and _get_dm_comp(cm, "cardiovascular")
        ),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2008, Lipid Management',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0092',
        "description": 'Use aspirin (75-162 mg/day) as secondary prevention in diabetic patients with established CVD.',
        "condition":   lambda cm, pd, yr: (yr >= 2009 and _get_dm_type(cm) in ("T1DM", "T2DM") and _get_dm_comp(cm, "cardiovascular")),
        "any_of_cats": ['Antiplatelet'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2009, Aspirin for secondary prevention',
        "domain":      'ascvd_antiplatelet',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0093',
        "description": 'Use aspirin therapy (75–162 mg/day) as a secondary prevention strategy in those with diabetes with a history of CVD (Grade A)',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2010
            and _get_dm_comp(cm, "cardiovascular")
        ),
        "any_of_cats": ['Antiplatelet'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2010, Recommendations',
        "domain":      'ascvd_antiplatelet',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0095',
        "description": 'For patients with diabetes and hypertension, pharmacologic therapy should include ACE inhibitor or ARB as first-line',
        "condition":   lambda cm, pd, yr: (yr >= 2012 and _get_dm_type(cm) in ("T1DM", "T2DM") and cm.get("has_hypertension")),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2012, Treatment section',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0097',
        "description": 'All patients with diabetes and hypertension should receive a regimen including an ACE inhibitor or ARB',
        "condition":   lambda cm, pd, yr: (yr >= 2008 and _get_dm_type(cm) in ("T1DM", "T2DM") and cm.get("has_hypertension")),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2008, Hypertension section',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0098',
        "description": 'Children with T1DM over 10 years with elevated LDL ( >160 mg/dL or >130 mg/dL with CVD risk factors) should receive a statin after lifestyle therapy',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T1DM" and yr >= 2008 and (_get_age(cm, pd) or 0) >= 10 and ((lambda l: l is not None and l > 160)((cm.get("lipids") or {}).get("ldl")) or ((lambda l: l is not None and l > 130)((cm.get("lipids") or {}).get("ldl")) and _get_dm_comp(cm, "cardiovascular")))),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2008, Treatment section (Level E)',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0099',
        "description": 'For children with type 1 diabetes aged >10 years with persistent LDL elevation despite lifestyle therapy, statins should be considered.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T1DM" and yr >= 2009 and (_get_age(cm, pd) or 0) >= 10 and (lambda l: l is not None and l > 130)((cm.get("lipids") or {}).get("ldl"))),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2009, Retinopathy section, Level E',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0100',
        "description": 'For patients with type 1 diabetes, hypertension, and any degree of albuminuria, ACE inhibitors should be used to delay nephropathy progression',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T1DM" and yr >= 2006 and cm.get("has_hypertension") and _has_ckd_with_proteinuria(cm)),
        "any_of_cats": ['Hypertension_ACEi'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2006, Treatment (hypertension/renal)',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0101',
        "description": 'For patients with type 2 diabetes, hypertension, and microalbuminuria, ACE inhibitors or ARBs should be used to delay progression to macroalbuminuria',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2006 and cm.get("has_hypertension") and _has_albuminuria(cm)),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2006, Treatment (hypertension/renal)',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0102',
        "description": 'For patients with type 2 diabetes, hypertension, macroalbuminuria, and renal insufficiency, ARBs should be used to delay nephropathy progression',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2006 and cm.get("has_hypertension") and (lambda u: u is not None and u >= 300)((cm.get("kidney") or {}).get("uacr")) and (lambda egfr: egfr is not None and egfr <= 60)(_get_egfr(cm))),
        "any_of_cats": ['Hypertension_ARB'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2006, Treatment (hypertension/renal)',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0103',
        "description": 'Combination therapy with clopidogrel in addition to aspirin should be used in patients with severe and progressive CVD.',
        "condition":   lambda cm, pd, yr: (yr >= 2006 and _get_dm_type(cm) in ("T1DM", "T2DM") and _get_dm_comp(cm, "cardiovascular")),
        "any_of_cats": ['Antiplatelet'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Medical Care 2006, Antiplatelet therapy',
        "domain":      'ascvd_antiplatelet',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0105',
        "description": 'Statins should be prescribed for diabetic patients with ASCVD',
        "condition":   lambda cm, pd, yr: (yr >= 2008 and _get_dm_type(cm) in ("T1DM", "T2DM") and _get_dm_comp(cm, "cardiovascular")),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2008, Dyslipidemia treatment and target lipid levels',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0106',
        "description": 'For patients with type 2 diabetes and established cardiovascular disease, prescribe statin therapy at discharge for LDL cholesterol lowering.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2009
            and _get_dm_comp(cm, "cardiovascular")
        ),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2009 - Dyslipidemia treatment and target lipid levels',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0107',
        "description": 'For all diabetes patients, initiate statin therapy to achieve LDL cholesterol <100 mg/dl (2.6 mmol/l) for primary prevention.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2010),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2010 - Treatment recommendations and goals',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0108',
        "description": 'For diabetic patients with microalbuminuria (UACR ≥30 mg/g), use ACE inhibitor or ARB to slow nephropathy progression',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) in ("T1DM", "T2DM") and yr >= 2010 and _has_albuminuria(cm)),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2010 - Treatment (Chunk 3)',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0109',
        "description": 'For children with diabetes aged >10 years, add a statin if LDL-C >160 mg/dL, or LDL-C >130 mg/dL with one or more CVD risk factors',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) in ("T1DM", "T2DM") and yr >= 2011 and (_get_age(cm, pd) or 0) >= 10 and (lambda l: l is not None and (l > 160 or (l > 130 and cm.get("has_hypertension"))))((cm.get("lipids") or {}).get("ldl"))),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2011 - Treatment of Dyslipidemia in Children',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0110',
        "description": 'In children aged ≥10 years with type 1 diabetes, add statin if LDL cholesterol >160 mg/dL, or >130 mg/dL with one or more CVD risk factors after MNT and lifestyle changes, at discharge.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T1DM" and yr >= 2012 and (_get_age(cm, pd) or 0) >= 10 and (lambda l: l is not None and (l > 160 or (l > 130 and cm.get("has_hypertension"))))((cm.get("lipids") or {}).get("ldl"))),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2012, Section: Treatment (Lipid management), Level=E',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0111',
        "description": 'Glucagon should be prescribed for all individuals at significant risk of severe hypoglycemia',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) in ("T1DM", "T2DM") and yr >= 2010 and _has_hypoglycemia_hx(pd)),
        "any_of_cats": ['Hypoglycemia_Rescue'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2010, Hypoglycemia section',
        "domain":      'hypoglycemia_rescue',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0112',
        "description": 'Prescribe glucagon for all individuals at significant risk of severe hypoglycemia',
        "condition":   lambda cm, pd, yr: (yr >= 2011 and _get_dm_type(cm) in ("T1DM", "T2DM") and (_has_hypoglycemia_hx(pd) or cm.get("hypo_history") or (lambda x: x is not None and x > 0)((cm.get("glucose") or {}).get("severe_hypoglycemia_events")))),
        "any_of_cats": ['Hypoglycemia_Rescue'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Medical Care 2011, Hypoglycemia section',
        "domain":      'hypoglycemia_rescue',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0113',
        "description": 'In older adults with diabetes (age ≥65) and sufficient life expectancy, statin therapy is recommended for lipid management to prevent cardiovascular events.',
        "condition":   lambda cm, pd, yr: (yr >= 2008 and _get_dm_type(cm) in ("T1DM", "T2DM") and (_get_age(cm, pd) or 0) >= 65),
        "any_of_cats": ['Lipid_Statins', 'Lipid_NonStatins'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2008, Chapter: Older Adults, Recommendation (E)',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0115',
        "description": 'Older adults (≥65 yr) with diabetes and cardiovascular disease should be considered for statin and aspirin therapy',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) in ("T1DM", "T2DM") and yr >= 2009 and _get_dm_comp(cm, "cardiovascular") and (_get_age(cm, pd) or 0) >= 65),
        "any_of_cats": ['Lipid_Statins', 'Antiplatelet'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2009 – Older Adults Recommendations (Level E)',
        "domain":      'ascvd_antiplatelet',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0117',
        "description": 'Antihypertensive therapy for older adults with diabetes',
        "condition":   lambda cm, pd, yr: (yr >= 2011 and _get_dm_type(cm) in ("T1DM", "T2DM") and (_get_age(cm, pd) or 0) >= 65 and bool(pd.get("has_hypertension"))),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB', 'Hypertension_CCB', 'Hypertension_Diuretics', 'Hypertension_BetaBlockers'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Medical Care 2011, Older Adults recommendations',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0120',
        "description": 'Statins should not be used in children with type 1 diabetes under the age of 10.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T1DM" and yr >= 2013 and (_get_age(cm, pd) or 99) < 10),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": ['Atorvastatin', 'Rosuvastatin', 'Simvastatin', 'Pravastatin', 'Lovastatin', 'Fluvastatin', 'Pitavastatin'],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2013, Section VIII, Children',
        "domain":      'lipid_management',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0121',
        "description": 'Initiate ACE inhibitor as first-line antihypertensive therapy in pediatric diabetes patients with hypertension',
        "condition":   lambda cm, pd, yr: (yr >= 2006 and _get_dm_type(cm) in ("T1DM", "T2DM") and cm.get("has_hypertension") is True and 0 <= (_get_age(cm, pd) or 0) < 18),
        "any_of_cats": ['Hypertension_ACEi'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2006, Recommendations (Level E)',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0122',
        "description": 'For T1DM children aged ≥10 years with diabetes duration ≥5 years and confirmed persistent microalbuminuria, initiate ACE inhibitor titrated to normalization of albumin excretion.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T1DM" and yr >= 2011 and _has_ckd_with_proteinuria(cm) and 10 <= (_get_age(cm, pd) or 0) < 18),
        "any_of_cats": ['Hypertension_ACEi'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      "ADA Standards of Medical Care 2011 - Section 'Nephropathy' (Chunk 4)",
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0123',
        "description": 'For children with T1DM aged ≥10 years with diabetes duration ≥5 years and confirmed persistent microalbuminuria, an ACE inhibitor should be initiated and titrated to normalize microalbumin excretion.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T1DM" and yr >= 2008 and _has_ckd_with_proteinuria(cm) and 10 <= (_get_age(cm, pd) or 0) < 18),
        "any_of_cats": ['Hypertension_ACEi'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2008, Microalbuminuria',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0124',
        "description": 'In children with hypertension (BP consistently ≥95th percentile for age/sex/height or consistently >130/80 mmHg), pharmacologic treatment should be initiated after 3-6 months of lifestyle intervention if BP target not reached, or immediately upon diagnosis; ACE inhibitors should be considered for initial therapy.',
        "condition":   lambda cm, pd, yr: (yr >= 2008 and _get_dm_type(cm) in ("T1DM", "T2DM") and cm.get("has_hypertension") is True and 0 <= (_get_age(cm, pd) or 0) < 18),
        "any_of_cats": ['Hypertension_ACEi'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2008, Hypertension',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0125',
        "description": 'For children/adolescents with diabetes and confirmed persistent microalbuminuria, initiate ACE inhibitor, titrate to normalize microalbumin excretion if possible.',
        "condition":   lambda cm, pd, yr: (yr >= 2009 and _get_dm_type(cm) in ("T1DM", "T2DM") and _has_ckd_with_proteinuria(cm) and 10 <= (_get_age(cm, pd) or 0) < 18),
        "any_of_cats": ['Hypertension_ACEi'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Medical Care in Diabetes - 2009, Microalbuminuria section',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0126',
        "description": 'For T1DM pediatric patients with confirmed persistent microalbuminuria (age≥10 years, diabetes duration≥5 years), treat with ACE inhibitor',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T1DM" and yr >= 2010 and _has_albuminuria(cm) and (_get_age(cm, pd) or 0) >= 10),
        "any_of_cats": ['Hypertension_ACEi'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2010, Microalbuminuria screening and treatment (Section ii. Hypertension)',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0127',
        "description": 'For diabetic patients with hypertension, especially those with CKD, prescribe ACEi or ARB for BP control, goal BP <130/80 mmHg.',
        "condition":   lambda cm, pd, yr: (_has_ckd_g3plus(cm)
            and yr >= 2003 and _get_dm_type(cm) in ("T1DM", "T2DM") and cm.get("has_hypertension") is True),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'JNC 7 (2003) - Classification of BP & Compelling Indications',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0128',
        "description": 'Tricyclic drugs are recommended for the management of painful diabetic peripheral neuropathy',
        "condition":   lambda cm, pd, yr: (yr >= 2006 and _get_dm_type(cm) in ("T1DM", "T2DM") and _get_dm_comp(cm, "neuropathy")),
        "any_of_cats": ['Neuropathy_TCA'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Medical Care 2006 - Neuropathy/Symptomatic treatments',
        "domain":      'neuropathy',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0130',
        "description": 'Duloxetine (SNRI) is recommended for the treatment of painful diabetic neuropathy',
        "condition":   lambda cm, pd, yr: (yr >= 2006 and _get_dm_type(cm) in ("T1DM", "T2DM") and _get_dm_comp(cm, "neuropathy")),
        "any_of_cats": ['Neuropathy_SNRI'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Medical Care 2006 - Neuropathy/Symptomatic treatments',
        "domain":      'neuropathy',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0131',
        "description": 'If eGFR < 30 ml/min/1.73m2, a loop diuretic rather than HCTZ or chlorthalidone should be prescribed.',
        "condition":   lambda cm, pd, yr: (yr >= 2012 and _get_dm_type(cm) in ("T1DM", "T2DM") and (lambda egfr: egfr is not None and egfr < 30)(_get_egfr(cm))),
        "any_of_cats": ['Hypertension_Diuretics'],
        "exclude_drugs": ['hydrochlorothiazide', 'chlorthalidone', 'thiazide'],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2012 - Hypertension',
        "domain":      'bp_other',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'P2-A: exclude 指向策展中有意移除的药(噻嗪/非DHP-CCB/gemfibrozil/保钾/跨类别),commission 永不触发,孤儿规则',
    },
    {
        "id":          'FINAL-0132',
        "description": 'Prescribe glucagon for all individuals at significant risk of severe hypoglycemia',
        "condition":   lambda cm, pd, yr: (yr >= 2013 and _get_dm_type(cm) in ("T1DM", "T2DM") and (_has_hypoglycemia_hx(pd) or cm.get("hypo_history") or (lambda x: x is not None and x > 0)((cm.get("glucose") or {}).get("severe_hypoglycemia_events")))),
        "any_of_cats": ['Hypoglycemia_Rescue'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2013 - K. Hypoglycemia Recommendations',
        "domain":      'hypoglycemia_rescue',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0133',
        "description": 'Prescribe glucagon for patients at significant risk of severe hypoglycemia',
        "condition":   lambda cm, pd, yr: (yr >= 2006 and _get_dm_type(cm) in ("T1DM", "T2DM") and (_has_hypoglycemia_hx(pd) or (lambda x: x is not None and x > 0)((cm.get("glucose") or {}).get("severe_hypoglycemia_events")))),
        "any_of_cats": ['Hypoglycemia_Rescue'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2006 - Hypoglycemia',
        "domain":      'hypoglycemia_rescue',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0134',
        "description": 'Prescribe glucagon for all individuals at significant risk of severe hypoglycemia',
        "condition":   lambda cm, pd, yr: (yr >= 2008 and _get_dm_type(cm) in ("T1DM", "T2DM") and (_has_hypoglycemia_hx(pd) or cm.get("hypo_history") or (lambda x: x is not None and x > 0)((cm.get("glucose") or {}).get("severe_hypoglycemia_events")))),
        "any_of_cats": ['Hypoglycemia_Rescue'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Medical Care 2008 - Hypoglycemia',
        "domain":      'hypoglycemia_rescue',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0135',
        "description": 'Prescribe glucagon for individuals at significant risk of severe hypoglycemia',
        "condition":   lambda cm, pd, yr: (yr >= 2010 and _get_dm_type(cm) in ("T1DM", "T2DM") and (_has_hypoglycemia_hx(pd) or (lambda x: x is not None and x > 0)((cm.get("glucose") or {}).get("severe_hypoglycemia_events")))),
        "any_of_cats": ['Hypoglycemia_Rescue'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2010 - Hypoglycemia Recommendations',
        "domain":      'hypoglycemia_rescue',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0136',
        "description": 'Prescribe glucagon for all individuals at significant risk of severe hypoglycemia',
        "condition":   lambda cm, pd, yr: (yr >= 2012 and _get_dm_type(cm) in ("T1DM", "T2DM") and (_has_hypoglycemia_hx(pd) or (lambda x: x is not None and x > 0)((cm.get("glucose") or {}).get("severe_hypoglycemia_events")))),
        "any_of_cats": ['Hypoglycemia_Rescue'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2012 K. Hypoglycemia Recommendations',
        "domain":      'hypoglycemia_rescue',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0137',
        "description": 'Glucose (15-20g) is the preferred treatment for the conscious individual with hypoglycemia',
        "condition":   lambda cm, pd, yr: (yr >= 2008 and _get_dm_type(cm) in ("T1DM", "T2DM") and _has_hypoglycemia_hx(pd)),
        "any_of_cats": ['Hypoglycemia_Rescue'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2008 - Hypoglycemia',
        "domain":      'hypoglycemia_rescue',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0138',
        "description": 'Patients at risk for severe hypoglycemia should have glucagon emergency kits available for rescue.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T1DM"
            and yr >= 2009
        ),
        "any_of_cats": ['Hypoglycemia_Rescue'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Medical Care 2009 - Diabetes in Correctional Institutions/Disaster Planning',
        "domain":      'hypoglycemia_rescue',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0139',
        "description": 'Statin initiation for adults >=50 years old with CKD and eGFR>=60 for primary prevention',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2013 and not _get_dm_comp(cm, "cardiovascular") and (lambda egfr: egfr is not None and 60 <= egfr)(_get_egfr(cm)) and (_get_age(cm, pd) or 0) >= 50 and _has_albuminuria(cm)),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Consensus Reports & Supplements 2022 | Lipid Management',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0142',
        "description": 'Prescribe ACE inhibitors or ARBs for patients with UACR ≥30 mg/g to manage diabetic kidney disease.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) in ("T1DM", "T2DM") and yr >= 2013 and _has_albuminuria(cm)),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2013 | Treatment | Level=A',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0144',
        "description": 'ACE inhibitors or ARBs should be prescribed for patients with albuminuria (UACR >30 mg/g) to improve renal and cardiovascular outcomes',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) in ("T1DM", "T2DM") and yr >= 2008 and _has_albuminuria(cm)),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2008 | Assessment of albuminuria status and renal function',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0145',
        "description": 'ACE inhibitors or ARBs should be prescribed for diabetic patients with UACR ≥30 mg/g to delay nephropathy progression',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2010 and (cm.get("kidney") or {}).get("uacr") is not None and (cm.get("kidney") or {}).get("uacr") >= 30),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2010, Treatment',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0198',
        "description": 'ACE inhibitors or ARBs should be used in nonpregnant diabetic patients with micro- or macroalbuminuria',
        "condition":   lambda cm, pd, yr: (yr >= 2001 and _get_dm_type(cm) in ('T1DM', 'T2DM') and not _has_pregnancy(pd) and _has_ckd_with_proteinuria(cm)),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2008, Treatment of Diabetic Nephropathy',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0199',
        "description": 'Statin therapy should be added to lifestyle therapy for diabetic patients with overt CVD regardless of baseline lipid levels.',
        "condition":   lambda cm, pd, yr: (yr >= 2001 and _get_dm_type(cm) in ('T1DM', 'T2DM') and _get_dm_comp(cm, 'cardiovascular')),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2013 - Treatment recommendations and goals (Lipid Management)',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0200',
        "description": 'Use aspirin therapy (75–162 mg/day) as a secondary prevention strategy in those with diabetes with a history of CVD.',
        "condition":   lambda cm, pd, yr: (yr >= 2001 and _get_dm_type(cm) in ('T1DM', 'T2DM') and _get_dm_comp(cm, 'cardiovascular')),
        "any_of_cats": ['Antiplatelet'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2014 - Antiplatelet Agents Recommendations',
        "domain":      'ascvd_antiplatelet',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0201',
        "description": 'Statin therapy should be considered for diabetic patients without CVD, aged ≥40 with one or more CVD risk factors',
        "condition":   lambda cm, pd, yr: (yr >= 2001 and _get_dm_type(cm) in ('T1DM', 'T2DM') and not _get_dm_comp(cm, 'cardiovascular') and (_get_age(cm, pd) or 0) >= 40),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2014, Treatment Recommendations and Goals, Level=A',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0203',
        "description": 'For diabetic patients with heart failure, use ACE inhibitor or ARB (can be combined) to reduce risk of cardiovascular death or heart failure hospitalization.',
        "condition":   lambda cm, pd, yr: (yr >= 2001 and _get_dm_type(cm) in ('T1DM', 'T2DM') and _get_dm_comp(cm, 'heart_failure')),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2006, Section VI (Treatment)',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0204',
        "description": 'For diabetic patients with albuminuria or renal insufficiency, use ACE inhibitor or ARB to slow progression of kidney disease.',
        "condition":   lambda cm, pd, yr: (yr >= 2001 and _get_dm_type(cm) in ('T1DM', 'T2DM') and (_has_ckd_g3plus(cm) or _has_albuminuria(cm))),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2006, Section VI (Treatment)',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0206',
        "description": 'Prescribe glucagon for all individuals taking insulin or at high risk for hypoglycemia at discharge; prefer formulations that do not require reconstitution.',
        "condition":   lambda cm, pd, yr: (yr >= 2001 and _get_dm_type(cm) in ('T1DM', 'T2DM') and (bool(cm.get('hypoglycemia_history') or cm.get('hypo_history')) or _get_dm_type(cm) == 'T1DM')),
        "any_of_cats": ['Hypoglycemia_Rescue'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2026, Section 9, Recommendation 9.28',
        "domain":      'hypoglycemia_rescue',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0208',
        "description": 'In patients with known CVD, aspirin should be used to reduce the risk of cardiovascular events.',
        "condition":   lambda cm, pd, yr: (yr >= 2001 and _get_dm_type(cm) in ('T1DM', 'T2DM') and _get_dm_comp(cm, 'cardiovascular')),
        "any_of_cats": ['Antiplatelet'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2011, Treatment',
        "domain":      'ascvd_antiplatelet',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0209',
        "description": 'In patients with known CVD, statin therapy should be used to reduce the risk of cardiovascular events.',
        "condition":   lambda cm, pd, yr: (yr >= 2001 and _get_dm_type(cm) in ("T1DM", "T2DM") and _get_dm_comp(cm, "cardiovascular")),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2011, Treatment',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0211',
        "description": 'For diabetic patients with eGFR<30 mL/min, use loop diuretics instead of thiazide diuretics for hypertension management',
        "condition":   lambda cm, pd, yr: (lambda egfr: egfr is not None and egfr < 30)(_get_egfr(cm)),
        "any_of_cats": ['Hypertension_Diuretics'],
        "exclude_drugs": ['hydrochlorothiazide', 'chlorthalidone'],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2012, Position Statement on Hypertension',
        "domain":      'bp_other',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'P2-A: exclude 指向策展中有意移除的药(噻嗪/非DHP-CCB/gemfibrozil/保钾/跨类别),commission 永不触发,孤儿规则',
    },
    {
        "id":          'FINAL-0213',
        "description": 'Prescribe glucagon for all individuals at significant risk of severe hypoglycemia, and instruct caregivers on its administration at discharge',
        "condition":   lambda cm, pd, yr: (yr >= 2000 and _get_dm_type(cm) in ("T1DM", "T2DM") and _has_hypoglycemia_hx(pd)),
        "any_of_cats": ['Hypoglycemia_Rescue'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2013, Section K. Hypoglycemia',
        "domain":      'hypoglycemia_rescue',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0214',
        "description": 'Statin therapy for secondary prevention in T2DM patients with established ASCVD',
        "condition":   lambda cm, pd, yr: (yr >= 2001 and _get_dm_type(cm) == "T2DM" and _get_dm_comp(cm, "cardiovascular")),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2013 - Dyslipidemia treatment',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0215',
        "description": 'For T1DM patients >10 years old with LDL >160 mg/dL or LDL >130 mg/dL plus at least one CVD risk factor, consider adding a statin.',
        "condition":   lambda cm, pd, yr: (yr >= 2001 and _get_dm_type(cm) == "T1DM" and (_get_age(cm, pd) or 0) >= 10 and ((lambda l: l is not None and l > 160)((cm.get("lipids") or {}).get("ldl")) or ((lambda l: l is not None and l > 130)((cm.get("lipids") or {}).get("ldl")) and cm.get("has_hypertension")))),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       0.25,
        "source":      'ADA Standards of Care 2013, Treatment section',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0218',
        "description": 'For diabetic patients with confirmed persistently elevated microalbuminuria, ACE inhibitor therapy should be initiated and titrated to normalization of microalbumin excretion if possible.',
        "condition":   lambda cm, pd, yr: (yr >= 2001 and _get_dm_type(cm) in ('T1DM', 'T2DM') and _has_ckd_with_proteinuria(cm)),
        "any_of_cats": ['Hypertension_ACEi'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2006 - Nephropathy Screening and Treatment',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0219',
        "description": 'For hospitalized diabetic patients with CKD, recommend ACEi or ARB to improve kidney outcomes',
        "condition":   lambda cm, pd, yr: (_has_ckd_g3plus(cm)
            and yr >= 2001 and _get_dm_type(cm) in ('T1DM', 'T2DM') and (lambda egfr: egfr is not None and egfr <= 60)(_get_egfr(cm))),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'JAMA 2014;311(5):507-520, Report From the Panel Members Appointed to the Eighth Joint, Recommendation 3',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0220',
        "description": 'For patients with diabetes aged 20-39 years with CKD (eGFR<60 or nephropathy) without ASCVD, consider moderate-intensity statin for primary prevention at discharge.',
        "condition":   lambda cm, pd, yr: (_has_ckd_g3plus(cm)
            and yr >= 2001 and _get_dm_type(cm) in ('T1DM', 'T2DM') and not _get_dm_comp(cm, 'cardiovascular') and (lambda egfr: egfr is not None and egfr <= 60)(_get_egfr(cm)) and (_get_age(cm, pd) or 0) >= 20 and (_get_age(cm, pd) or 0) <= 39),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Consensus Reports & Supplements 2022 Lipid Management, Level=A',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0221',
        "description": 'Patients with history of hypoglycemia should be discharged with glucagon or glucose rescue kit',
        "condition":   lambda cm, pd, yr: (yr >= 2001 and _get_dm_type(cm) in ('T1DM', 'T2DM') and _has_hypoglycemia_hx(pd)),
        "any_of_cats": ['Hypoglycemia_Rescue'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Consensus Reports & Supplements 2012, Recommendations Table 5',
        "domain":      'hypoglycemia_rescue',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0224',
        "description": 'Loop diuretic rather than HCTZ or chlorthalidone should be prescribed if eGFR < 30',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and (lambda egfr: egfr is not None and egfr <= 29)(_get_egfr(cm))
        ),
        "any_of_cats": ['Hypertension_Diuretics'],
        "exclude_drugs": ['hydrochlorothiazide', 'chlorthalidone'],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2013 - Treatment strategies',
        "domain":      'bp_other',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'P2-A: exclude 指向策展中有意移除的药(噻嗪/非DHP-CCB/gemfibrozil/保钾/跨类别),commission 永不触发,孤儿规则',
    },
    {
        "id":          'FINAL-0333',
        "description": 'For T2DM with HF or CKD, prefer SGLT2i with proven CV benefit.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2019 and (_get_dm_comp(cm, "heart_failure") or _has_ckd_g3plus(cm) or _has_albuminuria(cm))),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2019, Initial Therapy (Figure 9.1)',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0334',
        "description": 'For T2DM with established ASCVD or high risk of ASCVD, use SGLT2i or GLP-1 RA with proven CV benefit',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2016 and _get_dm_comp(cm, "cardiovascular")),
        "any_of_cats": ['SGLT2i', 'GLP-1RA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2025, Section 9.10',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0335',
        "description": 'In patients with type 2 diabetes and established ASCVD, antihyperglycemic therapy should include an agent proven to reduce MACE and CV mortality (empagliflozin or liraglutide) after metformin.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2016
            and _get_dm_comp(cm, "cardiovascular")
        ),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2018, Section 8',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0336',
        "description": 'In patients with type 2 diabetes and established ASCVD, consider adding a GLP-1 RA with proven CV benefit (liraglutide) after metformin.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2017
            and _get_dm_comp(cm, "cardiovascular")
        ),
        "any_of_cats": ['GLP-1RA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2018, Section 8',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0337',
        "description": 'In T2DM patients with ASCVD, heart failure, or CKD, SGLT2i or GLP-1RA with CV benefit should be used for cardiorenal protection independent of glycemic control.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2021 and (_get_dm_comp(cm, "cardiovascular") or _get_dm_comp(cm, "heart_failure") or _has_ckd_g3plus(cm))),
        "any_of_cats": ['SGLT2i', 'GLP-1RA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2021, Section 9, Recommendation 9.9',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0338',
        "description": 'In T2DM patients with established ASCVD, prioritize SGLT2i (e.g., empagliflozin) or GLP-1 RA (e.g., liraglutide) with proven cardiovascular benefit to reduce MACE.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2017
            and _get_dm_comp(cm, "cardiovascular")
        ),
        "any_of_cats": ['SGLT2i', 'GLP-1RA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2017, Section: Cardiovascular Outcome Trials (EMPA-REG OUTCOME, LEADER)',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0339',
        "description": 'In adults with T2DM and advanced CKD (eGFR<30), prefer GLP-1 RA for glycemic management and cardiovascular event reduction',
        "condition":   lambda cm, pd, yr: (_has_ckd_g3plus(cm)
            and _get_dm_type(cm) == "T2DM" and yr >= 2025 and (lambda egfr: egfr is not None and egfr < 30)(_get_egfr(cm))),
        "any_of_cats": ['GLP-1RA'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2025, Section 9, Recommendation 9.14',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0340',
        "description": 'Diabetic patients with clinical CVD or age >40 years with other CVD risk factors should receive statin therapy regardless of baseline lipid levels',
        "condition":   lambda cm, pd, yr: (yr >= 2014 and _get_dm_type(cm) in ("T1DM", "T2DM") and (_get_dm_comp(cm, "cardiovascular") or ((_get_age(cm, pd) or 0) >= 40 and cm.get("has_hypertension")))),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2014, Dyslipidemia Treatment section',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0341',
        "description": 'Children with type 1 diabetes aged >10 years with LDL >160 mg/dL or LDL >130 mg/dL plus one or more CVD risk factors may receive statin therapy, target LDL <100 mg/dL',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T1DM" and yr >= 2014 and (_get_age(cm, pd) or 0) >= 10 and ((lambda l: l is not None and l > 160)((cm.get("lipids") or {}).get("ldl")) or ((lambda l: l is not None and l > 130)((cm.get("lipids") or {}).get("ldl")) and cm.get("has_hypertension")))),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2014, Treatment (Child section)',
        "domain":      'lipid_management',
        "polarity":    'ALLOW',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0342',
        "description": 'For women of childbearing potential with diabetes, before conception, medications contraindicated in pregnancy (statins, ACEi, ARB, most noninsulin therapies) should be evaluated and avoided.',
        "condition":   lambda cm, pd, yr: yr >= 2014,
        "any_of_cats": ['Lipid_Statins', 'Hypertension_ACEi', 'Hypertension_ARB'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2014 - Preconception Counseling (Chunk 2)',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'audit:avoid_uncheckable',
    },
    {
        "id":          'FINAL-0344',
        "description": 'Diabetes with acute coronary syndrome: prescribe aspirin plus a P2Y12 inhibitor (clopidogrel/ticagrelor/prasugrel) for at least 1 year post discharge.',
        "condition":   lambda cm, pd, yr: (yr >= 2014 and _get_dm_type(cm) in ("T1DM", "T2DM") and _has_acs(pd)),
        "any_of_cats": ['Antiplatelet'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2014, Chunk 1: Antiplatelet Agents',
        "domain":      'ascvd_antiplatelet',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0345',
        "description": 'Diabetes with moderately elevated albuminuria (UACR 30-299 mg/g): recommend ACE inhibitor or ARB monotherapy.',
        "condition":   lambda cm, pd, yr: (yr >= 2014 and _get_dm_type(cm) in ("T1DM", "T2DM") and (lambda u: u is not None and 30 <= u <= 299)((cm.get("kidney") or {}).get("uacr"))),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2014, Chunk 2: Treatment (Level C)',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0346',
        "description": 'Use loop diuretics instead of thiazide diuretics when eGFR<30 mL/min/1.73m² in diabetic hypertensive patients',
        "condition":   lambda cm, pd, yr: (yr >= 2014 and _get_dm_type(cm) in ("T1DM", "T2DM") and cm.get("has_hypertension") is True and (lambda egfr: egfr is not None and egfr < 30)(_get_egfr(cm))),
        "any_of_cats": ['Hypertension_Diuretics'],
        "exclude_drugs": ['hydrochlorothiazide', 'chlorthalidone'],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2014, Treatment Strategies section',
        "domain":      'bp_other',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'P2-A: exclude 指向策展中有意移除的药(噻嗪/非DHP-CCB/gemfibrozil/保钾/跨类别),commission 永不触发,孤儿规则',
    },
    {
        "id":          'FINAL-0347',
        "description": 'Prescribe glucagon for individuals at significant risk of severe hypoglycemia and instruct caregivers on administration',
        "condition":   lambda cm, pd, yr: (yr >= 2014 and _get_dm_type(cm) in ("T1DM", "T2DM") and _has_hypoglycemia_hx(pd)),
        "any_of_cats": ['Hypoglycemia_Rescue'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2014, Recommendations section',
        "domain":      'hypoglycemia_rescue',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0348',
        "description": 'For T2DM patients with CKD (with or without ASCVD), consider using an SGLT2 inhibitor shown to reduce CKD progression',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2018 and (_has_ckd_g3plus(cm) or _has_albuminuria(cm))),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Consensus Reports & Supplements 2018 | Consensus recommendation (Fig 2 & 3 text)',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0349',
        "description": 'For T2DM patients with CKD where SGLT2i is contraindicated or not preferred, use a GLP-1 receptor agonist shown to reduce CKD progression',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2018 and (_has_ckd_g3plus(cm) or _has_albuminuria(cm))),
        "any_of_cats": ['GLP-1RA'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Consensus Reports & Supplements 2018 | Consensus recommendation (Fig 2 & 3 text)',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0350',
        "description": 'In T2DM patients without established CVD but with multiple CV risk factors (age≥55, obesity, hypertension, smoking, dyslipidemia, or albuminuria), consider a GLP-1RA or SGLT2i with proven benefit to reduce MACE and improve cardiorenal outcomes.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2019 and not _get_dm_comp(cm, "cardiovascular") and (_get_age(cm, pd) or 0) >= 55 and (cm.get("has_hypertension") is True or _has_albuminuria(cm) or (lambda bmi: bmi is not None and bmi >= 30)(_get_bmi_val(pd)))),
        "any_of_cats": ['GLP-1RA', 'SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Consensus Reports & Supplements 2022, Consensus Recommendations',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0351',
        "description": 'Recommend GLP-1 receptor agonists in T2DM patients with CKD or ASCVD for cardiovascular and kidney protection.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2017 and (_get_dm_comp(cm, "cardiovascular") or _has_ckd_g3plus(cm) or _has_albuminuria(cm))),
        "any_of_cats": ['GLP-1RA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Consensus Reports & Supplements 2022, Use of Additional Glucose-Lowering Agents',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0352',
        "description": 'Prioritize SGLT2i in T2DM patients with HF (especially HFrEF) or CKD for reducing HHF and CKD progression',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2019 and (_get_dm_comp(cm, "heart_failure") or _has_ckd_g3plus(cm) or _has_albuminuria(cm))),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Consensus Reports & Supplements 2022, People With Cardiorenal Comorbidities',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0353',
        "description": 'For outpatients with severe hypoglycemia, recommend glucagon preparations that do not require reconstitution over those that require reconstitution.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2019 and ((cm.get("glucose") or {}).get("severe_hypoglycemia_events") or 0) > 0),
        "any_of_cats": ['Hypoglycemia_Rescue'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Consensus Reports & Supplements, Recommendation 10',
        "domain":      'hypoglycemia_rescue',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0354',
        "description": 'For T2DM with CKD, SGLT2i and/or GLP-1RA with demonstrated CV/kidney benefit is recommended independent of A1C.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2019 and (_has_ckd_g3plus(cm) or _has_albuminuria(cm))),
        "any_of_cats": ['SGLT2i', 'GLP-1RA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2024 - Section 9: Combination Therapy',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0355',
        "description": 'SGLT2i or GLP-1 RA with demonstrated CVD benefit recommended for T2DM with CKD',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2019 and (_has_ckd_g3plus(cm) or _has_albuminuria(cm))),
        "any_of_cats": ['SGLT2i', 'GLP-1RA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2021 Section 9 Rec 9.9',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0356',
        "description": 'Combination therapy has not shown additional CV benefit above statin alone and is not generally recommended',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2014
        ),
        "any_of_cats": ['Lipid_NonStatins'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2014 - Treatment Recommendations and Goals',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'audit:avoid_uncheckable',
    },
    {
        "id":          'FINAL-0357',
        "description": 'ACE inhibitors provide selective benefit in delaying progression of increased urinary albumin excretion in T1DM',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T1DM"
            and yr >= 2014
            and _has_ckd_with_proteinuria(cm)
        ),
        "any_of_cats": ['Hypertension_ACEi'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2014 - Treatment',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0358',
        "description": 'Ezetimibe, PCSK9i, or icosapent ethyl if indicated based on ASCVD risk and lipids.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2015
            and _get_dm_comp(cm, "cardiovascular")
        ),
        "any_of_cats": ['Lipid_NonStatins'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Consensus Reports & Supplements 2022 Figure',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0359',
        "description": 'Consider specific SGLT2i or GLP-1 RA to improve HF and cardiovascular outcomes in T2D patients with HF',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2018
            and _get_dm_comp(cm, "heart_failure")
        ),
        "any_of_cats": ['SGLT2i', 'GLP-1RA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Consensus Reports & Supplements 2018 - Implications of New Evidence From CVOTs',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0435',
        "description": 'For T2DM patients with heart failure (especially HFrEF), recommend SGLT2i to reduce HHF and CV death.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2020
            and _has_hfref(cm, pd)
        ),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2024, Table 9.2',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0436',
        "description": 'For T2DM patients with CKD (eGFR≥20-30 with albuminuria), recommend SGLT2i to slow DKD progression.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2020 and (lambda egfr: egfr is not None and egfr >= 25)(_get_egfr(cm)) and _has_albuminuria(cm)),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2024, Table 9.2',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0437',
        "description": 'For T2DM patients with established ASCVD or high risk, preferentially prescribe SGLT2i or GLP-1RA with proven CV benefit, regardless of A1C or metformin use.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2024 and _get_dm_comp(cm, "cardiovascular")),
        "any_of_cats": ['SGLT2i', 'GLP-1RA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2024, Cardiovascular Outcomes Trials section',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0438',
        "description": 'For T2DM patients with CKD, recommend SGLT2i with renal benefit.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2024 and (_has_ckd_g3plus(cm) or _has_albuminuria(cm)) and (lambda egfr: egfr is not None and egfr >= 30)(_get_egfr(cm))),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2024, Cardiovascular Outcomes Trials section',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0439',
        "description": 'For T2DM with ASCVD, HF, or CKD, recommend SGLT2i and/or GLP-1RA with CV benefit, independent of A1C.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2026 and (_get_dm_comp(cm, "cardiovascular") or _get_dm_comp(cm, "heart_failure") or _has_ckd_g3plus(cm) or _has_albuminuria(cm))),
        "any_of_cats": ['SGLT2i', 'GLP-1RA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2026, Glucose-Lowering Therapy for People With Cardiovascular Disease',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0440',
        "description": 'For T2DM with obesity and HFpEF, recommend GLP-1RA or dual GIP/GLP-1 RA with demonstrated benefits.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2026 and _has_hfpef(cm, pd) and (lambda bmi: bmi is not None and bmi >= 30)(_get_bmi_val(pd))),
        "any_of_cats": ['GLP-1RA', 'GIP_GLP1_RA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2026, Glucose-Lowering Therapy for People With Cardiovascular Disease',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0441',
        "description": 'For T2DM patients with established ASCVD, HF, or CKD, recommend SGLT2 inhibitor and/or GLP-1 RA with proven cardiovascular benefit, independent of A1C',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2020 and (_get_dm_comp(cm, "cardiovascular") or _get_dm_comp(cm, "heart_failure") or _has_ckd_g3plus(cm) or _has_albuminuria(cm))),
        "any_of_cats": ['SGLT2i', 'GLP-1RA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2025 Chunk 2 & 2026 Chunk 3',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0442',
        "description": 'For T2DM with heart failure (reduced or preserved ejection fraction), use SGLT2i to prevent HF hospitalization',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2022
            and _get_dm_comp(cm, "heart_failure")
        ),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2025, Section 9.11',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0443',
        "description": 'For T2DM with CKD (eGFR 20-60 and/or albuminuria), use SGLT2i or GLP-1 RA to slow CKD progression',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2020 and ((lambda egfr: egfr is not None and 20 <= egfr <= 60)(_get_egfr(cm)) or _has_albuminuria(cm))),
        "any_of_cats": ['SGLT2i', 'GLP-1RA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2025, Section 9.13',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0444',
        "description": 'For T2DM with symptomatic HFpEF and obesity, use GLP-1 RA with demonstrated reduction in HF-related symptoms',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2024 and _has_hfpef(cm, pd) and (lambda b: b is not None and b >= 30)(_get_bmi_val(pd))),
        "any_of_cats": ['GLP-1RA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2025, Section 9.12',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0445',
        "description": 'SGLT2i and/or GLP-1RA with CVD benefit for T2DM + HF',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2020
            and _get_dm_comp(cm, "heart_failure")
        ),
        "any_of_cats": ['SGLT2i', 'GLP-1RA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2022, Section 9.4b, 9.9',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0446',
        "description": 'SGLT2i recommended for T2DM with CKD (eGFR >20) to slow CKD progression',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2020 and (_has_ckd_g3plus(cm) or _has_albuminuria(cm)) and (lambda egfr: egfr is not None and egfr > 20)(_get_egfr(cm))),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2026, Table 9.2',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0447',
        "description": 'In T2DM with established ASCVD, heart failure, and/or CKD, discharge regimen should include agents that reduce cardiorenal risk (e.g., SGLT2i, GLP-1RA).',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2020 and (_get_dm_comp(cm, "cardiovascular") or _get_dm_comp(cm, "heart_failure") or _has_ckd_g3plus(cm) or _has_albuminuria(cm))),
        "any_of_cats": ['SGLT2i', 'GLP-1RA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2023, Recommendation 9.4b',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0448',
        "description": 'Prescribe glucagon for all individuals requiring intensive insulin therapy or at high risk for hypoglycemia',
        "condition":   lambda cm, pd, yr: (yr >= 2025 and (_requires_insulin(cm, pd) or _is_on_sulfonylurea(pd))),
        "any_of_cats": ['Hypoglycemia_Rescue'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2025 Section 9, Recommendation 9.28 (Chunk 4)',
        "domain":      'hypoglycemia_rescue',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0449',
        "description": 'Prescribe glucagon for all individuals treated with insulin or at high risk for hypoglycemia',
        "condition":   lambda cm, pd, yr: (yr >= 2026 and (_requires_insulin(cm, pd) or _is_on_sulfonylurea(pd))),
        "any_of_cats": ['Hypoglycemia_Rescue'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2026, Section 9 (Glucagon)',
        "domain":      'hypoglycemia_rescue',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0450',
        "description": 'For resistant hypertension with eGFR≥45 in CKD, consider Finerenone (ns-MRA)',
        "condition":   lambda cm, pd, yr: (yr >= 2022 and _get_dm_type(cm) == 'T2DM' and _has_ckd_g3plus(cm) and (lambda egfr: egfr is not None and egfr >= 45)(_get_egfr(cm)) and cm.get('has_hypertension') == True),
        "any_of_cats": ['CKD_MRA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Consensus Reports 2022, Figure 3 Holistic Approach',
        "domain":      'ckd_mra',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0451',
        "description": 'T2DM patients with CKD (eGFR 25-75 and UACR≥200 mg/g) should be prescribed SGLT2i (dapagliflozin or empagliflozin) at discharge for kidney and cardiovascular protection',
        "condition":   lambda cm, pd, yr: (_has_ckd_g3plus(cm)
            and _get_dm_type(cm) == 'T2DM' and yr >= 2020 and (lambda egfr: egfr is not None and 25 <= egfr <= 75)(_get_egfr(cm)) and (lambda uacr: uacr is not None and uacr >= 200)((cm.get('kidney') or {}).get('uacr'))),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Consensus Reports & Supplements 2022, SGLT2i section (#5)',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0452',
        "description": 'For T2D with CKD (eGFR 25-75, albuminuria), use finerenone to reduce kidney and cardiovascular risk',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == 'T2DM' and yr >= 2022 and (lambda egfr: egfr is not None and 25 <= egfr <= 75)(_get_egfr(cm)) and _has_albuminuria(cm)),
        "any_of_cats": ['CKD_MRA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Consensus Reports & Supplements 2022, Management of Hypertension in Individuals With Diabetes',
        "domain":      'ckd_mra',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0453',
        "description": 'For T2DM patients hospitalized with heart failure, initiate or continue SGLT2 inhibitor during hospitalization and at discharge, if no contraindications and after recovery from acute illness.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2024 and _get_dm_comp(cm, "heart_failure") and (lambda egfr: egfr is not None and egfr >= 20)(_get_egfr(cm))),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Consensus Reports & Supplements 2024, Recommendation 16.11',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0454',
        "description": 'All individuals with diabetes and CKD (eGFR<60 or UACR>30) should receive SGLT2i with proven kidney benefit; if SGLT2i contraindicated/not preferred, use GLP-1 RA',
        "condition":   lambda cm, pd, yr: (yr >= 2022 and _get_dm_type(cm) in ("T1DM", "T2DM") and (_has_ckd_g3plus(cm) or _has_albuminuria(cm))),
        "any_of_cats": ['SGLT2i', 'GLP-1RA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Consensus Reports & Supplements 2022 - Choice of Glucose-Lowering Medication',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0455',
        "description": 'In T2DM patients with NASH at intermediate to high risk of hepatic fibrosis, consider treatment with pioglitazone and/or a GLP-1 RA (off-label).',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2022),
        "any_of_cats": ['Thiazolidinediones', 'GLP-1RA'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Consensus Reports & Supplements 2022, OBESITY AND WEIGHT-RELATED COMORBIDITIES, PARTICULARLY NAFLD',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'audit:NASH + hepatic fibrosis risk not extractable from cohort; rule would activate for all T2DM losing NASH-specificity (TZD/GLP-1 already covered by other glycemic rules -> zero scoring loss)',
    },
    {
        "id":          'FINAL-0456',
        "description": 'Statin recommended for all T1D or T2D patients with CKD for primary or secondary ASCVD prevention',
        "condition":   lambda cm, pd, yr: (yr >= 2022 and _get_dm_type(cm) in ("T1DM", "T2DM") and (_has_ckd_g3plus(cm) or _has_albuminuria(cm) or _get_dm_comp(cm, "nephropathy"))),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Consensus Reports & Supplements 2022, Consensus Statement',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0458',
        "description": 'For diabetic patients at risk for severe hypoglycemia (e.g., history of hypoglycemia), prescribe glucagon preparations that do not require reconstitution (intranasal or autoinjector) over traditional reconstitution forms for outpatient rescue use.',
        "condition":   lambda cm, pd, yr: (yr >= 2020 and _get_dm_type(cm) in ("T1DM", "T2DM") and _has_hypoglycemia_hx(pd)),
        "any_of_cats": ['Hypoglycemia_Rescue'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Consensus Reports & Supplements, Recommendation 10 and Justification (年份推断为2020)',
        "domain":      'hypoglycemia_rescue',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0459',
        "description": 'In T2DM with heart failure, dual GIP and GLP-1 RA (tirzepatide) may be used for HF benefit',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2026
            and _get_dm_comp(cm, "heart_failure")
        ),
        "any_of_cats": ['GIP_GLP1_RA'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2026 - Table 9 Glucose-Lowering Medications',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0460',
        "description": 'Continue or start SGLT2i for cardiovascular and kidney benefit if eGFR > 20 mL/min/1.73 m2',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2026 and (lambda egfr: egfr is not None and egfr > 20)(_get_egfr(cm))),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2026 Table 9.2',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0461',
        "description": 'Glucagon is recommended for emergent hypoglycemia in T1DM patients with hypoglycemia history.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T1DM" and yr >= 2026 and _has_hypoglycemia_hx(pd)),
        "any_of_cats": ['Hypoglycemia_Rescue'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ADA Standards of Care 2026 - Fig 9.2',
        "domain":      'hypoglycemia_rescue',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0462',
        "description": 'Recommend ns-MRA (Finerenone) for DKD with resistant HTN and eGFR >= 45',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2022 and (lambda egfr: egfr is not None and egfr >= 45)(_get_egfr(cm)) and _has_albuminuria(cm) and (lambda k: k is None or k <= 5.0)(_get_potassium(pd))),
        "any_of_cats": ['CKD_MRA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Consensus Reports & Supplements 2022 - Holistic approach for DM and CKD',
        "domain":      'ckd_mra',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0463',
        "description": 'GDMT for patients with HF and diabetes should include ARNI (or ACEi/ARB), evidence-based beta-blockers, MRA, and SGLT2i',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) in ("T1DM", "T2DM") and yr >= 2020 and _get_dm_comp(cm, "heart_failure") and (lambda egfr: egfr is None or egfr >= 30)(_get_egfr(cm)) and (lambda k: k is None or k <= 5.0)(_get_potassium(pd))),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB', 'Hypertension_BetaBlockers', 'CKD_MRA', 'SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Consensus Reports & Supplements 2022 - SUMMARY AND CONCLUSIONS',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0503',
        "description": 'In T2DM with TG >200 mg/dL despite maximally tolerated statin, add fenofibrate and/or high-dose prescription omega-3 fatty acids.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2001 and (lambda tg: tg is not None and tg > 200)((cm.get("lipids") or {}).get("triglycerides"))),
        "any_of_cats": ['Lipid_NonStatins'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      '',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0505',
        "description": 'For T2DM patients with CKD (eGFR <60 or UACR ≥30 mg/g), an ACEi or ARB should be prescribed as first-line antihypertensive therapy to slow nephropathy progression.',
        "condition":   lambda cm, pd, yr: (_has_ckd_g3plus(cm)
            and _get_dm_type(cm) == "T2DM" and yr >= 2001 and ((lambda egfr: egfr is not None and egfr < 60)(_get_egfr(cm)) or (lambda uacr: uacr is not None and uacr >= 30)((cm.get("kidney") or {}).get("uacr")))),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      '',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0528',
        "description": 'Avoid alogliptin and saxagliptin in T2DM patients with heart failure due to increased hospitalization risk',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2014
            and _get_dm_comp(cm, "heart_failure")
        ),
        "any_of_cats": ['DPP-4i'],
        "exclude_drugs": ['alogliptin', 'saxagliptin'],
        "grade":       0.75,
        "source":      '',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0530',
        "description": 'In T2DM with ASCVD or ≥2 CV risk factors and TG >135 mg/dL, consider adding icosapent ethyl.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2019 and _get_dm_comp(cm, "cardiovascular") and (lambda tg: tg is not None and tg > 135)((cm.get("lipids") or {}).get("triglycerides"))),
        "any_of_cats": ['Lipid_NonStatins'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      '',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0532',
        "description": 'Add ezetimibe to maximally tolerated statin if lipid targets not achieved in T2DM patients',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2016 and (lambda l: l is not None and l > 70)((cm.get("lipids") or {}).get("ldl"))),
        "any_of_cats": ['Lipid_NonStatins'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      '',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0533',
        "description": 'T2DM patients with ASCVD or heart failure should be prescribed SGLT2 inhibitors for cardiorenal protection at discharge',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2016 and (_get_dm_comp(cm, "cardiovascular") or _get_dm_comp(cm, "heart_failure")) and (lambda egfr: egfr is not None and 45 <= egfr)(_get_egfr(cm))),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      '',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0534',
        "description": 'ACEIs and ARBs are preferred for patients with T2D to slow progression of nephropathy and retinopathy',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2018 and _get_dm_comp(cm, "nephropathy")),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      '',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0536',
        "description": 'T2DM with CKD stage 3 or 4 should receive statin therapy for extreme risk lipid lowering',
        "condition":   lambda cm, pd, yr: (_has_ckd_g3plus(cm)
            and 
            _get_dm_type(cm) == "T2DM"
            and yr >= 2018
            and (lambda egfr: egfr is not None and 15 <= egfr and egfr <= 59)(_get_egfr(cm))
        ),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      '',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0545',
        "description": 'Consider finerenone in T2DM with CKD (eGFR>25, UACR≥30) at discharge to reduce CKD progression and CV events',
        "condition":   lambda cm, pd, yr: (_has_ckd_g3plus(cm)
            and _get_dm_type(cm) == "T2DM" and yr >= 2023 and (lambda egfr: egfr is not None and egfr > 25)(_get_egfr(cm)) and (lambda u: u is not None and u >= 30)((cm.get("kidney") or {}).get("uacr"))),
        "any_of_cats": ['CKD_MRA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      '',
        "domain":      'ckd_mra',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0547',
        "description": 'For T2DM patients with prior TIA or stroke, consider pioglitazone to reduce recurrent stroke risk',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2023 and (_has_stroke_hx(pd) or _has_tia(pd))),
        "any_of_cats": ['Thiazolidinediones'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      '',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0678',
        "description": 'For T2DM patients with heart failure and LVEF ≤40%, an ACE-I (or ARB if intolerant) and a beta-blocker should be prescribed.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2013
            and _has_hfref(cm, pd)
        ),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB', 'Hypertension_BetaBlockers'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 2013, Section 8.3',
        "domain":      'hf_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0679',
        "description": 'For T2DM patients with heart failure, LVEF ≤35%, and persistent symptoms (NYHA II-IV) despite ACE-I/ARB and beta-blocker, an MRA should be added.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2013 and _get_dm_comp(cm, "heart_failure") and (lambda lvef: lvef is not None and lvef <= 35)(_get_lvef(pd)) and (lambda egfr: egfr is not None and egfr >= 30)(_get_egfr(cm)) and (lambda k: k is None or k <= 5.0)(_get_potassium(pd))),
        "any_of_cats": ['CKD_MRA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 2013, Section 8.3',
        "domain":      'ckd_mra',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0681',
        "description": 'Antiplatelet therapy may be considered for primary prevention in high-risk DM patients on individual basis',
        "condition":   lambda cm, pd, yr: (yr >= 2013 and _get_dm_type(cm) in ("T1DM", "T2DM") and not _get_dm_comp(cm, "cardiovascular")),
        "any_of_cats": ['Antiplatelet'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ESC CVD in Diabetes 2023 (2013 version) 6.5.4',
        "domain":      'ascvd_antiplatelet',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0685',
        "description": 'Statin therapy indicated in DM with CAD to reduce CV risk',
        "condition":   lambda cm, pd, yr: (yr >= 2013 and _get_dm_type(cm) in ("T1DM", "T2DM") and _has_cad(pd)),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 (2013 version) 7.1.9',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0686',
        "description": 'Beta-blockers should be considered to reduce mortality and morbidity in DM with ACS',
        "condition":   lambda cm, pd, yr: (yr >= 2013 and _get_dm_type(cm) in ("T1DM", "T2DM") and _has_acs(pd)),
        "any_of_cats": ['Hypertension_BetaBlockers'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ESC CVD in Diabetes 2023 (2013 version) 7.1.9',
        "domain":      'bp_other',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0687',
        "description": 'ACE-I recommended in addition to beta-blockers in systolic HF and T2DM',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2013
            and _has_hfref(cm, pd)
        ),
        "any_of_cats": ['Hypertension_ACEi'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 (2013 version) 8.7',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0688',
        "description": 'ARB may be used as alternative to ACE-I in systolic HF with ACE-I intolerance',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2013 and _has_hfref(cm, pd)),
        "any_of_cats": ['Hypertension_ARB'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 (2013 version) 8.7',
        "domain":      'bp_raas',
        "polarity":    'ALLOW',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0690',
        "description": 'Prescribe MRA for hospitalized DM patients with HFrEF at discharge',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) in ("T1DM", "T2DM") and yr >= 2003 and _has_hfref(cm, pd)),
        "any_of_cats": ['CKD_MRA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 (2021 version) - Quality indicators, Domain 3, Main (4)',
        "domain":      'ckd_mra',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0693',
        "description": 'In diabetic patients with heart failure but without established ASCVD (e.g., CAD), routine initiation of statins is not recommended; continuation is safe if already on statin.',
        "condition":   lambda cm, pd, yr: (
            yr >= 2008
            and not _has_cad(pd)
            and _get_dm_comp(cm, "heart_failure")
        ),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ESC CVD in Diabetes 2023 (2021 version), Section 13.9',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'audit:avoid_uncheckable',
    },
    {
        "id":          'FINAL-0694',
        "description": 'In hospitalized diabetic patients with HFrEF (LVEF ≤40%) and NYHA class II–IV symptoms, initiate MRA (spironolactone or eplerenone) if eGFR >30 and potassium <5.0 to reduce morbidity and mortality.',
        "condition":   lambda cm, pd, yr: (yr >= 2003 and _get_dm_type(cm) in ("T1DM", "T2DM") and _has_hfref(cm, pd) and (lambda egfr: egfr is not None and egfr > 30)(_get_egfr(cm)) and (lambda k: k is None or k < 5.0)(_get_potassium(pd))),
        "any_of_cats": ['CKD_MRA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ACC-AHA Heart Failure Guideline 2022, Section 7.3.3 Mineralocorticoid Receptor Antagonists',
        "domain":      'ckd_mra',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0695',
        "description": 'ACE inhibitors are recommended for all patients with HFrEF (LVEF≤40%) in diabetic hospitalized patients.',
        "condition":   lambda cm, pd, yr: (yr >= 2013 and _get_dm_type(cm) in ("T1DM", "T2DM") and _has_hfref(cm, pd)),
        "any_of_cats": ['Hypertension_ACEi'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ACC-AHA 2013 Heart Failure Guideline - Class I A',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0696',
        "description": 'ARBs are recommended in patients with HFrEF who are ACE inhibitor intolerant in diabetic hospitalized patients.',
        "condition":   lambda cm, pd, yr: (yr >= 2013 and _get_dm_type(cm) in ("T1DM", "T2DM") and _has_hfref(cm, pd)),
        "any_of_cats": ['Hypertension_ARB'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ACC-AHA 2013 Heart Failure Guideline - Class I A',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0697',
        "description": 'Aldosterone receptor antagonists are recommended in patients following an acute MI who have LVEF ≤40% with symptoms of HF or DM.',
        "condition":   lambda cm, pd, yr: (yr >= 2013 and _get_dm_type(cm) in ("T1DM", "T2DM") and _has_acute_mi(pd) and (_has_hfref(cm, pd) or _get_dm_type(cm) in ("T1DM", "T2DM")) and (lambda e: e is not None and e > 30)(_get_egfr(cm)) and (lambda k: k is None or k <= 5)(_get_potassium(pd))),
        "any_of_cats": ['CKD_MRA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ACC-AHA 2013 Heart Failure Guideline - Class I B',
        "domain":      'ckd_mra',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0698',
        "description": 'Aldosterone receptor antagonists are recommended in patients following an acute MI who have LVEF ≤40% with DM (even without HF symptoms).',
        "condition":   lambda cm, pd, yr: (yr >= 2013 and _get_dm_type(cm) in ("T1DM", "T2DM") and _has_acute_mi(pd) and not _has_hfref(cm, pd) and (lambda e: e is not None and e > 30)(_get_egfr(cm)) and (lambda k: k is None or k <= 5)(_get_potassium(pd))),
        "any_of_cats": ['CKD_MRA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ACC-AHA 2013 Heart Failure Guideline - Class I B',
        "domain":      'ckd_mra',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0699',
        "description": 'Aldosterone receptor antagonists are recommended in patients with NYHA class II-IV who have LVEF ≤35% in diabetic hospitalized patients.',
        "condition":   lambda cm, pd, yr: (yr >= 2013 and _get_dm_type(cm) in ("T1DM", "T2DM") and _has_hfref(cm, pd) and (lambda e: e is not None and e > 30)(_get_egfr(cm)) and (lambda k: k is None or k <= 5)(_get_potassium(pd))),
        "any_of_cats": ['CKD_MRA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ACC-AHA 2013 Heart Failure Guideline - Class I A',
        "domain":      'ckd_mra',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0700',
        "description": 'For diabetic patients aged 40-75 years without ASCVD, moderate-intensity statin therapy is recommended; high-intensity may be considered based on risk',
        "condition":   lambda cm, pd, yr: (yr >= 2002 and _get_dm_type(cm) in ("T1DM", "T2DM") and not _get_dm_comp(cm, "cardiovascular") and (_get_age(cm, pd) or 0) >= 40 and (_get_age(cm, pd) or 0) <= 75),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ACC-AHA Cholesterol Guideline 2018, Figure 2 Primary Prevention',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0701',
        "description": 'In patients with clinical ASCVD (including diabetes), use high-intensity statin if age ≤75, or moderate-intensity statin if age >75 or contraindicated',
        "condition":   lambda cm, pd, yr: (yr >= 2013 and _get_dm_type(cm) in ("T1DM", "T2DM") and _get_dm_comp(cm, "cardiovascular")),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ACC-AHA Cholesterol/Cardiovascular 2013 Section 4.3 Secondary Prevention',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0702',
        "description": 'Patients (≥21 years) with primary LDL-C ≥190 mg/dL should receive statin therapy for primary prevention',
        "condition":   lambda cm, pd, yr: (yr >= 2013 and _get_dm_type(cm) in ("T1DM", "T2DM") and not _get_dm_comp(cm, "cardiovascular") and (_get_age(cm, pd) or 0) >= 21 and (lambda l: l is not None and l >= 190)((cm.get("lipids") or {}).get("ldl"))),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ACC-AHA Cholesterol/Cardiovascular 2013 Section 4.4 Primary Prevention',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0703',
        "description": 'For diabetic patients with current or prior symptoms of HFrEF, one of the three beta blockers proven to reduce mortality (bisoprolol, carvedilol, metoprolol succinate) is recommended to reduce morbidity and mortality',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) in ("T1DM", "T2DM") and yr >= 2013 and _has_hfref(cm, pd)),
        "any_of_cats": ['Hypertension_BetaBlockers'],
        "exclude_drugs": ['bucindolol', 'metoprolol tartrate', 'nebivolol'],
        "grade":       1.0,
        "source":      'ACC-AHA 2013 HF Guideline, Section 7.3.2.4.1',
        "domain":      'hf_management',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'P2-A: exclude 指向策展中有意移除的药(噻嗪/非DHP-CCB/gemfibrozil/保钾/跨类别),commission 永不触发,孤儿规则',
    },
    {
        "id":          'FINAL-0704',
        "description": 'Diuretics are recommended in diabetic patients with HF and evidence of fluid retention to improve symptoms',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) in ("T1DM", "T2DM") and yr >= 2013 and _get_dm_comp(cm, "heart_failure")),
        "any_of_cats": ['Hypertension_Diuretics'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ACC-AHA 2013 HF Guideline, Section 7.3.2.1.1',
        "domain":      'bp_other',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0705',
        "description": 'In patients with type 2 diabetes without clinical ASCVD, aged 40-75 years and LDL-C 70-189 mg/dL, moderate-intensity statin should be initiated (LDL-C data unavailable; but diabetes alone qualifies as high risk).',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2013 and not _get_dm_comp(cm, "cardiovascular") and 40 <= (_get_age(cm, pd) or 0) <= 75 and (lambda l: l is None or (70 <= l <= 189))((cm.get("lipids") or {}).get("ldl"))),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ACC-AHA Cholesterol / Cardiovascular 2013, Section 2.1',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0706',
        "description": 'Moderate-intensity statin therapy is indicated for primary prevention in patients 40-75 years of age with diabetes.',
        "condition":   lambda cm, pd, yr: (yr >= 2019 and _get_dm_type(cm) in ("T1DM", "T2DM") and 40 <= (_get_age(cm, pd) or 0) <= 75),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ACC-AHA Cholesterol / Cardiovascular 2019, Section 4.3, Figure 3 and Recommendation-Specific Supportive Text',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0707',
        "description": 'For diabetic patients with HFrEF, recommend ACEi or ARB combined with beta blocker as GDMT.',
        "condition":   lambda cm, pd, yr: (yr >= 2013 and _get_dm_type(cm) in ("T1DM", "T2DM") and _has_hfref(cm, pd)),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB', 'Hypertension_BetaBlockers'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ACC-AHA Cholesterol/Cardiovascular 2013, HFrEF Treatment Algorithm, Figure 1, Class I LOE A',
        "domain":      'hf_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0708',
        "description": 'For diabetic patients with HFrEF (LVEF≤35%), eGFR≥30, K≤5.0, recommend adding aldosterone antagonist.',
        "condition":   lambda cm, pd, yr: (yr >= 2013 and _get_dm_type(cm) in ("T1DM", "T2DM") and _has_hfref(cm, pd) and (lambda egfr: egfr is not None and egfr >= 30)(_get_egfr(cm)) and (lambda k: k is not None and k <= 5.0)(_get_potassium(pd))),
        "any_of_cats": ['CKD_MRA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ACC-AHA Cholesterol/Cardiovascular 2013, Section 7.3.2.5.1, Figure 1, Class I LOE A',
        "domain":      'ckd_mra',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0709',
        "description": 'Avoid aldosterone antagonists if eGFR<30 or K>5.0 due to risk of life-threatening hyperkalemia or renal insufficiency.',
        "condition":   lambda cm, pd, yr: (yr >= 2013 and ((lambda egfr: egfr is not None and egfr < 30)(_get_egfr(cm)) or (lambda k: k is not None and k > 5.0)(_get_potassium(pd)))),
        "any_of_cats": ['CKD_MRA'],
        "exclude_drugs": ['spironolactone', 'eplerenone'],
        "grade":       0.75,
        "source":      'ACC-AHA Cholesterol/Cardiovascular 2013, Section 7.3.2.5, Class III Harm, LOE B',
        "domain":      'ckd_mra',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0710',
        "description": 'Initiate high-intensity or moderate-intensity statin therapy for patients with diabetes and clinical ASCVD for secondary prevention.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2013 and _get_dm_comp(cm, "cardiovascular")),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ACC-AHA Cholesterol Guideline 2013, Section 2.2',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0712',
        "description": 'No recommendation for initiation or discontinuation of statins in patients with NYHA class II-IV ischemic systolic HF or on maintenance hemodialysis.',
        "condition":   lambda cm, pd, yr: (
            yr >= 2013
            and _has_hfref(cm, pd)
        ),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ACC-AHA Cholesterol / Cardiovascular 2013, Table 4, Heart Failure and Hemodialysis section',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0713',
        "description": 'T2DM patients aged 40-75 with predicted 10-year ASCVD risk ≥7.5% should initiate moderate or high-intensity statin',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2013 and (_get_age(cm, pd) or 0) >= 40 and (_get_age(cm, pd) or 0) <= 75),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ACC-AHA Cholesterol/Cardiovascular 2013 Conclusion',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0714',
        "description": 'Thiazolidinediones should be avoided in diabetic patients with heart failure due to fluid retention and worsening HF risk',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2013
            and _get_dm_comp(cm, "heart_failure")
        ),
        "any_of_cats": ['Thiazolidinediones'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ACC-AHA Cholesterol/Cardiovascular 2013, Section 5.3.2 Diabetic Cardiomyopathy',
        "domain":      'glycemic_intensification',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0715',
        "description": 'For hospitalized diabetic patients with HFpEF and volume overload, diuretics should be used for symptom relief.',
        "condition":   lambda cm, pd, yr: (yr >= 2013 and _get_dm_type(cm) in ("T1DM", "T2DM") and _has_hfpef(cm, pd)),
        "any_of_cats": ['Hypertension_Diuretics'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ACC-AHA Cholesterol/Cardiovascular 2013, HFpEF management, Section',
        "domain":      'bp_other',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0716',
        "description": 'ACEi for all DM patients with reduced EF (without MI/ACS) to prevent symptomatic HF.',
        "condition":   lambda cm, pd, yr: (yr >= 2013 and _get_dm_type(cm) in ("T1DM", "T2DM") and not _has_acs(pd) and _has_hfref(cm, pd)),
        "any_of_cats": ['Hypertension_ACEi'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ACC-AHA Cholesterol/Cardiovascular 2013 CLASS I',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0718',
        "description": 'In diabetic patients with heart failure (NYHA class II-IV), omega-3 PUFA supplementation may be reasonable as adjunctive therapy to reduce mortality and cardiovascular hospitalizations.',
        "condition":   lambda cm, pd, yr: (yr >= 2009 and _get_dm_type(cm) in ("T1DM", "T2DM") and _get_dm_comp(cm, "heart_failure")),
        "any_of_cats": ['Lipid_NonStatins'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ACC-AHA Cholesterol / Cardiovascular 2022, Recommendations for Other Drug Treatment',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0719',
        "description": 'Moderate-intensity statin therapy indicated for adults 40-75 years with diabetes mellitus without ASCVD',
        "condition":   lambda cm, pd, yr: (yr >= 2013 and _get_dm_type(cm) in ("T1DM", "T2DM") and not _get_dm_comp(cm, "cardiovascular") and 40 <= (_get_age(cm, pd) or 0) <= 75),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ACC-AHA Cholesterol/Cardiovascular 2018, Section 4.3',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0720',
        "description": 'Intensive statin therapy with LDL-C goal <70 mg/dL should be considered for diabetic patients hospitalized with acute coronary syndrome.',
        "condition":   lambda cm, pd, yr: (yr >= 2004 and _get_dm_type(cm) in ("T1DM", "T2DM") and _has_acs(pd)),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ACC-AHA Cholesterol 2004, Patients With Acute Coronary Syndromes',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0721',
        "description": 'For diabetic patients with heart failure, ACE inhibitors, beta-blockers, ARBs, or aldosterone blockers are recommended at discharge.',
        "condition":   lambda cm, pd, yr: (yr >= 2003 and _get_dm_type(cm) in ("T1DM", "T2DM") and _get_dm_comp(cm, "heart_failure")),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB', 'Hypertension_BetaBlockers', 'CKD_MRA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ACC-AHA Cholesterol 2003, Special Considerations - Heart Failure',
        "domain":      'hf_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0722',
        "description": 'Aldosterone receptor antagonists (MRA) recommended in DM patients with NYHA class II–IV HF and LVEF ≤35%, provided eGFR >30 mL/min and potassium <5.0 mEq/L.',
        "condition":   lambda cm, pd, yr: (yr >= 2013 and _get_dm_type(cm) in ("T1DM", "T2DM") and _get_dm_comp(cm, "heart_failure") and (lambda l: l is not None and l <= 35)(_get_lvef(pd)) and (lambda egfr: egfr is not None and egfr > 30)(_get_egfr(cm)) and (lambda k: k is not None and k < 5.0)(_get_potassium(pd))),
        "any_of_cats": ['CKD_MRA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ACC-AHA Cholesterol / Cardiovascular 2013, CLASS I, Level of Evidence A',
        "domain":      'ckd_mra',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0723',
        "description": 'Aldosterone receptor antagonists recommended in DM patients post-AMI with LVEF ≤40%, regardless of HF symptoms.',
        "condition":   lambda cm, pd, yr: (yr >= 2013 and _get_dm_type(cm) in ("T1DM", "T2DM") and _has_acute_mi(pd) and (lambda l: l is not None and l <= 40)(_get_lvef(pd)) and (lambda egfr: egfr is not None and egfr > 30)(_get_egfr(cm)) and (lambda k: k is not None and k < 5.0)(_get_potassium(pd))),
        "any_of_cats": ['CKD_MRA'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ACC-AHA Cholesterol / Cardiovascular 2013, CLASS I, Level of Evidence B',
        "domain":      'ckd_mra',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0724',
        "description": 'In diabetic patients with HF and LVEF <40%, prescribe ACE inhibitor or ARB at discharge.',
        "condition":   lambda cm, pd, yr: (yr >= 2013 and _get_dm_type(cm) in ("T1DM", "T2DM") and _has_hfref(cm, pd)),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ACC-AHA Cholesterol / Cardiovascular 2013, Section 13: Evidence Gaps and Future Research Directions (Quality Measure #7)',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0725',
        "description": 'Calcium channel blockers are not recommended as routine treatment for patients with HFrEF, including those with diabetes',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2013
            and _has_hfref(cm, pd)
        ),
        "any_of_cats": ['Hypertension_CCB'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ACC-AHA Cholesterol / Cardiovascular 2013, Section 7.3.2.9.3',
        "domain":      'bp_other',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'audit:inert — Hypertension_CCB has no live taxonomy members (amlodipine/verapamil/diltiazem/nifedipine → None in DrugNormalizer); and would mis-flag HFrEF-safe amlodipine if reinstated',
    },
    {
        "id":          'FINAL-0726',
        "description": 'African American diabetic patients with HFrEF and persistent symptoms despite ACEi/ARB, beta blocker, and MRA should be considered for hydralazine/isosorbide dinitrate.',
        "condition":   lambda cm, pd, yr: (yr >= 2004 and _get_dm_type(cm) in ("T1DM", "T2DM") and _has_hfref(cm, pd)),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB', 'Hypertension_BetaBlockers', 'CKD_MRA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ACC-AHA Cholesterol / Cardiovascular 2022, Chunk 2',
        "domain":      'hf_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0728',
        "description": 'For T2DM patients with established ASCVD, statin therapy is recommended for intensive lipid lowering to achieve LDL-C <100 mg/dL',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2004
            and _get_dm_comp(cm, "cardiovascular")
        ),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ACC-AHA Cholesterol/Cardiovascular 2004 - Diabetes Without CVD & Older Persons With Established CVD',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0729',
        "description": 'For T2DM patients without CVD aged ≥65 years, statin therapy may be considered based on clinical judgment',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2004
            and not _get_dm_comp(cm, "cardiovascular")
            and (_get_age(cm, pd) or 0) >= 65
        ),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ACC-AHA Cholesterol/Cardiovascular 2004 - Older Persons at High Risk Without Established CVD',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0730',
        "description": 'For hospitalized diabetic patients, ACE inhibitors or ARBs are recommended to prevent renal dysfunction and reduce the risk of heart failure',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2013 and bool(pd.get("has_hypertension"))),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ACC-AHA Cholesterol / Cardiovascular 2013, 7.1.3. Obesity and Diabetes Mellitus',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0731',
        "description": 'Statins should be considered for LDL-C lowering in adults older than 40 years with T1DM without CVD history',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T1DM"
            and yr >= 2004
            and not _get_dm_comp(cm, "cardiovascular")
            and (_get_age(cm, pd) or 0) >= 41
        ),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ESC CVD in Diabetes 2023 | 11.4 Recommendation Table 26',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0732',
        "description": 'In diabetes with hypertension, antihypertensive treatment should preferentially be based on combinations of an ACE inhibitor or ARB with a CCB and/or thiazide/thiazide-like diuretic',
        "condition":   lambda cm, pd, yr: (yr >= 2008 and _get_dm_type(cm) in ("T1DM", "T2DM") and cm.get("has_hypertension") is True),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB', 'Hypertension_CCB', 'Hypertension_Diuretics'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 | 7.5.2.1 Drug combinations for hypertension treatment | 2018',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0733',
        "description": 'Post-myocardial infarction is a compelling indication for β-Blocker, ACE Inhibitor, and Aldosterone Antagonist',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) in ("T1DM", "T2DM") and yr >= 2003 and _get_dm_comp(cm, "cardiovascular") and (lambda egfr: egfr is not None and egfr >= 30)(_get_egfr(cm)) and (lambda k: k is None or k <= 5)(_get_potassium(pd))),
        "any_of_cats": ['Hypertension_BetaBlockers', 'Hypertension_ACEi', 'CKD_MRA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ACC-AHA Cholesterol / Cardiovascular 2003 | Special Considerations',
        "domain":      'ckd_mra',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0734',
        "description": 'Avoid negative inotropic agents such as nondihydropyridine CCBs in HF',
        "condition":   lambda cm, pd, yr: (
            yr >= 2013
            and _get_dm_comp(cm, "heart_failure")
        ),
        "any_of_cats": ['Hypertension_CCB'],
        "exclude_drugs": ['diltiazem', 'verapamil', 'nondihydropyridine'],
        "grade":       1.0,
        "source":      'ACC-AHA Cardiovascular 2013 | 7.2.1. Management Strategies for Stage B',
        "domain":      'bp_other',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'P2-A: exclude 指向策展中有意移除的药(噻嗪/非DHP-CCB/gemfibrozil/保钾/跨类别),commission 永不触发,孤儿规则',
    },
    {
        "id":          'FINAL-0735',
        "description": 'Fenofibrate should not be used if moderate or severe renal impairment, defined as eGFR < 30 mL/min per 1.73 m², is present.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2013
            and (lambda egfr: egfr is not None and egfr < 30)(_get_egfr(cm))
        ),
        "any_of_cats": ['Lipid_NonStatins'],
        "exclude_drugs": ['fenofibrate'],
        "grade":       0.75,
        "source":      'ACC/AHA Cholesterol 2013 | 5. Safety: Recommendations',
        "domain":      'lipid_management',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0736',
        "description": 'Intensive lipid-lowering therapy should be considered for patients admitted with ACS',
        "condition":   lambda cm, pd, yr: (yr >= 2004 and _get_dm_type(cm) in ("T1DM", "T2DM") and _has_acs(pd)),
        "any_of_cats": ['Lipid_Statins', 'Lipid_NonStatins'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ACC-AHA Cholesterol/Cardiovascular 2004 - Patients With Acute Coronary Syndromes',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0737',
        "description": 'Continue ACEi or ARB at discharge for HFrEF patients without hemodynamic instability',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2013 and _has_hfref(cm, pd) and not (cm.get("dm_complications") or {}).get("vasopressor_use")),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ACC-AHA Cardiovascular 2013 - Hospitalized HF Patient',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0738',
        "description": 'Increasing doses of loop diuretics are usually needed in advanced renal disease (eGFR < 30)',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2003 and (lambda egfr: egfr is not None and egfr < 30)(_get_egfr(cm))),
        "any_of_cats": ['Hypertension_Diuretics'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ACC-AHA Cholesterol/Cardiovascular 2003 - Special Considerations',
        "domain":      'bp_other',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0739',
        "description": 'Initiate and maintain ACE inhibitor in HFrEF to reduce cardiovascular events',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2013
            and _has_hfref(cm, pd)
        ),
        "any_of_cats": ['Hypertension_ACEi'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ACC-AHA Cardiovascular 2013 | 7.3.2.2.2',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0740',
        "description": 'Initiation of GDMT for heart failure patients before hospital discharge if not contraindicated',
        "condition":   lambda cm, pd, yr: (yr >= 2013 and _get_dm_type(cm) in ("T1DM", "T2DM") and _get_dm_comp(cm, "heart_failure") and (lambda egfr: egfr is not None and egfr >= 30)(_get_egfr(cm)) and (lambda k: k is None or k <= 5)(_get_potassium(pd))),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB', 'Hypertension_BetaBlockers', 'CKD_MRA', 'Hypertension_Diuretics'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ACC-AHA Cardiovascular 2013 | CLASS I',
        "domain":      'hf_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0743',
        "description": 'For diabetic patients with cardiovascular disease (including peripheral artery disease), statins and antiplatelet drugs should be prescribed at discharge for secondary prevention.',
        "condition":   lambda cm, pd, yr: (yr >= 2013 and _get_dm_type(cm) in ("T1DM", "T2DM") and _get_dm_comp(cm, "cardiovascular")),
        "any_of_cats": ['Lipid_Statins', 'Antiplatelet'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ESC CVD in Diabetes 2013 | 10.1 Peripheral artery disease',
        "domain":      'ascvd_antiplatelet',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0745',
        "description": 'Avoid aldosterone receptor antagonists if eGFR <30 mL/min or potassium >5.0 mEq/L due to harm risk',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2013 and ((lambda egfr: egfr is not None and egfr < 30)(_get_egfr(cm)) or (lambda k: k is not None and k > 5.0)(_get_potassium(pd)))),
        "any_of_cats": ['CKD_MRA'],
        "exclude_drugs": ['spironolactone', 'eplerenone'],
        "grade":       0.75,
        "source":      'ACC-AHA Cholesterol / Cardiovascular 2013, Section 7.3.2.5.1 and CLASS III',
        "domain":      'ckd_mra',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0746',
        "description": 'For diabetic patients with heart failure (symptomatic ventricular dysfunction), prescribe ACE inhibitors, ARBs, beta-blockers, or MRAs at discharge for HF management',
        "condition":   lambda cm, pd, yr: (yr >= 2003 and _get_dm_type(cm) in ("T1DM", "T2DM") and _get_dm_comp(cm, "heart_failure") and (lambda egfr: egfr is not None and egfr >= 30)(_get_egfr(cm)) and (lambda k: k is None or k <= 5)(_get_potassium(pd))),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB', 'Hypertension_BetaBlockers', 'CKD_MRA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ACC-AHA Cholesterol / Cardiovascular 2003 | Special Considerations',
        "domain":      'hf_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0747',
        "description": 'Statins should not be prescribed solely for HF diagnosis in the absence of other indications',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2013
            and _get_dm_comp(cm, "heart_failure")
        ),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ACC-AHA Cholesterol / Cardiovascular 2013, Section 7.3.2.7',
        "domain":      'lipid_management',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'audit:anti-signal — statin "solely for HF/absence of other indications" uncheckable; in diabetes cohort statins are near-universally indicated → would penalise appropriate use (severity-confounded)',
    },
    {
        "id":          'FINAL-0748',
        "description": 'For diabetic patients hospitalized with heart failure, guideline-directed medical therapy (GDMT) including beta-blockers, ACEi/ARB, and MRAs should be continued during hospitalization to avoid risks of withdrawal.',
        "condition":   lambda cm, pd, yr: (yr >= 2008 and _get_dm_type(cm) in ("T1DM", "T2DM") and _get_dm_comp(cm, "heart_failure")),
        "any_of_cats": ['Hypertension_BetaBlockers', 'Hypertension_ACEi', 'Hypertension_ARB', 'CKD_MRA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ACC-AHA Cholesterol / Cardiovascular 2022 | 9.2. Maintenance or Optimization of GDMT During Hospitalization',
        "domain":      'hf_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0749',
        "description": 'High-intensity statins mandatory for ASCVD',
        "condition":   lambda cm, pd, yr: (yr >= 2013 and _get_dm_type(cm) in ("T1DM", "T2DM") and _get_dm_comp(cm, "cardiovascular") and (lambda egfr: egfr is not None and egfr >= 30)(_get_egfr(cm))),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ACC-AHA Cholesterol / Cardiovascular 2022 | Section 8',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0756',
        "description": 'For diabetic patients with heart failure, prescribe ACE-i/ARB/ARNI, beta-blockers, and MRA at discharge unless contraindicated',
        "condition":   lambda cm, pd, yr: (yr >= 2001 and _get_dm_type(cm) in ("T1DM", "T2DM") and _get_dm_comp(cm, "heart_failure") and (lambda e: e is not None and e >= 30)(_get_egfr(cm)) and (lambda k: k is None or k <= 5)(_get_potassium(pd))),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB', 'Hypertension_BetaBlockers', 'CKD_MRA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023, Section 5.3 (2021 version)',
        "domain":      'hf_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0757',
        "description": 'ACE-I (or ARB) is recommended in diabetic patients with HFrEF to reduce HF hospitalization and death.',
        "condition":   lambda cm, pd, yr: (yr >= 2001 and _get_dm_type(cm) in ("T1DM", "T2DM") and _has_hfref(cm, pd)),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 (2016 Chapter 7.2.1)',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0758',
        "description": 'Beta-blocker is recommended in diabetic patients with stable symptomatic HFrEF in addition to ACE-I to reduce HF hospitalization and death.',
        "condition":   lambda cm, pd, yr: (yr >= 2001 and _get_dm_type(cm) in ("T1DM", "T2DM") and _has_hfref(cm, pd)),
        "any_of_cats": ['Hypertension_BetaBlockers'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 (2016 Chapter 7.2.1)',
        "domain":      'hf_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0759',
        "description": 'MRA is recommended in diabetic patients with HFrEF who remain symptomatic despite ACE-I and beta-blocker to reduce HF hospitalization and death.',
        "condition":   lambda cm, pd, yr: (yr >= 2001 and _get_dm_type(cm) in ("T1DM", "T2DM") and _has_hfref(cm, pd) and (lambda egfr: egfr is not None and egfr >= 30)(_get_egfr(cm)) and (lambda k: k is None or k <= 5)(_get_potassium(pd))),
        "any_of_cats": ['CKD_MRA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 (2016 Chapter 7.2.1)',
        "domain":      'ckd_mra',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0760',
        "description": 'For T2DM patients with established ASCVD, discharge prescription should include high-intensity statin.',
        "condition":   lambda cm, pd, yr: (yr >= 2001 and _get_dm_type(cm) == "T2DM" and _get_dm_comp(cm, "cardiovascular")),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ESC CVD in Diabetes 2023, Chunk4 (2018) & 2021 prevention section',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0761',
        "description": 'For T2DM patients with ASCVD, discharge prescription should include low-dose aspirin for secondary prevention.',
        "condition":   lambda cm, pd, yr: (yr >= 2001 and _get_dm_type(cm) == "T2DM" and _get_dm_comp(cm, "cardiovascular")),
        "any_of_cats": ['Antiplatelet'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023, Chunk4 (2018) antiplatelet section',
        "domain":      'ascvd_antiplatelet',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0762',
        "description": 'Low-dose aspirin (75-100 mg o.d.) recommended long-term in T2DM patients with prior MI or revascularization',
        "condition":   lambda cm, pd, yr: (yr >= 2001 and _get_dm_type(cm) == "T2DM" and _has_prior_mi(pd)),
        "any_of_cats": ['Antiplatelet'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 2023 | 5.6.2.2',
        "domain":      'ascvd_antiplatelet',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0763',
        "description": 'In patients with symptomatic HFrEF (LVEF<40%), ACE inhibitors, beta-blockers, and MR antagonists are recommended as part of optimal medical therapy',
        "condition":   lambda cm, pd, yr: (yr >= 2001 and _get_dm_type(cm) in ("T1DM", "T2DM") and _has_hfref(cm, pd) and (lambda egfr: egfr is not None and egfr >= 30)(_get_egfr(cm)) and (lambda k: k is None or k <= 5)(_get_potassium(pd))),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_BetaBlockers', 'CKD_MRA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 (2016) - Figure 7.1 and Section 7.3.1',
        "domain":      'hf_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0764',
        "description": 'Statin therapy is recommended in T2DM patients at high risk (without any other CV risk factor and free of target organ damage) with LDL-C target <2.5 mmol/L.',
        "condition":   lambda cm, pd, yr: (yr >= 2001 and _get_dm_type(cm) == "T2DM" and not _get_dm_comp(cm, "cardiovascular") and not _get_dm_comp(cm, "nephropathy") and not _get_dm_comp(cm, "retinopathy") and not _get_dm_comp(cm, "neuropathy")),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 (2013 edition) section 6.4.5',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0766',
        "description": 'ACEi/ARB recommended for T2DM patients with CAD and HF or CKD to reduce CV events',
        "condition":   lambda cm, pd, yr: (yr >= 2001 and _get_dm_type(cm) == "T2DM" and _has_cad(pd) and (_get_dm_comp(cm, "heart_failure") or _has_ckd_g3plus(cm))),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023, 6.1.3.1.2',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0767',
        "description": 'ACEi/ARB recommended for T2DM patients with HFrEF as part of quadruple therapy',
        "condition":   lambda cm, pd, yr: (yr >= 2001 and _get_dm_type(cm) == "T2DM" and _has_hfref(cm, pd) and (lambda e: e is None or e >= 30)(_get_egfr(cm))),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023, 7.4.1',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0769',
        "description": 'Prescribe beta-blocker (bisoprolol, carvedilol, metoprolol succinate, nebivolol) for hospitalized DM patients with HFrEF at discharge',
        "condition":   lambda cm, pd, yr: (yr >= 2001 and _get_dm_type(cm) == "T2DM" and _has_hfref(cm, pd) and not (cm.get("kidney") or {}).get("aki_suspected")),
        "any_of_cats": ['Hypertension_BetaBlockers'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 (2021 version) - Quality indicators, Domain 3, Main (1)',
        "domain":      'hf_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0770',
        "description": 'Prescribe diuretic therapy for hospitalized DM patients with HF and evidence of fluid retention at discharge',
        "condition":   lambda cm, pd, yr: (yr >= 2000 and _get_dm_type(cm) == "T2DM" and _get_dm_comp(cm, "heart_failure")),
        "any_of_cats": ['Hypertension_Diuretics'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 (2021 version) - Quality indicators, Domain 3, Main (3)',
        "domain":      'bp_other',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0771',
        "description": 'Initiate antihypertensive therapy with RAS inhibitor plus CCB or thiazide/thiazide-like diuretic in diabetic patients with office BP ≥ 140/90 mmHg',
        "condition":   lambda cm, pd, yr: (yr >= 2001 and _get_dm_type(cm) in ('T1DM', 'T2DM') and pd.get('has_hypertension', False) and not _is_hospice(pd)),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB', 'Hypertension_CCB', 'Hypertension_Diuretics'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 Section 5.4.4',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0772',
        "description": 'For diabetic patients with HFpEF/HFmrEF and signs/symptoms of fluid congestion, recommend diuretics to improve symptoms, exercise capacity, and HF hospitalization',
        "condition":   lambda cm, pd, yr: (yr >= 2001 and _get_dm_type(cm) in ('T1DM', 'T2DM') and (_has_hfmref(cm, pd) or _has_hfpef(cm, pd))),
        "any_of_cats": ['Hypertension_Diuretics'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ESC CVD in Diabetes 2023, Recommendation Table 21',
        "domain":      'bp_other',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0773',
        "description": 'For diabetic patients with HFrEF (LVEF≤40%), ARB (candesartan, valsartan) is recommended to reduce HF hospitalization and CV death, especially in ACEi-intolerant patients.',
        "condition":   lambda cm, pd, yr: (yr >= 2001 and _get_dm_type(cm) in ('T1DM', 'T2DM') and _has_hfref(cm, pd)),
        "any_of_cats": ['Hypertension_ARB'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 2016, Section 3 Key evidence (Chunk 3 & 4)',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0774',
        "description": 'For diabetic patients with heart failure and LVEF<40%, recommend beta-blockers (prefer carvedilol or nebivolol) to improve outcomes.',
        "condition":   lambda cm, pd, yr: (yr >= 1999 and _get_dm_type(cm) in ('T1DM', 'T2DM') and _has_hfref(cm, pd)),
        "any_of_cats": ['Hypertension_BetaBlockers'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 (2019 version) 7.1.3.1',
        "domain":      'hf_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0775',
        "description": 'ARB recommended for HFrEF patients who cannot tolerate ARNI or ACEi, with or without diabetes',
        "condition":   lambda cm, pd, yr: (yr >= 2001 and _get_dm_type(cm) in ('T1DM', 'T2DM') and _has_hfref(cm, pd)),
        "any_of_cats": ['Hypertension_ARB'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ESC CVD in Diabetes 2023, 7.4.1.5',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0776',
        "description": 'In hospitalized diabetic patients with HFrEF (LVEF ≤40%), ACE inhibitors are recommended at discharge unless contraindicated or not tolerated.',
        "condition":   lambda cm, pd, yr: (yr >= 2001 and _get_dm_type(cm) == "T2DM" and _has_hfref(cm, pd)),
        "any_of_cats": ['Hypertension_ACEi'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 (2021 version), section 5.3.1',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0777',
        "description": 'Fenofibrate is contraindicated if eGFR <30 mL/min; dose should not exceed 54 mg/day if eGFR 30-59 mL/min in patients with diabetes.',
        "condition":   lambda cm, pd, yr: (lambda egfr: egfr is not None and egfr < 30)(_get_egfr(cm)),
        "any_of_cats": ['Lipid_NonStatins'],
        "exclude_drugs": ['fenofibrate'],
        "grade":       0.75,
        "source":      'ACC-AHA Cholesterol / Cardiovascular 2013, Safety of Fibrates',
        "domain":      'lipid_management',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0778',
        "description": 'For hospitalized patients with diabetes and HFrEF, beta-blocker therapy should be continued or initiated at discharge unless contraindicated (e.g., bradycardia, cardiogenic shock).',
        "condition":   lambda cm, pd, yr: (yr >= 2000 and _get_dm_type(cm) == "T2DM" and _has_hfref(cm, pd)),
        "any_of_cats": ['Hypertension_BetaBlockers'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ACC-AHA Cholesterol/Cardiovascular 2022, Recommendation-Specific Supportive Text (Chunk 1-2)',
        "domain":      'hf_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0779',
        "description": 'ACE inhibitors or ARBs are recommended for patients with type 2 diabetes to reduce incidence of heart failure.',
        "condition":   lambda cm, pd, yr: (yr >= 2000 and _get_dm_type(cm) == "T2DM" and bool(pd.get("has_hypertension"))),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ACC-AHA Heart Failure 2013, Section 7.2.1, Management Strategies for Stage B',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0780',
        "description": 'Recommended use of ACEi in patients with HFrEF to reduce morbidity and mortality (as alternative when ARNi not feasible).',
        "condition":   lambda cm, pd, yr: (yr >= 2001 and _get_dm_type(cm) in ("T1DM", "T2DM") and _has_hfref(cm, pd)),
        "any_of_cats": ['Hypertension_ACEi'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ACC-AHA Cholesterol/Cardiovascular 2022, Section 7.3.1, Recommendation 2',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0781',
        "description": 'For diabetic patients with SIHD and hypertension (BP≥130/80 mmHg), treat with GDMT beta blockers, ACE inhibitors, or ARBs as first-line therapy',
        "condition":   lambda cm, pd, yr: (yr >= 2001 and _get_dm_type(cm) in ("T1DM", "T2DM") and _get_dm_comp(cm, "cardiovascular") and cm.get("has_hypertension")),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB', 'Hypertension_BetaBlockers'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ACC-AHA Cholesterol / Cardiovascular 2017, Section 9.1 Stable Ischemic Heart Disease',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0782',
        "description": 'In diabetic patients with hypertension and CKD stage 3 or higher, or stage 1-2 with albuminuria ≥300 mg/g, treatment with an ARB may be reasonable if ACE inhibitor is not tolerated.',
        "condition":   lambda cm, pd, yr: (yr >= 2001 and _get_dm_type(cm) in ("T1DM", "T2DM") and cm.get("has_hypertension") and (_has_ckd_g3plus(cm) or (lambda u: u is not None and u >= 300)((cm.get("kidney") or {}).get("uacr")))),
        "any_of_cats": ['Hypertension_ARB'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ACC-AHA Cholesterol/Cardiovascular 2017, Section 9.3 Chronic Kidney Disease',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0783',
        "description": 'In diabetic patients with HFrEF, MRA (spironolactone or eplerenone) is recommended to reduce mortality and HF hospitalization, provided eGFR>30 and potassium<5.0.',
        "condition":   lambda cm, pd, yr: (yr >= 2001 and _get_dm_type(cm) in ("T1DM", "T2DM") and _has_hfref(cm, pd) and (lambda egfr: egfr is not None and egfr > 30)(_get_egfr(cm)) and (lambda k: k is not None and k < 5.0)(_get_potassium(pd))),
        "any_of_cats": ['CKD_MRA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ACC-AHA Cholesterol / Cardiovascular 2022, Chunk 1',
        "domain":      'ckd_mra',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0784',
        "description": 'Statins should be considered for LDL-C lowering in adults older than 40 years with T1DM without a history of CVD',
        "condition":   lambda cm, pd, yr: (yr >= 2001 and _get_dm_type(cm) == "T1DM" and not _get_dm_comp(cm, "cardiovascular") and (_get_age(cm, pd) or 0) >= 41),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ESC CVD in Diabetes 2023 Section 11',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0785',
        "description": 'Statins should be considered in patients with T1DM at high CV risk, irrespective of the baseline LDL-C level.',
        "condition":   lambda cm, pd, yr: (yr >= 2001 and _get_dm_type(cm) == "T1DM" and _get_dm_comp(cm, "cardiovascular")),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 | 6.4.1.4 Fibrates | 2019',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0786',
        "description": 'Beta-blockers are recommended for DM patients with heart failure and after acute MI to prevent sudden cardiac death.',
        "condition":   lambda cm, pd, yr: (yr >= 2001 and _get_dm_type(cm) in ("T1DM", "T2DM") and (_get_dm_comp(cm, "heart_failure") or _has_acute_mi(pd))),
        "any_of_cats": ['Hypertension_BetaBlockers'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2013 | 9.4 Recommendations for the management of arrhythmias',
        "domain":      'hf_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0787',
        "description": 'ARB should be administered in patients intolerant of ACEIs with CCS/ACS and systolic LV dysfunction',
        "condition":   lambda cm, pd, yr: (yr >= 2001 and _get_dm_type(cm) in ("T1DM", "T2DM") and _has_acs(pd) and _has_hfref(cm, pd)),
        "any_of_cats": ['Hypertension_ARB'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 | 7.1.3.2 | 2019',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0789',
        "description": 'In HF, MRA (spironolactone/eplerenone) is always preferred over other potassium-sparing diuretics; avoid amiloride, triamterene, and potassium supplements',
        "condition":   lambda cm, pd, yr: (yr >= 2001 and _get_dm_type(cm) == "T2DM" and _get_dm_comp(cm, "heart_failure")),
        "any_of_cats": ['CKD_MRA'],
        "exclude_drugs": ['amiloride', 'triamterene', 'potassium_supplements'],
        "grade":       0.75,
        "source":      'ESC CVD in Diabetes 2023 (2016) | 7.5 Diuretics',
        "domain":      'ckd_mra',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'P2-A: exclude 指向策展中有意移除的药(噻嗪/非DHP-CCB/gemfibrozil/保钾/跨类别),commission 永不触发,孤儿规则',
    },
    {
        "id":          'FINAL-0791',
        "description": 'Loop diuretics should replace thiazide diuretics when eGFR < 30 in hypertension with CKD',
        "condition":   lambda cm, pd, yr: (_has_ckd_g3plus(cm)
            and yr >= 2001 and _get_dm_type(cm) == "T2DM" and cm.get("has_hypertension") and (lambda egfr: egfr is not None and egfr < 30)(_get_egfr(cm))),
        "any_of_cats": ['Hypertension_Diuretics'],
        "exclude_drugs": ['thiazide'],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 2018 | 8.12',
        "domain":      'bp_other',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'P2-A: exclude 指向策展中有意移除的药(噻嗪/非DHP-CCB/gemfibrozil/保钾/跨类别),commission 永不触发,孤儿规则',
    },
    {
        "id":          'FINAL-0792',
        "description": 'Use ACE inhibitor in HF-REF/LVSD with adequate renal function (eGFR >= 30) and normal potassium',
        "condition":   lambda cm, pd, yr: (yr >= 2001 and _get_dm_type(cm) in ("T1DM", "T2DM") and _has_hfref(cm, pd) and (lambda egfr: egfr is not None and egfr >= 30)(_get_egfr(cm)) and (lambda k: k is None or k <= 5.5)(_get_potassium(pd))),
        "any_of_cats": ['Hypertension_ACEi'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 7.2.1 (2016)',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0793',
        "description": 'For hypertensive patients with stable ischemic heart disease (ASCVD), ACEi/ARB/Beta-blockers are recommended as first-line GDMT',
        "condition":   lambda cm, pd, yr: (yr >= 2001 and _get_dm_type(cm) == "T2DM" and cm.get("has_hypertension") is True and _has_cad(pd)),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB', 'Hypertension_BetaBlockers'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ACC-AHA Cardiovascular 2017, Figure 5',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0794',
        "description": 'Beta-blocker therapy is recommended to prevent symptomatic HF in patients with reduced LVEF',
        "condition":   lambda cm, pd, yr: (yr >= 2001 and _get_dm_type(cm) in ("T1DM", "T2DM") and _get_lvef(pd) is not None and _get_lvef(pd) <= 40 and not _is_hospice(pd)),
        "any_of_cats": ['Hypertension_BetaBlockers'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ACC-AHA Cardiovascular 2022 - Section 7.3.6',
        "domain":      'bp_other',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0795',
        "description": 'ACE inhibitors, beta-blockers, ARBs, aldosterone blockers and loop diuretics recommended for symptomatic HF',
        "condition":   lambda cm, pd, yr: (yr >= 2001 and _get_dm_type(cm) in ("T1DM", "T2DM") and _get_dm_comp(cm, "heart_failure") and (lambda e: e is not None and e >= 30)(_get_egfr(cm)) and (lambda k: k is None or k <= 5)(_get_potassium(pd))),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB', 'Hypertension_BetaBlockers', 'Hypertension_Diuretics', 'CKD_MRA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ACC-AHA Cholesterol/Cardiovascular 2003 - Special Considerations',
        "domain":      'hf_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0820',
        "description": 'GLP-1 RA recommended at eGFR>15 in T2DM with CKD for glycemic control, CV risk reduction, and albuminuria benefit',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2017 and _get_dm_comp(cm, "nephropathy") and (lambda egfr: egfr is not None and egfr >= 15)(_get_egfr(cm))),
        "any_of_cats": ['GLP-1RA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023, Chunk 2, Recommendations for patients with chronic kidney disease and diabetes',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0821',
        "description": 'For T2DM patients with ASCVD, prescribe SGLT2i or GLP-1 RA with proven CV benefit at discharge',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2019
            and _get_dm_comp(cm, "cardiovascular")
        ),
        "any_of_cats": ['SGLT2i', 'GLP-1RA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023, Section 5.3.5, Figure 7',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0822',
        "description": 'SGLT2i recommended in T2DM with CKD and eGFR ≥20 to reduce CVD and kidney failure',
        "condition":   lambda cm, pd, yr: (_has_ckd_g3plus(cm)
            and 
            _get_dm_type(cm) == "T2DM"
            and yr >= 2019
            and (lambda egfr: egfr is not None and 20 <= egfr)(_get_egfr(cm))
        ),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023, Section 9',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0823',
        "description": 'RAAS blocker (ACEI or ARB) recommended in DM patients with hypertension, especially with microalbuminuria, albuminuria, proteinuria, or LV hypertrophy',
        "condition":   lambda cm, pd, yr: (yr >= 2019 and _get_dm_type(cm) in ("T1DM", "T2DM") and cm.get("has_hypertension") is True and _has_ckd_with_proteinuria(cm)),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 (2019 version) 6.3.2.3',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0824',
        "description": 'Statins recommended for diabetic patients at high or very high cardiovascular risk.',
        "condition":   lambda cm, pd, yr: (yr >= 2018 and _get_dm_type(cm) in ("T1DM", "T2DM") and _get_dm_comp(cm, "cardiovascular")),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      "ESC CVD in Diabetes 2023 2018, 13 'What to Do' and 'What Not to Do' Messages",
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0825',
        "description": 'Antiplatelet therapy (low-dose aspirin) recommended for secondary prevention in diabetic patients with cardiovascular disease.',
        "condition":   lambda cm, pd, yr: (yr >= 2018 and _get_dm_type(cm) in ("T1DM", "T2DM") and _get_dm_comp(cm, "cardiovascular")),
        "any_of_cats": ['Antiplatelet'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      "ESC CVD in Diabetes 2023 2018, 13 'What to Do' and 'What Not to Do' Messages",
        "domain":      'ascvd_antiplatelet',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0826',
        "description": 'SGLT2 inhibitors are recommended in T2DM patients with CKD to reduce progression of diabetic kidney disease.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2019 and (_has_ckd_g3plus(cm) or _has_albuminuria(cm))),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 (2019 version) - Management of CKD',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0827',
        "description": 'SGLT2 inhibitors are recommended in T2DM patients at high CV risk or with CV disease to prevent HF hospitalizations.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2016
            and _get_dm_comp(cm, "cardiovascular")
            and (lambda egfr: egfr is not None and 30 <= egfr)(_get_egfr(cm))
        ),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 (2021 Chapter 8.4)',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0829',
        "description": 'Ezetimibe recommended in combination with statins if LDL-C target not achieved in very high risk T2DM patients',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2016 and _get_dm_comp(cm, "cardiovascular") and (lambda l: l is not None and l > 70)((cm.get("lipids") or {}).get("ldl"))),
        "any_of_cats": ['Lipid_NonStatins'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 2023 | 5.5.3.2',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0830',
        "description": 'ACE inhibitor is recommended for all diabetic patients with symptomatic systolic HF (NYHA II-IV, EF≤40%) in addition to a beta-blocker to reduce HF hospitalization and premature death.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2016
            and _has_hfref(cm, pd)
        ),
        "any_of_cats": ['Hypertension_ACEi'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 (2016 version) section 7.2.1',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0832',
        "description": 'An MRA is recommended for all diabetic patients with persisting symptoms (NYHA II-IV) and EF ≤35% despite treatment with ACE inhibitor and beta-blocker to reduce HF hospitalization and premature death.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2016 and _get_dm_comp(cm, "heart_failure") and (lambda lvef: lvef is not None and lvef <= 35)(_get_lvef(pd))),
        "any_of_cats": ['CKD_MRA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 (2016 version) section 7.2.1',
        "domain":      'ckd_mra',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0833',
        "description": 'Beta-blockers and MRA recommended for diabetic patients with HFrEF to reduce risk of sudden death',
        "condition":   lambda cm, pd, yr: (yr >= 2016 and _get_dm_type(cm) in ("T1DM", "T2DM") and _has_hfref(cm, pd) and (lambda e: e is not None and e >= 30)(_get_egfr(cm)) and (lambda k: k is None or k <= 5)(_get_potassium(pd))),
        "any_of_cats": ['Hypertension_BetaBlockers', 'CKD_MRA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 (2016 edition), Section 10.2 Ventricular arrhythmias',
        "domain":      'hf_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0834',
        "description": 'For diabetic patients requiring antihypertensive therapy, ACE inhibitors, ARBs, beta-blockers, CCBs, or diuretics are indicated as the basis of treatment',
        "condition":   lambda cm, pd, yr: (yr >= 2018 and _get_dm_type(cm) in ("T1DM", "T2DM") and cm.get("has_hypertension") == True),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB', 'Hypertension_BetaBlockers', 'Hypertension_CCB', 'Hypertension_Diuretics'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 2018, Section 7.6.1',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0835',
        "description": 'For diabetic patients with angina, post-MI, heart failure, or requiring heart rate control, beta-blockers are recommended in combination with other antihypertensives',
        "condition":   lambda cm, pd, yr: (yr >= 2018 and _get_dm_type(cm) in ("T1DM", "T2DM") and (_has_cad(pd) or _has_prior_mi(pd) or _get_dm_comp(cm, "heart_failure"))),
        "any_of_cats": ['Hypertension_BetaBlockers'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 2018, Section 7.6.1',
        "domain":      'hf_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0837',
        "description": 'Statins recommended in patients with or at high-risk of CAD, regardless of LV systolic dysfunction, to prevent/delay HF and prolong life',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2016
            and _has_cad(pd)
        ),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 (2016 version), Section 6',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0838',
        "description": 'Empagliflozin should be considered in patients with T2DM to prevent or delay HF and prolong life',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2016 and (_get_dm_comp(cm, "cardiovascular") or _has_ckd_g3plus(cm) or _has_albuminuria(cm) or _get_dm_comp(cm, "heart_failure") or _has_hfref(cm, pd))),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ESC CVD in Diabetes 2023 (2016 version), Section 6',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0839',
        "description": 'ACE-I recommended in patients with asymptomatic LV systolic dysfunction and history of MI to prevent/delay HF and prolong life',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2016 and _has_prior_mi(pd) and _get_lvef(pd) is not None and _get_lvef(pd) <= 40),
        "any_of_cats": ['Hypertension_ACEi'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 (2016 version), Section 6',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0840',
        "description": 'ACE-I recommended in patients with asymptomatic LV systolic dysfunction without MI history to prevent/delay HF',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2016 and not _has_prior_mi(pd) and _get_lvef(pd) is not None and _get_lvef(pd) <= 40),
        "any_of_cats": ['Hypertension_ACEi'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ESC CVD in Diabetes 2023 (2016 version), Section 6',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0842',
        "description": 'Eplerenone recommended in patients with ACS and EF≤40% to reduce death and CV hospitalization',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) in ("T1DM", "T2DM") and yr >= 2016 and _has_acs(pd) and _get_lvef(pd) is not None and _get_lvef(pd) <= 40),
        "any_of_cats": ['CKD_MRA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 (2016 version), Section 12.2.2',
        "domain":      'ckd_mra',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0843',
        "description": 'For diabetes patients with HFrEF and hypertension, ACEI/ARB, beta-blocker, or MRA is recommended as first-line antihypertensive therapy.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) in ("T1DM", "T2DM") and yr >= 2016 and _has_hfref(cm, pd) and cm.get("has_hypertension") is True),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB', 'Hypertension_BetaBlockers', 'CKD_MRA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 (2016) Section 11.11',
        "domain":      'hf_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0844',
        "description": 'For diabetes patients with HFrEF and persistent hypertension despite ACEI/ARB, beta-blocker, and MRA, a thiazide or loop diuretic is recommended.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) in ("T1DM", "T2DM") and yr >= 2016 and _has_hfref(cm, pd) and cm.get("has_hypertension") is True),
        "any_of_cats": ['Hypertension_Diuretics'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 (2016) Section 11.11',
        "domain":      'bp_other',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0845',
        "description": 'For diabetes patients with HFrEF and resistant hypertension, amlodipine is recommended.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) in ("T1DM", "T2DM") and yr >= 2016 and _has_hfref(cm, pd) and cm.get("has_hypertension") is True),
        "any_of_cats": ['Hypertension_CCB'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 (2016) Section 11.11',
        "domain":      'bp_other',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0847',
        "description": 'For T2DM patients with heart failure (LVEF≤40%), recommend MRA (eplerenone or spironolactone) at discharge.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2016 and _has_hfref(cm, pd) and (lambda egfr: egfr is not None and egfr > 30)(_get_egfr(cm)) and (lambda k: k is not None and k < 5)(_get_potassium(pd))),
        "any_of_cats": ['CKD_MRA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 (based on 2016 guideline), section 7.2.2 Mineralocorticoid/aldosterone receptor antagonists',
        "domain":      'ckd_mra',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0849',
        "description": 'Antiplatelet therapy is recommended in patients with DM and symptomatic LEAD',
        "condition":   lambda cm, pd, yr: (yr >= 2019 and _get_dm_type(cm) in ("T1DM", "T2DM") and _has_pad(pd)),
        "any_of_cats": ['Antiplatelet'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 (2019) 11.1.1 LEAD management',
        "domain":      'ascvd_antiplatelet',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0850',
        "description": 'Statin therapy targeting LDL-C <1.4 mmol/L (55 mg/dL) or at least 50% reduction is recommended in DM patients with LEAD',
        "condition":   lambda cm, pd, yr: (yr >= 2019 and _get_dm_type(cm) in ("T1DM", "T2DM") and _has_pad(pd)),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ESC CVD in Diabetes 2023 (2019) 11.1.1 LEAD management',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0851',
        "description": 'In hypertensive diabetic patients with lower extremity arterial disease (LEAD), initial combination therapy with a RAS blocker (ACEi/ARB) plus a CCB or diuretic should be considered for blood pressure lowering.',
        "condition":   lambda cm, pd, yr: (yr >= 2018 and _get_dm_type(cm) in ("T1DM", "T2DM") and _has_pad(pd) and cm.get("has_hypertension") is True),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB', 'Hypertension_CCB', 'Hypertension_Diuretics'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ESC CVD in Diabetes 2023 (based on 2018 version), Section 8.17.3 Lower extremity arterial disease',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0852',
        "description": 'Add ezetimibe to statin for DM patients with recent ACS if LDL-C not <1.4 mmol/L on statin alone',
        "condition":   lambda cm, pd, yr: (yr >= 2019 and _get_dm_type(cm) in ("T1DM", "T2DM") and _has_acs(pd) and (cm.get("lipids") or {}).get("ldl") is not None and (cm.get("lipids") or {}).get("ldl") >= 1.4),
        "any_of_cats": ['Lipid_NonStatins'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ESC CVD in Diabetes 2023 (2019 version) Section 6.4.1.2',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0853',
        "description": 'Fibrates may be considered in DM patients with high triglycerides (≥2.3 mmol/L) who are statin intolerant, or with elevated TG and low HDL-C on statin',
        "condition":   lambda cm, pd, yr: (yr >= 2019 and _get_dm_type(cm) in ("T1DM", "T2DM") and (cm.get("lipids") or {}).get("triglycerides") is not None and (cm.get("lipids") or {}).get("triglycerides") >= 2.3),
        "any_of_cats": ['Lipid_NonStatins'],
        "exclude_drugs": ['gemfibrozil'],
        "grade":       0.75,
        "source":      'ESC CVD in Diabetes 2023 (2019 version) Section 6.4.1.4',
        "domain":      'lipid_management',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'P2-A: exclude 指向策展中有意移除的药(噻嗪/非DHP-CCB/gemfibrozil/保钾/跨类别),commission 永不触发,孤儿规则',
    },
    {
        "id":          'FINAL-0854',
        "description": 'Beta-blocker should be part of multifactorial management in DM patients with known CAD',
        "condition":   lambda cm, pd, yr: (yr >= 2019 and _get_dm_type(cm) in ("T1DM", "T2DM") and _has_cad(pd)),
        "any_of_cats": ['Hypertension_BetaBlockers'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ESC CVD in Diabetes 2023 (2019 version) Section 6.6.1',
        "domain":      'bp_other',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0856',
        "description": 'For diabetic patients post-myocardial infarction or with chronic HFrEF, ACEi or ARB is recommended',
        "condition":   lambda cm, pd, yr: (yr >= 2018 and _get_dm_type(cm) in ("T1DM", "T2DM") and (_has_prior_mi(pd) or _has_hfref(cm, pd))),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ESC CVD in Diabetes 2023 (2018 version), 7.5.1.1 Blockers of the renin-angiotensin system',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0857',
        "description": 'For hypertensive diabetic patients with CKD, recommend initial combination therapy: RAS blocker (ACEi/ARB) plus CCB or diuretic.',
        "condition":   lambda cm, pd, yr: (yr >= 2018 and _get_dm_type(cm) in ("T1DM", "T2DM") and cm.get("has_hypertension") is True and _has_ckd_g3plus(cm)),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB', 'Hypertension_CCB', 'Hypertension_Diuretics'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ESC CVD in Diabetes 2023 (2018) - 8.12 Hypertension and chronic kidney disease',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0858',
        "description": 'For hypertensive diabetic patients with CKD and eGFR<30 ml/min/1.73m², loop diuretics should replace thiazide diuretics.',
        "condition":   lambda cm, pd, yr: (_has_ckd_g3plus(cm)
            and yr >= 2018 and _get_dm_type(cm) in ("T1DM", "T2DM") and cm.get("has_hypertension") is True and (lambda egfr: egfr is not None and egfr < 30)(_get_egfr(cm))),
        "any_of_cats": ['Hypertension_Diuretics'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ESC CVD in Diabetes 2023 (2018) - 8.12 Hypertension and chronic kidney disease',
        "domain":      'bp_other',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0860',
        "description": 'Aspirin 75-100 mg/day may be considered for primary prevention in DM patients at high/very high CV risk without ASCVD',
        "condition":   lambda cm, pd, yr: (yr >= 2019 and _get_dm_type(cm) in ("T1DM", "T2DM") and not _get_dm_comp(cm, "cardiovascular")),
        "any_of_cats": ['Antiplatelet'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 (2019 version), section 6.5.1.1 Primary prevention',
        "domain":      'ascvd_antiplatelet',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0861',
        "description": 'For T2DM patients with elevated triglycerides despite statin therapy, consider icosapent ethyl 2g bid to reduce cardiovascular events.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2019 and (lambda tg: tg is not None and tg > 150)((cm.get("lipids") or {}).get("triglycerides"))),
        "any_of_cats": ['Lipid_NonStatins'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023, section 5.5.2.4',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0862',
        "description": 'In patients with heart failure and CAD who are already receiving a statin, continuation may be considered.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) in ("T1DM", "T2DM") and yr >= 2016 and _has_cad(pd) and _get_dm_comp(cm, "heart_failure")),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ESC CVD in Diabetes 2023 (2016 version), Chunk 1',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0863',
        "description": 'In diabetic patients with hypertension, if eGFR <30 ml/min, loop diuretics (e.g., furosemide) should replace thiazide/thiazide-like diuretics for blood pressure control',
        "condition":   lambda cm, pd, yr: (yr >= 2018 and _get_dm_type(cm) in ("T1DM", "T2DM") and cm.get("has_hypertension") is True and (lambda egfr: egfr is not None and egfr < 30)(_get_egfr(cm))),
        "any_of_cats": ['Hypertension_Diuretics'],
        "exclude_drugs": ['thiazide', 'thiazide-like'],
        "grade":       0.75,
        "source":      'ESC CVD in Diabetes 2023 (2018 version) Section 7.5.1.3',
        "domain":      'bp_other',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'P2-A: exclude 指向策展中有意移除的药(噻嗪/非DHP-CCB/gemfibrozil/保钾/跨类别),commission 永不触发,孤儿规则',
    },
    {
        "id":          'FINAL-0865',
        "description": 'In hypertensive patients undergoing non-cardiac surgery, transient preoperative discontinuation of RAS blockers (ACEi/ARB) should be considered.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2018
        ),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 (2018 edition), Section 8.21 Perioperative management of hypertension',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'judge:scope_out_of_discharge',
    },
    {
        "id":          'FINAL-0866',
        "description": 'In diabetic patients with prior MI (1-3 years) who have tolerated DAPT without major bleeding, consider extended DAPT up to 3 years with reduced-dose ticagrelor (60mg bid)',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2016 and _has_prior_mi(pd) and not _has_gi_bleed_hx(pd)),
        "any_of_cats": ['Antiplatelet'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ESC CVD in Diabetes 2023, Section 5.6.2.2 Acute coronary syndrome',
        "domain":      'ascvd_antiplatelet',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0867',
        "description": 'Continue RAAS blockers (ACEi/ARB/MRA) in hospitalized diabetic patients with heart failure, unless marked GFR decline or hyperkalemia occurs.',
        "condition":   lambda cm, pd, yr: (yr >= 2016 and _get_dm_type(cm) in ("T1DM", "T2DM") and _get_dm_comp(cm, "heart_failure") and not (cm.get("kidney", {}).get("aki_suspected") or (lambda k: k is not None and k > 5.5)(_get_potassium(pd)))),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB', 'CKD_MRA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 (2016 version), Section 11.15',
        "domain":      'ckd_mra',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0868',
        "description": 'In patients with chronic HF (HFrEF) without established ASCVD, initiation of statins is not recommended for HF management',
        "condition":   lambda cm, pd, yr: (
            yr >= 2016
            and not _get_dm_comp(cm, "cardiovascular")
            and _has_hfref(cm, pd)
        ),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ESC CVD in Diabetes 2023 (2016 version) 7.5.1',
        "domain":      'lipid_management',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'audit:anti-signal — "statin not for HF management" != contraindicated; diabetes itself is a statin indication → flagging as commission penalises appropriate statin use',
    },
    {
        "id":          'FINAL-0869',
        "description": 'In patients with chronic HF who already receive a statin because of underlying CAD or hyperlipidaemia, continuation of statin therapy should be considered',
        "condition":   lambda cm, pd, yr: (yr >= 2016 and _get_dm_type(cm) in ("T1DM", "T2DM") and _get_dm_comp(cm, "heart_failure") and (_has_cad(pd) or (lambda l: l is not None and l > 100)((cm.get("lipids") or {}).get("ldl")))),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ESC CVD in Diabetes 2023 (2016 version) 7.5.1',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0870',
        "description": 'For diabetic patients with resistant hypertension, if eGFR ≥45 ml/min and potassium ≤4.5 mmol/L, consider MRA (spironolactone) as fourth-line antihypertensive',
        "condition":   lambda cm, pd, yr: (yr >= 2018 and _get_dm_type(cm) in ("T1DM", "T2DM") and (lambda egfr: egfr is not None and egfr >= 45)(_get_egfr(cm)) and (lambda k: k is not None and k <= 4.5)(_get_potassium(pd))),
        "any_of_cats": ['CKD_MRA'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ESC CVD in Diabetes 2023 (2018 version) 8.1.4 Treatment of resistant hypertension',
        "domain":      'ckd_mra',
        "polarity":    'ALLOW',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0873',
        "description": 'For inpatients with T2DM and history of ischemic stroke or TIA, discharge prescription should include a RAS blocker (ACEi/ARB) for secondary stroke prevention',
        "condition":   lambda cm, pd, yr: (yr >= 2018 and _get_dm_type(cm) in ("T1DM", "T2DM") and cm.get("has_hypertension") is True and (_has_stroke_hx(pd) or _has_tia(pd))),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 2018, 8.15.3 Previous stroke or transient ischaemic attack',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0876',
        "description": 'For hospitalized T2DM patients with HFrEF (symptomatic, LVEF≤40%), with adequate renal function (eGFR≥30) and normal potassium, prescribe ACEi at discharge.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2016 and _has_hfref(cm, pd) and (lambda egfr: egfr is not None and egfr >= 30)(_get_egfr(cm)) and (lambda k: k is not None and k <= 5.0)(_get_potassium(pd))),
        "any_of_cats": ['Hypertension_ACEi'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 (2016 version), section 7.2.1',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0877',
        "description": 'For hospitalized T2DM patients with HFrEF (symptomatic, LVEF≤40%), prescribe beta-blocker at discharge in addition to ACEi.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2016
            and _has_hfref(cm, pd)
        ),
        "any_of_cats": ['Hypertension_BetaBlockers'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 (2016 version), section 7.2.1',
        "domain":      'hf_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0878',
        "description": 'For T2DM patients with asymptomatic LV systolic dysfunction (LVEF≤40%, no HF symptoms) or post-AMI with HF/LV dysfunction, prescribe ACEi at discharge.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2016 and (lambda egfr: egfr is not None and egfr >= 30)(_get_egfr(cm)) and ((not _has_hfref(cm, pd) and (lambda lvef: lvef is not None and lvef <= 40)(_get_lvef(pd))) or (_has_acute_mi(pd) and _has_hfref(cm, pd)))),
        "any_of_cats": ['Hypertension_ACEi'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 (2016 version), section 7.2.1 (SOLVD-Prevention, SAVE, AIRE, TRACE trials)',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0879',
        "description": 'Recommend MRA in diabetic patients with heart failure, adequate renal function, and normal serum potassium',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2016 and _get_dm_comp(cm, "heart_failure") and (lambda egfr: egfr is not None and 30 <= egfr)(_get_egfr(cm)) and (lambda k: k is not None and k <= 5.0)(_get_potassium(pd))),
        "any_of_cats": ['CKD_MRA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 (2016 version), Section 7.2.2',
        "domain":      'ckd_mra',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0880',
        "description": 'In diabetic patients with reduced ejection fraction heart failure (HFrEF), MRA should be started as soon as possible if renal function and potassium permit.',
        "condition":   lambda cm, pd, yr: (yr >= 2016 and _get_dm_type(cm) in ("T1DM", "T2DM") and _has_hfref(cm, pd) and (lambda egfr: egfr is not None and 30 <= egfr)(_get_egfr(cm)) and (lambda k: k is not None and k <= 5.0)(_get_potassium(pd))),
        "any_of_cats": ['CKD_MRA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 (2016 version), Section Mineralocorticoid receptor antagonist',
        "domain":      'ckd_mra',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0881',
        "description": 'For diabetic patients with HFrEF, first-line antihypertensive therapy includes ACEi/ARB, MRA, diuretics, or GDMT beta-blockers; avoid nondihydropyridine CCBs',
        "condition":   lambda cm, pd, yr: (
            yr >= 2017
            and _has_hfref(cm, pd)
        ),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB', 'CKD_MRA', 'Hypertension_Diuretics', 'Hypertension_BetaBlockers'],
        "exclude_drugs": ['verapamil', 'diltiazem'],
        "grade":       1.0,
        "source":      'ACC-AHA Cholesterol/Cardiovascular 2017 (Recommendation-Specific Supportive Text)',
        "domain":      'hf_management',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'P2-A: exclude 指向策展中有意移除的药(噻嗪/非DHP-CCB/gemfibrozil/保钾/跨类别),commission 永不触发,孤儿规则',
    },
    {
        "id":          'FINAL-0883',
        "description": 'For hospitalized diabetic patients with HFrEF (LVEF≤40%) who are intolerant to ACE inhibitors due to cough or angioedema, ARB is recommended to reduce morbidity and mortality (Class I, Level B-R)',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2017 and _has_hfref(cm, pd)),
        "any_of_cats": ['Hypertension_ARB'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ACC-AHA Cholesterol/Cardiovascular 2017, HFrEF ARB recommendation (based on 2013 update repeated in this section)',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0884',
        "description": 'For hospitalized diabetic patients with chronic symptomatic HFrEF (LVEF≤40%, NYHA II/III) who tolerate an ACE inhibitor or ARB, replacement by ARNI (sacubitril/valsartan) is recommended to further reduce morbidity and mortality (Class I, Level B-R)',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2017 and _has_hfref(cm, pd)),
        "any_of_cats": ['Hypertension_ARNI'],
        "exclude_drugs": ['ace_inhibitors', 'arbs'],
        "grade":       0.75,
        "source":      'ACC-AHA Cholesterol/Cardiovascular 2017, ARNI recommendation (NEW, based on PARADIGM-HF 2014)',
        "domain":      'hf_management',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'P2-A: exclude 指向策展中有意移除的药(噻嗪/非DHP-CCB/gemfibrozil/保钾/跨类别),commission 永不触发,孤儿规则',
    },
    {
        "id":          'FINAL-0885',
        "description": 'In adults aged 40-75 years with CKD not treated with dialysis or kidney transplantation, LDL-C 70-189 mg/dL, and 10-year ASCVD risk ≥7.5%, initiation of moderate-intensity statin or moderate-intensity statin plus ezetimibe can be useful.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) in ("T1DM", "T2DM") and yr >= 2018 and 40 <= (_get_age(cm, pd) or 0) <= 75 and not _has_esrd_dialysis(pd) and (_has_ckd_g3plus(cm) or _has_albuminuria(cm)) and (lambda l: l is not None and 70 <= l <= 189)((cm.get("lipids") or {}).get("ldl"))),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ACC-AHA Cholesterol/Cardiovascular 2018, Section 4.5.4 Adults With CKD',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0886',
        "description": 'In CKD patients with hypertension (including diabetes), initiate or add-on ACEi or ARB to improve kidney outcomes',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) in ("T1DM", "T2DM") and yr >= 2014 and (_get_age(cm, pd) or 0) >= 18 and (_has_ckd_g3plus(cm) or _has_albuminuria(cm)) and cm.get("has_hypertension")),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ACC-AHA Cholesterol/Cardiovascular 2014 Recommendation 8',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0887',
        "description": 'For diabetic patients with ASCVD who cannot tolerate high-intensity statin, moderate-intensity statin plus ezetimibe may be reasonable.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) in ("T1DM", "T2DM") and yr >= 2018 and _get_dm_comp(cm, "cardiovascular")),
        "any_of_cats": ['Lipid_Statins', 'Lipid_NonStatins'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ACC-AHA Cholesterol / Cardiovascular 2018, Recommendation-Specific Supportive Text, Chunk 2',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0888',
        "description": 'For diabetic patients with severe hypercholesterolemia (LDL-C≥190mg/dL), maximally tolerated statin therapy is recommended.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) in ("T1DM", "T2DM") and yr >= 2018 and (lambda l: l is not None and l >= 190)((cm.get("lipids") or {}).get("ldl"))),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ACC-AHA Cholesterol / Cardiovascular 2018, Recommendation-Specific Supportive Text, Chunk 3',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0889',
        "description": 'For hospitalized diabetic patients with CKD (eGFR<60 mL/min or UACR≥30 mg/g), statin therapy should be prescribed at discharge.',
        "condition":   lambda cm, pd, yr: (_has_ckd_g3plus(cm)
            and _get_dm_type(cm) in ("T1DM", "T2DM") and yr >= 2018 and ((lambda egfr: egfr is not None and egfr < 60)(_get_egfr(cm)) or (lambda u: u is not None and u >= 30)((cm.get("kidney") or {}).get("uacr")))),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ACC-AHA Cholesterol / Cardiovascular 2018, Section 4.4.2 Primary Prevention Adults 40-75 Years, CKD supportive text',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0890',
        "description": 'In appropriately selected symptomatic HFpEF patients with EF≥45%, elevated BNP or HF hospitalization within 1 year, eGFR>30, K<5.0, spironolactone may be considered.',
        "condition":   lambda cm, pd, yr: (yr >= 2017 and _get_dm_type(cm) in ("T1DM", "T2DM") and _has_hfpef(cm, pd) and (lambda egfr: egfr is not None and egfr > 30)(_get_egfr(cm)) and (lambda k: k is None or k < 5.0)(_get_potassium(pd))),
        "any_of_cats": ['CKD_MRA'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ACC-AHA Cholesterol/Cardiovascular 2017, HFpEF management, TOPCAT post-hoc analysis',
        "domain":      'ckd_mra',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0891',
        "description": 'For diabetic patients with CKD and albuminuria (UACR ≥300 mg/g), discharge prescription should include ACEi (first-line) or ARB (alternative).',
        "condition":   lambda cm, pd, yr: (yr >= 2017 and _get_dm_type(cm) in ("T1DM", "T2DM") and _has_ckd_g3plus(cm) and (lambda u: u is not None and u >= 300)((cm.get("kidney") or {}).get("uacr"))),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ACC-AHA Cholesterol / Cardiovascular 2017, Section 9.3, Figure 6',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0892',
        "description": 'In adults 40-75 years of age with diabetes, regardless of estimated 10-year ASCVD risk, moderate-intensity statin therapy is indicated.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2019 and 40 <= (_get_age(cm, pd) or 0) <= 75),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ACC-AHA Cholesterol / Cardiovascular 2019, Section 4.3 (Adults With High Blood Cholesterol)',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0893',
        "description": 'In adults 40-70 years of age with diabetes, low-dose aspirin may be considered for primary prevention of ASCVD (Class IIb).',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2019 and not _get_dm_comp(cm, "cardiovascular") and 40 <= (_get_age(cm, pd) or 0) <= 70),
        "any_of_cats": ['Antiplatelet'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ACC-AHA Cholesterol / Cardiovascular 2019, Section 4.6 (Prophylactic Aspirin)',
        "domain":      'ascvd_antiplatelet',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0894',
        "description": 'For diabetic patients aged 40-75 years with LDL-C >=70 mg/dL, initiate moderate-intensity statin; for high-risk patients (multiple risk factors or age 50-75), high-intensity statin is reasonable.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2018 and 40 <= (_get_age(cm, pd) or 0) <= 75 and (lambda l: l is not None and l >= 70)((cm.get("lipids") or {}).get("ldl"))),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ACC-AHA Cholesterol Guideline 2018, Top 10 Take-Home Messages, #5',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0895',
        "description": 'Initiate or add-on ACEi or ARB for CKD patients with hypertension (applicable to hospitalized DM patients with CKD)',
        "condition":   lambda cm, pd, yr: (_has_ckd_g3plus(cm)
            and yr >= 2014 and _get_dm_type(cm) in ("T1DM", "T2DM") and cm.get("has_hypertension") and ((lambda egfr: egfr is not None and 0 <= egfr and egfr <= 59)(_get_egfr(cm)) or (lambda uacr: uacr is not None and uacr >= 31)((cm.get("kidney") or {}).get("uacr"))) and (_get_age(cm, pd) or 0) >= 18),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ACC-AHA Cholesterol/Cardiovascular 2014, Recommendation 8, Level B',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0896',
        "description": 'Diuretics recommended for relief of volume overload symptoms in diabetic patients with HFpEF',
        "condition":   lambda cm, pd, yr: (yr >= 2017 and _get_dm_type(cm) in ("T1DM", "T2DM") and _has_hfpef(cm, pd)),
        "any_of_cats": ['Hypertension_Diuretics'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ACC-AHA Cholesterol/Cardiovascular 2017, Stage C HFpEF Recommendations',
        "domain":      'bp_other',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0897',
        "description": 'Beta-blockers, ACE inhibitors, or ARBs are reasonable for blood pressure control in diabetic patients with HFpEF and hypertension',
        "condition":   lambda cm, pd, yr: (yr >= 2017 and _get_dm_type(cm) in ("T1DM", "T2DM") and _has_hfpef(cm, pd) and cm.get("has_hypertension")),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB', 'Hypertension_BetaBlockers'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ACC-AHA Cholesterol/Cardiovascular 2017, Stage C HFpEF Recommendations',
        "domain":      'hf_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0898',
        "description": 'SGLT2i recommended for T2DM patients with established CVD or at high CVD risk to prevent HF hospitalization.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2016
            and _get_dm_comp(cm, "cardiovascular")
            and not _get_dm_comp(cm, "heart_failure")
            and (lambda egfr: egfr is not None and 25 <= egfr)(_get_egfr(cm))
        ),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ACC-AHA Cholesterol / Cardiovascular 2022, Recommendation-Specific Supportive Text (Chunk 2), Figure 5',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0900',
        "description": 'In diabetic patients with severe hypertriglyceridemia (TG ≥500 mg/dL), consider adding fibrates (prefer fenofibrate over gemfibrozil) or omega-3 fatty acids to prevent pancreatitis when lifestyle modification is insufficient.',
        "condition":   lambda cm, pd, yr: (yr >= 2018 and _get_dm_type(cm) in ("T1DM", "T2DM") and (lambda tg: tg is not None and tg >= 500)((cm.get("lipids") or {}).get("triglycerides"))),
        "any_of_cats": ['Lipid_NonStatins'],
        "exclude_drugs": ['gemfibrozil'],
        "grade":       0.75,
        "source":      'ACC-AHA Cholesterol/Cardiovascular 2018, Section 4.5.2 (Recommendation-Specific Supportive Text)',
        "domain":      'lipid_management',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'P2-A: exclude 指向策展中有意移除的药(噻嗪/非DHP-CCB/gemfibrozil/保钾/跨类别),commission 永不触发,孤儿规则',
    },
    {
        "id":          'FINAL-0901',
        "description": 'Initiate first-line antihypertensive agents (diuretics, ACEi, ARB, or CCB) in adults with DM and hypertension at BP ≥130/80 mmHg',
        "condition":   lambda cm, pd, yr: (yr >= 2017 and _get_dm_type(cm) in ("T1DM", "T2DM") and cm.get("has_hypertension") is True),
        "any_of_cats": ['Hypertension_Diuretics', 'Hypertension_ACEi', 'Hypertension_ARB', 'Hypertension_CCB'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ACC-AHA Cholesterol/Cardiovascular 2017 - 9.6 Diabetes Mellitus',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0902',
        "description": 'In adults with HFpEF and volume overload, diuretics should be prescribed to control hypertension',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2017 and _has_hfpef(cm, pd)),
        "any_of_cats": ['Hypertension_Diuretics'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ACC-AHA Cholesterol/Cardiovascular 2017 - 9.2.2 Heart Failure With Preserved Ejection Fraction',
        "domain":      'bp_other',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0903',
        "description": 'In HFpEF with persistent hypertension after volume control, ACE inhibitors or ARBs should be prescribed to achieve SBP <130 mm Hg',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2017 and _has_hfpef(cm, pd) and cm.get("has_hypertension") is True),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ACC-AHA Cholesterol/Cardiovascular 2017 - 9.2.2 Heart Failure With Preserved Ejection Fraction',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0904',
        "description": 'In HFpEF with persistent hypertension after volume control, beta blockers should be prescribed to achieve SBP <130 mm Hg',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2017 and _has_hfpef(cm, pd) and cm.get("has_hypertension") is True),
        "any_of_cats": ['Hypertension_BetaBlockers'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ACC-AHA Cholesterol/Cardiovascular 2017 - 9.2.2 Heart Failure With Preserved Ejection Fraction',
        "domain":      'hf_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0905',
        "description": 'For non-black patients with diabetes and hypertension, initial antihypertensive therapy at discharge should include a thiazide diuretic, CCB, ACEi, or ARB.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2014 and cm.get("has_hypertension") is True),
        "any_of_cats": ['Hypertension_Diuretics', 'Hypertension_CCB', 'Hypertension_ACEi', 'Hypertension_ARB'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ACC-AHA Cholesterol / Cardiovascular 2014, Recommendation 6, Level B',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0906',
        "description": 'For patients with diabetes and CVD without HF, prefer SGLT2i to improve survival and reduce HF hospitalization',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2016
            and _get_dm_comp(cm, "cardiovascular")
            and not _get_dm_comp(cm, "heart_failure")
        ),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ACC-AHA Cholesterol / Cardiovascular 2022 Synopsis',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0907',
        "description": 'For diabetic patients with prior hypertension who experience stroke or TIA, antihypertensive treatment should be restarted after the first few days to reduce recurrent stroke risk',
        "condition":   lambda cm, pd, yr: (yr >= 2017 and _get_dm_type(cm) in ("T1DM", "T2DM") and cm.get("has_hypertension") and (_has_stroke_hx(pd) or _has_tia(pd))),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB', 'Hypertension_Diuretics'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ACC-AHA Cholesterol / Cardiovascular 2017, Section 9.4.3 Secondary Stroke Prevention',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0909',
        "description": 'SGLT2i is recommended in T2DM and CKD with eGFR 30 to <90 to reduce renal endpoints',
        "condition":   lambda cm, pd, yr: (_has_ckd_g3plus(cm)
            and _get_dm_type(cm) == "T2DM" and yr >= 2019 and (lambda egfr: egfr is not None and 30 <= egfr < 90)(_get_egfr(cm))),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ESC CVD in Diabetes 2023 (2019) - 11.1.2 New approaches to nephroprotection',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0910',
        "description": 'GLP-1 RA (liraglutide/semaglutide) considered for T2DM at eGFR>30 to lower risk of renal endpoints',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2017 and (lambda egfr: egfr is not None and egfr > 30)(_get_egfr(cm))),
        "any_of_cats": ['GLP-1RA'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ESC CVD in Diabetes 2023 | Section 9 | Chronic kidney disease and diabetes',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0911',
        "description": 'An ARB is recommended to reduce the risk of HF hospitalization and cardiovascular death in symptomatic patients unable to tolerate an ACE-I.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2016 and _get_dm_comp(cm, "heart_failure")),
        "any_of_cats": ['Hypertension_ARB'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2016 - Other pharmacological treatments',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0912',
        "description": 'Treatment with beta-blocker and MRA reduces the risk of sudden death and is recommended for patients with HFrEF and ventricular arrhythmias.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) in ('T1DM', 'T2DM') and yr >= 2016 and _has_hfref(cm, pd) and (lambda e: e is not None and e >= 30)(_get_egfr(cm)) and (lambda k: k is None or k <= 5)(_get_potassium(pd))),
        "any_of_cats": ['Hypertension_BetaBlockers', 'CKD_MRA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2016 - 10.2 Ventricular arrhythmias',
        "domain":      'hf_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0914',
        "description": 'ARBs are recommended only as an alternative in patients intolerant of an ACEI',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == 'T2DM' and yr >= 2016 and _get_dm_comp(cm, 'heart_failure')),
        "any_of_cats": ['Hypertension_ARB'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 2016 | 7.3.4 Angiotensin II type I receptor blockers',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0915',
        "description": 'Combination of loop diuretic with thiazide or spironolactone may be considered for resistant oedema',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) in ('T1DM', 'T2DM') and yr >= 2016 and _get_dm_comp(cm, 'heart_failure') and (lambda e: e is not None and e >= 30)(_get_egfr(cm)) and (lambda k: k is None or k <= 5)(_get_potassium(pd))),
        "any_of_cats": ['Hypertension_Diuretics', 'CKD_MRA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 (2016) | Pharmacological therapy',
        "domain":      'ckd_mra',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0916',
        "description": 'ARB is recommended for HF patients with EF≤40% unable to tolerate ACEi to reduce HF hospitalization and death',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) in ('T1DM', 'T2DM') and yr >= 2016 and _has_hfref(cm, pd)),
        "any_of_cats": ['Hypertension_ARB'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 (2016 chunk) | Other treatments with less-certain benefits',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0917',
        "description": 'Most CCBs should not be used in HF due to negative inotropic effect, except amlodipine and felodipine',
        "condition":   lambda cm, pd, yr: (
            yr >= 2016
            and _get_dm_comp(cm, "heart_failure")
        ),
        "any_of_cats": ['Hypertension_CCB'],
        "exclude_drugs": ['non_amlodipine_non_felodipine_ccb'],
        "grade":       0.75,
        "source":      'ESC CVD in Diabetes 2023 (2016 chunk) | 7.4 Treatments not recommended',
        "domain":      'bp_other',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'audit:avoid_uncheckable',
    },
    {
        "id":          'FINAL-0918',
        "description": 'In patients with LEAD, antihypertensive treatment should be complemented by statin and antiplatelet therapy',
        "condition":   lambda cm, pd, yr: (
            yr >= 2018
            and _has_pad(pd)
        ),
        "any_of_cats": ['Lipid_Statins', 'Antiplatelet'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 2018 | 8.17.3 Lower extremity arterial disease',
        "domain":      'ascvd_antiplatelet',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0920',
        "description": 'In T2DM with albuminuric CKD (eGFR 30-90), SGLT2i is recommended to delay renal progression and reduce HF hospitalization',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2019 and (lambda egfr: egfr is not None and 30 <= egfr and egfr <= 90)(_get_egfr(cm)) and _has_albuminuria(cm)),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2019 | 7.1.2.2 Newer oral glucose-lowering drugs',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0921',
        "description": 'ARBs are recommended as an alternative in patients intolerant of an ACE inhibitor',
        "condition":   lambda cm, pd, yr: (yr >= 2016 and _get_dm_comp(cm, "heart_failure")),
        "any_of_cats": ['Hypertension_ARB'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 2016 | 7.2.4 Angiotensin receptor blockers',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0922',
        "description": 'Preferred MRA (Spironolactone or Eplerenone) as add-on therapy for diabetic patients with resistant hypertension or primary aldosteronism',
        "condition":   lambda cm, pd, yr: (yr >= 2017 and _get_dm_type(cm) in ("T1DM", "T2DM") and cm.get("has_hypertension") and (lambda egfr: egfr is not None and egfr >= 30)(_get_egfr(cm)) and (lambda k: k is None or k <= 5)(_get_potassium(pd))),
        "any_of_cats": ['CKD_MRA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ACC-AHA Cholesterol / Cardiovascular 2017 / Synopsis',
        "domain":      'ckd_mra',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0923',
        "description": 'ACE inhibitor is reasonable to slow kidney disease progression in hypertensive patients with CKD (stage 3+ or stage 1/2 with albuminuria >=300 mg/g)',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2017 and cm.get("has_hypertension") and (_has_ckd_g3plus(cm) or (_has_albuminuria(cm) and (lambda uacr: uacr is not None and uacr >= 300)((cm.get("kidney") or {}).get("uacr"))))),
        "any_of_cats": ['Hypertension_ACEi'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ACC-AHA Cardiovascular 2017 - 9.3. Chronic Kidney Disease',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0925',
        "description": 'In nonblack hypertensive population with diabetes, initiate antihypertensive with ACEi, ARB, CCB, or thiazide-type diuretic',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2014 and cm.get("has_hypertension")),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB', 'Hypertension_CCB', 'Hypertension_Diuretics'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ACC-AHA JNC 8 2014 - Recommendation',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0935',
        "description": 'Statins should be considered for LDL-C lowering in adults >40 years with T1DM without CVD to reduce CV risk',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T1DM" and yr >= 2023 and not _get_dm_comp(cm, "cardiovascular") and (_get_age(cm, pd) or 0) > 40),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ESC CVD in Diabetes 2023 2023, Section 11 Type 1 diabetes and cardiovascular disease',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0936',
        "description": 'For T2DM patients with severe TOD (eGFR<45 or eGFR 45-59 with UACR≥30), prescribe SGLT2i or GLP-1 RA with proven CV/renal benefit at discharge',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2020 and ((lambda egfr: egfr is not None and egfr < 45)(_get_egfr(cm)) or ((lambda egfr: egfr is not None and 45 <= egfr <= 59)(_get_egfr(cm)) and (lambda u: u is not None and u >= 30)((cm.get("kidney") or {}).get("uacr"))))),
        "any_of_cats": ['SGLT2i', 'GLP-1RA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023, Section 5.3.5, Figure 7',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0937',
        "description": 'For T2DM patients with HFpEF, SGLT2i (empagliflozin or dapagliflozin) is recommended to reduce CV death or HF hospitalization',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2023
            and _has_hfpef(cm, pd)
        ),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023, Section 7.4.3',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0938',
        "description": 'Continue SGLT2i in diabetic patients with chronic HF when hemodynamically stable, as part of optimal medical therapy for HF',
        "condition":   lambda cm, pd, yr: (yr >= 2021 and _get_dm_type(cm) in ("T1DM", "T2DM") and _get_dm_comp(cm, "heart_failure") and (lambda egfr: egfr is not None and egfr >= 20)(_get_egfr(cm)) and not (cm.get("kidney") or {}).get("aki_suspected", False)),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 (引用2021心衰指南) - Section 14.1.1 / Table 24',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0939',
        "description": 'SGLT2i and/or Finerenone recommended in T2DM with CKD for CV and kidney failure risk reduction.',
        "condition":   lambda cm, pd, yr: (_has_ckd_g3plus(cm)
            and _get_dm_type(cm) == "T2DM" and yr >= 2022 and (lambda egfr: egfr is not None and egfr >= 25)(_get_egfr(cm)) and (lambda u: u is not None and u >= 30)((cm.get("kidney") or {}).get("uacr")) and (lambda k: k is None or k <= 5)(_get_potassium(pd))),
        "any_of_cats": ['SGLT2i', 'CKD_MRA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 2023, section 2 Introduction',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0940',
        "description": 'Diuretics are recommended in patients with congestion and HFmrEF (NYHA II–IV) to alleviate symptoms and signs.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2021 and _has_hfmref(cm, pd)),
        "any_of_cats": ['Hypertension_Diuretics'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ESC CVD in Diabetes 2023 (2021 edition) section 7.3',
        "domain":      'bp_other',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0941',
        "description": 'ACE-I, ARB, beta-blocker, MRA, or sacubitril/valsartan may be considered in patients with HFmrEF (NYHA II–IV) to reduce HF hospitalization and death (grade IIb C).',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) in ("T1DM", "T2DM") and yr >= 2021 and _has_hfmref(cm, pd) and (lambda egfr: egfr is not None and egfr >= 30)(_get_egfr(cm)) and (lambda k: k is None or k <= 5)(_get_potassium(pd))),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB', 'Hypertension_BetaBlockers', 'CKD_MRA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 (2021 edition) section 7.3',
        "domain":      'hf_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0942',
        "description": 'SGLT2i (dapagliflozin or empagliflozin) recommended for HFrEF patients (LVEF≤40%, eGFR≥20) regardless of diabetes status, on top of OMT',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) in ("T1DM", "T2DM") and yr >= 2021 and _has_hfref(cm, pd) and (lambda egfr: egfr is not None and egfr >= 20)(_get_egfr(cm))),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ESC CVD in Diabetes 2023 (2021) | Section 5.3.5',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0943',
        "description": 'ARB recommended to reduce risk of HF hospitalization and CV death in symptomatic HFrEF patients unable to tolerate ACEi/ARNI',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) in ("T1DM", "T2DM") and yr >= 2021 and _has_hfref(cm, pd)),
        "any_of_cats": ['Hypertension_ARB'],
        "exclude_drugs": ['acei', 'arni'],
        "grade":       0.75,
        "source":      'ESC CVD in Diabetes 2023 (2021) | Section 5.4, Table',
        "domain":      'bp_raas',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'P2-A: exclude 指向策展中有意移除的药(噻嗪/非DHP-CCB/gemfibrozil/保钾/跨类别),commission 永不触发,孤儿规则',
    },
    {
        "id":          'FINAL-0944',
        "description": 'SGLT2 inhibitors recommended for patients with HFrEF',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) in ("T1DM", "T2DM") and yr >= 2021 and _has_hfref(cm, pd) and (lambda egfr: egfr is not None and egfr >= 20)(_get_egfr(cm))),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 (2021 version), Key message 3',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0945',
        "description": 'ACEi/ARNI recommended for patients with HFrEF',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) in ("T1DM", "T2DM") and yr >= 2021 and _has_hfref(cm, pd) and (lambda egfr: egfr is not None and egfr >= 30)(_get_egfr(cm))),
        "any_of_cats": ['Hypertension_ACEi'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 (2021 version), Key message 3',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0946',
        "description": 'Beta-blockers recommended for patients with HFrEF',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) in ("T1DM", "T2DM") and yr >= 2021 and _has_hfref(cm, pd)),
        "any_of_cats": ['Hypertension_BetaBlockers'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 (2021 version), Key message 3',
        "domain":      'hf_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0947',
        "description": 'MRA recommended for patients with HFrEF',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2021
            and _has_hfref(cm, pd)
        ),
        "any_of_cats": ['CKD_MRA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 (2021 version), Key message 3',
        "domain":      'ckd_mra',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0948',
        "description": 'Prescribe SGLT2 inhibitor for hospitalized DM patients with HFrEF at discharge',
        "condition":   lambda cm, pd, yr: (yr >= 2020 and _get_dm_type(cm) in ("T1DM", "T2DM") and _has_hfref(cm, pd)),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 (2021 version) - Quality indicators, Domain 3, Main (5)',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0949',
        "description": 'For patients with diabetes and ASCVD or high/very high CV risk, prescribe high-intensity statins at discharge for primary or secondary prevention.',
        "condition":   lambda cm, pd, yr: (yr >= 2023 and _get_dm_type(cm) in ("T1DM", "T2DM") and _get_dm_comp(cm, "cardiovascular")),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023, Section 5.5.2.1 Statins',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0950',
        "description": 'In patients with diabetes and ACS, prolonging DAPT beyond 12 months for up to 3 years should be considered if they have tolerated DAPT without major bleeding complications.',
        "condition":   lambda cm, pd, yr: (yr >= 2023 and _get_dm_type(cm) in ("T1DM", "T2DM") and _has_acs(pd)),
        "any_of_cats": ['Antiplatelet'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023, 5.6.2.2 Acute coronary syndrome',
        "domain":      'ascvd_antiplatelet',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0952',
        "description": 'SGLT2 inhibitors should be used in T2DM patients with CKD to reduce risk of kidney failure and CVD',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2023 and (_has_ckd_g3plus(cm) or _has_albuminuria(cm))),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023, Chronic kidney disease',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0953',
        "description": 'Finerenone should be used in T2DM patients with CKD to reduce risk of kidney failure and CVD',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2023 and (lambda egfr: egfr is not None and egfr >= 25)(_get_egfr(cm)) and _has_albuminuria(cm) and (lambda k: k is None or k <= 5.0)(_get_potassium(pd))),
        "any_of_cats": ['CKD_MRA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023, Chronic kidney disease',
        "domain":      'ckd_mra',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0954',
        "description": 'For T2DM patients with CKD, a statin-based regimen is recommended to reduce cardiovascular risk.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2023 and (_has_ckd_g3plus(cm) or _has_albuminuria(cm))),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023, Chapter: Treatment of patients with T2DM and CKD',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0956',
        "description": 'Beta-blockers are the mainstay of therapy in patients with T2DM, HFrEF, and CAD.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2021
            and _has_cad(pd)
            and _has_hfref(cm, pd)
        ),
        "any_of_cats": ['Hypertension_BetaBlockers'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023, Section 12.2.1 (2021 version)',
        "domain":      'hf_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0957',
        "description": 'For T2DM patients without symptomatic ASCVD or revascularization, ASA (75-100 mg o.d.) may be considered for primary prevention of first major vascular event, in absence of contraindications (e.g., recent GI bleeding, active liver disease, ASA allergy).',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2023 and not _get_dm_comp(cm, "cardiovascular") and not _has_gi_bleed_hx(pd) and not (cm.get("liver") or {}).get("liver_dysfunction")),
        "any_of_cats": ['Antiplatelet'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023, Section 5.6.1, Table 12',
        "domain":      'ascvd_antiplatelet',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'audit:desc permissive("ASA may be considered" for primary prevention); hard commission penalty contradicts wording; primary-prevention aspirin is a shared-decision judgement not adjudicable per-patient -> disable to avoid anti-signal',
    },
    {
        "id":          'FINAL-0958',
        "description": 'Consider antiplatelet agents in T1DM patients without symptomatic ASCVD but with at least one additional major CV risk factor',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T1DM" and yr >= 2023 and not _get_dm_comp(cm, "cardiovascular") and (cm.get("has_hypertension") or _has_ckd_g3plus(cm) or (_get_age(cm, pd) is not None and _get_age(cm, pd) >= 55))),
        "any_of_cats": ['Antiplatelet'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ESC CVD in Diabetes 2023 | 11.2.4. Antiplatelet therapy',
        "domain":      'ascvd_antiplatelet',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0959',
        "description": 'For hospitalized diabetic patients with HFpEF (LVEF≥50%), MRA can be considered to reduce HF hospitalizations (Class 2b recommendation)',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2022 and _has_hfpef(cm, pd) and (lambda e: e is not None and e >= 30)(_get_egfr(cm)) and (lambda k: k is None or k <= 5)(_get_potassium(pd))),
        "any_of_cats": ['CKD_MRA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ACC-AHA Cholesterol/Cardiovascular 2022, Top 10 Take-Home Messages Message 3',
        "domain":      'ckd_mra',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0960',
        "description": 'Avoid nondihydropyridine CCBs (diltiazem, verapamil) in patients with reduced LVEF or post-MI due to negative inotropic effects and increased HF risk',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2022
            and _get_dm_comp(cm, "heart_failure")
        ),
        "any_of_cats": ['Hypertension_CCB'],
        "exclude_drugs": ['diltiazem', 'verapamil'],
        "grade":       1.0,
        "source":      'ACC-AHA Cholesterol / Cardiovascular 2022, Chunk 1, Level A',
        "domain":      'bp_other',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'P2-A: exclude 指向策展中有意移除的药(噻嗪/非DHP-CCB/gemfibrozil/保钾/跨类别),commission 永不触发,孤儿规则',
    },
    {
        "id":          'FINAL-0961',
        "description": 'In patients with symptomatic chronic HFrEF, SGLT2i are recommended to reduce hospitalization for HF and cardiovascular mortality, irrespective of presence of type 2 diabetes.',
        "condition":   lambda cm, pd, yr: (yr >= 2020 and _has_hfref(cm, pd)),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ACC-AHA Cholesterol / Cardiovascular 2022, Section 7.3.4, Sodium-Glucose Cotransporter 2 Inhibitors',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0962',
        "description": 'Consider MRA for diabetic patients with HFpEF (LVEF≥50%) (COR 2b)',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) in ("T1DM", "T2DM") and yr >= 2022 and _has_hfpef(cm, pd) and (lambda e: e is not None and e >= 30)(_get_egfr(cm)) and (lambda k: k is None or k <= 5)(_get_potassium(pd))),
        "any_of_cats": ['CKD_MRA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ACC-AHA Cholesterol/Cardiovascular 2022, 7.7 Preserved EF (HFpEF), Figure 12',
        "domain":      'ckd_mra',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0963',
        "description": 'Consider ARB for diabetic patients with HFpEF (LVEF≥50%) (COR 2b)',
        "condition":   lambda cm, pd, yr: (yr >= 2022 and _get_dm_type(cm) in ("T1DM", "T2DM") and _has_hfpef(cm, pd)),
        "any_of_cats": ['Hypertension_ARB'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ACC-AHA Cholesterol/Cardiovascular 2022, 7.7 Preserved EF (HFpEF), Figure 12',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0964',
        "description": 'For hospitalized diabetic patients with HFpEF (LVEF≥50%), SGLT2 inhibitors should be preferred at discharge to reduce HF hospitalization and cardiovascular mortality.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2022
            and _has_hfpef(cm, pd)
        ),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ACC-AHA Cholesterol/Cardiovascular 2022, Section 7.7.1 HF With Preserved Ejection Fraction, Recommendation 2',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0965',
        "description": 'In selected hospitalized diabetic patients with HFpEF, particularly those with LVEF on the lower end, ARB may be considered at discharge to decrease hospitalizations.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2022 and _has_hfpef(cm, pd)),
        "any_of_cats": ['Hypertension_ARB'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ACC-AHA Cholesterol/Cardiovascular 2022, Section 7.7.1 HF With Preserved Ejection Fraction, Recommendation 5',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0966',
        "description": 'In patients with type 2 diabetes and established CVD, SGLT2i should be used to prevent hospitalization for HF.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2022
            and _get_dm_comp(cm, "cardiovascular")
        ),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ACC-AHA Cholesterol/Cardiovascular 2022, Section 5.1 Patients at Risk for HF',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0967',
        "description": 'Beta blockers should be prescribed to all patients with HFrEF, unless contraindicated or not tolerated.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2022
            and _has_hfref(cm, pd)
        ),
        "any_of_cats": ['Hypertension_BetaBlockers'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ACC-AHA Cholesterol/Cardiovascular 2022, Synopsis Section',
        "domain":      'hf_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0968',
        "description": 'In diabetic patients with HFrEF (LVEF<45%), recent worsening HF (hospitalization within 6 months or IV diuretics), eGFR≥15, SBP≥100, vericiguat may be considered.',
        "condition":   lambda cm, pd, yr: (yr >= 2021 and _get_dm_type(cm) in ("T1DM", "T2DM") and _has_hfref(cm, pd) and (lambda egfr: egfr is not None and egfr >= 15)(_get_egfr(cm))),
        "any_of_cats": ['Microcirculation_Vasodilators'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ACC-AHA Cholesterol / Cardiovascular 2022, Chunk 3',
        "domain":      'microvascular',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0969',
        "description": 'For hospitalized diabetic patients with HFrEF, continue and optimize preexisting GDMT (ACEi/ARB/ARNI, beta-blockers, MRA, etc.); if discontinued, reinitiate as soon as possible; initiate GDMT after clinical stability.',
        "condition":   lambda cm, pd, yr: (yr >= 2022 and _get_dm_type(cm) in ("T1DM", "T2DM") and _has_hfref(cm, pd) and (lambda egfr: egfr is not None and egfr >= 30)(_get_egfr(cm)) and (lambda k: k is None or k <= 5)(_get_potassium(pd))),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB', 'Hypertension_BetaBlockers', 'CKD_MRA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      '2022 ACC-AHA Cholesterol/Cardiovascular Guideline, Section 9.2 Maintenance or Optimization of GDMT During Hospitalization',
        "domain":      'hf_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0970',
        "description": 'In diabetic patients with heart failure and fluid retention, diuretics are recommended to relieve congestion, improve symptoms, and prevent worsening heart failure.',
        "condition":   lambda cm, pd, yr: (yr >= 2022 and _get_dm_type(cm) in ("T1DM", "T2DM") and _get_dm_comp(cm, "heart_failure")),
        "any_of_cats": ['Hypertension_Diuretics'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ACC-AHA Cholesterol / Cardiovascular 2022, Section 7.2',
        "domain":      'bp_other',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0971',
        "description": 'A transient decrease in renal function with RAASi/ARNI/SGLT2i should not prompt interruption in HF/CKD patients if eGFR>25 and dip<10%',
        "condition":   lambda cm, pd, yr: (_has_ckd_g3plus(cm)
            and _get_dm_type(cm) == "T2DM" and yr >= 2021 and _get_dm_comp(cm, "heart_failure") and (lambda egfr: egfr is not None and egfr >= 25)(_get_egfr(cm))),
        "any_of_cats": ['SGLT2i', 'Hypertension_ACEi', 'Hypertension_ARB'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 | 13.6 Kidney dysfunction | 2021',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0972',
        "description": 'Beta-blockers are recommended for patients with HFrEF and CAD due to prognostic benefit',
        "condition":   lambda cm, pd, yr: (yr >= 2021 and _get_dm_type(cm) in ("T1DM", "T2DM") and _has_cad(pd) and _has_hfref(cm, pd)),
        "any_of_cats": ['Hypertension_BetaBlockers'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 | 12.2.1 Medical therapy | 2021',
        "domain":      'hf_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0973',
        "description": 'Diltiazem and verapamil are contraindicated in patients with HFrEF as they increase HF-related events',
        "condition":   lambda cm, pd, yr: (
            yr >= 2021
            and _has_hfref(cm, pd)
        ),
        "any_of_cats": ['Hypertension_CCB'],
        "exclude_drugs": ['diltiazem', 'verapamil'],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 | 12.2.1 Medical therapy | 2021',
        "domain":      'bp_other',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'P2-A: exclude 指向策展中有意移除的药(噻嗪/非DHP-CCB/gemfibrozil/保钾/跨类别),commission 永不触发,孤儿规则',
    },
    {
        "id":          'FINAL-0976',
        "description": 'In patients with a recent or remote history of MI or ACS, statins should be used to prevent symptomatic HF and adverse cardiovascular events.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2022
            and (_has_prior_mi(pd) or _has_acs(pd))
        ),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ACC-AHA Cardiovascular 2022 - Management of Stage B',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0977',
        "description": 'SGLT2i is recommended for patients with T2DM and established CVD or high CVD risk to prevent HF hospitalizations',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2022
            and _get_dm_comp(cm, "cardiovascular")
            and (lambda egfr: egfr is not None and 25 <= egfr)(_get_egfr(cm))
        ),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ACC-AHA Cardiovascular 2022 - Stage A/B Recommendations',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0978',
        "description": 'MRA is recommended for HFrEF to improve outcomes; contraindicated if eGFR<=30 or K+>=5.0',
        "condition":   lambda cm, pd, yr: (
            yr >= 2022
            and _has_hfref(cm, pd)
            and (lambda egfr: egfr is not None and 30 < egfr)(_get_egfr(cm))
        ),
        "any_of_cats": ['CKD_MRA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ACC-AHA Cardiovascular 2022 - Synopsis',
        "domain":      'ckd_mra',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0980',
        "description": 'Loop diuretics are recommended over thiazides for diabetic patients with heart failure and fluid overload.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2013 and _get_dm_comp(cm, "heart_failure")),
        "any_of_cats": ['Hypertension_Diuretics'],
        "exclude_drugs": ['thiazides'],
        "grade":       0.75,
        "source":      'ESC CVD in Diabetes 2023 (2013) | 8.3 Pharmacological management of heart failure in type 2 di | Level=B',
        "domain":      'bp_other',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'P2-A: exclude 指向策展中有意移除的药(噻嗪/非DHP-CCB/gemfibrozil/保钾/跨类别),commission 永不触发,孤儿规则',
    },
    {
        "id":          'FINAL-0982',
        "description": 'Beta-blockers recommended in patients with HFrEF and diabetes to reduce risk of HF hospitalization and death.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2001 and _has_hfref(cm, pd)),
        "any_of_cats": ['Hypertension_BetaBlockers'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 Recommendation Table 20 — Heart failure',
        "domain":      'hf_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0983',
        "description": 'In patients with CAD, without LV systolic dysfunction or HF, ACEIs are recommended to prevent HF and reduce cardiovascular and all-cause mortality',
        "condition":   lambda cm, pd, yr: (yr >= 2001 and _get_dm_type(cm) in ("T1DM", "T2DM") and _has_cad(pd) and not _get_dm_comp(cm, "heart_failure")),
        "any_of_cats": ['Hypertension_ACEi'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 (2016) - Section 6: Delaying or preventing the development of overt heart failure',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0984',
        "description": 'For T2DM patients with HFrEF (LVEF ≤40%), prescribe an ACE-I at discharge to reduce risk of HF hospitalization and death.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2001 and _has_hfref(cm, pd)),
        "any_of_cats": ['Hypertension_ACEi'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 2021 | 5.3 Drugs recommended in all patients with heart failure with reduced ejection fraction',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0985',
        "description": 'For T2DM patients with HFrEF (LVEF ≤40%), prescribe an MRA at discharge to reduce risk of HF hospitalization and death.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2001 and _has_hfref(cm, pd)),
        "any_of_cats": ['CKD_MRA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 2021 | 5.3 Drugs recommended in all patients with heart failure with reduced ejection fraction',
        "domain":      'ckd_mra',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0986',
        "description": 'In HF patients with HFrEF, beta-blockers and RAAS blockers are recommended to reduce the risk of sudden cardiac death',
        "condition":   lambda cm, pd, yr: (yr >= 2001 and _get_dm_type(cm) in ('T1DM', 'T2DM') and _has_hfref(cm, pd)),
        "any_of_cats": ['Hypertension_BetaBlockers', 'Hypertension_ACEi', 'Hypertension_ARB'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2019 | Section 9.2.3 Sudden cardiac death',
        "domain":      'hf_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0997',
        "description": 'SGLT2 inhibitors are recommended in T2DM patients with heart failure to reduce HF hospitalization and CV death',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2016
            and _get_dm_comp(cm, "heart_failure")
        ),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 2021 | Recommendations for treatment of patients with HF and diabetes',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-0998',
        "description": 'For diabetic patients with HF, prescribe ACE-I or ARB at discharge to reduce HF hospitalization and mortality',
        "condition":   lambda cm, pd, yr: (yr >= 2016 and _get_dm_type(cm) in ("T1DM", "T2DM") and _get_dm_comp(cm, "heart_failure")),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 (引用2016 HF指南) - Section 16',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1000',
        "description": 'For diabetic patients with HFrEF (LVEF≤35%) who remain symptomatic despite ACE-I and beta-blocker, add MRA to reduce HF hospitalization and death',
        "condition":   lambda cm, pd, yr: (yr >= 2016 and _get_dm_type(cm) in ("T1DM", "T2DM") and _has_hfref(cm, pd) and (lambda e: e is not None and e >= 30)(_get_egfr(cm)) and (lambda k: k is None or k <= 5.0)(_get_potassium(pd))),
        "any_of_cats": ['CKD_MRA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 (引用2016 HF指南) - Section 16',
        "domain":      'ckd_mra',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1001',
        "description": 'Sacubitril/valsartan or an ACE-I recommended in all patients with HFrEF and diabetes to reduce risk of HF hospitalization and death.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2016
            and _has_hfref(cm, pd)
        ),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 Recommendation Table 20 — Heart failure',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1002',
        "description": 'In HF patients with HFrEF, beta-blockers, RAAS blockers (including sacubitril/valsartan), and MRAs are recommended to reduce sudden cardiac death',
        "condition":   lambda cm, pd, yr: (yr >= 2019 and _get_dm_type(cm) in ("T1DM", "T2DM") and _has_hfref(cm, pd) and (lambda e: e is not None and e >= 30)(_get_egfr(cm)) and (lambda k: k is None or k <= 5.0)(_get_potassium(pd))),
        "any_of_cats": ['Hypertension_BetaBlockers', 'Hypertension_ACEi', 'Hypertension_ARB', 'CKD_MRA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 2019, 9.2.3 Sudden cardiac death in diabetes',
        "domain":      'hf_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1003',
        "description": 'Beta-blockers recommended for patients with DM with HF and after acute MI with LVEF <40% to prevent sudden cardiac death.',
        "condition":   lambda cm, pd, yr: (yr >= 2019 and _get_dm_type(cm) in ("T1DM", "T2DM") and (_has_hfref(cm, pd) or (_has_acute_mi(pd) and (lambda l: l is not None and l < 40)(_get_lvef(pd))))),
        "any_of_cats": ['Hypertension_BetaBlockers'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 2019, 9.2.3 Sudden cardiac death in diabetes',
        "domain":      'hf_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1004',
        "description": 'Diuretics are recommended in hospitalized diabetic patients with HFpEF or HFmrEF who present with congestion, to alleviate symptoms and signs.',
        "condition":   lambda cm, pd, yr: (yr >= 2016 and _get_dm_type(cm) in ("T1DM", "T2DM") and (_has_hfmref(cm, pd) or _has_hfpef(cm, pd))),
        "any_of_cats": ['Hypertension_Diuretics'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ESC CVD in Diabetes 2023 (2016 version), Section 10.1.3, Recommendations for treatment of patients with heart failure',
        "domain":      'bp_other',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1009',
        "description": 'Basal insulins (glargine, degludec) should be considered for glucose-lowering treatment in T2DM patients at risk of or with HF',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2023 and _get_dm_comp(cm, "heart_failure")),
        "any_of_cats": ['Insulins_Basal'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ESC CVD in Diabetes 2023 2023, Section 7 Heart failure and diabetes',
        "domain":      'glycemic_insulin',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1010',
        "description": 'For T2DM patients with HFrEF (LVEF≤40%), discharge prescription should include ACEi or ARB.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2021 and _has_hfref(cm, pd)),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023, Chunk1 (2021) HFrEF treatment',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1012',
        "description": 'For T2DM patients with HFrEF (LVEF ≤40%), prescribe dapagliflozin or empagliflozin at discharge to reduce risk of HF hospitalization and death.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2021
            and _has_hfref(cm, pd)
            and (lambda egfr: egfr is not None and 20 <= egfr)(_get_egfr(cm))
        ),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 2021 | 5.3 Drugs recommended in all patients with heart failure with reduced ejection fraction',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1013',
        "description": 'For T2DM patients with HFmrEF (LVEF 40-49%), an ACE-I may be considered at discharge to reduce risk of HF hospitalization and death.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2021
            and _has_hfmref(cm, pd)
        ),
        "any_of_cats": ['Hypertension_ACEi'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ESC CVD in Diabetes 2023 2021 | New concepts | Recommendations for treatment of chronic HF | HFmrEF',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1014',
        "description": 'For T2DM patients with HFmrEF (LVEF 40-49%), an ARB may be considered at discharge to reduce risk of HF hospitalization and death.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2021
            and _has_hfmref(cm, pd)
        ),
        "any_of_cats": ['Hypertension_ARB'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ESC CVD in Diabetes 2023 2021 | New concepts | Recommendations for treatment of chronic HF | HFmrEF',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1015',
        "description": 'For T2DM patients with HFmrEF (LVEF 40-49%), a beta-blocker may be considered at discharge to reduce risk of HF hospitalization and death.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2021
            and _has_hfmref(cm, pd)
        ),
        "any_of_cats": ['Hypertension_BetaBlockers'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ESC CVD in Diabetes 2023 2021 | New concepts | Recommendations for treatment of chronic HF | HFmrEF',
        "domain":      'hf_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1016',
        "description": 'For T2DM patients with HFmrEF (LVEF 40-49%), an MRA may be considered at discharge to reduce risk of HF hospitalization and death.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2021
            and _has_hfmref(cm, pd)
        ),
        "any_of_cats": ['CKD_MRA'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ESC CVD in Diabetes 2023 2021 | New concepts | Recommendations for treatment of chronic HF | HFmrEF',
        "domain":      'ckd_mra',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1017',
        "description": 'MRAs are recommended in patients with HFrEF and diabetes to reduce the risk of HF hospitalization and death.',
        "condition":   lambda cm, pd, yr: (yr >= 2023 and _get_dm_type(cm) in ("T1DM", "T2DM") and _has_hfref(cm, pd)),
        "any_of_cats": ['CKD_MRA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023, Recommendation Table 20 — Heart failure',
        "domain":      'ckd_mra',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1018',
        "description": 'Initiation of SGLT2 inhibitors before discharge following HF hospitalization is recommended as part of evidence-based treatment to reduce re-admission or mortality.',
        "condition":   lambda cm, pd, yr: (yr >= 2023 and _get_dm_type(cm) in ("T1DM", "T2DM") and _get_dm_comp(cm, "heart_failure") and (lambda egfr: egfr is not None and 20 <= egfr)(_get_egfr(cm))),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'ESC CVD in Diabetes 2023, Recommendation Table 20 — Heart failure',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1019',
        "description": 'ARBs are recommended in symptomatic HFrEF patients with diabetes who do not tolerate sacubitril/valsartan or ACE-Is.',
        "condition":   lambda cm, pd, yr: (yr >= 2023 and _get_dm_type(cm) in ("T1DM", "T2DM") and _has_hfref(cm, pd)),
        "any_of_cats": ['Hypertension_ARB'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023, Recommendation Table 20 — Heart failure',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1020',
        "description": 'Early initiation of evidence-based treatment (SGLT2i, ARNI/ACE-Is, beta-blockers, MRAs) before discharge is recommended',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) in ("T1DM", "T2DM") and yr >= 2023 and _get_dm_comp(cm, "heart_failure") and (lambda egfr: egfr is None or egfr >= 30)(_get_egfr(cm)) and (lambda k: k is None or k <= 5)(_get_potassium(pd))),
        "any_of_cats": ['SGLT2i', 'Hypertension_ACEi', 'Hypertension_BetaBlockers', 'CKD_MRA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 | Evaluation for heart failure in diabetes',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1021',
        "description": 'SGLT2 inhibitors are recommended in patients with diabetes at high risk of CV disease or with CV disease in order to prevent HF hospitalizations.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) in ("T1DM", "T2DM") and yr >= 2021 and _get_dm_comp(cm, "cardiovascular")),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 (2021 version) - Recommendations for the prevention of chronic HF',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1029',
        "description": 'In adults aged ≥50 years with eGFR <60 ml/min (GFR G3a–G5) not on dialysis or transplant, recommend statin or statin/ezetimibe combination.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) in ("T1DM", "T2DM") and yr >= 2024 and (lambda egfr: egfr is not None and egfr <= 59)(_get_egfr(cm)) and (_get_age(cm, pd) or 0) >= 50 and not _has_esrd_dialysis(pd)),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'KDIGO 2024 CKD & Diabetes Guideline, Section 3.15.1 Lipid management, Recommendation 3.15.1.1',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1030',
        "description": 'In adults aged ≥50 years with CKD and eGFR ≥60 ml/min (GFR G1–G2), recommend treatment with a statin.',
        "condition":   lambda cm, pd, yr: (_has_ckd_g3plus(cm)
            and yr >= 2013 and _get_dm_type(cm) in ("T1DM", "T2DM") and (lambda egfr: egfr is not None and 60 <= egfr)(_get_egfr(cm)) and (_get_age(cm, pd) or 0) >= 50),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'KDIGO 2024 CKD & Diabetes Guideline, Section 3.15.1, Recommendation 3.15.1.2',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1031',
        "description": 'In adults aged 18–49 years with CKD, suggest statin treatment in people with diabetes mellitus (or other specified risk factors).',
        "condition":   lambda cm, pd, yr: (yr >= 2013 and _get_dm_type(cm) in ("T1DM", "T2DM") and 18 <= (_get_age(cm, pd) or 0) < 50 and (_has_ckd_g3plus(cm) or _has_albuminuria(cm))),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'KDIGO 2024 CKD & Diabetes Guideline, Section 3.15.1, Recommendation 3.15.1.3',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1032',
        "description": 'In adults aged ≥50 years with eGFR <60 ml/min/1.73m² (CKD G3a–G5) not on dialysis or transplant, recommend statin or statin/ezetimibe combination',
        "condition":   lambda cm, pd, yr: (_has_ckd_g3plus(cm)
            and _get_dm_type(cm) in ("T1DM", "T2DM") and yr >= 2013 and (lambda egfr: egfr is not None and egfr < 60)(_get_egfr(cm)) and (_get_age(cm, pd) or 0) >= 50 and not _has_esrd_dialysis(pd)),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'KDIGO CKD & Diabetes-CKD Guidelines 2024, Section 3.15.1',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1033',
        "description": 'In adults aged ≥50 years with CKD and eGFR ≥60 ml/min/1.73m² (CKD G1–G2), recommend statin treatment',
        "condition":   lambda cm, pd, yr: (_has_ckd_g3plus(cm)
            and _get_dm_type(cm) in ("T1DM", "T2DM") and yr >= 2013 and (lambda egfr: egfr is not None and 60 <= egfr)(_get_egfr(cm)) and (_get_age(cm, pd) or 0) >= 50),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'KDIGO CKD & Diabetes-CKD Guidelines 2024, Section 3.15.1',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1034',
        "description": 'In adults aged 18–49 years with CKD not on dialysis or transplant, suggest statin if any of: known coronary disease, diabetes mellitus, prior ischemic stroke, or estimated 10-year risk of coronary death or MI >10%',
        "condition":   lambda cm, pd, yr: (_has_ckd_g3plus(cm)
            and _get_dm_type(cm) in ("T1DM", "T2DM") and yr >= 2013 and 18 <= (_get_age(cm, pd) or 0) < 50 and not _has_esrd_dialysis(pd)),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'KDIGO CKD & Diabetes-CKD Guidelines 2024, Section 3.15.1',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1036',
        "description": 'Temporarily discontinue RAAS blockers and diuretics in people with GFR < 60 who have serious intercurrent illness that increases the risk of AKI',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2012
            and (lambda egfr: egfr is not None and egfr <= 59)(_get_egfr(cm))
            and True
        ),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB', 'Hypertension_Diuretics'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'KDIGO CKD 2012 4.4.3',
        "domain":      'bp_raas',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'judge:scope_out_of_discharge',
    },
    {
        "id":          'FINAL-1039',
        "description": 'Recommend statins for all patients with T1D or T2D and CKD.',
        "condition":   lambda cm, pd, yr: (yr >= 2001 and _get_dm_type(cm) in ("T1DM", "T2DM") and _has_ckd_g3plus(cm) and not _is_hospice(pd)),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'KDIGO CKD & Diabetes-CKD Guidelines 2022, Section 1.1, Figure 1',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1040',
        "description": 'For T2D with CKD and UACR≥30mg/g, recommend ACEi or ARB for kidney protection',
        "condition":   lambda cm, pd, yr: (yr >= 2001 and _get_dm_type(cm) == "T2DM" and _has_ckd_with_proteinuria(cm)),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'KDIGO 2022 CKD & Diabetes Guidelines, Table of ACEi/ARB dosing (Practice Point 1.2.1 supported by RENAAL,IDNT)',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1041',
        "description": 'For patients with diabetes and CKD with moderately to severely increased albuminuria (A2 or A3), recommend ACEi or ARB.',
        "condition":   lambda cm, pd, yr: (yr >= 2001 and _get_dm_type(cm) in ("T1DM", "T2DM") and (lambda egfr: egfr is not None and egfr >= 15)(_get_egfr(cm)) and _has_albuminuria(cm)),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'KDIGO 2024 Clinical Practice Guideline for Diabetes Management in Chronic Kidney Disease, Recommendation 3.6.3',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1042',
        "description": 'Recommend starting ACEi or ARB (RASi) for people with CKD, diabetes, and moderately-to-severely increased albuminuria (A2 and A3), CKD G1–G4',
        "condition":   lambda cm, pd, yr: (yr >= 2001 and _get_dm_type(cm) in ("T1DM", "T2DM") and _has_albuminuria(cm) and (lambda egfr: egfr is not None and egfr >= 15)(_get_egfr(cm))),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'KDIGO 2024 Clinical Practice Guideline for Diabetes Management in CKD, Section 3.6, Recommendation 3.6.3',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1043',
        "description": 'Recommend moderate- or high-intensity statin therapy for T2DM patients with CKD or ASCVD.',
        "condition":   lambda cm, pd, yr: (yr >= 2001 and _get_dm_type(cm) == "T2DM" and (_get_dm_comp(cm, "nephropathy") or _get_dm_comp(cm, "cardiovascular"))),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'KDIGO CKD & Diabetes-CKD Guidelines 2024, Figure 17 (statin-based therapy)',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1044',
        "description": 'For patients with resistant hypertension, no history of hyperkalemia, and eGFR>45, consider adding MRA (spironolactone/eplerenone) for BP and albuminuria reduction.',
        "condition":   lambda cm, pd, yr: (yr >= 2001 and _get_dm_type(cm) in ("T1DM", "T2DM") and cm.get("has_hypertension") == True and (lambda egfr: egfr is not None and egfr > 45)(_get_egfr(cm))),
        "any_of_cats": ['CKD_MRA'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'KDIGO CKD & Diabetes-CKD Guidelines 2020, Practice Point 1.2.1 (Chunk 4)',
        "domain":      'ckd_mra',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1045',
        "description": 'Statin therapy for primary prevention in T2DM with CKD stages 1-4 and age > 40.',
        "condition":   lambda cm, pd, yr: (yr >= 2001 and _get_dm_type(cm) == "T2DM" and _get_dm_comp(cm, "nephropathy") and (lambda egfr: egfr is not None and egfr >= 15)(_get_egfr(cm)) and (_get_age(cm, pd) or 0) > 40),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'KDIGO CKD & Diabetes-CKD Guidelines 2022 Chapter 1.1',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1047',
        "description": 'Statin treatment for patients aged >=50 with CKD and eGFR >=60 (G1-G2)',
        "condition":   lambda cm, pd, yr: (yr >= 2001 and _get_dm_type(cm) in ("T1DM", "T2DM") and _get_dm_comp(cm, "nephropathy") and (lambda egfr: egfr is not None and egfr >= 60)(_get_egfr(cm)) and (_get_age(cm, pd) or 0) >= 50),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'KDIGO CKD & Diabetes-CKD Guidelines 2024 | 3.15.1 | Rec 3.15.1.2',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1048',
        "description": 'Steroidal MRA are standard of care for treatment of heart failure; adding a nonsteroidal MRA should not be done',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and _get_dm_comp(cm, "heart_failure")
        ),
        "any_of_cats": ['CKD_MRA'],
        "exclude_drugs": ['finerenone'],
        "grade":       1.0,
        "source":      'KDIGO CKD & Diabetes-CKD Guidelines 2022, Practice Point 1.4.5',
        "domain":      'ckd_mra',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'audit:avoid_uncheckable',
    },
    {
        "id":          'FINAL-1052',
        "description": 'SGLT2i recommended for T2DM with CKD (eGFR≥25) to reduce kidney progression and CV events.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2019 and (_has_ckd_g3plus(cm) or _has_albuminuria(cm)) and (lambda egfr: egfr is not None and 25 <= egfr)(_get_egfr(cm)) and not (cm.get("kidney") or {}).get("aki_suspected", False)),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'KDIGO CKD & Diabetes-CKD Guidelines 2022 References (chunk2)',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1053',
        "description": 'For T2D patients with CKD, GLP-1 RA are recommended to reduce MACE and composite kidney outcomes.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2017 and (_has_ckd_g3plus(cm) or _has_albuminuria(cm))),
        "any_of_cats": ['GLP-1RA'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'KDIGO CKD & Diabetes-CKD Guidelines 2022, Chapter 4: Key information on GLP-1 RA',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1054',
        "description": 'SGLT2i provides kidney and cardioprotective benefits for T2DM patients with CKD and eGFR >= 30 ml/min/1.73m2.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2019 and (_has_ckd_g3plus(cm) or _has_albuminuria(cm)) and (lambda egfr: egfr is not None and egfr >= 30)(_get_egfr(cm))),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'KDIGO CKD & Diabetes-CKD Guidelines 2022 - Implementation (CREDENCE 2019)',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1081',
        "description": 'Add ns-MRA (Finerenone) for T2D patients with persistent albuminuria >30 mg/g and normal potassium.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2022 and _has_ckd_with_proteinuria(cm) and (lambda egfr: egfr is not None and egfr >= 25)(_get_egfr(cm)) and (lambda k: k is not None and k <= 5.0)(_get_potassium(pd))),
        "any_of_cats": ['CKD_MRA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'KDIGO CKD & Diabetes-CKD Guidelines 2022, Section 1.1, Figure 1',
        "domain":      'ckd_mra',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1082',
        "description": 'For T2D with CKD and eGFR≥20, SGLT2i are preferred for kidney and heart protective agents',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2022 and (_has_ckd_g3plus(cm) or _has_albuminuria(cm)) and (lambda egfr: egfr is not None and egfr >= 20)(_get_egfr(cm))),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'KDIGO 2022 CKD & Diabetes Guidelines, Practice Point 1.2.1 (Chunk 2)',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1083',
        "description": 'Finerenone recommended for T2DM with CKD, albuminuria (UACR≥30), eGFR≥25, and potassium≤5.0.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2021 and _has_ckd_with_proteinuria(cm) and (lambda egfr: egfr is not None and egfr >= 25)(_get_egfr(cm)) and (lambda k: k is not None and k <= 5.0)(_get_potassium(pd))),
        "any_of_cats": ['CKD_MRA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'KDIGO CKD & Diabetes-CKD Guidelines 2022 References (chunk3)',
        "domain":      'ckd_mra',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1084',
        "description": 'For T2DM patients with CKD, recommend SGLT2i for kidney protection and cardiovascular risk reduction',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2020 and (_has_ckd_g3plus(cm) or _has_albuminuria(cm))),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'KDIGO CKD & Diabetes-CKD Guidelines 2020 - 1.1 Comprehensive diabetes and CKD management',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1085',
        "description": 'For diabetes patients with albuminuria and hypertension, recommend RAS inhibition (ACEi/ARB) for renoprotection',
        "condition":   lambda cm, pd, yr: (yr >= 2020 and _get_dm_type(cm) in ("T1DM", "T2DM") and _has_ckd_with_proteinuria(cm) and cm.get("has_hypertension") is True),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'KDIGO CKD & Diabetes-CKD Guidelines 2020 - 1.1 Comprehensive diabetes and CKD management',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1086',
        "description": 'For diabetes patients with established cardiovascular disease (ASCVD), recommend lifelong aspirin for secondary prevention',
        "condition":   lambda cm, pd, yr: (yr >= 2020 and _get_dm_type(cm) in ("T1DM", "T2DM") and _get_dm_comp(cm, "cardiovascular")),
        "any_of_cats": ['Antiplatelet'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'KDIGO CKD & Diabetes-CKD Guidelines 2020 - 1.1 Comprehensive diabetes and CKD management',
        "domain":      'ascvd_antiplatelet',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1087',
        "description": 'For T2DM with CKD and albuminuria (UACR ≥30 mg/g), use ACEi or ARB at maximum tolerated dose',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2022 and _has_ckd_with_proteinuria(cm) and (lambda egfr: egfr is not None and egfr >= 30)(_get_egfr(cm))),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'KDIGO CKD & Diabetes-CKD Guidelines 2022, Figure 1 and Section 1.4.3',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1088',
        "description": 'For T2DM with CKD, albuminuria (UACR ≥30 mg/g), potassium ≤5.0 mmol/L, eGFR ≥25, add nonsteroidal MRA (finerenone) on top of ACEi/ARB',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2022 and _has_ckd_with_proteinuria(cm) and (lambda egfr: egfr is not None and egfr >= 25)(_get_egfr(cm)) and (lambda k: k is not None and k <= 5.0)(_get_potassium(pd))),
        "any_of_cats": ['CKD_MRA'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'KDIGO CKD & Diabetes-CKD Guidelines 2022, Section 1.4.3',
        "domain":      'ckd_mra',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1089',
        "description": 'For T2DM with CKD, recommend moderate- or high-intensity statins',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2022 and _has_ckd_g3plus(cm)),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'KDIGO CKD & Diabetes-CKD Guidelines 2022, Figure 1',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1090',
        "description": 'For patients with diabetes and heart failure (any LVEF), prescribe SGLT2i at discharge to reduce CV death and HF hospitalization',
        "condition":   lambda cm, pd, yr: (yr >= 2022 and _get_dm_type(cm) in ("T1DM", "T2DM") and _get_dm_comp(cm, "heart_failure")),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'KDIGO CKD & Diabetes-CKD Guidelines 2022, Section 1.3, EMPEROR-Preserved and SOLOIST trials',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1091',
        "description": 'Recommend SGLT2i for T2DM patients with CKD eGFR≥20 ml/min and heart failure, irrespective of albuminuria.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2020 and _get_dm_comp(cm, "heart_failure") and (_has_ckd_g3plus(cm) or _has_albuminuria(cm)) and (lambda egfr: egfr is not None and egfr >= 20)(_get_egfr(cm))),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'KDIGO CKD & Diabetes-CKD Guidelines 2024, Recommendation 3.7.2',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1092',
        "description": 'Recommend SGLT2i for cardiovascular and renal protection in T2DM patients with CKD and eGFR ≥30 ml/min per 1.73 m²',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2020 and (_has_ckd_g3plus(cm) or _has_albuminuria(cm)) and (lambda egfr: egfr is not None and egfr >= 30)(_get_egfr(cm))),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'KDIGO 2020 CKD & Diabetes Guidelines, Recommendation 4.2.1',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1093',
        "description": 'For patients with CKD (with or without diabetes), recommend SGLT2i to reduce risk of kidney failure, AKI, HF hospitalization, and cardiovascular death.',
        "condition":   lambda cm, pd, yr: (yr >= 2024 and (_has_ckd_g3plus(cm) or _has_albuminuria(cm)) and (lambda egfr: egfr is not None and egfr >= 20)(_get_egfr(cm))),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'KDIGO CKD & Diabetes-CKD Guidelines 2024 | Key information (Chunk 2)',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1094',
        "description": 'For patients with T2DM and CKD (eGFR≥25, UACR≥30, potassium≤5.5 mmol/L), recommend Finerenone to reduce cardiovascular events, especially HF hospitalization.',
        "condition":   lambda cm, pd, yr: (_has_ckd_g3plus(cm)
            and _get_dm_type(cm) == "T2DM" and yr >= 2024 and (lambda egfr: egfr is not None and egfr >= 25)(_get_egfr(cm)) and (lambda uacr: uacr is not None and uacr >= 30)((cm.get("kidney") or {}).get("uacr")) and (lambda k: k is None or k <= 5.5)(_get_potassium(pd))),
        "any_of_cats": ['CKD_MRA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'KDIGO CKD & Diabetes-CKD Guidelines 2024 | K+ >5.5 mmol/l (Chunk 3)',
        "domain":      'ckd_mra',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1095',
        "description": 'For T2DM patients with CKD (eGFR≥25 and UACR≥200 mg/g), SGLT2i is recommended for renal and cardiovascular protection.',
        "condition":   lambda cm, pd, yr: (_has_ckd_g3plus(cm)
            and _get_dm_type(cm) == "T2DM" and yr >= 2022 and (lambda egfr: egfr is not None and egfr >= 25)(_get_egfr(cm)) and (lambda uacr: uacr is not None and uacr >= 200)((cm.get("kidney") or {}).get("uacr"))),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'KDIGO CKD & Diabetes-CKD Guidelines 2022, Chunk4: KIDNEY TRIALS table',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1096',
        "description": 'For diabetes patients (T1DM or T2DM) with CKD (eGFR≥15 and not on dialysis), ACEi or ARB is recommended for renal protection.',
        "condition":   lambda cm, pd, yr: (_has_ckd_g3plus(cm)
            and yr >= 2022 and _get_dm_type(cm) in ("T1DM", "T2DM") and (lambda egfr: egfr is not None and 15 <= egfr)(_get_egfr(cm)) and not _has_esrd_dialysis(pd)),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'KDIGO CKD & Diabetes-CKD Guidelines 2022, Chunk2: RAS blockade recommendation',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1097',
        "description": 'For diabetes patients with heart failure with reduced ejection fraction (LVEF ≤40%) and eGFR ≥30 ml/min per 1.73m², SGLT2 inhibitor is recommended to reduce cardiovascular death and heart failure hospitalization.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2020
            and _has_hfref(cm, pd)
            and (lambda egfr: egfr is not None and 30 <= egfr)(_get_egfr(cm))
        ),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'KDIGO 2020 Clinical Practice Guideline for Diabetes Management in CKD, Section on SGLT2i and Heart Failure',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1099',
        "description": 'Recommend SGLT2i for CKD adults with eGFR≥20 and UACR≥200',
        "condition":   lambda cm, pd, yr: (_has_ckd_g3plus(cm)
            and _get_dm_type(cm) == "T2DM" and yr >= 2024 and (lambda egfr: egfr is not None and 20 <= egfr)(_get_egfr(cm)) and (lambda uacr: uacr is not None and uacr >= 200)((cm.get("kidney") or {}).get("uacr"))),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'KDIGO 2024 CKD & Diabetes Guideline, Section 3.7, Recommendation 3.7.2',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1100',
        "description": 'Recommend SGLT2i for CKD adults with heart failure, irrespective of albuminuria',
        "condition":   lambda cm, pd, yr: (_has_ckd_g3plus(cm)
            and 
            _get_dm_type(cm) == "T2DM"
            and yr >= 2024
            and _get_dm_comp(cm, "heart_failure")
        ),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'KDIGO 2024 CKD & Diabetes Guideline, Section 3.7, Recommendation 3.7.2',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1101',
        "description": 'Suggest SGLT2i for CKD adults with eGFR 20-45 and UACR<200',
        "condition":   lambda cm, pd, yr: (_has_ckd_g3plus(cm)
            and _get_dm_type(cm) == "T2DM" and yr >= 2024 and (lambda egfr: egfr is not None and 20 <= egfr and egfr <= 45)(_get_egfr(cm)) and (lambda uacr: uacr is not None and uacr < 200)((cm.get("kidney") or {}).get("uacr"))),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'KDIGO 2024 CKD & Diabetes Guideline, Section 3.7, Recommendation 3.7.3',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1102',
        "description": 'Suggest nonsteroidal MRA for T2D with eGFR>25, normal potassium, UACR>30, on max RASi',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2022 and (lambda egfr: egfr is not None and egfr > 25)(_get_egfr(cm)) and (lambda k: k is not None and k <= 5.0)(_get_potassium(pd)) and _has_albuminuria(cm)),
        "any_of_cats": ['CKD_MRA'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'KDIGO 2024 CKD & Diabetes Guideline, Section 3.8, Recommendation 3.8.1',
        "domain":      'ckd_mra',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1103',
        "description": 'For adults with eGFR 20-45 ml/min/1.73m² and UACR <200 mg/g, SGLT2i is suggested to reduce risk of kidney failure.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2020 and (lambda egfr: egfr is not None and 20 <= egfr and egfr <= 45)(_get_egfr(cm)) and (lambda u: u is not None and u < 200)((cm.get("kidney") or {}).get("uacr"))),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'KDIGO CKD & Diabetes-CKD Guidelines 2024, Recommendation 3.7.3, Level B',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1104',
        "description": 'Initiate ACEi or ARB in patients with diabetes, hypertension, and albuminuria; titrate to highest tolerated dose',
        "condition":   lambda cm, pd, yr: (yr >= 2022 and _get_dm_type(cm) in ("T1DM", "T2DM") and cm.get("has_hypertension") == True and _has_ckd_with_proteinuria(cm)),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'KDIGO 2022 CKD & Diabetes-CKD Guidelines, Section 1.2, Recommendation 1.2.1',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1105',
        "description": 'Recommend SGLT2i for T2DM patients with HFrEF (LVEF≤40%) to reduce heart failure hospitalization and cardiovascular death.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2022
            and _has_hfref(cm, pd)
            and (lambda egfr: egfr is not None and 20 <= egfr)(_get_egfr(cm))
        ),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'KDIGO CKD & Diabetes-CKD Guidelines 2022, Section: SGLT2i in Heart Failure',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1106',
        "description": 'For diabetic CKD patients with heart failure (particularly HFrEF), if MRA is indicated, a steroidal MRA (e.g., spironolactone, eplerenone) should be used preferentially over nonsteroidal MRA (e.g., finerenone).',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2022 and _has_hfref(cm, pd) and (lambda egfr: egfr is not None and egfr >= 25)(_get_egfr(cm)) and _has_albuminuria(cm) and (lambda k: k is not None and k <= 5.0)(_get_potassium(pd))),
        "any_of_cats": ['CKD_MRA'],
        "exclude_drugs": ['finerenone'],
        "grade":       1.0,
        "source":      'KDIGO CKD & Diabetes-CKD Guidelines 2022, Practice Point 1.4.5',
        "domain":      'ckd_mra',
        "polarity":    'AVOID',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1107',
        "description": 'For patients with T2DM and established cardiovascular disease or heart failure, recommend SGLT2i to reduce risk of hospitalization for heart failure and MACE.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2020 and (_get_dm_comp(cm, "cardiovascular") or _get_dm_comp(cm, "heart_failure"))),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'KDIGO 2020 Clinical Practice Guideline for Diabetes Management in CKD, Chapter 4.2 SGLT2i (citing EMPA-REG, CANVAS, DECLARE)',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1108',
        "description": 'Continue ACEi or ARB in people with CKD even when eGFR falls below 30 ml/min/1.73m².',
        "condition":   lambda cm, pd, yr: (_has_ckd_g3plus(cm)
            and 
            _get_dm_type(cm) == "T2DM"
            and yr >= 2024
            and (lambda egfr: egfr is not None and egfr <= 30)(_get_egfr(cm))
        ),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB'],
        "exclude_drugs": [],
        "grade":       0.75,
        "source":      'KDIGO CKD & Diabetes-CKD Guidelines 2024, Section 3.6 Renin-angiotensin system inhibitors',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1109',
        "description": 'For diabetic patients with CKD, recommend SGLT2i to reduce risk of CKD progression, AKI, and cardiovascular outcomes; can continue or initiate even when eGFR falls below 20 ml/min/1.73m² until KRT initiation.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2024 and (_has_ckd_g3plus(cm) or _has_albuminuria(cm))),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'KDIGO CKD & Diabetes-CKD Guidelines 2024, Section 3.7.1 (1A recommendation)',
        "domain":      'cardiorenal_protection',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1111',
        "description": 'In patients with T2DM and CKD with eGFR <30 ml/min/1.73m², ACEi/ARB should be continued to delay kidney replacement therapy, and should not be routinely discontinued.',
        "condition":   lambda cm, pd, yr: (_has_ckd_g3plus(cm)
            and 
            _get_dm_type(cm) == "T2DM"
            and yr >= 2022
            and (lambda egfr: egfr is not None and egfr < 30)(_get_egfr(cm))
        ),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'KDIGO CKD & Diabetes-CKD Guidelines 2024, 3.6 Renin-angiotensin system inhibitors',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'FINAL-1112',
        "description": 'Avoid steroidal MRA (spironolactone/eplerenone) added to ACEi/ARB if eGFR < 45 due to hyperkalemia risk',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2020
            and (lambda egfr: egfr is not None and egfr <= 44)(_get_egfr(cm))
        ),
        "any_of_cats": ['CKD_MRA'],
        "exclude_drugs": ['spironolactone', 'eplerenone'],
        "grade":       0.75,
        "source":      'KDIGO CKD & Diabetes-CKD Guidelines 2020 | Practice Point 1.2.1',
        "domain":      'ckd_mra',
        "polarity":    'AVOID',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'audit:avoid_uncheckable',
    },
    {
        "id":          'CPR-01',
        "description": 'T2DM + ASCVD or age>=40: Statin (ADA+ACC-AHA Cholesterol 2019)',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and (
                _get_dm_comp(cm, "cardiovascular")          # ASCVD 确诊 → Grade A
                or (                                         # 年龄 40-75 → Grade A (ADA 2014+)
                    yr >= 2004
                    and (_get_age(cm, pd) or 0) >= 40
                )
            )
        ),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care S9 + ACC-AHA Cholesterol 2019',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'CPR-02',
        "description": 'T2DM + CKD/proteinuria: ACEi or ARB (ADA S9 + KDIGO CKD 2022)',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and _has_ckd_with_proteinuria(cm)
        ),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA S9 + KDIGO CKD 2022 Chapter 3',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'CPR-03',
        "description": 'T2DM + established ASCVD: Antiplatelet (ADA S9)',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and _get_dm_comp(cm, "cardiovascular")
        ),
        "any_of_cats": ['Antiplatelet'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care S9',
        "domain":      'ascvd_antiplatelet',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'CPR-05',
        "description": 'T2DM + DKD (eGFR 25-75) post-2021: Finerenone (KDIGO 2022)',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2021
            and _has_ckd_with_proteinuria(cm)
            and (lambda egfr: egfr is not None and 25 <= egfr <= 75)(_get_egfr(cm))
        ),
        "any_of_cats": ['CKD_MRA'],
        "exclude_drugs": ['spironolactone', 'eplerenone'],
        "grade":       1.0,
        "source":      'KDIGO Diabetes-CKD 2022 + Finerenone FDA 2021',
        "domain":      'ckd_mra',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'CPR-06',
        "description": 'In adults with T2DM and heart failure, the treatment plan should include SGLT2 inhibitors to reduce cardiovascular risk.',
        "condition":   lambda cm, pd, yr: (
    _get_dm_type(cm) == "T2DM"
    and yr >= 2020
    and _get_dm_comp(cm, "heart_failure")
    and (lambda e: e is not None and e >= 20)(_get_egfr(cm))
),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2024 Section 9 (Pharmacologic Approaches)',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'CPR-07',
        "description": 'Statin therapy should be added for T1DM patients with overt CVD.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T1DM"
            and _get_dm_comp(cm, "cardiovascular")
        ),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2010 - Treatment recommendations and goals',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'CPR-08',
        "description": 'ACE inhibitors are preferred for T1DM patients with micro- or macroalbuminuria to delay nephropathy progression.',
        "condition":   lambda cm, pd, yr: (
    _get_dm_type(cm) == "T1DM"
    and _has_albuminuria(cm)
),
        "any_of_cats": ['Hypertension_ACEi'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Standards of Care 2010 - Treatment',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'CPR-11',
        "description": 'Recommend RASi (ACEi or ARB) for patients with diabetes, moderately-to-severely increased albuminuria (A2/A3), and eGFR >= 15 mL/min/1.73m².',
        "condition":   lambda cm, pd, yr: (
    _get_dm_type(cm) in ["T1DM", "T2DM"]
    and _has_albuminuria(cm)
    and (lambda e: e is not None and e >= 15)(_get_egfr(cm))
),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'KDIGO CKD & Diabetes-CKD Guidelines 2024 | 3.6 Renin-angiotensin system inhibitors | Recommendation 3.6.3',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'CPR-12',
        "description": 'In T2DM with heart failure and eGFR ≥20, SGLT2i is recommended to reduce CV mortality and HF hospitalization.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2020
            and _get_dm_comp(cm, "heart_failure")
            and (lambda egfr: egfr is not None and 20 <= egfr)(_get_egfr(cm))
        ),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'KDIGO CKD & Diabetes-CKD Guidelines 2024, Recommendation 3.7.2',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'CPR-13',
        "description": 'In T2DM with CKD (eGFR ≥20, UACR ≥200), SGLT2i is recommended to slow kidney disease progression.',
        "condition":   lambda cm, pd, yr: (
    _get_dm_type(cm) == "T2DM"
    and yr >= 2020
    and _has_albuminuria(cm)
    and (lambda e: e is not None and e >= 20)(_get_egfr(cm))
),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'KDIGO CKD & Diabetes-CKD Guidelines 2024, Recommendation 3.7.2',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'audit:avoid_uncheckable',
    },
    {
        "id":          'CPR-15',
        "description": 'In patients with T2D, CKD, and an eGFR ≥ 20 ml/min per 1.73 m², treatment with an SGLT2i is recommended to prevent CKD progression and cardiovascular events.',
        "condition":   lambda cm, pd, yr: (
    _get_dm_type(cm) == "T2DM"
    and yr >= 2020
    and (lambda egfr: egfr is not None and 20 <= egfr)(_get_egfr(cm))
    and (_has_ckd_g3plus(cm) or _has_albuminuria(cm))
),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'KDIGO CKD & Diabetes-CKD Guidelines 2022, Recommendation 1.3.1 (Chunk 2)',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'CPR-16',
        "description": 'Recommend SGLT2i for adults with CKD and heart failure',
        "condition":   lambda cm, pd, yr: (
    _get_dm_type(cm) == "T2DM"
    and yr >= 2020
    and _get_dm_comp(cm, "heart_failure")
    and (lambda egfr: egfr is not None and 20 <= egfr)(_get_egfr(cm))
    and (_has_ckd_g3plus(cm) or _has_albuminuria(cm))
),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'KDIGO CKD & Diabetes-CKD Guidelines 2024 | Recommendation 3.7.2',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'CPR-17',
        "description": 'Recommend SGLT2i for CKD with eGFR ≥20 and urine ACR ≥200 mg/g',
        "condition":   lambda cm, pd, yr: (
    _get_dm_type(cm) == "T2DM"
    and yr >= 2020
    and (lambda egfr: egfr is not None and 20 <= egfr)(_get_egfr(cm))
    and _has_albuminuria(cm)
),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'KDIGO CKD & Diabetes-CKD Guidelines 2024 | Recommendation 3.7.2',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'audit:avoid_uncheckable',
    },
    {
        "id":          'CPR-19',
        "description": 'Suggest nonsteroidal MRA for T2D, eGFR>25, normal K+, albuminuria >30mg/g',
        "condition":   lambda cm, pd, yr: (
    _get_dm_type(cm) == "T2DM"
    and yr >= 2021
    and (lambda egfr: egfr is not None and 25 <= egfr)(_get_egfr(cm))
    and _has_albuminuria(cm)
    # [P4 fix] Description says "normal K+" but original had no K+ check.
    # Add via _get_potassium(pd). None = unavailable → conservative: do not exclude.
    and (lambda k: k is None or float(k) <= 5.0)(_get_potassium(pd))
),
        "any_of_cats": ['CKD_MRA'],
        "exclude_drugs": ['spironolactone', 'eplerenone'],
        "grade":       0.5,
        "source":      'KDIGO CKD & Diabetes-CKD Guidelines 2024 | Recommendation 3.8.1',
        "domain":      'ckd_mra',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'CPR-21',
        "description": 'T1DM patients with albuminuria are recommended to initiate ACEi or ARB to slow CKD progression.',
        "condition":   lambda cm, pd, yr: (
    _get_dm_type(cm) == "T1DM"
    and _has_albuminuria(cm)
),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'KDIGO CKD & Diabetes-CKD Guidelines 2022, Recommendation 1.2.1',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'CPR-25',
        "description": 'Restart Finerenone 10 mg if previously held for hyperkalemia and K+ is now <= 5.0 mmol/L.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2021
            # [P4 critical fix] Original: cm.get("potassium") always returns None because
            # run_extraction stores K+ in clinical_metrics.electrolytes.potassium (nested).
            # When K+ is None, the original lambda (k is not None and ...) returns False,
            # so the K+ gate was SILENTLY INACTIVE. The rule fired on all T2DM+yr>=2021
            # patients (~51,746) with zero Finerenone satisfaction, massively inflating n_comp.
            # Fix: (1) use _get_potassium(pd) for correct nested lookup
            #      (2) add mandatory CKD+albuminuria conditions (this rule is for DKD patients
            #          who previously stopped Finerenone for hyperkalemia — not all T2DM)
            #      (3) None K+ → conservative: assume K+ is safe (do not block)
            and _has_ckd_g3plus(cm)
            and _has_albuminuria(cm)
            and (lambda k: k is None or float(k) <= 5.0)(_get_potassium(pd))
        ),
        "any_of_cats": ['CKD_MRA'],
        "exclude_drugs": ['spironolactone', 'eplerenone'],
        "grade":       1.0,
        "source":      'KDIGO CKD & Diabetes-CKD Guidelines 2024 | K+ ≤4.8 mmol/l',
        "domain":      'ckd_mra',
        "polarity":    'RECOMMEND',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'judge:uncheckable:?',
    },
    {
        "id":          'CPR-27',
        "description": 'In T2DM patients with eGFR > 30 mL/min/1.73m², treatment with GLP-1 RAs (liraglutide/semaglutide) should be considered for lower risk of renal endpoints.',
        "condition":   lambda cm, pd, yr: (
    _get_dm_type(cm) == "T2DM"
    and yr >= 2020
    and _get_egfr(cm) is not None
    and _get_egfr(cm) > 30
),
        "any_of_cats": ['GLP-1RA'],
        "exclude_drugs": [],
        "grade":       0.5,
        "source":      'ESC CVD in Diabetes 2023 | Section 9 Chronic kidney disease and diabetes',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'CPR-28',
        "description": 'SGLT2 inhibitors are recommended in patients with diabetes and CV disease to prevent HF hospitalizations.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2020
            and _get_dm_comp(cm, "cardiovascular")
        ),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023, Section 8.4 (HFpEF), 2021 update',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'CPR-29',
        "description": 'ACE-I is recommended for patients with HFrEF to reduce the risk of HF hospitalization and death.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and _has_hfref(cm, pd)
        ),
        "any_of_cats": ['Hypertension_ACEi'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 (Chunk 1) | Recommendations for the treatment of HFrEF',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'CPR-30',
        "description": 'MRA is recommended for patients with HFrEF to reduce the risk of HF hospitalization and death.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2004
            and _has_hfref(cm, pd)
        ),
        "any_of_cats": ['CKD_MRA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 (Chunk 1) | Recommendations for the treatment of HFrEF',
        "domain":      'ckd_mra',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'CPR-31',
        "description": 'In T2DM patients with macroalbuminuria (UACR>300) and eGFR 30-90, SGLT2i is recommended for renal protection.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2020 and _has_albuminuria(cm) and (lambda egfr: egfr is not None and 30 <= egfr and egfr <= 90)(_get_egfr(cm))),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       0.5,
        "source":      'ESC CVD in Diabetes 2019 | Section 4 Key messages (CKD)',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'audit:avoid_uncheckable',
    },
    {
        "id":          'CPR-32',
        "description": 'In T2DM patients at very high CV risk (e.g., ASCVD), combination with ezetimibe is recommended to further lower LDL-C.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2015
            and _get_dm_comp(cm, "cardiovascular")
        ),
        "any_of_cats": ['Lipid_NonStatins'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 | 5.5.2.2. Ezetimibe',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'CPR-33',
        "description": 'In patients with HFrEF and diabetes, SGLT inhibitors are recommended to reduce CV death and HF hospitalization.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2020
            and _has_hfref(cm, pd)
        ),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 | 7.4.1.1. Sodium–glucose co-transporter-2 inhibitors',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'CPR-34',
        "description": 'In patients with T2DM and HFrEF (LVEF <= 40%), an MRA is recommended to reduce the risk of HF hospitalization and death.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2004 and _has_hfref(cm, pd)),
        "any_of_cats": ['CKD_MRA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 | Section 5.3 Drugs recommended in all patients with HFrEF',
        "domain":      'ckd_mra',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'CPR-35',
        "description": 'In patients with T2DM and HFrEF (LVEF ≤ 40%), SGLT2 inhibitors are recommended to reduce the risk of HF hospitalization and cardiovascular death.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2020
            and _has_hfref(cm, pd)
        ),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 (Referencing ESC HF 2021) / Figure 3',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'CPR-37',
        "description": 'In T2DM patients with ASCVD, SGLT2i or GLP-1RA should be given first line for CV protection, independent of metformin use.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2016
            and _get_dm_comp(cm, "cardiovascular")
        ),
        "any_of_cats": ['SGLT2i', 'GLP-1RA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 | 5.3.3.1. Metformin',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'CPR-39',
        "description": 'Diuretics recommended in HFrEF patients with signs/symptoms of congestion.',
        "condition":   lambda cm, pd, yr: _has_hfref(cm, pd),
        "any_of_cats": ['Hypertension_Diuretics'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 (2021) Section 5.4',
        "domain":      'bp_other',
        "polarity":    'RECOMMEND',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'audit:avoid_uncheckable',
    },
    {
        "id":          'CPR-40',
        "description": 'ARB recommended in symptomatic HFrEF patients intolerant to ACE-I or ARNI.',
        "condition":   lambda cm, pd, yr: _has_hfref(cm, pd),
        "any_of_cats": ['Hypertension_ARB'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 (2021) Section 5.4',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'judge:uncheckable:?',
    },
    {
        "id":          'CPR-42',
        "description": 'SGLT2 inhibitors (dapagliflozin or empagliflozin) are recommended for patients with HFrEF (LVEF ≤ 40%) regardless of diabetes status.',
        "condition":   lambda cm, pd, yr: (
            yr >= 2020
            and _has_hfref(cm, pd)
            and (lambda egfr: egfr is not None and 20 <= egfr)(_get_egfr(cm))
        ),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       0.5,
        "source":      'ESC CVD in Diabetes 2023 | 5.3.5 Sodium-glucose co-transporter 2 inhibitors',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'CPR-43',
        "description": 'Treatment should aim at reducing symptoms of congestion with diuretics. Loop diuretics are preferred.',
        "condition":   lambda cm, pd, yr: _get_dm_comp(cm, "heart_failure"),
        "any_of_cats": ['Hypertension_Diuretics'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 | 8.4 Treatment of heart failure with preserved ejection fraction | 2021',
        "domain":      'bp_other',
        "polarity":    'RECOMMEND',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'audit:avoid_uncheckable',
    },
    {
        "id":          'CPR-46',
        "description": 'In T2DM patients with HFrEF, RAAS blockers (ACEI or ARB) are recommended to reduce the risk of sudden cardiac death and mortality.',
        "condition":   lambda cm, pd, yr: (
    _get_dm_type(cm) == "T2DM"
    and yr >= 2000
    and _has_hfref(cm, pd)
    and (_get_lvef(pd) is not None and _get_lvef(pd) <= 40)
),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 | 9.2.3 Sudden cardiac death in diabetes | 2019',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'CPR-47',
        "description": 'In patients with diabetes and CAD, ACE-Is or ARBs are recommended to reduce the risk of CV events.',
        "condition":   lambda cm, pd, yr: (
    _get_dm_type(cm) == "T2DM"
    and yr >= 2019
    and _has_cad(pd)
),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 | 6.1.3.1.2. Other medications',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'CPR-48',
        "description": 'In T2DM patients with HFpEF or HFmrEF (LVEF > 40%), SGLT2i is recommended to reduce HF hospitalization or CV death.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2020
            and (_has_hfmref(cm, pd) or _has_hfpef(cm, pd))
        ),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 | Recommendation Table 21',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'CPR-49',
        "description": 'In DM patients with CCS or ACS and systolic LV dysfunction, ACEIs (or ARBs if intolerant) are recommended to prevent major CV events and HF.',
        "condition":   lambda cm, pd, yr: (
    _get_dm_type(cm) in ["T1DM", "T2DM"]
    and yr >= 2019
    and (_has_acs(pd) or _has_cad(pd))
    and ((_get_lvef(pd) is not None and _get_lvef(pd) <= 40) or _has_hfref(cm, pd))
),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 (2019) Section 7.1.3.2',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'CPR-52',
        "description": 'In T2DM with ACS who tolerated DAPT without major bleeding, prolonging DAPT beyond 12 months should be considered for up to 3 years.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2023
            and _has_acs(pd)
        ),
        "any_of_cats": ['Antiplatelet'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 | 5.6.2.2. Acute coronary syndrome',
        "domain":      'ascvd_antiplatelet',
        "polarity":    'RECOMMEND',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'audit:avoid_uncheckable',
    },
    {
        "id":          'CPR-55',
        "description": 'In T2DM undergoing stent implantation with indication for anticoagulation, triple therapy... is recommended for at least 1 week, followed by dual therapy.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2023
            and _get_dm_comp(cm, "cardiovascular")
        ),
        "any_of_cats": ['Antiplatelet'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 | 5.6.3. Patients with atherosclerotic cardiovascular disease',
        "domain":      'ascvd_antiplatelet',
        "polarity":    'RECOMMEND',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'judge:uncheckable:?',
    },
    {
        "id":          'CPR-59',
        "description": 'In DM patients with recent ACS, the combination of ezetimibe with a statin is recommended.',
        "condition":   lambda cm, pd, yr: (yr >= 2015 and _has_acs(pd) and _get_dm_type(cm) is not None),
        "any_of_cats": ['Lipid_NonStatins'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 (2019) Section 6.4.1.2',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'CPR-66',
        "description": 'In T2DM with CKD (eGFR 25-75, UACR>30) and K+ <= 4.8 mmol/L, Finerenone is recommended in addition to RAS inhibitor.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2021
            and (lambda egfr: egfr is not None and 25 <= egfr and egfr <= 75)(_get_egfr(cm))
            and _has_albuminuria(cm)
            and (_get_potassium(pd) is not None and _get_potassium(pd) <= 4.8)
        ),
        "any_of_cats": ['CKD_MRA'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 | 7.4.1.3 & For additional glucose control',
        "domain":      'ckd_mra',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'CPR-67',
        "description": 'Loop diuretics are recommended to reduce the signs and/or symptoms of congestion in patients with HFrEF.',
        "condition":   lambda cm, pd, yr: (
            yr >= 2021
            and _has_hfref(cm, pd)
        ),
        "any_of_cats": ['Hypertension_Diuretics'],
        "exclude_drugs": [],
        "grade":       0.5,
        "source":      'ESC CVD in Diabetes 2023 | 5.4.1 Diuretics | 2021',
        "domain":      'bp_other',
        "polarity":    'RECOMMEND',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'audit:avoid_uncheckable',
    },
    {
        "id":          'CPR-68',
        "description": 'In adults with T2DM without a history of symptomatic ASCVD or revascularization, ASA may be considered to prevent the first severe vascular event.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2023
            and not _get_dm_comp(cm, "cardiovascular")
        ),
        "any_of_cats": ['Antiplatelet'],
        "exclude_drugs": [],
        "grade":       0.5,
        "source":      'ESC CVD in Diabetes 2023 | 5.6.1 | Recommendation Table 12',
        "domain":      'ascvd_antiplatelet',
        "polarity":    'ALLOW',
        "scorable":    True,
    },
    {
        "id":          'CPR-69',
        "description": 'Patients with DM and HFrEF should receive ACEI or ARB to reduce mortality and morbidity.',
        "condition":   lambda cm, pd, yr: _get_dm_type(cm) is not None and _has_hfref(cm, pd),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 | 7.4.1.2 & 7.4.1.5',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'CPR-70',
        "description": 'SGLT2 inhibitors are recommended for patients with T2DM and established ASCVD to reduce HF hospitalization.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2016
            and _get_dm_comp(cm, "cardiovascular")
        ),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 | 7.5.1',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'CPR-71',
        "description": 'In hypertensive patients with diabetes, RAS inhibitor (ACEi or ARB) is recommended as part of the treatment regimen.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) in ["T1DM", "T2DM"] and pd.get("has_hypertension", False)),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_ARB'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ESC CVD in Diabetes 2023 | Section: Blood pressure',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'CPR-78',
        "description": 'In patients with clinical ASCVD who are 75 years of age or younger, high-intensity statin therapy should be initiated or continued.',
        "condition":   lambda cm, pd, yr: _get_dm_comp(cm, "cardiovascular") and (_get_age(cm, pd) is not None and _get_age(cm, pd) <= 75),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ACC-AHA Cholesterol Guideline 2018, Section 4.1 Secondary ASCVD Prevention',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'CPR-79',
        "description": 'In patients older than 75 years of age with clinical ASCVD, it is reasonable to initiate moderate- or high-intensity statin therapy.',
        "condition":   lambda cm, pd, yr: (_has_complication(cm, 'cardiovascular') or _get_dm_comp(cm, 'cardiovascular')) and (_get_age(cm, pd) or 0) > 75,
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       0.5,
        "source":      'ACC-AHA Cholesterol Guideline 2018, Section 4.1 Secondary ASCVD Prevention',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'CPR-80',
        "description": 'Moderate-intensity statin therapy is indicated for primary prevention in patients aged 40-75 with Type 1 Diabetes Mellitus.',
        "condition":   lambda cm, pd, yr: _get_dm_type(cm) == "T1DM" and yr >= 2004 and not _get_dm_comp(cm, "cardiovascular") and 40 <= (_get_age(cm, pd) or 0) <= 75,
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ACC-AHA Cholesterol Guideline 2018/2019, Section 4.3, Recommendation 3',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'CPR-82',
        "description": 'In patients with ASCVD, consider add-on Ezetimibe or PCSK9 inhibitors for intensive lipid lowering.',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) in ("T1DM", "T2DM") and yr >= 2015 and _get_dm_comp(cm, "cardiovascular")),
        "any_of_cats": ['Lipid_NonStatins'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ACC-AHA Cholesterol/CV 2018 Sec 3.2.3 & 4.1',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'CPR-84',
        "description": 'Intensive statin therapy should be considered for all patients admitted for ACS.',
        "condition":   lambda cm, pd, yr: _has_acs(pd),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ACC-AHA Cholesterol / Cardiovascular 2004 | Patients With Acute Coronary Syndromes',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'CPR-85',
        "description": 'In T2DM patients with ASCVD, consider Ezetimibe or PCSK9 inhibitors if statin intolerance or further lipid lowering is needed.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2015
            and _get_dm_comp(cm, "cardiovascular")
        ),
        "any_of_cats": ['Lipid_NonStatins'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ACC-AHA Cholesterol 2018 Sec 7.1 / 2016 ACC ECDP',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'judge:uncheckable:?',
    },
    {
        "id":          'CPR-87',
        "description": 'T2DM patients should be treated with statins (considered CHD risk equivalent)',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2004),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ACC-AHA Cholesterol / Cardiovascular 2004 (ATP III Update)',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'CPR-90',
        "description": 'In patients at increased ASCVD risk with chronic, stable liver disease, it is reasonable to use statins.',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2004
            and _get_dm_comp(cm, "cardiovascular")
        ),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       0.5,
        "source":      'ACC-AHA Cholesterol Guideline 2018, Section 5 (Statin Safety), Recommendation 7',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
        "disabled":    True,
        "disabled_reason": 'judge:uncheckable:?',
    },
    {
        "id":          'CPR-94',
        "description": 'T2DM with asymptomatic significantly reduced LVEF should receive ACEi or Beta-blocker to slow HF progression',
        "condition":   lambda cm, pd, yr: (
    _get_dm_type(cm) == "T2DM"
    and not _get_dm_comp(cm, "heart_failure")
    and (_get_lvef(pd) is not None and _get_lvef(pd) < 40)
),
        "any_of_cats": ['Hypertension_ACEi', 'Hypertension_BetaBlockers'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Consensus Reports & Supplements 2022 | Pharmacologic Treatment',
        "domain":      'bp_raas',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'CPR-95',
        "description": 'In individuals with HFpEF, treatment with an SGLT2i is clinically proven therapy to reduce HF hospitalizations.',
        "condition":   lambda cm, pd, yr: (
            yr >= 2020
            and _has_hfpef(cm, pd)
        ),
        "any_of_cats": ['SGLT2i'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      'ADA Consensus Reports & Supplements 2022 | Key Points',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'CPR-99',
        "description": 'In T2DM with CKD stage 3-4 (Very High Risk), consider adding Ezetimibe or PCSK9i for intensive lipid lowering.',
        "condition":   lambda cm, pd, yr: (_has_ckd_g3plus(cm)
            and 
            _get_dm_type(cm) == "T2DM"
            and yr >= 2015
            and (lambda egfr: egfr is not None and 15 <= egfr and egfr <= 59)(_get_egfr(cm))
        ),
        "any_of_cats": ['Lipid_NonStatins'],
        "exclude_drugs": [],
        "grade":       1.0,
        "source":      '',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'CPR-24',
        "description": 'Patients with CKD and heart failure should receive diuretic therapy; CKD should not be a reason to avoid it.',
        "condition":   lambda cm, pd, yr: (
    _get_dm_type(cm) == "T2DM"
    and _has_ckd_g3plus(cm)
    and _get_dm_comp(cm, "heart_failure")
),
        "any_of_cats": ['Hypertension_Diuretics'],
        "exclude_drugs": [],
        "grade":       0.5,
        "source":      'KDIGO CKD & Diabetes-CKD Guidelines 2012 - RATIONALE',
        "domain":      'bp_other',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'CPR-09',
        "description": 'T1DM + age>=40 + yr>=2001: statin primary prevention (HPS 2001, ADA 2010)',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T1DM"
            and yr >= 2001
            and (_get_age(cm, pd) or 0) >= 40
            and not _is_hospice(pd)
        ),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       0.5,
        "source":      'ADA Standards of Medical Care 2010 - Lipid Management; HPS 2001 T1DM subgroup; CPR-09',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'CPR-23',
        "description": 'CKD + ASCVD (any DM incl T1DM) + yr>=2012: antiplatelet secondary prevention (KDIGO 2012 2B)',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) in ("T1DM", "T2DM") and yr >= 2012 and _has_ckd_g3plus(cm) and _get_dm_comp(cm, "cardiovascular") and not _is_hospice(pd)),
        "any_of_cats": ['Antiplatelet'],
        "exclude_drugs": [],
        "grade":       0.5,
        "source":      'KDIGO CKD Guidelines 2012 | 4.1.3 Antiplatelet agents (2B); CPR-23',
        "domain":      'ascvd_antiplatelet',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'CPR-83',
        "description": 'T2DM + CKD G3+ + yr>=2018: non-statin lipid lowering (ezetimibe; SHARP 2011, ACC-AHA 2018)',
        "condition":   lambda cm, pd, yr: (
    _get_dm_type(cm) == "T2DM"
    and yr >= 2018
    and _has_ckd_g3plus(cm)
    and not _is_hospice(pd)
),
        "any_of_cats": ['Lipid_NonStatins'],
        "exclude_drugs": [],
        "grade":       0.5,
        "source":      'ACC-AHA Cholesterol Guideline 2018 Sec 4.4.2; SHARP trial 2011; CPR-83',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'CPR-88',
        "description": 'T2DM + age 20-39 + UACR>=30 + yr>=2018: statin primary prevention (ACC-AHA 2018 Table 5 IIa)',
        "condition":   lambda cm, pd, yr: (_get_dm_type(cm) == "T2DM" and yr >= 2018 and 20 <= (_get_age(cm, pd) or 0) <= 39 and _has_albuminuria(cm) and not _is_hospice(pd)),
        "any_of_cats": ['Lipid_Statins'],
        "exclude_drugs": [],
        "grade":       0.5,
        "source":      'ACC-AHA Cholesterol Guideline 2018 Section 4.3 & Table 5 (IIa); CPR-88',
        "domain":      'lipid_management',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'CPR-105',
        "description": 'T2DM + CKD G3+ + yr>=2016: GLP-1RA for renal/CV protection (LEADER 2016, CN guideline, CPR-105)',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2016
            and _has_ckd_g3plus(cm)
            and not _is_hospice(pd)
        ),
        "any_of_cats": ['GLP-1RA', 'GIP_GLP1_RA'],
        "exclude_drugs": [],
        "grade":       0.5,
        "source":      'ESC CVD 2023 Sec 9; CPR-105',
        "domain":      'glycemic_intensification',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
    {
        "id":          'CPR-109',
        "description": 'T2DM + CKD eGFR<30 + yr>=2022: Finerenone as RASi alternative (CN Consensus 2023, CPR-109)',
        "condition":   lambda cm, pd, yr: (
            _get_dm_type(cm) == "T2DM"
            and yr >= 2022
            and _has_ckd_g3plus(cm)
            and (lambda egfr: egfr is not None and float(egfr) < 30)(_get_egfr(cm))
            and not _is_hospice(pd)
        ),
        "any_of_cats": ['CKD_MRA'],
        "exclude_drugs": ['spironolactone', 'eplerenone'],
        "grade":       0.5,
        "source":      'KDIGO 2022; CPR-109',
        "domain":      'ckd_mra',
        "polarity":    'RECOMMEND',
        "scorable":    True,
    },
]




# ═══════════════════════════════════════════════════════════════════════════════
# 五、OSEMetrics 数据结构(v4 扩展)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class OSEMetrics:
    """单患者 OSE 指标(v4:双路径 GAS)。"""
    outcome_label:   str   = "Unknown"
    severity_tier:   str   = "Unknown"
    charlson_cci:    Optional[float] = None
    lab_sofa_proxy:  Optional[float] = None

    # GAS 最终分(双路径加权融合)
    gas_score:        float = 0.0
    gas_grade_a_hits: int   = 0
    gas_grade_b_hits: int   = 0
    gas_era_penalty:  float = 0.0

    # GAS 路径分解(v4 双路径,诊断/论文 Table 用)
    gas_dm_score:     float = 0.0   # Path-DM: DM 规则满足率
    gas_comp_score:   float = 0.0   # Path-C:  并发症规则满足率
    gas_n_dm_rules:   int   = 0     # 适用的 DM 规则数
    gas_n_comp_rules: int   = 0     # 适用的并发症规则数

    # DM Rules 明细(溯源用)
    gas_dm_applied:   str = ""      # "|" 分隔的适用 DM 规则 ID
    gas_dm_satisfied: str = ""      # "|" 分隔的满足 DM 规则 ID

    # Path-C 明细(溯源用)
    gas_comp_applied:   str = ""    # "|" 分隔的适用并发症规则 ID
    gas_comp_satisfied: str = ""    # "|" 分隔的满足并发症规则 ID

    # GAS v5 双轴(域级 Concordance × Appropriateness)
    gas_concordance:     float = 0.0   # 域覆盖 / recall
    gas_appropriateness: float = 1.0   # 处方适当性 / precision
    gas_f:               float = 0.0   # 两轴调和均值 F = 2CA/(C+A)
    gas_commission_rules: str = ""     # [接入] "|" 分隔 commission 规则ID(供分专科 appropriateness)
    gas_macro: float = 0.0             # [接入] 专科等权 macro concordance
    gas_concordance_by_specialty: str = ""   # [接入] JSON: {专科: concordance}

    # GAS mode_b(真实世界预测路径,用于 CII gap 量化)
    # [FIX-1] 修复:原版只计算 mode_a,CII gap 无法正确量化
    gas_score_mode_b: float = 0.0   # Path-DM/C 对 mode_b 处方的评分
    cii_gap:          float = 0.0   # = gas_score(mode_a) - gas_score(mode_b)
                                    # > 0 表示 AI 最优解比模拟临床处方更合规

    # CII(定性标记,基于 mode_a 空处方 + HbA1c 检测)
    is_clinical_inertia: bool = False
    cii_evidence:        str  = ""

    # Rx Fidelity
    rx_fidelity:             float = 0.0
    prescription_divergence: float = 0.0

    # 时代穿越
    temporal_violation_count: int       = 0
    temporal_violation_drugs: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# 六、OSECalculator(v4)
# ═══════════════════════════════════════════════════════════════════════════════

class OSECalculator:
    """OSE 指标计算器(v4:双路径 GAS,降糖药 + 并发症管理)。"""

    def __init__(self, normalizer: "DrugNormalizer"):
        self.normalizer = normalizer

    def calculate(
        self,
        patient_metrics:  "PatientMetrics",
        rec_result:       Dict,
        patient_data:     Dict,
        outcome_entry:    Optional[Dict],
    ) -> OSEMetrics:
        ose = OSEMetrics()

        if outcome_entry:
            ose.outcome_label  = str(outcome_entry.get("outcome_label", "Unknown"))
            ose.severity_tier  = str(outcome_entry.get("severity_tier", "Unknown"))
            ose.charlson_cci   = _safe_float(outcome_entry.get("charlson_cci"))
            ose.lab_sofa_proxy = _safe_float(outcome_entry.get("lab_sofa_proxy"))

        (
            ose.gas_score,
            ose.gas_grade_a_hits,
            ose.gas_grade_b_hits,
            ose.gas_era_penalty,
            ose.gas_dm_score,
            ose.gas_comp_score,
            ose.gas_n_dm_rules,
            ose.gas_n_comp_rules,
            ose.gas_dm_applied,
            ose.gas_dm_satisfied,
            ose.gas_comp_applied,
            ose.gas_comp_satisfied,
            ose.gas_concordance,
            ose.gas_appropriateness,
            ose.gas_f,
            ose.gas_commission_rules,
            ose.gas_macro,
            ose.gas_concordance_by_specialty,
        ) = self._compute_gas(rec_result, patient_metrics, patient_data, mode="mode_a")

        # ── GAS mode_b(CII gap 计算)──────────────────────────────────────────
        # [FIX-4] dual_mode_raw=None 时退化为 mode_a,gap=0(无 CII 信息)
        _dual = (rec_result.get("diamas_metadata", {}) or {}).get("dual_mode_raw")
        if _dual:
            ose.gas_score_mode_b = self._compute_gas(
                rec_result, patient_metrics, patient_data, mode="mode_b"
            )[0]
        else:
            ose.gas_score_mode_b = ose.gas_score  # 无 dual_mode_raw,两者等价

        # cii_gap > 0 → AI 指南最优解依从度高于模拟临床处方 → CII 改进空间
        ose.cii_gap = round(max(0.0, ose.gas_score - ose.gas_score_mode_b), 4)

        ose.is_clinical_inertia, ose.cii_evidence = (
            self._detect_clinical_inertia(patient_data, patient_metrics)
        )

        ose.temporal_violation_count, ose.temporal_violation_drugs = (
            self._detect_temporal_violations(rec_result, patient_data)
        )

        ose.rx_fidelity             = patient_metrics.drug_f1
        ose.prescription_divergence = 1.0 - ose.rx_fidelity

        return ose

    # ── 双路径 GAS 核心计算 ────────────────────────────────────────────────────

    def _compute_gas(
        self,
        rec_result:      Dict,
        patient_metrics: "PatientMetrics",
        patient_data:    Dict,
        mode:            str = "mode_a",
    ) -> Tuple[float, int, int, float, float, float, int, int, str, str, str, str, float, float, float]:
        """
        双路径 GAS 计算(v4+v5)。

        Path-DM: 规则化降糖药依从性(DM_RULES)
        Path-C:  并发症药物条件性评分(COMPLICATION_DRUG_GRADES,不变)

        公式:
          GAS_dm   = Σ(satisfied DM rule weights) / Σ(applicable DM rule weights)
          GAS_comp = Σ(satisfied CPR weights)     / Σ(applicable CPR weights)
          GAS_raw  = (n_dm × GAS_dm + n_comp × GAS_comp) / (n_dm + n_comp)
          GAS      = max(0, min(1, GAS_raw - safety_penalty))

        Returns:
          (gas_final, grade_a_hits, grade_b_hits, era_penalty,
           gas_dm_raw, gas_comp_raw,
           n_dm_rules, n_comp_rules,
           dm_applied_str, dm_satisfied_str,
           comp_applied_str, comp_satisfied_str,
           gas_concordance, gas_appropriateness, gas_f)
        """
        NULL = (0.0, 0, 0, 0.0, 0.0, 0.0, 0, 0, "", "", "", "", 0.0, 1.0, 0.0, "", 0.0, "")

        # Layer 0:critical 安全违反 → GAS=0
        if patient_metrics.n_violations_critical > 0:
            return NULL

        # ── 读取 clinical_metrics 和入院年份 ──────────────────────────────────
        cm = dict(patient_data.get("clinical_metrics") or {})

        # [Y1 FIX] 原实现直接取 admission_time[:4],而 MIMIC 该字段是脱敏偏移年
        # (MIMIC-IV ~2158, MIMIC-III ~2117) -> 798/834 条 `yr >= N` 门控全部通过,
        # 历史患者被 2023-2026 年的规则评分(其中 73 条阈值 >2022)。
        # 现统一走 safety_checker.resolve_admission_year(单一权威来源):
        #   MIMIC-IV -> anchor_year_group 末年 + (入院移位年份 - anchor_year);MIMIC-III -> 2008;eICU -> 真实年份。
        admission_year = resolve_admission_year(patient_data)
        if admission_year:
            cm["admission_year"] = admission_year
        else:
            # 解析失败:保持原有回退语义(9999 => 不触发时代穿越误判)
            admission_year = cm.get("admission_year", 9999)

        # ═══ [PDATA-ISOLATION] 打分幂等性修复 ═══════════════════════════════
        # _build_rule_eval_context 会把推断出的字段回写 patient_data 顶层
        #   (on_systemic_steroids / has_hypertension / on_iv_insulin / _icd_codes_cache),
        # 而各推断块的守卫是 `if out.get("X") is None:`。回写会在**第二次**打分时
        # 让守卫短路 → out["X"] 不再被设置 → 依赖该字段的规则不触发 → 域退出分母
        # → (若该域原为 OMITTED)反而抬高 concordance/gas_f。
        # 实测:同一 rec 第 1 次 gas_f=0.6514,第 2 次起 =0.6738(+0.0224)。
        #
        # 回写本身**不能删** —— DR-09 / CPR-71 等规则的 lambda 直接读
        # pd.get("has_hypertension"),回写是让本次调用内这些规则可见的必要手段。
        # 故改为:对顶层做浅拷贝,回写落在本次调用的副本上,调用方对象保持原始。
        # 实测键级 diff 显示污染全部发生在顶层(无既有键被改写),浅拷贝足够。
        patient_data = dict(patient_data or {})

        # 规则评估上下文:补齐 run_extraction 实际存储结构的别名键
        rule_cm = _build_rule_eval_context(cm, patient_data, admission_year)

        # ── 解析推荐药物 ───────────────────────────────────────────────────────
        rec_drug_entries: List[Tuple[str, str]] = []
        for r in _extract_drug_list_from_rec(rec_result, mode=mode):
            if isinstance(r, dict):
                name = str(r.get("drug", "") or r.get("name", "")).strip().lower()
                cat  = str(r.get("category", "")).strip()   # LLM 原始 category(备用)
            elif isinstance(r, str):
                name = r.strip().lower()
                cat  = ""
            else:
                continue
            if not name:
                continue
            # ── FIX: 始终调用 normalizer,taxonomy 分类拥有最终决定权 ──────────
            # LLM 输出 category 是自然语言("Basal insulin"),
            # DM_RULES / EVAL_DM_ONLY_CATEGORIES 用 taxonomy 枚举("Insulins_Basal")。
            # 只有 normalizer 能把 INN 名映射到 taxonomy,所以不能以 cat 非空为由跳过。
            mapping = self.normalizer.normalize(name)
            if mapping.drug_category:
                cat = mapping.drug_category        # taxonomy 优先("Insulins_Basal")
            # else: cat 保留 LLM 原始值,至少能参与 Path-C 的宽松匹配
            rec_drug_entries.append((name, cat))

        if not rec_drug_entries:
            return NULL

        # ── Layer 1:患者特征安全门控 ────────────────────────────────────────
        gated_out: List[str] = []
        safe_entries: List[Tuple[str, str]] = []
        for drug_name, drug_cat in rec_drug_entries:
            gated = False
            for gate in _GAS_PATIENT_GATES:
                try:
                    if gate["drug_match"](drug_name, drug_cat) and gate["condition"](rule_cm):
                        gated = True
                        gated_out.append(f"{drug_name}[{gate['gate_id']}]")
                        break
                except Exception:
                    pass
            if not gated:
                safe_entries.append((drug_name, drug_cat))
        if gated_out:
            logger.debug("GAS Layer-1 gate-out (%d): %s", len(gated_out), gated_out)

        # [TEMPORAL-CONSISTENCY-V1] prescription-side availability mask.
        # A not-yet-available drug remains visible to the independent temporal
        # violation detector, but must not receive GAS credit or enter its denominator.
        _era_safe_entries: List[Tuple[str, str]] = []
        _temporal_gated_out: List[str] = []
        for _drug_name, _drug_cat in safe_entries:
            try:
                _m = self.normalizer.normalize(_drug_name)
                _availability_name = str(
                    getattr(_m, "mapped_name", None) or _drug_name
                ).strip()
                _approval = DrugEraConfig.get_drug_approval_year(_availability_name)
                if _approval is not None and int(_approval) > int(admission_year):
                    _temporal_gated_out.append(
                        f"{_availability_name}[available={_approval},year={admission_year}]"
                    )
                    continue
            except Exception:
                pass  # preserve historical conservative fail-open behavior
            _era_safe_entries.append((_drug_name, _drug_cat))
        safe_entries = _era_safe_entries
        if _temporal_gated_out:
            logger.debug(
                "GAS temporal availability gate-out (%d): %s",
                len(_temporal_gated_out), _temporal_gated_out
            )

        # ── GAS v5:域级双轴评分(Concordance × Appropriateness)——唯一评分口径 ──
        # (v4 双路径 _family_dedup_eval / _evaluate_dm_rules / _evaluate_path_c 已移除;
        #  gas_f 即最终 GAS。slot0 现返回 gas_f 供 gas_score / cii_gap 使用。)
        concordance_v5 = 0.0
        appropriateness_v5 = 1.0
        gas_f_v5 = 0.0
        commission_ids_v5 = ""
        gas_macro_v5 = 0.0
        spec_conc_v5 = ""
        try:
            from evaluation.gas_v5_domain_eval import domain_eval
            ALL_RULES_V5 = DM_RULES + DM_DRUG_CONDITIONS + COMPLICATION_DRUG_GRADES
            cat_to_drug_names = {}
            for name, cat in safe_entries:
                if cat:
                    cat_to_drug_names.setdefault(cat, []).append(name)
            r5 = domain_eval(ALL_RULES_V5, cat_to_drug_names, rule_cm, patient_data, admission_year)
            concordance_v5 = r5.concordance
            appropriateness_v5 = r5.appropriateness
            gas_f_v5 = r5.gas_f
            commission_ids_v5 = "|".join(sorted({rid for rid, _drug in r5.commission_violations}))
            gas_macro_v5 = float(r5.gas_macro) if r5.gas_macro is not None else 0.0
            spec_conc_v5 = _json.dumps(r5.concordance_by_specialty or {}, ensure_ascii=False)
        except Exception as e5:
            # [P0 OBSERVABILITY FIX] 原为 logger.debug,导致 v5 打分整体失败时
            # concordance_v5 静默保持 0.0 -> gas_f=0 -> 全体患者 GAS=0,日志无任何提示。
            # 现改为:首次失败打 ERROR + traceback(只打一次),并累计计数。
            import os as _os
            import traceback as _tb
            _g = globals()
            _n = _g.get("_GAS_V5_FAIL_COUNT", 0) + 1
            _g["_GAS_V5_FAIL_COUNT"] = _n
            if _n == 1:
                logger.error(
                    "GAS v5 domain_eval FAILED (%s: %s) -> concordance 回退 0.0,"
                    " 该患者 gas_f 将为 0。完整 traceback:\n%s",
                    type(e5).__name__, e5, _tb.format_exc(),
                )
                logger.error(
                    "若此错误对全体患者持续出现,physician_gas_final 将恒为 0"
                    "(Good/Bad 无差异, Cliff's d=0)。请先修复本异常再解读任何 GAS 结果。"
                )
            elif _n % 5000 == 0:
                logger.error("GAS v5 domain_eval 已累计失败 %d 次(仍在回退 0.0)", _n)
            if _os.environ.get("GAS_V5_STRICT") == "1":
                raise

        logger.debug("GAS v5: C=%.3f A=%.3f F=%.3f commission=%s",
                     concordance_v5, appropriateness_v5, gas_f_v5, commission_ids_v5)

        # ── 激活规则 ID(applicable = 未禁用且 condition 为真)──────────────────────
        # v4 的 gas_dm_applied / gas_comp_applied 被 temporal_validity_v3.applied_rule_ids()
        # 消费(判断病人是否命中"时代变革规则")。v4 引擎已删,故在此用 v5 同一规则集与
        # 同一 rule_cm 重新计算 applicability,保持该字段语义不变。satisfied 字段不再单独
        # 追踪(v5 用 domain_status / commission 表达),留空。
        _dm_applied: List[str] = []
        _comp_applied: List[str] = []
        for _rlist, _bucket in (
            (DM_RULES, _dm_applied),
            (DM_DRUG_CONDITIONS, _dm_applied),
            (COMPLICATION_DRUG_GRADES, _comp_applied),
        ):
            for _r in _rlist:
                if _r.get("disabled"):
                    continue
                try:
                    if _r["condition"](rule_cm, patient_data, admission_year):
                        _bucket.append(_r["id"])
                except Exception:
                    pass
        dm_applied_v5 = "|".join(_dm_applied)
        comp_applied_v5 = "|".join(_comp_applied)

        # slot0 = gas_f(最终 GAS);v4-only 诊断槽位保留占位(0/"")兼容 OSEMetrics 解包。
        return (
            round(gas_f_v5, 4),              # gas_score  ← v5 gas_f(v4 已删)
            0,                               # gas_grade_a_hits (v4 弃用)
            0,                               # gas_grade_b_hits (v4 弃用)
            0.0,                             # gas_era_penalty  (弃用)
            0.0,                             # gas_dm_score     (设计中已不使用)
            0.0,                             # gas_comp_score   (设计中已不使用)
            len(_dm_applied),                # gas_n_dm_rules   ← v5 applicable 计数
            len(_comp_applied),              # gas_n_comp_rules ← v5 applicable 计数
            dm_applied_v5,                   # gas_dm_applied   ← v5 applicable IDs(temporal 消费)
            "",                              # gas_dm_satisfied (v5 用 domain_status 表达,留空)
            comp_applied_v5,                 # gas_comp_applied ← v5 applicable IDs(temporal 消费)
            "",                              # gas_comp_satisfied(留空)
            round(concordance_v5, 4),        # gas_concordance
            round(appropriateness_v5, 4),    # gas_appropriateness
            round(gas_f_v5, 4),              # gas_f
            commission_ids_v5,               # gas_commission_rules
            gas_macro_v5,                    # gas_macro
            spec_conc_v5,                    # gas_concordance_by_specialty
        )

    # ── CII 检测(继承 v2)────────────────────────────────────────────────────

    def _detect_clinical_inertia(
        self,
        patient_data:    Dict,
        patient_metrics: "PatientMetrics",
    ) -> Tuple[bool, str]:
        if patient_metrics.actual_drugs:
            return False, ""

        cm    = patient_data.get("clinical_metrics", {}) or {}
        hba1c = cm.get("hba1c")
        egfr  = _get_egfr(cm)

        try:
            hba1c_val = float(hba1c) if hba1c is not None else None
        except (ValueError, TypeError):
            hba1c_val = None

        if hba1c_val is None or hba1c_val < 8.0:
            return False, ""
        if egfr is not None and egfr < 15:
            return False, "eGFR<15, empty Rx acceptable (ESRD)"

        try:
            from evaluation.evaluation_extensions import EmptyGTType
            if patient_metrics.empty_gt_type == EmptyGTType.SAFE_DEESCALATION.value:
                return False, "Safe de-escalation, empty Rx acceptable"
        except ImportError:
            pass

        evidence = f"HbA1c={hba1c_val:.1f}% (>8.0%), empty discharge Rx -> suspected clinical inertia"
        if egfr is not None:
            evidence += f", eGFR={egfr}"
        return True, evidence

    # ── 时代穿越检测(继承 v2 FIX-3)─────────────────────────────────────────

    def _detect_temporal_violations(
        self,
        rec_result:   Dict,
        patient_data: Dict,
    ) -> Tuple[int, List[str]]:
        # [FIX 2026-09-07] 原实现只读 anchor_year/admission_year/admission_time[:4]。
        # MIMIC 前两者恒为 None，第三者是去标识化偏移年份（如 2193/2164）⇒
        # admission_year < approval_year 结构上永假 ⇒ 检测在全部 MIMIC 病人上失效
        # （实测：违规 4,401 例全部来自 eICU，mimic3/mimic4 精确 0）。
        # 改用 safety_checker.resolve_admission_year()（MIMIC-IV 取 anchor_year_group 末年 +
        # 入院移位差 / MIMIC-III 取 2008 / eICU 取真实年份），与全系统赋年口径同源。
        # 注：本字段不参与 gas_f 计算（见 :14802，赋值在 _compute_gas 之后），
        # 故本修复不改变任何已发表 GAS 分数。
        from evaluation.safety_checker import resolve_admission_year
        try:
            admission_year = resolve_admission_year(patient_data)
        except Exception:
            admission_year = None
        if admission_year is None:
            return 0, []

        # [TEMPORAL-CONSISTENCY-V1] Use the same recommendation extractor,
        # normalizer and DrugEraConfig registry as GAS/generation-time filtering.
        violations: List[str] = []
        seen: Set[str] = set()
        for r in _extract_drug_list_from_rec(rec_result, mode="mode_a"):
            if isinstance(r, dict):
                raw_drug = str(
                    r.get("drug", "") or r.get("name", "") or r.get("drug_name", "")
                ).strip()
            else:
                raw_drug = str(r).strip()
            if not raw_drug:
                continue

            try:
                mapping = self.normalizer.normalize(raw_drug)
                std_drug = str(getattr(mapping, "mapped_name", None) or raw_drug).strip()
            except Exception:
                std_drug = raw_drug

            key = std_drug.lower()
            if key in seen:
                continue
            seen.add(key)

            approval_year = DrugEraConfig.get_drug_approval_year(std_drug)
            if approval_year is not None and int(approval_year) > int(admission_year):
                violations.append(
                    f"{std_drug.lower()} (available {approval_year}, patient year {admission_year})"
                )

        return len(violations), violations


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def _safe_float(v) -> Optional[float]:
    try:
        return float(v) if v is not None and str(v) not in ("nan", "None", "") else None
    except (ValueError, TypeError):
        return None
