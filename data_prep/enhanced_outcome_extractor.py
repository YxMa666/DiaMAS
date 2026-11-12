#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强的Outcome数据提取器 v2.1
============================

版本变更 (v2.0 → v2.1):
  关键Bug修复:
    [FIX-P0] extract_all_outcomes 逐 hadm 提取：原逻辑每个 subject 仅提取最新
             一次住院的结局，多次住院的早期 hadm 在 outcome CSV 中无记录，导致
             pilot_sampler 标记为 Unknown（直接原因：46.6% Unknown 率）。
             修复后对每个 hadm 独立调用 extract_patient_outcomes([single_hid])，
             输出 CSV 从「每 subject 一行」变为「每 hadm 一行」。

  临床修正:
    [FIX-C1] BAD_DISCHARGE_KEYWORDS 移除 'snf'：SNF 是 post-acute care 正常
             过渡路径（CMS 定义），不应归为不良出院去向
    [FIX-C2] severity_tier SOFA 升档阈值从 ≥8 修正为 ≥4：排除标准已将 SOFA ≥8
             移出队列，原阈值 ≥8 为永远无法触发的死代码

  新增敏感性分析列:
    [SA-1]   bad_outcome_flags:           触发 Bad 判定的具体组件（逗号分隔）
    [SA-2]   outcome_label_core:          排除 AKI 进展后的核心复合终点
    [SA-3]   outcome_label_expanded_mod:  TIR<50% + CV>36% 的 Good 升级为 Moderate

  重构:
    [REFACTOR] generate() 内部委托给 generate_with_flags()，保持接口向后兼容
    [REFACTOR] _apply_sensitivity_columns() 统一三数据库的标签生成逻辑

版本变更 (v1.0 → v2.0):
  Bug修复:
    [FIX-1] MIMIC-III 院内死亡：dod_hosp 非空不等于本次住院死亡，加入时间窗校验
    [FIX-2] MIMIC-IV  院内死亡：dod 时间窗校验（原逻辑实际是正确的，保持但增加注释）
    [FIX-3] MIMIC-III/IV 再入院：admissions 已 set_index('hadm_id')，subject_id 查询逻辑修正
    [FIX-4] eICU 再入院：原代码直接赋 None，现通过 uniquepid 追踪多次 ICU 入住
    [FIX-5] eICU DKA检测：icd9code 大量为 NULL，新增 diagnosisstring 自由文本扫描

  新增变量:
    [NEW-1]  Charlson 共病指数（CCI）：PSM 核心控制变量，支持 ICD-9 和 ICD-10
    [NEW-2]  Lab-SOFA 代理评分：肌酐+胆红素+血小板三组件，用于急性严重度控制
    [NEW-3]  出院去向（discharge_destination）：功能结局代理指标
    [NEW-4]  TIR（Time in Range 70-180 mg/dL）：糖尿病专属血糖质量指标
    [NEW-5]  血糖变异系数（CV%）和高血糖负担（TAR）
    [NEW-6]  AKI 动态分期（KDIGO）：肌酐轨迹计算，比 ICD 诊断码更敏感
    [NEW-7]  28日死亡率：比院内死亡更敏感，捕获出院后早期死亡
    [NEW-8]  胰岛素使用强度（基础版）：血糖管理积极性的直接代理
    [NEW-9]  eGFR 入院基线（CKD-EPI 2021）

作者：Yuxi（v2.1 临床修正 + 敏感性分析扩展）

Run from the repository root after data_prep/run_extraction.py has written the patient lists:
  python data_prep/enhanced_outcome_extractor.py
Reads raw tables from PathConfig.MIMIC3_PATH / MIMIC4_PATH / EICU_PATH and writes
diabetes_subsets/<db>_outcomes.csv (one row per admission).
"""

import json
import re
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from abc import ABC, abstractmethod
import logging
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repository root (agents.config)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 一、Charlson 共病指数 ICD 映射表（Quan et al. 2005 适配版）
# ═══════════════════════════════════════════════════════════════════════════════

# ICD-9：前缀列表（无点号格式，匹配时两侧统一去点后做 startswith）
# 格式：{condition_name: ([prefix_list], weight)}
_CCI_ICD9 = {
    'MI':               (['410', '412'],                                     1),
    'CHF':              (['428'],                                             1),
    'PVD':              (['440', '441', '4439', '7854', 'V434'],             1),
    'CVD':              (['430', '431', '432', '433', '434', '435',
                          '436', '437', '438'],                              1),
    'Dementia':         (['290'],                                             1),
    'COPD':             (['490', '491', '492', '493', '494', '495',
                          '496', '500', '501', '502', '503', '504',
                          '505', '5064'],                                    1),
    'Rheumatic':        (['7100', '7101', '7104', '7140', '7141',
                          '7142', '71481', '725'],                           1),
    'PUD':              (['531', '532', '533', '534'],                       1),
    'MildLiver':        (['5712', '5714', '5715', '5716',
                          '5733', '5734', '5738', '5739'],                  1),
    'DM_noCC':          (['2500', '2501', '2502', '2503',
                          '2508', '2509'],                                   1),
    'DM_CC':            (['2504', '2505', '2506', '2507'],                   2),
    'Hemiplegia':       (['342', '3441'],                                    2),
    'RenalDisease':     (['582', '583', '585', '586', '5880',
                          'V420', 'V451', 'V56'],                           2),
    'Cancer':           (['140', '141', '143', '144', '145', '146',
                          '147', '148', '149', '150', '151', '152',
                          '153', '154', '155', '156', '157', '158',
                          '159', '160', '161', '162', '163', '164',
                          '165', '166', '167', '168', '169', '170',
                          '171', '172', '174', '175', '176', '177',
                          '178', '179', '180', '181', '182', '183',
                          '184', '185', '186', '187', '188', '189',
                          '190', '191', '192', '193', '194', '195',
                          '200', '201', '202', '203', '204', '205',
                          '206', '207', '208'],                             2),
    'ModSevereLiver':   (['4560', '4561', '4562', '5722', '5723',
                          '5724', '5725', '5726', '5727', '5728'],          3),
    'MetastaticCancer': (['196', '197', '198', '1990', '1991'],             6),
    'AIDS':             (['042', '043', '044'],                              6),
}

# ICD-10：Quan et al. 2005 + van Walraven 更新
_CCI_ICD10 = {
    'MI':               (['I21', 'I22', 'I252'],                             1),
    'CHF':              (['I099', 'I110', 'I130', 'I132', 'I255',
                          'I420', 'I425', 'I426', 'I427', 'I428',
                          'I429', 'I43', 'I50', 'P290'],                    1),
    'PVD':              (['I70', 'I71', 'I731', 'I738', 'I739',
                          'I771', 'I790', 'I792', 'K551', 'K558',
                          'K559', 'Z958', 'Z959'],                          1),
    'CVD':              (['G45', 'G46', 'H340', 'I60', 'I61',
                          'I62', 'I63', 'I64', 'I65', 'I66',
                          'I67', 'I68', 'I69'],                             1),
    'Dementia':         (['F00', 'F01', 'F02', 'F03', 'F051',
                          'G30', 'G311'],                                    1),
    'COPD':             (['I278', 'I279', 'J40', 'J41', 'J42',
                          'J43', 'J44', 'J45', 'J46', 'J47',
                          'J60', 'J61', 'J62', 'J63', 'J64',
                          'J65', 'J66', 'J67', 'J684', 'J701', 'J703'],    1),
    'Rheumatic':        (['M05', 'M06', 'M315', 'M32', 'M33',
                          'M34', 'M351', 'M353', 'M360'],                   1),
    'PUD':              (['K25', 'K26', 'K27', 'K28'],                      1),
    'MildLiver':        (['B18', 'K700', 'K701', 'K702', 'K703',
                          'K709', 'K713', 'K714', 'K715', 'K717',
                          'K73', 'K74', 'K760', 'K762', 'K763',
                          'K764', 'K768', 'K769', 'Z944'],                  1),
    'DM_noCC':          (['E100', 'E101', 'E106', 'E108', 'E109',
                          'E110', 'E111', 'E116', 'E118', 'E119',
                          'E120', 'E121', 'E126', 'E128', 'E129',
                          'E130', 'E131', 'E136', 'E138', 'E139',
                          'E140', 'E141', 'E146', 'E148', 'E149'],         1),
    'DM_CC':            (['E102', 'E103', 'E104', 'E105', 'E107',
                          'E112', 'E113', 'E114', 'E115', 'E117',
                          'E122', 'E123', 'E124', 'E125', 'E127',
                          'E132', 'E133', 'E134', 'E135', 'E137',
                          'E142', 'E143', 'E144', 'E145', 'E147'],         2),
    'Hemiplegia':       (['G041', 'G114', 'G801', 'G802', 'G81',
                          'G82', 'G830', 'G831', 'G832', 'G833',
                          'G834', 'G839'],                                  2),
    'RenalDisease':     (['I120', 'I131', 'N032', 'N033', 'N034',
                          'N035', 'N036', 'N037', 'N052', 'N053',
                          'N054', 'N055', 'N056', 'N057', 'N18',
                          'N19', 'N250', 'Z490', 'Z491', 'Z492',
                          'Z940', 'Z992'],                                  2),
    'Cancer':           (['C00', 'C01', 'C02', 'C03', 'C04', 'C05',
                          'C06', 'C07', 'C08', 'C09', 'C10', 'C11',
                          'C12', 'C13', 'C14', 'C15', 'C16', 'C17',
                          'C18', 'C19', 'C20', 'C21', 'C22', 'C23',
                          'C24', 'C25', 'C26', 'C30', 'C31', 'C32',
                          'C33', 'C34', 'C37', 'C38', 'C39', 'C40',
                          'C41', 'C43', 'C45', 'C46', 'C47', 'C48',
                          'C49', 'C50', 'C51', 'C52', 'C53', 'C54',
                          'C55', 'C56', 'C57', 'C58', 'C60', 'C61',
                          'C62', 'C63', 'C64', 'C65', 'C66', 'C67',
                          'C68', 'C69', 'C70', 'C71', 'C72', 'C73',
                          'C74', 'C75', 'C76', 'C81', 'C82', 'C83',
                          'C84', 'C85', 'C88', 'C90', 'C91', 'C92',
                          'C93', 'C94', 'C95', 'C96', 'C97'],             2),
    'ModSevereLiver':   (['I850', 'I859', 'I864', 'I982', 'K704',
                          'K711', 'K721', 'K729', 'K765', 'K766', 'K767'], 3),
    'MetastaticCancer': (['C77', 'C78', 'C79', 'C80'],                     6),
    'AIDS':             (['B20', 'B21', 'B22', 'B24'],                      6),
}

# ─── Lab ItemIDs ──────────────────────────────────────────────────────────────
_MIMIC3_LAB = {
    'hba1c':         [50852, 51002],
    'glucose':       [50809, 50931],
    'creatinine':    [50912],
    'bun':           [51006],
    'bilirubin':     [50885],
    'platelet':      [51265],
    'pao2':          [50821],
    'fio2':          [50816],
}

_MIMIC4_LAB = {
    'hba1c':         [51222, 52027],
    'glucose':       [50809, 50931],
    'creatinine':    [50912, 52546],
    'bun':           [51006],
    'bilirubin':     [50885, 52730],
    'platelet':      [51265, 51704],
    'pao2':          [50821],
    'fio2':          [50816],
}

# Vasopressor itemids（MIMIC-III inputevents_cv/mv 以及 MIMIC-IV inputevents）
_VASOPRESSOR_ITEMIDS_M3 = {220045, 30047, 30120, 221906, 30044, 30119, 30309}
_VASOPRESSOR_ITEMIDS_M4 = {221906, 221289, 222315, 30044, 30119}

# DKA 关键词（用于 eICU diagnosisstring 自由文本匹配）
_DKA_KEYWORDS = [
    'diabetic ketoacidosis', 'dka', 'ketoacidosis',
    'diabetes.*ketoacid', 'hyperglycemic crisis',
]
_DKA_PATTERN = re.compile('|'.join(_DKA_KEYWORDS), re.IGNORECASE)

_HHS_KEYWORDS = ['hyperosmolar', 'hhs', 'hyperglycemic hyperosmolar', 'hhns']
_HHS_PATTERN = re.compile('|'.join(_HHS_KEYWORDS), re.IGNORECASE)


# ═══════════════════════════════════════════════════════════════════════════════
# 二、通用工具函数
# ═══════════════════════════════════════════════════════════════════════════════

def _norm_icd(code: str) -> str:
    """标准化 ICD 编码：去点号、大写、去空白。"""
    return str(code).replace('.', '').upper().strip()


def _matches_prefix(norm_code: str, prefixes: List[str]) -> bool:
    """检查标准化 ICD 编码是否匹配任意前缀（前缀亦已标准化）。"""
    for p in prefixes:
        if norm_code.startswith(p):
            return True
    return False


def compute_charlson_icd9(icd_codes: List[str]) -> int:
    """
    计算 Charlson 共病指数（ICD-9 版）。
    使用 Quan et al. 2005 适配版映射。

    Args:
        icd_codes: ICD-9 编码列表（含或不含点号均可）
    Returns:
        CCI 整数分数
    """
    norm_codes = [_norm_icd(c) for c in icd_codes if c and str(c).lower() != 'nan']

    found = set()
    for cond, (prefixes, _) in _CCI_ICD9.items():
        for nc in norm_codes:
            if _matches_prefix(nc, prefixes):
                found.add(cond)
                break

    # 互斥去重：较重的分类覆盖较轻的
    if 'DM_CC' in found:
        found.discard('DM_noCC')
    if 'MetastaticCancer' in found:
        found.discard('Cancer')
    if 'ModSevereLiver' in found:
        found.discard('MildLiver')

    return sum(_CCI_ICD9[c][1] for c in found)


def compute_charlson_icd10(icd_codes: List[str]) -> int:
    """
    计算 Charlson 共病指数（ICD-10 版）。
    使用 Quan et al. 2005 适配版映射。
    """
    norm_codes = [_norm_icd(c) for c in icd_codes if c and str(c).lower() != 'nan']

    found = set()
    for cond, (prefixes, _) in _CCI_ICD10.items():
        for nc in norm_codes:
            if _matches_prefix(nc, prefixes):
                found.add(cond)
                break

    if 'DM_CC' in found:
        found.discard('DM_noCC')
    if 'MetastaticCancer' in found:
        found.discard('Cancer')
    if 'ModSevereLiver' in found:
        found.discard('MildLiver')

    return sum(_CCI_ICD10[c][1] for c in found)


def compute_lab_sofa_proxy(
    creatinine_mg: Optional[float],
    bilirubin_mg:  Optional[float],
    platelet_k:    Optional[float],
) -> Optional[float]:
    """
    计算实验室 SOFA 代理评分（3 组件）。
    
    组件：
      肾脏（肌酐 mg/dL）: 0-4 分
      肝脏（总胆红素 mg/dL）: 0-4 分
      凝血（血小板 ×10³/μL）: 0-4 分
    
    注意：真实 SOFA 还包括呼吸、心血管、神经组件；
    此代理评分命名为 lab_sofa_proxy，在论文中需说明局限性。
    
    Returns:
        代理分数（0-12），若三项均缺失返回 None
    """
    score = 0
    n_available = 0

    # 肌酐组件
    if creatinine_mg is not None and not np.isnan(creatinine_mg):
        n_available += 1
        if creatinine_mg >= 5.0:
            score += 4
        elif creatinine_mg >= 3.5:
            score += 3
        elif creatinine_mg >= 2.0:
            score += 2
        elif creatinine_mg >= 1.2:
            score += 1

    # 胆红素组件
    if bilirubin_mg is not None and not np.isnan(bilirubin_mg):
        n_available += 1
        if bilirubin_mg >= 12.0:
            score += 4
        elif bilirubin_mg >= 6.0:
            score += 3
        elif bilirubin_mg >= 2.0:
            score += 2
        elif bilirubin_mg >= 1.2:
            score += 1

    # 血小板组件
    if platelet_k is not None and not np.isnan(platelet_k):
        n_available += 1
        if platelet_k < 20:
            score += 4
        elif platelet_k < 50:
            score += 3
        elif platelet_k < 100:
            score += 2
        elif platelet_k < 150:
            score += 1

    return float(score) if n_available >= 1 else None


def compute_tir_metrics(
    glucose_values: List[float],
    low_threshold:  float = 70.0,
    high_threshold: float = 180.0,
) -> Dict:
    """
    计算住院血糖质量指标（国际共识推荐，Diabetes Care 2019）。

    Args:
        glucose_values: 住院期间所有血糖测量值（mg/dL）
        low_threshold:  低血糖阈值（默认 70 mg/dL）
        high_threshold: 高血糖阈值（默认 180 mg/dL）

    Returns:
        {
            'tir':   float  # Time in Range 比例（0-1）
            'tbr':   float  # Time Below Range 比例
            'tar':   float  # Time Above Range 比例
            'cv_pct': float # 血糖变异系数 %
            'mean':  float
            'std':   float
            'min':   float
            'max':   float
            'n_readings': int
            'severe_hypo_count': int   # <54 mg/dL 次数
            'hypo_count':        int   # <70 mg/dL 次数
        }
    """
    empty = {
        'tir': None, 'tbr': None, 'tar': None, 'cv_pct': None,
        'mean': None, 'std': None, 'min': None, 'max': None,
        'n_readings': 0, 'severe_hypo_count': 0, 'hypo_count': 0,
    }
    if not glucose_values:
        return empty

    vals = np.array([v for v in glucose_values if v is not None and not np.isnan(v) and v > 0])
    if len(vals) == 0:
        return empty

    n = len(vals)
    mean_val = float(np.mean(vals))
    std_val  = float(np.std(vals))
    cv_pct   = float(std_val / mean_val * 100) if mean_val > 0 else None

    in_range = np.sum((vals >= low_threshold) & (vals <= high_threshold))
    below    = np.sum(vals < low_threshold)
    above    = np.sum(vals > high_threshold)

    return {
        'tir':               round(float(in_range / n), 4),
        'tbr':               round(float(below / n), 4),
        'tar':               round(float(above / n), 4),
        'cv_pct':            round(cv_pct, 2) if cv_pct is not None else None,
        'mean':              round(mean_val, 2),
        'std':               round(std_val, 2),
        'min':               round(float(np.min(vals)), 2),
        'max':               round(float(np.max(vals)), 2),
        'n_readings':        int(n),
        'severe_hypo_count': int(np.sum(vals < 54)),
        'hypo_count':        int(np.sum(vals < 70)),
    }


def compute_aki_kdigo_dynamic(
    creatinine_series: List[Tuple[Optional[datetime], float]],
    admission_creat:   Optional[float] = None,
) -> Dict:
    """
    计算 AKI KDIGO 动态分期（基于肌酐轨迹）。

    KDIGO 2012 标准：
      Stage 1: ≥0.3 mg/dL 在 48h 内升高，或 ≥1.5-1.9× 基线（7天内）
      Stage 2: ≥2.0-2.9× 基线
      Stage 3: ≥3.0× 基线，或肌酐 ≥4.0 mg/dL 急性升高 ≥0.5 mg/dL

    Args:
        creatinine_series: [(datetime_or_None, value_mg_dL), ...]
        admission_creat:   入院时基线肌酐（如果 series 包含则可从 series 取最低值）

    Returns:
        {
            'aki_present':       bool
            'max_kdigo_stage':   int   # 0-3
            'baseline_creat':    float
            'peak_creat':        float
            'creat_fold_change': float # peak/baseline
            'aki_on_discharge':  bool  # 出院时 AKI 是否仍存在（肌酐未恢复 < 1.5× 基线）
            'aki_progression':   bool  # AKI 分期是否上升（结局恶化标志）
        }
    """
    empty = {
        'aki_present': False, 'max_kdigo_stage': 0,
        'baseline_creat': None, 'peak_creat': None,
        'creat_fold_change': None, 'aki_on_discharge': False,
        'aki_progression': False,
    }

    # 过滤有效值
    valid = [(t, v) for t, v in creatinine_series if v is not None and not np.isnan(v) and v > 0]
    if len(valid) < 2:
        return empty

    # 按时间排序（None 时间置尾）
    valid.sort(key=lambda x: (x[0] is None, x[0] or datetime.min))
    times  = [t for t, _ in valid]
    values = [v for _, v in valid]

    # 基线：入院后前 48h 的最低值（或 admission_creat 若已知）
    if admission_creat is not None and admission_creat > 0:
        baseline = admission_creat
    else:
        # 取前两个值的最低值作为基线（保守估计）
        baseline = min(values[:min(3, len(values))])

    if baseline <= 0:
        return empty

    peak = max(values)
    fold = peak / baseline

    # KDIGO 分期（基于倍数变化）
    stage = 0
    if fold >= 3.0 or (peak >= 4.0 and peak - baseline >= 0.5):
        stage = 3
    elif fold >= 2.0:
        stage = 2
    elif fold >= 1.5:
        stage = 1
    else:
        # 检查 48h 内绝对值升高 ≥0.3（更敏感的早期 AKI 检测）
        for i, (t_i, v_i) in enumerate(valid):
            for j in range(i + 1, len(valid)):
                t_j, v_j = valid[j]
                if t_i is not None and t_j is not None:
                    delta_h = (t_j - t_i).total_seconds() / 3600
                    if delta_h <= 48 and (v_j - v_i) >= 0.3:
                        stage = max(stage, 1)
                        break

    # 出院时 AKI 是否持续（末尾值是否仍 ≥1.5× 基线）
    last_creat = values[-1]
    aki_on_discharge = (last_creat / baseline) >= 1.5 if baseline > 0 else False

    # AKI 进展：早期分期 < 后期分期
    early_stage = _kdigo_stage_from_value(values[0], baseline)
    late_stage  = _kdigo_stage_from_value(values[-1], baseline)
    aki_progression = late_stage > early_stage

    return {
        'aki_present':       stage > 0,
        'max_kdigo_stage':   stage,
        'baseline_creat':    round(baseline, 3),
        'peak_creat':        round(peak, 3),
        'creat_fold_change': round(fold, 3),
        'aki_on_discharge':  bool(aki_on_discharge),
        'aki_progression':   bool(aki_progression),
    }


def _kdigo_stage_from_value(creat: float, baseline: float) -> int:
    """从单个肌酐值和基线计算 KDIGO 分期。"""
    if baseline <= 0:
        return 0
    fold = creat / baseline
    if fold >= 3.0 or (creat >= 4.0 and creat - baseline >= 0.5):
        return 3
    if fold >= 2.0:
        return 2
    if fold >= 1.5:
        return 1
    return 0


def compute_egfr_ckd_epi_2021(
    creatinine: float,
    age: float,
    gender: str,
) -> Optional[float]:
    """
    CKD-EPI 2021 公式（无种族校正）。
    Inker LA, et al. N Engl J Med. 2021;385(19):1737-1749.
    """
    if creatinine is None or age is None or creatinine <= 0 or age <= 0:
        return None
    gender_upper = str(gender).upper()
    is_female = gender_upper in ('F', 'FEMALE')
    kappa = 0.7 if is_female else 0.9
    alpha = -0.241 if is_female else -0.302
    sex_factor = 1.012 if is_female else 1.0
    ratio = creatinine / kappa
    egfr = (142
            * (min(ratio, 1) ** alpha)
            * (max(ratio, 1) ** (-1.200))
            * (0.9938 ** age)
            * sex_factor)
    return round(egfr, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# 三、基类定义
# ═══════════════════════════════════════════════════════════════════════════════

class OutcomeExtractorBase(ABC):
    """Outcome 提取器基类（v2.1）"""

    def __init__(self, data_path: Path, dataset_name: str):
        self.data_path    = data_path
        self.dataset_name = dataset_name
        logger.info(f"初始化 {dataset_name} 提取器: {data_path}")

    @abstractmethod
    def extract_patient_outcomes(self, subject_id: int, hadm_ids: List[int]) -> Dict:
        pass

    def extract_all_outcomes(
        self,
        patient_list_path: Path,
        output_path:       Optional[Path] = None,
    ) -> pd.DataFrame:
        """
        批量提取所有患者所有住院的 outcomes，含预加载优化。

        v2.1 修复 [FIX-P0]：
          原逻辑每个 subject 仅提取最新一次住院的结局，导致多次住院的早期
          hadm 在 outcome CSV 中无记录 → pilot_sampler 标记为 Unknown。
          修复后逐 hadm 提取，每次住院独立产出一行结局记录。
        """
        logger.info(f"从 {patient_list_path} 加载患者列表...")
        patient_df = pd.read_csv(patient_list_path)
        logger.info(f"加载了 {len(patient_df)} 个患者")

        required_cols = ['subject_id', 'hadm_ids', 'dm_type']
        missing = [c for c in required_cols if c not in patient_df.columns]
        if missing:
            raise ValueError(f"患者列表缺少必需列: {missing}")

        # ── 诊断：打印前 3 行 hadm_ids 原始值，便于确认 CSV 格式 ──────────────
        _sample = patient_df['hadm_ids'].head(3).tolist()
        logger.info(f"[诊断] hadm_ids 列前3行原始值: {_sample}")

        all_hadm_ids    = set()
        subject_to_hadm = {}
        n_parse_ok      = 0
        n_parse_fail    = 0

        for _, row in patient_df.iterrows():
            sid = row['subject_id']
            raw_val = row['hadm_ids']
            try:
                if not pd.notna(raw_val):
                    n_parse_fail += 1
                    continue
                raw = json.loads(raw_val)
                if not isinstance(raw, list):
                    raw = [raw]   # 单值兼容：json.loads('123') → 123 (int)
                # [BUG-FIX-v2] hadm_id 经 pandas float64 提升后以 '193457.0'
                # 形式存入 CSV；int('193457.0') 抛 ValueError，必须先 float() 再 int()。
                # 同时兼容：int / float / '193457' / '193457.0'
                hids = [
                    int(float(h))
                    for h in raw
                    if h is not None and str(h).strip() not in ('', 'nan', 'None', 'null')
                ]
                if hids:
                    all_hadm_ids.update(hids)
                    subject_to_hadm[sid] = hids
                    n_parse_ok += 1
                else:
                    n_parse_fail += 1
            except Exception as e:
                n_parse_fail += 1
                if n_parse_fail <= 5:   # 打印前 5 个失败行，而非静默吞掉
                    logger.warning(
                        f"[解析] subject_id={sid} hadm_ids={raw_val!r} 解析失败: {e}"
                    )

        logger.info(
            f"[解析] hadm_ids 解析完成: {n_parse_ok} 行成功 / {n_parse_fail} 行失败"
        )

        n_total_hadm = len(all_hadm_ids)
        logger.info(f"共 {n_total_hadm} 个唯一住院 ID，开始预加载...")
        self._preload_data(list(all_hadm_ids))

        # [FIX-P0] 逐 hadm 提取：每次住院独立产出一行结局记录
        # extract_patient_outcomes(sid, [single_hid]) 传入单元素列表，
        # 方法内部会定位到该 hadm 的入院记录；再入院计算使用预建的
        # subject→hadm 辅助索引（不依赖传入的 hadm_ids 列表）。
        all_outcomes = []
        n_extracted  = 0
        n_failed     = 0
        for _, row in tqdm(patient_df.iterrows(), total=len(patient_df), desc="提取 outcomes"):
            sid    = row['subject_id']
            dm_typ = row['dm_type']
            if sid not in subject_to_hadm:
                continue
            for hid in subject_to_hadm[sid]:
                try:
                    outcomes = self.extract_patient_outcomes(sid, [hid])
                    if outcomes:
                        outcomes['subject_id']    = sid
                        outcomes['diabetes_type'] = dm_typ
                        all_outcomes.append(outcomes)
                        n_extracted += 1
                except Exception as e:
                    n_failed += 1
                    if n_failed <= 10:
                        logger.error(f"提取患者 {sid} hadm {hid} 失败: {e}")
                    elif n_failed == 11:
                        logger.error("后续失败不再逐条打印（超过 10 条）...")

        logger.info(
            f"逐 hadm 提取完成: {n_extracted} 条成功 / {n_failed} 条失败 "
            f"（{len(subject_to_hadm)} subjects × {n_total_hadm} hadm）"
        )

        if all_outcomes:
            result_df = pd.DataFrame(all_outcomes)
            logger.info(f"成功提取 {len(result_df)} 条住院结局记录（逐 hadm）")
            if output_path:
                result_df.to_csv(output_path, index=False)
                logger.info(f"已保存到: {output_path}")
            return result_df
        else:
            logger.warning("未提取到任何 outcomes")
            return pd.DataFrame()

    @abstractmethod
    def _preload_data(self, hadm_ids: List[int]):
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# 四、MIMIC-III 提取器（v2.0）
# ═══════════════════════════════════════════════════════════════════════════════

class MIMICIII_OutcomeExtractor(OutcomeExtractorBase):
    """MIMIC-III Outcome 提取器 v2.0"""

    def __init__(self, data_path: Path):
        super().__init__(data_path, 'mimic-iii')
        logger.info("加载基础表...")
        self.admissions = self._load_admissions()   # index: hadm_id；保留 subject_id 列
        self.diagnoses  = self._load_diagnoses()    # index: hadm_id
        self.patients   = self._load_patients()     # index: subject_id

        # subject_id → list[hadm_id] 辅助索引（用于再入院查询）[FIX-3]
        if not self.admissions.empty and 'subject_id' in self.admissions.columns:
            self._subj_to_hadms: Dict[int, List[int]] = (
                self.admissions.reset_index()
                .groupby('subject_id')['hadm_id']
                .apply(list)
                .to_dict()
            )
        else:
            self._subj_to_hadms = {}

        self.labevents = None  # 由 _preload_data 填充

    # ── 基础表加载 ────────────────────────────────────────────────────────────

    def _load_admissions(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(
                self.data_path / 'ADMISSIONS.csv.gz',
                usecols=['subject_id', 'hadm_id', 'admittime', 'dischtime',
                         'discharge_location'],
                parse_dates=['admittime', 'dischtime'],
            )
            df.columns = df.columns.str.lower()
            df = df.set_index('hadm_id')
            logger.info(f"ADMISSIONS: {len(df)} 条")
            return df
        except Exception as e:
            logger.error(f"加载 ADMISSIONS 失败: {e}")
            return pd.DataFrame()

    def _load_diagnoses(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(
                self.data_path / 'DIAGNOSES_ICD.csv.gz',
                usecols=['subject_id', 'hadm_id', 'icd9_code'],
            )
            df.columns = df.columns.str.lower()
            df['icd9_code'] = df['icd9_code'].astype(str).str.strip()
            df = df.set_index('hadm_id')
            logger.info(f"DIAGNOSES: {len(df)} 条")
            return df
        except Exception as e:
            logger.error(f"加载 DIAGNOSES 失败: {e}")
            return pd.DataFrame()

    def _load_patients(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(
                self.data_path / 'PATIENTS.csv.gz',
                usecols=['subject_id', 'gender', 'dob', 'dod', 'dod_hosp'],
                parse_dates=['dob', 'dod', 'dod_hosp'],
            )
            df.columns = df.columns.str.lower()
            df = df.set_index('subject_id')
            logger.info(f"PATIENTS: {len(df)} 条")
            return df
        except Exception as e:
            logger.error(f"加载 PATIENTS 失败: {e}")
            return pd.DataFrame()

    def _preload_data(self, hadm_ids: List[int]):
        """预加载 LABEVENTS（过滤到目标 hadm_id + itemid）。"""
        all_itemids = set()
        for ids in _MIMIC3_LAB.values():
            all_itemids.update(ids)

        hadm_set  = set(hadm_ids)
        chunks    = []
        try:
            for chunk in tqdm(
                pd.read_csv(
                    self.data_path / 'LABEVENTS.csv.gz',
                    usecols=['subject_id', 'hadm_id', 'itemid', 'charttime', 'valuenum'],
                    parse_dates=['charttime'],
                    chunksize=1_000_000,
                ),
                desc="加载 LABEVENTS",
            ):
                chunk.columns = chunk.columns.str.lower()
                filt = chunk[
                    (chunk['hadm_id'].isin(hadm_set)) &
                    (chunk['itemid'].isin(all_itemids))
                ].copy()
                if not filt.empty:
                    chunks.append(filt)

            if chunks:
                self.labevents = pd.concat(chunks, ignore_index=True)
                self.labevents = self.labevents.set_index('hadm_id')
                logger.info(f"LABEVENTS: {len(self.labevents)} 条目标记录")
            else:
                self.labevents = pd.DataFrame()
                logger.warning("LABEVENTS: 未找到匹配记录")
        except Exception as e:
            logger.error(f"加载 LABEVENTS 失败: {e}")
            self.labevents = pd.DataFrame()

    # ── 核心提取 ──────────────────────────────────────────────────────────────

    def extract_patient_outcomes(self, subject_id: int, hadm_ids: List[int]) -> Dict:
        outcomes: Dict = {}

        # ── 1. 定位最近住院 ───────────────────────────────────────────────────
        try:
            adm_rows = self.admissions.loc[
                self.admissions.index.isin(hadm_ids)
            ].copy()
        except Exception:
            adm_rows = pd.DataFrame()

        if adm_rows.empty:
            return {}

        adm_rows = adm_rows.sort_values('admittime', ascending=False)
        adm      = adm_rows.iloc[0]
        hadm_id  = adm.name

        outcomes['hadm_id']  = int(hadm_id)
        outcomes['admittime'] = adm['admittime']
        outcomes['dischtime'] = adm['dischtime']

        # ── 2. 住院时长 ───────────────────────────────────────────────────────
        if pd.notna(adm['dischtime']) and pd.notna(adm['admittime']):
            los = (adm['dischtime'] - adm['admittime']).total_seconds() / 86400
            outcomes['length_of_stay'] = round(max(0.0, los), 2)
        else:
            outcomes['length_of_stay'] = None

        # ── 3. 出院去向 [NEW-3] ──────────────────────────────────────────────
        outcomes['discharge_destination'] = (
            str(adm.get('discharge_location', '')).strip() or None
        )

        # ── 4. 诊断：CCI + 急性并发症 ────────────────────────────────────────
        try:
            diag_df = self.diagnoses.loc[[hadm_id]].copy()
        except KeyError:
            diag_df = pd.DataFrame()

        icd_codes = list(diag_df['icd9_code'].dropna()) if not diag_df.empty else []

        # [NEW-1] Charlson CCI（ICD-9）
        outcomes['charlson_cci'] = compute_charlson_icd9(icd_codes)

        # 急性并发症（ICD-9 前缀匹配）
        dka_prefixes = ['2501', '25011', '25013']  # 250.1x
        aki_prefixes = ['584']
        norm_icds = [_norm_icd(c) for c in icd_codes]

        outcomes['dka_event'] = int(
            any(_matches_prefix(c, ['2501']) for c in norm_icds)
        )
        outcomes['aki_event'] = int(
            any(_matches_prefix(c, ['584']) for c in norm_icds)
        )

        # ── 5. 实验室数据 ─────────────────────────────────────────────────────
        lab_df = pd.DataFrame()
        if self.labevents is not None and not self.labevents.empty:
            try:
                lab_df = self.labevents.loc[[hadm_id]].copy()
            except KeyError:
                pass

        if not lab_df.empty:
            lab_df['valuenum'] = pd.to_numeric(lab_df['valuenum'], errors='coerce')

            # HbA1c
            hba1c_df = lab_df[lab_df['itemid'].isin(_MIMIC3_LAB['hba1c'])].dropna(
                subset=['valuenum']).sort_values('charttime')
            if len(hba1c_df) >= 1:
                outcomes['hba1c_admission'] = float(hba1c_df.iloc[0]['valuenum'])
            if len(hba1c_df) >= 2:
                outcomes['hba1c_discharge'] = float(hba1c_df.iloc[-1]['valuenum'])
                outcomes['hba1c_change'] = (
                    outcomes['hba1c_discharge'] - outcomes['hba1c_admission']
                )

            # 血糖指标 [NEW-4/5]
            glc_vals = lab_df[lab_df['itemid'].isin(_MIMIC3_LAB['glucose'])][
                'valuenum'].dropna().tolist()
            tir_metrics = compute_tir_metrics(glc_vals)
            outcomes.update({'glucose_' + k: v for k, v in tir_metrics.items()})

            # 肌酐序列（用于 AKI 动态分期 + SOFA）
            creat_df = lab_df[lab_df['itemid'].isin(_MIMIC3_LAB['creatinine'])].dropna(
                subset=['valuenum']).sort_values('charttime')
            creat_series = list(zip(
                creat_df['charttime'].tolist(),
                creat_df['valuenum'].astype(float).tolist(),
            ))
            aki_result = compute_aki_kdigo_dynamic(creat_series)
            outcomes.update({'aki_kdigo_' + k: v for k, v in aki_result.items()})

            # SOFA 代理组件 [NEW-2]
            def _first_val(item_key: str) -> Optional[float]:
                sub = lab_df[lab_df['itemid'].isin(_MIMIC3_LAB[item_key])].dropna(
                    subset=['valuenum'])
                return float(sub.iloc[0]['valuenum']) if not sub.empty else None

            cr_val = _first_val('creatinine')
            bi_val = _first_val('bilirubin')
            pl_val = _first_val('platelet')
            outcomes['lab_sofa_proxy'] = compute_lab_sofa_proxy(cr_val, bi_val, pl_val)

            # 入院 eGFR [NEW-9]
            try:
                pat_row = self.patients.loc[subject_id]
                gender  = str(pat_row.get('gender', ''))
                # 年龄估算（dob → admittime）
                dob      = pat_row.get('dob')
                admittime = adm['admittime']
                age = None
                if pd.notna(dob) and pd.notna(admittime):
                    age = (admittime - dob).total_seconds() / (365.25 * 86400)
                    if age > 150:
                        age = 91.4  # MIMIC-III 高龄脱敏截断处理
                outcomes['egfr_admission'] = (
                    compute_egfr_ckd_epi_2021(cr_val, age, gender)
                    if (cr_val and age) else None
                )
            except Exception:
                outcomes['egfr_admission'] = None

        # ── 6. 院内死亡（含时间窗校验）[FIX-1] ──────────────────────────────
        try:
            pat_row  = self.patients.loc[subject_id]
            dod_hosp = pat_row.get('dod_hosp')
            dod      = pat_row.get('dod')

            in_hosp = False
            if pd.notna(dod_hosp) and pd.notna(adm['admittime']) and pd.notna(adm['dischtime']):
                # [FIX-1] 原逻辑：dod_hosp 非空即判死亡，忽略了时间窗
                # 修复：dod_hosp 必须在本次入院的 admittime ~ dischtime+1天 内
                dod_hosp_dt = pd.to_datetime(dod_hosp)
                window_end  = adm['dischtime'] + timedelta(days=1)
                in_hosp = adm['admittime'] <= dod_hosp_dt <= window_end
            elif pd.notna(dod_hosp):
                # dischtime 缺失时退化到旧逻辑（保守处理）
                in_hosp = True

            outcomes['in_hospital_mortality'] = int(in_hosp)

            # 28日死亡率 [NEW-7]
            if pd.notna(dod) and pd.notna(adm.get('dischtime')):
                dod_dt   = pd.to_datetime(dod)
                dischtime = adm['dischtime']
                outcomes['mortality_28d'] = int(
                    (dod_dt - dischtime).total_seconds() / 86400 <= 28
                )
            else:
                outcomes['mortality_28d'] = int(in_hosp)  # 院内死亡算入28日

        except (KeyError, Exception):
            outcomes['in_hospital_mortality'] = 0
            outcomes['mortality_28d']          = 0

        # ── 7. 再入院 [FIX-3] ───────────────────────────────────────────────
        # FIX-3: admissions 已 set_index('hadm_id')，使用预建的 subject→hadm 辅助索引
        if pd.notna(adm.get('dischtime')):
            try:
                all_hids = self._subj_to_hadms.get(subject_id, [])
                future_adms = []
                for hid in all_hids:
                    if hid == hadm_id:
                        continue
                    try:
                        row = self.admissions.loc[hid]
                        if pd.notna(row['admittime']) and row['admittime'] > adm['dischtime']:
                            future_adms.append(
                                (row['admittime'] - adm['dischtime']).total_seconds() / 86400
                            )
                    except KeyError:
                        continue

                if future_adms:
                    outcomes['readmission_30d'] = int(min(future_adms) <= 30)
                    outcomes['readmission_90d'] = int(min(future_adms) <= 90)
                else:
                    outcomes['readmission_30d'] = 0
                    outcomes['readmission_90d'] = 0
            except Exception:
                outcomes['readmission_30d'] = 0
                outcomes['readmission_90d'] = 0

        return outcomes


# ═══════════════════════════════════════════════════════════════════════════════
# 五、MIMIC-IV 提取器（v2.0）
# ═══════════════════════════════════════════════════════════════════════════════

class MIMICIV_OutcomeExtractor(OutcomeExtractorBase):
    """MIMIC-IV Outcome 提取器 v2.0"""

    def __init__(self, data_path: Path):
        super().__init__(data_path, 'mimic-iv')
        logger.info("加载基础表...")
        self.admissions = self._load_admissions()
        self.diagnoses  = self._load_diagnoses()
        self.patients   = self._load_patients()

        # [FIX-3] subject→hadm 辅助索引
        if not self.admissions.empty and 'subject_id' in self.admissions.columns:
            self._subj_to_hadms: Dict[int, List[int]] = (
                self.admissions.reset_index()
                .groupby('subject_id')['hadm_id']
                .apply(list)
                .to_dict()
            )
        else:
            self._subj_to_hadms = {}

        self.labevents = None

    def _load_admissions(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(
                self.data_path / 'hosp' / 'admissions.csv.gz',
                usecols=['subject_id', 'hadm_id', 'admittime', 'dischtime',
                         'discharge_location'],
                parse_dates=['admittime', 'dischtime'],
            )
            df = df.set_index('hadm_id')
            logger.info(f"ADMISSIONS: {len(df)} 条")
            return df
        except Exception as e:
            logger.error(f"加载 ADMISSIONS 失败: {e}")
            return pd.DataFrame()

    def _load_diagnoses(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(
                self.data_path / 'hosp' / 'diagnoses_icd.csv.gz',
                usecols=['subject_id', 'hadm_id', 'icd_code', 'icd_version'],
            )
            df['icd_code'] = df['icd_code'].astype(str).str.strip().str.upper()
            df = df.set_index('hadm_id')
            logger.info(f"DIAGNOSES: {len(df)} 条")
            return df
        except Exception as e:
            logger.error(f"加载 DIAGNOSES 失败: {e}")
            return pd.DataFrame()

    def _load_patients(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(
                self.data_path / 'hosp' / 'patients.csv.gz',
                usecols=['subject_id', 'gender', 'anchor_age', 'dod'],
                parse_dates=['dod'],
            )
            df = df.set_index('subject_id')
            logger.info(f"PATIENTS: {len(df)} 条")
            return df
        except Exception as e:
            logger.error(f"加载 PATIENTS 失败: {e}")
            return pd.DataFrame()

    def _preload_data(self, hadm_ids: List[int]):
        all_itemids = set()
        for ids in _MIMIC4_LAB.values():
            all_itemids.update(ids)

        hadm_set = set(hadm_ids)
        chunks   = []
        try:
            for chunk in tqdm(
                pd.read_csv(
                    self.data_path / 'hosp' / 'labevents.csv.gz',
                    usecols=['subject_id', 'hadm_id', 'itemid', 'charttime', 'valuenum'],
                    parse_dates=['charttime'],
                    chunksize=1_000_000,
                ),
                desc="加载 LABEVENTS",
            ):
                filt = chunk[
                    (chunk['hadm_id'].isin(hadm_set)) &
                    (chunk['itemid'].isin(all_itemids))
                ].copy()
                if not filt.empty:
                    chunks.append(filt)

            if chunks:
                self.labevents = pd.concat(chunks, ignore_index=True)
                self.labevents = self.labevents.set_index('hadm_id')
                logger.info(f"LABEVENTS: {len(self.labevents)} 条目标记录")
            else:
                self.labevents = pd.DataFrame()
        except Exception as e:
            logger.error(f"加载 LABEVENTS 失败: {e}")
            self.labevents = pd.DataFrame()

    def extract_patient_outcomes(self, subject_id: int, hadm_ids: List[int]) -> Dict:
        outcomes: Dict = {}

        try:
            adm_rows = self.admissions.loc[
                self.admissions.index.isin(hadm_ids)
            ].copy()
        except Exception:
            adm_rows = pd.DataFrame()

        if adm_rows.empty:
            return {}

        adm_rows = adm_rows.sort_values('admittime', ascending=False)
        adm      = adm_rows.iloc[0]
        hadm_id  = adm.name

        outcomes['hadm_id']  = int(hadm_id)
        outcomes['admittime'] = adm['admittime']
        outcomes['dischtime'] = adm['dischtime']

        if pd.notna(adm['dischtime']) and pd.notna(adm['admittime']):
            los = (adm['dischtime'] - adm['admittime']).total_seconds() / 86400
            outcomes['length_of_stay'] = round(max(0.0, los), 2)
        else:
            outcomes['length_of_stay'] = None

        # 出院去向 [NEW-3]
        outcomes['discharge_destination'] = (
            str(adm.get('discharge_location', '')).strip() or None
        )

        # ── 诊断 ──────────────────────────────────────────────────────────────
        try:
            diag_df = self.diagnoses.loc[[hadm_id]].copy()
        except KeyError:
            diag_df = pd.DataFrame()

        icd10_codes = (
            list(diag_df.loc[diag_df['icd_version'] == 10, 'icd_code'].dropna())
            if not diag_df.empty and 'icd_version' in diag_df.columns
            else list(diag_df['icd_code'].dropna()) if not diag_df.empty else []
        )
        icd9_codes = (
            list(diag_df.loc[diag_df['icd_version'] == 9, 'icd_code'].dropna())
            if not diag_df.empty and 'icd_version' in diag_df.columns
            else []
        )

        # [NEW-1] Charlson CCI：优先用 ICD-10，fallback ICD-9
        if icd10_codes:
            outcomes['charlson_cci'] = compute_charlson_icd10(icd10_codes)
        else:
            outcomes['charlson_cci'] = compute_charlson_icd9(icd9_codes)

        norm_icd10 = [_norm_icd(c) for c in icd10_codes]
        outcomes['dka_event'] = int(
            any(_matches_prefix(c, ['E101', 'E111', 'E131', 'E141']) for c in norm_icd10)
        )
        outcomes['aki_event'] = int(
            any(_matches_prefix(c, ['N17']) for c in norm_icd10)
        )

        # ── 实验室 ────────────────────────────────────────────────────────────
        lab_df = pd.DataFrame()
        if self.labevents is not None and not self.labevents.empty:
            try:
                lab_df = self.labevents.loc[[hadm_id]].copy()
            except KeyError:
                pass

        if not lab_df.empty:
            lab_df['valuenum'] = pd.to_numeric(lab_df['valuenum'], errors='coerce')

            hba1c_df = lab_df[lab_df['itemid'].isin(_MIMIC4_LAB['hba1c'])].dropna(
                subset=['valuenum']).sort_values('charttime')
            if len(hba1c_df) >= 1:
                outcomes['hba1c_admission'] = float(hba1c_df.iloc[0]['valuenum'])
            if len(hba1c_df) >= 2:
                outcomes['hba1c_discharge'] = float(hba1c_df.iloc[-1]['valuenum'])
                outcomes['hba1c_change'] = (
                    outcomes['hba1c_discharge'] - outcomes['hba1c_admission']
                )

            # 血糖 TIR [NEW-4/5]
            glc_vals = lab_df[lab_df['itemid'].isin(_MIMIC4_LAB['glucose'])][
                'valuenum'].dropna().tolist()
            tir_metrics = compute_tir_metrics(glc_vals)
            outcomes.update({'glucose_' + k: v for k, v in tir_metrics.items()})

            # AKI 动态分期 [NEW-6]
            creat_df = lab_df[lab_df['itemid'].isin(_MIMIC4_LAB['creatinine'])].dropna(
                subset=['valuenum']).sort_values('charttime')
            creat_series = list(zip(
                creat_df['charttime'].tolist(),
                creat_df['valuenum'].astype(float).tolist(),
            ))
            aki_result = compute_aki_kdigo_dynamic(creat_series)
            outcomes.update({'aki_kdigo_' + k: v for k, v in aki_result.items()})

            # Lab-SOFA 代理 [NEW-2]
            def _first_val(key: str) -> Optional[float]:
                sub = lab_df[lab_df['itemid'].isin(_MIMIC4_LAB[key])].dropna(
                    subset=['valuenum'])
                return float(sub.iloc[0]['valuenum']) if not sub.empty else None

            cr_val = _first_val('creatinine')
            bi_val = _first_val('bilirubin')
            pl_val = _first_val('platelet')
            outcomes['lab_sofa_proxy'] = compute_lab_sofa_proxy(cr_val, bi_val, pl_val)

            # 入院 eGFR [NEW-9]
            try:
                pat_row    = self.patients.loc[subject_id]
                gender     = str(pat_row.get('gender', ''))
                anchor_age = pat_row.get('anchor_age')
                age        = float(anchor_age) if pd.notna(anchor_age) else None
                outcomes['egfr_admission'] = (
                    compute_egfr_ckd_epi_2021(cr_val, age, gender)
                    if (cr_val and age) else None
                )
            except Exception:
                outcomes['egfr_admission'] = None

        # ── 院内死亡（时间窗校验，MIMIC-IV dod 是全死亡日期）[FIX-2 确认] ───
        try:
            pat_row = self.patients.loc[subject_id]
            dod     = pat_row.get('dod')
            in_hosp = False
            if pd.notna(dod) and pd.notna(adm['admittime']) and pd.notna(adm['dischtime']):
                dod_dt     = pd.to_datetime(dod)
                window_end = adm['dischtime'] + timedelta(days=1)
                in_hosp    = adm['admittime'] <= dod_dt <= window_end
            outcomes['in_hospital_mortality'] = int(in_hosp)

            # 28日死亡率 [NEW-7]
            if pd.notna(dod) and pd.notna(adm.get('dischtime')):
                dod_dt = pd.to_datetime(dod)
                outcomes['mortality_28d'] = int(
                    (dod_dt - adm['dischtime']).total_seconds() / 86400 <= 28
                )
            else:
                outcomes['mortality_28d'] = int(in_hosp)
        except (KeyError, Exception):
            outcomes['in_hospital_mortality'] = 0
            outcomes['mortality_28d']          = 0

        # ── 再入院 [FIX-3] ──────────────────────────────────────────────────
        if pd.notna(adm.get('dischtime')):
            try:
                all_hids = self._subj_to_hadms.get(subject_id, [])
                future_days = []
                for hid in all_hids:
                    if hid == hadm_id:
                        continue
                    try:
                        row = self.admissions.loc[hid]
                        if pd.notna(row['admittime']) and row['admittime'] > adm['dischtime']:
                            delta = (row['admittime'] - adm['dischtime']).total_seconds() / 86400
                            future_days.append(delta)
                    except KeyError:
                        continue

                if future_days:
                    outcomes['readmission_30d'] = int(min(future_days) <= 30)
                    outcomes['readmission_90d'] = int(min(future_days) <= 90)
                else:
                    outcomes['readmission_30d'] = 0
                    outcomes['readmission_90d'] = 0
            except Exception:
                outcomes['readmission_30d'] = 0
                outcomes['readmission_90d'] = 0

        return outcomes


# ═══════════════════════════════════════════════════════════════════════════════
# 六、eICU 提取器（v2.0）
# ═══════════════════════════════════════════════════════════════════════════════

class eICU_OutcomeExtractor(OutcomeExtractorBase):
    """eICU Outcome 提取器 v2.0"""

    def __init__(self, data_path: Path):
        super().__init__(data_path, 'eicu')
        logger.info("加载基础表...")
        self.patient   = self._load_patient()    # index: patientunitstayid
        self.diagnosis = self._load_diagnosis()  # index: patientunitstayid

        # [FIX-4] uniquepid → list[patientunitstayid] 辅助索引（再入院用）
        if not self.patient.empty and 'uniquepid' in self.patient.columns:
            self._uid_to_stays: Dict[str, List[int]] = (
                self.patient.reset_index()
                .groupby('uniquepid')['patientunitstayid']
                .apply(list)
                .to_dict()
            )
        else:
            self._uid_to_stays = {}

        self.lab = None

    def _load_patient(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(
                self.data_path / 'patient.csv.gz',
                usecols=['patientunitstayid', 'uniquepid', 'unitdischargeoffset',
                         'hospitaldischargestatus', 'unitdischargestatus',
                         'hospitaldischargelocation', 'age', 'gender',
                         'unitadmittime24'],
            )
            df.columns = df.columns.str.lower()
            df = df.set_index('patientunitstayid')
            logger.info(f"PATIENT: {len(df)} 条")
            return df
        except Exception as e:
            logger.error(f"加载 patient 失败: {e}")
            return pd.DataFrame()

    def _load_diagnosis(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(
                self.data_path / 'diagnosis.csv.gz',
                usecols=['patientunitstayid', 'icd9code', 'diagnosisstring'],
            )
            df.columns = df.columns.str.lower()
            df['icd9code']       = df['icd9code'].astype(str).str.strip()
            df['diagnosisstring'] = df['diagnosisstring'].astype(str).fillna('')
            df = df.set_index('patientunitstayid')
            logger.info(f"DIAGNOSIS: {len(df)} 条")
            return df
        except Exception as e:
            logger.error(f"加载 diagnosis 失败: {e}")
            return pd.DataFrame()

    def _preload_data(self, hadm_ids: List[int]):
        """eICU hadm_ids 实为 patientunitstayid。"""
        stay_set = set(hadm_ids)
        chunks   = []
        try:
            for chunk in tqdm(
                pd.read_csv(
                    self.data_path / 'lab.csv.gz',
                    usecols=['patientunitstayid', 'labname', 'labresult', 'labresultoffset'],
                    chunksize=1_000_000,
                ),
                desc="加载 LAB",
            ):
                chunk.columns = chunk.columns.str.lower()
                filt = chunk[chunk['patientunitstayid'].isin(stay_set)].copy()
                if not filt.empty:
                    chunks.append(filt)

            if chunks:
                self.lab = pd.concat(chunks, ignore_index=True)
                self.lab['labname'] = self.lab['labname'].str.lower().str.strip()
                self.lab = self.lab.set_index('patientunitstayid')
                logger.info(f"LAB: {len(self.lab)} 条目标记录")
            else:
                self.lab = pd.DataFrame()
        except Exception as e:
            logger.error(f"加载 lab 失败: {e}")
            self.lab = pd.DataFrame()

    def extract_patient_outcomes(self, subject_id: int, hadm_ids: List[int]) -> Dict:
        """eICU 中 hadm_ids 实为 patientunitstayid 列表。"""
        stay_ids = hadm_ids
        outcomes: Dict = {}

        try:
            pat_rows = self.patient.loc[self.patient.index.isin(stay_ids)].copy()
        except Exception:
            pat_rows = pd.DataFrame()

        if pat_rows.empty:
            return {}

        # 选最新 stay（offset 最大即住院最久的那次）
        pat_row = pat_rows.sort_index(ascending=False).iloc[0]
        stay_id = int(pat_row.name)

        outcomes['hadm_id'] = stay_id

        # 住院时长（ICU unit）
        los_min = pat_row.get('unitdischargeoffset')
        outcomes['length_of_stay'] = (
            round(max(0.0, float(los_min) / 1440), 2) if pd.notna(los_min) else None
        )

        # 出院去向 [NEW-3]
        outcomes['discharge_destination'] = (
            str(pat_row.get('hospitaldischargelocation', '')).strip() or None
        )

        # ── 诊断 ──────────────────────────────────────────────────────────────
        try:
            diag_df = self.diagnosis.loc[[stay_id]].copy()
        except KeyError:
            diag_df = pd.DataFrame()

        icd_codes = []
        diag_strings = []
        if not diag_df.empty:
            icd_codes     = list(diag_df['icd9code'].dropna())
            diag_strings  = list(diag_df['diagnosisstring'].fillna(''))

        # [NEW-1] CCI（eICU 用 ICD-9）
        outcomes['charlson_cci'] = compute_charlson_icd9(icd_codes)

        # [FIX-5] DKA：ICD 匹配 + diagnosisstring 自由文本扫描
        norm_icds = [_norm_icd(c) for c in icd_codes]
        dka_by_icd  = any(_matches_prefix(c, ['2501']) for c in norm_icds)
        dka_by_text = any(_DKA_PATTERN.search(s) for s in diag_strings)
        outcomes['dka_event'] = int(dka_by_icd or dka_by_text)

        hhs_by_text = any(_HHS_PATTERN.search(s) for s in diag_strings)
        outcomes['hhs_event'] = int(hhs_by_text)

        aki_by_icd  = any(_matches_prefix(c, ['584']) for c in norm_icds)
        aki_by_text = any(
            re.search(r'acute kidney injury|aki\b|acute renal failure', s, re.IGNORECASE)
            for s in diag_strings
        )
        outcomes['aki_event'] = int(aki_by_icd or aki_by_text)

        # ── 实验室 ────────────────────────────────────────────────────────────
        lab_df = pd.DataFrame()
        if self.lab is not None and not self.lab.empty:
            try:
                lab_df = self.lab.loc[[stay_id]].copy()
            except KeyError:
                pass

        if not lab_df.empty:
            lab_df['labresult'] = pd.to_numeric(lab_df['labresult'], errors='coerce')

            def _get_vals(keyword: str) -> List[float]:
                sub = lab_df[lab_df['labname'].str.contains(keyword, na=False)]
                return sub['labresult'].dropna().tolist()

            # HbA1c
            hba1c_vals = _get_vals('hba1c')
            if hba1c_vals:
                outcomes['hba1c_admission'] = float(hba1c_vals[0])

            # 血糖 TIR [NEW-4/5]
            glc_vals = _get_vals('glucose')
            tir_metrics = compute_tir_metrics(glc_vals)
            outcomes.update({'glucose_' + k: v for k, v in tir_metrics.items()})

            # AKI 动态分期 [NEW-6]（eICU labresultoffset 为入 ICU 后分钟数）
            # [FIX-P0-3] 旧代码将所有时间戳设为 None，导致 compute_aki_kdigo_dynamic
            # 中基于 48h 窗口的"绝对值升高 ≥0.3"判据完全失效（该分支要求 t_i/t_j 非空）。
            # 修复：用 labresultoffset（分钟）构造伪 datetime，恢复时间窗判据。
            creat_df = lab_df[lab_df['labname'].str.contains('creatinine', na=False)].dropna(
                subset=['labresult']).sort_values('labresultoffset')
            _EICU_PSEUDO_BASE = datetime(2000, 1, 1)  # 任意基准点，只需相对间隔正确
            creat_series = []
            for _, crow in creat_df.iterrows():
                offset_min = crow.get('labresultoffset')
                if pd.notna(offset_min):
                    pseudo_t = _EICU_PSEUDO_BASE + timedelta(minutes=float(offset_min))
                else:
                    pseudo_t = None
                creat_series.append((pseudo_t, float(crow['labresult'])))
            aki_result   = compute_aki_kdigo_dynamic(creat_series)
            outcomes.update({'aki_kdigo_' + k: v for k, v in aki_result.items()})

            # Lab-SOFA 代理 [NEW-2]
            cr_vals = _get_vals('creatinine')
            bi_vals = _get_vals('bilirubin')
            pl_vals = _get_vals('platelets')
            outcomes['lab_sofa_proxy'] = compute_lab_sofa_proxy(
                cr_vals[0] if cr_vals else None,
                bi_vals[0] if bi_vals else None,
                pl_vals[0] if pl_vals else None,
            )

            # 入院 eGFR [NEW-9]
            try:
                age_raw = pat_row.get('age')
                gender  = str(pat_row.get('gender', ''))
                # eICU age 有时是 "> 89" 字符串
                if isinstance(age_raw, str) and '>' in age_raw:
                    age = 91.4
                else:
                    age = float(age_raw) if pd.notna(age_raw) else None
                cr_val = cr_vals[0] if cr_vals else None
                outcomes['egfr_admission'] = (
                    compute_egfr_ckd_epi_2021(cr_val, age, gender)
                    if (cr_val and age) else None
                )
            except Exception:
                outcomes['egfr_admission'] = None

        # ── 院内死亡 ────────────────────────────────────────────────────────
        disch_status = str(pat_row.get('unitdischargestatus', '') or '').lower()
        hosp_status  = str(pat_row.get('hospitaldischargestatus', '') or '').lower()
        outcomes['in_hospital_mortality'] = int(
            'expired' in disch_status or 'expired' in hosp_status
        )
        outcomes['mortality_28d'] = outcomes['in_hospital_mortality']  # eICU 无随访数据

        # ── 再入院 [FIX-4] → [FIX-P0-2] ────────────────────────────────────
        # [FIX-P0-2] eICU 没有跨 stay 的绝对时间戳，无法精确计算两次入住间隔天数。
        # 旧逻辑将所有有后续 stay 的患者标记为 readmission_30d=1，
        # 这会系统性膨胀 Bad Outcome（OutcomeLabelGenerator 将 readmission_30d=1 判为 Bad）。
        #
        # 修复策略：
        #   - 有后续 stay → readmission_30d=None（未知，不参与终点判断）
        #   - 无后续 stay → readmission_30d=0（确定未再入院）
        #   - 90d 同理
        # OutcomeLabelGenerator 中 None 值不触发 Bad Outcome 判定。
        # 论文中需在 Limitations 里说明 eICU 再入院数据缺失。
        try:
            uniquepid  = str(pat_row.get('uniquepid', ''))
            all_stays  = self._uid_to_stays.get(uniquepid, [])
            future_stays = [s for s in all_stays if s > stay_id]

            if future_stays:
                # 存在后续 ICU 入住，但时间间隔未知
                outcomes['readmission_30d'] = None   # [FIX-P0-2] 无法确定是否 ≤30d
                outcomes['readmission_90d'] = None
                outcomes['readmission_any'] = 1      # 补充：确定存在再入院（不限时间窗）
            else:
                outcomes['readmission_30d'] = 0
                outcomes['readmission_90d'] = 0
                outcomes['readmission_any'] = 0
        except Exception:
            outcomes['readmission_30d'] = None
            outcomes['readmission_90d'] = None
            outcomes['readmission_any'] = None

        return outcomes


# ═══════════════════════════════════════════════════════════════════════════════
# 七、结局标签生成器（供 pilot_sampler.py 使用）
# ═══════════════════════════════════════════════════════════════════════════════

class OutcomeLabelGenerator:
    """
    根据提取的 outcomes 生成标准化的 Bad/Good/Moderate 结局标签。
    
    Bad Outcome（复合终点，任意一项即可）：
      - 院内死亡
      - 严重低血糖（<54 mg/dL，次数 ≥1）
      - 新发 DKA 或 HHS
      - AKI 进展（分期上升 ≥1 期）
      - 30 日非计划再入院

    Good Outcome：
      - 以上均未发生
      - 且顺利出院（非 expired/hospice）

    Moderate：介于两者之间（如发生轻度 AKI 但未进展，无再入院）
    """

    # SNF (Skilled Nursing Facility) 已移除：CMS 定义下 SNF 是 post-acute care
    # 正常过渡路径（尤其老年 ICU 患者），不应归为不良出院去向。
    # MIMIC-IV 官方文档亦将 SNF 与 home/rehab 并列为常规出院去向。
    BAD_DISCHARGE_KEYWORDS = ['expired', 'hospice', 'dead', 'died']

    def generate(self, outcomes: Dict) -> str:
        """
        Returns:
            'Good' | 'Bad' | 'Moderate' | 'Unknown'
        """
        label, _ = self.generate_with_flags(outcomes)
        return label

    def generate_with_flags(self, outcomes: Dict) -> Tuple[str, List[str]]:
        """
        与 generate() 逻辑完全一致，但同时返回触发 Bad 判定的具体组件列表。

        Returns:
            (outcome_label, bad_flags)
            bad_flags 可能的值：
              'in_hospital_mortality', 'severe_hypoglycemia', 'dka_or_hhs',
              'aki_progression', 'readmission_30d', 'bad_discharge_destination'
            若 label != 'Bad'，bad_flags 为空列表。
        """
        if not outcomes:
            return 'Unknown', []

        # Bad Outcome 判断
        bad_flags = []

        # 1. 院内死亡
        if outcomes.get('in_hospital_mortality', 0) == 1:
            bad_flags.append('in_hospital_mortality')

        # 2. 严重低血糖
        if (outcomes.get('glucose_severe_hypo_count') or 0) >= 1:
            bad_flags.append('severe_hypoglycemia')

        # 3. DKA / HHS
        if outcomes.get('dka_event', 0) == 1 or outcomes.get('hhs_event', 0) == 1:
            bad_flags.append('dka_or_hhs')

        # 4. AKI 进展
        if outcomes.get('aki_kdigo_aki_progression', False):
            bad_flags.append('aki_progression')

        # 5. 30 日再入院
        # [FIX-P0-2] readmission_30d=None 表示数据不可用（如 eICU 无绝对时间戳），
        # None == 1 为 False，不触发 Bad 判定——这是正确行为。
        readmit_30 = outcomes.get('readmission_30d')
        if readmit_30 is not None and pd.notna(readmit_30) and int(readmit_30) == 1:
            bad_flags.append('readmission_30d')

        if bad_flags:
            return 'Bad', bad_flags

        # Good Outcome 判断（额外检查出院去向）
        dest = str(outcomes.get('discharge_destination', '') or '').lower()
        if any(kw in dest for kw in self.BAD_DISCHARGE_KEYWORDS):
            return 'Bad', ['bad_discharge_destination']

        # Moderate：发生了轻度 AKI 但未进展，或 90 日再入院（但非 30 日）
        aki_present = outcomes.get('aki_kdigo_aki_present', False)
        # [FIX-P0-2] readmission_90d=None 不触发 Moderate
        readmit_90_val = outcomes.get('readmission_90d')
        readmit_90  = readmit_90_val is not None and pd.notna(readmit_90_val) and int(readmit_90_val) == 1

        if aki_present or readmit_90:
            return 'Moderate', []

        return 'Good', []

    def get_severity_tier(self, outcomes: Dict) -> str:
        """
        基于 Charlson CCI 和 Lab-SOFA 代理评分确定疾病严重度层。

        Returns:
            'Low' | 'Medium' | 'High'
        """
        cci  = outcomes.get('charlson_cci', 0) or 0
        sofa = outcomes.get('lab_sofa_proxy')

        # 优先基于 CCI（普遍可得）
        if cci <= 2:
            tier = 'Low'
        elif cci <= 5:
            tier = 'Medium'
        else:
            tier = 'High'

        # Lab-SOFA 代理升档修正
        # 排除阈值为 8（pilot_sampler.py），队列内实际范围 0-7
        # ≥ 4 ≈ 至少一个器官系统重度异常（单项 4 分）或两个中度异常（各 2 分）
        # 类比完整 SOFA 24 分制中 ≥ 6 的临床含义
        if sofa is not None and sofa >= 4 and tier == 'Low':
            tier = 'Medium'
        elif sofa is not None and sofa >= 4 and tier == 'Medium':
            tier = 'High'

        return tier


# ═══════════════════════════════════════════════════════════════════════════════
# 八、敏感性分析辅助函数
# ═══════════════════════════════════════════════════════════════════════════════

# 核心复合终点组件（排除 AKI 进展，用于敏感性分析）
_CORE_BAD_COMPONENTS = {
    'in_hospital_mortality', 'severe_hypoglycemia',
    'dka_or_hhs', 'readmission_30d', 'bad_discharge_destination',
}


def _compute_outcome_label_core(row) -> str:
    """
    敏感性分析：排除 AKI 进展后的核心复合终点。

    若原始 Bad 仅由 aki_progression 触发（无其他组件），则降级为原始的
    Moderate/Good 判定逻辑。否则保持 Bad。
    非 Bad 的行保持原始 outcome_label 不变。
    """
    if row.get('outcome_label') != 'Bad':
        return row.get('outcome_label', 'Unknown')

    flags_str = str(row.get('bad_outcome_flags', '') or '')
    if not flags_str:
        return row.get('outcome_label', 'Unknown')

    flags = [f.strip() for f in flags_str.split(',') if f.strip()]
    core_flags = [f for f in flags if f in _CORE_BAD_COMPONENTS]

    if core_flags:
        # 仍有核心组件触发，保持 Bad
        return 'Bad'
    else:
        # 仅 AKI 进展触发的 Bad → 降级
        # 检查是否应为 Moderate（有 AKI 存在但无进展 → 逻辑上对）
        # 但这里 AKI 进展 = True 意味着 AKI present 也为 True → Moderate
        return 'Moderate'


def _compute_expanded_moderate(row) -> str:
    """
    敏感性分析：TIR < 50% 且 CV > 36% 的 Good 患者升级为 Moderate。

    基于 Diabetes Care 2019 国际共识住院血糖管理的 "可接受" 下界。
    仅对 Good 患者生效；Bad 和 Moderate 保持不变。
    """
    if row.get('outcome_label') != 'Good':
        return row.get('outcome_label', 'Unknown')

    tir = row.get('glucose_tir')
    cv = row.get('glucose_cv_pct')

    if (tir is not None and cv is not None
            and pd.notna(tir) and pd.notna(cv)
            and float(tir) < 0.50 and float(cv) > 36):
        return 'Moderate'

    return 'Good'


def _apply_sensitivity_columns(df: pd.DataFrame, label_gen: 'OutcomeLabelGenerator') -> pd.DataFrame:
    """
    对 outcome DataFrame 批量添加敏感性分析列：
      - bad_outcome_flags:           触发 Bad 判定的具体组件（逗号分隔）
      - outcome_label_core:          排除 AKI 进展后的核心复合终点
      - outcome_label_expanded_mod:  TIR/CV 扩展 Moderate 定义
    """
    # generate_with_flags → bad_outcome_flags
    _label_flags = df.apply(
        lambda r: label_gen.generate_with_flags(r.to_dict()), axis=1
    )
    df['outcome_label'] = _label_flags.apply(lambda x: x[0])
    df['bad_outcome_flags'] = _label_flags.apply(
        lambda x: ','.join(x[1]) if x[1] else ''
    )

    # severity_tier（使用修正后的阈值）
    df['severity_tier'] = df.apply(
        lambda r: label_gen.get_severity_tier(r.to_dict()), axis=1
    )

    # 核心复合终点（排除 AKI 进展）
    df['outcome_label_core'] = df.apply(_compute_outcome_label_core, axis=1)

    # 扩展 Moderate（TIR/CV 阈值）
    df['outcome_label_expanded_mod'] = df.apply(_compute_expanded_moderate, axis=1)

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 九、主函数
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """使用示例（与 v1.0 兼容）"""
    try:
        from agents.config import PathConfig
    except ImportError:
        logger.error("无法导入 PathConfig，请检查项目结构")
        return

    label_gen = OutcomeLabelGenerator()

    # MIMIC-III
    mimic3_path   = PathConfig.MIMIC3_PATH
    patient_list3 = PathConfig.MIMIC3_PATIENCE_PATH
    if mimic3_path.exists() and patient_list3.exists():
        logger.info("=" * 70)
        logger.info("提取 MIMIC-III outcomes（v2.1）")
        extractor = MIMICIII_OutcomeExtractor(mimic3_path)
        df3 = extractor.extract_all_outcomes(
            patient_list_path=patient_list3,
            output_path=PathConfig.MIMIC3_PATIENCE_OUTCOME_PATH,
        )
        if not df3.empty:
            df3 = _apply_sensitivity_columns(df3, label_gen)
            logger.info(f"✅ MIMIC-III: {len(df3)} 例，结局分布:\n{df3['outcome_label'].value_counts()}")
            # 敏感性分析对比
            n_core_changed = (df3['outcome_label'] != df3['outcome_label_core']).sum()
            n_mod_expanded = (df3['outcome_label'] != df3['outcome_label_expanded_mod']).sum()
            logger.info(
                f"  敏感性分析: Core(排除AKI)改判 {n_core_changed} 例, "
                f"Expanded-Moderate 改判 {n_mod_expanded} 例"
            )
            df3.to_csv(PathConfig.MIMIC3_PATIENCE_OUTCOME_PATH, index=False)

    # MIMIC-IV
    mimic4_path   = PathConfig.MIMIC4_PATH
    patient_list4 = PathConfig.MIMIC4_PATIENCE_PATH
    if mimic4_path.exists() and patient_list4.exists():
        logger.info("提取 MIMIC-IV outcomes（v2.1）...")
        extractor = MIMICIV_OutcomeExtractor(mimic4_path)
        df4 = extractor.extract_all_outcomes(
            patient_list_path=patient_list4,
            output_path=PathConfig.MIMIC4_PATIENCE_OUTCOME_PATH,
        )
        if not df4.empty:
            df4 = _apply_sensitivity_columns(df4, label_gen)
            logger.info(f"✅ MIMIC-IV: {len(df4)} 例，结局分布:\n{df4['outcome_label'].value_counts()}")
            n_core_changed = (df4['outcome_label'] != df4['outcome_label_core']).sum()
            n_mod_expanded = (df4['outcome_label'] != df4['outcome_label_expanded_mod']).sum()
            logger.info(
                f"  敏感性分析: Core(排除AKI)改判 {n_core_changed} 例, "
                f"Expanded-Moderate 改判 {n_mod_expanded} 例"
            )
            df4.to_csv(PathConfig.MIMIC4_PATIENCE_OUTCOME_PATH, index=False)

    # eICU
    eicu_path   = PathConfig.EICU_PATH
    patient_liste = PathConfig.EICU_PATIENCE_PATH
    if eicu_path.exists() and patient_liste.exists():
        logger.info("提取 eICU outcomes（v2.1）...")
        extractor = eICU_OutcomeExtractor(eicu_path)
        dfe = extractor.extract_all_outcomes(
            patient_list_path=patient_liste,
            output_path=PathConfig.EICU_PATIENCE_OUTCOME_PATH,
        )
        if not dfe.empty:
            dfe = _apply_sensitivity_columns(dfe, label_gen)
            logger.info(f"✅ eICU: {len(dfe)} 例，结局分布:\n{dfe['outcome_label'].value_counts()}")
            n_core_changed = (dfe['outcome_label'] != dfe['outcome_label_core']).sum()
            n_mod_expanded = (dfe['outcome_label'] != dfe['outcome_label_expanded_mod']).sum()
            logger.info(
                f"  敏感性分析: Core(排除AKI)改判 {n_core_changed} 例, "
                f"Expanded-Moderate 改判 {n_mod_expanded} 例"
            )
            dfe.to_csv(PathConfig.EICU_PATIENCE_OUTCOME_PATH, index=False)


if __name__ == "__main__":
    main()