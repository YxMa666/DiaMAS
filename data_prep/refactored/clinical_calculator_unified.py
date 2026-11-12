#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Clinical Calculator - 统一临床指标计算器

核心改进：
1. eGFR：统一使用 CKD-EPI 2021 (无种族校正) —— KDIGO/NKF 当前推荐标准
2. 所有计算公式 **唯一** 存在于本文件，其他模块禁止自行计算
3. 增加 BMI 计算
4. 基于 LOINC itemid 的精确匹配替代模糊字符串搜索

文献依据：
- eGFR: Inker LA, et al. N Engl J Med. 2021;385(19):1737-1749
- ASCVD: Pooled Cohort Equations (2013 ACC/AHA)
- eAG: Nathan DM, et al. Diabetes Care. 2008;31(8):1473-1478
- BNP/HF: McDonagh TA, et al. Eur Heart J. 2021;42(36):3599-3726 (ESC HF Guidelines)
          Januzzi JL, et al. JACC. 2019;74(13):1701-1716 (NT-proBNP age-stratified thresholds)

[FIX-v3.1] 修复列表：
  FIX-ALG-1: _resolve_lab_value 增加 charttime 排序，确保 first/last 有时间语义
  FIX-ALG-2: calculate_all_metrics 优先读取已有 eGFR lab，不再重复计算
  FIX-ALG-3: anion_gap 使用 is not None 判断，修复 falsy 0 导致的静默跳过
  FIX-ALG-4: 新增 MIMIC4_LAB_MAP（基于 MIMIC-IV d_labitems）
  FIX-ALG-5: eGFR 使用基准肌酐（入院24h内最低值），并标记 AKI 疑似
  FIX-CLN-4: glucose 指标重命名 time_in_range → glucose_in_range_rate，增加采样频率
  FIX-CLN-6: 增加 HbA1c 推算 eAG (estimated average glucose)
  FIX-CLN-2: 增加 UACR 提取（支持 CKD G+A 完整分期）

[FIX-v3.2] 新增：
  FIX-CLN-7: 新增 BNP / NT-proBNP 提取（itemid 精确匹配 + 关键词回退）
    背景：ClinicalPhenotyper Stage 2 的"HFrEF 高度怀疑"表型依赖
          flat["bnp"] 和 flat["nt_probnp"]，但 v3.1 的 calculate_all_metrics()
          未写入这两个字段，导致该路径下 hf_score 永远 ≤ 0.20（只有利尿剂贡献），
          无法触发 0.35 阈值 → 心衰表型在无 HF ICD 编码时静默漏检。
    itemid 来源：
      MIMIC-III D_LABITEMS: BNP=51214, NT-proBNP=50963
      MIMIC-IV D_LABITEMS:  BNP=52070, NT-proBNP=52075
    输出位置：metrics["bnp"] 和 metrics["nt_probnp"]（顶层，供 flatten_clinical_metrics 读取）
    单位说明：pg/mL（两者单位相同，但临床阈值不同，使用时必须区分）

作者：Yuxi
日期：2025-02（v3.1）/ 2025 update（v3.2）
"""

import math
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple

from .data_models import (
    CompletePatientData, LabResult, Diagnosis, VitalSign, DataQualityFlags
)

logger = logging.getLogger(__name__)


# ==================== MIMIC-III itemid 精确映射表 ====================
# 来源：MIMIC-III D_LABITEMS

MIMIC3_LAB_MAP = {
    # 血糖相关
    50931: 'glucose',
    50809: 'glucose',
    51478: 'glucose_urine',
    # HbA1c
    50852: 'hba1c',
    # 肾功能
    50912: 'creatinine',
    51081: 'creatinine_urine',
    50862: 'albumin',
    51006: 'bun',
    # 尿白蛋白（用于 UACR）
    51002: 'microalbumin_urine',
    51003: 'albumin_urine',
    # 血脂
    50907: 'cholesterol_total',
    50904: 'cholesterol_hdl',
    50905: 'cholesterol_ldl_direct',
    50906: 'cholesterol_ldl_calc',
    51000: 'triglycerides',
    # 电解质
    50983: 'sodium',
    50971: 'potassium',
    50902: 'chloride',
    50882: 'bicarbonate',
    # 血常规
    51222: 'hemoglobin',
    51279: 'rbc',
    51301: 'wbc',
    51265: 'platelet',
    51250: 'mcv',
    # 肝功能
    50861: 'alt',
    50863: 'alkaline_phosphatase',
    50878: 'ast',
    50885: 'bilirubin_total',
    # 凝血
    51237: 'inr',
    # 血气
    50820: 'ph',
    50821: 'po2',
    50818: 'pco2',
    50813: 'lactate',
    # [FIX-CLN-7] 心脏标志物（用于 HFrEF 表型推断）
    # MIMIC-III D_LABITEMS 来源：
    #   51214: B-type Natriuretic Peptide (BNP)         单位 pg/mL
    #   50963: N-terminal pro-B-type Natriuretic Peptide 单位 pg/mL
    # 临床阈值（ACC/AHA 2022 HF Guideline）：
    #   BNP > 100 pg/mL → 急性 HF 诊断参考值
    #   NT-proBNP: 年龄分层（<50岁>450; 50-75岁>900; >75岁>1800）
    51214: 'bnp',
    50963: 'nt_probnp',
}


# ==================== [FIX-ALG-4] MIMIC-IV itemid 精确映射表 ====================
# 来源：MIMIC-IV d_labitems.csv（itemid 与 III 不同，需独立维护）

MIMIC4_LAB_MAP = {
    # 血糖相关
    52027: 'glucose',           # Glucose（血清）
    50931: 'glucose',           # Glucose（部分 IV 数据继承 III 编码）
    51478: 'glucose_urine',
    # HbA1c
    51637: 'hba1c',             # Hemoglobin A1c（IV 新编码）
    50852: 'hba1c',             # 兼容
    # 肾功能
    52024: 'creatinine',        # Creatinine（血清）
    50912: 'creatinine',        # 兼容
    51081: 'creatinine_urine',
    52023: 'albumin',
    52025: 'bun',
    # 尿白蛋白
    51002: 'microalbumin_urine',
    51003: 'albumin_urine',
    # 血脂
    52019: 'cholesterol_total',
    52016: 'cholesterol_hdl',
    52017: 'cholesterol_ldl_direct',
    52018: 'cholesterol_ldl_calc',
    52020: 'triglycerides',
    # 电解质
    52033: 'sodium',
    52032: 'potassium',
    52028: 'chloride',
    52022: 'bicarbonate',
    # 血常规
    51222: 'hemoglobin',
    51301: 'wbc',
    51265: 'platelet',
    # 肝功能
    52012: 'alt',
    52013: 'ast',
    52014: 'bilirubin_total',
    # 凝血
    51237: 'inr',
    # 血气
    50820: 'ph',
    50821: 'po2',
    50818: 'pco2',
    50813: 'lactate',
    # [FIX-CLN-7] 心脏标志物
    # MIMIC-IV D_LABITEMS 来源：
    #   52070: B-type Natriuretic Peptide (BNP)         单位 pg/mL
    #   52075: NT-proBNP                                单位 pg/mL
    # 注意：MIMIC-IV 部分记录继承了 MIMIC-III 的 itemid，
    # 两个编码均保留以确保兼容性（_resolve_lab_value 会去重）
    52070: 'bnp',
    51214: 'bnp',       # MIMIC-III 兼容编码（部分 MIMIC-IV 沿用）
    52075: 'nt_probnp',
    50963: 'nt_probnp', # MIMIC-III 兼容编码
}


# ==================== 关键词回退匹配规则 ====================
# 仅当 itemid 精确匹配失败时使用（主要用于 eICU）

KEYWORD_FALLBACK_MAP = {
    'glucose':             (['glucose'],              ['urine', 'tolerance', 'csf', 'body fluid']),
    'hba1c':              (['a1c', 'hemoglobin a1c', 'glycated'],     []),
    'creatinine':         (['creatinine'],            ['urine', 'clearance', 'ratio']),
    'sodium':             (['sodium'],                ['urine', 'fractional']),
    'potassium':          (['potassium'],             ['urine']),
    'hemoglobin':         (['hemoglobin'],            ['a1c', 'glycated', 'fetal']),
    'cholesterol_total':  (['cholesterol, total', 'total cholesterol'], []),
    'cholesterol_ldl':    (['ldl', 'low density'],   ['ratio']),
    'cholesterol_hdl':    (['hdl', 'high density'],  ['ratio']),
    'triglycerides':      (['triglyceride'],          []),
    'albumin':            (['albumin'],               ['urine', 'micro']),
    'bun':                (['urea nitrogen', 'bun'],  ['urine']),
    'bicarbonate':        (['bicarbonate', 'hco3'],   ['urine']),
    'chloride':           (['chloride'],              ['urine']),
    # 尿白蛋白
    'microalbumin_urine': (['microalbumin', 'albumin'], ['serum', 'plasma', '/']),
    'creatinine_urine':   (['creatinine'],            ['serum', 'plasma', 'blood']),
    # [FIX-CLN-7] 心脏标志物关键词回退
    # 排除词设计原则：
    #   BNP 排除 'pro' 防止 NT-proBNP 被误匹配为 BNP
    #   nt_probnp 排除 'fragment' / 'precursor' 等非标准报告名
    #   两者均排除 'urine'（尿液 BNP 无临床意义）
    'bnp':       (
        ['bnp', 'b-type natriuretic', 'brain natriuretic'],
        ['pro', 'nt-', 'n-terminal', 'urine'],
    ),
    'nt_probnp': (
        ['nt-probnp', 'nt probnp', 'n-terminal', 'proBNP', 'pro-bnp',
         'pro bnp', 'NT-proBNP'],
        ['urine'],
    ),
}


# ==================== [FIX-ALG-1] 检验值精确提取（含时序排序）====================

def _parse_charttime(ct: str) -> Optional[datetime]:
    """
    解析 charttime 字符串为 datetime 对象

    支持格式：
    - MIMIC: 'YYYY-MM-DD HH:MM:SS'
    - eICU: 'admission+XXhYYm'（返回等效 timedelta 的伪时间，用于排序）
    """
    if not ct:
        return None
    # MIMIC 标准格式
    for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%Y-%m-%d'):
        try:
            return datetime.strptime(ct, fmt)
        except ValueError:
            continue
    # eICU offset 格式：转为从 epoch 起的 timedelta（仅用于排序）
    if ct.startswith('admission+'):
        try:
            parts = ct.replace('admission+', '').replace('h', ' ').replace('m', '').split()
            h, m = int(parts[0]), int(parts[1])
            return datetime(1970, 1, 1) + timedelta(hours=h, minutes=m)
        except Exception:
            pass
    return None


def _resolve_lab_value(
    lab_results: List[LabResult],
    semantic_name: str,
    itemid_map: Dict[int, str] = None,
    strategy: str = 'first',
) -> Optional[Any]:
    """
    精确提取检验值

    [FIX-ALG-1] 核心修复：
    - 匹配到的 lab 先按 charttime 升序排序
    - 'first' = 时间最早，'last' = 时间最近
    - 旧实现依赖列表插入顺序，导致语义错误

    优先级：
    1. itemid 精确匹配（如果提供 itemid_map）
    2. 关键词回退匹配（带排除词过滤）

    Args:
        strategy: 'first' 取时间最早, 'last' 取时间最近,
                  'mean' 取均值, 'all' 返回值列表, 'min' 取最小值
    """
    if itemid_map is None:
        itemid_map = MIMIC3_LAB_MAP

    matched_labs: List[Tuple[Optional[datetime], float]] = []

    # --- Pass 1: itemid 精确匹配 ---
    for lab in lab_results:
        if isinstance(lab.itemid, int) and itemid_map.get(lab.itemid) == semantic_name:
            try:
                matched_labs.append((_parse_charttime(lab.charttime), float(lab.value)))
            except (ValueError, TypeError):
                continue

    # --- Pass 2: 关键词回退 ---
    if not matched_labs and semantic_name in KEYWORD_FALLBACK_MAP:
        include_kws, exclude_kws = KEYWORD_FALLBACK_MAP[semantic_name]
        for lab in lab_results:
            if not lab.label or not isinstance(lab.label, str):
                continue
            label_lower = lab.label.lower()
            if any(kw in label_lower for kw in include_kws):
                if not any(ex in label_lower for ex in exclude_kws):
                    try:
                        matched_labs.append((_parse_charttime(lab.charttime), float(lab.value)))
                    except (ValueError, TypeError):
                        continue

    if not matched_labs:
        return None

    # [FIX-ALG-1] 按时间排序（None 时间置于末尾）
    matched_labs.sort(key=lambda x: (x[0] is None, x[0] or datetime.min))
    values = [v for _, v in matched_labs]

    if strategy == 'first':
        return values[0]
    elif strategy == 'last':
        return values[-1]
    elif strategy == 'mean':
        return sum(values) / len(values)
    elif strategy == 'min':
        return min(values)
    elif strategy == 'all':
        return values
    return values[0]


# ==================== eGFR 计算（唯一实现）====================

def calculate_egfr_ckd_epi_2021(
    creatinine: float,
    age: float,
    gender: str,
) -> Optional[float]:
    """
    CKD-EPI 2021 公式（无种族校正）—— 唯一的 eGFR 计算入口

    公式：
        Female: 142 × min(Scr/0.7, 1)^(−0.241) × max(Scr/0.7, 1)^(−1.200) × 0.9938^age × 1.012
        Male:   142 × min(Scr/0.9, 1)^(−0.302) × max(Scr/0.9, 1)^(−1.200) × 0.9938^age × 1.0

    文献：Inker LA, et al. N Engl J Med. 2021;385(19):1737-1749.
    """
    if creatinine is None or creatinine <= 0:
        return None
    if age is None or age <= 0 or age > 150:
        return None
    if not gender:
        return None

    gender_upper = gender.upper().strip()
    is_female = gender_upper in ('F', 'FEMALE')

    kappa = 0.7 if is_female else 0.9
    alpha = -0.241 if is_female else -0.302
    gender_factor = 1.012 if is_female else 1.0

    scr_ratio = creatinine / kappa
    egfr = (
        142
        * (min(scr_ratio, 1.0) ** alpha)
        * (max(scr_ratio, 1.0) ** (-1.200))
        * (0.9938 ** age)
        * gender_factor
    )

    return round(egfr, 1)


# ==================== BMI 计算 ====================

def calculate_bmi(height_cm: Optional[float], weight_kg: Optional[float]) -> Optional[float]:
    """计算 BMI"""
    if not height_cm or not weight_kg:
        return None
    if height_cm <= 0 or weight_kg <= 0:
        return None
    height_m = height_cm / 100.0
    bmi = weight_kg / (height_m ** 2)
    if bmi < 10 or bmi > 80:
        logger.warning(f"BMI 值异常: {bmi:.1f} (h={height_cm}, w={weight_kg})")
        return None
    return round(bmi, 1)


# ==================== CKD 分期 ====================

def classify_ckd_stage(egfr: Optional[float]) -> Optional[str]:
    """
    CKD eGFR 分期（KDIGO 2012，G 分期）

    注意：完整 KDIGO 分期需结合白蛋白尿（A 分期）。
    此函数仅返回 G 分期，完整 G+A 分期在 calculate_all_metrics 中组合。
    """
    if egfr is None:
        return None
    if egfr >= 90:   return 'G1'
    if egfr >= 60:   return 'G2'
    if egfr >= 45:   return 'G3a'
    if egfr >= 30:   return 'G3b'
    if egfr >= 15:   return 'G4'
    return 'G5'


def classify_ckd_albuminuria_stage(acr: Optional[float]) -> Optional[str]:
    """
    CKD 白蛋白尿分期（KDIGO 2012，A 分期）

    [FIX-CLN-2] 新增：支持 eGFR+ACR 完整分期
    ACR 单位：mg/g

    Returns:
        'A1' (<30, 正常至轻度), 'A2' (30-300, 中度), 'A3' (>300, 重度) or None
    """
    if acr is None:
        return None
    if acr < 30:    return 'A1'
    if acr <= 300:  return 'A2'
    return 'A3'


# ==================== LDL 估算 ====================

def calculate_ldl_friedewald(
    total_cholesterol: Optional[float],
    hdl: Optional[float],
    triglycerides: Optional[float]
) -> Optional[float]:
    """
    Friedewald 公式估算 LDL-C

    前提：TG < 400 mg/dL（TG ≥ 400 时 Friedewald 公式误差不可接受）
    LDL = TC - HDL - TG/5
    """
    if total_cholesterol is None or hdl is None or triglycerides is None:
        return None
    if triglycerides >= 400:
        return None
    ldl = total_cholesterol - hdl - (triglycerides / 5.0)
    return round(ldl, 1)


# ==================== [FIX-CLN-6] eAG 计算 ====================

def calculate_eag_from_hba1c(hba1c: Optional[float]) -> Optional[float]:
    """
    从 HbA1c 推算估计平均血糖（eAG）

    公式：eAG (mg/dL) = 28.7 × HbA1c(%) - 46.7
    文献：Nathan DM, et al. Diabetes Care. 2008;31(8):1473-1478

    与离散血糖均值的关系：
    - eAG 代表过去 2-3 个月的平均血糖
    - 住院期间的即时血糖均值反映急性状态，二者不可相互替代

    Args:
        hba1c: HbA1c 百分比（如 7.5）

    Returns:
        eAG (mg/dL) 或 None
    """
    if hba1c is None or hba1c < 4.0 or hba1c > 20.0:
        return None
    return round(28.7 * hba1c - 46.7, 1)


# ==================== [FIX-ALG-5] 基准肌酐提取 ====================

def _get_baseline_creatinine(
    data: CompletePatientData,
    itemid_map: Dict[int, str],
    baseline_window_hours: float = 24.0,
) -> Tuple[Optional[float], str]:
    """
    提取基准肌酐（用于 eGFR 计算）

    [FIX-ALG-5] ICU 场景下的关键改进：
    ICU 患者入院时常因脓毒症/失血/造影剂等导致 AKI，入院即刻肌酐可能已显著升高。
    KDIGO 建议使用能代表稳态 GFR 的肌酐。

    策略：
    1. 获取入院 24h 内的所有肌酐值
    2. 取最低值（最低肌酐最接近 AKI 发生前的真实基线）
    3. 若 24h 内仅有一条，则使用该条（并标注来源）
    4. 同时计算 AKI 疑似指标（入院首条 / 住院最低 ≥ 1.5）

    Args:
        baseline_window_hours: 基线窗口（默认入院后24h）

    Returns:
        (creatinine_value, source_description)
    """
    if not data.admission_time:
        # 无入院时间时回退到时序最早值
        val = _resolve_lab_value(data.all_lab_results, 'creatinine', itemid_map, 'first')
        return val, 'admission_earliest_no_time'

    admit_dt = None
    for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%d %H:%M', '%Y-%m-%d'):
        try:
            admit_dt = datetime.strptime(data.admission_time, fmt)
            break
        except ValueError:
            continue

    if admit_dt is None:
        val = _resolve_lab_value(data.all_lab_results, 'creatinine', itemid_map, 'first')
        return val, 'admission_earliest_no_time'

    # 收集入院 baseline_window_hours 内的所有肌酐值
    window_end = admit_dt + timedelta(hours=baseline_window_hours)
    window_creat = []

    for lab in data.all_lab_results:
        # 判断是否为肌酐
        is_creat = False
        if isinstance(lab.itemid, int) and itemid_map.get(lab.itemid) == 'creatinine':
            is_creat = True
        elif lab.label and 'creatinine' in lab.label.lower() and 'urine' not in lab.label.lower():
            is_creat = True

        if not is_creat or lab.is_calculated:
            continue

        lab_dt = _parse_charttime(lab.charttime)
        if lab_dt is None:
            continue

        # eICU offset（伪时间 < 1971 年）
        if lab_dt.year == 1970:
            offset_h = (lab_dt - datetime(1970, 1, 1)).total_seconds() / 3600
            if offset_h <= baseline_window_hours:
                try:
                    window_creat.append(float(lab.value))
                except (ValueError, TypeError):
                    pass
        else:
            if admit_dt <= lab_dt <= window_end:
                try:
                    window_creat.append(float(lab.value))
                except (ValueError, TypeError):
                    pass

    if window_creat:
        baseline_val = min(window_creat)  # 取最低值
        source = 'first_24h_min'
        return baseline_val, source
    else:
        # 24h 内无肌酐，回退到最早一条
        val = _resolve_lab_value(data.all_lab_results, 'creatinine', itemid_map, 'first')
        return val, 'admission_earliest'


def _detect_aki(data: CompletePatientData, itemid_map: Dict[int, str]) -> bool:
    """
    AKI 疑似检测（KDIGO Stage 1 标准：肌酐升高 ≥ 1.5 倍基线）

    [FIX-ALG-5] 将入院时肌酐与住院期间最低肌酐比较。
    如果入院肌酐 ÷ 住院最低肌酐 ≥ 1.5，提示入院时已有 AKI。

    注意：这是简化的 AKI 筛查，不替代 KDIGO 完整标准（需要时序分析）
    """
    all_creat = _resolve_lab_value(data.all_lab_results, 'creatinine', itemid_map, 'all')
    if not all_creat or len(all_creat) < 2:
        return False

    admission_creat = all_creat[0]   # 时序最早（_resolve_lab_value 已排序）
    min_creat = min(all_creat)

    if min_creat <= 0:
        return False

    ratio = admission_creat / min_creat
    return ratio >= 1.5


# ==================== UACR 计算 ====================

def _calculate_uacr(
    lab_results: List[LabResult],
    itemid_map: Dict[int, str],
) -> Optional[float]:
    """
    计算尿白蛋白/肌酐比（UACR，mg/g）

    [FIX-CLN-2] 支持 CKD G+A 完整分期（KDIGO 2012）

    方法：
    1. 优先使用直接测量的 UACR（如果检验名含 'albumin/creatinine'）
    2. 否则用 尿白蛋白(mg/L) ÷ 尿肌酐(g/L) 计算

    Returns:
        UACR (mg/g) 或 None
    """
    # 尝试直接 UACR 结果
    for lab in lab_results:
        if lab.label and any(
            kw in lab.label.lower()
            for kw in ['albumin/creatinine', 'acr', 'uacr', 'alb/cr']
        ):
            try:
                return float(lab.value)
            except (ValueError, TypeError):
                continue

    # 计算法：尿白蛋白 / 尿肌酐
    urine_albumin = _resolve_lab_value(lab_results, 'microalbumin_urine', itemid_map, 'first')
    urine_creatinine = _resolve_lab_value(lab_results, 'creatinine_urine', itemid_map, 'first')

    if urine_albumin is not None and urine_creatinine is not None and urine_creatinine > 0:
        # 尿肌酐单位通常为 mg/dL，需转换为 g/L (×0.1)
        # 尿白蛋白单位通常为 mg/L
        # UACR = albumin(mg/L) / creatinine(g/L) = albumin / (creatinine_mgdL × 0.1)
        urine_creat_gl = urine_creatinine * 0.1  # mg/dL → g/L
        if urine_creat_gl > 0:
            uacr = urine_albumin / urine_creat_gl
            return round(uacr, 1)

    return None


# ==================== 主计算器 ====================

class UnifiedClinicalCalculator:
    """
    统一临床指标计算器

    所有临床计算的 **唯一入口**，确保公式一致性
    """

    def __init__(self, database: str = 'mimic3'):
        """
        Args:
            database: 'mimic3' / 'mimic4' / 'eicu' → 选择对应的 itemid 映射表
        """
        self.database = database
        if database == 'mimic3':
            self.itemid_map = MIMIC3_LAB_MAP
        elif database == 'mimic4':
            # [FIX-ALG-4] MIMIC-IV 使用独立映射表
            self.itemid_map = MIMIC4_LAB_MAP
        else:
            # eICU：无 itemid，完全依赖关键词
            self.itemid_map = {}

    def calculate_and_append_egfr(self, data: CompletePatientData) -> CompletePatientData:
        """
        计算 eGFR 并追加到检验列表

        [FIX-ALG-5] 使用基准肌酐（入院24h内最低值）而非"列表第一条"
        同时更新 data_quality 中的 baseline_creatinine_source 和 aki_suspected

        这是整个系统中 eGFR 计算的 **唯一入口**
        """
        # 检查是否已经计算过（避免重复追加）
        for lab in data.all_lab_results:
            if lab.is_calculated and lab.label and 'egfr' in lab.label.lower():
                logger.debug(f"  eGFR 已存在，跳过计算 ({data.subject_id}/{data.hadm_id})")
                return data

        # [FIX-ALG-5] 使用基准肌酐
        creatinine, source = _get_baseline_creatinine(data, self.itemid_map)

        # 同时检测 AKI
        aki = _detect_aki(data, self.itemid_map)

        if creatinine is None:
            logger.debug(f"  无法计算 eGFR: 缺少肌酐值 ({data.subject_id}/{data.hadm_id})")
            return data

        egfr = calculate_egfr_ckd_epi_2021(creatinine, data.age, data.gender)
        if egfr is None:
            return data

        ckd_stage = classify_ckd_stage(egfr)

        egfr_lab = LabResult(
            itemid=999999,
            label='eGFR (CKD-EPI 2021)',
            loinc_code='98979-8',
            value=egfr,
            unit='mL/min/1.73m²',
            charttime=data.admission_time or '',
            is_abnormal=(egfr < 60),
            is_calculated=True,
            time_window='baseline',
        )
        data.all_lab_results.append(egfr_lab)

        # 更新数据质量标记
        if data.data_quality is None:
            data.data_quality = DataQualityFlags()
        data.data_quality.baseline_creatinine_source = source
        data.data_quality.aki_suspected = aki

        logger.debug(
            f"  eGFR={egfr:.1f} (CKD {ckd_stage}), "
            f"Scr={creatinine:.2f} [{source}], "
            f"age={data.age}, sex={data.gender}, AKI={aki}"
        )
        return data

    def calculate_all_metrics(
        self,
        data: CompletePatientData,
        levels: List[int] = None,
    ) -> Dict[str, Any]:
        """
        计算所有临床指标

        [FIX-ALG-2] 优先读取已经计算好的 eGFR lab，不再重复调用 calculate_egfr_ckd_epi_2021
        [FIX-CLN-2] 增加 UACR 和 CKD G+A 完整分期
        [FIX-CLN-4] glucose 指标重命名，增加采样频率
        [FIX-CLN-6] 增加 eAG
        [FIX-CLN-7] 新增 BNP / NT-proBNP（顶层字段，供 flatten_clinical_metrics 读取）

        Args:
            data: 完整患者数据
            levels: 计算级别 [1=核心, 2=推荐, 3=可选]
        """
        if levels is None:
            levels = [1, 2]

        metrics = {
            'version': 'v3.2_unified',
            'egfr_formula': 'CKD-EPI 2021 (race-free)',
            'levels_calculated': levels,
        }

        labs = data.all_lab_results
        resolve = lambda name, strat='first': _resolve_lab_value(
            labs, name, self.itemid_map, strat
        )

        # ---- Level 1: 核心指标 ----
        if 1 in levels:
            # 血糖
            glucose_vals = _resolve_lab_value(labs, 'glucose', self.itemid_map, 'all')
            metrics['glucose'] = self._calc_glucose_metrics(glucose_vals, data.los_days)

            # HbA1c
            hba1c = resolve('hba1c')
            metrics['hba1c'] = hba1c

            # [FIX-CLN-6] eAG 推算
            metrics['eag_from_hba1c'] = calculate_eag_from_hba1c(hba1c)

            # [FIX-ALG-2] 肾功能：优先读取已计算的 eGFR，不重复计算
            creatinine = resolve('creatinine')
            egfr = self._get_cached_egfr(data)
            if egfr is None and creatinine is not None:
                # 缓存中没有，才重新计算（理论上此处不应到达）
                egfr = calculate_egfr_ckd_epi_2021(creatinine, data.age, data.gender)

            ckd_g_stage = classify_ckd_stage(egfr)

            # [FIX-CLN-2] UACR 和 G+A 分期
            uacr = _calculate_uacr(labs, self.itemid_map)
            ckd_a_stage = classify_ckd_albuminuria_stage(uacr)
            ckd_full_stage = (
                f"{ckd_g_stage}{ckd_a_stage}" if ckd_g_stage and ckd_a_stage
                else ckd_g_stage or None
            )

            metrics['kidney'] = {
                'creatinine': creatinine,
                'egfr': egfr,
                'ckd_g_stage': ckd_g_stage,            # G1-G5
                'uacr': uacr,                           # [FIX-CLN-2] 新增
                'ckd_a_stage': ckd_a_stage,             # [FIX-CLN-2] A1/A2/A3
                'ckd_full_stage': ckd_full_stage,       # [FIX-CLN-2] 如 'G2A2'
                'aki_suspected': (
                    data.data_quality.aki_suspected
                    if data.data_quality else None
                ),
                'baseline_creatinine_source': (
                    data.data_quality.baseline_creatinine_source
                    if data.data_quality else None
                ),
                'bun': resolve('bun'),
                'albumin': resolve('albumin'),
                'data_available': creatinine is not None,
            }

            # BMI
            metrics['bmi'] = data.bmi or calculate_bmi(data.height_cm, data.weight_kg)

        # ---- Level 2: 推荐指标 ----
        if 2 in levels:
            tc = resolve('cholesterol_total')
            hdl = resolve('cholesterol_hdl')
            ldl = resolve('cholesterol_ldl_direct') or resolve('cholesterol_ldl_calc')
            tg = resolve('triglycerides')

            if ldl is None:
                ldl = calculate_ldl_friedewald(tc, hdl, tg)

            metrics['lipids'] = {
                'total_cholesterol': tc,
                'hdl': hdl,
                'ldl': ldl,
                'triglycerides': tg,
                'data_available': any(v is not None for v in [tc, hdl, ldl, tg]),
            }

            # [FIX-ALG-3] 电解质：用 is not None 代替 all([...])，修复 falsy 0 问题
            na = resolve('sodium')
            k = resolve('potassium')
            cl = resolve('chloride')
            hco3 = resolve('bicarbonate')

            anion_gap = None
            if na is not None and cl is not None and hco3 is not None:
                anion_gap = round(na - (cl + hco3), 1)

            metrics['electrolytes'] = {
                'sodium': na,
                'potassium': k,
                'chloride': cl,
                'bicarbonate': hco3,
                'anion_gap': anion_gap,
                'data_available': any(v is not None for v in [na, k]),
            }

            # 血常规
            hgb = resolve('hemoglobin')
            metrics['cbc'] = {
                'hemoglobin': hgb,
                'wbc': resolve('wbc'),
                'platelet': resolve('platelet'),
                'data_available': hgb is not None,
            }

            # [FIX-CLN-7] 心脏标志物
            # 取住院期间首次测量值（first），与肾功能 / 电解质保持一致。
            # 使用首次而非峰值的理由：
            #   ClinicalPhenotyper Stage 2 用 BNP/NT-proBNP 推断是否存在心衰，
            #   目的是判断"入院时心脏状态"而非追踪动态变化。
            #   峰值更适合预后研究，首次更适合入院诊断推断。
            # 单位：pg/mL（两者相同，但阈值差异悬殊，使用方必须区分）
            bnp_val      = resolve('bnp',      'first')
            nt_probnp_val = resolve('nt_probnp', 'first')

            # 顶层写入：flatten_clinical_metrics 通过 cm.get("bnp") 读取
            # 注意：不写入嵌套 section，因为 diamas_inference_stage.py 的
            # flatten_clinical_metrics 直接读 cm.get("bnp") 和 cm.get("nt_probnp")
            metrics['bnp']       = bnp_val
            metrics['nt_probnp'] = nt_probnp_val

            logger.debug(
                f"  心脏标志物: BNP={bnp_val} pg/mL, NT-proBNP={nt_probnp_val} pg/mL"
            )

        return metrics

    @staticmethod
    def _get_cached_egfr(data: CompletePatientData) -> Optional[float]:
        """
        从 all_lab_results 中读取已计算的 eGFR 值

        [FIX-ALG-2] 避免在 calculate_all_metrics 中重复计算 eGFR
        """
        for lab in data.all_lab_results:
            if lab.is_calculated and lab.label and 'egfr' in lab.label.lower():
                try:
                    return float(lab.value)
                except (ValueError, TypeError):
                    continue
        return None

    @staticmethod
    def _calc_glucose_metrics(values, los_days: Optional[float] = None) -> Dict[str, Any]:
        """
        血糖指标计算

        [FIX-CLN-4] 核心修复：
        - 将 time_in_range_70_180 重命名为 glucose_in_range_rate，并加注释说明
          原名称暗示"时间"在范围内，但实际是"测量点"比例，与 CGM-TIR 概念冲突
        - 增加 glucose_sampling_frequency（次/天）用于可靠性评估
        - 保留 time_in_range_70_180 作为向后兼容别名（下一版本移除）
        """
        if not values:
            return {'data_available': False, 'measurement_count': 0}

        import numpy as np
        arr = np.array(values, dtype=float)
        mean_g = float(np.mean(arr))
        std_g = float(np.std(arr))

        # [FIX-CLN-4] 采样频率（次/天）
        sampling_freq = None
        if los_days and los_days > 0:
            sampling_freq = round(len(arr) / los_days, 1)

        # 达标率（测量点比例，非时间比例）
        in_range_rate = round(float(np.sum((arr >= 70) & (arr <= 180)) / len(arr) * 100), 1)

        return {
            # [FIX-CLN-4] 正确命名：测量点达标率（非 CGM-TIR）
            'glucose_in_range_rate': in_range_rate,
            # 向后兼容别名（逐步废弃，下版本移除）
            'time_in_range_70_180': in_range_rate,

            'mean_glucose': round(mean_g, 1),
            'std_glucose': round(std_g, 1),
            'glucose_cv': round(std_g / mean_g * 100, 1) if mean_g > 0 else None,
            'min_glucose': round(float(np.min(arr)), 1),
            'max_glucose': round(float(np.max(arr)), 1),
            'time_above_180': round(float(np.sum(arr > 180) / len(arr) * 100), 1),
            'time_below_70': round(float(np.sum(arr < 70) / len(arr) * 100), 1),
            'hypoglycemia_events': int(np.sum(arr < 70)),
            'severe_hypoglycemia_events': int(np.sum(arr < 54)),
            'measurement_count': len(arr),
            # [FIX-CLN-4] 采样频率（用于 TIR 可靠性评估）
            'glucose_sampling_frequency_per_day': sampling_freq,
            'data_available': True,
        }
