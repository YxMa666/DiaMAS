#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clinical Validator - 临床数据质量校验器

核心功能：
1. 检验值合理性校验（极端值检测与标记）
2. 数据完整性评估（关键指标缺失率）
3. 时间窗口分类（基线 vs 动态监测 vs 出院前）
4. 糖尿病分型证据链验证
5. 糖尿病慢性并发症结构化提取（新增）

临床依据：
- 检验参考范围来自 ARUP Laboratories / Mayo Clinic reference ranges
- 极端值阈值参考 Hamill J, et al. JAMIA 2013 ("10× rule")
- DM 分型：ADA Standards of Medical Care in Diabetes 2024
- CKD 分期：KDIGO 2012 Clinical Practice Guideline for CKD

[FIX-v3.1] 修复列表：
  FIX-CLN-3a: 降低"仅胰岛素"的 T1DM 权重（ICU 场景下不可靠），需结合 LOS 和出院带药
  FIX-CLN-3b: 增加 LADA 可能性标记（年龄 30-60 + T1DM ICD + 口服药史）
  FIX-CLN-3c: 调整 ICD 编码和用药的相对权重（ICD 来源质量参差，不应独自主导分型）
  FIX-CLN-5: 增加 dm_complications 结构化提取（基于 ICD 编码）
  FIX-CLN-6: HbA1c 时机分类（baseline vs during_stay）
  FIX-ALG-7: validate_all 中缓存 DM 分型结果到 data_quality，避免质量报告重复计算

[FIX-v3.2]
  FIX-CLN-8: extract_dm_complications 新增 BNP / NT-proBNP 补充判定 heart_failure。
    背景：
      extract_dm_complications 在 validate_all 中调用时，clinical_metrics 里已经有
      calculator.calculate_all_metrics() 写入的 bnp 和 nt_probnp（v3.2 新增）。
      但原实现只读 all_diagnoses，对无 HF ICD 编码但 BNP 显著升高的患者，
      heart_failure 恒为 False → has_heart_failure = False →
      ClinicalPhenotyper Stage 1 HF 路径静默跳过，只能靠 Stage 2 BNP 概率路径兜底。

    修复策略（三档证据强度）：
      Level A（ICD 确认，conf 0.95）：I50.*/428.* → heart_failure=True, source='icd'
      Level B（BNP 确认，conf 0.80）：
        BNP > 100 pg/mL（ACC/AHA 急性 HF 诊断阈值）
        NT-proBNP > 年龄分层阈值（Januzzi 2019: <50→450, 50-75→900, >75→1800）
        → heart_failure=True, source='bnp'/'nt_probnp'
      Level C（BNP 提示，conf 0.35-0.85）：
        BNP 35-100 pg/mL 或 NT-proBNP > 0.5× 阈值但未过阈值
        → heart_failure=False（交由 ClinicalPhenotyper Stage 2 处理）

    输出新增字段：
      heart_failure_source: 'icd' | 'bnp' | 'nt_probnp' | None
        供论文 Methods 统计各路径覆盖率，也供 ClinicalPhenotyper
        按来源区分置信度（Stage 1 读 has_heart_failure 时已有此分层）

作者：Yuxi
日期：2025-02（v3.1）/ 2025 update（v3.2）
"""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from .data_models import (
    CompletePatientData, LabResult, Diagnosis, Medication,
    DataQualityFlags, LabTimeWindow, MedTimeWindow
)

logger = logging.getLogger(__name__)


# ==================== 临床合理范围定义 ====================

PLAUSIBLE_RANGES = {
    'glucose':          (1,    2000,  'mg/dL'),
    'hba1c':            (2.0,  20.0,  '%'),
    'creatinine':       (0.1,  30.0,  'mg/dL'),
    'sodium':           (100,  180,   'mEq/L'),
    'potassium':        (1.0,  12.0,  'mEq/L'),
    'hemoglobin':       (1.0,  25.0,  'g/dL'),
    'platelet':         (1,    2000,  'K/uL'),
    'wbc':              (0.1,  500,   'K/uL'),
    'bilirubin':        (0.0,  60.0,  'mg/dL'),
    'alt':              (0,    10000, 'IU/L'),
    'ast':              (0,    10000, 'IU/L'),
    'albumin':          (0.5,  7.0,   'g/dL'),
    'bun':              (0,    300,   'mg/dL'),
    'cholesterol':      (10,   1000,  'mg/dL'),
    'ldl':              (5,    600,   'mg/dL'),
    'hdl':              (2,    200,   'mg/dL'),
    'triglycerides':    (10,   5000,  'mg/dL'),
    'temperature':      (25.0, 45.0,  '°C'),
    'heart rate':       (10,   300,   'bpm'),
    'systolic':         (30,   350,   'mmHg'),
    'diastolic':        (10,   250,   'mmHg'),
    'spo2':             (30,   100,   '%'),
    'respiratory rate': (2,    80,    'breaths/min'),
}

# 糖尿病相关药物关键词
DIABETES_DRUG_KEYWORDS = {
    # T2DM 口服药
    'metformin', 'glyburide', 'glipizide', 'glimepiride',
    'pioglitazone', 'rosiglitazone',
    'sitagliptin', 'saxagliptin', 'linagliptin', 'alogliptin',
    'empagliflozin', 'dapagliflozin', 'canagliflozin', 'ertugliflozin',
    'liraglutide', 'semaglutide', 'dulaglutide', 'exenatide', 'tirzepatide',
    'acarbose', 'miglitol', 'colesevelam', 'bromocriptine',
    # 胰岛素
    'insulin', 'lantus', 'levemir', 'tresiba', 'basaglar',
    'humalog', 'novolog', 'apidra', 'fiasp',
    'humulin', 'novolin', 'nph',
    # INN 命名（论文要求与 diamas_inference_stage 保持一致）
    'glargine', 'detemir', 'degludec', 'aspart', 'lispro', 'glulisine',
}

# [FIX-CLN-5] 糖尿病慢性并发症 ICD 映射表
# 来源：ICD-10-CM 2024, ICD-9-CM
DM_COMPLICATION_ICD = {
    # 糖尿病视网膜病变
    'retinopathy': {
        'icd10_prefixes': ['E10.3', 'E11.3', 'E13.3'],
        'icd9_codes': ['250.5', '362.01', '362.02', '362.03', '362.04', '362.05', '362.06'],
        'description': '糖尿病视网膜病变',
    },
    # 糖尿病肾病
    'nephropathy': {
        'icd10_prefixes': ['E10.2', 'E11.2', 'N08'],
        'icd9_codes': ['250.4', '583.81'],
        'description': '糖尿病肾病',
    },
    # 糖尿病神经病变（纯神经病变编码，不含周围血管/足部）
    # [FIX-C1] 移除 E10.5/E11.5（周围循环并发症），这两个编码已归入 peripheral_vascular
    #           E10.4/E11.4 = 糖尿病伴神经并发症（纯感觉/自主/运动神经病变）
    'neuropathy': {
        'icd10_prefixes': ['E10.4', 'E11.4'],
        'icd9_codes': ['250.6', '357.2', '713.5'],
        'description': '糖尿病神经病变',
    },
    # 糖尿病足 / 外周血管病变
    # E10.5/E11.5 = 周围循环并发症（血管为主）；E10.6/E11.6 = 其他特定并发症（含足部溃疡）
    'peripheral_vascular': {
        'icd10_prefixes': ['E10.5', 'E11.5', 'E10.6', 'E11.6'],
        'icd9_codes': ['250.7', '785.4'],
        'description': '糖尿病外周血管病变/糖尿病足',
    },
    # 心血管病 / ASCVD（糖尿病相关，不含 HF——HF 单独列项）
    'cardiovascular': {
        'icd10_prefixes': ['I21', 'I22', 'I25', 'I63', 'I64'],
        'icd9_codes': ['410', '411', '412', '414', '433', '434', '435'],
        'description': '心血管合并症（MI/Stroke/ASCVD）',
    },
    # 心力衰竭（ESC 2021 / ACC-AHA 2022 定义，独立于 ASCVD）
    # I50.x = HF (any); ICD-9 428.x = HF
    # 之所以独立：HF 对 SGLT2i 选择影响最强（DAPA-HF / EMPEROR），且 TZD 为绝对禁忌
    'heart_failure': {
        'icd10_prefixes': ['I50'],
        'icd9_codes': ['428'],
        'description': '心力衰竭（HF）',
    },
    # 糖尿病酮症酸中毒（DKA）——提示血糖控制极差或 T1DM
    'dka': {
        'icd10_prefixes': ['E10.1', 'E11.1'],
        'icd9_codes': ['250.1'],
        'description': '糖尿病酮症酸中毒（DKA）',
    },
    # 低血糖
    'hypoglycemia': {
        'icd10_prefixes': ['E10.64', 'E11.64', 'E16.0'],
        'icd9_codes': ['250.8', '251.0'],
        'description': '低血糖事件',
    },
}

# T1DM 高权重 ICD（含 DKA，强烈提示 T1DM）
T1DM_HIGH_WEIGHT_ICD10 = {'E10.1', 'E10.10', 'E10.11'}
T1DM_ICD10_PREFIX = 'E10'
T2DM_ICD10_PREFIX = 'E11'

# [FIX-CLN-8] BNP / NT-proBNP HF 确认阈值
# 来源：
#   BNP:       ACC/AHA 2022 HF Guideline; McDonagh TA, et al. Eur Heart J. 2021
#              > 100 pg/mL = 急性 HF 诊断参考值（rule-in 阈值）
#   NT-proBNP: Januzzi JL, et al. JACC. 2019;74(13):1701-1716
#              年龄分层 rule-in 阈值（急性 HF）：
#                < 50  岁：> 450  pg/mL
#                50-75 岁：> 900  pg/mL
#                > 75  岁：> 1800 pg/mL
# 使用 rule-in 而非 rule-out 阈值的理由：
#   目的是在 dm_complications 中设置 heart_failure=True 以激活 ClinicalPhenotyper Stage 1
#   (conf=0.80)，而非排除 HF。过低的 BNP（35-100 pg/mL）仍由 Stage 2 概率路径处理，
#   避免 Stage 1 被假阳性污染。
_BNP_HF_THRESHOLD = 100.0          # pg/mL，急性 HF rule-in
_NTPROBNP_AGE_THRESHOLDS = {        # (max_age_exclusive, threshold_pg_ml)
    50:  450.0,
    75:  900.0,
    999: 1800.0,
}


def _get_ntprobnp_threshold(age: Optional[float]) -> float:
    """根据年龄返回 NT-proBNP 的 HF rule-in 阈值（pg/mL）。"""
    if age is None:
        return _NTPROBNP_AGE_THRESHOLDS[75]   # 默认中间档
    for max_age, threshold in sorted(_NTPROBNP_AGE_THRESHOLDS.items()):
        if age < max_age:
            return threshold
    return _NTPROBNP_AGE_THRESHOLDS[999]


class ClinicalValidator:
    """
    临床数据质量校验器

    功能：
    1. validate_lab_values：检验值合理性校验
    2. classify_time_windows：时间窗口分类
    3. assess_data_quality：数据完整性评估（含HbA1c时机、UACR）
    4. validate_dm_type：糖尿病分型证据链（修复ICU偏差）
    5. extract_dm_complications：糖尿病慢性并发症结构化提取（新增）
    6. tag_diabetes_medications：标记糖尿病相关用药
    """

    # ==================== 检验值校验 ====================

    @staticmethod
    def validate_lab_values(data: CompletePatientData) -> CompletePatientData:
        """
        校验检验值的临床合理性

        - 不删除极端值（保留原始数据完整性）
        - 仅在 data_quality.extreme_value_details 中标记
        """
        if data.data_quality is None:
            data.data_quality = DataQualityFlags()

        extreme_details = []
        missing_units = 0

        for lab in data.all_lab_results:
            if not lab.unit or lab.unit.strip() == '':
                missing_units += 1

            label_lower = (lab.label or '').lower()
            for keyword, (lo, hi, expected_unit) in PLAUSIBLE_RANGES.items():
                if keyword in label_lower:
                    if lab.value < lo or lab.value > hi:
                        extreme_details.append({
                            'lab_label': lab.label,
                            'value': lab.value,
                            'unit': lab.unit,
                            'expected_range': f'{lo}-{hi} {expected_unit}',
                            'charttime': lab.charttime,
                            'reason': 'out_of_plausible_range',
                        })
                    break

        data.data_quality.total_labs = len(data.all_lab_results)
        data.data_quality.labs_with_extreme_values = len(extreme_details)
        data.data_quality.labs_missing_units = missing_units
        data.data_quality.extreme_value_details = extreme_details

        if extreme_details:
            logger.warning(
                f"  {data.subject_id}/{data.hadm_id}: "
                f"{len(extreme_details)} 个极端检验值被标记"
            )

        return data

    # ==================== 时间窗口分类 ====================

    @staticmethod
    def classify_lab_time_windows(
        data: CompletePatientData,
        baseline_hours: float = 6.0,
        predischarge_hours: float = 24.0,
    ) -> CompletePatientData:
        """对检验结果进行时间窗口分类"""
        if data.database == 'eicu':
            ClinicalValidator._classify_eicu_labs(data, baseline_hours)
        else:
            ClinicalValidator._classify_mimic_labs(data, baseline_hours, predischarge_hours)
        return data

    @staticmethod
    def _classify_mimic_labs(
        data: CompletePatientData,
        baseline_hours: float,
        predischarge_hours: float,
    ):
        """MIMIC 时间窗口分类"""
        admit_dt = None
        discharge_dt = None
        try:
            if data.admission_time:
                admit_dt = datetime.strptime(data.admission_time, '%Y-%m-%d %H:%M:%S')
            if data.discharge_time:
                discharge_dt = datetime.strptime(data.discharge_time, '%Y-%m-%d %H:%M:%S')
        except (ValueError, TypeError):
            pass

        for lab in data.all_lab_results:
            if lab.is_calculated:
                lab.time_window = LabTimeWindow.BASELINE
                continue

            try:
                lab_dt = datetime.strptime(lab.charttime, '%Y-%m-%d %H:%M:%S')
            except (ValueError, TypeError):
                lab.time_window = LabTimeWindow.DURING_STAY
                continue

            if admit_dt and (lab_dt - admit_dt) <= timedelta(hours=baseline_hours):
                lab.time_window = LabTimeWindow.BASELINE
            elif discharge_dt and (discharge_dt - lab_dt) <= timedelta(hours=predischarge_hours):
                lab.time_window = LabTimeWindow.PREDISCHARGE
            else:
                lab.time_window = LabTimeWindow.DURING_STAY

    @staticmethod
    def _classify_eicu_labs(data: CompletePatientData, baseline_hours: float):
        """eICU 时间窗口分类（基于 offset）"""
        baseline_min = baseline_hours * 60
        for lab in data.all_lab_results:
            if lab.is_calculated:
                lab.time_window = LabTimeWindow.BASELINE
                continue

            try:
                ct = lab.charttime
                if ct.startswith('admission+'):
                    parts = ct.replace('admission+', '').replace('h', ' ').replace('m', '')
                    h, m = parts.split()
                    offset_min = int(h) * 60 + int(m)
                    lab.time_window = (
                        LabTimeWindow.BASELINE
                        if offset_min <= baseline_min
                        else LabTimeWindow.DURING_STAY
                    )
                else:
                    lab.time_window = LabTimeWindow.DURING_STAY
            except (ValueError, AttributeError):
                lab.time_window = LabTimeWindow.DURING_STAY

    # ==================== 用药时间窗口分类 ====================

    @staticmethod
    def classify_medication_time_windows(
        data: CompletePatientData,
    ) -> CompletePatientData:
        """对结构化用药进行时间窗口分类，同时标记糖尿病相关用药"""
        for med in data.all_medications:
            drug_lower = (med.drug_name or '').lower()
            med.is_diabetes_related = any(
                kw in drug_lower for kw in DIABETES_DRUG_KEYWORDS
            )
        return data

    # ==================== 数据完整性评估 ====================

    @staticmethod
    def assess_data_quality(
        data: CompletePatientData,
        baseline_hours: float = 48.0,
    ) -> CompletePatientData:
        """
        评估数据完整性

        [FIX-CLN-6] HbA1c 时机分类：
            区分入院 48h 内（baseline）vs 住院中后期（during_stay）
            入院 48h 内的 HbA1c 更能代表患者基础控制水平

        [FIX-CLN-2] UACR 可用性：
            检查是否提取到尿白蛋白相关指标（用于 CKD G+A 分期）

        [FIX-ALG-7] 增加 glucose_sampling_frequency（次/天）
        """
        if data.data_quality is None:
            data.data_quality = DataQualityFlags()

        q = data.data_quality

        # HbA1c 可用性（含时机分类）
        hba1c_labs = [
            lab for lab in data.all_lab_results
            if 'a1c' in (lab.label or '').lower()
        ]
        q.hba1c_available = len(hba1c_labs) > 0

        # [FIX-CLN-6] HbA1c 时机分类
        if hba1c_labs and data.admission_time:
            try:
                admit_dt = datetime.strptime(data.admission_time, '%Y-%m-%d %H:%M:%S')
                window_end = admit_dt + timedelta(hours=baseline_hours)
                earliest_hba1c = hba1c_labs[0]  # 假设已排序（validate 调用前已分类）
                lab_dt = None
                for fmt in ('%Y-%m-%d %H:%M:%S', '%Y-%m-%d'):
                    try:
                        lab_dt = datetime.strptime(earliest_hba1c.charttime, fmt)
                        break
                    except (ValueError, TypeError):
                        continue
                if lab_dt and lab_dt <= window_end:
                    q.hba1c_timing = 'baseline'
                else:
                    q.hba1c_timing = 'during_stay'
            except (ValueError, TypeError):
                q.hba1c_timing = None
        elif hba1c_labs:
            q.hba1c_timing = 'unknown'

        # 肌酐可用性
        q.creatinine_available = any(
            'creatinine' in (lab.label or '').lower()
            and 'urine' not in (lab.label or '').lower()
            for lab in data.all_lab_results
        )

        # eGFR 是否已计算
        q.egfr_calculated = any(
            lab.is_calculated and 'egfr' in (lab.label or '').lower()
            for lab in data.all_lab_results
        )

        # BMI
        q.bmi_available = data.bmi is not None

        # 血糖测量次数
        glucose_labs = [
            lab for lab in data.all_lab_results
            if 'glucose' in (lab.label or '').lower()
            and 'urine' not in (lab.label or '').lower()
        ]
        q.glucose_measurement_count = len(glucose_labs)

        # [FIX-CLN-4] 血糖采样频率（次/天）
        if data.los_days and data.los_days > 0 and q.glucose_measurement_count > 0:
            q.glucose_sampling_frequency = round(
                q.glucose_measurement_count / data.los_days, 1
            )

        # [FIX-CLN-2] UACR 可用性
        q.acr_available = any(
            any(kw in (lab.label or '').lower()
                for kw in ['albumin/creatinine', 'uacr', 'acr', 'microalbumin'])
            for lab in data.all_lab_results
        )

        # UMLS 映射成功率
        total_mappable = sum(
            1 for lab in data.all_lab_results if lab.loinc_code
        ) + sum(
            1 for diag in data.all_diagnoses if diag.icd_code
        )
        mapped = sum(
            1 for lab in data.all_lab_results if lab.umls_cui
        ) + sum(
            1 for diag in data.all_diagnoses if diag.umls_cui
        )
        q.umls_mapping_rate = (mapped / total_mappable) if total_mappable > 0 else 0.0

        return data

    # ==================== [FIX-CLN-5/CLN-8] 糖尿病慢性并发症提取 ====================

    @staticmethod
    def extract_dm_complications(
        data: CompletePatientData,
        clinical_metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        基于 ICD 编码 + BNP/NT-proBNP 提取糖尿病慢性并发症

        [FIX-CLN-5] 这些信息是 MDT 药物推荐的核心依据：
        - 视网膜病变 → 避免强化降糖导致视网膜快速改善综合征
        - 肾病（CKD G3b+）→ 二甲双胍禁忌、SGLT-2 效益降低
        - 神经病变 → 优先 SGLT-2 或 GLP-1（有神经保护证据）
        - 心血管病 → SGLT-2（EMPA-REG, CREDENCE）或 GLP-1（LEADER, SUSTAIN-6）
        - DKA 病史 → 高度提示 T1DM 或胰岛素绝对缺乏

        [FIX-CLN-8] heart_failure 增加 BNP / NT-proBNP 补充判定：
          当 ICD 编码未记录 HF 但 BNP/NT-proBNP 已超过急性 HF rule-in 阈值时，
          仍将 heart_failure 置为 True，激活 ClinicalPhenotyper Stage 1 HF 路径
          （conf=0.80，高于单靠 Stage 2 BNP 概率路径的理论上限 0.85 且更稳定）。

          证据强度分层：
            'icd'      → ICD 确认，ClinicalPhenotyper Stage 1, conf=0.95
            'bnp'      → BNP > 100 pg/mL，Stage 1, conf=0.80
            'nt_probnp'→ NT-proBNP > 年龄阈值，Stage 1, conf=0.80
            None       → 无证据，heart_failure=False

          注意：BNP 35-100 pg/mL 及低于阈值的 NT-proBNP 仍由 Stage 2 处理，
          此处不设置 heart_failure=True，避免 Stage 1 被假阳性污染。

        Args:
            data:             患者数据（含 all_diagnoses, age）
            clinical_metrics: calculate_all_metrics() 的输出（含 bnp, nt_probnp）。
                              在 validate_all 中由调用方传入；若为 None 则跳过 BNP 判定
                              （backward compat：直接调用 extract_dm_complications 的旧代码）。

        Returns:
            {
                'retinopathy':        bool,
                'nephropathy':        bool,
                'neuropathy':         bool,
                'peripheral_vascular': bool,
                'cardiovascular':     bool,
                'heart_failure':      bool,
                'heart_failure_source': 'icd' | 'bnp' | 'nt_probnp' | None,
                'dka':                bool,
                'hypoglycemia':       bool,
                'complication_codes': [...],
                'complication_count': int,
            }
        """
        result = {comp: False for comp in DM_COMPLICATION_ICD}
        result['heart_failure_source'] = None   # [FIX-CLN-8] 新增字段
        complication_codes = []

        # ── Step 1: ICD 编码匹配（原有逻辑，完全不变）────────────────────────
        for diag in data.all_diagnoses:
            code = (diag.icd_code or '').strip()
            # [FIX-eICU-4] 跳过空 ICD 编码（eICU 无 ICD 的诊断）和旧版 "Unknown" 占位
            if not code or code == "Unknown":
                continue
            version = diag.icd_version

            for comp_name, comp_def in DM_COMPLICATION_ICD.items():
                matched = False
                if version == 10:
                    for prefix in comp_def['icd10_prefixes']:
                        if code.startswith(prefix):
                            matched = True
                            break
                elif version == 9:
                    for prefix in comp_def['icd9_codes']:
                        if code.startswith(prefix):
                            matched = True
                            break

                if matched and not result[comp_name]:
                    result[comp_name] = True
                    complication_codes.append({
                        'type': comp_name,
                        'icd_code': code,
                        'icd_version': version,
                        'description': comp_def['description'],
                        'seq_num': diag.seq_num,
                    })

        # 记录 ICD 确认来源
        if result['heart_failure']:
            result['heart_failure_source'] = 'icd'

        # ── Step 2: [FIX-CLN-8] BNP / NT-proBNP 补充判定 ─────────────────────
        # 仅在 ICD 未确认 HF 时才检查（ICD 确认优先，不降级）
        if not result['heart_failure'] and clinical_metrics is not None:
            bnp_val      = clinical_metrics.get('bnp')
            nt_probnp_val = clinical_metrics.get('nt_probnp')
            age           = data.age  # 用于 NT-proBNP 年龄分层

            # BNP 判断：> 100 pg/mL = 急性 HF rule-in（ACC/AHA 2022）
            if bnp_val is not None:
                try:
                    bnp_float = float(bnp_val)
                    if bnp_float > _BNP_HF_THRESHOLD:
                        result['heart_failure'] = True
                        result['heart_failure_source'] = 'bnp'
                        complication_codes.append({
                            'type': 'heart_failure',
                            'source': 'bnp',
                            'value': bnp_float,
                            'threshold': _BNP_HF_THRESHOLD,
                            'description': (
                                f'BNP={bnp_float:.0f} pg/mL > {_BNP_HF_THRESHOLD:.0f} pg/mL'
                                f'（ACC/AHA 急性 HF rule-in 阈值）'
                            ),
                        })
                        logger.debug(
                            f"  {data.subject_id}/{data.hadm_id}: "
                            f"heart_failure 由 BNP={bnp_float:.0f} pg/mL 确认（无 HF ICD）"
                        )
                except (ValueError, TypeError):
                    pass

            # NT-proBNP 判断（年龄分层阈值，Januzzi 2019）
            # 仅在 BNP 未触发时再判断（避免重复标记）
            if not result['heart_failure'] and nt_probnp_val is not None:
                try:
                    nt_float = float(nt_probnp_val)
                    nt_threshold = _get_ntprobnp_threshold(age)
                    if nt_float > nt_threshold:
                        result['heart_failure'] = True
                        result['heart_failure_source'] = 'nt_probnp'
                        complication_codes.append({
                            'type': 'heart_failure',
                            'source': 'nt_probnp',
                            'value': nt_float,
                            'threshold': nt_threshold,
                            'age_used': age,
                            'description': (
                                f'NT-proBNP={nt_float:.0f} pg/mL > {nt_threshold:.0f} pg/mL'
                                f'（年龄分层阈值，age={age}，Januzzi 2019）'
                            ),
                        })
                        logger.debug(
                            f"  {data.subject_id}/{data.hadm_id}: "
                            f"heart_failure 由 NT-proBNP={nt_float:.0f} pg/mL 确认"
                            f"（阈值={nt_threshold:.0f}，age={age}，无 HF ICD）"
                        )
                except (ValueError, TypeError):
                    pass

        # ── Step 3: 汇总 ──────────────────────────────────────────────────────
        result['complication_codes'] = complication_codes
        result['complication_count'] = sum(
            1 for k, v in result.items()
            if isinstance(v, bool) and v
        )

        return result

    # ==================== [FIX-CLN-3] 糖尿病分型验证 ====================

    @staticmethod
    def validate_dm_type(data: CompletePatientData) -> Dict[str, Any]:
        """
        糖尿病分型的多维证据链验证

        [FIX-CLN-3a] ICU 胰岛素权重修正：
            ICU 中 T2DM 患者因应激/禁食常全程使用胰岛素输注，
            "仅胰岛素"不能区分 T1/T2。
            新策略：仅在 LOS>5天 且 出院时仍只带胰岛素（无口服药）时，
            才将"仅胰岛素"计为 T1DM 支持证据，且权重降为 +0.5

        [FIX-CLN-3b] LADA 识别：
            年龄 30-60 + ICD 编码为 T1DM + 用药史含口服降糖药
            → 标记 dm_type='LADA_possible'（而非硬分为 T1DM）

        [FIX-CLN-3c] ICD 权重调整：
            ICD 编码仍是最强单一证据（+2），但 DKA 病史额外+1（强烈提示 T1DM）
            用药权重调整：口服降糖药 +1.5 > 胰岛素+口服 +1.0 > 仅胰岛素（条件性）+0.5

        证据维度：
        1. ICD 编码 (E10.x / 250.x1 = T1DM; E11.x / 250.x0 = T2DM)
        2. 用药类型（DKA 史加权后的胰岛素 vs 口服药）
        3. 年龄（<30 / 30-60 / >60）
        4. 并发症模式（DKA 史→T1DM, 心血管病史→T2DM更常见）

        Returns:
            {
                'dm_type': 'T1DM' / 'T2DM' / 'LADA_possible' / 'uncertain',
                'confidence': 'high' / 'medium' / 'low',
                'evidence': [...],
                'conflicts': [...],
                'lada_flag': bool,
                't1_score': float,
                't2_score': float,
            }
        """
        evidence = []
        conflicts = []
        t1_score = 0.0
        t2_score = 0.0

        # --- 证据1: ICD 编码（权重保持 +2，但 DKA 额外加分）---
        has_t1_icd = False
        has_t2_icd = False
        for diag in data.all_diagnoses:
            code = (diag.icd_code or '').strip()
            if diag.icd_version == 10:
                if code.startswith('E10'):
                    t1_score += 2
                    has_t1_icd = True
                    evidence.append(f'ICD-10 {code}: Type 1')
                    # [FIX-CLN-3c] DKA 额外加分
                    if code in T1DM_HIGH_WEIGHT_ICD10:
                        t1_score += 1
                        evidence.append(f'  → DKA 病史（强 T1DM 支持，+1）')
                elif code.startswith('E11'):
                    t2_score += 2
                    has_t2_icd = True
                    evidence.append(f'ICD-10 {code}: Type 2')
            elif diag.icd_version == 9:
                if code.startswith('250') and len(code) >= 5:
                    last_digit = code[-1]
                    if last_digit in ('1', '3'):
                        t1_score += 2
                        has_t1_icd = True
                        evidence.append(f'ICD-9 {code}: Type 1')
                    elif last_digit in ('0', '2'):
                        t2_score += 2
                        has_t2_icd = True
                        evidence.append(f'ICD-9 {code}: Type 2')

        # --- 证据2: 用药分析（[FIX-CLN-3a][FIX-CLN-3c] 权重修正）---
        all_drugs = [m.drug_name.lower() for m in data.all_medications if m.drug_name]
        if not all_drugs:
            all_drugs = [d.lower() for d in (data.prior_medications + data.discharge_medications) if d]

        has_insulin = any(
            'insulin' in d or d in ('lantus', 'levemir', 'tresiba', 'humalog', 'novolog', 'apidra')
            for d in all_drugs
        )
        has_oral = any(
            any(kw in d for kw in (
                'metformin', 'glipizide', 'glyburide', 'glimepiride',
                'sitagliptin', 'empagliflozin', 'dapagliflozin',
                'pioglitazone', 'liraglutide', 'semaglutide', 'dulaglutide'
            ))
            for d in all_drugs
        )

        # 检查出院用药（比住院用药更能代表长期方案）
        discharge_drugs = [d.lower() for d in data.discharge_medications if d]
        discharge_insulin_only = (
            has_insulin and not has_oral
            and discharge_drugs
            and any('insulin' in d for d in discharge_drugs)
            and not any(
                any(kw in d for kw in ('metformin', 'glipizide', 'glyburide'))
                for d in discharge_drugs
            )
        )

        if has_insulin and not has_oral:
            # [FIX-CLN-3a] ICU 修正：不再直接 +1 T1DM
            # 仅在 LOS>5 天且出院仍维持纯胰岛素方案时，才计为弱证据
            los = data.los_days or 0
            if los > 5 and discharge_insulin_only:
                t1_score += 0.5
                evidence.append(
                    f'出院纯胰岛素方案（LOS={los:.0f}d，弱 T1DM 支持，+0.5）'
                )
            else:
                evidence.append(
                    f'仅胰岛素（ICU 场景下不足以区分 T1/T2，不计分）'
                )

        if has_oral:
            # [FIX-CLN-3c] 口服药存在强烈提示 T2DM
            t2_score += 1.5
            evidence.append(f'口服降糖药存在（T2DM 支持，+1.5）')

        if has_insulin and has_oral:
            # T2DM 晚期/需要强化的患者常并用
            t2_score += 1.0
            evidence.append('胰岛素+口服药并用（提示进展期 T2DM，额外+1.0）')

        # --- 证据3: 年龄 ---
        if data.age is not None:
            if data.age < 30:
                t1_score += 1.5
                evidence.append(f'年龄 {data.age:.0f} < 30（支持 T1DM，+1.5）')
            elif 30 <= data.age <= 60:
                # LADA 高发年龄段，不直接加分，但影响 LADA 判断
                evidence.append(f'年龄 {data.age:.0f}（30-60，LADA 可能年龄段）')
            elif data.age > 60:
                t2_score += 0.5
                evidence.append(f'年龄 {data.age:.0f} > 60（弱 T2DM 支持，+0.5）')

        # --- 冲突检测 ---
        if t1_score > 0 and t2_score > 0:
            conflicts.append(
                f'混合证据：T1 score={t1_score:.1f}, T2 score={t2_score:.1f}'
            )

        # --- [FIX-CLN-3b] LADA 检测 ---
        # 条件：年龄 30-60 + T1DM ICD + 曾用口服药
        lada_flag = (
            data.age is not None
            and 30 <= data.age <= 60
            and has_t1_icd
            and has_oral
        )
        if lada_flag:
            evidence.append(
                'LADA 可能：年龄 30-60 + T1DM ICD + 口服药史'
                '（建议查 GAD 抗体 / C 肽）'
            )

        # --- 最终判定 ---
        if lada_flag:
            dm_type = 'LADA_possible'
            confidence = 'low'
        elif t1_score > t2_score and t1_score >= 2:
            dm_type = 'T1DM'
            confidence = 'high' if t1_score - t2_score >= 2 else 'medium'
        elif t2_score > t1_score and t2_score >= 2:
            dm_type = 'T2DM'
            confidence = 'high' if t2_score - t1_score >= 2 else 'medium'
        elif t1_score == 0 and t2_score == 0:
            dm_type = 'uncertain'
            confidence = 'low'
        else:
            dm_type = 'T2DM' if t2_score >= t1_score else 'T1DM'
            confidence = 'low'

        return {
            'dm_type': dm_type,
            'confidence': confidence,
            'evidence': evidence,
            'conflicts': conflicts,
            'lada_flag': lada_flag,
            't1_score': t1_score,
            't2_score': t2_score,
        }

    # ==================== [FIX-ALG-7] 一键全量校验（含缓存） ====================

    @classmethod
    def validate_all(
        cls,
        data: CompletePatientData,
        baseline_hours: float = 6.0,
    ) -> CompletePatientData:
        """
        一键执行所有校验步骤

        [FIX-ALG-7] 在 validate_all 结束时计算并缓存 DM 分型结果
        → 质量报告阶段直接读取，无需对所有患者重跑 validate_dm_type

        [FIX-CLN-8] 将 clinical_metrics 传入 extract_dm_complications，
        使其能用 BNP/NT-proBNP 补充判定 heart_failure。
        前提：run_extraction.py 中 calculate_all_metrics 在 validate_all 之前调用，
        因此调用本方法时 data.clinical_metrics 里已有 bnp 和 nt_probnp。

        顺序：
        1. 检验值合理性校验
        2. 时间窗口分类
        3. 用药分类
        4. 数据完整性评估（含HbA1c时机、血糖采样频率）
        5. DM 分型（缓存结果）
        6. 并发症提取（注入 clinical_metrics，[FIX-CLN-8] 传入 clinical_metrics 用于 BNP 判定）
        """
        data = cls.validate_lab_values(data)
        data = cls.classify_lab_time_windows(data, baseline_hours=baseline_hours)
        data = cls.classify_medication_time_windows(data)
        data = cls.assess_data_quality(data)

        # [FIX-ALG-7] 缓存 DM 分型
        dm_info = cls.validate_dm_type(data)
        if data.data_quality:
            data.data_quality.dm_type_cached = dm_info['dm_type']
            data.data_quality.dm_confidence_cached = dm_info['confidence']

        # [FIX-CLN-5 + FIX-CLN-8] 并发症提取，写入 clinical_metrics
        # 传入当前 clinical_metrics（含 bnp/nt_probnp）以激活 BNP 补充判定
        complications = cls.extract_dm_complications(
            data,
            clinical_metrics=data.clinical_metrics,  # [FIX-CLN-8] 新增
        )
        if data.clinical_metrics is None:
            data.clinical_metrics = {}
        data.clinical_metrics['dm_complications'] = complications
        data.clinical_metrics['dm_classification'] = dm_info

        return data
