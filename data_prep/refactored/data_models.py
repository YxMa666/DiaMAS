#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Data Models - 增强数据模型

核心改进：
1. LabResult/Diagnosis/VitalSign 直接声明 UMLS 属性（不再动态 setattr）
2. Medication 从纯字符串升级为结构体（保留剂量、途径、频率）
3. CompletePatientData 增加 BMI、过敏史、用药时间窗口分离
4. 所有 ID 字段统一为 Union[int, str] 以兼容 eICU
5. 增加数据质量元信息（data_quality_flags）

[FIX-v3.1] DataQualityFlags 新增字段：
  - hba1c_timing: HbA1c 测量时机分类（baseline/during_stay）
  - acr_available: 是否有尿白蛋白/肌酐比
  - baseline_creatinine_source: 基准肌酐来源说明
  - glucose_sampling_frequency: 血糖采样频率（次/天）
  - aki_suspected: 是否疑似 AKI（入院肌酐 vs 住院最低肌酐差异 ≥ 50%）

作者：Yuxi
日期：2025-02
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum


# ==================== 枚举类型 ====================

class LabTimeWindow(str, Enum):
    """检验时间窗口分类"""
    BASELINE = "baseline"          # 入院首次 / 入院前24h内
    DURING_STAY = "during_stay"    # 住院期间动态监测
    PREDISCHARGE = "predischarge"  # 出院前24h


class MedTimeWindow(str, Enum):
    """用药时间窗口分类"""
    PREADMISSION = "preadmission"    # 入院前长期用药
    ADMISSION_ORDER = "admission_order"  # 入院后初始医嘱
    DURING_STAY = "during_stay"      # 住院期间用药
    DISCHARGE = "discharge"          # 出院带药


# ==================== 基础数据结构 ====================

@dataclass
class LabResult:
    """
    单个实验室检查结果

    改进：
    - UMLS/SNOMED 属性直接声明，不再依赖动态 setattr
    - 增加 time_window 分类
    - 增加 reference_range 和 specimen_type
    """
    itemid: Union[int, str]
    label: str
    loinc_code: Optional[str]
    value: float
    unit: str
    charttime: str
    is_abnormal: bool

    # UMLS 标准化属性（直接声明，默认 None）
    umls_cui: Optional[str] = None
    snomed_code: Optional[str] = None
    snomed_name: Optional[str] = None

    # 增强属性
    time_window: Optional[str] = None          # baseline / during_stay / predischarge
    reference_low: Optional[float] = None      # 参考范围下限
    reference_high: Optional[float] = None     # 参考范围上限
    specimen_type: Optional[str] = None        # 标本类型（血清/尿/全血…）
    is_calculated: bool = False                # 是否为计算值（如 eGFR）


@dataclass
class Diagnosis:
    """
    单个诊断

    改进：
    - UMLS/SNOMED 属性直接声明
    - 增加 is_primary 标记
    """
    icd_code: str
    icd_version: int            # 9 or 10
    description: str
    seq_num: int

    # UMLS 标准化
    umls_cui: Optional[str] = None
    snomed_code: Optional[str] = None
    snomed_name: Optional[str] = None

    # 增强属性
    is_primary: bool = False    # 是否主诊断（seq_num == 1）


@dataclass
class VitalSign:
    """
    生命体征

    改进：
    - 增加 time_window 分类
    """
    itemid: Union[int, str]
    label: str
    value: float
    unit: str
    charttime: str

    time_window: Optional[str] = None


@dataclass
class Medication:
    """
    结构化用药信息（从纯字符串升级）

    核心改进：
    - 保留剂量、途径、频率等关键信息
    - 标注时间窗口（preadmission / admission / discharge）
    - 支持药物标准化名称
    """
    drug_name: str                              # 原始药物名称
    drug_name_standard: Optional[str] = None    # 标准化名称（RxNorm等）
    dosage: Optional[str] = None                # 剂量（如 "500mg"）
    route: Optional[str] = None                 # 给药途径（PO/IV/SC…）
    frequency: Optional[str] = None             # 频率（BID/QD/PRN…）
    start_time: Optional[str] = None            # 开始时间
    stop_time: Optional[str] = None             # 结束时间
    time_window: Optional[str] = None           # preadmission / admission / discharge
    is_diabetes_related: bool = False           # 是否糖尿病相关用药


@dataclass
class DataQualityFlags:
    """
    数据质量标记

    记录提取过程中发现的质量问题，用于下游分析时过滤或加权

    [FIX-v3.1] 新增字段说明：
    - hba1c_timing: HbA1c 测量时机，'baseline'=入院48h内，'during_stay'=住院中后期
      临床意义：入院48h内的 HbA1c 更能代表患者长期控制水平，住院期间因急性应激干扰更大
    - acr_available: 是否提取到尿白蛋白/肌酐比（UACR）
      临床意义：KDIGO 2012 CKD 完整分期需要 eGFR+ACR 双维度
    - baseline_creatinine_source: 基准肌酐来源
      'admission_earliest'=入院最早一条，'first_24h_min'=入院24h内最低值
      临床意义：ICU 患者入院时常有 AKI，最低肌酐更接近真实基线 GFR
    - glucose_sampling_frequency: 血糖采样频率（次/天），用于 TIR 可靠性加权
      临床意义：CGM 的 TIR 与离散指尖血糖的 TIR 含义不同，采样越稀疏越不可靠
    - aki_suspected: 是否疑似 AKI（入院肌酐 ÷ 住院最低肌酐 ≥ 1.5，KDIGO AKI 标准）
    - dm_type_cached: 缓存的 DM 分型结果，避免质量报告阶段重复计算
    """
    total_labs: int = 0
    labs_with_extreme_values: int = 0           # 超出临床合理范围的检验值
    labs_missing_units: int = 0                 # 缺失单位的检验项
    hba1c_available: bool = False               # HbA1c 是否可用
    creatinine_available: bool = False          # 肌酐是否可用
    egfr_calculated: bool = False               # eGFR 是否已计算
    bmi_available: bool = False                 # BMI 是否可用
    glucose_measurement_count: int = 0          # 血糖测量次数
    umls_mapping_rate: float = 0.0              # UMLS 映射成功率
    extreme_value_details: List[Dict] = field(default_factory=list)  # 极端值详情

    # [FIX-v3.1] 新增：HbA1c 时机
    hba1c_timing: Optional[str] = None          # 'baseline' / 'during_stay' / None

    # [FIX-v3.1] 新增：蛋白尿
    acr_available: bool = False                 # 是否有 UACR（尿白蛋白/肌酐比）

    # [FIX-v3.1] 新增：基准肌酐来源
    baseline_creatinine_source: Optional[str] = None  # 'admission_earliest' / 'first_24h_min'

    # [FIX-v3.1] 新增：血糖采样频率
    glucose_sampling_frequency: Optional[float] = None  # 次/天（基于 LOS 计算）

    # [FIX-v3.1] 新增：AKI 疑似标记
    aki_suspected: bool = False                 # 入院肌酐/最低肌酐 ≥ 1.5

    # [FIX-v3.1] 新增：DM 分型缓存（避免质量报告重复计算）
    dm_type_cached: Optional[str] = None        # 'T1DM' / 'T2DM' / 'LADA_possible' / 'uncertain'
    dm_confidence_cached: Optional[str] = None  # 'high' / 'medium' / 'low'


# ==================== 完整患者数据 ====================

@dataclass
class CompletePatientData:
    """
    完整的患者数据（增强版）

    核心改进：
    1. ID 统一为 Union[int, str] 以兼容 eICU 的字符串 uniquepid
    2. 用药从 List[str] 升级为 List[Medication]（保留剂量信息）
    3. 增加 BMI、ethnicity 等影响药物选择的关键字段
    4. 增加 data_quality 质量元信息
    5. 保留旧接口的兼容属性（prior_medications_names 等）
    """
    # ===== 基本信息 =====
    database: str
    subject_id: Union[int, str]
    hadm_id: Union[int, str]
    age: Optional[float]
    gender: str

    # ===== 增强的人口学信息 =====
    ethnicity: Optional[str] = None
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    bmi: Optional[float] = None

    # ===== 完整的原始数据 =====
    all_lab_results: List[LabResult] = field(default_factory=list)
    all_diagnoses: List[Diagnosis] = field(default_factory=list)
    all_vitals: List[VitalSign] = field(default_factory=list)

    # ===== 结构化用药（新） =====
    all_medications: List[Medication] = field(default_factory=list)

    # ===== 旧接口兼容：纯药名列表 =====
    prior_medications: List[str] = field(default_factory=list)
    discharge_medications: List[str] = field(default_factory=list)

    # ===== 元数据 =====
    admission_time: Optional[str] = None
    discharge_time: Optional[str] = None
    los_days: Optional[float] = None
    # [FIX-ERA] MIMIC-IV 真实临床年份分组（patients.csv.gz anchor_year_group 字段）
    # admittime 经去标识化偏移约+100年，不可用于 Era 分类。
    # 该字段存储未偏移的真实年份分组，供 classify_era() 正确区分 Era-1/2/3。
    # 格式示例: "2008 - 2010" / "2014 - 2016" / "2020 - 2022"
    # 对 MIMIC-III / eICU 为 None（不需要，Era 由 database 字段直接路由）
    anchor_year_group: Optional[str] = None
    # MIMIC-IV 移位后的 anchor_year（与 admission_time 同一移位），用于按入院推算真实年份
    anchor_year: Optional[int] = None

    # ===== 临床计算指标 =====
    clinical_metrics: Optional[Dict[str, Any]] = None

    # ===== 数据质量 =====
    data_quality: Optional[DataQualityFlags] = None

    # ==================== 便捷属性 ====================

    @property
    def prior_medication_names(self) -> List[str]:
        """入院前用药名称列表（兼容旧接口）"""
        if self.prior_medications:
            return self.prior_medications
        return [m.drug_name for m in self.all_medications
                if m.time_window == MedTimeWindow.PREADMISSION]

    @property
    def discharge_medication_names(self) -> List[str]:
        """出院用药名称列表（兼容旧接口）"""
        if self.discharge_medications:
            return self.discharge_medications
        return [m.drug_name for m in self.all_medications
                if m.time_window == MedTimeWindow.DISCHARGE]

    @property
    def baseline_labs(self) -> List[LabResult]:
        """基线检验（入院首次）"""
        return [l for l in self.all_lab_results
                if l.time_window == LabTimeWindow.BASELINE]

    @property
    def diabetes_medications(self) -> List[Medication]:
        """糖尿病相关用药"""
        return [m for m in self.all_medications if m.is_diabetes_related]


# ==================== 序列化 / 反序列化 ====================

def patient_data_to_dict(data: CompletePatientData) -> Dict:
    """
    将 CompletePatientData 序列化为字典（用于 JSON 缓存）

    改进：不再依赖 hasattr 判断，所有属性都在 dataclass 中声明
    """
    labs_list = []
    for lab in data.all_lab_results:
        labs_list.append({
            'itemid': lab.itemid,
            'label': lab.label,
            'loinc_code': lab.loinc_code,
            'value': lab.value,
            'unit': lab.unit,
            'charttime': lab.charttime,
            'is_abnormal': lab.is_abnormal,
            'umls_cui': lab.umls_cui,
            'snomed_code': lab.snomed_code,
            'snomed_name': lab.snomed_name,
            'time_window': lab.time_window,
            'reference_low': lab.reference_low,
            'reference_high': lab.reference_high,
            'specimen_type': lab.specimen_type,
            'is_calculated': lab.is_calculated,
        })

    diag_list = []
    for diag in data.all_diagnoses:
        diag_list.append({
            'icd_code': diag.icd_code,
            'icd_version': diag.icd_version,
            'description': diag.description,
            'seq_num': diag.seq_num,
            'umls_cui': diag.umls_cui,
            'snomed_code': diag.snomed_code,
            'snomed_name': diag.snomed_name,
            'is_primary': diag.is_primary,
        })

    vitals_list = []
    for v in data.all_vitals:
        vitals_list.append({
            'itemid': v.itemid,
            'label': v.label,
            'value': v.value,
            'unit': v.unit,
            'charttime': v.charttime,
            'time_window': v.time_window,
        })

    meds_list = []
    for m in data.all_medications:
        meds_list.append({
            'drug_name': m.drug_name,
            'drug_name_standard': m.drug_name_standard,
            'dosage': m.dosage,
            'route': m.route,
            'frequency': m.frequency,
            'start_time': m.start_time,
            'stop_time': m.stop_time,
            'time_window': m.time_window,
            'is_diabetes_related': m.is_diabetes_related,
        })

    quality_dict = None
    if data.data_quality:
        q = data.data_quality
        quality_dict = {
            'total_labs': q.total_labs,
            'labs_with_extreme_values': q.labs_with_extreme_values,
            'labs_missing_units': q.labs_missing_units,
            'hba1c_available': q.hba1c_available,
            'creatinine_available': q.creatinine_available,
            'egfr_calculated': q.egfr_calculated,
            'bmi_available': q.bmi_available,
            'glucose_measurement_count': q.glucose_measurement_count,
            'umls_mapping_rate': q.umls_mapping_rate,
            'extreme_value_details': q.extreme_value_details,
            # [FIX-v3.1] 新增字段序列化
            'hba1c_timing': q.hba1c_timing,
            'acr_available': q.acr_available,
            'baseline_creatinine_source': q.baseline_creatinine_source,
            'glucose_sampling_frequency': q.glucose_sampling_frequency,
            'aki_suspected': q.aki_suspected,
            'dm_type_cached': q.dm_type_cached,
            'dm_confidence_cached': q.dm_confidence_cached,
        }

    return {
        '_format': 'json_complete_v3.1',
        'database': data.database,
        'subject_id': data.subject_id,
        'hadm_id': data.hadm_id,
        'age': data.age,
        'gender': data.gender,
        'ethnicity': data.ethnicity,
        'height_cm': data.height_cm,
        'weight_kg': data.weight_kg,
        'bmi': data.bmi,
        'all_lab_results': labs_list,
        'all_diagnoses': diag_list,
        'all_vitals': vitals_list,
        'all_medications': meds_list,
        'prior_medications': data.prior_medications,
        'discharge_medications': data.discharge_medications,
        'admission_time': data.admission_time,
        'discharge_time': data.discharge_time,
        'los_days': data.los_days,
        'anchor_year_group': data.anchor_year_group,
        'anchor_year': data.anchor_year,
        'clinical_metrics': data.clinical_metrics,
        'data_quality': quality_dict,
    }


def patient_data_from_dict(d: Dict) -> CompletePatientData:
    """
    从字典恢复 CompletePatientData

    改进：类型安全，不再依赖动态属性；向后兼容 v3.0 格式
    """
    lab_results = []
    for ld in d.get('all_lab_results', []):
        lab_results.append(LabResult(
            itemid=ld['itemid'],
            label=ld['label'],
            loinc_code=ld.get('loinc_code'),
            value=ld['value'],
            unit=ld['unit'],
            charttime=ld['charttime'],
            is_abnormal=ld.get('is_abnormal', False),
            umls_cui=ld.get('umls_cui'),
            snomed_code=ld.get('snomed_code'),
            snomed_name=ld.get('snomed_name'),
            time_window=ld.get('time_window'),
            reference_low=ld.get('reference_low'),
            reference_high=ld.get('reference_high'),
            specimen_type=ld.get('specimen_type'),
            is_calculated=ld.get('is_calculated', False),
        ))

    diagnoses = []
    for dd in d.get('all_diagnoses', []):
        diagnoses.append(Diagnosis(
            icd_code=dd['icd_code'],
            icd_version=dd.get('icd_version', 9),
            description=dd['description'],
            seq_num=dd.get('seq_num', 0),
            umls_cui=dd.get('umls_cui'),
            snomed_code=dd.get('snomed_code'),
            snomed_name=dd.get('snomed_name'),
            is_primary=dd.get('is_primary', False),
        ))

    vitals = []
    for vd in d.get('all_vitals', []):
        vitals.append(VitalSign(
            itemid=vd['itemid'],
            label=vd['label'],
            value=vd['value'],
            unit=vd['unit'],
            charttime=vd['charttime'],
            time_window=vd.get('time_window'),
        ))

    medications = []
    for md in d.get('all_medications', []):
        medications.append(Medication(
            drug_name=md['drug_name'],
            drug_name_standard=md.get('drug_name_standard'),
            dosage=md.get('dosage'),
            route=md.get('route'),
            frequency=md.get('frequency'),
            start_time=md.get('start_time'),
            stop_time=md.get('stop_time'),
            time_window=md.get('time_window'),
            is_diabetes_related=md.get('is_diabetes_related', False),
        ))

    quality = None
    qd = d.get('data_quality')
    if qd:
        quality = DataQualityFlags(
            total_labs=qd.get('total_labs', 0),
            labs_with_extreme_values=qd.get('labs_with_extreme_values', 0),
            labs_missing_units=qd.get('labs_missing_units', 0),
            hba1c_available=qd.get('hba1c_available', False),
            creatinine_available=qd.get('creatinine_available', False),
            egfr_calculated=qd.get('egfr_calculated', False),
            bmi_available=qd.get('bmi_available', False),
            glucose_measurement_count=qd.get('glucose_measurement_count', 0),
            umls_mapping_rate=qd.get('umls_mapping_rate', 0.0),
            extreme_value_details=qd.get('extreme_value_details', []),
            # [FIX-v3.1] 向后兼容：旧格式缓存中无此字段时用 None/False
            hba1c_timing=qd.get('hba1c_timing'),
            acr_available=qd.get('acr_available', False),
            baseline_creatinine_source=qd.get('baseline_creatinine_source'),
            glucose_sampling_frequency=qd.get('glucose_sampling_frequency'),
            aki_suspected=qd.get('aki_suspected', False),
            dm_type_cached=qd.get('dm_type_cached'),
            dm_confidence_cached=qd.get('dm_confidence_cached'),
        )

    return CompletePatientData(
        database=d['database'],
        subject_id=d['subject_id'],
        hadm_id=d['hadm_id'],
        age=d.get('age'),
        gender=d.get('gender', 'unknown'),
        ethnicity=d.get('ethnicity'),
        height_cm=d.get('height_cm'),
        weight_kg=d.get('weight_kg'),
        bmi=d.get('bmi'),
        all_lab_results=lab_results,
        all_diagnoses=diagnoses,
        all_vitals=vitals,
        all_medications=medications,
        prior_medications=d.get('prior_medications', []),
        discharge_medications=d.get('discharge_medications', []),
        admission_time=d.get('admission_time'),
        discharge_time=d.get('discharge_time'),
        los_days=d.get('los_days'),
        anchor_year_group=d.get('anchor_year_group'),
        anchor_year=d.get('anchor_year'),
        clinical_metrics=d.get('clinical_metrics'),
        data_quality=quality,
    )
