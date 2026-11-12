#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
语义化患者数据提取器 - 消除预定义规则的局限性

核心思想：
1. 提取患者的**所有**检查和诊断，不预设过滤
2. 使用UMLS进行语义标准化
3. 由LLM基于临床知识进行智能归纳

作者：Yuxi
日期：2025-02-01
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field

# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== 数据结构 ====================

@dataclass
class LabResult:
    """单个实验室检查结果"""
    itemid: int
    label: str  # 从D_LABITEMS获取
    loinc_code: Optional[str]  # LOINC编码
    value: float
    unit: str
    charttime: str
    is_abnormal: bool  # 是否异常（基于参考范围）


@dataclass
class Diagnosis:
    """单个诊断"""
    icd_code: str
    icd_version: int  # 9 or 10
    description: str  # 从D_ICD_DIAGNOSES获取
    seq_num: int  # 诊断序号
    snomed_cui: Optional[str] = None  # SNOMED CT CUI (通过UMLS映射)


@dataclass
class VitalSign:
    """生命体征"""
    itemid: int
    label: str
    value: float
    unit: str
    charttime: str


@dataclass
class CompletePatientData:
    """完整的患者数据（无信息丢失）"""
    # 基本信息
    database: str
    subject_id: int
    hadm_id: int
    age: Optional[float]
    gender: str
    
    # ===== 完整的原始数据 =====
    all_lab_results: List[LabResult] = field(default_factory=list)
    all_diagnoses: List[Diagnosis] = field(default_factory=list)
    all_vitals: List[VitalSign] = field(default_factory=list)
    
    # 用药信息
    prior_medications: List[str] = field(default_factory=list)
    discharge_medications: List[str] = field(default_factory=list)
    
    # 元数据
    anchor_year_group: Optional[str] = None  # [FIX-ERA] MIMIC-IV 真实临床年份分组
    anchor_year: Optional[int] = None        # MIMIC-IV 移位后的 anchor_year
    admission_time: Optional[str] = None
    discharge_time: Optional[str] = None
    los_days: Optional[float] = None

    # [FIX-N1] eICUPatientExtractor 构造时传入这四个字段；
    # 旧版 dataclass 没有声明 → TypeError crash。
    # 声明为 Optional 并默认 None，对 MIMIC 提取器完全透明（它们不传这些字段）。
    ethnicity: Optional[str] = None
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    bmi: Optional[float] = None


@dataclass
class ClinicalSummary:
    """LLM生成的临床摘要（结构化）"""
    # 主要诊断
    primary_diagnosis: str
    dm_type: str  # Type 1 / Type 2 / Unknown
    
    # 关键检查（LLM识别的重要指标）
    key_labs: Dict[str, float]  # {'HbA1c': 8.5, 'eGFR': 45, ...}
    
    # 重要合并症（LLM识别的影响治疗的合并症）
    critical_comorbidities: List[str]  # ['CKD Stage 3', 'ASCVD', ...]
    
    # 禁忌证和注意事项
    contraindications: List[str]  # ['eGFR <30: avoid metformin', ...]
    
    # 临床推理
    clinical_reasoning: str  # LLM对患者情况的综合分析


# ==================== 语义化提取器 ====================

class SemanticPatientExtractor:
    """
    语义化患者数据提取器
    
    核心改进：
    1. 不预定义ITEMID/ICD，提取所有数据
    2. 使用D_LABITEMS/D_ICD_DIAGNOSES获取标准名称
    3. （可选）使用UMLS进行语义映射
    4. 由LLM进行智能归纳
    """
    
    def __init__(self, database: str, data_path: Path, batch_loader=None):
        self.database = database
        self.data_path = data_path
        
        # BatchDataLoader：外部传入则共享实例，否则自建一个（单患者模式也能用）
        from baseline.batch_data_loader import BatchDataLoader
        self.batch_loader = batch_loader if batch_loader is not None else BatchDataLoader()
        
        # 预加载字典表（用于获取标准名称）
        self._load_dictionaries()
        
        # 预加载患者数据表
        self._load_patient_tables()
        
        logger.info(f"✅ 语义化提取器初始化完成: {database}")
    
    def _load_dictionaries(self):
        """加载字典表（D_LABITEMS, D_ICD_DIAGNOSES等）"""
        logger.info("加载字典表...")
        
        if self.database == 'mimic3':
            # 实验室检查字典
            self.d_labitems = pd.read_csv(
                self.data_path / 'D_LABITEMS.csv.gz',
                usecols=['itemid', 'label', 'fluid', 'category', 'loinc_code']
            )
            self.d_labitems.columns = self.d_labitems.columns.str.lower()
            
            # ICD-9诊断字典
            self.d_icd_diagnoses = pd.read_csv(
                self.data_path / 'D_ICD_DIAGNOSES.csv.gz',
                usecols=['icd9_code', 'short_title', 'long_title']
            )
            self.d_icd_diagnoses.columns = self.d_icd_diagnoses.columns.str.lower()
            
            # CHARTEVENTS字典
            self.d_items = pd.read_csv(
                self.data_path / 'D_ITEMS.csv.gz',
                usecols=['itemid', 'label', 'category', 'unitname']
            )
            self.d_items.columns = self.d_items.columns.str.lower()
            
            logger.info(f"  - D_LABITEMS: {len(self.d_labitems)} 项")
            logger.info(f"  - D_ICD_DIAGNOSES: {len(self.d_icd_diagnoses)} 编码")
            logger.info(f"  - D_ITEMS: {len(self.d_items)} 项")
        
        elif self.database == 'mimic4':
            # MIMIC-IV结构类似，但文件名和字段略有不同
            self.d_labitems = pd.read_csv(
                self.data_path / 'hosp/d_labitems.csv.gz'
            )
            self.d_labitems.columns = self.d_labitems.columns.str.lower()
            
            self.d_icd_diagnoses = pd.read_csv(
                self.data_path / 'hosp/d_icd_diagnoses.csv.gz'
            )
            self.d_icd_diagnoses.columns = self.d_icd_diagnoses.columns.str.lower()
    
    def _load_patient_tables(self):
        """加载患者数据表"""
        logger.info("加载患者数据表...")
        
        if self.database == 'mimic3':
            self.admissions = pd.read_csv(
                self.data_path / 'ADMISSIONS.csv.gz',
                usecols=['subject_id', 'hadm_id', 'admittime', 'dischtime'],
                parse_dates=['admittime', 'dischtime']
            )
            self.admissions.columns = self.admissions.columns.str.lower()
            
            self.patients = pd.read_csv(
                self.data_path / 'PATIENTS.csv.gz',
                usecols=['subject_id', 'gender', 'dob'],
                parse_dates=['dob']
            )
            self.patients.columns = self.patients.columns.str.lower()
            
            self.diagnoses = pd.read_csv(
                self.data_path / 'DIAGNOSES_ICD.csv.gz',
                usecols=['subject_id', 'hadm_id', 'icd9_code', 'seq_num']
            )
            self.diagnoses.columns = self.diagnoses.columns.str.lower()
            
            self.prescriptions = pd.read_csv(
                self.data_path / 'PRESCRIPTIONS.csv.gz',
                usecols=['subject_id', 'hadm_id', 'drug', 'startdate', 'enddate'],
                parse_dates=['startdate', 'enddate']
            )
            self.prescriptions.columns = self.prescriptions.columns.str.lower()

            # --- O(1) 按 hadm_id 查询索引 ---
            # 用 dict-of-DataFrame 代替每次全表 boolean mask（O(n) → O(1)）
            self._admissions_by_hadm    = {k: v for k, v in self.admissions.groupby('hadm_id')}
            self._patients_by_subj      = {k: v for k, v in self.patients.groupby('subject_id')}
            self._diagnoses_by_hadm     = {k: v for k, v in self.diagnoses.groupby('hadm_id')}
            self._prescriptions_by_hadm = {k: v for k, v in self.prescriptions.groupby('hadm_id')}
            logger.info(
                f"  - 索引构建完成: {len(self._admissions_by_hadm)} 次住院 / "
                f"{len(self._patients_by_subj)} 个患者"
            )
    
    def preload_batch(self, hadm_ids: List[int], preload_vitals: bool = False):
        if self.database == 'eicu':
            logger.info("✅ eICU 数据已在初始化时装入内存，无需预加载。")
            return
        
        if self.database != 'mimic3':
            logger.warning(f"preload_batch 当前仅针对 mimic3，当前 database={self.database}")
            return

        if hadm_ids and isinstance(hadm_ids[0], tuple):
            target_set = set(hadm_id for _, hadm_id in hadm_ids)
        else:
            target_set = set(hadm_ids)

        labevents_path = self.data_path / 'LABEVENTS.csv.gz'
        chartevents_path = self.data_path / 'CHARTEVENTS.csv.gz'

        # LABEVENTS：核心数据，必须加载
        if labevents_path.exists():
            self.batch_loader.load_filtered(
                file_path=labevents_path,
                id_column='hadm_id',
                target_ids=target_set,
                usecols=['hadm_id', 'itemid', 'charttime', 'valuenum', 'valueuom'],
                parse_dates=['charttime'],
            )
        else:
            logger.warning(f"  LABEVENTS.csv.gz 不存在: {labevents_path}")

        # CHARTEVENTS：用 vital-sign itemid 白名单过滤，大幅减少扫描行数
        # 不用白名单则 CareVue subset 解压后数 GB，直接全量加载容易 OOM
        if preload_vitals and chartevents_path.exists():
            # 从已加载的 D_ITEMS 提取 vital-sign 相关 itemid 白名单
            vital_itemids: Set[int] = set()
            if hasattr(self, 'd_items') and not self.d_items.empty:
                vital_mask = self.d_items['category'].str.contains(
                    'Vital Sign|vital sign|Vital|vital|Routine Vital|Bedside Monitor',
                    case=False, na=False
                )
                vital_itemids = set(self.d_items.loc[vital_mask, 'itemid'].tolist())
                logger.info(f"  [CHARTEVENTS] vital-sign itemid 白名单: {len(vital_itemids)} 项")

            if vital_itemids:
                # 带 itemid 白名单的预加载（双重过滤：hadm_id ∩ itemid）
                self.batch_loader.load_filtered_with_item_filter(
                    file_path=chartevents_path,
                    id_column='hadm_id',
                    target_ids=target_set,
                    item_column='itemid',
                    target_items=vital_itemids,
                    usecols=['hadm_id', 'itemid', 'charttime', 'valuenum', 'valueuom'],
                    parse_dates=['charttime'],
                )
            else:
                # 白名单为空（D_ITEMS 未加载）→ 退化为全量扫描，不推荐但不崩溃
                logger.warning("  [CHARTEVENTS] 无 vital itemid 白名单，走全量扫描（内存风险）")
                self.batch_loader.load_filtered(
                    file_path=chartevents_path,
                    id_column='hadm_id',
                    target_ids=target_set,
                    usecols=['hadm_id', 'itemid', 'charttime', 'valuenum', 'valueuom'],
                    parse_dates=['charttime'],
                )
        else:
            logger.info("  [preload_vitals=False] 跳过 CHARTEVENTS 预加载（懒加载模式）")

        logger.info(
            f"✅ MIMIC-III预加载完成 | 缓存内存: {self.batch_loader.memory_usage_mb():.1f}MB | "
            f"统计: {self.batch_loader.get_stats()}"
        )
    
    def extract_complete_patient_data(
        self, 
        subject_id: int, 
        hadm_id: int,
        lab_window_hours: int = 48
    ) -> Optional[CompletePatientData]:
        """
        提取患者的**完整**数据（无信息丢失）
        
        关键改进：
        - 不预定义ITEMID/ICD
        - 提取所有检查和诊断
        - 保留原始信息的完整性
        - 自动计算eGFR（如果有肌酐数据）
        """
        try:
            # 1. 基本信息
            demographics = self._extract_demographics(subject_id, hadm_id)
            
            # 2. 所有实验室检查
            all_labs = self._extract_all_labs(hadm_id, lab_window_hours)
            
            # # 3. 计算eGFR（如果有肌酐值）
            # egfr_result = self._calculate_and_add_egfr(
            #     all_labs,
            #     age=demographics.get('age'),
            #     gender=demographics.get('gender')
            # )
            
            # if egfr_result:
            #     all_labs.append(egfr_result)
            #     logger.debug(f"  计算eGFR: {egfr_result.value} mL/min/1.73m²")
            
            # 4. 所有诊断
            all_diagnoses = self._extract_all_diagnoses(hadm_id)
            
            # 5. 所有生命体征
            all_vitals = self._extract_all_vitals(hadm_id)
            
            # 6. 用药
            prior_meds, discharge_meds = self._extract_medications(hadm_id)
            
            # 构建完整数据
            complete_data = CompletePatientData(
                database=self.database,
                subject_id=subject_id,
                hadm_id=hadm_id,
                age=demographics.get('age'),
                gender=demographics.get('gender', 'unknown'),
                all_lab_results=all_labs,
                all_diagnoses=all_diagnoses,
                all_vitals=all_vitals,
                prior_medications=prior_meds,
                discharge_medications=discharge_meds,
                # [FIX-ERA] anchor_year_group 由 MIMIC4PatientExtractor._extract_demographics()
                # 写入 demographics dict，此处透传进 CompletePatientData。
                # 对 MIMIC-III / eICU 路径，demographics 中无该键，get() 返回 None，无副作用。
                anchor_year_group=demographics.get('anchor_year_group'),
                anchor_year=demographics.get('anchor_year'),
                admission_time=demographics.get('admission_time'),
                discharge_time=demographics.get('discharge_time'),
                los_days=demographics.get('los_days')
            )
            
            logger.info(f"✅ 提取完成: {len(all_labs)}个检查, {len(all_diagnoses)}个诊断, {len(all_vitals)}个生命体征")
            
            return complete_data
            
        except Exception as e:
            logger.error(f"提取失败 ({subject_id}/{hadm_id}): {e}")
            return None
    
    def _extract_demographics(self, subject_id: int, hadm_id: int) -> Dict:
        """提取人口统计学特征"""
        # 优先走 O(1) 索引（批量模式）；回退到全表过滤（单患者/旧路径兼容）
        if hasattr(self, '_patients_by_subj'):
            patient = self._patients_by_subj.get(subject_id, pd.DataFrame())
        else:
            patient = self.patients[self.patients['subject_id'] == subject_id]
        if patient.empty:
            return {'age': None, 'gender': 'unknown'}

        if hasattr(self, '_admissions_by_hadm'):
            admission = self._admissions_by_hadm.get(hadm_id, pd.DataFrame())
        else:
            admission = self.admissions[self.admissions['hadm_id'] == hadm_id]
        if admission.empty:
            return {'age': None, 'gender': patient.iloc[0]['gender']}
        
        # 计算年龄
        dob = patient.iloc[0]['dob']
        admittime = admission.iloc[0]['admittime']
        
        age = None
        if pd.notna(dob) and pd.notna(admittime):
            try:
                age = (admittime.to_pydatetime() - dob.to_pydatetime()).days / 365.25
                # 合理性校验：去识别化移位不均匀时偶尔会算出荒谬值
                if age < 0:
                    age = None
                elif age > 150:
                    # [CLINICAL-FIX] MIMIC-III 高龄脱敏修复
                    # 大于 89 岁的患者会被计算为 ~300 岁。统一截断为 90.0 岁
                    age = 90.0
            except (OverflowError, ValueError, AttributeError):
                age = None
        
        return {
            'age': age,
            'gender': patient.iloc[0]['gender'],
            'admission_time': admittime.strftime('%Y-%m-%d %H:%M:%S'),
            'discharge_time': admission.iloc[0]['dischtime'].strftime('%Y-%m-%d %H:%M:%S'),
            'los_days': (admission.iloc[0]['dischtime'] - admittime).days,
        }
    
    def _extract_all_labs(
        self, 
        hadm_id: int, 
        window_hours: int = 48
    ) -> List[LabResult]:
        """
        提取患者的**所有**实验室检查（不预定义ITEMID）
        
        读取策略：
            - 如果已 preload_batch → 直接从缓存拿（O(1)）
            - 否则 → 逐块扫描文件（单患者模式 fallback）
        """
        if hasattr(self, '_admissions_by_hadm'):
            admission = self._admissions_by_hadm.get(hadm_id, pd.DataFrame())
        else:
            admission = self.admissions[self.admissions['hadm_id'] == hadm_id]
        if admission.empty:
            return []
        
        admit_time = admission.iloc[0]['admittime']
        cutoff_time = admit_time + timedelta(hours=window_hours)
        
        labevents_path = self.data_path / 'LABEVENTS.csv.gz'

        # --- 路径A：缓存命中 ---
        if self.batch_loader.has(labevents_path):
            labs_df = self.batch_loader.get(labevents_path, hadm_id)
            if labs_df is None:
                return []
            labs_df = labs_df[labs_df['charttime'] <= cutoff_time].copy()

        # --- 路径B：逐块扫描（单患者 fallback） ---
        else:
            collected = []
            for chunk in pd.read_csv(
                labevents_path,
                usecols=['hadm_id', 'itemid', 'charttime', 'valuenum', 'valueuom'],
                parse_dates=['charttime'],
                chunksize=500000,
                low_memory=False
            ):
                sub = chunk[
                    (chunk['hadm_id'] == hadm_id) &
                    (chunk['charttime'] <= cutoff_time)
                ]
                if not sub.empty:
                    collected.append(sub)
            if not collected:
                return []
            labs_df = pd.concat(collected, ignore_index=True)

        # 合并 D_LABITEMS 获取标准名称和 LOINC 编码（路径A/B之后统一一次）
        labs_df = labs_df.merge(
            self.d_labitems[['itemid', 'label', 'loinc_code']],
            on='itemid',
            how='left'
        )

        # 转换为 LabResult 对象
        lab_results = []
        for _, row in labs_df.iterrows():
            if pd.notna(row['valuenum']):
                lab_results.append(LabResult(
                    itemid=int(row['itemid']),
                    label=str(row.get('label') or f"ITEMID_{int(row['itemid'])}"),
                    loinc_code=row.get('loinc_code'),
                    value=float(row['valuenum']),
                    unit=str(row.get('valueuom') or ''),
                    charttime=row['charttime'].strftime('%Y-%m-%d %H:%M:%S'),
                    is_abnormal=False
                ))
        
        logger.debug(f"  提取了 {len(lab_results)} 个实验室检查结果")
        return lab_results
    
    def _extract_all_diagnoses(self, hadm_id: int) -> List[Diagnosis]:
        """
        提取患者的**所有**诊断（不预定义ICD）
        
        关键改进：
        - 提取所有ICD编码
        - 从D_ICD_DIAGNOSES获取标准描述
        - 保留完整诊断信息
        """
        if hasattr(self, '_diagnoses_by_hadm'):
            diags = self._diagnoses_by_hadm.get(hadm_id, pd.DataFrame())
        else:
            diags = self.diagnoses[self.diagnoses['hadm_id'] == hadm_id].copy()
        
        if diags.empty:
            return []
        
        # 合并D_ICD_DIAGNOSES获取标准描述
        diags_with_desc = diags.merge(
            self.d_icd_diagnoses[['icd9_code', 'long_title']],
            on='icd9_code',
            how='left'
        )
        
        # 转换为Diagnosis对象
        diagnoses = []
        for _, row in diags_with_desc.iterrows():
            diagnoses.append(Diagnosis(
                icd_code=row['icd9_code'],
                icd_version=9,
                description=row.get('long_title', f"ICD9: {row['icd9_code']}"),
                seq_num=int(row['seq_num'])
            ))
        
        # 按seq_num排序（主诊断在前）
        diagnoses.sort(key=lambda d: d.seq_num)
        
        logger.debug(f"  提取了 {len(diagnoses)} 个诊断")
        return diagnoses
    
    def _extract_all_vitals(self, hadm_id: int) -> List[VitalSign]:
        """
        提取患者的**所有**生命体征（不预定义ITEMID）
        
        读取策略与 _extract_all_labs 一致。
        时间窗口：入院后24小时。
        """
        if hasattr(self, '_admissions_by_hadm'):
            admission = self._admissions_by_hadm.get(hadm_id, pd.DataFrame())
        else:
            admission = self.admissions[self.admissions['hadm_id'] == hadm_id]
        if admission.empty:
            return []
        
        admit_time = admission.iloc[0]['admittime']
        cutoff_time = admit_time + timedelta(hours=24)
        
        chartevents_path = self.data_path / 'CHARTEVENTS.csv.gz'

        # --- 路径A：缓存命中 ---
        if self.batch_loader.has(chartevents_path):
            vitals_df = self.batch_loader.get(chartevents_path, hadm_id)
            if vitals_df is None:
                return []
            vitals_df = vitals_df[vitals_df['charttime'] <= cutoff_time].copy()

        # --- 路径B：逐块扫描（单患者 fallback） ---
        else:
            # [SAFETY-FIX] 加上并发锁，防止 preload_vitals=False 时导致硬盘 I/O 踩踏爆炸
            if not hasattr(self, '_chartevents_scan_lock'):
                import threading
                self._chartevents_scan_lock = threading.Lock()
                
            with self._chartevents_scan_lock:
                collected = []
                for chunk in pd.read_csv(
                    chartevents_path,
                    usecols=['hadm_id', 'itemid', 'charttime', 'valuenum', 'valueuom'],
                    parse_dates=['charttime'],
                    chunksize=1000000,
                    low_memory=False
                ):
                    sub = chunk[
                        (chunk['hadm_id'] == hadm_id) &
                        (chunk['charttime'] <= cutoff_time)
                    ]
                    if not sub.empty:
                        collected.append(sub)
                
                if not collected:
                    return []
                vitals_df = pd.concat(collected, ignore_index=True)

        # 合并 D_ITEMS 获取标准名称（路径A/B之后统一一次）
        vitals_df = vitals_df.merge(
            self.d_items[['itemid', 'label', 'unitname']],
            on='itemid',
            how='left'
        )

        vitals = []
        for _, row in vitals_df.iterrows():
            if pd.notna(row['valuenum']):
                vitals.append(VitalSign(
                    itemid=int(row['itemid']),
                    label=str(row.get('label') or f"ITEMID_{int(row['itemid'])}"),
                    value=float(row['valuenum']),
                    unit=str(row.get('valueuom') or row.get('unitname') or ''),
                    charttime=row['charttime'].strftime('%Y-%m-%d %H:%M:%S')
                ))
        
        logger.debug(f"  提取了 {len(vitals)} 个生命体征记录")
        return vitals
    
    def _extract_medications(self, hadm_id: int) -> Tuple[List[str], List[str]]:
        """提取入院和出院用药"""
        if hasattr(self, '_admissions_by_hadm'):
            admission = self._admissions_by_hadm.get(hadm_id, pd.DataFrame())
        else:
            admission = self.admissions[self.admissions['hadm_id'] == hadm_id]
        if admission.empty:
            return ([], [])

        admittime = admission.iloc[0]['admittime']
        dischtime = admission.iloc[0]['dischtime']

        if hasattr(self, '_prescriptions_by_hadm'):
            prescriptions = self._prescriptions_by_hadm.get(hadm_id, pd.DataFrame()).copy()
        else:
            prescriptions = self.prescriptions[
                self.prescriptions['hadm_id'] == hadm_id
            ].copy()
        
        if prescriptions.empty:
            return ([], [])
        
        # 入院用药（入院前后24小时）
        prescriptions['time_to_admit'] = (
            prescriptions['startdate'] - admittime
        ).dt.total_seconds() / 3600
        
        prior_meds = prescriptions[
            (prescriptions['time_to_admit'] >= -24) & 
            (prescriptions['time_to_admit'] <= 24)
        ]['drug'].dropna().unique().tolist()
        
        # 出院用药（出院前24小时）
        prescriptions['time_to_discharge'] = (
            dischtime - prescriptions['enddate']
        ).dt.total_seconds() / 3600
        
        discharge_meds = prescriptions[
            (prescriptions['time_to_discharge'] >= -24) & 
            (prescriptions['time_to_discharge'] <= 24)
        ]['drug'].dropna().unique().tolist()
        
        return (prior_meds, discharge_meds)
    
    def _calculate_and_add_egfr(
        self,
        lab_results: List[LabResult],
        age: Optional[float],
        gender: Optional[str]
    ) -> Optional[LabResult]:
        """
        计算eGFR并返回为LabResult
        
        使用CKD-EPI 2021公式
        
        Args:
            lab_results: 实验室检查结果列表
            age: 患者年龄
            gender: 患者性别
        
        Returns:
            eGFR的LabResult对象，如果无法计算则返回None
        """
        # 查找肌酐值
        creatinine = None
        for lab in lab_results:
            if 'creatinine' in lab.label.lower() and 'serum' in lab.label.lower():
                creatinine = lab.value
                break
        
        if not creatinine or not age or not gender:
            logger.debug("  无法计算eGFR: 缺少肌酐、年龄或性别")
            return None
        
        # CKD-EPI 2021公式
        try:
            # 性别系数
            if gender.upper() in ['F', 'FEMALE']:
                kappa = 0.7
                alpha = -0.241
                gender_factor = 1.012
            else:  # 'M', 'MALE'
                kappa = 0.9
                alpha = -0.302
                gender_factor = 1.0
            
            # 计算
            min_ratio = min(creatinine / kappa, 1.0)
            max_ratio = max(creatinine / kappa, 1.0)
            
            egfr = 142 * (min_ratio ** alpha) * (max_ratio ** -1.200) * (0.9938 ** age) * gender_factor
            egfr = round(egfr, 1)
            
            # 创建LabResult对象
            egfr_result = LabResult(
                itemid=999999,  # 虚拟ITEMID表示计算值
                label='eGFR (Calculated)',
                loinc_code='33914-3',  # eGFR的标准LOINC编码
                value=egfr,
                unit='mL/min/1.73m2',
                charttime='',  # 计算值没有时间戳
                is_abnormal=(egfr < 60)  # eGFR <60 表示CKD
            )
            
            return egfr_result
            
        except Exception as e:
            logger.debug(f"  eGFR计算失败: {e}")
            return None
    
    def generate_clinical_summary(
        self,
        complete_data: CompletePatientData,
        llm_client
    ) -> ClinicalSummary:
        """
        使用LLM生成临床摘要
        
        核心思想：
        - 将完整的检查和诊断信息提供给LLM
        - LLM基于临床知识进行智能归纳
        - 识别关键指标、重要合并症、禁忌证
        """
        # 构建提示词
        prompt = self._build_summary_prompt(complete_data)
        
        # 调用LLM（使用正确的接口）
        messages = [
            {"role": "system", "content": "你是一位资深的内分泌科医生。"},
            {"role": "user", "content": prompt}
        ]
        
        response = llm_client.chat_for_task("diagnosis", messages)
        response_text = llm_client.get_response_text(response)
        
        # 解析LLM返回的结构化摘要
        summary = self._parse_llm_summary(response_text)
        
        return summary
    
    def _build_summary_prompt(self, data: CompletePatientData) -> str:
        """构建LLM提示词"""
        
        # 格式化检查结果
        lab_text = "\n".join([
            f"- {lab.label}: {lab.value} {lab.unit} (LOINC: {lab.loinc_code or 'N/A'})"
            for lab in data.all_lab_results
        ])
        
        # 格式化诊断
        diag_text = "\n".join([
            f"{diag.seq_num}. {diag.description} ({diag.icd_code})"
            for diag in data.all_diagnoses
        ])
        
        # 格式化生命体征（只取最新值）
        vital_latest = {}
        for vital in data.all_vitals:
            if vital.label not in vital_latest:
                vital_latest[vital.label] = vital
        
        vital_text = "\n".join([
            f"- {label}: {vital.value} {vital.unit}"
            for label, vital in vital_latest.items()
        ])
        
        prompt = f"""你是一位资深的内分泌科医生。请基于以下患者的完整数据，生成一份临床摘要。

# 患者信息
- 年龄: {data.age:.1f}岁
- 性别: {data.gender}
- 住院天数: {data.los_days}天

# 所有实验室检查（入院48小时内）
{lab_text}

# 所有诊断（按序号排列）
{diag_text}

# 生命体征（最新值）
{vital_text}

# 入院用药
{', '.join(data.prior_medications) if data.prior_medications else '无'}

# 请提供以下结构化摘要（JSON格式）：

```json
{{
  "primary_diagnosis": "主要诊断（中文）",
  "dm_type": "Type 1 或 Type 2 或 Unknown",
  "key_labs": {{
    "HbA1c": 数值或null,
    "eGFR": 数值或null,
    "血肌酐": 数值或null,
    "空腹血糖": 数值或null,
    "LDL-C": 数值或null,
    "其他关键指标": 数值或null
  }},
  "critical_comorbidities": [
    "列出影响药物选择的重要合并症，如：",
    "CKD Stage 3 (eGFR 45)",
    "ASCVD",
    "心力衰竭",
    "肝功能异常",
    "..."
  ],
  "contraindications": [
    "基于检查和诊断，列出药物禁忌和注意事项，如：",
    "eGFR <30: 禁用二甲双胍",
    "肝功能异常: 慎用噻唑烷二酮类",
    "..."
  ],
  "clinical_reasoning": "综合分析患者情况的临床推理（2-3句话）"
}}
```

注意：
1. 只输出JSON，不要其他文字
2. key_labs只包含对糖尿病管理重要的指标
3. critical_comorbidities只包含影响药物选择的合并症
4. 基于ADA/KDIGO等指南判断禁忌证
"""
        return prompt
    
    def _parse_llm_summary(self, llm_response: str) -> ClinicalSummary:
        """解析LLM返回的JSON摘要"""
        import json
        
        # 提取JSON（去掉```json```标记）
        json_text = llm_response.strip()
        if json_text.startswith('```json'):
            json_text = json_text.split('```json')[1].split('```')[0].strip()
        elif json_text.startswith('```'):
            json_text = json_text.split('```')[1].split('```')[0].strip()
        
        # 解析JSON
        data = json.loads(json_text)
        
        # 转换为ClinicalSummary对象
        summary = ClinicalSummary(
            primary_diagnosis=data['primary_diagnosis'],
            dm_type=data['dm_type'],
            key_labs={k: v for k, v in data['key_labs'].items() if v is not None},
            critical_comorbidities=data['critical_comorbidities'],
            contraindications=data['contraindications'],
            clinical_reasoning=data['clinical_reasoning']
        )
        
        return summary


# ==================== 使用示例 ====================

if __name__ == '__main__':
    from pathlib import Path
    
    # 初始化提取器
    extractor = SemanticPatientExtractor(
        database='mimic3',
        data_path=Path('ori_data_zip/mimic-iii-clinical-database-carevue-subset-1.4')
    )
    
    # 提取完整数据
    complete_data = extractor.extract_complete_patient_data(
        subject_id=10006,
        hadm_id=142345
    )
    
    if complete_data:
        print(f"\n✅ 成功提取患者数据:")
        print(f"  - 实验室检查: {len(complete_data.all_lab_results)} 项")
        print(f"  - 诊断: {len(complete_data.all_diagnoses)} 个")
        print(f"  - 生命体征: {len(complete_data.all_vitals)} 条")
        
        print(f"\n前5个检查结果:")
        for lab in complete_data.all_lab_results[:5]:
            print(f"  - {lab.label}: {lab.value} {lab.unit}")
        
        print(f"\n所有诊断:")
        for diag in complete_data.all_diagnoses:
            print(f"  {diag.seq_num}. {diag.description}")
        
        # 如果有LLM，生成临床摘要
        # from agents.llm_client import DeepseekClient
        # llm = DeepseekClient(...)
        # summary = extractor.generate_clinical_summary(complete_data, llm)
        # print(f"\n临床摘要:\n{summary.clinical_reasoning}")
