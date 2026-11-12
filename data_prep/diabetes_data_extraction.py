#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
糖尿病患者数据提取与子集构建脚本 (修复版 v2.0)
支持：MIMIC-III, MIMIC-IV, eICU

修复记录 (v1.0 → v2.0):
  [FIX-1] MIMIC-IV ICD-9 遗漏（最重要）：
    原代码 _load_diagnoses() 只保留 icd_version==10，丢弃了所有 ICD-9 记录。
    MIMIC-IV 中 2008-2013 年（Era-1）的住院大量使用 ICD-9 编码（250.xx），
    导致 Era-1 的 T2DM 患者系统性低估。
    修复：新增 filter_by_icd9()，合并 ICD-9（250.xx）和 ICD-10（E10/E11）两路结果，
    DM 类型判定优先级：ICD-10 > ICD-9 > medication_only。

  [FIX-2] evidence_count 语义错误：
    原代码统计的是"行数"（同一来源类型的多行处方会重复计数），
    导致只有 medication 来源（无 ICD 确认）的患者也能通过 min_evidence=2 筛选，
    混入大量应激性高血糖（非真性糖尿病）患者。
    修复：改为统计"独立来源类型数"（nunique of source categories），
    min_evidence=2 现在要求至少两种独立来源（如 ICD + medication）。

  [FIX-3] MIMIC-IV anchor_year_group 缺失：
    _load_patients() 未读取 anchor_year_group 字段，
    导致该字段无法传递到下游 classify_era()。
    修复：在 usecols 中补充 'anchor_year_group'，并写入 demographics。

  [FIX-4] eICU hospitaldischargeyear 缺失：
    _load_patient() 未读取 hospitaldischargeyear，
    导致 eICU 患者的精细化 Era 分类依赖 fallback 而非真实年份。
    修复：在 usecols 中补充 'hospitaldischargeyear'，并写入 demographics。

  [FIX-5] eICU dm_type 丢失：
    原代码 icd_df 在 combined merge 后 dm_type 列可能因键不匹配而全部 NaN。
    修复：在 filter_by_icd9() 阶段就在 uniquepid 粒度建立 dm_type_map，
    与 combined 的 uniquepid 直接对齐。

作者：Yuxi
版本：v2.0（2025 修复版）
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional, Set
import logging
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from agents.config import DIABETES_MEDICATIONS

# ── 日志配置 ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('loggings/diabetes_extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 一、全局配置
# ═══════════════════════════════════════════════════════════════════════════════

class Config:
    """全局配置"""

    # ICD-9 编码（MIMIC-III、eICU，以及 MIMIC-IV 早期住院）
    ICD9_CODES = {
        'primary': ['250'],
        'T1DM': [
            '250.01', '250.03', '250.11', '250.13', '250.21', '250.23',
            '250.31', '250.33', '250.41', '250.43', '250.51', '250.53',
            '250.61', '250.63', '250.71', '250.73', '250.81', '250.83',
            '250.91', '250.93'
        ],
        'T2DM': [
            '250.00', '250.02', '250.10', '250.12', '250.20', '250.22',
            '250.30', '250.32', '250.40', '250.42', '250.50', '250.52',
            '250.60', '250.62', '250.70', '250.72', '250.80', '250.82',
            '250.90', '250.92'
        ],
    }

    # ICD-10 编码（MIMIC-IV 主体）
    ICD10_CODES = {
        'T1DM': ['E10'],
        'T2DM': ['E11'],
        # E08=drug/chem induced, E09=drug/chem induced, E13=other specified
        # 这三类不属于原发性 T1DM/T2DM，排除以避免混入继发性糖尿病
        'exclude': ['E08', 'E09', 'E13'],
    }

    DIABETES_MEDICATIONS = DIABETES_MEDICATIONS

    # 并发症 ICD 编码
    COMPLICATION_ICDS = {
        'CKD':           {'icd9': ['585', '586', '403', '404'],        'icd10': ['N18', 'N19', 'I12', 'I13']},
        'heart_failure': {'icd9': ['428'],                             'icd10': ['I50']},
        'retinopathy':   {'icd9': ['362.0', '250.5'],                  'icd10': ['E10.3', 'E11.3', 'H36']},
        'neuropathy':    {'icd9': ['357.2', '250.6'],                  'icd10': ['E10.4', 'E11.4', 'G63']},
        'CAD':           {'icd9': ['414'],                             'icd10': ['I25']},
        'stroke':        {'icd9': ['430', '431', '432', '433', '434', '436'], 'icd10': ['I60', 'I61', 'I62', 'I63', 'I64']},
        'PVD':           {'icd9': ['443', '444'],                      'icd10': ['I73', 'I74']},
    }

    # [FIX-2] 最少独立来源类型数（不是行数）
    # 2 = 至少两种独立来源（如 ICD + medication，或 ICD + lab），
    # 防止仅有 medication 记录的应激性高血糖患者混入
    MIN_EVIDENCE_TYPES = 2

    LAB_THRESHOLDS = {
        'hba1c_diagnostic': 6.5,   # %
        'glucose_fasting':  126,   # mg/dL
        'glucose_random':   200,   # mg/dL（随机血糖，ICU 常用）
    }

    EVIDENCE_WEIGHTS = {
        'ICD9_diagnosis':  1.0,
        'ICD10_diagnosis': 1.0,
        'medication':      0.8,
        'lab_abnormal':    0.7,
    }


# 构建药物正则（全局，避免重复编译）
_all_meds = []
for _med_list in Config.DIABETES_MEDICATIONS.values():
    _all_meds.extend(_med_list)
MED_PATTERN = '|'.join(re.escape(m) for m in _all_meds)


# ═══════════════════════════════════════════════════════════════════════════════
# 二、工具函数
# ═══════════════════════════════════════════════════════════════════════════════

def load_csv(path: Path, **kwargs) -> pd.DataFrame:
    """安全加载 CSV 文件"""
    try:
        if not path.exists():
            logger.warning(f"文件不存在: {path}")
            return pd.DataFrame()
        df = pd.read_csv(path, low_memory=False, **kwargs)
        logger.info(f"加载 {path.name}: {len(df):,} 行")
        return df
    except Exception as e:
        logger.error(f"加载文件失败 {path}: {e}")
        return pd.DataFrame()


def identify_dm_type_icd9(icd_code: str) -> str:
    """从 ICD-9 编码识别糖尿病类型（去点号后前缀匹配）"""
    code = str(icd_code).strip().replace('.', '')
    # T1DM: 250 第 5 位为奇数（1, 3）
    t1dm_prefixes = {'25001','25003','25011','25013','25021','25023',
                     '25031','25033','25041','25043','25051','25053',
                     '25061','25063','25071','25073','25081','25083',
                     '25091','25093'}
    # T2DM: 250 第 5 位为偶数（0, 2）
    t2dm_prefixes = {'25000','25002','25010','25012','25020','25022',
                     '25030','25032','25040','25042','25050','25052',
                     '25060','25062','25070','25072','25080','25082',
                     '25090','25092'}
    code5 = code[:5]
    if code5 in t1dm_prefixes:
        return 'T1DM'
    if code5 in t2dm_prefixes:
        return 'T2DM'
    if code.startswith('250'):
        return 'DM_unspecified'
    return 'unknown'


def identify_dm_type_icd10(icd_code: str) -> str:
    """从 ICD-10 编码识别糖尿病类型"""
    code = str(icd_code).strip().upper()
    if code.startswith('E10'):
        return 'T1DM'
    if code.startswith('E11'):
        return 'T2DM'
    return 'unknown'


def _dm_type_priority(dm_type: str) -> int:
    """结果优先级：越高的值在合并时越优先保留"""
    return {'T2DM': 3, 'T1DM': 2, 'DM_unspecified': 1}.get(dm_type, 0)


def calculate_evidence_quality(evidence_sources: str) -> float:
    """计算证据质量评分（基于独立来源类型组合）"""
    has_icd = 'ICD' in evidence_sources or 'diagnosis' in evidence_sources.lower()
    has_med = 'medication' in evidence_sources
    has_lab = 'lab_abnormal' in evidence_sources
    if has_icd and has_med and has_lab:
        return 1.0
    if has_icd and has_med:
        return 0.8
    if has_med and has_lab:
        return 0.7
    return 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# 三、并发症检测器（三库通用）
# ═══════════════════════════════════════════════════════════════════════════════

class ComplicationDetector:
    """糖尿病并发症检测器（ICD-9 / ICD-10 混合）"""

    def __init__(self, diagnoses_df: pd.DataFrame):
        self.diagnoses = diagnoses_df.copy()
        if 'icd_code' in self.diagnoses.columns:
            self.diagnoses['icd_code'] = (
                self.diagnoses['icd_code'].astype(str).str.strip().str.upper()
            )
        # 预建索引，避免 O(n) 全表扫描
        if 'subject_id' in self.diagnoses.columns:
            self._idx = self.diagnoses.groupby('subject_id')['icd_code'].apply(list).to_dict()
        else:
            self._idx = {}
        logger.info(f"并发症检测器初始化: {len(self.diagnoses):,} 条诊断记录")

    def detect_complications(self, subject_id) -> Dict[str, int]:
        codes = self._idx.get(subject_id, [])
        result = {}
        for comp_name, icd_map in Config.COMPLICATION_ICDS.items():
            all_prefixes = icd_map.get('icd9', []) + icd_map.get('icd10', [])
            has_comp = any(
                c.startswith(tuple(p.upper().replace('.', '') for p in all_prefixes))
                for c in codes
            )
            result[f'has_{comp_name}'] = int(has_comp)
        return result


# ═══════════════════════════════════════════════════════════════════════════════
# 四、共用的证据汇总逻辑
# ═══════════════════════════════════════════════════════════════════════════════

def _aggregate_evidence(
    combined:         pd.DataFrame,   # 含 subject_id / hadm_id / source / dm_type(可选)
    id_col:           str = 'subject_id',
    hadm_col:         str = 'hadm_id',
    min_evidence_types: int = Config.MIN_EVIDENCE_TYPES,
) -> pd.DataFrame:
    """
    [FIX-2] 以"独立来源类型数"而非"行数"统计证据强度。

    步骤：
      1. 在 (id_col, source) 粒度去重（同一患者同一来源类型只算一次）
      2. 统计每个患者的独立来源类型数（evidence_type_count）
      3. 筛选 >= min_evidence_types 的患者
      4. 保留 hadm_id 列表（JSON 格式）

    Returns:
        patient_stats DataFrame，含列：
          subject_id, evidence_type_count, evidence_sources,
          n_admissions, hadm_ids, dm_type
    """
    if combined.empty:
        return pd.DataFrame()

    # ── hadm_id 列表（不去重，保留全部 hadm）
    hadm_by_subject = (
        combined.groupby(id_col)[hadm_col]
        .apply(lambda x: list(set(str(v) for v in x if pd.notna(v))))
        .to_dict()
    )

    # ── 在 (subject, source) 粒度去重 → 统计独立来源类型数
    deduped = combined[[id_col, 'source']].drop_duplicates()
    type_counts = deduped.groupby(id_col)['source'].agg(
        evidence_type_count='count',
        evidence_sources=lambda x: ','.join(sorted(set(x))),
    ).reset_index()

    # ── 住院次数（唯一 hadm_id 数）
    n_admissions = (
        combined.groupby(id_col)[hadm_col].nunique().rename('n_admissions').reset_index()
    )
    patient_stats = type_counts.merge(n_admissions, on=id_col, how='left')

    # ── 筛选
    patient_stats = patient_stats[
        patient_stats['evidence_type_count'] >= min_evidence_types
    ].copy()

    # ── 添加 hadm_ids
    patient_stats['hadm_ids'] = patient_stats[id_col].map(
        lambda sid: json.dumps(hadm_by_subject.get(sid, []))
    )

    # ── DM 类型（最高优先级）+ [FIX-Bug8] T1+T2 共存检测
    if 'dm_type' in combined.columns:
        dm_priority = combined[combined['dm_type'].notna()].copy()
        dm_priority['_p'] = dm_priority['dm_type'].map(_dm_type_priority)

        # [FIX-Bug8] 检测同一 subject 是否同时出现 T1DM 和 T2DM 编码
        # 这类患者可能是 LADA 或数据质量问题，需单独标记用于敏感性分析
        dm_types_per_subject = (
            dm_priority[dm_priority['dm_type'].isin(['T1DM', 'T2DM'])]
            .groupby(id_col)['dm_type']
            .apply(lambda x: set(x))
        )
        mixed_subjects = set(
            dm_types_per_subject[
                dm_types_per_subject.apply(lambda s: 'T1DM' in s and 'T2DM' in s)
            ].index
        )
        if mixed_subjects:
            logger.warning(
                f"[Bug8] 检测到 {len(mixed_subjects)} 例患者同时具有 T1DM 和 T2DM 编码"
                f"（可能为 LADA 或编码错误）。这些患者的 dm_type 将标记为 'DM_mixed'。"
                f"敏感性分析时建议单独处理此子组。"
            )

        dm_type_map = (
            dm_priority.sort_values('_p', ascending=False)
            .groupby(id_col)['dm_type'].first()
            .to_dict()
        )
        patient_stats['dm_type'] = patient_stats[id_col].map(dm_type_map).fillna('unknown')

        # 将共存患者覆盖标记为 DM_mixed（必须在 dm_type_map 赋值后覆盖）
        if mixed_subjects:
            patient_stats.loc[
                patient_stats[id_col].isin(mixed_subjects), 'dm_type'
            ] = 'DM_mixed'
    else:
        patient_stats['dm_type'] = 'unknown'

    patient_stats = patient_stats.rename(columns={id_col: 'subject_id'})
    patient_stats['evidence_quality'] = patient_stats['evidence_sources'].apply(
        calculate_evidence_quality
    )
    return patient_stats.reset_index(drop=True)


def _apply_age_filter(patient_stats: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """
    [FIX-Bug4] 年龄过滤：排除 < 18 岁（儿科指南不适用）和 > 89 岁（MIMIC 去标识化置为 300）。

    临床依据：
      - < 18 岁：ADA Section 9 为成人 T2DM 指南，不适用于儿科患者（另有 ADA 青少年章节）。
        若对儿科患者评价，ADA 成人用药标准将产生系统性误判（二甲双胍在儿科有年龄限制，
        SGLT-2i/GLP-1RA 在 18 岁以下适应证有限）。
      - > 89 岁：MIMIC-III/IV 将 > 89 岁患者年龄统一置为 300（去标识化），
        用于 PSM 和 CCI 计算时会产生异常值，且高龄患者的血糖目标宽松化
        （ADA 对衰弱老人 HbA1c 目标放宽至 < 8.5%），评价逻辑与主队列不同。
      - 年龄缺失（NaN）：保留（无信息不等于不合格），在报告中单独列示缺失率。

    Args:
        patient_stats: 含 'age' 列的患者 DataFrame
        dataset_name:  用于日志的数据库名称

    Returns:
        过滤后的 DataFrame，同时记录被排除患者数量（用于 CONSORT 流程图）
    """
    if 'age' not in patient_stats.columns:
        logger.warning(f"[Bug4] {dataset_name}: 无 'age' 列，跳过年龄过滤")
        return patient_stats

    n_before = len(patient_stats)
    valid_age = patient_stats['age'].notna()
    too_young = valid_age & (patient_stats['age'] < 18)
    too_old   = valid_age & (patient_stats['age'] > 89)

    n_young = int(too_young.sum())
    n_old   = int(too_old.sum())
    n_miss  = int((~valid_age).sum())

    excluded_mask = too_young | too_old
    filtered = patient_stats[~excluded_mask].copy()

    logger.info(
        f"[FIX-Bug4] {dataset_name} 年龄过滤: "
        f"原始={n_before:,} → 保留={len(filtered):,} "
        f"（排除 <18岁={n_young:,}, >89岁={n_old:,}; 年龄缺失保留={n_miss:,}）"
    )
    return filtered


# ═══════════════════════════════════════════════════════════════════════════════
# 五、MIMIC-III 提取器
# ═══════════════════════════════════════════════════════════════════════════════

class MIMICIIIExtractor:
    """MIMIC-III 糖尿病数据提取器（v2.0）"""

    def __init__(self, data_path: Path):
        self.data_path = data_path
        logger.info(f"初始化 MIMIC-III 提取器: {data_path}")
        self.admissions = self._load_admissions()
        self.diagnoses  = self._load_diagnoses()
        self.patients   = self._load_patients()
        self.comp_detector = (
            ComplicationDetector(self.diagnoses) if not self.diagnoses.empty else None
        )

    def _load_admissions(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(
                self.data_path / 'ADMISSIONS.csv.gz',
                usecols=['subject_id', 'hadm_id', 'admittime', 'dischtime'],
                parse_dates=['admittime', 'dischtime'],
            )
            df.columns = df.columns.str.lower()
            return df.set_index('hadm_id')
        except Exception as e:
            logger.error(f"加载 admissions 失败: {e}")
            return pd.DataFrame()

    def _load_diagnoses(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(
                self.data_path / 'DIAGNOSES_ICD.csv.gz',
                usecols=['subject_id', 'hadm_id', 'icd9_code'],
            )
            df.columns = df.columns.str.lower()
            df['icd9_code'] = df['icd9_code'].astype(str).str.strip()
            df['icd_code']  = df['icd9_code']
            df['icd_version'] = 9
            logger.info(f"MIMIC-III 诊断记录: {len(df):,} 行")
            return df
        except Exception as e:
            logger.error(f"加载 diagnoses 失败: {e}")
            return pd.DataFrame()

    def _load_patients(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(
                self.data_path / 'PATIENTS.csv.gz',
                usecols=['subject_id', 'gender', 'dob'],
            )
            df.columns = df.columns.str.lower()
            df['dob'] = pd.to_datetime(df['dob'], errors='coerce')
            return df.set_index('subject_id')
        except Exception as e:
            logger.error(f"加载 patients 失败: {e}")
            return pd.DataFrame()

    def filter_by_icd9(self) -> pd.DataFrame:
        """基于 ICD-9 250.xx 筛选"""
        if self.diagnoses.empty:
            return pd.DataFrame()
        mask  = self.diagnoses['icd9_code'].str.startswith('250', na=False)
        df_dm = self.diagnoses[mask].copy()
        df_dm['dm_type'] = df_dm['icd9_code'].apply(identify_dm_type_icd9)
        df_dm['source']  = 'ICD9_diagnosis'
        logger.info(
            f"MIMIC-III ICD-9 筛选: {len(df_dm):,} 条 "
            f"(T1DM={sum(df_dm['dm_type']=='T1DM'):,}, "
            f"T2DM={sum(df_dm['dm_type']=='T2DM'):,})"
        )
        return df_dm[['subject_id', 'hadm_id', 'dm_type', 'source']]

    def filter_by_medication(self) -> pd.DataFrame:
        """基于药物处方筛选"""
        df = load_csv(
            self.data_path / 'PRESCRIPTIONS.csv.gz',
            usecols=['subject_id', 'hadm_id', 'drug'],
        )
        if df.empty:
            return pd.DataFrame()
        df.columns = df.columns.str.lower()
        mask  = df['drug'].astype(str).str.contains(MED_PATTERN, case=False, na=False, regex=True)
        df_dm = df[mask].copy()
        df_dm['source'] = 'medication'
        logger.info(f"MIMIC-III 药物筛选: {len(df_dm):,} 条")
        return df_dm[['subject_id', 'hadm_id', 'source']].drop_duplicates()

    def filter_by_lab(self) -> pd.DataFrame:
        """
        基于 HbA1c / 血糖筛选。

        [FIX-Bug5] ICU 中不存在真正意义上的"空腹"状态，glucose_fasting（126 mg/dL）
        在危重症患者中因应激性高血糖极易触发，会将大量非糖尿病患者混入队列。
        修复：血糖证据使用 glucose_random（200 mg/dL，ADA 随机血糖诊断阈值），
        并将 HbA1c 和血糖拆分为两个独立 source 标签：
          - 'lab_hba1c'    : HbA1c ≥ 6.5%（诊断性，单独即可作为 evidence type）
          - 'lab_glucose'  : 随机血糖 ≥ 200 mg/dL（支持性，需与其他来源联合才达标）
        这样 min_evidence_types=2 对纯血糖来源仍有保护作用。
        """
        d_lab = load_csv(self.data_path / 'D_LABITEMS.csv.gz', usecols=['itemid', 'label'])
        if d_lab.empty:
            return pd.DataFrame()
        d_lab['label'] = d_lab['label'].str.lower()
        hba1c_items   = d_lab[d_lab['label'].str.contains('hba1c|hemoglobin a1c|glycohemoglobin', na=False)]['itemid'].tolist()
        glucose_items = d_lab[d_lab['label'].str.contains('glucose', na=False) &
                               ~d_lab['label'].str.contains('csf|urine', na=False)]['itemid'].tolist()
        target_items  = hba1c_items + glucose_items
        if not target_items:
            return pd.DataFrame()

        lab_df = load_csv(
            self.data_path / 'LABEVENTS.csv.gz',
            usecols=['subject_id', 'hadm_id', 'itemid', 'valuenum'],
        )
        if lab_df.empty:
            return pd.DataFrame()
        lab_df = lab_df[lab_df['itemid'].isin(target_items)].copy()
        lab_df = lab_df[lab_df['valuenum'].notna()]

        rows = []
        # HbA1c ≥ 6.5%
        hba1c_df = lab_df[
            lab_df['itemid'].isin(hba1c_items) &
            (lab_df['valuenum'] >= Config.LAB_THRESHOLDS['hba1c_diagnostic'])
        ][['subject_id', 'hadm_id']].drop_duplicates().copy()
        hba1c_df['source'] = 'lab_hba1c'
        rows.append(hba1c_df)

        # 随机血糖 ≥ 200 mg/dL（[FIX-Bug5] 从 126 改为 200）
        gluc_df = lab_df[
            lab_df['itemid'].isin(glucose_items) &
            (lab_df['valuenum'] >= Config.LAB_THRESHOLDS['glucose_random'])
        ][['subject_id', 'hadm_id']].drop_duplicates().copy()
        gluc_df['source'] = 'lab_glucose'
        rows.append(gluc_df)

        df_dm = pd.concat(rows, ignore_index=True).drop_duplicates()
        logger.info(
            f"MIMIC-III 实验室筛选: HbA1c={len(hba1c_df):,} 条, "
            f"血糖≥200={len(gluc_df):,} 条"
        )
        return df_dm[['subject_id', 'hadm_id', 'source']].drop_duplicates()

    def extract_patient_demographics(self, subject_ids: List) -> pd.DataFrame:
        if self.patients.empty:
            return pd.DataFrame()
        subset = self.patients[self.patients.index.isin(subject_ids)]
        rows = []
        for sid in subject_ids:
            if sid not in subset.index:
                continue
            p = subset.loc[sid]
            rows.append({'subject_id': sid, 'gender': p.get('gender'), 'age': None})
        return pd.DataFrame(rows)

    def get_diabetes_patients(self, min_evidence_types: int = Config.MIN_EVIDENCE_TYPES) -> pd.DataFrame:
        logger.info(f"\nMIMIC-III 综合筛选（最少独立来源类型数 = {min_evidence_types}）...")

        pieces = []
        icd_df = self.filter_by_icd9()
        if not icd_df.empty:
            pieces.append(icd_df)
        med_df = self.filter_by_medication()
        if not med_df.empty:
            pieces.append(med_df.assign(dm_type=None))
        lab_df = self.filter_by_lab()
        if not lab_df.empty:
            pieces.append(lab_df.assign(dm_type=None))

        if not pieces:
            logger.error("MIMIC-III: 未找到任何证据")
            return pd.DataFrame()

        combined = pd.concat(pieces, ignore_index=True)
        # 确保有 dm_type 列
        if 'dm_type' not in combined.columns:
            combined['dm_type'] = None

        patient_stats = _aggregate_evidence(
            combined, id_col='subject_id', hadm_col='hadm_id',
            min_evidence_types=min_evidence_types,
        )

        demographics = self.extract_patient_demographics(patient_stats['subject_id'].tolist())
        if not demographics.empty:
            patient_stats = patient_stats.merge(demographics, on='subject_id', how='left')

        if self.comp_detector:
            comp_rows = [
                {**self.comp_detector.detect_complications(sid), 'subject_id': sid}
                for sid in tqdm(patient_stats['subject_id'], desc="MIMIC-III 并发症检测")
            ]
            patient_stats = patient_stats.merge(pd.DataFrame(comp_rows), on='subject_id', how='left')

        logger.info(
            f"MIMIC-III 筛选完成（过滤前）: {len(patient_stats):,} 例 "
            f"(T1DM={sum(patient_stats['dm_type']=='T1DM'):,}, "
            f"T2DM={sum(patient_stats['dm_type']=='T2DM'):,}, "
            f"DM_mixed={sum(patient_stats['dm_type']=='DM_mixed'):,}, "
            f"unknown={sum(patient_stats['dm_type']=='unknown'):,})"
        )
        # [FIX-Bug4] 年龄过滤（必须在 demographics merge 之后执行）
        patient_stats = _apply_age_filter(patient_stats, "MIMIC-III")
        return patient_stats.sort_values('evidence_quality', ascending=False).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 六、MIMIC-IV 提取器（核心修复）
# ═══════════════════════════════════════════════════════════════════════════════

class MIMICIVExtractor:
    """
    MIMIC-IV 糖尿病数据提取器（v2.0）

    [FIX-1] 关键修复：同时使用 ICD-9（250.xx）和 ICD-10（E10/E11）识别糖尿病。

    背景：MIMIC-IV 数据库包含 2008-2022 年的住院记录。
      - 2008-2015 年：大量住院使用 ICD-9 编码（过渡期）
      - 2016 年起：美国强制切换到 ICD-10（美国 ICD-10-CM 正式实施日期为 2015-10-01）
      - 实际上，MIMIC-IV 的 diagnoses_icd 表同时包含 icd_version=9 和 icd_version=10 的行

    旧代码只保留 icd_version==10，导致：
      - Era-1（2008-2013）的所有 ICD-9 T2DM 患者被丢弃
      - Era-2 早期（2014-2015）的 ICD-9 记录也被丢弃
      - 实际纳入率在 Era-1 段极低，严重影响时序分析的平衡性

    修复方案：
      - _load_diagnoses_icd9()：读取 icd_version==9 的 250.xx 记录
      - _load_diagnoses_icd10()：读取 icd_version==10 的 E10/E11 记录
      - filter_by_icd()：合并两路结果，dm_type 以 ICD-10 优先
    """

    def __init__(self, data_path: Path):
        self.data_path = data_path
        logger.info(f"初始化 MIMIC-IV 提取器: {data_path}")
        self.admissions      = self._load_admissions()
        self.diagnoses_icd9  = self._load_diagnoses_icd9()
        self.diagnoses_icd10 = self._load_diagnoses_icd10()
        self.patients        = self._load_patients()

        # 合并诊断用于并发症检测
        diag_all = pd.concat([self.diagnoses_icd9, self.diagnoses_icd10], ignore_index=True)
        self.comp_detector = (
            ComplicationDetector(diag_all) if not diag_all.empty else None
        )

    def _load_admissions(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(
                self.data_path / 'hosp' / 'admissions.csv.gz',
                usecols=['subject_id', 'hadm_id', 'admittime', 'dischtime'],
                parse_dates=['admittime', 'dischtime'],
            )
            return df.set_index('hadm_id')
        except Exception as e:
            logger.error(f"加载 admissions 失败: {e}")
            return pd.DataFrame()

    def _load_diagnoses_icd9(self) -> pd.DataFrame:
        """
        [FIX-1] 加载 MIMIC-IV 中的 ICD-9 诊断记录（icd_version == 9）。
        覆盖 2008-2015 年的早期住院（Era-1 全部 + Era-2 早期）。

        [OPT] 与 _load_diagnoses_icd10() 共享同一次 CSV 读取，避免重复 I/O。
        """
        if not hasattr(self, '_diagnoses_raw'):
            self._diagnoses_raw = self._load_diagnoses_all()
        df9 = self._diagnoses_raw[self._diagnoses_raw['icd_version'] == 9].copy()
        df9['icd_code'] = df9['icd_code'].astype(str).str.strip()
        logger.info(f"MIMIC-IV ICD-9 诊断记录: {len(df9):,} 行")
        return df9[['subject_id', 'hadm_id', 'icd_code', 'icd_version']]

    def _load_diagnoses_icd10(self) -> pd.DataFrame:
        """加载 MIMIC-IV 中的 ICD-10 诊断记录（icd_version == 10）。"""
        if not hasattr(self, '_diagnoses_raw'):
            self._diagnoses_raw = self._load_diagnoses_all()
        df10 = self._diagnoses_raw[self._diagnoses_raw['icd_version'] == 10].copy()
        df10['icd_code'] = df10['icd_code'].astype(str).str.strip().str.upper()
        logger.info(f"MIMIC-IV ICD-10 诊断记录: {len(df10):,} 行")
        return df10[['subject_id', 'hadm_id', 'icd_code', 'icd_version']]

    def _load_diagnoses_all(self) -> pd.DataFrame:
        """[OPT] 一次性读取 diagnoses_icd.csv.gz，由 _load_diagnoses_icd9/10 共享。"""
        try:
            df = pd.read_csv(
                self.data_path / 'hosp' / 'diagnoses_icd.csv.gz',
                usecols=['subject_id', 'hadm_id', 'icd_code', 'icd_version'],
            )
            logger.info(f"MIMIC-IV 诊断记录（ICD-9+10 合并读取）: {len(df):,} 行")
            return df
        except Exception as e:
            logger.error(f"加载 MIMIC-IV diagnoses 失败: {e}")
            return pd.DataFrame(columns=['subject_id', 'hadm_id', 'icd_code', 'icd_version'])

    def _load_patients(self) -> pd.DataFrame:
        """
        [FIX-3] 补充读取 anchor_year_group，供 classify_era() 使用。
        """
        try:
            df = pd.read_csv(
                self.data_path / 'hosp' / 'patients.csv.gz',
                usecols=['subject_id', 'gender', 'anchor_age', 'anchor_year', 'anchor_year_group'],
            )
            return df.set_index('subject_id')
        except Exception as e:
            logger.error(f"加载 patients 失败: {e}")
            return pd.DataFrame()

    def filter_by_icd(self) -> pd.DataFrame:
        """
        [FIX-1] 合并 ICD-9（250.xx）和 ICD-10（E10/E11）两路糖尿病诊断。

        优先级：同一 hadm 若同时有 ICD-9 和 ICD-10 记录，以 ICD-10 为准
        （因为 ICD-10 更精确且更新）。
        """
        rows = []

        # ── ICD-9 路：250.xx
        if not self.diagnoses_icd9.empty:
            mask9 = self.diagnoses_icd9['icd_code'].str.startswith('250', na=False)
            df9   = self.diagnoses_icd9[mask9].copy()
            df9['dm_type'] = df9['icd_code'].apply(identify_dm_type_icd9)
            df9['source']  = 'ICD9_diagnosis'
            n_t1 = sum(df9['dm_type'] == 'T1DM')
            n_t2 = sum(df9['dm_type'] == 'T2DM')
            logger.info(
                f"MIMIC-IV ICD-9 筛选: {len(df9):,} 条 "
                f"(T1DM={n_t1:,}, T2DM={n_t2:,})"
            )
            rows.append(df9[['subject_id', 'hadm_id', 'dm_type', 'source']])

        # ── ICD-10 路：E10 / E11（排除 E08/E09/E13/E14 继发性糖尿病）
        if not self.diagnoses_icd10.empty:
            mask10 = (
                self.diagnoses_icd10['icd_code'].str.startswith('E10', na=False) |
                self.diagnoses_icd10['icd_code'].str.startswith('E11', na=False)
            )
            # [FIX-Bug9] 追踪继发性糖尿病排除数（CONSORT 流程图必要数字）
            exclude_prefixes = tuple(Config.ICD10_CODES['exclude'])  # ('E08','E09','E13')
            exclude_mask = self.diagnoses_icd10['icd_code'].str.startswith(
                exclude_prefixes, na=False
            )
            n_secondary_dx = int(
                self.diagnoses_icd10[exclude_mask]['subject_id'].nunique()
            )
            if n_secondary_dx > 0:
                logger.info(
                    f"[FIX-Bug9] MIMIC-IV 排除继发性糖尿病（E08/E09/E13）: "
                    f"{n_secondary_dx:,} 例患者（仅含这些编码，无 E10/E11 的将不进入队列）。"
                    f"此数字用于 CONSORT 流程图'Excluded — secondary DM'。"
                )
            df10 = self.diagnoses_icd10[mask10 & ~exclude_mask].copy()
            df10['dm_type'] = df10['icd_code'].apply(identify_dm_type_icd10)
            df10['source']  = 'ICD10_diagnosis'
            n_t1 = sum(df10['dm_type'] == 'T1DM')
            n_t2 = sum(df10['dm_type'] == 'T2DM')
            logger.info(
                f"MIMIC-IV ICD-10 筛选: {len(df10):,} 条 "
                f"(T1DM={n_t1:,}, T2DM={n_t2:,})"
            )
            rows.append(df10[['subject_id', 'hadm_id', 'dm_type', 'source']])

        if not rows:
            return pd.DataFrame()

        combined_icd = pd.concat(rows, ignore_index=True)

        # 同一 (subject_id, hadm_id) 的 dm_type 以 ICD-10 优先
        combined_icd['_p'] = combined_icd['dm_type'].map(_dm_type_priority)
        combined_icd = (
            combined_icd.sort_values('_p', ascending=False)
            .drop_duplicates(subset=['subject_id', 'hadm_id', 'source'])
        )
        return combined_icd[['subject_id', 'hadm_id', 'dm_type', 'source']]

    def filter_by_medication(self) -> pd.DataFrame:
        df = load_csv(
            self.data_path / 'hosp' / 'prescriptions.csv.gz',
            usecols=['subject_id', 'hadm_id', 'drug'],
        )
        if df.empty:
            return pd.DataFrame()
        mask  = df['drug'].astype(str).str.contains(MED_PATTERN, case=False, na=False, regex=True)
        df_dm = df[mask].copy()
        df_dm['source'] = 'medication'
        logger.info(f"MIMIC-IV 药物筛选: {len(df_dm):,} 条")
        return df_dm[['subject_id', 'hadm_id', 'source']].drop_duplicates()

    def filter_by_lab(self) -> pd.DataFrame:
        """
        [FIX-Bug5] ICU 血糖阈值修正：使用 glucose_random（200 mg/dL）替代
        glucose_fasting（126 mg/dL）；HbA1c 与血糖使用独立 source 标签。
        """
        d_lab = load_csv(
            self.data_path / 'hosp' / 'd_labitems.csv.gz',
            usecols=['itemid', 'label'],
        )
        if d_lab.empty:
            return pd.DataFrame()
        d_lab['label']    = d_lab['label'].str.lower()
        hba1c_items       = d_lab[d_lab['label'].str.contains('hba1c|hemoglobin a1c|glycohemoglobin', na=False)]['itemid'].tolist()
        glucose_items     = d_lab[d_lab['label'].str.contains('glucose', na=False) &
                                  ~d_lab['label'].str.contains('csf|urine', na=False)]['itemid'].tolist()
        target_items      = hba1c_items + glucose_items
        if not target_items:
            return pd.DataFrame()

        lab_df = load_csv(
            self.data_path / 'hosp' / 'labevents.csv.gz',
            usecols=['subject_id', 'hadm_id', 'itemid', 'valuenum'],
        )
        if lab_df.empty:
            return pd.DataFrame()
        lab_df = lab_df[lab_df['itemid'].isin(target_items) & lab_df['valuenum'].notna()]

        rows = []
        hba1c_df = lab_df[
            lab_df['itemid'].isin(hba1c_items) &
            (lab_df['valuenum'] >= Config.LAB_THRESHOLDS['hba1c_diagnostic'])
        ][['subject_id', 'hadm_id']].drop_duplicates().copy()
        hba1c_df['source'] = 'lab_hba1c'
        rows.append(hba1c_df)

        gluc_df = lab_df[
            lab_df['itemid'].isin(glucose_items) &
            (lab_df['valuenum'] >= Config.LAB_THRESHOLDS['glucose_random'])
        ][['subject_id', 'hadm_id']].drop_duplicates().copy()
        gluc_df['source'] = 'lab_glucose'
        rows.append(gluc_df)

        df_dm = pd.concat(rows, ignore_index=True).drop_duplicates()
        logger.info(
            f"MIMIC-IV 实验室筛选: HbA1c={len(hba1c_df):,} 条, "
            f"血糖≥200={len(gluc_df):,} 条"
        )
        return df_dm[['subject_id', 'hadm_id', 'source']].drop_duplicates()

    def extract_patient_demographics(self, subject_ids: List) -> pd.DataFrame:
        """[FIX-3] 包含 anchor_year_group 字段"""
        if self.patients.empty:
            return pd.DataFrame()
        subset = self.patients[self.patients.index.isin(subject_ids)]
        rows = []
        for sid in subject_ids:
            if sid not in subset.index:
                continue
            p = subset.loc[sid]
            rows.append({
                'subject_id':       sid,
                'gender':           p.get('gender'),
                'age':              p.get('anchor_age'),
                'anchor_year':      p.get('anchor_year'),
                'anchor_year_group': p.get('anchor_year_group'),   # [FIX-3]
            })
        return pd.DataFrame(rows)

    def get_diabetes_patients(self, min_evidence_types: int = Config.MIN_EVIDENCE_TYPES) -> pd.DataFrame:
        logger.info(f"\nMIMIC-IV 综合筛选（最少独立来源类型数 = {min_evidence_types}）...")

        pieces = []
        icd_df = self.filter_by_icd()   # [FIX-1] 合并 ICD-9 + ICD-10
        if not icd_df.empty:
            pieces.append(icd_df)

        med_df = self.filter_by_medication()
        if not med_df.empty:
            pieces.append(med_df.assign(dm_type=None))

        lab_df = self.filter_by_lab()
        if not lab_df.empty:
            pieces.append(lab_df.assign(dm_type=None))

        if not pieces:
            logger.error("MIMIC-IV: 未找到任何证据")
            return pd.DataFrame()

        combined = pd.concat(pieces, ignore_index=True)
        if 'dm_type' not in combined.columns:
            combined['dm_type'] = None

        patient_stats = _aggregate_evidence(
            combined, id_col='subject_id', hadm_col='hadm_id',
            min_evidence_types=min_evidence_types,
        )

        demographics = self.extract_patient_demographics(patient_stats['subject_id'].tolist())
        if not demographics.empty:
            patient_stats = patient_stats.merge(demographics, on='subject_id', how='left')

        if self.comp_detector:
            comp_rows = [
                {**self.comp_detector.detect_complications(sid), 'subject_id': sid}
                for sid in tqdm(patient_stats['subject_id'], desc="MIMIC-IV 并发症检测")
            ]
            patient_stats = patient_stats.merge(pd.DataFrame(comp_rows), on='subject_id', how='left')

        logger.info(
            f"MIMIC-IV 筛选完成（过滤前）: {len(patient_stats):,} 例 "
            f"(T1DM={sum(patient_stats['dm_type']=='T1DM'):,}, "
            f"T2DM={sum(patient_stats['dm_type']=='T2DM'):,}, "
            f"DM_mixed={sum(patient_stats['dm_type']=='DM_mixed'):,}, "
            f"unknown={sum(patient_stats['dm_type']=='unknown'):,})"
        )

        # 额外日志：各 anchor_year_group 的患者数（验证 Era-1 是否得到修复）
        if 'anchor_year_group' in patient_stats.columns:
            ayg_dist = patient_stats['anchor_year_group'].value_counts().sort_index()
            logger.info("anchor_year_group 分布（Era 分类验证）:")
            for ayg, cnt in ayg_dist.items():
                logger.info(f"  {ayg}: {cnt:,}")

        # [FIX-Bug4] 年龄过滤（必须在 demographics merge 之后执行）
        patient_stats = _apply_age_filter(patient_stats, "MIMIC-IV")
        return patient_stats.sort_values('evidence_quality', ascending=False).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 七、eICU 提取器
# ═══════════════════════════════════════════════════════════════════════════════

class eICUExtractor:
    """eICU 糖尿病数据提取器（v2.0）"""

    def __init__(self, data_path: Path):
        self.data_path = data_path
        logger.info(f"初始化 eICU 提取器: {data_path}")
        self.patients  = self._load_patient()
        self.diagnosis = self._load_diagnosis()
        self.comp_detector = (
            ComplicationDetector(self.diagnosis) if not self.diagnosis.empty else None
        )

    def _load_patient(self) -> pd.DataFrame:
        """[FIX-4] 补充读取 hospitaldischargeyear"""
        try:
            df = pd.read_csv(
                self.data_path / 'patient.csv.gz',
                usecols=['patientunitstayid', 'uniquepid', 'gender', 'age',
                         'hospitaldischargeyear'],   # [FIX-4]
            )
            df.columns = df.columns.str.lower()
            return df.set_index('patientunitstayid')
        except Exception as e:
            # 若字段不存在则降级（兼容旧版 eICU）
            logger.warning(f"加载 patient（含 hospitaldischargeyear）失败，降级: {e}")
            try:
                df = pd.read_csv(
                    self.data_path / 'patient.csv.gz',
                    usecols=['patientunitstayid', 'uniquepid', 'gender', 'age'],
                )
                df.columns = df.columns.str.lower()
                df['hospitaldischargeyear'] = None
                return df.set_index('patientunitstayid')
            except Exception as e2:
                logger.error(f"加载 patient 失败: {e2}")
                return pd.DataFrame()

    def _load_diagnosis(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(
                self.data_path / 'diagnosis.csv.gz',
                usecols=['patientunitstayid', 'icd9code', 'diagnosisstring'],
            )
            df.columns = df.columns.str.lower()
            df['icd9code'] = df['icd9code'].astype(str).str.strip()
            df['icd_code'] = df['icd9code']
            df['icd_version'] = 9
            # 并发症检测器需要 subject_id 列，用 patientunitstayid 代替
            df['subject_id'] = df['patientunitstayid']
            logger.info(f"eICU 诊断记录: {len(df):,} 行")
            return df
        except Exception as e:
            logger.error(f"加载 diagnosis 失败: {e}")
            return pd.DataFrame()

    def filter_by_icd9(self) -> pd.DataFrame:
        """ICD-9 250.xx 筛选（同时利用 diagnosisstring 兜底 icd9code 缺失的情况）"""
        df = load_csv(
            self.data_path / 'diagnosis.csv.gz',
            usecols=['patientunitstayid', 'icd9code', 'diagnosisstring'],
        )
        if df.empty:
            return pd.DataFrame()
        df.columns = df.columns.str.lower()
        df['icd9code'] = df['icd9code'].astype(str).str.strip()

        # 路 1：icd9code 以 250 开头
        mask_icd = df['icd9code'].str.startswith('250', na=False)
        # 路 2：diagnosisstring 包含 diabetes/diabetic 关键词（icd9code=NULL 的兜底）
        mask_str = df['diagnosisstring'].str.contains(
            r'diabet|250\.', case=False, na=False, regex=True
        )

        df_dm = df[mask_icd | mask_str].copy()
        df_dm['dm_type'] = df_dm['icd9code'].apply(
            lambda c: identify_dm_type_icd9(c) if c.startswith('250') else 'DM_unspecified'
        )
        df_dm['source'] = 'ICD9_diagnosis'
        logger.info(
            f"eICU ICD-9 + string 筛选: {len(df_dm):,} 条 "
            f"(T1DM={sum(df_dm['dm_type']=='T1DM'):,}, "
            f"T2DM={sum(df_dm['dm_type']=='T2DM'):,})"
        )
        return df_dm[['patientunitstayid', 'dm_type', 'source']].rename(
            columns={'patientunitstayid': 'stay_id'}
        )

    def filter_by_medication(self) -> pd.DataFrame:
        df = load_csv(
            self.data_path / 'medication.csv.gz',
            usecols=['patientunitstayid', 'drugname'],
        )
        if df.empty:
            return pd.DataFrame()
        df.columns = df.columns.str.lower()
        mask  = df['drugname'].astype(str).str.contains(MED_PATTERN, case=False, na=False, regex=True)
        df_dm = df[mask].copy()
        df_dm['source'] = 'medication'
        logger.info(f"eICU 药物筛选: {len(df_dm):,} 条")
        return df_dm[['patientunitstayid', 'source']].drop_duplicates().rename(
            columns={'patientunitstayid': 'stay_id'}
        )

    def filter_by_lab(self) -> pd.DataFrame:
        """
        [FIX-Bug5] ICU 血糖阈值修正：使用 glucose_random（200 mg/dL）替代
        glucose_fasting（126 mg/dL）；HbA1c 与血糖使用独立 source 标签。
        """
        df = load_csv(
            self.data_path / 'lab.csv.gz',
            usecols=['patientunitstayid', 'labname', 'labresult'],
        )
        if df.empty:
            return pd.DataFrame()
        df.columns  = df.columns.str.lower()
        df['labname']   = df['labname'].str.lower()
        df['labresult'] = pd.to_numeric(df['labresult'], errors='coerce')
        df = df[df['labresult'].notna()]

        rows = []
        hba1c_df = df[
            df['labname'].str.contains('hba1c', na=False) &
            (df['labresult'] >= Config.LAB_THRESHOLDS['hba1c_diagnostic'])
        ][['patientunitstayid']].drop_duplicates().copy()
        hba1c_df['source'] = 'lab_hba1c'
        rows.append(hba1c_df)

        gluc_df = df[
            df['labname'].str.contains('glucose', na=False) &
            (df['labresult'] >= Config.LAB_THRESHOLDS['glucose_random'])
        ][['patientunitstayid']].drop_duplicates().copy()
        gluc_df['source'] = 'lab_glucose'
        rows.append(gluc_df)

        df_dm = pd.concat(rows, ignore_index=True).drop_duplicates()
        logger.info(
            f"eICU 实验室筛选: HbA1c={len(hba1c_df):,} 条, "
            f"血糖≥200={len(gluc_df):,} 条"
        )
        return df_dm[['patientunitstayid', 'source']].drop_duplicates().rename(
            columns={'patientunitstayid': 'stay_id'}
        )

    def extract_patient_demographics(self, uniquepids: List) -> pd.DataFrame:
        """[FIX-4] 包含 hospitaldischargeyear 字段"""
        if self.patients.empty:
            return pd.DataFrame()
        # patients 表 indexed by patientunitstayid，需要通过 uniquepid 关联
        patient_reset = self.patients.reset_index()
        subset = patient_reset[patient_reset['uniquepid'].isin(uniquepids)]
        # 每个 uniquepid 可能对应多行（多次 ICU stays），取第一行的人口学信息
        subset_first = subset.sort_index().groupby('uniquepid').first().reset_index()
        rows = []
        for _, row in subset_first.iterrows():
            age_raw = row.get('age', None)
            try:
                age = float(age_raw) if str(age_raw) != '> 89' and pd.notna(age_raw) else (90.0 if str(age_raw) == '> 89' else None)
            except (ValueError, TypeError):
                age = None
            rows.append({
                'subject_id':              row['uniquepid'],
                'gender':                  row.get('gender'),
                'age':                     age,
                'hospitaldischargeyear':   row.get('hospitaldischargeyear'),  # [FIX-4]
            })
        return pd.DataFrame(rows)

    def get_diabetes_patients(self, min_evidence_types: int = Config.MIN_EVIDENCE_TYPES) -> pd.DataFrame:
        logger.info(f"\neICU 综合筛选（最少独立来源类型数 = {min_evidence_types}）...")

        pieces = []
        icd_df = self.filter_by_icd9()
        if not icd_df.empty:
            pieces.append(icd_df.rename(columns={'stay_id': 'hadm_id'}))

        med_df = self.filter_by_medication()
        if not med_df.empty:
            pieces.append(med_df.rename(columns={'stay_id': 'hadm_id'}).assign(dm_type=None))

        lab_df = self.filter_by_lab()
        if not lab_df.empty:
            pieces.append(lab_df.rename(columns={'stay_id': 'hadm_id'}).assign(dm_type=None))

        if not pieces:
            logger.error("eICU: 未找到任何证据")
            return pd.DataFrame()

        combined = pd.concat(pieces, ignore_index=True)
        if 'dm_type' not in combined.columns:
            combined['dm_type'] = None

        # [FIX-5] 建立 patientunitstayid → uniquepid 映射，在汇总前完成
        patient_df = load_csv(
            self.data_path / 'patient.csv.gz',
            usecols=['patientunitstayid', 'uniquepid'],
        )
        if not patient_df.empty:
            patient_df.columns = patient_df.columns.str.lower()
            combined = combined.merge(
                patient_df.rename(columns={'patientunitstayid': 'hadm_id'}),
                on='hadm_id', how='left',
            )
            # 用 uniquepid 作为 subject_id
            combined['subject_id'] = combined['uniquepid'].fillna(combined['hadm_id'].astype(str))
        else:
            combined['subject_id'] = combined['hadm_id'].astype(str)

        patient_stats = _aggregate_evidence(
            combined, id_col='subject_id', hadm_col='hadm_id',
            min_evidence_types=min_evidence_types,
        )

        demographics = self.extract_patient_demographics(patient_stats['subject_id'].tolist())
        if not demographics.empty:
            patient_stats = patient_stats.merge(demographics, on='subject_id', how='left')

        if self.comp_detector and 'subject_id' in self.comp_detector._idx:
            comp_rows = [
                {**self.comp_detector.detect_complications(sid), 'subject_id': sid}
                for sid in tqdm(patient_stats['subject_id'], desc="eICU 并发症检测")
            ]
            patient_stats = patient_stats.merge(pd.DataFrame(comp_rows), on='subject_id', how='left')

        logger.info(
            f"eICU 筛选完成（过滤前）: {len(patient_stats):,} 例 "
            f"(T1DM={sum(patient_stats['dm_type']=='T1DM'):,}, "
            f"T2DM={sum(patient_stats['dm_type']=='T2DM'):,}, "
            f"DM_mixed={sum(patient_stats['dm_type']=='DM_mixed'):,}, "
            f"unknown={sum(patient_stats['dm_type']=='unknown'):,})"
        )
        # [FIX-Bug4] 年龄过滤（必须在 demographics merge 之后执行）
        patient_stats = _apply_age_filter(patient_stats, "eICU")
        return patient_stats.sort_values('evidence_quality', ascending=False).reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 八、数据质量检查
# ═══════════════════════════════════════════════════════════════════════════════

def quality_check(df_patients: pd.DataFrame, dataset_name: str):
    logger.info("\n" + "=" * 60)
    logger.info(f"{dataset_name} 数据质量检查")
    logger.info("=" * 60)

    logger.info(f"\n总患者数: {len(df_patients):,}")

    # 证据来源组合分布
    logger.info("\n证据来源组合分布（独立类型）:")
    for combo, cnt in df_patients['evidence_sources'].value_counts().items():
        logger.info(f"  {combo}: {cnt:,} ({cnt/len(df_patients)*100:.1f}%)")

    # DM 类型分布
    logger.info("\nDM 类型分布:")
    for dm_type, cnt in df_patients['dm_type'].value_counts().items():
        logger.info(f"  {dm_type}: {cnt:,} ({cnt/len(df_patients)*100:.1f}%)")

    # 证据质量
    if 'evidence_quality' in df_patients.columns:
        hq = sum(df_patients['evidence_quality'] >= 0.8)
        logger.info(f"\n证据质量: 平均={df_patients['evidence_quality'].mean():.2f}, "
                    f"高质量(≥0.8)={hq:,} ({hq/len(df_patients)*100:.1f}%)")

    # 入院次数
    logger.info(f"\n入院次数: 平均={df_patients['n_admissions'].mean():.1f}, "
                f"中位数={df_patients['n_admissions'].median():.0f}, "
                f"最大={df_patients['n_admissions'].max()}")

    # 年龄
    if 'age' in df_patients.columns:
        valid_age = df_patients['age'].dropna()
        if len(valid_age) > 0:
            logger.info(f"\n年龄: 平均={valid_age.mean():.1f}, "
                        f"中位数={valid_age.median():.1f}, "
                        f"范围=[{valid_age.min():.0f}, {valid_age.max():.0f}]")

    # anchor_year_group（MIMIC-IV 专用）
    if 'anchor_year_group' in df_patients.columns:
        logger.info("\nanchor_year_group 分布（Era 分类验证）:")
        for ayg, cnt in df_patients['anchor_year_group'].value_counts().sort_index().items():
            logger.info(f"  {ayg}: {cnt:,}")

    # 并发症
    comp_cols = [c for c in df_patients.columns if c.startswith('has_')]
    if comp_cols:
        logger.info("\n并发症分布:")
        for col in comp_cols:
            cnt = int(df_patients[col].sum())
            logger.info(f"  {col.replace('has_', '')}: {cnt:,} ({cnt/len(df_patients)*100:.1f}%)")

    # 异常检查
    try:
        no_hadm = df_patients[df_patients['hadm_ids'].apply(lambda x: len(json.loads(x)) == 0)]
        if len(no_hadm) > 0:
            logger.error(f"\n❌ {len(no_hadm):,} 例患者没有有效的 hadm_ids！")
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# 九、主流程
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 60)
    logger.info("糖尿病数据提取与子集构建 v2.0 — 开始")
    logger.info("=" * 60)

    from agents.config import PathConfig

    MIMIC3_PATH  = PathConfig.MIMIC3_PATH
    MIMIC4_PATH  = PathConfig.MIMIC4_PATH
    EICU_PATH    = PathConfig.EICU_PATH
    OUTPUT_BASE  = PathConfig.PATIENCE_OUTPUT_BASE
    OUTPUT_BASE.mkdir(exist_ok=True)

    results_summary = {}

    # ── MIMIC-III
    if MIMIC3_PATH.exists():
        logger.info("\n" + "─" * 60)
        logger.info("处理 MIMIC-III")
        logger.info("─" * 60)
        try:
            extractor   = MIMICIIIExtractor(MIMIC3_PATH)
            dm_patients = extractor.get_diabetes_patients()
            if not dm_patients.empty:
                out = PathConfig.MIMIC3_PATIENCE_PATH
                dm_patients.to_csv(out, index=False)
                logger.info(f"MIMIC-III 患者列表已保存: {out}")
                quality_check(dm_patients, "MIMIC-III")
                results_summary['mimiciii'] = {
                    'patients': len(dm_patients),
                    'T1DM': int(sum(dm_patients['dm_type'] == 'T1DM')),
                    'T2DM': int(sum(dm_patients['dm_type'] == 'T2DM')),
                }
        except Exception as e:
            import traceback; traceback.print_exc()
            results_summary['mimiciii'] = {'error': str(e)}
    else:
        logger.warning(f"MIMIC-III 路径不存在: {MIMIC3_PATH}")

    # ── MIMIC-IV
    if MIMIC4_PATH.exists():
        logger.info("\n" + "─" * 60)
        logger.info("处理 MIMIC-IV  [含 ICD-9 + ICD-10 双路筛选，FIX-1]")
        logger.info("─" * 60)
        try:
            extractor   = MIMICIVExtractor(MIMIC4_PATH)
            dm_patients = extractor.get_diabetes_patients()
            if not dm_patients.empty:
                out = PathConfig.MIMIC4_PATIENCE_PATH
                dm_patients.to_csv(out, index=False)
                logger.info(f"MIMIC-IV 患者列表已保存: {out}")
                quality_check(dm_patients, "MIMIC-IV")
                results_summary['mimiciv'] = {
                    'patients': len(dm_patients),
                    'T1DM': int(sum(dm_patients['dm_type'] == 'T1DM')),
                    'T2DM': int(sum(dm_patients['dm_type'] == 'T2DM')),
                }
        except Exception as e:
            import traceback; traceback.print_exc()
            results_summary['mimiciv'] = {'error': str(e)}
    else:
        logger.warning(f"MIMIC-IV 路径不存在: {MIMIC4_PATH}")

    # ── eICU
    if EICU_PATH.exists():
        logger.info("\n" + "─" * 60)
        logger.info("处理 eICU  [含 diagnosisstring 兜底，FIX-4/5]")
        logger.info("─" * 60)
        try:
            extractor   = eICUExtractor(EICU_PATH)
            dm_patients = extractor.get_diabetes_patients()
            if not dm_patients.empty:
                out = PathConfig.EICU_PATIENCE_PATH
                dm_patients.to_csv(out, index=False)
                logger.info(f"eICU 患者列表已保存: {out}")
                quality_check(dm_patients, "eICU")
                results_summary['eicu'] = {
                    'patients': len(dm_patients),
                    'T1DM': int(sum(dm_patients['dm_type'] == 'T1DM')),
                    'T2DM': int(sum(dm_patients['dm_type'] == 'T2DM')),
                }
        except Exception as e:
            import traceback; traceback.print_exc()
            results_summary['eicu'] = {'error': str(e)}
    else:
        logger.warning(f"eICU 路径不存在: {EICU_PATH}")

    # ── 总结
    logger.info("\n" + "=" * 60)
    logger.info("处理完成！总体摘要：")
    logger.info("=" * 60)
    for dataset, info in results_summary.items():
        logger.info(f"\n{dataset.upper()}:")
        for k, v in info.items():
            logger.info(f"  {k}: {v}")
    logger.info(f"\n所有结果保存在: {OUTPUT_BASE}")


if __name__ == "__main__":
    main()
