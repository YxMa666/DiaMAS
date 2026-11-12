#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_extraction.py - 重构后的统一数据提取脚本

使用方式：
    # 提取单个数据库
    python run_extraction.py --database mimic3
    python run_extraction.py --database mimic4
    python run_extraction.py --database eicu

    # 提取全部三个数据库
    python run_extraction.py --database all

    # 自定义参数
    python run_extraction.py --database mimic3 --workers 8 --no-umls --force-refresh

    # 诊断单个患者
    python run_extraction.py --database mimic3 --diagnose 10094 122928

    # 生成数据质量报告
    python run_extraction.py --database mimic3 --quality-report

作者：Yuxi
日期：2025-02
"""

import sys
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict
from typing import List, Tuple, Optional, Union

# ==================== 路径设置 ====================
# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# ==================== 日志配置 ====================
LOG_DIR = PROJECT_ROOT / 'loggings'
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(
            LOG_DIR / f'extraction_{datetime.now():%Y%m%d_%H%M%S}.log',
            encoding='utf-8'
        ),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('run_extraction')


# ==================== 数据库配置 ====================

from agents.config import PathConfig

class DatabaseConfig:
    """三个数据库的路径与参数集中配置 — 从 PathConfig 读取（Single Source of Truth）"""

    DATABASES = {
        'mimic3': {
            'data_path':         PathConfig.MIMIC3_PATH,
            'patient_list_csv':  PathConfig.MIMIC3_PATIENCE_PATH,
            'umls_cache_dir':    PathConfig.UMLS_CACHE_BASE / 'mimiciii',
        },
        'mimic4': {
            'data_path':         PathConfig.MIMIC4_PATH,
            'patient_list_csv':  PathConfig.MIMIC4_PATIENCE_PATH,
            'umls_cache_dir':    PathConfig.UMLS_CACHE_BASE / 'mimiciv',
        },
        'eicu': {
            'data_path':         PathConfig.EICU_PATH,
            'patient_list_csv':  PathConfig.EICU_PATIENCE_PATH,
            'umls_cache_dir':    PathConfig.UMLS_CACHE_BASE / 'eicu',
        },
    }

    # 输出目录（统一从 PathConfig 读取，禁止硬编码版本号）
    CACHE_BASE = PathConfig.PATIENT_CACHE_BASE

    @classmethod
    def get(cls, database: str) -> dict:
        db = database.lower()
        if db not in cls.DATABASES:
            raise ValueError(f"Unsupported database: {db}. Choose from {list(cls.DATABASES)}")
        return cls.DATABASES[db]

    @classmethod
    def cache_dir(cls, database: str) -> Path:
        return cls.CACHE_BASE / database.lower()


# ==================== 患者列表提取 ====================

def load_patient_list(
    database: str,
    data_path: Path,
    min_evidence: int = 2,
    patient_list_override: Optional[Path] = None,
) -> List[Tuple[Union[int, str], Union[int, str]]]:
    """
    加载或提取患者列表

    优先从已有 CSV 读取，否则调用 diabetes_data_extraction 重新提取

    patient_list_override: 若指定，则完全跳过默认 CSV 路径，直接使用该文件。
        用于 Era-2 补充提取等场景，无需修改 DatabaseConfig。
        示例：--patient_list diabetes_subsets/mimiciv_patient_list_era2.csv
    """
    from diabetes_data_extraction import (
        MIMICIIIExtractor, MIMICIVExtractor, eICUExtractor
    )

    # ── [NEW] 外部 patient_list 覆盖路径 ────────────────────────────────────
    if patient_list_override is not None:
        if not patient_list_override.exists():
            raise FileNotFoundError(
                f"--patient_list 指定的文件不存在: {patient_list_override}\n"
                f"请先运行 generate_era2_patient_list.py 生成该文件。"
            )
        logger.info(f"[patient_list_override] 使用指定患者列表: {patient_list_override}")
        return _parse_patient_csv(patient_list_override, database)

    db_conf = DatabaseConfig.get(database)
    csv_path = db_conf['patient_list_csv']

    # 尝试从 CSV 加载
    if csv_path.exists():
        logger.info(f"从缓存加载患者列表: {csv_path}")
        return _parse_patient_csv(csv_path, database)

    # 从原始数据提取
    logger.info(f"未找到缓存，从原始数据提取患者列表...")
    extractors = {
        'mimic3': MIMICIIIExtractor,
        'mimic4': MIMICIVExtractor,
        'eicu': eICUExtractor,
    }

    extractor = extractors[database](data_path)
    df_patients = extractor.get_diabetes_patients(min_evidence=min_evidence)

    if df_patients.empty:
        logger.warning(f"未找到糖尿病患者 ({database})")
        return []

    # 保存 CSV
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df_patients.to_csv(csv_path, index=False)
    logger.info(f"患者列表已保存: {csv_path}")

    return _parse_patient_csv(csv_path, database)


def _parse_patient_csv(
    csv_path: Path,
    database: str,
) -> List[Tuple[Union[int, str], Union[int, str]]]:
    """解析患者列表 CSV → [(subject_id, hadm_id), ...]"""
    import pandas as pd

    df = pd.read_csv(csv_path)
    cases = []

    if database == 'eicu':
        # eICU: subject_id 可能是字符串 uniquepid
        # 需要 fallback 到 patient.csv.gz 的映射
        db_conf = DatabaseConfig.get('eicu')
        data_path = db_conf['data_path']

        # 构建 uniquepid → [patientunitstayid] 映射
        pid_map = {}
        try:
            pmap_df = pd.read_csv(
                data_path / 'patient.csv.gz',
                usecols=['patientunitstayid', 'uniquepid']
            )
            for uid, grp in pmap_df.groupby('uniquepid'):
                # [FIX-A2] 强制 str 键，避免 pandas 推断类型差异导致 .get() 静默失败
                pid_map[str(uid)] = grp['patientunitstayid'].tolist()
        except Exception as e:
            logger.warning(f"加载 patient.csv.gz 映射失败: {e}")

        for _, row in df.iterrows():
            subject_id = row['subject_id']
            hadm_ids = []

            if 'hadm_ids' in row and pd.notna(row['hadm_ids']):
                hadm_ids = json.loads(row['hadm_ids'])

            if not hadm_ids:
                # [FIX-A2] 与 pid_map 构建时的 str(uid) 保持一致
                hadm_ids = pid_map.get(str(subject_id), [])

            for hid in hadm_ids:
                cases.append((subject_id, int(float(hid))))
    else:
        # MIMIC-III / IV
        for _, row in df.iterrows():
            subject_id = int(row['subject_id'])
            # [FIX-BUG-NAN] eICU 分支已有 pd.notna 防护；MIMIC 分支同样需要，
            # 否则 hadm_ids 为 null 时 json.loads(float('nan')) → TypeError 崩溃
            if not pd.notna(row.get('hadm_ids')):
                logger.warning(f"  跳过 subject_id={subject_id}：hadm_ids 为 null")
                continue
            hadm_ids = json.loads(row['hadm_ids'])
            for hid in hadm_ids:
                cases.append((subject_id, int(float(hid))))

    logger.info(f"  解析到 {len(cases)} 个病例 ({database})")
    return cases


# ==================== 底层提取器创建 ====================

def create_base_extractor(database: str, data_path: Path):
    """
    创建底层数据库特定的提取器

    复用原有的 SemanticPatientExtractor / MIMIC4PatientExtractor / eICUPatientExtractor
    它们负责表结构和字段处理
    """
    from baseline.batch_data_loader import BatchDataLoader

    if database == 'mimic3':
        from baseline.semantic_patient_extractor import SemanticPatientExtractor
        batch_loader = BatchDataLoader()
        return SemanticPatientExtractor('mimic3', data_path, batch_loader=batch_loader)

    elif database == 'mimic4':
        from baseline.multi_database_extractors import MIMIC4PatientExtractor
        batch_loader = BatchDataLoader()
        return MIMIC4PatientExtractor(data_path, batch_loader=batch_loader)

    elif database == 'eicu':
        from baseline.multi_database_extractors import eICUPatientExtractor
        return eICUPatientExtractor(data_path)


# ==================== UMLS 映射器创建 ====================

def create_umls_mapper(database: str, enabled: bool = True):
    """创建 UMLS 映射器（带缓存）"""
    if not enabled:
        logger.info("UMLS mapping disabled")
        return None

    try:
        from agents.config import APIConfig
        from baseline.enhanced_umls_mapper import EnhancedUMLSMapperWithCache

        db_conf = DatabaseConfig.get(database)
        cache_dir = db_conf['umls_cache_dir']
        cache_dir.parent.mkdir(parents=True, exist_ok=True)

        mapper = EnhancedUMLSMapperWithCache(
            api_key=APIConfig.UMLS_KEY,
            cache_dir=str(cache_dir)
        )
        logger.info(f"✅ UMLS mapper initialized (cache={cache_dir})")
        return mapper

    except Exception as e:
        logger.warning(f"UMLS mapper creation failed: {e}")
        logger.warning("Continuing without UMLS mapping")
        return None


# ==================== 旧版→新版数据转换 ====================

def convert_legacy_patient_data(old_data):
    """
    将旧版 CompletePatientData（来自 semantic_patient_extractor.py）
    转换为新版（refactored/data_models.py）

    旧版缺少: bmi, data_quality, all_medications, clinical_metrics,
              ethnicity, height_cm, weight_kg
    旧版 LabResult 缺少: umls_cui, snomed_code, snomed_name, time_window, is_calculated
    旧版 Diagnosis 用 snomed_cui 而非 umls_cui/snomed_code/snomed_name

    [FIX-ALG-8] snomed_cui vs snomed_code 语义修正：
        旧版字段 snomed_cui 存的是 UMLS CUI（格式 C0XXXXXXX），
        不是 SNOMED ConceptId（纯数字，如 73211009）。
        错误映射到 snomed_code 字段会导致下游 SNOMED API 查询失败。
        修正：旧版 snomed_cui → 优先填入 umls_cui（若 umls_cui 为空）
    """
    from refactored.data_models import (
        CompletePatientData as NewPatientData,
        LabResult as NewLabResult,
        Diagnosis as NewDiagnosis,
        VitalSign as NewVitalSign,
    )

    # 如果已经是新版（有 bmi 属性且是 NewPatientData），直接返回
    if isinstance(old_data, NewPatientData):
        return old_data

    # 转换 LabResult
    new_labs = []
    for lab in (old_data.all_lab_results or []):
        new_labs.append(NewLabResult(
            itemid=lab.itemid,
            label=lab.label,
            loinc_code=getattr(lab, 'loinc_code', None),
            value=lab.value,
            unit=lab.unit,
            charttime=lab.charttime,
            is_abnormal=getattr(lab, 'is_abnormal', False),
            umls_cui=getattr(lab, 'umls_cui', None),
            snomed_code=getattr(lab, 'snomed_code', None),
            snomed_name=getattr(lab, 'snomed_name', None),
        ))

    # 转换 Diagnosis
    new_diags = []
    for diag in (old_data.all_diagnoses or []):
        # [FIX-ALG-8] 旧版 snomed_cui 实为 UMLS CUI，应映射到 umls_cui 字段
        # 旧版字段优先级：umls_cui > snomed_cui（作为 UMLS CUI 的备用来源）
        legacy_snomed_cui = getattr(diag, 'snomed_cui', None)
        new_umls_cui = getattr(diag, 'umls_cui', None) or legacy_snomed_cui

        # snomed_code 只保留真正的 SNOMED ConceptId（纯数字格式）
        raw_snomed_code = getattr(diag, 'snomed_code', None)
        if raw_snomed_code and not str(raw_snomed_code).startswith('C'):
            # 纯数字或不以 'C' 开头 → 可能是真实 SNOMED Code
            new_snomed_code = raw_snomed_code
        elif legacy_snomed_cui and not str(legacy_snomed_cui).startswith('C'):
            # snomed_cui 字段里存的其实是 SNOMED ConceptId
            new_snomed_code = legacy_snomed_cui
        else:
            new_snomed_code = None

        new_diags.append(NewDiagnosis(
            icd_code=diag.icd_code,
            icd_version=diag.icd_version,
            description=diag.description,
            seq_num=diag.seq_num,
            umls_cui=new_umls_cui,
            snomed_code=new_snomed_code,
            snomed_name=getattr(diag, 'snomed_name', None),
            is_primary=(diag.seq_num == 1),
        ))

    # 转换 VitalSign
    new_vitals = []
    for v in (old_data.all_vitals or []):
        new_vitals.append(NewVitalSign(
            itemid=v.itemid,
            label=v.label,
            value=v.value,
            unit=v.unit,
            charttime=v.charttime,
        ))

    return NewPatientData(
        database=old_data.database,
        subject_id=old_data.subject_id,
        hadm_id=old_data.hadm_id,
        age=old_data.age,
        gender=old_data.gender,
        # [FIX-N2] eICU 提取器已写入这四个字段，但旧版→新版转换时遗漏
        # 导致 bmi/ethnicity 全部丢失、data_quality.bmi_available 恒 False
        # getattr 对没有该属性的旧对象安全返回 None，不影响 MIMIC 路径
        ethnicity=getattr(old_data, 'ethnicity', None),
        height_cm=getattr(old_data, 'height_cm', None),
        weight_kg=getattr(old_data, 'weight_kg', None),
        bmi=getattr(old_data, 'bmi', None),
        all_lab_results=new_labs,
        all_diagnoses=new_diags,
        all_vitals=new_vitals,
        prior_medications=old_data.prior_medications or [],
        discharge_medications=old_data.discharge_medications or [],
        admission_time=old_data.admission_time,
        discharge_time=old_data.discharge_time,
        los_days=old_data.los_days,
        # [FIX-ERA] anchor_year_group 透传：MIMIC-IV 新提取的缓存已含此字段，
        # getattr 对无该属性的旧对象安全返回 None，不影响 MIMIC-III / eICU 路径
        anchor_year_group=getattr(old_data, 'anchor_year_group', None),
        anchor_year=getattr(old_data, 'anchor_year', None),
        clinical_metrics=getattr(old_data, 'clinical_metrics', None),
    )


# ==================== 主提取流程 ====================

def run_extraction(
    database: str,
    max_workers: int = 4,
    enable_umls: bool = True,
    force_refresh: bool = False,
    min_evidence: int = 2,
    max_patients: Optional[int] = None,
    lab_window_hours: int = 48,
    skip_eicu_vitals: bool = False,
    patient_list_override: Optional[Path] = None,
):
    """
    运行单个数据库的完整提取流程

    流程：
    1. 加载患者列表
    2. 初始化提取器 + UMLS mapper + 临床计算器 + 校验器
    3. 预加载大文件
    4. 并行提取（pipeline: 原始提取 → UMLS → eGFR → 校验 → 缓存）
    5. 生成摘要
    """
    from refactored.cache_manager import EnhancedCacheManager
    from refactored.clinical_calculator_unified import UnifiedClinicalCalculator
    from refactored.clinical_validator import ClinicalValidator
    from refactored.umls_mapping_unified import apply_umls_mapping

    db_conf = DatabaseConfig.get(database)
    data_path = db_conf['data_path']
    cache_dir = DatabaseConfig.cache_dir(database)

    logger.info(f"\n{'='*80}")
    logger.info(f"  DATABASE: {database.upper()}")
    logger.info(f"  Data:     {data_path}")
    logger.info(f"  Cache:    {cache_dir}")
    logger.info(f"  Workers:  {max_workers}")
    logger.info(f"  UMLS:     {'ON' if enable_umls else 'OFF'}")
    if database == 'eicu':
        logger.info(f"  eICU生命体征: {'跳过（--skip-eicu-vitals）' if skip_eicu_vitals else '扫描（预计 20-60 分钟）'}")
    logger.info(f"{'='*80}\n")

    # --- Step 1: 患者列表 ---
    t0 = time.time()
    cases = load_patient_list(database, data_path, min_evidence,
                              patient_list_override=patient_list_override)
    if not cases:
        logger.error("无患者可提取，退出")
        return []

    if max_patients:
        cases = cases[:max_patients]
        logger.info(f"  限制为前 {max_patients} 个患者")

    logger.info(f"  待提取: {len(cases)} 个病例 ({time.time()-t0:.1f}s)")

    # --- Step 2: 初始化组件 ---
    umls_mapper = create_umls_mapper(database, enable_umls)
    cache_manager = EnhancedCacheManager(cache_dir, fmt='json')
    calculator = UnifiedClinicalCalculator(database=database)
    extractor = create_base_extractor(database, data_path)

    # --- Step 3: 预加载 ---
    if hasattr(extractor, 'preload_batch'):
        logger.info("预加载大文件到内存...")
        t1 = time.time()
        # [FIX-P1] MIMIC-III/IV 的 chartevents.csv.gz 解压后数十GB，
        # 直接全量读入内存必然 OOM（尤其 MIMIC-IV + 8 workers）。
        # 与 refactored/extraction_pipeline.py 的 ExtractionPipeline.preload_batch 保持一致：
        # MIMIC 传 preload_vitals=False，跳过 chartevents 扫描，生命体征走按需懒加载。
        # eICU 仍需 preload_vitals_eicu（按目标 patientunitstayid 分块过滤，可控内存）。
        if database in ('mimic3', 'mimic4'):
            # [PERF-FIX] mimic3: preload_vitals=True + vital itemid 白名单过滤
            # 首次扫描多耗几分钟，但之后每患者 vitals 提取从 O(n_file) 降为 O(1)
            extractor.preload_batch(cases, preload_vitals=(database == 'mimic3'))
        else:
            # eICU：先在 preload_batch 里完成 lab/diag/med 预加载，
            # 再通过 preload_vitals_eicu 按目标 ID 过滤加载生命体征
            extractor.preload_batch(cases)
            target_ids = {int(hid) for _, hid in cases}
            if hasattr(extractor, 'preload_vitals_eicu'):
                extractor.preload_vitals_eicu(target_ids, skip=skip_eicu_vitals)
            else:
                logger.warning("eICUPatientExtractor 无 preload_vitals_eicu 方法，生命体征将为空")
        logger.info(f"  预加载完成 ({time.time()-t1:.1f}s)")

    # --- Step 4: 并行提取 ---
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = []
    failed = []
    cached_count = 0

    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False

    def extract_one(subject_id, hadm_id):
        """单个患者的完整 pipeline"""
        # 4a. 检查缓存
        if not force_refresh:
            cached = cache_manager.load(database, subject_id, hadm_id)
            if cached:
                return cached, True  # (data, is_cached)

        # 4b. 原始提取
        data = extractor.extract_complete_patient_data(
            subject_id=subject_id,
            hadm_id=hadm_id,
            lab_window_hours=lab_window_hours,
        )
        if data is None:
            return None, False

        # 4b+. 将旧版 dataclass 转换为新版（补充 bmi/data_quality 等字段）
        data = convert_legacy_patient_data(data)

        # 4c. UMLS 映射（统一实现）
        if umls_mapper:
            try:
                data, _ = apply_umls_mapping(data, umls_mapper, stop_on_permission_error=False)
            except Exception as e:
                logger.debug(f"UMLS mapping error ({subject_id}/{hadm_id}): {e}")

        # 4d. eGFR 计算（CKD-EPI 2021，唯一入口）
        data = calculator.calculate_and_append_egfr(data)

        # 4e. 完整临床指标
        data.clinical_metrics = calculator.calculate_all_metrics(data, levels=[1, 2])

        # 4f. 临床校验（极端值 + 时间窗口 + 数据质量）
        # 注意：validate_all 会初始化 data.data_quality（含 aki_suspected）
        # 必须在 calculate_all_metrics 之后调用，否则 aki_suspected 无法写入 metrics
        data = ClinicalValidator.validate_all(data, baseline_hours=6.0)

        # [FIX-O3] 回填 aki_suspected 和 baseline_creatinine_source：
        # calculate_all_metrics（4e）调用时 data_quality 尚未初始化，
        # metrics['kidney']['aki_suspected'] 因此写入 None 并永久留在缓存中。
        # validate_all（4f）完成后 data_quality 才有值，此处补充回填。
        if data.clinical_metrics and data.data_quality:
            kidney = data.clinical_metrics.get('kidney')
            if isinstance(kidney, dict):
                kidney['aki_suspected'] = data.data_quality.aki_suspected
                kidney['baseline_creatinine_source'] = data.data_quality.baseline_creatinine_source

        # 4g. 保存缓存
        cache_manager.save(data)

        return data, False

    logger.info(f"\n开始并行提取 ({max_workers} workers)...")
    t2 = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_case = {
            executor.submit(extract_one, sid, hid): (sid, hid)
            for sid, hid in cases
        }

        iterator = as_completed(future_to_case)
        if use_tqdm:
            iterator = tqdm(iterator, total=len(cases), desc=f"[{database}]")

        for future in iterator:
            sid, hid = future_to_case[future]
            try:
                data, is_cached = future.result()
                if data:
                    results.append(data)
                    if is_cached:
                        cached_count += 1
                else:
                    failed.append((sid, hid))
            except Exception as e:
                logger.error(f"  Failed ({sid}/{hid}): {e}")
                failed.append((sid, hid))

    elapsed = time.time() - t2

    # --- Step 5: 摘要 ---
    logger.info(f"\n{'='*80}")
    logger.info(f"  {database.upper()} 提取完成")
    logger.info(f"  ─────────────────────────")
    logger.info(f"  总数:     {len(cases)}")
    logger.info(f"  成功:     {len(results)} ({len(results)/max(len(cases),1)*100:.1f}%)")
    logger.info(f"  缓存命中: {cached_count}")
    logger.info(f"  新提取:   {len(results) - cached_count}")
    logger.info(f"  失败:     {len(failed)}")
    logger.info(f"  耗时:     {elapsed:.1f}s ({elapsed/max(len(cases),1):.2f}s/case)")
    logger.info(f"  保存目录: {cache_dir}")
    logger.info(f"{'='*80}\n")

    # 缓存统计
    stats = cache_manager.get_stats()
    for db_name, info in stats.items():
        logger.info(f"  Cache [{db_name}]: {info['count']} files, {info['size_mb']:.1f} MB")

    return results


# ==================== 数据质量报告 ====================

def generate_quality_report(
    results: list,
    database: str,
    output_path: Optional[Path] = None,
):
    """
    生成数据质量报告

    内容：
    - 总体统计（患者数、检验数、诊断数）
    - 关键指标缺失率（HbA1c、肌酐、eGFR、BMI）
    - 极端值统计
    - 糖尿病分型分布
    - 数据完整性评分分布
    """
    from refactored.clinical_validator import ClinicalValidator

    if not results:
        logger.warning("无数据，跳过质量报告")
        return

    n = len(results)
    report = {
        'database': database,
        'total_patients': n,
        'generated_at': datetime.now().isoformat(),
        'egfr_formula': 'CKD-EPI 2021 (race-free)',
    }

    # 关键指标可用率
    hba1c_count = sum(1 for d in results if d.data_quality and d.data_quality.hba1c_available)
    creat_count = sum(1 for d in results if d.data_quality and d.data_quality.creatinine_available)
    egfr_count = sum(1 for d in results if d.data_quality and d.data_quality.egfr_calculated)
    bmi_count = sum(1 for d in results if d.data_quality and d.data_quality.bmi_available)

    report['data_availability'] = {
        'hba1c': f'{hba1c_count}/{n} ({hba1c_count/n*100:.1f}%)',
        'creatinine': f'{creat_count}/{n} ({creat_count/n*100:.1f}%)',
        'egfr_calculated': f'{egfr_count}/{n} ({egfr_count/n*100:.1f}%)',
        'bmi': f'{bmi_count}/{n} ({bmi_count/n*100:.1f}%)',
    }

    # 极端值
    total_extreme = sum(
        d.data_quality.labs_with_extreme_values
        for d in results if d.data_quality
    )
    patients_with_extreme = sum(
        1 for d in results
        if d.data_quality and d.data_quality.labs_with_extreme_values > 0
    )
    report['extreme_values'] = {
        'total_flagged': total_extreme,
        'patients_affected': f'{patients_with_extreme}/{n} ({patients_with_extreme/n*100:.1f}%)',
    }

    # 血糖测量
    glucose_counts = [
        d.data_quality.glucose_measurement_count
        for d in results if d.data_quality
    ]
    if glucose_counts:
        import statistics
        report['glucose_measurements'] = {
            'mean': round(statistics.mean(glucose_counts), 1),
            'median': round(statistics.median(glucose_counts), 1),
            'min': min(glucose_counts),
            'max': max(glucose_counts),
            'patients_with_zero': sum(1 for c in glucose_counts if c == 0),
        }

    # 糖尿病分型
    # [FIX-ALG-7] 优先读取缓存的分型结果（validate_all 阶段已计算并缓存到 data_quality）
    # 避免对所有患者（可能数千例）重复运行 validate_dm_type
    dm_types = Counter()
    dm_confidence = Counter()
    recalc_count = 0
    for d in results:
        # 优先读缓存
        if d.data_quality and d.data_quality.dm_type_cached:
            dm_types[d.data_quality.dm_type_cached] += 1
            dm_confidence[d.data_quality.dm_confidence_cached or 'unknown'] += 1
        elif d.clinical_metrics and 'dm_classification' in d.clinical_metrics:
            # 兼容：从 clinical_metrics 读取（validate_all 写入的位置）
            dm_cls = d.clinical_metrics['dm_classification']
            dm_types[dm_cls.get('dm_type', 'uncertain')] += 1
            dm_confidence[dm_cls.get('confidence', 'unknown')] += 1
        else:
            # 缓存缺失（如旧格式缓存文件），才重新计算
            dm_info = ClinicalValidator.validate_dm_type(d)
            dm_types[dm_info['dm_type']] += 1
            dm_confidence[dm_info['confidence']] += 1
            recalc_count += 1

    if recalc_count > 0:
        logger.debug(f"  质量报告：{recalc_count} 例患者 DM 分型重新计算（缓存缺失）")

    report['dm_classification'] = {
        'type_distribution': dict(dm_types),
        'confidence_distribution': dict(dm_confidence),
    }

    # [FIX-CLN-5] 并发症分布统计（从 clinical_metrics 读取）
    complication_keys = [
        'retinopathy', 'nephropathy', 'neuropathy',
        'peripheral_vascular', 'cardiovascular', 'dka'
    ]
    complication_counts = {k: 0 for k in complication_keys}
    for d in results:
        comp = (d.clinical_metrics or {}).get('dm_complications', {})
        for k in complication_keys:
            if comp.get(k):
                complication_counts[k] += 1

    report['dm_complications'] = {
        k: f'{v}/{n} ({v/n*100:.1f}%)' for k, v in complication_counts.items()
    }

    # [FIX-CLN-2] UACR 可用率
    acr_count = sum(1 for d in results if d.data_quality and d.data_quality.acr_available)
    report['data_availability']['acr_uacr'] = f'{acr_count}/{n} ({acr_count/n*100:.1f}%)'

    # [FIX-CLN-6] HbA1c 时机分布
    hba1c_timing = Counter()
    for d in results:
        timing = d.data_quality.hba1c_timing if d.data_quality else None
        hba1c_timing[timing or 'N/A'] += 1
    report['hba1c_timing_distribution'] = dict(hba1c_timing)

    # [FIX-ALG-5] AKI 疑似统计
    aki_count = sum(1 for d in results if d.data_quality and d.data_quality.aki_suspected)
    report['aki_suspected'] = f'{aki_count}/{n} ({aki_count/n*100:.1f}%)'

    # 数据量统计
    lab_counts = [len(d.all_lab_results) for d in results]
    diag_counts = [len(d.all_diagnoses) for d in results]
    vital_counts = [len(d.all_vitals) for d in results]

    report['data_volume'] = {
        'labs_per_patient': {
            'mean': round(sum(lab_counts)/n, 1),
            'min': min(lab_counts),
            'max': max(lab_counts),
        },
        'diagnoses_per_patient': {
            'mean': round(sum(diag_counts)/n, 1),
            'min': min(diag_counts),
            'max': max(diag_counts),
        },
        'vitals_per_patient': {
            'mean': round(sum(vital_counts)/n, 1),
            'min': min(vital_counts),
            'max': max(vital_counts),
        },
    }

    # UMLS 映射率
    umls_rates = [
        d.data_quality.umls_mapping_rate
        for d in results if d.data_quality
    ]
    if umls_rates:
        import statistics
        report['umls_mapping'] = {
            'mean_rate': f'{statistics.mean(umls_rates)*100:.1f}%',
            'median_rate': f'{statistics.median(umls_rates)*100:.1f}%',
        }

    # 输出
    if output_path is None:
        output_path = Path(f'data/quality_report_{database}_{datetime.now():%Y%m%d}.json')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info(f"\n📊 数据质量报告已保存: {output_path}")

    # 打印摘要
    print(f"\n{'='*60}")
    print(f"  📊 Data Quality Report: {database.upper()}")
    print(f"{'='*60}")
    print(f"  Patients:           {n}")
    print(f"  HbA1c available:    {report['data_availability']['hba1c']}")
    print(f"  Creatinine:         {report['data_availability']['creatinine']}")
    print(f"  eGFR calculated:    {report['data_availability']['egfr_calculated']}")
    print(f"  BMI available:      {report['data_availability']['bmi']}")
    print(f"  Extreme values:     {total_extreme} (in {patients_with_extreme} patients)")
    print(f"  DM types:           {dict(dm_types)}")
    print(f"  DM confidence:      {dict(dm_confidence)}")
    print(f"{'='*60}\n")

    return report


# ==================== 单患者诊断 ====================

def diagnose_patient(database: str, subject_id, hadm_id):
    """对单个患者运行完整诊断并打印详细报告"""
    from refactored.cache_manager import EnhancedCacheManager
    from refactored.clinical_calculator_unified import UnifiedClinicalCalculator
    from refactored.clinical_validator import ClinicalValidator
    from refactored.umls_mapping_unified import apply_umls_mapping

    db_conf = DatabaseConfig.get(database)
    data_path = db_conf['data_path']
    cache_dir = DatabaseConfig.cache_dir(database)

    extractor = create_base_extractor(database, data_path)
    calculator = UnifiedClinicalCalculator(database=database)
    umls_mapper = create_umls_mapper(database, enabled=True)

    # 提取
    data = extractor.extract_complete_patient_data(subject_id, hadm_id)
    if data is None:
        print(f"❌ Patient not found: {subject_id}/{hadm_id}")
        return

    # 转换为新版格式（补充 bmi/data_quality 等字段）
    data = convert_legacy_patient_data(data)

    # UMLS
    if umls_mapper:
        data, stats = apply_umls_mapping(data, umls_mapper)

    # eGFR
    data = calculator.calculate_and_append_egfr(data)
    data.clinical_metrics = calculator.calculate_all_metrics(data)

    # 校验
    data = ClinicalValidator.validate_all(data)
    dm_info = ClinicalValidator.validate_dm_type(data)

    # 打印报告
    print(f"\n{'='*60}")
    print(f"  Patient Diagnosis Report")
    print(f"  {database.upper()}: {subject_id} / {hadm_id}")
    print(f"{'='*60}")
    print(f"  Age:      {data.age}")
    print(f"  Gender:   {data.gender}")
    print(f"  BMI:      {data.bmi or 'N/A'}")
    print(f"  LOS:      {data.los_days} days")
    print(f"  Labs:     {len(data.all_lab_results)}")
    print(f"  Diagnoses:{len(data.all_diagnoses)}")
    print(f"  Vitals:   {len(data.all_vitals)}")
    print(f"  Meds:     {len(data.prior_medications)} prior, {len(data.discharge_medications)} discharge")

    print(f"\n  --- DM Classification ---")
    dm_info = data.clinical_metrics.get('dm_classification', {}) if data.clinical_metrics else {}
    if not dm_info:
        dm_info = ClinicalValidator.validate_dm_type(data)
    print(f"  Type:       {dm_info.get('dm_type')} ({dm_info.get('confidence')})")
    if dm_info.get('lada_flag'):
        print(f"  ⚠️  LADA_possible: 建议检查 GAD 抗体 / C 肽")
    print(f"  Evidence:   {dm_info.get('evidence', [])}")
    if dm_info.get('conflicts'):
        print(f"  Conflicts:  {dm_info.get('conflicts')}")

    # 并发症
    comp = (data.clinical_metrics or {}).get('dm_complications', {})
    if comp:
        active_comps = [k for k in ['retinopathy','nephropathy','neuropathy',
                                     'peripheral_vascular','cardiovascular','dka']
                        if comp.get(k)]
        if active_comps:
            print(f"\n  --- DM Complications ---")
            for c in active_comps:
                print(f"  ✓ {c}")

    if data.clinical_metrics:
        print(f"\n  --- Clinical Metrics ---")
        kidney = data.clinical_metrics.get('kidney', {})
        if kidney.get('data_available'):
            print(f"  Creatinine: {kidney.get('creatinine')} mg/dL"
                  f"  [{kidney.get('baseline_creatinine_source', 'unknown')}]")
            print(f"  eGFR:       {kidney.get('egfr')} mL/min/1.73m² (CKD-EPI 2021)")
            g_stage = kidney.get('ckd_g_stage') or kidney.get('ckd_stage')
            a_stage = kidney.get('ckd_a_stage')
            full_stage = kidney.get('ckd_full_stage') or g_stage
            print(f"  CKD Stage:  G={g_stage}  A={a_stage or 'N/A'}  Full={full_stage}")
            if kidney.get('aki_suspected'):
                print(f"  ⚠️  AKI Suspected (入院肌酐 ≥ 1.5× 住院最低值)")
            if kidney.get('uacr') is not None:
                print(f"  UACR:       {kidney.get('uacr')} mg/g")

        glucose = data.clinical_metrics.get('glucose', {})
        if glucose.get('data_available'):
            print(f"  Glucose:    mean={glucose.get('mean_glucose')}, "
                  f"SD={glucose.get('std_glucose')}, "
                  f"range={glucose.get('min_glucose')}-{glucose.get('max_glucose')} mg/dL")
            # [FIX-CLN-4] 使用正确名称（保留旧名作为向后兼容）
            in_range = glucose.get('glucose_in_range_rate', glucose.get('time_in_range_70_180'))
            freq = glucose.get('glucose_sampling_frequency_per_day')
            freq_str = f" (采样频率={freq}次/天)" if freq else ""
            print(f"  In Range:   {in_range}% 测量点达标率 (70-180 mg/dL){freq_str}")
            print(f"  Hypo (<70): {glucose.get('hypoglycemia_events')} events")
        hba1c = data.clinical_metrics.get('hba1c')
        if hba1c:
            eag = data.clinical_metrics.get('eag_from_hba1c')
            timing = data.data_quality.hba1c_timing if data.data_quality else None
            timing_str = f" [{timing}]" if timing else ""
            print(f"  HbA1c:      {hba1c}%{timing_str}"
                  + (f" → eAG={eag} mg/dL" if eag else ""))

    if data.data_quality:
        q = data.data_quality
        print(f"\n  --- Data Quality ---")
        print(f"  Total labs:       {q.total_labs}")
        print(f"  Extreme values:   {q.labs_with_extreme_values}")
        print(f"  Missing units:    {q.labs_missing_units}")
        print(f"  HbA1c available:  {q.hba1c_available}")
        print(f"  UMLS mapping:     {q.umls_mapping_rate:.1%}")

    print(f"{'='*60}\n")


# ==================== CLI ====================

def main():
    parser = argparse.ArgumentParser(
        description='Dia-MAS 重构后的统一数据提取脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run_extraction.py --database mimic3                    # 提取 MIMIC-III
  python run_extraction.py --database all --workers 8           # 全部数据库, 8线程
  python run_extraction.py --database mimic3 --no-umls          # 跳过 UMLS 映射
  python run_extraction.py --database mimic3 --max-patients 50  # 只提取前50个
  python run_extraction.py --database mimic3 --quality-report   # 提取后生成质量报告
  python run_extraction.py --database mimic3 --diagnose 10094 122928  # 诊断单患者
  python run_extraction.py --database eicu --skip-eicu-vitals          # 跳过eICU生命体征（快速）
        """
    )

    parser.add_argument(
        '--database', '-d',
        choices=['mimic3', 'mimic4', 'eicu', 'all'],
        default='mimic4',
        help='数据库名称，或 "all" 提取全部三个库'
    )
    parser.add_argument(
        '--workers', '-w',
        type=int, default=4,
        help='并行工作线程数 (default: 4)'
    )
    parser.add_argument(
        '--no-umls',
        action='store_true',
        help='跳过 UMLS 映射（加速测试）'
    )
    parser.add_argument(
        '--force-refresh',
        action='store_true',
        help='强制重新提取，忽略缓存'
    )
    parser.add_argument(
        '--min-evidence',
        type=int, default=2,
        help='糖尿病筛选最小证据数 (default: 2)'
    )
    parser.add_argument(
        '--max-patients',
        type=int, default=None,
        help='限制提取患者数（调试用）'
    )
    parser.add_argument(
        '--lab-window',
        type=int, default=48,
        help='检验时间窗口(小时, default: 48)'
    )
    parser.add_argument(
        '--quality-report',
        action='store_true',
        help='提取后生成数据质量报告'
    )
    parser.add_argument(
        '--diagnose',
        nargs=2, metavar=('SUBJECT_ID', 'HADM_ID'),
        help='诊断单个患者（调试用）'
    )
    parser.add_argument(
        '--skip-eicu-vitals',
        action='store_true',
        help=(
            '跳过 eICU vitalPeriodic/vitalAperiodic 扫描（仅对 eICU 有效）。'
            'vitalPeriodic.csv.gz 约 1.5-2 亿行，全量扫描需 20-60 分钟。'
            '生命体征是辅助数据，跳过后 ICU 生命体征将为空列表，'
            '核心临床指标（lab/diagnosis/medication/eGFR）不受影响。'
            '建议首次全量提取时加此参数，后续按需重跑补充生命体征。'
        )
    )
    parser.add_argument(
        '--patient_list',
        type=Path,
        default=None,
        metavar='CSV_PATH',
        help=(
            '覆盖默认患者列表 CSV（可选）。\n'
            '指定后完全跳过 DatabaseConfig 中配置的 patient_list_csv 路径，\n'
            '直接使用该文件作为提取输入。\n'
            '格式与 diabetes_subsets/mimiciv_patient_list.csv 相同：\n'
            '  subject_id,hadm_ids\n'
            '  10000032,"[20141430]"\n'
            '主要用途：Era-2 补充提取\n'
            '  python run_extraction.py \\\n'
            '      --database mimic4 \\\n'
            '      --patient_list diabetes_subsets/mimiciv_patient_list_era2.csv\n'
            '新患者直接追加进现有缓存目录，已有 Era-3 缓存不会被重新提取。'
        ),
    )

    args = parser.parse_args()

    # 单患者诊断模式
    if args.diagnose:
        sid, hid = args.diagnose
        try:
            sid = int(sid)
        except ValueError:
            pass  # eICU 可能是字符串
        diagnose_patient(args.database, sid, int(hid))
        return

    # 确定要处理的数据库列表
    databases = ['mimic3', 'mimic4', 'eicu'] if args.database == 'all' else [args.database]

    total_start = time.time()
    all_results = {}

    for db in databases:
        try:
            results = run_extraction(
                database=db,
                max_workers=args.workers,
                enable_umls=not args.no_umls,
                force_refresh=args.force_refresh,
                min_evidence=args.min_evidence,
                max_patients=args.max_patients,
                lab_window_hours=args.lab_window,
                skip_eicu_vitals=args.skip_eicu_vitals,
                patient_list_override=args.patient_list,
            )
            all_results[db] = results

            if args.quality_report and results:
                generate_quality_report(results, db)

        except Exception as e:
            logger.error(f"❌ {db} extraction failed: {e}", exc_info=True)
            all_results[db] = []

    # 总结
    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  🎯 ALL EXTRACTIONS COMPLETE")
    print(f"  Total time: {total_elapsed:.1f}s")
    for db, res in all_results.items():
        print(f"  {db:8s}: {len(res)} patients")
    print(f"  Output:    {DatabaseConfig.CACHE_BASE}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
