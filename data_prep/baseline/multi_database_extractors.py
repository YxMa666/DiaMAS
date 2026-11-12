#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Database Patient Extractors

为MIMIC-IV和eICU提供专用提取器，并解决批量处理时的大文件重复读取问题。

核心变更：
1. BatchDataLoader：大文件只扫描一次，按ID分组缓存，后续查询O(1)
2. MIMIC4PatientExtractor._extract_all_vitals：完整实现
   - 通过icu/icustays做hadm_id→stay_id映射
   - 从icu/chartevents读取生命体征
   - 合并icu/d_items获取标签
3. eICUPatientExtractor._extract_all_labs_eicu：单位和时间修复
   - 使用labunits字段（之前硬编码为空）
   - 使用laboffset做时间窗口过滤和时间记录
4. eICUPatientExtractor._extract_all_vitals_eicu：新增实现
   - 从vitalPeriodic表读取连续监控数据

作者：Yuxi
日期：2025-02-03
"""

import sys
import time
import threading
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass

# 导入基础提取器
sys.path.append(str(Path(__file__).parent))
from semantic_patient_extractor import (
    SemanticPatientExtractor,
    CompletePatientData,
    LabResult,
    Diagnosis,
    VitalSign
)

# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from baseline.batch_data_loader import BatchDataLoader

# ==================== MIMIC-IV专用提取器 ====================

class MIMIC4PatientExtractor(SemanticPatientExtractor):
    """
    MIMIC-IV数据提取器
    
    关键差异：
        - 年龄: anchor_age + anchor_year，而非dob
        - ICD: 同时包含ICD-9和ICD-10，字段为icd_code + icd_version
        - 目录: hosp/ 放医院级别表，icu/ 放ICU级别表
        - 生命体征: 在icu/chartevents.csv.gz中，以stay_id为键（非hadm_id）
          需通过icu/icustays做hadm_id→stay_id映射
    """
    
    def __init__(self, data_path: Path, batch_loader: Optional[BatchDataLoader] = None):
        self.database = "mimic4"
        self.data_path = Path(data_path)
        # 如果外部没传batch_loader，自建一个（单独patient模式也能用）
        self.batch_loader = batch_loader if batch_loader is not None else BatchDataLoader()
        # [FIX-A1] 懒加载 chartevents 时防止多线程重复扫描同一文件的锁
        self._chartevents_scan_lock = threading.Lock()
        logger.info(f"初始化MIMIC-IV提取器: {data_path}")
        self._load_tables()
    
    def _load_tables(self):
        """加载所有需要的表。大文件（labevents/chartevents）不在这里加载。"""
        try:
            # --- hosp/ 层 ---
            self.patients = pd.read_csv(
                self.data_path / "hosp" / "patients.csv.gz",
                usecols=["subject_id", "gender", "anchor_age", "anchor_year",
                         "anchor_year_group"],
            )
            self.admissions = pd.read_csv(
                self.data_path / "hosp" / "admissions.csv.gz",
                usecols=["subject_id", "hadm_id", "admittime", "dischtime", "admission_type"],
                parse_dates=["admittime", "dischtime"],
            )
            self.diagnoses = pd.read_csv(
                self.data_path / "hosp" / "diagnoses_icd.csv.gz",
                usecols=["subject_id", "hadm_id", "icd_code", "icd_version", "seq_num"],
            )
            self.d_icd = pd.read_csv(
                self.data_path / "hosp" / "d_icd_diagnoses.csv.gz",
                usecols=["icd_code", "icd_version", "long_title"],
            )
            self.d_labitems = pd.read_csv(
                self.data_path / "hosp" / "d_labitems.csv.gz",
                usecols=["itemid", "label", "fluid", "category"],
            )
            self.prescriptions = pd.read_csv(
                self.data_path / "hosp" / "prescriptions.csv.gz",
                usecols=["subject_id", "hadm_id", "drug", "starttime", "stoptime"],
                parse_dates=["starttime", "stoptime"],
            )

            # --- icu/ 层（生命体征相关） ---
            # icustays: hadm_id ↔ stay_id 映射
            icustays_path = self.data_path / "icu" / "icustays.csv.gz"
            if icustays_path.exists():
                self.icustays = pd.read_csv(
                    icustays_path,
                    usecols=["subject_id", "stay_id", "hadm_id", "intime", "outtime"],
                    parse_dates=["intime", "outtime"],
                )
            else:
                self.icustays = pd.DataFrame()
                logger.warning("icu/icustays.csv.gz 不存在，生命体征将无法提取")

            # d_items: ICU事件项目字典（标签+单位）
            d_items_path = self.data_path / "icu" / "d_items.csv.gz"
            if d_items_path.exists():
                self.d_items = pd.read_csv(
                    d_items_path,
                    usecols=["itemid", "label", "category", "unitname"],
                )
            else:
                self.d_items = pd.DataFrame()
                logger.warning("icu/d_items.csv.gz 不存在")

            # --- O(1) 按 ID 查询索引（与 MIMIC-III 保持一致）---
            # 用 dict-of-DataFrame 代替每次全表 boolean mask（O(n) → O(1)）
            # MIMIC-IV patients/admissions 均为十万~百万行级别，
            # 不加索引时每个患者 4 次 O(n) 扫描，几千患者就是数百万次扫描。
            self._patients_by_subj      = {
                k: v for k, v in self.patients.groupby("subject_id")
            }
            self._admissions_by_hadm    = {
                k: v for k, v in self.admissions.groupby("hadm_id")
            }
            self._diagnoses_by_hadm     = {
                k: v for k, v in self.diagnoses.groupby("hadm_id")
            }
            self._prescriptions_by_hadm = {
                k: v for k, v in self.prescriptions.groupby("hadm_id")
            }
            # icustays 索引：hadm_id → stay rows（用于 _extract_all_vitals 的 hadm→stay 映射）
            if not self.icustays.empty:
                self._icustays_by_hadm = {
                    k: v for k, v in self.icustays.groupby("hadm_id")
                }
            else:
                self._icustays_by_hadm = {}

            logger.info(
                f"  - 索引构建完成: {len(self._admissions_by_hadm)} 次住院 / "
                f"{len(self._patients_by_subj)} 个患者"
            )
            logger.info("✅ MIMIC-IV数据表加载完成")
        except Exception as e:
            logger.error(f"加载MIMIC-IV数据表失败: {e}")
            raise
    
    # ---- 批量预加载入口 ----

    def preload_batch(self, hadm_ids: List[int], preload_vitals: bool = False):
        """
        批量处理前调用一次。把大文件按目标ID预筛选装进内存。

        之后对这些hadm_id调用extract_complete_patient_data，
        内部的_extract_all_labs和_extract_all_vitals会自动走缓存路径。

        Args:
            hadm_ids: 本次要处理的所有入院ID
            preload_vitals: 是否预加载 chartevents（ICU生命体征）。
                默认 False。
                chartevents.csv.gz 是 MIMIC-IV 最大的文件（5-8GB压缩），
                全量扫描需 30分钟~3小时，而生命体征对糖尿病用药推荐属于
                辅助数据，不预加载则按需懒加载（单患者~秒级）。
                仅在明确需要大批量生命体征时才设为 True。
        """
        if hadm_ids and isinstance(hadm_ids[0], tuple):
            target_set = set(hadm_id for _, hadm_id in hadm_ids)
        else:
            target_set = set(hadm_ids)

        # 1. labevents → 按hadm_id分组（必须预加载，检验值是核心数据）
        labevents_path = self.data_path / "hosp" / "labevents.csv.gz"
        if labevents_path.exists():
            self.batch_loader.load_filtered(
                file_path=labevents_path,
                id_column="hadm_id",
                target_ids=target_set,
                usecols=["subject_id", "hadm_id", "itemid", "charttime", "valuenum", "valueuom"],
                parse_dates=["charttime"],
            )

        # 2. chartevents → 默认跳过（文件巨大，生命体征为辅助数据）
        # [FIX-F1] 原代码无条件扫描 chartevents.csv.gz（5-8GB），
        #   对 ~63K stay_id 扫描需 30分钟~3小时，导致命令卡死。
        #   修复：默认 preload_vitals=False，走懒加载路径（_extract_all_vitals fallback）。
        if preload_vitals:
            chartevents_path = self.data_path / "icu" / "chartevents.csv.gz"
            if chartevents_path.exists() and not self.icustays.empty:
                target_stay_ids = set(
                    self.icustays.loc[self.icustays["hadm_id"].isin(target_set), "stay_id"].unique()
                )
                if target_stay_ids:
                    logger.info(
                        f"  [preload_vitals=True] 扫描 chartevents.csv.gz "
                        f"({len(target_stay_ids)} stay_ids) — 预计耗时较长..."
                    )
                    self.batch_loader.load_filtered(
                        file_path=chartevents_path,
                        id_column="stay_id",
                        target_ids=target_stay_ids,
                        usecols=["subject_id", "stay_id", "itemid", "charttime", "valuenum", "valueuom"],
                        parse_dates=["charttime"],
                    )
        else:
            logger.info(
                "  [preload_vitals=False] 跳过 chartevents 预加载（生命体征将按需懒加载）"
            )

        # [FIX-LAZY] 记录目标患者对应的 stay_ids，供懒加载时精准过滤。
        # 修复前：懒加载扫描全库所有 stay_id（MIMIC-IV 全量 ~6.3 万条），
        #         导致 chartevents.csv.gz（5-8GB）扫描耗时 30 分钟~3 小时。
        # 修复后：只扫目标患者的 stay_ids（通常几百~几千条），扫描降至 2-10 分钟。
        if not self.icustays.empty:
            self._target_stay_ids: set = set(
                self.icustays.loc[
                    self.icustays["hadm_id"].isin(target_set), "stay_id"
                ].dropna().astype(int).tolist()
            )
            logger.info(
                f"  [FIX-LAZY] 目标 stay_ids 缓存: {len(self._target_stay_ids)} 个"
                f"（来自 {len(target_set)} 个目标 hadm_id）"
            )
        else:
            self._target_stay_ids = set()

        logger.info(
            f"✅ MIMIC-IV预加载完成 | 缓存内存: {self.batch_loader.memory_usage_mb():.1f}MB | "
            f"统计: {self.batch_loader.get_stats()}"
        )
    
    def _extract_demographics(self, subject_id: int, hadm_id: int) -> Dict:
        """人口统计。MIMIC-IV无dob，用anchor_age + anchor_year推算。"""
        # O(1) dict 索引（_load_tables 已构建）；回退到全表扫描（兼容旧路径）
        if hasattr(self, '_patients_by_subj'):
            patient = self._patients_by_subj.get(subject_id, pd.DataFrame())
        else:
            patient = self.patients[self.patients["subject_id"] == subject_id]
        if patient.empty:
            return {"age": None, "gender": "unknown"}

        if hasattr(self, '_admissions_by_hadm'):
            admission = self._admissions_by_hadm.get(hadm_id, pd.DataFrame())
        else:
            admission = self.admissions[self.admissions["hadm_id"] == hadm_id]
        if admission.empty:
            return {"age": None, "gender": patient.iloc[0]["gender"]}

        p = patient.iloc[0]
        a = admission.iloc[0]
        age = float(p["anchor_age"]) + (a["admittime"].year - int(p["anchor_year"]))

        return {
            "age": age,
            "gender": p["gender"],
            "admission_time": a["admittime"].strftime("%Y-%m-%d %H:%M:%S"),
            "discharge_time": a["dischtime"].strftime("%Y-%m-%d %H:%M:%S"),
            "los_days": (a["dischtime"] - a["admittime"]).days,
            # [FIX-ERA] anchor_year_group 存储患者真实临床年份分组（未经日期偏移）。
            # MIMIC-IV admittime 经去标识化偏移约+100年（真实2016→存储~2165），
            # 导致 classify_era() 用 admittime 年份判断 Era 完全失效。
            # 该字段写入 JSON 缓存后，classify_era() 使用它正确区分 Era-1/2/3。
            "anchor_year_group": str(p.get("anchor_year_group", "") or "").strip(),
            # 移位后的 anchor_year：与 admission_time 的年份差用于推算每次入院的真实年份
            "anchor_year": int(p["anchor_year"]),
        }
    
    def _extract_all_diagnoses(self, hadm_id: int) -> List[Diagnosis]:
        """提取诊断。MIMIC-IV包含ICD-9和ICD-10。"""
        if hasattr(self, '_diagnoses_by_hadm'):
            rows = self._diagnoses_by_hadm.get(hadm_id, pd.DataFrame()).copy()
        else:
            rows = self.diagnoses[self.diagnoses["hadm_id"] == hadm_id].copy()
        if rows.empty:
            return []

        rows = rows.merge(self.d_icd, on=["icd_code", "icd_version"], how="left")

        result = []
        for _, r in rows.iterrows():
            result.append(Diagnosis(
                icd_code=str(r["icd_code"]),
                icd_version=int(r.get("icd_version", 10)),
                description=str(r.get("long_title", "Unknown")),
                seq_num=int(r.get("seq_num", 0)),
            ))
        return sorted(result, key=lambda d: d.seq_num)
    
    def _extract_all_labs(self, hadm_id: int, window_hours: int = 48) -> List[LabResult]:
        """
        提取实验室检查。

        读取策略：
            - 如果已preload_batch → 直接从缓存拿（O(1)）
            - 否则 → 逐块扫描文件（单患者模式fallback）
        """
        if hasattr(self, '_admissions_by_hadm'):
            admission = self._admissions_by_hadm.get(hadm_id, pd.DataFrame())
        else:
            admission = self.admissions[self.admissions["hadm_id"] == hadm_id]
        if admission.empty:
            return []
        admittime = admission.iloc[0]["admittime"]
        cutoff = admittime + pd.Timedelta(hours=window_hours)

        labevents_path = self.data_path / "hosp" / "labevents.csv.gz"

        # --- 路径A：缓存命中 ---
        if self.batch_loader.has(labevents_path):
            labs_df = self.batch_loader.get(labevents_path, hadm_id)
            if labs_df is None:
                return []
            labs_df = labs_df[labs_df["charttime"] <= cutoff].copy()

        # --- 路径B：逐块扫描（单患者fallback） ---
        else:
            collected = []
            for chunk in pd.read_csv(
                labevents_path,
                usecols=["subject_id", "hadm_id", "itemid", "charttime", "valuenum", "valueuom"],
                parse_dates=["charttime"],
                chunksize=500000,
                low_memory=False,
            ):
                sub = chunk[(chunk["hadm_id"] == hadm_id) & (chunk["charttime"] <= cutoff)]
                if not sub.empty:
                    collected.append(sub)
            if not collected:
                return []
            labs_df = pd.concat(collected, ignore_index=True)

        # 合并标签
        labs_df = labs_df.merge(self.d_labitems[["itemid", "label"]], on="itemid", how="left")

        results = []
        for _, r in labs_df.iterrows():
            if pd.isna(r["valuenum"]):
                continue
            results.append(LabResult(
                itemid=int(r["itemid"]),
                label=str(r.get("label") or f"ITEMID_{int(r['itemid'])}"),
                loinc_code=None,  # MIMIC-IV d_labitems无loinc_code列
                value=float(r["valuenum"]),
                unit=str(r.get("valueuom") or ""),
                charttime=str(r["charttime"]),
                is_abnormal=False,
            ))
        return results
    
    def _extract_all_vitals(self, hadm_id: int) -> List[VitalSign]:
        """
        提取生命体征（MIMIC-IV完整实现）。

        数据流程：
            hadm_id
              → icu/icustays 查找 stay_id（不存在则无ICU记录，返回空）
              → icu/chartevents 按 stay_id 过滤（走缓存或逐块扫描）
              → icu/d_items 合并标签和单位
              → 构建 VitalSign 列表

        时间窗口：ICU入院后24小时。
        """
        if self.icustays.empty or self.d_items.empty:
            return []

        # Step1: hadm_id → stay_id
        if hasattr(self, '_icustays_by_hadm'):
            icu_rows = self._icustays_by_hadm.get(hadm_id, pd.DataFrame())
        else:
            icu_rows = self.icustays[self.icustays["hadm_id"] == hadm_id]
        if icu_rows.empty:
            # 该入院无ICU住院记录，正常情况
            logger.debug(f"  hadm_id={hadm_id} 无ICU住院记录")
            return []

        # 取第一次ICU入院（多次ICU入院取最早的）
        icu_rows = icu_rows.sort_values("intime")
        stay_id = int(icu_rows.iloc[0]["stay_id"])
        icu_intime = icu_rows.iloc[0]["intime"]
        cutoff = icu_intime + pd.Timedelta(hours=24)

        chartevents_path = self.data_path / "icu" / "chartevents.csv.gz"

        # --- 路径A：缓存命中 ---
        if self.batch_loader.has(chartevents_path):
            vitals_df = self.batch_loader.get(chartevents_path, stay_id)
            if vitals_df is None:
                return []
            vitals_df = vitals_df[vitals_df["charttime"] <= cutoff].copy()

        # --- 路径B：懒加载（第一个到达的 worker 扫描，结果写入 batch_loader 供后续复用）---
        else:
            if not chartevents_path.exists():
                logger.debug("  icu/chartevents.csv.gz 不存在")
                return []

            # [FIX-A1] 加锁：只让第一个 worker 真正扫描文件，其余阻塞等待。
            # 锁释放后，后来的 worker 检查 batch_loader.has() 已为 True，走路径A。
            with self._chartevents_scan_lock:
                # double-check：可能在等锁期间已被别的 worker 扫完
                if self.batch_loader.has(chartevents_path):
                    vitals_df = self.batch_loader.get(chartevents_path, stay_id)
                    if vitals_df is None:
                        return []
                    vitals_df = vitals_df[vitals_df["charttime"] <= cutoff].copy()
                else:
                    # [FIX-LAZY] 使用 preload_batch 记录的目标 stay_ids，而非全库 stay_ids。
                    # 修复前：all_stay_ids = 全库 ~6.3 万 stay_id → 扫描耗时 30 分钟~3 小时
                    # 修复后：_target_stay_ids = 目标患者 stay_ids（几百~几千）→ 2-10 分钟
                    if hasattr(self, "_target_stay_ids") and self._target_stay_ids:
                        all_stay_ids = self._target_stay_ids
                        logger.info(
                            f"  [懒加载][FIX-LAZY] 首次扫描 chartevents.csv.gz，"
                            f"目标 {len(all_stay_ids)} 个 stay_id（仅目标患者）..."
                        )
                    elif not self.icustays.empty:
                        # 降级：preload_batch 未调用时（单患者模式）扫全库
                        all_stay_ids = set(self.icustays["stay_id"].dropna().astype(int).tolist())
                        logger.info(
                            f"  [懒加载][降级] _target_stay_ids 未设置，"
                            f"扫描全库 {len(all_stay_ids)} stay_ids（耗时较长）..."
                        )
                    else:
                        all_stay_ids = {stay_id}
                        logger.info(
                            f"  [懒加载][降级] icustays 为空，仅扫 stay_id={stay_id}"
                        )

                    self.batch_loader.load_filtered(
                        file_path=chartevents_path,
                        id_column="stay_id",
                        target_ids=all_stay_ids,
                        usecols=["subject_id", "stay_id", "itemid", "charttime", "valuenum", "valueuom"],
                        parse_dates=["charttime"],
                    )
                    vitals_df = self.batch_loader.get(chartevents_path, stay_id)
                    if vitals_df is None:
                        return []
                    vitals_df = vitals_df[vitals_df["charttime"] <= cutoff].copy()

        # 合并d_items标签
        vitals_df = vitals_df.merge(
            self.d_items[["itemid", "label", "unitname"]], on="itemid", how="left"
        )

        results = []
        for _, r in vitals_df.iterrows():
            if pd.isna(r.get("valuenum")):
                continue
            results.append(VitalSign(
                itemid=int(r["itemid"]),
                label=str(r.get("label") or f"ITEMID_{int(r['itemid'])}"),
                value=float(r["valuenum"]),
                # valueuom优先，无则用d_items的unitname
                unit=str(r.get("valueuom") or r.get("unitname") or ""),
                charttime=str(r["charttime"]),
            ))

        logger.debug(f"  提取了 {len(results)} 个生命体征 (stay_id={stay_id})")
        return results
 
    def _extract_medications(self, hadm_id: int) -> Tuple[List[str], List[str]]:
        """
        提取用药信息（MIMIC-IV版本）
        
        MIMIC-IV的用药在hosp/prescriptions中
        """
        if hasattr(self, '_prescriptions_by_hadm'):
            hadm_prescriptions = self._prescriptions_by_hadm.get(hadm_id, pd.DataFrame())
        else:
            hadm_prescriptions = self.prescriptions[
                self.prescriptions['hadm_id'] == hadm_id
            ]
        
        if hadm_prescriptions.empty:
            return [], []
        
        # 获取入院时间
        if hasattr(self, '_admissions_by_hadm'):
            admission = self._admissions_by_hadm.get(hadm_id, pd.DataFrame())
        else:
            admission = self.admissions[self.admissions['hadm_id'] == hadm_id]
        if admission.empty:
            return [], []
        
        admittime = admission.iloc[0]['admittime']
        dischtime = admission.iloc[0]['dischtime']
        
        # 入院前用药（starttime在入院前）
        prior_meds = hadm_prescriptions[
            hadm_prescriptions['starttime'] < admittime
        ]['drug'].unique().tolist()
        
        # 出院用药（stoptime在出院时或之后）
        discharge_meds = hadm_prescriptions[
            (hadm_prescriptions['stoptime'].isna()) |
            (hadm_prescriptions['stoptime'] >= dischtime)
        ]['drug'].unique().tolist()
        
        return prior_meds, discharge_meds

# ==================== eICU专用提取器 ====================

class eICUPatientExtractor(SemanticPatientExtractor):
    """
    eICU数据提取器
    
    关键差异：
        - ID: patientunitstayid（对应hadm_id）、patienthealthsystemstayid（对应subject_id）
        - 时间: 没有绝对时间戳，用offset（入院后分钟数）表示
        - Lab单位: labunits字段（之前代码漏用了）
        - 生命体征: vitalPeriodic表
        - 已全部装内存: lab/diagnosis/medication表不大，
        _load_tables时就读完了，不需要BatchDataLoader
    """
    
    def __init__(self, data_path: Path, batch_loader: Optional[BatchDataLoader] = None):
        self.database = "eicu"
        self.data_path = Path(data_path)
        self.batch_loader = batch_loader  # eICU本身不需要，但接口一致
        logger.info(f"初始化eICU提取器: {data_path}")
        self._load_tables()
    
    def _load_tables(self):
        """加载eICU数据表。"""
        try:
            self.patients = pd.read_csv(
                self.data_path / "patient.csv.gz",
                usecols=[
                    "patienthealthsystemstayid", "patientunitstayid",
                    "gender", "age", "ethnicity",
                    "unitadmittime24", "unitdischargetime24",
                    "unitdischargeoffset",   # ICU 住院时长（分钟），用于计算 los_days
                    "hospitaldischargeyear",
                    "admissionheight",       # cm，用于 BMI 计算
                    "admissionweight",       # kg，用于 BMI 计算
                ],
            )

            self.diagnosis = pd.read_csv(
                self.data_path / "diagnosis.csv.gz",
                usecols=["patientunitstayid", "diagnosisstring", "icd9code", "diagnosisoffset"],
            )

            # lab.csv.gz 实际列名：
            #   单位在 labmeasurenamesystem（如 "mg/dL"、"ratio"）
            #   时间在 labresultoffset（入院后分钟数）
            #   不存在 labunits / laboffset
            self.lab = pd.read_csv(
                self.data_path / "lab.csv.gz",
                usecols=[
                    "patientunitstayid",
                    "labname",
                    "labresult",
                    "labresultoffset",          # 入院后分钟数（实际列名）
                    "labmeasurenamesystem",     # 单位（实际列名）
                ],
            )

            self.medication = pd.read_csv(
                self.data_path / "medication.csv.gz",
                usecols=[
                    "patientunitstayid", "drugname", "dosage",
                    "routeadmin", "drugstartoffset", "drugstopoffset",
                ],
            )

            # --- O(1) 按 patientunitstayid 查询索引 ---
            # eICU 的 lab/diagnosis/medication 表均以 patientunitstayid 为键，
            # 行数分别在百万~千万级别，不加索引时每患者 4 次 O(n) 扫描。
            self._patients_by_stay  = {
                k: v for k, v in self.patients.groupby("patientunitstayid")
            }
            self._diagnosis_by_stay = {
                k: v for k, v in self.diagnosis.groupby("patientunitstayid")
            }
            self._lab_by_stay       = {
                k: v for k, v in self.lab.groupby("patientunitstayid")
            }
            self._medication_by_stay = {
                k: v for k, v in self.medication.groupby("patientunitstayid")
            }
            logger.info(
                f"  - 索引构建完成: {len(self._patients_by_stay)} 个 ICU stay"
            )

            # [FIX-F2] vitalPeriodic / vitalAperiodic 全量加载移除
            # ──────────────────────────────────────────────────────
            # 原问题：
            #   vitalPeriodic.csv.gz 在完整 eICU 数据集中有 ~1.5-2 亿行（5分钟级ICU监护）
            #   vitalAperiodic.csv.gz 亦有数千万行
            #   pd.read_csv() 全量读取需要 20-60 分钟，期间无任何日志输出 → "静默卡死"
            #
            # 修复策略：
            #   ① _load_tables() 阶段：仅初始化为空 DataFrame（亚秒级完成）
            #   ② 在知道目标 patientunitstayid 后，调用 preload_vitals_eicu(target_ids)
            #      用分块扫描过滤出目标患者的数据（通常只需 2-5 分钟）
            #   ③ 若 preload_vitals_eicu 未调用，_extract_all_vitals_eicu 将返回空列表
            #      （ICU 生命体征是辅助数据，糖尿病用药推荐的核心依赖检验值和诊断）
            self.vitalperiodic = pd.DataFrame()
            self.vitalaperiodic = pd.DataFrame()
            logger.info(
                "  vitalPeriodic/vitalAperiodic: 延迟加载模式 "
                "（调用 preload_vitals_eicu(target_ids) 后才加载）"
            )

            logger.info("✅ eICU数据表加载完成")
        except Exception as e:
            logger.error(f"加载eICU数据表失败: {e}")
            raise

    # ---- 生命体征延迟加载接口 ----

    def preload_vitals_eicu(self, target_ids: set, skip: bool = False):
        """
        [FIX-F2] 按需加载 eICU 生命体征，仅保留目标患者数据。

        调用时机：在知道所有目标 patientunitstayid 后调用一次。
        通常在 ExtractionPipeline.preload_batch() 中调用。

        分块扫描策略：
            chunksize=500_000 行/次，每次只保留目标ID的行。
            对完整 eICU (1.5亿行 vitalPeriodic)，扫描约需 20-60 分钟。
            生命体征是辅助数据，不影响核心临床指标（lab/diagnosis/medication）。
            传 skip=True 可完全跳过，提取速度提升 10x。

        Args:
            target_ids: 目标 patientunitstayid 集合
            skip:        True = 跳过生命体征加载（vitalperiodic/vitalaperiodic 保持空 DataFrame）
        """
        if skip:
            logger.info(
                "  [preload_vitals_eicu] skip=True，跳过 vitalPeriodic/vitalAperiodic 扫描。"
                "后续 _extract_all_vitals_eicu 将返回空列表（生命体征为辅助数据，不影响核心指标）。"
            )
            return
        PERIODIC_COLS = [
            "patientunitstayid", "observationoffset",
            "temperature", "sao2", "heartrate", "respiration", "cvp", "etco2",
            "systemicsystolic", "systemicdiastolic", "systemicmean",
            "pasystolic", "padiastolic", "pamean", "st1", "st2", "st3", "icp",
        ]
        APERIODIC_COLS = [
            "patientunitstayid", "observationoffset",
            "noninvasivesystolic", "noninvasivediastolic", "noninvasivemean",
            "paop", "cardiacoutput", "cardiacinput", "svr", "svri", "pvr", "pvri",
        ]

        def _scan_filtered(path, usecols, target_ids):
            """分块扫描，只保留 target_ids 对应行（含进度日志）"""
            if not path.exists():
                logger.warning(f"  {path.name} 不存在，跳过")
                return pd.DataFrame()
            chunks = []
            rows_read = rows_kept = chunk_idx = 0
            # vitalPeriodic.csv.gz 约 1.5-2 亿行，500K/块 = 300-400 个 chunk
            # 每 50 块打一次进度（约 2500 万行/次），让用户确认程序在运行
            PROGRESS_EVERY = 50
            logger.info(
                f"  [扫描开始] {path.name} | "
                f"目标 {len(target_ids)} 个 stay_id | "
                f"每 {PROGRESS_EVERY * 500_000 // 10000}万 行报告一次进度..."
            )
            import time as _time
            t_start = _time.time()
            for chunk in pd.read_csv(path, usecols=usecols, chunksize=500_000, low_memory=False):
                chunk_idx += 1
                rows_read += len(chunk)
                sub = chunk[chunk["patientunitstayid"].isin(target_ids)]
                if not sub.empty:
                    chunks.append(sub)
                    rows_kept += len(sub)
                # 进度日志：每 PROGRESS_EVERY 块 或 首块完成时打印
                if chunk_idx == 1 or chunk_idx % PROGRESS_EVERY == 0:
                    elapsed = _time.time() - t_start
                    rate = rows_read / elapsed if elapsed > 0 else 0
                    logger.info(
                        f"  [进度] {path.name}: "
                        f"已扫 {rows_read:,} 行 ({chunk_idx} 块) | "
                        f"保留 {rows_kept:,} 行 | "
                        f"速度 {rate/1e6:.1f}M行/s | "
                        f"已用 {elapsed:.0f}s"
                    )
            total_elapsed = _time.time() - t_start
            logger.info(
                f"  [扫描完成] {path.name}: "
                f"共 {rows_read:,} 行, 保留 {rows_kept:,} 行 "
                f"({len(target_ids)} 个目标患者), 耗时 {total_elapsed:.0f}s"
            )
            return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()

        logger.info(f"开始 eICU 生命体征延迟加载 (target={len(target_ids)} patients)...")
        import time
        t0 = time.time()

        self.vitalperiodic = _scan_filtered(
            self.data_path / "vitalPeriodic.csv.gz",
            usecols=PERIODIC_COLS,
            target_ids=target_ids,
        )
        self.vitalaperiodic = _scan_filtered(
            self.data_path / "vitalAperiodic.csv.gz",
            usecols=APERIODIC_COLS,
            target_ids=target_ids,
        )
        logger.info(f"✅ eICU 生命体征加载完成，耗时 {time.time() - t0:.1f}s")

    # ---- 完整提取入口 ----

    def extract_complete_patient_data(
        self,
        subject_id: int,
        hadm_id: int,
        lab_window_hours: int = 48,
    ) -> Optional[CompletePatientData]:
        """
        提取eICU患者完整数据。

        参数映射:
            subject_id → patienthealthsystemstayid
            hadm_id    → patientunitstayid

        修复记录：
          [FIX-eICU-1] los_days：从 unitdischargeoffset（分钟）换算，而非硬编码 None
          [FIX-eICU-2] admission_time：构造 "{year}-01-01 00:00:00" 占位符，
                        避免存入纯时分字符串 "14:30:00" 误导下游计算
          [FIX-eICU-3] height_cm / weight_kg / ethnicity：从 patient 表直接读取
        """
        try:
            patientunitstayid = hadm_id

            if hasattr(self, '_patients_by_stay'):
                row_df = self._patients_by_stay.get(patientunitstayid, pd.DataFrame())
            else:
                row_df = self.patients[self.patients["patientunitstayid"] == patientunitstayid]
            if row_df.empty:
                logger.warning(f"未找到患者: patientunitstayid={patientunitstayid}")
                return None
            row = row_df.iloc[0]

            # 年龄（可能是字符串 ">89"）
            age_raw = str(row.get("age", ""))
            if ">" in age_raw:
                age = 90.0
            else:
                try:
                    age = float(age_raw)
                except (ValueError, TypeError):
                    age = None

            gender = str(row.get("gender", "Unknown"))
            ethnicity = str(row.get("ethnicity", "")) or None

            # [FIX-eICU-3] 身高体重
            # eICU 的 admissionheight 单位标称是 cm，但少量记录实为英寸（inches）
            # 成人合理身高范围：100–250 cm；体重：20–300 kg；BMI：10–70
            try:
                height_cm = float(row.get("admissionheight") or 0) or None
                if height_cm is not None and not (100 <= height_cm <= 250):
                    logger.debug(f"  身高异常值 {height_cm}cm，已清除（patientunitstayid={patientunitstayid}）")
                    height_cm = None
            except (ValueError, TypeError):
                height_cm = None
            try:
                weight_kg = float(row.get("admissionweight") or 0) or None
                if weight_kg is not None and not (20 <= weight_kg <= 300):
                    logger.debug(f"  体重异常值 {weight_kg}kg，已清除（patientunitstayid={patientunitstayid}）")
                    weight_kg = None
            except (ValueError, TypeError):
                weight_kg = None
            bmi = None
            if height_cm and weight_kg:
                raw_bmi = weight_kg / (height_cm / 100) ** 2
                if 10 <= raw_bmi <= 70:
                    bmi = round(raw_bmi, 1)
                else:
                    logger.debug(f"  BMI 异常值 {raw_bmi:.1f}，已清除（height={height_cm}, weight={weight_kg}）")

            # [FIX-eICU-1] LOS：unitdischargeoffset 是入院后分钟数
            try:
                discharge_offset_min = float(row.get("unitdischargeoffset") or 0)
                los_days = round(discharge_offset_min / 1440, 2) if discharge_offset_min > 0 else None
            except (ValueError, TypeError):
                los_days = None

            # [FIX-eICU-2] 时间戳：eICU 只有年份，用 "{year}-01-01 00:00:00" 作占位
            # 这样下游 classify_time_windows 不会因解析失败而静默报错
            discharge_year = row.get("hospitaldischargeyear")
            try:
                year = int(discharge_year)
                admission_time_str = f"{year}-01-01 00:00:00"
                discharge_time_str = f"{year}-01-01 00:00:00"
            except (ValueError, TypeError):
                admission_time_str = ""
                discharge_time_str = ""

            all_diagnoses = self._extract_all_diagnoses_eicu(patientunitstayid)
            all_labs = self._extract_all_labs_eicu(patientunitstayid, lab_window_hours)
            all_vitals = self._extract_all_vitals_eicu(patientunitstayid)
            prior_meds, discharge_meds = self._extract_medications_eicu(patientunitstayid)

            data = CompletePatientData(
                database="eicu",
                subject_id=subject_id,
                hadm_id=hadm_id,
                age=age,
                gender=gender,
                ethnicity=ethnicity,
                height_cm=height_cm,
                weight_kg=weight_kg,
                bmi=bmi,
                all_lab_results=all_labs,
                all_diagnoses=all_diagnoses,
                all_vitals=all_vitals,
                prior_medications=prior_meds,
                discharge_medications=discharge_meds,
                admission_time=admission_time_str,
                discharge_time=discharge_time_str,
                los_days=los_days,
            )

            logger.info(
                f"✅ eICU提取完成: {len(all_labs)}个检查, "
                f"{len(all_diagnoses)}个诊断, {len(all_vitals)}个生命体征, "
                f"LOS={los_days}天, BMI={bmi}"
            )
            return data

        except Exception as e:
            logger.error(f"eICU提取失败 ({subject_id}/{hadm_id}): {e}")
            return None

    # ---- 子提取方法 ----

    def _extract_all_diagnoses_eicu(self, patientunitstayid: int) -> List[Diagnosis]:
        """
        提取诊断（eICU版本）。

        [FIX-eICU-4] ICD 编码策略：
          eICU 的 diagnosis 表中 icd9code 大量为 NULL。
          修复前：NULL → "Unknown"，导致 extract_dm_complications 匹配
                  "Unknown".startswith("428") 永远失败，has_heart_failure 恒 False。
          修复后：
            - icd9code 有值 → 正常使用，icd_version=9
            - icd9code 为 NULL → icd_code="" (空字符串)，不参与 ICD 前缀匹配，
              但 diagnosisstring（自由文本）保留在 description 中，
              供 LLM prompt 和 UMLS 映射使用
        """
        if hasattr(self, '_diagnosis_by_stay'):
            rows = self._diagnosis_by_stay.get(patientunitstayid, pd.DataFrame()).copy()
        else:
            rows = self.diagnosis[
                self.diagnosis["patientunitstayid"] == patientunitstayid
            ].copy()
        if rows.empty:
            return []

        rows = rows.sort_values("diagnosisoffset")

        result = []
        # [FIX-O2] seq 从 1 开始（与 MIMIC 统一）
        # 修复前：enumerate 从 0 开始 → convert_legacy 中 is_primary=(seq==1) 对 eICU 永远 False
        for seq, (_, r) in enumerate(rows.iterrows(), start=1):
            raw_icd = r.get("icd9code")
            if pd.isna(raw_icd) or str(raw_icd).strip() == "":
                # 没有 ICD 编码：保留自由文本描述，icd_code 置空
                icd = ""
            else:
                icd = str(raw_icd).strip()

            result.append(Diagnosis(
                icd_code=icd,
                icd_version=9,
                description=str(r.get("diagnosisstring", "") or ""),
                seq_num=seq,
            ))
        return result

    def _extract_all_labs_eicu(
        self, patientunitstayid: int, window_hours: int = 48
    ) -> List[LabResult]:
        """
        提取实验室检查（eICU版本）。

        列名对照（实际表结构）：
            单位  → labmeasurenamesystem（如 "mg/dL"、"ratio"）
            时间  → labresultoffset（入院后分钟数）
            名称  → labname
        """
        if hasattr(self, '_lab_by_stay'):
            rows = self._lab_by_stay.get(patientunitstayid, pd.DataFrame()).copy()
        else:
            rows = self.lab[
                self.lab["patientunitstayid"] == patientunitstayid
            ].copy()
        if rows.empty:
            return []

        # 时间窗口过滤
        # [FIX-eICU-A1] 同时过滤上下界：
        #   下界：>= -60 分钟（保留 ICU 入院前 1h 内的急诊检查，视为入院基础值）
        #         < -60 的记录来自住院早期普通病房，不代表基础状态，排除
        #   上界：<= window_min（默认 48h，只取入院早期数据）
        window_min = window_hours * 60
        rows = rows[
            (rows["labresultoffset"] >= -60) &
            (rows["labresultoffset"] <= window_min)
        ]

        # 按时间排序
        rows = rows.sort_values("labresultoffset")

        results = []
        for _, r in rows.iterrows():
            try:
                value = float(r["labresult"])
            except (ValueError, TypeError):
                continue  # 非数值跳过

            unit = str(r.get("labmeasurenamesystem") or "")

            # 负值 offset（入院前急诊检查）统一归为 admission+00h00m，
            # 下游 classify_eicu_labs 会将其标记为 BASELINE
            raw_offset = int(r.get("labresultoffset", 0) or 0)
            offset_min = max(raw_offset, 0)  # 负值归零，不产生负时间字符串
            h, m = divmod(offset_min, 60)
            charttime_str = f"admission+{h:02d}h{m:02d}m"

            results.append(LabResult(
                itemid=0,  # eICU无itemid
                label=str(r.get("labname") or "Unknown"),
                loinc_code=None,
                value=value,
                unit=unit,
                charttime=charttime_str,
                is_abnormal=False,
            ))
        return results

    def _extract_all_vitals_eicu(self, patientunitstayid: int) -> List[VitalSign]:
        """
        提取生命体征（eICU版本）。

        数据来源: vitalPeriodic + vitalAperiodic（均为宽格式表）

        宽格式表的处理流程::

            1. 按 patientunitstayid filter（此时数据量很小）
            2. pd.melt：把每个生命体征列展开为长格式的一行
            3. dropna：宽格式表里大量 NaN，melt 后一起去掉
            4. 用 VITAL_META 映射列名 → （显示名, 单位）
        """
        # ---- 列名 → (显示名, 单位) 映射 ----
        # vitalPeriodic 的指标
        PERIODIC_META = {
            "temperature":        ("Temperature",          "°F"),   # [FIX-C1] eICU vitalPeriodic.temperature 实际单位为华氏度
            "sao2":               ("SpO2",                 "%"),
            "heartrate":          ("Heart Rate",           "bpm"),
            "respiration":        ("Respiratory Rate",     "breaths/min"),
            "cvp":                ("CVP",                  "mmHg"),
            "etco2":              ("EtCO2",                "mmHg"),
            "systemicsystolic":   ("Systemic Systolic BP", "mmHg"),
            "systemicdiastolic":  ("Systemic Diastolic BP","mmHg"),
            "systemicmean":       ("Systemic Mean BP",     "mmHg"),
            "pasystolic":         ("PA Systolic",          "mmHg"),
            "padiastolic":        ("PA Diastolic",         "mmHg"),
            "pamean":             ("PA Mean",              "mmHg"),
            "st1":                ("ST1",                  "mV"),
            "st2":                ("ST2",                  "mV"),
            "st3":                ("ST3",                  "mV"),
            "icp":                ("ICP",                  "mmHg"),
        }
        # vitalAperiodic 的指标
        APERIODIC_META = {
            "noninvasivesystolic":   ("NonInvasive Systolic BP",  "mmHg"),
            "noninvasivediastolic":  ("NonInvasive Diastolic BP", "mmHg"),
            "noninvasivemean":       ("NonInvasive Mean BP",      "mmHg"),
            "paop":                  ("PA Occluded Pressure",     "mmHg"),
            "cardiacoutput":         ("Cardiac Output",           "L/min"),
            "cardiacinput":          ("Cardiac Input",            "L/min"),
            "svr":                   ("SVR",                      "dyn·s/cm5"),
            "svri":                  ("SVRI",                     "dyn·s·cm⁻⁵/m²"),
            "pvr":                   ("PVR",                      "dyn·s/cm5"),
            "pvri":                  ("PVRI",                     "dyn·s·cm⁻⁵/m²"),
        }

        long_frames = []  # 收集两个表 melt 后的长格式片段

        # ---- vitalPeriodic ----
        if not self.vitalperiodic.empty:
            peri = self.vitalperiodic[
                self.vitalperiodic["patientunitstayid"] == patientunitstayid
            ]
            if not peri.empty:
                value_vars = [c for c in PERIODIC_META if c in peri.columns]
                melted = peri.melt(
                    id_vars=["patientunitstayid", "observationoffset"],
                    value_vars=value_vars,
                    var_name="col_name",
                    value_name="value",
                ).dropna(subset=["value"])
                if not melted.empty:
                    long_frames.append(melted)

        # ---- vitalAperiodic ----
        if not self.vitalaperiodic.empty:
            aperi = self.vitalaperiodic[
                self.vitalaperiodic["patientunitstayid"] == patientunitstayid
            ]
            if not aperi.empty:
                value_vars = [c for c in APERIODIC_META if c in aperi.columns]
                melted = aperi.melt(
                    id_vars=["patientunitstayid", "observationoffset"],
                    value_vars=value_vars,
                    var_name="col_name",
                    value_name="value",
                ).dropna(subset=["value"])
                if not melted.empty:
                    long_frames.append(melted)

        # 两个表都空 → 无生命体征
        if not long_frames:
            return []

        # 合并为一个长格式表
        combined = pd.concat(long_frames, ignore_index=True)
        combined = combined.sort_values("observationoffset")

        # 合并两个映射用于统一查找
        all_meta = {**PERIODIC_META, **APERIODIC_META}

        # ---- 构建 VitalSign 对象 ----
        results = []
        for _, r in combined.iterrows():
            col = r["col_name"]
            label, unit = all_meta.get(col, (col, ""))

            try:
                value = float(r["value"])
            except (ValueError, TypeError):
                continue

            offset_min = int(r.get("observationoffset", 0) or 0)
            h, m = divmod(offset_min, 60)

            results.append(VitalSign(
                itemid=0,
                label=label,
                value=value,
                unit=unit,
                charttime=f"admission+{h:02d}h{m:02d}m",
            ))
        return results

    def _extract_medications_eicu(self, patientunitstayid: int) -> Tuple[List[str], List[str]]:
        """
        提取用药（eICU版本）。

        [FIX-eICU-5] 用药分类逻辑修正：
          修复前：drugstartoffset < 0 → prior（语义错误）
            eICU 中 drugstartoffset < 0 表示药物在 ICU 入院前已开始，
            可能是同次住院普通病房的用药，并非"出院前用药历史"。

          修复后：
            - prior_medications（入院前已在使用）：
                drugstartoffset < 0（ICU 入院前开始），视为既往用药
            - discharge_medications（出院时仍在使用）：
                drugstopoffset 为 NULL 或 > 0（ICU 期间未停药），
                且 drugstartoffset 不超过 ICU 住院时长
                → 最接近"出院带药"语义
        """
        if hasattr(self, '_medication_by_stay'):
            rows = self._medication_by_stay.get(patientunitstayid, pd.DataFrame()).copy()
        else:
            rows = self.medication[
                self.medication["patientunitstayid"] == patientunitstayid
            ].copy()
        if rows.empty:
            return [], []

        # prior：ICU 入院前已在使用的药物（offset < 0）
        prior = rows[rows["drugstartoffset"] < 0]["drugname"].dropna().unique().tolist()

        # discharge：最接近"出院带药"的近似
        # [FIX-eICU-A3] 修复前：drugstopoffset > 0 包含所有 ICU 期间已停药的药物
        #               → discharge_meds 过度膨胀，系统性压低评估 F1
        # 修复后：只保留以下两类：
        #   ① drugstopoffset 为 NULL（未记录停药时间，视为持续使用）
        #   ② drugstopoffset >= ICU 住院时长 × 80%（停药时间接近出院，视为出院带药）
        # 获取该患者的 ICU 时长（分钟）用于设定阈值
        if hasattr(self, '_patients_by_stay'):
            patient_row = self._patients_by_stay.get(patientunitstayid, pd.DataFrame())
        else:
            patient_row = self.patients[self.patients["patientunitstayid"] == patientunitstayid]
        los_min = 0.0
        if not patient_row.empty:
            try:
                los_min = float(patient_row.iloc[0].get("unitdischargeoffset") or 0)
            except (ValueError, TypeError):
                los_min = 0.0

        in_icu = rows[rows["drugstartoffset"] >= 0]

        # [FIX-N3] los_min=0 时 discharge_threshold=0，导致 >= 0 对所有药成立，
        # A-3 修复完全失效，回退到膨胀状态。
        # 修复：los_min 无效时退化为保守策略：只保留未记录停药时间的药物。
        if los_min > 0:
            discharge_threshold = los_min * 0.8  # 住院时长 80% 时点之后停药 → 视为出院带药
            discharge = in_icu[
                in_icu["drugstopoffset"].isna() |                        # ① 未记录停药
                (in_icu["drugstopoffset"] >= discharge_threshold)        # ② 接近出院时停药
            ]["drugname"].dropna().unique().tolist()
        else:
            # los_min 无效（unitdischargeoffset 缺失或为 0）
            # 保守策略：只保留未记录停药时间的药物，避免阈值=0 导致的膨胀
            discharge = in_icu[
                in_icu["drugstopoffset"].isna()
            ]["drugname"].dropna().unique().tolist()

        return prior, discharge

# ==================== 工厂函数 ====================

def create_patient_extractor(
    database: str,
    data_path: Path,
    batch_loader: Optional[BatchDataLoader] = None,
):
    """
    根据数据库类型创建提取器。

    Args:
        database: 'mimic3' | 'mimic4' | 'eicu'
        data_path: 数据根路径
        batch_loader: 可选的共享BatchDataLoader实例
    """
    database = database.lower()
    if database == "mimic3":
        from semantic_patient_extractor import SemanticPatientExtractor
        return SemanticPatientExtractor("mimic3", data_path)
    elif database == "mimic4":
        return MIMIC4PatientExtractor(data_path, batch_loader=batch_loader)
    elif database == "eicu":
        return eICUPatientExtractor(data_path, batch_loader=batch_loader)
    else:
        raise ValueError(f"不支持的数据库类型: {database}")


# ==================== 使用示例 ====================

def example_mimic4():
    """MIMIC-IV使用示例"""
    print("\n" + "="*80)
    print("MIMIC-IV数据提取示例")
    print("="*80)
    
    extractor = MIMIC4PatientExtractor(
        data_path=Path('ori_data_zip/mimic-iv-3.1')
    )
    
    # 示例：提取一个患者
    data = extractor.extract_complete_patient_data(
        subject_id=10000032,
        hadm_id=29079034
    )
    
    if data:
        print(f"\n✅ 提取成功:")
        print(f"  年龄: {data.age}")
        print(f"  性别: {data.gender}")
        print(f"  检查: {len(data.all_lab_results)} 项")
        print(f"  诊断: {len(data.all_diagnoses)} 个")
        
        # 显示前5个诊断
        print(f"\n前5个诊断:")
        for diag in data.all_diagnoses[:5]:
            print(f"  - {diag.icd_code}: {diag.description}")


def example_eicu():
    """eICU使用示例"""
    print("\n" + "="*80)
    print("eICU数据提取示例")
    print("="*80)
    
    extractor = eICUPatientExtractor(
        data_path=Path('ori_data_zip/eicu-collaborative-research-database-2.0')
    )
    
    # 示例：提取一个患者
    data = extractor.extract_complete_patient_data(
        subject_id=141168,  # patienthealthsystemstayid
        hadm_id=141168      # patientunitstayid
    )
    
    if data:
        print(f"\n✅ 提取成功:")
        print(f"  年龄: {data.age}")
        print(f"  性别: {data.gender}")
        print(f"  检查: {len(data.all_lab_results)} 项")
        print(f"  诊断: {len(data.all_diagnoses)} 个")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='多数据库患者提取器')
    parser.add_argument(
        '--database',
        choices=['mimic4', 'eicu'],
        required=True,
        help='数据库类型'
    )
    parser.add_argument(
        '--example',
        action='store_true',
        help='运行示例'
    )
    
    args = parser.parse_args()
    
    if args.example:
        if args.database == 'mimic4':
            example_mimic4()
        elif args.database == 'eicu':
            example_eicu()


if __name__ == '__main__':
    main()
