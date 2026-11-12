#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified UMLS Mapping - 统一 UMLS 映射模块

核心改进：
1. 将 3 处重复的 UMLS 映射逻辑合并为 1 处
2. 支持 ICD-9 / ICD-10 / LOINC 统一映射
3. 增强的错误处理和统计
4. 作为独立函数调用，而非继承 mixin

[FIX-CACHE-DICT] 修复缓存反序列化导致的 AttributeError：
    EnhancedUMLSMapperWithCache 将 MappingResult dataclass 存入 UMLSCache，
    UMLSCache 用 dataclasses.asdict() 序列化为 JSON，磁盘读回时 value 变成 dict。
    调用方用属性访问（mapping.umls_cui）→ AttributeError: 'dict' has no attribute 'umls_cui'。
    修复：在 umls_mapping_unified.py 调用方加 _ensure_mapping_result() 辅助函数，
    检测到 dict 时用 MappingResult(**d) 重建 dataclass，对正常 dataclass 返回透传。
    选择在此处修复而非 enhanced_umls_mapper.py，原因：
      1. 该文件编码完好，可安全编辑
      2. 修复在消费层（属性访问发生处），职责清晰
      3. enhanced_umls_mapper.py 存在编码损坏（mojibake），修改风险高

作者：Yuxi
日期：2025-02
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from .data_models import CompletePatientData

logger = logging.getLogger(__name__)


@dataclass
class MappingStats:
    """UMLS 映射统计"""
    lab_total: int = 0
    lab_success: int = 0
    lab_not_found: int = 0       # UMLS 中无对应条目（正常情况）
    lab_api_error: int = 0       # API 调用异常（网络/权限/超时等）
    diag_total: int = 0
    diag_success: int = 0
    diag_not_found: int = 0      # ICD→CUI 查找失败（编码格式/过时/UMLS未收录）
    diag_api_error: int = 0      # API 调用异常

    @property
    def summary(self) -> str:
        return (
            f"Lab: {self.lab_success}/{self.lab_total} mapped, "
            f"{self.lab_not_found} not found, {self.lab_api_error} api errors | "
            f"Diag: {self.diag_success}/{self.diag_total} mapped, "
            f"{self.diag_not_found} not found, {self.diag_api_error} api errors"
        )


# ==================== [FIX-CACHE-DICT] 辅助函数 ====================

def _ensure_mapping_result(raw):
    """
    确保返回值是 MappingResult dataclass，而非 dict。

    问题根源：
        UMLSCache._save_to_disk() 用 dataclasses.asdict(CacheEntry) 序列化，
        asdict() 会递归将嵌套的 MappingResult dataclass 也转为 dict。
        _load_from_disk() 读回时 CacheEntry.value 是 dict，不会自动还原。

    此函数在属性访问前做类型检查，检测到 dict 时动态导入 MappingResult 并重建，
    正常 dataclass 直接透传，None 透传。

    Args:
        raw: map_loinc_to_standard / map_icd9_to_standard 的返回值

    Returns:
        MappingResult dataclass 实例，或 None
    """
    if raw is None:
        return None

    if isinstance(raw, dict):
        # 从 enhanced_umls_mapper 动态导入 MappingResult，避免循环依赖
        try:
            from baseline.enhanced_umls_mapper import MappingResult
        except ImportError:
            from enhanced_umls_mapper import MappingResult
        return MappingResult(**raw)

    # 已经是 dataclass，透传
    return raw


# ==================== 主映射函数 ====================

def apply_umls_mapping(
    data: CompletePatientData,
    umls_mapper,
    stop_on_permission_error: bool = True,
) -> tuple:
    """
    统一的 UMLS 映射函数（替代 3 处重复实现）

    支持所有数据库：
    - MIMIC-III: ICD-9 + LOINC
    - MIMIC-IV: ICD-9/10 + LOINC (部分无 LOINC)
    - eICU: ICD-9（部分无 ICD code）+ 无 LOINC

    Args:
        data: 完整患者数据
        umls_mapper: UMLS 映射器实例 (EnhancedUMLSMapper 或 WithCache 版本)
        stop_on_permission_error: 遇到权限错误时是否抛出异常

    Returns:
        (data, stats): 映射后的数据和统计信息
    """
    if umls_mapper is None:
        return data, MappingStats()

    stats = MappingStats()

    # ==================== 映射检查项（LOINC → UMLS → SNOMED） ====================
    for lab in data.all_lab_results:
        # 跳过无 LOINC code 或已映射的
        if not lab.loinc_code or lab.umls_cui:
            continue

        loinc = lab.loinc_code
        if pd.isna(loinc) or not isinstance(loinc, str) or loinc.strip() == '':
            continue

        stats.lab_total += 1

        try:
            raw = umls_mapper.map_loinc_to_standard(
                loinc_code=loinc.strip(),
                loinc_name=(lab.label or '').strip()
            )
            # [FIX-CACHE-DICT] 磁盘缓存读回时 raw 可能是 dict，重建为 dataclass
            mapping = _ensure_mapping_result(raw)

            if mapping and mapping.umls_cui:
                lab.umls_cui = mapping.umls_cui
                lab.snomed_code = mapping.snomed_code
                lab.snomed_name = mapping.snomed_name
                stats.lab_success += 1
            elif mapping and mapping.error_message:
                stats.lab_not_found += 1
                logger.debug(f"LOINC not found in UMLS: {loinc} - {mapping.error_message}")
            else:
                stats.lab_not_found += 1

        except Exception as e:
            error_type = type(e).__name__
            if 'Permission' in error_type and stop_on_permission_error:
                logger.error(f"UMLS permission error on LOINC {loinc}: {e}")
                raise
            if 'RateLimit' in error_type or 'rate_limit' in str(e).lower():
                logger.warning(f"UMLS rate limit hit on LOINC {loinc} — consider reducing workers or adding delay")
            stats.lab_api_error += 1
            if stats.lab_api_error <= 3:
                logger.warning(f"UMLS exception ({error_type}) on LOINC {loinc}: {e}")
            else:
                logger.debug(f"UMLS exception ({error_type}) on LOINC {loinc}: {e}")

    # ==================== 映射诊断（ICD → UMLS → SNOMED） ====================
    for diag in data.all_diagnoses:
        if diag.umls_cui:
            continue

        icd = diag.icd_code
        if not icd or pd.isna(icd) or not isinstance(icd, str) or icd.strip() == '':
            continue
        if icd.strip().lower() == 'unknown':
            continue

        stats.diag_total += 1
        desc = diag.description if isinstance(diag.description, str) else f"ICD_{icd}"

        try:
            if diag.icd_version == 9:
                raw = umls_mapper.map_icd9_to_standard(
                    icd9_code=icd.strip(),
                    icd9_description=desc.strip()
                )
                # [FIX-CACHE-DICT] 磁盘缓存读回时 raw 可能是 dict，重建为 dataclass
                mapping = _ensure_mapping_result(raw)

                if mapping and mapping.umls_cui:
                    diag.umls_cui = mapping.umls_cui
                    diag.snomed_code = mapping.snomed_code
                    diag.snomed_name = mapping.snomed_name
                    stats.diag_success += 1
                elif mapping and mapping.error_message:
                    stats.diag_not_found += 1
                    logger.debug(f"ICD-9 not found in UMLS: {icd} - {mapping.error_message}")
                else:
                    stats.diag_not_found += 1

            elif diag.icd_version == 10:
                cui = umls_mapper.find_cui_from_code(
                    code=icd.strip(),
                    source_vocabulary='ICD10CM'
                )
                if cui:
                    diag.umls_cui = cui
                    snomed_result = umls_mapper.get_snomed_from_cui(cui)
                    if snomed_result:
                        # get_snomed_from_cui 返回 tuple；磁盘缓存读回后变 list，下标访问仍兼容
                        diag.snomed_code = snomed_result[0]
                        diag.snomed_name = snomed_result[1]
                    stats.diag_success += 1
                else:
                    stats.diag_not_found += 1

            else:
                stats.diag_not_found += 1

        except Exception as e:
            error_type = type(e).__name__
            if 'Permission' in error_type and stop_on_permission_error:
                logger.error(f"UMLS permission error on ICD {icd}: {e}")
                raise
            if 'RateLimit' in error_type or 'rate_limit' in str(e).lower():
                logger.warning(f"UMLS rate limit hit on ICD-{diag.icd_version} {icd} — consider reducing workers or adding delay")
            stats.diag_api_error += 1
            if stats.diag_api_error <= 3:
                logger.warning(
                    f"UMLS exception ({error_type}) on ICD-{diag.icd_version} {icd}: {e}"
                )
            else:
                logger.debug(
                    f"UMLS exception ({error_type}) on ICD-{diag.icd_version} {icd}: {e}"
                )

    # ==================== 标记主诊断 ====================
    for diag in data.all_diagnoses:
        diag.is_primary = (diag.seq_num == 1)

    logger.info(f"  UMLS mapping: {stats.summary}")
    return data, stats
