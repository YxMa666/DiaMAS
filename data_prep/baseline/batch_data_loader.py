#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BatchDataLoader - 大文件批量预加载缓存

解决的问题：
    MIMIC的LABEVENTS/CHARTEVENTS都是多GB文件。
    逐患者提取时，每人一次完整分块扫描，N个患者就扫N遍。
    这里做一次扫描，把目标ID的行按ID分组装进字典，后续查询O(1)。

使用规范：
    1. 创建实例（可多个提取器共享同一个实例）
    2. 调用 load_filtered() 预加载目标文件
    3. 提取器内部用 has()/get() 判断和获取

内存估算（5000患者）：
    labevents:    ~每人50行  → 250K行  → ~50MB
    chartevents:  ~每人200行 → 1M行    → ~200MB
    合计约250MB，研究环境无压力。

作者：Yuxi
日期：2025-02-03
"""

import time
from pathlib import Path
from typing import List, Dict, Optional, Set
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class BatchDataLoader:
    """
    批量大文件加载器。

    核心设计：
        - load_filtered(): 对文件扫描一次，按 id_column 分组缓存
        - 同一文件调用两次不会重复扫描（第二次直接返回缓存）
        - has()/get(): 提取器用来判断是否走缓存路径
        - clear(): 处理完一批后释放内存
    """

    def __init__(self):
        # {str(file_path): {id_value: DataFrame}}
        self._cache: Dict[str, Dict[int, pd.DataFrame]] = {}
        self._stats: Dict[str, dict] = {}

    # ---- 加载接口 ----

    def load_filtered(
        self,
        file_path: Path,
        id_column: str,
        target_ids: Set[int],
        usecols: List[str],
        parse_dates: Optional[List[str]] = None,
        chunksize: int = 500000,
    ) -> Dict[int, pd.DataFrame]:
        """
        单次扫描文件，按 id_column 分组缓存。

        同一文件已加载过则直接返回缓存，不重复扫描。

        Args:
            file_path:    要扫描的 csv.gz 文件路径
            id_column:    分组键列名（如 'hadm_id' 或 'stay_id'）
            target_ids:   只保留这些 ID 的行
            usecols:      需要读的列
            parse_dates:  需要解析日期的列
            chunksize:    分块大小（默认 500K 行）
        """
        key = str(file_path)
        if key in self._cache:
            logger.debug(f"  [BatchLoader] 命中缓存: {file_path.name}")
            return self._cache[key]

        logger.info(f"  [BatchLoader] 开始单次扫描: {file_path.name} (目标ID: {len(target_ids)}个)")
        t0 = time.time()

        # 用 list 先收集每个 ID 的碎片，最后统一 concat
        # （避免扫描过程中对同一 ID 反复 concat 大 DataFrame）
        collected: Dict[int, List[pd.DataFrame]] = {}
        rows_read = 0
        rows_kept = 0

        for chunk in pd.read_csv(
            file_path,
            usecols=usecols,
            parse_dates=parse_dates,
            chunksize=chunksize,
            low_memory=False,
        ):
            rows_read += len(chunk)
            filtered = chunk[chunk[id_column].isin(target_ids)]
            if filtered.empty:
                continue
            rows_kept += len(filtered)
            for id_val, group in filtered.groupby(id_column):
                collected.setdefault(id_val, []).append(group)

        # 合并各 ID 的碎片
        result: Dict[int, pd.DataFrame] = {
            id_val: pd.concat(dfs, ignore_index=True)
            for id_val, dfs in collected.items()
        }

        elapsed = time.time() - t0
        self._cache[key] = result
        self._stats[key] = {
            "file": str(file_path),
            "rows_read": rows_read,
            "rows_kept": rows_kept,
            "unique_ids_cached": len(result),
            "elapsed_s": round(elapsed, 2),
        }
        logger.info(
            f"  [BatchLoader] ✅ 完成: 扫描{rows_read:,}行, "
            f"保留{rows_kept:,}行, 缓存{len(result)}个ID, 耗时{elapsed:.1f}s"
        )
        return result

    def load_filtered_with_item_filter(
        self,
        file_path: Path,
        id_column: str,
        target_ids: Set[int],
        item_column: str,
        target_items: Set[int],
        usecols: List[str],
        parse_dates: Optional[List[str]] = None,
        chunksize: int = 500000,
    ) -> Dict[int, pd.DataFrame]:
        """
        单次扫描文件，同时按 id_column 和 item_column 双重过滤后分组缓存。

        与 load_filtered 相比额外做了 itemid 白名单过滤，
        主要用于 CHARTEVENTS（只保留 vital-sign 相关 itemids），
        大幅减少内存占用同时避免 per-patient 重复扫描。

        同一文件已加载过则直接返回缓存。
        """
        key = str(file_path)
        if key in self._cache:
            logger.debug(f"  [BatchLoader] 命中缓存: {file_path.name}")
            return self._cache[key]

        logger.info(
            f"  [BatchLoader] 双重过滤扫描: {file_path.name} "
            f"(目标ID: {len(target_ids)}个, 白名单items: {len(target_items)}个)"
        )
        t0 = time.time()

        collected: Dict[int, List[pd.DataFrame]] = {}
        rows_read = 0
        rows_kept = 0

        for chunk in pd.read_csv(
            file_path,
            usecols=usecols,
            parse_dates=parse_dates,
            chunksize=chunksize,
            low_memory=False,
        ):
            rows_read += len(chunk)
            # 双重过滤：hadm_id ∩ itemid 白名单
            filtered = chunk[
                chunk[id_column].isin(target_ids) &
                chunk[item_column].isin(target_items)
            ]
            if filtered.empty:
                continue
            rows_kept += len(filtered)
            for id_val, group in filtered.groupby(id_column):
                collected.setdefault(id_val, []).append(group)

        result: Dict[int, pd.DataFrame] = {
            id_val: pd.concat(dfs, ignore_index=True)
            for id_val, dfs in collected.items()
        }

        elapsed = time.time() - t0
        self._cache[key] = result
        self._stats[key] = {
            "file": str(file_path),
            "rows_read": rows_read,
            "rows_kept": rows_kept,
            "unique_ids_cached": len(result),
            "elapsed_s": round(elapsed, 2),
        }
        logger.info(
            f"  [BatchLoader] ✅ 完成(双重过滤): 扫描{rows_read:,}行, "
            f"保留{rows_kept:,}行, 缓存{len(result)}个ID, 耗时{elapsed:.1f}s"
        )
        return result

    # ---- 查询接口 ----

    def get(self, file_path: Path, target_id: int) -> Optional[pd.DataFrame]:
        """获取某个 ID 的已缓存数据。未加载或 ID 不存在返回 None。"""
        return self._cache.get(str(file_path), {}).get(target_id)

    def has(self, file_path: Path) -> bool:
        """该文件是否已经被加载过（无论目标 ID 是否在缓存里）。"""
        return str(file_path) in self._cache

    # ---- 管理接口 ----

    def clear(self, file_path: Optional[Path] = None):
        """
        清除缓存。

        Args:
            file_path: 传指定路径则只清该文件；不传则清全部。
        """
        if file_path:
            k = str(file_path)
            self._cache.pop(k, None)
            self._stats.pop(k, None)
        else:
            self._cache.clear()
            self._stats.clear()

    def memory_usage_mb(self) -> float:
        """估算当前缓存占用内存（MB）。"""
        total = 0
        for id_map in self._cache.values():
            for df in id_map.values():
                total += df.memory_usage(deep=True).sum()
        return total / (1024 * 1024)

    def get_stats(self) -> Dict[str, dict]:
        """返回各文件的加载统计信息。"""
        return self._stats
