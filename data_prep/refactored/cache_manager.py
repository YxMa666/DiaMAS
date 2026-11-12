#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Cache Manager - 增强缓存管理器

核心改进：
1. 使用 v3 格式（所有属性在 dataclass 中声明，无动态 setattr）
2. 去掉全局 Lock 瓶颈，改用 per-file 锁（或无锁 → 每个线程写独立文件）
3. 支持 Union[int, str] 的 subject_id/hadm_id
4. 自动兼容旧版缓存格式（v2 → v3 升级）

作者：Yuxi
日期：2025-02
"""

import json
import pickle
import logging
from pathlib import Path
from typing import Optional, Dict, Union

from .data_models import (
    CompletePatientData, patient_data_to_dict, patient_data_from_dict
)

logger = logging.getLogger(__name__)


class EnhancedCacheManager:
    """
    增强缓存管理器

    改进：
    - 无全局锁（每个患者是独立文件，并行写入无冲突）
    - 支持 JSON / pickle 格式
    - 自动兼容旧版缓存
    """

    def __init__(self, cache_dir: Path, fmt: str = 'json'):
        """
        Args:
            cache_dir: 缓存根目录
            fmt: 'json' 或 'pickle'
        """
        self.cache_dir = Path(cache_dir)
        self.fmt = fmt
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ CacheManager: {cache_dir} (format={fmt})")

    def _get_path(
        self, database: str,
        subject_id: Union[int, str],
        hadm_id: Union[int, str]
    ) -> Path:
        """
        获取缓存文件路径

        改进：对字符串 ID（eICU uniquepid）做安全处理，
        将 '002-10009' 这种格式中的 / 和特殊字符替换掉
        """
        db_dir = self.cache_dir / database
        db_dir.mkdir(parents=True, exist_ok=True)

        # 安全的文件名
        safe_sid = str(subject_id).replace('/', '_').replace('\\', '_')
        safe_hid = str(hadm_id).replace('/', '_').replace('\\', '_')
        ext = 'pkl' if self.fmt == 'pickle' else 'json'

        return db_dir / f"{safe_sid}_{safe_hid}.{ext}"

    def exists(
        self, database: str,
        subject_id: Union[int, str],
        hadm_id: Union[int, str]
    ) -> bool:
        return self._get_path(database, subject_id, hadm_id).exists()

    def save(self, data: CompletePatientData) -> bool:
        """保存（无全局锁 → 每个患者文件独立）"""
        path = self._get_path(data.database, data.subject_id, data.hadm_id)
        try:
            if self.fmt == 'json':
                d = patient_data_to_dict(data)
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(d, f, indent=2, ensure_ascii=False)
            else:
                with open(path, 'wb') as f:
                    pickle.dump(data, f)
            return True
        except Exception as e:
            logger.error(f"Cache save failed ({path}): {e}")
            return False

    def load(
        self, database: str,
        subject_id: Union[int, str],
        hadm_id: Union[int, str]
    ) -> Optional[CompletePatientData]:
        """加载（自动兼容 v2/v3 格式）"""
        path = self._get_path(database, subject_id, hadm_id)
        if not path.exists():
            return None

        try:
            if self.fmt == 'json':
                with open(path, 'r', encoding='utf-8') as f:
                    d = json.load(f)
                return self._load_json_with_compat(d)
            else:
                with open(path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            logger.error(f"Cache load failed ({path}): {e}")
            return None

    @staticmethod
    def _load_json_with_compat(d: Dict) -> CompletePatientData:
        """
        加载 JSON 并自动兼容旧版格式

        v2 格式特征：_format == 'json_complete_v2'
        v3 格式特征：_format == 'json_complete_v3'
        """
        fmt_version = d.get('_format', '')

        if fmt_version == 'json_complete_v3':
            return patient_data_from_dict(d)

        # v2 兼容：旧格式中 UMLS 属性可能不存在
        # patient_data_from_dict 已处理缺失字段（用 .get(key, default)）
        return patient_data_from_dict(d)

    def get_stats(self) -> Dict:
        """缓存统计"""
        stats = {}
        if not self.cache_dir.exists():
            return stats

        for db_dir in self.cache_dir.iterdir():
            if db_dir.is_dir():
                ext = 'pkl' if self.fmt == 'pickle' else 'json'
                files = list(db_dir.glob(f'*.{ext}'))
                stats[db_dir.name] = {
                    'count': len(files),
                    'size_mb': sum(f.stat().st_size for f in files) / (1024 * 1024),
                }
        return stats
