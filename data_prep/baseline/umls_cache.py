#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UMLS映射缓存系统

功能：
1. 缓存UMLS API查询结果，避免重复调用
2. 支持内存缓存和持久化缓存
3. 提供缓存统计和性能分析
4. 支持缓存过期和手动清理

性能提升：
- 首次查询：正常API调用（~200ms）
- 缓存命中：<1ms（200倍速度提升）
- 1000个病例，假设50%命中率，节约~50秒

作者：Yuxi
日期：2025-02-06
"""

import json
import pickle
import hashlib
import time
from pathlib import Path
from typing import Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
from threading import Lock

logger = logging.getLogger(__name__)


# ==================== 缓存数据结构 ====================

@dataclass
class CacheEntry:
    """缓存条目"""
    key: str                    # 缓存键
    value: Any                  # 缓存值
    cached_at: str             # 缓存时间
    hit_count: int = 0         # 命中次数
    last_accessed: str = None  # 最后访问时间
    
    def to_dict(self) -> Dict:
        """转换为字典（用于JSON序列化）"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CacheEntry':
        """从字典创建"""
        return cls(**data)


# ==================== UMLS缓存管理器 ====================

class UMLSCache:
    """
    UMLS映射缓存管理器
    
    功能：
    1. 内存缓存（快速访问）
    2. 持久化缓存（跨会话）
    3. 缓存统计（性能分析）
    4. 自动保存（定期写入磁盘）
    
    使用示例：
        cache = UMLSCache(cache_dir='cache/umls')
        
        # 查询缓存
        result = cache.get('loinc', '2345-7')
        if result is None:
            result = api_call(...)
            cache.set('loinc', '2345-7', result)
    """
    
    def __init__(
        self,
        cache_dir: str = 'cache/umls',
        max_memory_size: int = 10000,
        auto_save_interval: int = 100,
        cache_expiry_days: Optional[int] = None,
        enable_persistent: bool = True
    ):
        """
        初始化缓存管理器
        
        Args:
            cache_dir: 缓存目录
            max_memory_size: 内存缓存最大条目数
            auto_save_interval: 自动保存间隔（修改次数）
            cache_expiry_days: 缓存过期天数（None表示永不过期）
            enable_persistent: 是否启用持久化
        """
        self.cache_dir = Path(cache_dir)
        self.max_memory_size = max_memory_size
        self.auto_save_interval = auto_save_interval
        self.cache_expiry_days = cache_expiry_days
        self.enable_persistent = enable_persistent
        
        # 内存缓存
        self._memory_cache: Dict[str, CacheEntry] = {}
        self._cache_lock = Lock()  # 线程安全
        
        # 统计信息
        self.stats = {
            'total_queries': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'api_calls_saved': 0,
            'modifications': 0,
            'last_save_time': None
        }
        
        # 创建缓存目录
        if self.enable_persistent:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_from_disk()
        
        logger.info(f"✅ UMLS缓存已初始化")
        logger.info(f"   缓存目录: {self.cache_dir}")
        logger.info(f"   内存限制: {self.max_memory_size} 条目")
        logger.info(f"   持久化: {'启用' if self.enable_persistent else '禁用'}")
        if cache_expiry_days:
            logger.info(f"   过期时间: {cache_expiry_days} 天")
    
    def _generate_cache_key(self, mapping_type: str, source_code: str) -> str:
        """
        生成缓存键
        
        Args:
            mapping_type: 映射类型（'loinc', 'icd9', 'icd10', 'cui_to_snomed'）
            source_code: 源代码
        
        Returns:
            缓存键字符串
        """
        # 使用 type:code 格式作为键
        return f"{mapping_type}:{source_code.upper()}"
    
    def get(
        self, 
        mapping_type: str, 
        source_code: str
    ) -> Optional[Any]:
        """
        从缓存获取映射结果
        
        Args:
            mapping_type: 映射类型
            source_code: 源代码
        
        Returns:
            缓存的映射结果，如果不存在返回None
        """
        with self._cache_lock:
            self.stats['total_queries'] += 1
            
            cache_key = self._generate_cache_key(mapping_type, source_code)
            
            if cache_key not in self._memory_cache:
                self.stats['cache_misses'] += 1
                return None
            
            entry = self._memory_cache[cache_key]
            
            # 检查是否过期
            if self._is_expired(entry):
                del self._memory_cache[cache_key]
                self.stats['cache_misses'] += 1
                logger.debug(f"缓存过期: {cache_key}")
                return None
            
            # 更新访问统计
            entry.hit_count += 1
            entry.last_accessed = datetime.now().isoformat()
            
            self.stats['cache_hits'] += 1
            self.stats['api_calls_saved'] += 1
            
            logger.debug(f"✅ 缓存命中: {cache_key} (命中{entry.hit_count}次)")
            
            return entry.value
    
    def set(
        self, 
        mapping_type: str, 
        source_code: str, 
        value: Any
    ) -> None:
        """
        设置缓存
        
        Args:
            mapping_type: 映射类型
            source_code: 源代码
            value: 映射结果
        """
        with self._cache_lock:
            cache_key = self._generate_cache_key(mapping_type, source_code)
            
            # 检查内存限制
            if len(self._memory_cache) >= self.max_memory_size:
                self._evict_lru()
            
            # 创建缓存条目
            entry = CacheEntry(
                key=cache_key,
                value=value,
                cached_at=datetime.now().isoformat(),
                hit_count=0,
                last_accessed=datetime.now().isoformat()
            )
            
            self._memory_cache[cache_key] = entry
            self.stats['modifications'] += 1
            
            logger.debug(f"💾 已缓存: {cache_key}")
            
            # 自动保存
            if self.enable_persistent and \
               self.stats['modifications'] % self.auto_save_interval == 0:
                self._save_to_disk()
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """检查缓存是否过期"""
        if self.cache_expiry_days is None:
            return False
        
        cached_time = datetime.fromisoformat(entry.cached_at)
        expiry_time = cached_time + timedelta(days=self.cache_expiry_days)
        
        return datetime.now() > expiry_time
    
    def _evict_lru(self) -> None:
        """
        驱逐最少使用的缓存条目（LRU策略）
        """
        if not self._memory_cache:
            return
        
        # 找到最少访问的条目
        lru_key = min(
            self._memory_cache.keys(),
            key=lambda k: (
                self._memory_cache[k].hit_count,
                self._memory_cache[k].last_accessed
            )
        )
        
        logger.debug(f"驱逐LRU缓存: {lru_key}")
        del self._memory_cache[lru_key]
    
    def _save_to_disk(self) -> None:
        """保存缓存到磁盘"""
        if not self.enable_persistent:
            return
        
        try:
            # 分类保存（按映射类型）
            cache_by_type = {}
            
            for entry in self._memory_cache.values():
                mapping_type = entry.key.split(':')[0]
                if mapping_type not in cache_by_type:
                    cache_by_type[mapping_type] = []
                
                cache_by_type[mapping_type].append(entry.to_dict())
            
            # 保存为JSON文件
            for mapping_type, entries in cache_by_type.items():
                cache_file = self.cache_dir / f'{mapping_type}_cache.json'
                
                with open(cache_file, 'w', encoding='utf-8') as f:
                    json.dump(entries, f, indent=2, ensure_ascii=False)
            
            self.stats['last_save_time'] = datetime.now().isoformat()
            
            logger.info(f"💾 缓存已保存: {len(self._memory_cache)} 条目")
            
        except Exception as e:
            logger.error(f"保存缓存失败: {e}")
    
    def _load_from_disk(self) -> None:
        """从磁盘加载缓存"""
        if not self.enable_persistent:
            return
        
        try:
            loaded_count = 0
            
            # 加载所有JSON缓存文件
            for cache_file in self.cache_dir.glob('*_cache.json'):
                with open(cache_file, 'r', encoding='utf-8') as f:
                    entries = json.load(f)
                
                for entry_dict in entries:
                    entry = CacheEntry.from_dict(entry_dict)
                    
                    # 检查是否过期
                    if not self._is_expired(entry):
                        self._memory_cache[entry.key] = entry
                        loaded_count += 1
            
            if loaded_count > 0:
                logger.info(f"📂 已加载缓存: {loaded_count} 条目")
        
        except Exception as e:
            logger.warning(f"加载缓存失败: {e}")
    
    def clear(self, mapping_type: Optional[str] = None) -> int:
        """
        清空缓存
        
        Args:
            mapping_type: 指定映射类型清空，None表示清空所有
        
        Returns:
            清空的条目数
        """
        with self._cache_lock:
            if mapping_type is None:
                count = len(self._memory_cache)
                self._memory_cache.clear()
                logger.info(f"🗑️  已清空所有缓存: {count} 条目")
                return count
            
            else:
                # 清空特定类型
                keys_to_remove = [
                    k for k in self._memory_cache.keys()
                    if k.startswith(f"{mapping_type}:")
                ]
                
                for key in keys_to_remove:
                    del self._memory_cache[key]
                
                logger.info(f"🗑️  已清空 {mapping_type} 缓存: {len(keys_to_remove)} 条目")
                return len(keys_to_remove)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息
        
        Returns:
            统计信息字典
        """
        with self._cache_lock:
            hit_rate = (self.stats['cache_hits'] / self.stats['total_queries'] * 100) \
                       if self.stats['total_queries'] > 0 else 0
            
            return {
                **self.stats,
                'cache_size': len(self._memory_cache),
                'hit_rate': round(hit_rate, 2),
                'memory_usage_mb': self._estimate_memory_usage(),
            }
    
    def _estimate_memory_usage(self) -> float:
        """估算内存使用（MB）"""
        try:
            import sys
            total_size = 0
            
            for entry in self._memory_cache.values():
                total_size += sys.getsizeof(entry.key)
                total_size += sys.getsizeof(entry.value)
            
            return round(total_size / 1024 / 1024, 2)
        except:
            return 0.0
    
    def print_stats(self) -> None:
        """打印缓存统计信息"""
        stats = self.get_stats()
        
        print("\n" + "=" * 60)
        print("UMLS缓存统计")
        print("=" * 60)
        print(f"总查询次数:    {stats['total_queries']}")
        print(f"缓存命中:      {stats['cache_hits']}")
        print(f"缓存未命中:    {stats['cache_misses']}")
        print(f"命中率:        {stats['hit_rate']:.2f}%")
        print(f"节省API调用:   {stats['api_calls_saved']}")
        print(f"当前缓存大小:  {stats['cache_size']} 条目")
        print(f"内存使用:      {stats['memory_usage_mb']} MB")
        if stats['last_save_time']:
            print(f"最后保存时间:  {stats['last_save_time']}")
        print("=" * 60)
    
    def save(self) -> None:
        """手动保存缓存"""
        self._save_to_disk()
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器退出（自动保存）"""
        if self.enable_persistent:
            self._save_to_disk()


# ==================== 缓存装饰器 ====================

def cached_umls_mapping(cache: UMLSCache, mapping_type: str):
    """
    UMLS映射缓存装饰器
    
    Args:
        cache: UMLSCache实例
        mapping_type: 映射类型
    
    Usage:
        @cached_umls_mapping(cache, 'loinc')
        def map_loinc(code, name):
            # API调用
            return result
    """
    def decorator(func):
        def wrapper(self, source_code: str, *args, **kwargs):
            # 尝试从缓存获取
            cached_result = cache.get(mapping_type, source_code)
            
            if cached_result is not None:
                return cached_result
            
            # 缓存未命中，调用原函数
            result = func(self, source_code, *args, **kwargs)
            
            # 保存到缓存
            if result is not None:
                cache.set(mapping_type, source_code, result)
            
            return result
        
        return wrapper
    return decorator


# ==================== 性能比较工具 ====================

class CachePerformanceAnalyzer:
    """缓存性能分析器"""
    
    def __init__(self):
        self.timings = {
            'with_cache': [],
            'without_cache': []
        }
    
    def time_with_cache(self, func, *args, **kwargs):
        """计时（使用缓存）"""
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        
        self.timings['with_cache'].append(elapsed)
        return result, elapsed
    
    def time_without_cache(self, func, *args, **kwargs):
        """计时（不使用缓存）"""
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        
        self.timings['without_cache'].append(elapsed)
        return result, elapsed
    
    def print_comparison(self):
        """打印性能对比"""
        import statistics
        
        if not self.timings['with_cache'] or not self.timings['without_cache']:
            print("没有足够的数据进行对比")
            return
        
        avg_with = statistics.mean(self.timings['with_cache'])
        avg_without = statistics.mean(self.timings['without_cache'])
        speedup = avg_without / avg_with if avg_with > 0 else 0
        
        print("\n" + "=" * 60)
        print("缓存性能对比")
        print("=" * 60)
        print(f"使用缓存平均耗时:     {avg_with*1000:.2f} ms")
        print(f"不使用缓存平均耗时:   {avg_without*1000:.2f} ms")
        print(f"速度提升:             {speedup:.1f}x")
        print(f"时间节省:             {(avg_without - avg_with)*100:.0f}%")
        print("=" * 60)


# ==================== 使用示例 ====================

if __name__ == '__main__':
    # 示例1: 基本使用
    print("示例1: 基本缓存使用")
    print("-" * 60)
    
    cache = UMLSCache(cache_dir='test_cache/umls')
    
    # 模拟API调用
    def mock_api_call(code):
        time.sleep(0.2)  # 模拟200ms延迟
        return f"CUI_{code}"
    
    # 第一次查询（缓存未命中）
    code = '2345-7'
    result = cache.get('loinc', code)
    
    if result is None:
        print(f"缓存未命中，调用API...")
        result = mock_api_call(code)
        cache.set('loinc', code, result)
    
    print(f"结果: {result}")
    
    # 第二次查询（缓存命中）
    result = cache.get('loinc', code)
    print(f"缓存命中！结果: {result}")
    
    # 打印统计
    cache.print_stats()
    
    # 保存缓存
    cache.save()
    
    print("\n示例2: 性能对比")
    print("-" * 60)
    
    analyzer = CachePerformanceAnalyzer()
    
    # 测试10次
    for i in range(10):
        # 使用缓存
        cache.get('loinc', code)
        
        # 不使用缓存（模拟）
        if i < 5:  # 只测试前5次
            mock_api_call(code)
    
    print("\n✅ 缓存系统演示完成")
