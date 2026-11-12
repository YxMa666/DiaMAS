#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UMLS Relationships Mapper - 完全修复版 v4

关键修复：
✅ 正确的认证流程（TGT → Service Ticket → API调用）
✅ TGT缓存（8小时有效）
✅ Service Ticket每次获取新的（一次性使用）
✅ 401错误自动重试
✅ 404错误优雅处理（CUI不存在或已废弃）
✅ 语义类型过滤可选（默认返回所有相关概念）

作者：Yuxi
日期：2026-01-15
"""

import requests
import time
import logging
import re
import threading
from typing import List, Dict, Optional
from dataclasses import dataclass
from functools import wraps

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== 重试装饰器 ====================

def retry_with_backoff(max_retries=3, base_delay=2):
    """指数退避重试装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        logger.error(f"{func.__name__} 失败（已重试{max_retries}次）: {e}")
                        raise
                    
                    delay = base_delay * (2 ** attempt)
                    logger.warning(f"{func.__name__} 失败，{delay}秒后重试... (尝试 {attempt+1}/{max_retries})")
                    time.sleep(delay)
        return wrapper
    return decorator


# ==================== 数据结构 ====================

@dataclass
class UMLSConcept:
    """UMLS概念"""
    cui: str
    name: str
    semantic_types: List[str]
    
    def __repr__(self):
        return f"UMLSConcept({self.cui}, {self.name})"


# ==================== UMLS Relationships Mapper (完全修复版) ====================

class UMLSRelationshipMapper:
    """
    UMLS关系映射器（完全修复版 v4）
    
    关键修复：
    1. 正确的认证流程：TGT → Service Ticket → API
    2. TGT缓存（8小时有效）
    3. Service Ticket每次获取新的（一次性使用）
    4. 401错误自动重试
    5. 404错误优雅处理（CUI不存在或已废弃）✅
    6. 语义类型过滤可选（默认返回所有相关概念）✅
    """
    
    def __init__(self, api_key: str, filter_semantic_types: bool = False):
        """
        初始化UMLS Mapper
        
        Args:
            api_key: UMLS API key
            filter_semantic_types: 是否按语义类型过滤（默认False，返回所有相关概念）
        """
        if not api_key:
            raise ValueError("UMLS API key is required")
        
        self.api_key = api_key
        self.base_url = "https://uts-ws.nlm.nih.gov/rest"
        self.service_url = "http://umlsks.nlm.nih.gov"
        
        # ✅ 是否过滤语义类型（默认不过滤）
        self.filter_semantic_types = filter_semantic_types
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (compatible; UMLSMapper/2.1)'
        })
        
        # TGT缓存（8小时有效），ST不缓存（一次性）
        self._tgt_url = None
        self._tgt_expiry = 0
        # ✅ [FIX-TGT-RACE] 多worker并发时保证TGT刷新的原子性
        self._tgt_lock = threading.Lock()
        
        # 病理生理相关的语义类型（仅在filter_semantic_types=True时使用）
        self.pathophysiology_types = [
            'Pathologic Function',
            'Disease or Syndrome',
            'Finding',
            'Sign or Symptom',
            'Cell or Molecular Dysfunction',
            'Organ or Tissue Function'
        ]
        
        # 相关关系类型
        self.relevant_relations = [
            'PAR',   # Parent
            'CHD',   # Child
            'RO',    # Related
            'RB',    # Broader
            'RN'     # Narrower
        ]
        
        if filter_semantic_types:
            logger.info("✅ UMLS Relationships Mapper 初始化完成（语义类型过滤已启用）")
        else:
            logger.info("✅ UMLS Relationships Mapper 初始化完成（返回所有相关概念）")
    
    @retry_with_backoff(max_retries=5, base_delay=2)
    def _get_tgt(self) -> str:
        """
        获取TGT (Ticket Granting Ticket)
        
        Returns:
            TGT字符串
        """
        auth_url = "https://utslogin.nlm.nih.gov/cas/v1/api-key"
        
        response = self.session.post(
            auth_url, 
            data={'apikey': self.api_key},
            timeout=60
        )
        
        if response.status_code == 201:
            tgt_url = response.headers.get('Location')
            logger.debug(f"✅ 获取TGT成功: {tgt_url}")
            return tgt_url
        else:
            raise Exception(f"获取TGT失败: {response.status_code} - {response.text}")
    
    @retry_with_backoff(max_retries=3, base_delay=2)
    def _get_service_ticket(self, tgt_url: str, service: str) -> str:
        """
        用TGT获取Service Ticket（关键修复！）
        
        Args:
            tgt_url: TGT的完整URL
            service: 服务URL
        
        Returns:
            Service Ticket字符串
        """
        response = self.session.post(
            tgt_url,
            data={'service': service},
            timeout=30
        )
        
        if response.status_code == 200:
            service_ticket = response.text.strip()
            logger.debug(f"✅ 获取Service Ticket成功: {service_ticket[:20]}...")
            return service_ticket
        else:
            raise Exception(f"获取Service Ticket失败: {response.status_code} - {response.text}")
    
    def get_ticket(self) -> str:
        """
        获取Service Ticket（每次都是新的）
        
        关键修复：
        - TGT缓存（8小时有效）
        - Service Ticket每次获取新的（一次性使用）
        - [FIX-TGT-RACE] threading.Lock 保护 TGT 刷新，防止多 worker 并发踩踏
        
        Returns:
            Service Ticket字符串
        """
        # ✅ [FIX-TGT-RACE] 先在锁外做快速读检查，避免每次 ST 获取都争锁
        now = time.time()
        if self._tgt_url and now < self._tgt_expiry:
            tgt_url = self._tgt_url
        else:
            # 需要刷新 TGT，加锁确保只有一个线程执行
            with self._tgt_lock:
                # double-check：加锁后再检查一次，防止其他线程已刷新
                now = time.time()
                if self._tgt_url and now < self._tgt_expiry:
                    tgt_url = self._tgt_url
                else:
                    logger.debug("获取新的TGT...")
                    tgt_url = self._get_tgt()
                    self._tgt_url = tgt_url
                    self._tgt_expiry = now + 7 * 3600  # 7小时
                    logger.info("✅ TGT已更新（有效期7小时）")
        
        # ✅ 每次都获取新的Service Ticket（ST是一次性的）
        service_ticket = self._get_service_ticket(tgt_url, self.service_url)
        logger.debug(f"✅ 获取新的Service Ticket: {service_ticket[:20]}...")
        
        return service_ticket
    
    def _query_umls_search(
        self, 
        ticket: str, 
        query: str, 
        search_type: str = "words",
        page_size: int = 5
    ) -> List[Dict]:
        """
        统一的UMLS搜索方法（使用Service Ticket）
        
        Args:
            ticket: Service Ticket (不是TGT!)
            query: 搜索词
            search_type: 搜索类型（normalized/exact/words）
            page_size: 返回结果数量
        
        Returns:
            搜索结果列表
        """
        url = f"{self.base_url}/search/current"
        
        params = {
            "ticket": ticket,  # ✅ 使用Service Ticket
            "string": query,
            "pageSize": page_size
        }
        
        # 只有在非默认时才添加searchType参数
        if search_type and search_type != "words":
            params["searchType"] = search_type
        
        logger.debug(f"UMLS搜索: {query} (类型: {search_type})")
        logger.debug(f"请求URL: {url}")
        logger.debug(f"参数: ticket={ticket[:20]}..., string={query}")
        
        try:
            response = self.session.get(url, params=params, timeout=60)
            
            # 如果401，可能是TGT过期或ST无效
            if response.status_code == 401:
                logger.warning("认证失败（401），清除TGT缓存并重试...")
                self._tgt_url = None  # ✅ 清除TGT缓存
                self._tgt_expiry = 0
                raise Exception("认证失败，需要重新获取TGT")
            
            response.raise_for_status()
            
            data = response.json()
            results = data.get("result", {}).get("results", [])
            
            logger.debug(f"UMLS返回 {len(results)} 个结果")
            
            return results
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP错误: {e}")
            logger.error(f"响应状态码: {response.status_code}")
            logger.error(f"响应内容: {response.text[:500]}")
            raise
        except Exception as e:
            logger.error(f"请求异常: {e}")
            raise
    
    @retry_with_backoff(max_retries=3, base_delay=2)
    def find_disease_cui(self, disease_name: str) -> Optional[str]:
        """
        查找疾病的CUI（使用UMLS API）
        
        多策略搜索（按推荐顺序）：
        1. Words search (默认，最稳定)
        2. Exact match
        3. Normalized search
        
        Args:
            disease_name: 疾病名称
        
        Returns:
            CUI (如 "C0011881") 或 None
        """
        # 获取有效的Service Ticket
        ticket = self.get_ticket()
        
        name = disease_name.strip()
        
        # 策略1: Words search（默认方式，最稳定）
        try:
            results = self._query_umls_search(ticket, name, "words")
            if results:
                cui = results[0]["ui"]
                logger.info(f"[UMLS API] {disease_name} → {cui} (words)")
                return cui
        except Exception as e:
            logger.debug(f"Words search failed: {e}")
            # 如果是401错误，重新获取ticket再试一次
            if "401" in str(e) or "过期" in str(e):
                ticket = self.get_ticket()
                try:
                    results = self._query_umls_search(ticket, name, "words")
                    if results:
                        cui = results[0]["ui"]
                        logger.info(f"[UMLS API] {disease_name} → {cui} (words, 重试)")
                        return cui
                except:
                    pass
        
        # 策略2: Exact match
        try:
            results = self._query_umls_search(ticket, name, "exact")
            if results:
                cui = results[0]["ui"]
                logger.info(f"[UMLS API] {disease_name} → {cui} (exact)")
                return cui
        except Exception as e:
            logger.debug(f"Exact search failed: {e}")
        
        # 策略3: Normalized search
        try:
            results = self._query_umls_search(ticket, name, "normalized")
            if results:
                cui = results[0]["ui"]
                logger.info(f"[UMLS API] {disease_name} → {cui} (normalized)")
                return cui
        except Exception as e:
            logger.debug(f"Normalized search failed: {e}")
        
        logger.warning(f"[UMLS API] 未找到CUI: {disease_name}")
        return None
    
    @retry_with_backoff(max_retries=3, base_delay=2)
    def get_related_concepts(self, cui: str) -> List[Dict]:
        """
        获取相关概念（通过UMLS关系）
        
        Args:
            cui: 疾病的CUI
        
        Returns:
            相关概念列表
            - 如果filter_semantic_types=False（默认）：返回所有相关概念
            - 如果filter_semantic_types=True：只返回病理生理相关的概念
        """
        # 获取有效的Service Ticket
        ticket = self.get_ticket()
        
        # 查询关系
        relations_url = f"{self.base_url}/content/current/CUI/{cui}/relations"
        params = {
            'ticket': ticket, 
            'pageSize': 50
        }
        
        logger.debug(f"查询CUI关系: {cui}")
        
        try:
            response = self.session.get(relations_url, params=params, timeout=30)
            
            # 处理401错误
            if response.status_code == 401:
                logger.warning("认证失败，清除TGT并重新获取...")
                self._tgt_url = None
                self._tgt_expiry = 0
                # 获取新的ticket并重试
                ticket = self.get_ticket()
                params['ticket'] = ticket
                response = self.session.get(relations_url, params=params, timeout=30)
            
            response.raise_for_status()
        except Exception as e:
            logger.error(f"查询关系失败: {e}")
            return []
        
        data = response.json()
        
        related_concepts = []
        
        for relation in data.get('result', []):
            related_cui = relation.get('relatedId')
            relation_label = relation.get('relationLabel')
            
            # 只要特定关系类型
            if relation_label in self.relevant_relations:
                # 获取概念详情（方法内部会自己获取新的ticket）
                concept_info = self._get_concept_details(related_cui)
                
                if not concept_info:
                    continue
                
                # ✅ 根据配置决定是否过滤
                if self.filter_semantic_types:
                    # 只保留病理生理相关的概念
                    if not self._is_pathophysiology_concept(concept_info):
                        continue
                
                # 添加到结果
                related_concepts.append({
                    'cui': concept_info.cui,
                    'name': concept_info.name,
                    'semantic_type': ', '.join(concept_info.semantic_types),
                    'relation': relation_label
                })
        
        logger.info(f"[UMLS API] CUI {cui} → {len(related_concepts)} 个病理生理概念")
        
        return related_concepts
    
    @retry_with_backoff(max_retries=3, base_delay=2)
    def _get_concept_details(self, cui: str) -> Optional[UMLSConcept]:
        """
        获取概念详情
        
        Args:
            cui: CUI
        
        Returns:
            UMLSConcept对象，如果CUI不存在则返回None
        """
        # ✅ 自己获取新的Service Ticket
        ticket = self.get_ticket()
        
        detail_url = f"{self.base_url}/content/current/CUI/{cui}"
        
        try:
            response = self.session.get(
                detail_url, 
                params={'ticket': ticket}, 
                timeout=30
            )
            
            # ✅ 优雅处理404（CUI不存在或已废弃）
            if response.status_code == 404:
                logger.debug(f"CUI不存在或已废弃: {cui}")
                return None
            
            if response.status_code == 401:
                # Ticket过期，清除TGT缓存并重试
                logger.warning("认证失败，清除TGT缓存...")
                self._tgt_url = None
                self._tgt_expiry = 0
                # 重新获取ticket并重试
                ticket = self.get_ticket()
                response = self.session.get(
                    detail_url,
                    params={'ticket': ticket},
                    timeout=30
                )
            
            if response.status_code != 200:
                logger.debug(f"无法获取CUI详情: {cui} (状态码: {response.status_code})")
                return None
            
            data = response.json()
            result = data.get('result')
            
            if not result:
                return None
            
            semantic_types = [
                st.get('name') for st in result.get('semanticTypes', [])
            ]
            
            return UMLSConcept(
                cui=cui,
                name=result.get('name', ''),
                semantic_types=semantic_types
            )
            
        except Exception as e:
            logger.debug(f"获取CUI详情异常: {cui} - {e}")
            return None
    
    def _is_pathophysiology_concept(self, concept: UMLSConcept) -> bool:
        """判断是否为病理生理相关的概念"""
        return any(
            ptype in concept.semantic_types 
            for ptype in self.pathophysiology_types
        )
    
    def map_disease_to_pathophysiology(self, disease_name: str) -> List[str]:
        """
        将疾病名映射为病理生理状态列表（便捷方法）
        
        Args:
            disease_name: 疾病名称
        
        Returns:
            病理生理状态列表（标准化名称）
        """
        # Step 1: 查找CUI
        cui = self.find_disease_cui(disease_name)
        
        if not cui:
            logger.warning(f"无法找到CUI: {disease_name}")
            return []
        
        # Step 2: 获取相关概念
        concepts = self.get_related_concepts(cui)
        
        # Step 3: 标准化名称
        states = [self._standardize_name(c['name']) for c in concepts]
        
        return states
    
    def _standardize_name(self, name: str) -> str:
        """标准化概念名称"""
        import re
        
        name = name.lower().strip()
        name = re.sub(r'\([^)]*\)', '', name)
        name = name.replace(' ', '_').replace('-', '_').replace(',', '')
        name = re.sub(r'_+', '_', name).strip('_')
        
        return name


# ==================== 使用示例 ====================

def main():
    """使用示例"""
    
    print("\n" + "="*70)
    print("UMLS Relationships Mapper 测试（完全修复版）")
    print("="*70)
    
    import os
    from agents.config import APIConfig
    api_key = APIConfig.UMLS_KEY#os.environ.get('UMLS_API_KEY')
    
    if not api_key:
        print("\n⚠️ 请设置UMLS API key")
        print("  export UMLS_API_KEY='your_key_here'")
        api_key = input("\n或直接输入API key: ").strip()
        
        if not api_key:
            print("❌ 未提供API key，退出")
            return
    
    mapper = UMLSRelationshipMapper(api_key=api_key)
    
    # 测试疾病列表
    test_diseases = [
        "diabetes mellitus",
        "schizophrenia",
        "hypertension",
        "type 2 diabetes"
    ]
    
    for disease in test_diseases:
        print(f"\n{'='*70}")
        print(f"测试: {disease}")
        print(f"{'='*70}")
        
        # 查找CUI
        cui = mapper.find_disease_cui(disease)
        
        if cui:
            print(f"✅ CUI: {cui}")
            
            # 获取相关概念
            concepts = mapper.get_related_concepts(cui)
            
            if concepts:
                print(f"\n找到 {len(concepts)} 个病理生理概念:")
                for concept in concepts[:5]:
                    print(f"  - {concept['name']}")
                    print(f"    CUI: {concept['cui']}")
                    print(f"    类型: {concept['semantic_type']}")
        else:
            print(f"❌ 未找到CUI")
    
    print("\n✅ 测试完成！")


if __name__ == "__main__":
    main()
