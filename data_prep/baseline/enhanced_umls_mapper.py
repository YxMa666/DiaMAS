#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced UMLS Mapper - v2.1（修复CUI获取bug）

修复内容：
1. ✅ 修复find_cui_from_code返回源代码而非CUI的bug
2. ✅ 使用正确的UMLS API端点（/content/.../atoms）
3. ✅ 从concept URL中提取CUI
4. ✅ 保留细粒度错误处理

Bug说明：
- 旧版：使用 /crosswalk API，返回的ui字段是源代码本身
- 新版：使用 /atoms API，从concept URL提取真正的CUI

作者：Yuxi
日期：2025-02-05
"""

import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging
import time
import functools
import requests
import re

# 导入现有的UMLS映射器
# sys.path.append(str(Path(__file__).parent.parent / 'project'))
from drugs.umls_relationship_mapper import UMLSRelationshipMapper, UMLSConcept # , retry_with_backoff
from baseline.umls_cache import UMLSCache # , MappingResult
# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== 自定义异常 ====================

class UMLSAPIError(Exception):
    """UMLS API错误基类"""
    pass


class UMLSAuthenticationError(UMLSAPIError):
    """认证错误（401）- 可能可以通过重新认证解决"""
    pass


class UMLSPermissionError(UMLSAPIError):
    """权限错误（403）- 不可恢复，需要检查API key权限"""
    pass


class UMLSNotFoundError(UMLSAPIError):
    """资源不存在（404）- 正常情况，表示数据库中没有对应的映射"""
    pass


class UMLSRateLimitError(UMLSAPIError):
    """速率限制（429）- 可以通过等待后重试"""
    pass


class UMLSServerError(UMLSAPIError):
    """服务器错误（5xx）- 临时故障，可以重试"""
    pass


class UMLSClientError(UMLSAPIError):
    """客户端错误（其他4xx）- 请求格式错误，不可恢复"""
    pass


class UMLSNetworkError(UMLSAPIError):
    """网络错误（超时、连接失败）- 可以重试"""
    pass


# ==================== 智能重试装饰器 ====================

def smart_retry(
    max_retries: int = 3,
    base_delay: float = 2.0,
    retryable_errors: tuple = (UMLSServerError, UMLSRateLimitError, UMLSNetworkError)
):
    """
    智能重试装饰器
    
    特点：
    - 只重试可恢复的错误（服务器错误、速率限制、网络错误）
    - 不重试不可恢复的错误（权限错误、认证错误、客户端错误）
    - 使用指数退避策略
    
    Args:
        max_retries: 最大重试次数
        base_delay: 基础延迟（秒）
        retryable_errors: 可重试的错误类型
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                
                except retryable_errors as e:
                    last_error = e
                    
                    if attempt < max_retries - 1:
                        # 指数退避
                        delay = base_delay * (2 ** attempt)
                        
                        if isinstance(e, UMLSRateLimitError):
                            # 速率限制错误 - 等待更长时间
                            delay = max(delay, 60)
                            logger.warning(f"⏳ UMLS速率限制，等待{delay}秒后重试...")
                        else:
                            logger.warning(f"🔄 请求失败，{delay}秒后重试... ({type(e).__name__})")
                        
                        time.sleep(delay)
                    else:
                        logger.error(f"❌ 重试{max_retries}次后仍失败: {e}")
                        raise
                
                except UMLSAPIError:
                    # 不可恢复的错误，直接抛出
                    raise
            
            raise last_error
        
        return wrapper
    return decorator

# ==================== 新增数据结构 ====================

@dataclass
class MappingResult:
    """映射结果"""
    source_code: str  # 原始编码（如LOINC, ICD-9）
    source_system: str  # 编码系统（'LOINC', 'ICD9CM', 'ICD10CM'）
    source_name: str  # 原始名称
    
    umls_cui: Optional[str] = None  # UMLS CUI
    umls_name: Optional[str] = None  # UMLS首选名称
    
    snomed_code: Optional[str] = None  # SNOMED CT编码
    snomed_name: Optional[str] = None  # SNOMED CT名称
    
    confidence: float = 0.0  # 映射置信度 (0-1)
    error_message: Optional[str] = None  # 错误信息（如果映射失败）

# ==================== 增强的UMLS映射器 ====================

class EnhancedUMLSMapper(UMLSRelationshipMapper):
    """
    增强的UMLS映射器
    
    复用父类功能：
    - TGT/ST认证
    - CUI查找
    - 关系查询
    
    新增功能：
    - LOINC映射
    - ICD-9/ICD-10映射
    - 批量处理
    """
    
    def __init__(self, api_key: str, filter_semantic_types: bool = False):
        """初始化（复用父类）"""
        super().__init__(api_key, filter_semantic_types)
        
        # 错误统计
        self.error_stats = {
            'permission_errors': 0,
            'rate_limit_errors': 0,
            'server_errors': 0,
            'not_found': 0,
            'network_errors': 0,
            'other_errors': 0
        }
        
        logger.info("✅ Enhanced UMLS Mapper 初始化完成")
    
    def _handle_response(
        self, 
        response: requests.Response,
        context: str = ""
    ) -> Optional[dict]:
        """
        统一的响应处理
        
        Args:
            response: HTTP响应
            context: 上下文信息（用于日志）
        
        Returns:
            响应数据（JSON）或None（404情况）
        
        Raises:
            各种UMLS异常
        """
        status_code = response.status_code
        
        # ✅ 成功
        if status_code == 200:
            return response.json()
        
        # 🔐 认证失败
        elif status_code == 401:
            raise UMLSAuthenticationError(
                f"UMLS API认证失败 (401){context}: 需要重新获取ticket"
            )
        
        # 🚫 权限不足
        elif status_code == 403:
            self.error_stats['permission_errors'] += 1
            raise UMLSPermissionError(
                f"UMLS API权限不足 (403){context}: "
                f"请检查API key是否有访问权限。"
                f"可能原因：(1) API key过期 (2) 未授权访问该资源 (3) 超出配额"
            )
        
        # ✅ 数据不存在（正常情况）
        elif status_code == 404:
            self.error_stats['not_found'] += 1
            logger.debug(f"资源不存在 (404){context}")
            raise UMLSNotFoundError(f"资源不存在{context}")
        
        # ⏳ 速率限制
        elif status_code == 429:
            self.error_stats['rate_limit_errors'] += 1
            retry_after = response.headers.get('Retry-After', '60')
            raise UMLSRateLimitError(
                f"UMLS API请求过多 (429){context}: "
                f"建议等待{retry_after}秒后重试"
            )
        
        # 🔧 客户端错误（其他4xx）
        elif 400 <= status_code < 500:
            self.error_stats['other_errors'] += 1
            raise UMLSClientError(
                f"UMLS请求错误 ({status_code}){context}: {response.text[:200]}"
            )
        
        # 🔥 服务器错误（5xx）
        elif 500 <= status_code < 600:
            self.error_stats['server_errors'] += 1
            raise UMLSServerError(
                f"UMLS服务器错误 ({status_code}){context}: 可能是临时故障，建议重试"
            )
        
        # ❓ 未知状态码
        else:
            self.error_stats['other_errors'] += 1
            raise UMLSAPIError(
                f"UMLS未知响应 ({status_code}){context}: {response.text[:200]}"
            )
    
    # ==================== LOINC映射 ====================
    
    def _extract_cui_from_concept_url(self, concept_url: str) -> Optional[str]:
        """
        从concept URL中提取CUI
        
        Args:
            concept_url: 例如 'https://uts-ws.nlm.nih.gov/rest/content/current/CUI/C0017725'
        
        Returns:
            CUI（例如 'C0017725'）或None
        """
        if not concept_url:
            return None
        
        # 使用正则表达式提取CUI
        # CUI格式：C + 7位数字
        match = re.search(r'/CUI/(C\d{7})', concept_url)
        if match:
            return match.group(1)
        
        # 如果正则失败，尝试简单分割
        if '/CUI/' in concept_url:
            cui = concept_url.split('/CUI/')[-1]
            # 移除可能的查询参数
            cui = cui.split('?')[0]
            return cui
        
        return None
    
    def find_cui_from_code(
        self, 
        code: str, 
        source_vocabulary: str
    ) -> Optional[str]:
        """
        从源编码查找CUI（修复版）
        
        改进：
        1. ✅ 使用 /content/.../atoms API（而非/crosswalk）
        2. ✅ 从atom的concept字段提取真正的CUI
        3. ✅ 处理多个atoms的情况（选择第一个）
        
        Args:
            code: 源编码（如'2345-7'）
            source_vocabulary: 源词汇表
                - 'LNC' (LOINC)
                - 'ICD9CM' (ICD-9-CM)
                - 'ICD10CM' (ICD-10-CM)
                - 'SNOMEDCT_US' (SNOMED CT)
        
        Returns:
            CUI或None（404情况）
        """
        ticket = self.get_ticket()
        
        # ✅ 修复：使用atoms API而非crosswalk API
        url = f"{self.base_url}/content/current/source/{source_vocabulary}/{code}/atoms"
        params = {'ticket': ticket}
        
        context = f" ({source_vocabulary}:{code})"
        logger.debug(f"查找CUI{context}")
        
        try:
            response = self.session.get(url, params=params, timeout=90)
            
            # 处理401（认证失败）
            if response.status_code == 401:
                logger.warning(f"认证失败{context}，重新获取ticket...")
                self._tgt_url = None
                self._tgt_expiry = 0
                ticket = self.get_ticket()
                params['ticket'] = ticket
                response = self.session.get(url, params=params, timeout=90)
            
            # 统一处理响应
            data = self._handle_response(response, context)
            
            if data:
                results = data.get('result', [])
                
                if results and len(results) > 0:
                    # ✅ 修复：从atom的concept字段提取CUI
                    first_atom = results[0]
                    
                    # concept字段包含完整的concept URL
                    concept_url = first_atom.get('concept')
                    
                    if concept_url:
                        # 从URL中提取CUI
                        cui = self._extract_cui_from_concept_url(concept_url)
                        
                        if cui:
                            logger.debug(f"找到CUI{context} → {cui}")
                            return cui
                        else:
                            logger.warning(f"无法从concept URL提取CUI: {concept_url}")
                    else:
                        logger.warning(f"atom中没有concept字段{context}")
            
            return None
        
        except UMLSNotFoundError:
            # 404是正常情况
            return None
        
        except requests.exceptions.Timeout:
            self.error_stats['network_errors'] += 1
            raise UMLSNetworkError(f"UMLS API请求超时{context}")
        
        except requests.exceptions.ConnectionError as e:
            self.error_stats['network_errors'] += 1
            raise UMLSNetworkError(f"无法连接到UMLS API{context}: {e}")
        
        except UMLSAPIError:
            raise
        
        except Exception as e:
            self.error_stats['other_errors'] += 1
            logger.error(f"UMLS请求未知错误{context}: {e}")
            raise UMLSAPIError(f"UMLS请求失败{context}: {e}")
    
    @smart_retry(max_retries=3, base_delay=2)
    def get_snomed_from_cui(self, cui: str) -> Optional[Tuple[str, str]]:
        """
        从CUI获取SNOMED CT编码
        
        使用crosswalk API将CUI映射到SNOMED CT
        """
        ticket = self.get_ticket()
        
        # 使用crosswalk API从UMLS CUI映射到SNOMED
        url = f"{self.base_url}/crosswalk/current/source/UMLS/{cui}"
        params = {
            'ticket': ticket,
            'targetSource': 'SNOMEDCT_US'
        }
        
        context = f" (CUI:{cui}→SNOMED)"
        logger.debug(f"查找SNOMED{context}")
        
        try:
            response = self.session.get(url, params=params, timeout=90)
            
            # 处理401
            if response.status_code == 401:
                logger.warning(f"认证失败{context}，重新获取ticket...")
                self._tgt_url = None
                self._tgt_expiry = 0
                ticket = self.get_ticket()
                params['ticket'] = ticket
                response = self.session.get(url, params=params, timeout=90)
            
            # 统一处理响应
            data = self._handle_response(response, context)
            
            if data:
                results = data.get('result', [])
                if results and len(results) > 0:
                    result = results[0]
                    snomed_code = result.get('ui')
                    snomed_name = result.get('name')
                    logger.debug(f"找到SNOMED{context}: {snomed_code}")
                    return (snomed_code, snomed_name)
            
            return None
        
        except UMLSNotFoundError:
            return None
        
        except requests.exceptions.Timeout:
            self.error_stats['network_errors'] += 1
            raise UMLSNetworkError(f"UMLS API请求超时{context}")
        
        except requests.exceptions.ConnectionError as e:
            self.error_stats['network_errors'] += 1
            raise UMLSNetworkError(f"无法连接到UMLS API{context}: {e}")
        
        except UMLSAPIError:
            raise
        
        except Exception as e:
            self.error_stats['other_errors'] += 1
            logger.error(f"UMLS请求未知错误{context}: {e}")
            raise UMLSAPIError(f"UMLS请求失败{context}: {e}")
    
    def map_loinc_to_standard(
        self, 
        loinc_code: str, 
        loinc_name: str
    ) -> MappingResult:
        """
        将LOINC编码映射到标准术语
        
        流程：LOINC → CUI → SNOMED CT
        """
        logger.debug(f"映射LOINC: {loinc_code} - {loinc_name}")
        
        try:
            # Step 1: LOINC → CUI（使用修复后的方法）
            cui = self.find_cui_from_code(loinc_code, 'LNC')
            
            if not cui:
                return MappingResult(
                    source_code=loinc_code,
                    source_system='LOINC',
                    source_name=loinc_name,
                    confidence=0.0
                )
            
            # Step 2: 获取CUI详情
            try:
                concept = self._get_concept_details(cui)
                umls_name = concept.name if concept else None
            except Exception as e:
                logger.warning(f"获取CUI详情失败 ({cui}): {e}")
                umls_name = None
            
            # Step 3: CUI → SNOMED CT
            snomed_result = self.get_snomed_from_cui(cui)
            
            if snomed_result:
                snomed_code, snomed_name = snomed_result
                confidence = 1.0
            else:
                snomed_code, snomed_name = None, None
                confidence = 0.5
            
            return MappingResult(
                source_code=loinc_code,
                source_system='LOINC',
                source_name=loinc_name,
                umls_cui=cui,
                umls_name=umls_name,
                snomed_code=snomed_code,
                snomed_name=snomed_name,
                confidence=confidence
            )
        
        except UMLSPermissionError as e:
            logger.error(f"❌ LOINC映射权限错误: {e}")
            raise
        
        except (UMLSRateLimitError, UMLSServerError, UMLSNetworkError) as e:
            logger.warning(f"⚠️ LOINC映射失败 ({loinc_code}): {e}")
            return MappingResult(
                source_code=loinc_code,
                source_system='LOINC',
                source_name=loinc_name,
                confidence=0.0,
                error_message=str(e)
            )
        
        except Exception as e:
            logger.error(f"❌ LOINC映射未知错误 ({loinc_code}): {e}")
            return MappingResult(
                source_code=loinc_code,
                source_system='LOINC',
                source_name=loinc_name,
                confidence=0.0,
                error_message=f"未知错误: {e}"
            )
    
    def map_icd9_to_standard(
        self, 
        icd9_code: str, 
        icd9_description: str
    ) -> MappingResult:
        """将ICD-9编码映射到标准术语"""
        logger.debug(f"映射ICD-9: {icd9_code} - {icd9_description}")
        improved_mapper = ImprovedICD9Mapper(self)
        return improved_mapper.map_icd9_to_standard_improved(
            icd9_code,
            icd9_description
        )
        # try:
        #     # Step 1: ICD-9 → CUI（使用修复后的方法）
        #     cui = self.find_cui_from_code(icd9_code, 'ICD9CM')
            
        #     if not cui:
        #         return MappingResult(
        #             source_code=icd9_code,
        #             source_system='ICD9CM',
        #             source_name=icd9_description,
        #             confidence=0.0
        #         )
            
        #     # Step 2: 获取CUI详情
        #     try:
        #         concept = self._get_concept_details(cui)
        #         umls_name = concept.name if concept else None
        #     except Exception as e:
        #         logger.warning(f"获取CUI详情失败 ({cui}): {e}")
        #         umls_name = None
            
        #     # Step 3: CUI → SNOMED CT
        #     snomed_result = self.get_snomed_from_cui(cui)
            
        #     if snomed_result:
        #         snomed_code, snomed_name = snomed_result
        #         confidence = 1.0
        #     else:
        #         snomed_code, snomed_name = None, None
        #         confidence = 0.5
            
        #     return MappingResult(
        #         source_code=icd9_code,
        #         source_system='ICD9CM',
        #         source_name=icd9_description,
        #         umls_cui=cui,
        #         umls_name=umls_name,
        #         snomed_code=snomed_code,
        #         snomed_name=snomed_name,
        #         confidence=confidence
        #     )
        
        # except UMLSPermissionError as e:
        #     logger.error(f"❌ ICD-9映射权限错误: {e}")
        #     raise
        
        # except (UMLSRateLimitError, UMLSServerError, UMLSNetworkError) as e:
        #     logger.warning(f"⚠️ ICD-9映射失败 ({icd9_code}): {e}")
        #     return MappingResult(
        #         source_code=icd9_code,
        #         source_system='ICD9CM',
        #         source_name=icd9_description,
        #         confidence=0.0,
        #         error_message=str(e)
        #     )
        
        # except Exception as e:
        #     logger.error(f"❌ ICD-9映射未知错误 ({icd9_code}): {e}")
        #     return MappingResult(
        #         source_code=icd9_code,
        #         source_system='ICD9CM',
        #         source_name=icd9_description,
        #         confidence=0.0,
        #         error_message=f"未知错误: {e}"
        #     )
    
    # ==================== 批量映射 ====================
    
    # def map_batch_loinc(
    #     self, 
    #     loinc_pairs: List[Tuple[str, str]],
    #     show_progress: bool = True,
    #     stop_on_permission_error: bool = True
    # ) -> List[MappingResult]:
    #     """批量映射LOINC编码"""
    #     results = []
    #     total = len(loinc_pairs)
        
    #     logger.info(f"批量映射LOINC: {total} 个")
        
    #     for i, (code, name) in enumerate(loinc_pairs, 1):
    #         if show_progress and i % 10 == 0:
    #             logger.info(f"  进度: {i}/{total}")
            
    #         try:
    #             result = self.map_loinc_to_standard(code, name)
    #             results.append(result)
            
    #         except UMLSPermissionError:
    #             if stop_on_permission_error:
    #                 logger.error(f"❌ 遇到权限错误，停止批量处理 ({i}/{total})")
    #                 raise
    #             else:
    #                 logger.warning(f"⚠️ 遇到权限错误，跳过此项 ({i}/{total})")
    #                 results.append(MappingResult(
    #                     source_code=code,
    #                     source_system='LOINC',
    #                     source_name=name,
    #                     confidence=0.0,
    #                     error_message="权限错误"
    #                 ))
        
    #     # 统计
    #     successful = sum(1 for r in results if r.confidence >= 0.5)
    #     errors = sum(1 for r in results if r.error_message)
        
    #     logger.info(
    #         f"✅ LOINC映射完成: "
    #         f"成功={successful}/{total} ({successful/total*100:.1f}%), "
    #         f"错误={errors}"
    #     )
        
    #     return results
    
    # def map_batch_icd9(
    #     self, 
    #     icd9_pairs: List[Tuple[str, str]],
    #     show_progress: bool = True,
    #     stop_on_permission_error: bool = True
    # ) -> List[MappingResult]:
    #     """批量映射ICD-9编码"""
    #     results = []
    #     total = len(icd9_pairs)
        
    #     logger.info(f"批量映射ICD-9: {total} 个")
        
    #     for i, (code, description) in enumerate(icd9_pairs, 1):
    #         if show_progress and i % 10 == 0:
    #             logger.info(f"  进度: {i}/{total}")
            
    #         try:
    #             result = self.map_icd9_to_standard(code, description)
    #             results.append(result)
            
    #         except UMLSPermissionError:
    #             if stop_on_permission_error:
    #                 logger.error(f"❌ 遇到权限错误，停止批量处理 ({i}/{total})")
    #                 raise
    #             else:
    #                 logger.warning(f"⚠️ 遇到权限错误，跳过此项 ({i}/{total})")
    #                 results.append(MappingResult(
    #                     source_code=code,
    #                     source_system='ICD9CM',
    #                     source_name=description,
    #                     confidence=0.0,
    #                     error_message="权限错误"
    #                 ))
        
    #     # 统计
    #     successful = sum(1 for r in results if r.confidence >= 0.5)
    #     errors = sum(1 for r in results if r.error_message)
        
    #     logger.info(
    #         f"✅ ICD-9映射完成: "
    #         f"成功={successful}/{total} ({successful/total*100:.1f}%), "
    #         f"错误={errors}"
    #     )
        
    #     return results
    
    # def save_mapping_results(
    #     self, 
    #     results: List[MappingResult], 
    #     output_path: Path
    # ):
    #     """保存映射结果为CSV"""
    #     import pandas as pd
        
    #     data = []
    #     for r in results:
    #         data.append({
    #             'source_code': r.source_code,
    #             'source_system': r.source_system,
    #             'source_name': r.source_name,
    #             'umls_cui': r.umls_cui,
    #             'umls_name': r.umls_name,
    #             'snomed_code': r.snomed_code,
    #             'snomed_name': r.snomed_name,
    #             'confidence': r.confidence,
    #             'error_message': r.error_message
    #         })
        
    #     df = pd.DataFrame(data)
    #     df.to_csv(output_path, index=False)
        
    #     logger.info(f"✅ 映射结果已保存: {output_path}")
    
    # def print_error_summary(self):
    #     """打印错误统计摘要"""
    #     print("\n" + "="*80)
    #     print("UMLS API 错误统计")
    #     print("="*80)
    #     print(f"权限错误 (403):   {self.error_stats['permission_errors']}")
    #     print(f"速率限制 (429):   {self.error_stats['rate_limit_errors']}")
    #     print(f"服务器错误 (5xx): {self.error_stats['server_errors']}")
    #     print(f"网络错误:         {self.error_stats['network_errors']}")
    #     print(f"数据不存在 (404): {self.error_stats['not_found']}")
    #     print(f"其他错误:         {self.error_stats['other_errors']}")
    #     print("="*80)
        

class ImprovedICD9Mapper:
    """改进的ICD-9映射器"""
    
    def __init__(self, base_mapper):
        """
        Args:
            base_mapper: EnhancedUMLSMapper实例
        """
        self.mapper = base_mapper
        self.base_url = base_mapper.base_url
        self.session = base_mapper.session
    
    def _normalize_icd9_formats(self, code: str) -> List[str]:
        """
        生成ICD-9代码的多种可能格式
        
        Args:
            code: 输入的ICD-9代码（如'250.00'）
        
        Returns:
            可能的格式列表
        
        Examples:
            '250.00' → ['250.00', '25000', '250.0', '250']
            '585.3' → ['585.3', '5853', '585']
            '401.9' → ['401.9', '4019', '401']
        """
        formats = []
        
        # 1. 原格式
        formats.append(code)
        
        # 2. 去掉小数点
        if '.' in code:
            formats.append(code.replace('.', ''))
        
        # 3. 去掉尾部的0（但保留小数点）
        stripped = code.rstrip('0').rstrip('.')
        if stripped and stripped != code:
            formats.append(stripped)
        
        # 4. 只保留整数部分
        if '.' in code:
            integer_part = code.split('.')[0]
            if integer_part not in formats:
                formats.append(integer_part)
        
        # 5. 补齐到5位（如果少于5位）
        no_dot = code.replace('.', '')
        if len(no_dot) < 5:
            padded = no_dot.ljust(5, '0')
            if padded not in formats:
                formats.append(padded)
        
        # 去重并保持顺序
        seen = set()
        unique_formats = []
        for f in formats:
            if f not in seen:
                seen.add(f)
                unique_formats.append(f)
        
        return unique_formats
    
    def _try_atoms_api(
        self, 
        code: str, 
        source_vocabulary: str = 'ICD9CM'
    ) -> Optional[str]:
        """
        尝试使用atoms API获取CUI
        
        会尝试多种代码格式
        """
        formats = self._normalize_icd9_formats(code)
        
        logger.debug(f"尝试{len(formats)}种格式: {formats}")
        
        for test_code in formats:
            try:
                cui = self.mapper.find_cui_from_code(test_code, source_vocabulary)
                
                if cui:
                    if test_code != code:
                        logger.info(
                            f"✅ ICD-9格式转换成功: '{code}' → '{test_code}' → {cui}"
                        )
                    return cui
            
            except Exception as e:
                logger.debug(f"格式'{test_code}'失败: {e}")
                continue
        
        return None
    
    def _try_search_api(
        self, 
        code: str, 
        description: Optional[str] = None
    ) -> Optional[str]:
        """
        使用search API作为后备方案
        
        Args:
            code: ICD-9代码
            description: 可选的描述文本（可以提高准确性）
        
        Returns:
            CUI或None
        """
        ticket = self.mapper.get_ticket()
        
        # 方法1: 精确搜索代码
        url = f"{self.base_url}/search/current"
        
        # 尝试不同的格式
        for test_code in self._normalize_icd9_formats(code):
            params = {
                'ticket': ticket,
                'string': test_code,
                'inputType': 'sourceUi',
                'searchType': 'exact',
                'sabs': 'ICD9CM'
            }
            
            try:
                response = self.session.get(url, params=params, timeout=90)
                
                if response.status_code == 200:
                    data = response.json()
                    results = data.get('result', {}).get('results', [])
                    
                    if results:
                        # 从第一个结果提取CUI
                        first_result = results[0]
                        uri = first_result.get('uri', '')
                        
                        # 从URI中提取CUI
                        match = re.search(r'/CUI/(C\d{7})', uri)
                        if match:
                            cui = match.group(1)
                            logger.info(
                                f"✅ Search API成功: '{code}' (格式'{test_code}') → {cui}"
                            )
                            return cui
            
            except Exception as e:
                logger.debug(f"Search API失败 ('{test_code}'): {e}")
                continue
        
        # 方法2: 如果有描述，尝试搜索描述
        if description:
            params = {
                'ticket': ticket,
                'string': description,
                'searchType': 'words',
                'sabs': 'ICD9CM',
                'pageSize': 5
            }
            
            try:
                response = self.session.get(url, params=params, timeout=90)
                
                if response.status_code == 200:
                    data = response.json()
                    results = data.get('result', {}).get('results', [])
                    
                    # 在结果中查找匹配的代码
                    for result in results:
                        result_ui = result.get('ui', '')
                        
                        # 检查是否匹配原代码
                        if self._codes_match(code, result_ui):
                            uri = result.get('uri', '')
                            match = re.search(r'/CUI/(C\d{7})', uri)
                            if match:
                                cui = match.group(1)
                                logger.info(
                                    f"✅ Description Search成功: '{code}' → {cui}"
                                )
                                return cui
            
            except Exception as e:
                logger.debug(f"Description Search失败: {e}")
        
        return None
    
    def _codes_match(self, code1: str, code2: str) -> bool:
        """
        检查两个ICD-9代码是否匹配
        
        考虑不同的格式变体
        """
        # 标准化两个代码
        norm1 = code1.replace('.', '').rstrip('0')
        norm2 = code2.replace('.', '').rstrip('0')
        
        return norm1 == norm2
    
    def map_icd9_to_standard_improved(
        self,
        icd9_code: str,
        icd9_description: str
    ):
        """
        改进的ICD-9映射方法
        
        尝试策略：
        1. 使用atoms API + 多种格式
        2. 使用search API（代码精确搜索）
        3. 使用search API（描述搜索）
        
        Args:
            icd9_code: ICD-9代码
            icd9_description: ICD-9描述
        
        Returns:
            MappingResult
        """
        logger.debug(f"映射ICD-9: {icd9_code} - {icd9_description}")
        
        cui = None
        method_used = None
        
        # 策略1: atoms API（尝试多种格式）
        logger.debug("策略1: atoms API + 多格式")
        cui = self._try_atoms_api(icd9_code, 'ICD9CM')
        if cui:
            method_used = "atoms_api"
        
        # 策略2: search API（如果atoms失败）
        if not cui:
            logger.debug("策略2: search API")
            cui = self._try_search_api(icd9_code, icd9_description)
            if cui:
                method_used = "search_api"
        
        # 如果所有方法都失败
        if not cui:
            logger.warning(
                f"⚠️ 所有方法都无法找到CUI: {icd9_code} - {icd9_description}"
            )
            logger.warning(
                f"   尝试的格式: {self._normalize_icd9_formats(icd9_code)}"
            )
            
            return MappingResult(
                source_code=icd9_code,
                source_system='ICD9CM',
                source_name=icd9_description,
                confidence=0.0,
                error_message="未找到对应的CUI（已尝试多种格式和API）"
            )
        
        # 获取CUI详情
        try:
            concept = self.mapper._get_concept_details(cui)
            umls_name = concept.name if concept else None
        except Exception as e:
            logger.warning(f"获取CUI详情失败 ({cui}): {e}")
            umls_name = None
        
        # 获取SNOMED映射
        try:
            snomed_result = self.mapper.get_snomed_from_cui(cui)
            if snomed_result:
                snomed_code, snomed_name = snomed_result
                confidence = 1.0
            else:
                snomed_code, snomed_name = None, None
                confidence = 0.5
        except Exception as e:
            logger.warning(f"获取SNOMED失败 ({cui}): {e}")
            snomed_code, snomed_name = None, None
            confidence = 0.5
        
        logger.info(
            f"✅ ICD-9映射成功: {icd9_code} → {cui} (方法: {method_used})"
        )
        
        return MappingResult(
            source_code=icd9_code,
            source_system='ICD9CM',
            source_name=icd9_description,
            umls_cui=cui,
            umls_name=umls_name,
            snomed_code=snomed_code,
            snomed_name=snomed_name,
            confidence=confidence
        )        

class EnhancedUMLSMapperWithCache(EnhancedUMLSMapper):
    """
    带缓存的增强UMLS映射器
    
    继承自EnhancedUMLSMapper，添加缓存功能：
    1. 自动缓存所有API查询结果
    2. 优先从缓存读取，避免重复API调用
    3. 持久化缓存，跨会话复用
    4. 提供缓存统计和管理功能
    
    兼容原版：
    - API接口完全一致
    - 无需修改调用代码
    - 只需替换类名即可
    """
    
    def __init__(
        self,
        api_key: str,
        filter_semantic_types: bool = False,
        cache_dir: str = 'cache/umls',
        enable_cache: bool = True,
        cache_expiry_days: Optional[int] = None,
        max_cache_size: int = 50000,
        verbose: bool = False
    ):
        """
        初始化带缓存的UMLS映射器
        
        Args:
            api_key: UMLS API密钥
            filter_semantic_types: 是否过滤语义类型
            cache_dir: 缓存目录
            enable_cache: 是否启用缓存
            cache_expiry_days: 缓存过期天数（None=永不过期）
            max_cache_size: 最大缓存条目数
            verbose: 是否打印详细日志
        """
        # 初始化父类
        super().__init__(api_key, filter_semantic_types)
        
        # 初始化缓存
        self.enable_cache = enable_cache
        self.verbose = verbose
        
        if self.enable_cache:
            self.cache = UMLSCache(
                cache_dir=cache_dir,
                max_memory_size=max_cache_size,
                auto_save_interval=100,
                cache_expiry_days=cache_expiry_days,
                enable_persistent=True
            )
            
            logger.info(f"✅ 缓存系统已启用: {cache_dir}")
            
            # 加载现有缓存统计
            stats = self.cache.get_stats()
            if stats['cache_size'] > 0:
                logger.info(f"   已加载 {stats['cache_size']} 条缓存记录")
        else:
            self.cache = None
            logger.info("⚠️  缓存系统已禁用")
    
    def map_loinc_to_standard(
        self,
        loinc_code: str,
        loinc_name: str
    ) -> MappingResult:
        """
        映射LOINC到标准术语（带缓存）
        
        Args:
            loinc_code: LOINC代码
            loinc_name: LOINC名称
        
        Returns:
            MappingResult对象
        """
        # 尝试从缓存获取
        if self.enable_cache:
            cached_result = self.cache.get('loinc', loinc_code)
            
            if cached_result is not None:
                if self.verbose:
                    logger.debug(f"✅ 缓存命中: LOINC {loinc_code}")
                return cached_result
        
        # 缓存未命中，调用原版API
        if self.verbose:
            logger.debug(f"🔍 缓存未命中，查询API: LOINC {loinc_code}")
        
        result = super().map_loinc_to_standard(loinc_code, loinc_name)
        
        # 保存到缓存
        if self.enable_cache and result is not None:
            self.cache.set('loinc', loinc_code, result)
        
        return result
    
    def map_icd9_to_standard(
        self,
        icd9_code: str,
        icd9_description: str
    ) -> MappingResult:
        """
        映射ICD-9到标准术语（带缓存）
        
        Args:
            icd9_code: ICD-9代码
            icd9_description: ICD-9描述
        
        Returns:
            MappingResult对象
        """
        # 尝试从缓存获取
        if self.enable_cache:
            cached_result = self.cache.get('icd9', icd9_code)
            
            if cached_result is not None:
                if self.verbose:
                    logger.debug(f"✅ 缓存命中: ICD-9 {icd9_code}")
                return cached_result
        
        # 缓存未命中，调用原版API
        if self.verbose:
            logger.debug(f"🔍 缓存未命中，查询API: ICD-9 {icd9_code}")
        
        result = super().map_icd9_to_standard(icd9_code, icd9_description)
        
        # 保存到缓存
        if self.enable_cache and result is not None:
            self.cache.set('icd9', icd9_code, result)
        
        return result
    
    def find_cui_from_code(
        self,
        code: str,
        source_vocabulary: str
    ) -> Optional[str]:
        """
        从代码查找CUI（带缓存）
        
        Args:
            code: 源代码
            source_vocabulary: 源词汇表（如'ICD10CM'）
        
        Returns:
            CUI字符串
        """
        cache_key = f"{source_vocabulary.lower()}_{code}"
        
        # 尝试从缓存获取
        if self.enable_cache:
            cached_result = self.cache.get('cui', cache_key)
            
            if cached_result is not None:
                if self.verbose:
                    logger.debug(f"✅ 缓存命中: CUI查询 {source_vocabulary}:{code}")
                return cached_result
        
        # 缓存未命中，调用原版API
        if self.verbose:
            logger.debug(f"🔍 缓存未命中，查询API: CUI {source_vocabulary}:{code}")
        
        result = super().find_cui_from_code(code, source_vocabulary)
        
        # 保存到缓存
        if self.enable_cache and result is not None:
            self.cache.set('cui', cache_key, result)
        
        return result
    
    def get_snomed_from_cui(
        self,
        cui: str
    ) -> Optional[Tuple[str, str]]:
        """
        从CUI获取SNOMED-CT（带缓存）
        
        Args:
            cui: UMLS CUI
        
        Returns:
            (SNOMED代码, SNOMED名称) 元组
        """
        # 尝试从缓存获取
        if self.enable_cache:
            cached_result = self.cache.get('cui_to_snomed', cui)
            
            if cached_result is not None:
                if self.verbose:
                    logger.debug(f"✅ 缓存命中: SNOMED查询 CUI {cui}")
                return cached_result
        
        # 缓存未命中，调用原版API
        if self.verbose:
            logger.debug(f"🔍 缓存未命中，查询API: SNOMED CUI {cui}")
        
        result = super().get_snomed_from_cui(cui)
        
        # 保存到缓存
        if self.enable_cache and result is not None:
            self.cache.set('cui_to_snomed', cui, result)
        
        return result
    
    # ==================== 缓存管理方法 ====================
    
    def get_cache_stats(self) -> dict:
        """
        获取缓存统计信息
        
        Returns:
            缓存统计字典
        """
        if not self.enable_cache:
            return {'cache_enabled': False}
        
        stats = self.cache.get_stats()
        
        # 计算节省的时间（假设每次API调用200ms）
        time_saved_seconds = stats['api_calls_saved'] * 0.2
        time_saved_minutes = time_saved_seconds / 60
        
        stats['estimated_time_saved_seconds'] = round(time_saved_seconds, 2)
        stats['estimated_time_saved_minutes'] = round(time_saved_minutes, 2)
        
        return stats
    
    def print_cache_stats(self) -> None:
        """打印缓存统计信息"""
        if not self.enable_cache:
            print("缓存未启用")
            return
        
        stats = self.get_cache_stats()
        
        print("\n" + "=" * 70)
        print("UMLS映射缓存统计")
        print("=" * 70)
        print(f"缓存状态:        {'启用' if self.enable_cache else '禁用'}")
        print(f"总查询次数:      {stats['total_queries']}")
        print(f"缓存命中:        {stats['cache_hits']}")
        print(f"缓存未命中:      {stats['cache_misses']}")
        print(f"命中率:          {stats['hit_rate']:.2f}%")
        print(f"节省API调用:     {stats['api_calls_saved']} 次")
        print(f"节省时间:        {stats['estimated_time_saved_minutes']:.1f} 分钟 (~{stats['estimated_time_saved_seconds']:.0f}秒)")
        print(f"当前缓存大小:    {stats['cache_size']} 条目")
        print(f"内存使用:        {stats['memory_usage_mb']} MB")
        print("=" * 70)
    
    def clear_cache(self, mapping_type: Optional[str] = None) -> int:
        """
        清空缓存
        
        Args:
            mapping_type: 指定清空的类型（'loinc', 'icd9', etc.），None表示全部
        
        Returns:
            清空的条目数
        """
        if not self.enable_cache:
            logger.warning("缓存未启用")
            return 0
        
        count = self.cache.clear(mapping_type)
        logger.info(f"已清空缓存: {count} 条目")
        
        return count
    
    def save_cache(self) -> None:
        """手动保存缓存到磁盘"""
        if self.enable_cache:
            self.cache.save()
            logger.info("缓存已手动保存")
    
    def __del__(self):
        """析构函数：自动保存缓存"""
        if self.enable_cache and self.cache:
            try:
                self.cache.save()
                logger.info("缓存已自动保存")
            except:
                pass

# ==================== 使用示例 ====================

def main():
    """使用示例"""
    from agents.config import APIConfig
    
    print("\n" + "="*80)
    print("Enhanced UMLS Mapper v2.0 测试（改进错误处理）")
    print("="*80)
    
    api_key = APIConfig.UMLS_KEY
    
    if not api_key:
        print("\n⚠️  请在agents/config.py中设置UMLS_KEY")
        return
    
    mapper = EnhancedUMLSMapper(api_key=api_key)
    
    # 测试LOINC映射
    print("\n" + "="*80)
    print("测试 LOINC → UMLS → SNOMED CT")
    print("="*80)
    
    loinc_examples = [
        ('2345-7', 'Glucose [Mass/volume] in Serum or Plasma'),
        ('4548-4', 'Hemoglobin A1c/Hemoglobin.total in Blood'),
        ('2160-0', 'Creatinine [Mass/volume] in Serum or Plasma')
    ]
    
    try:
        loinc_results = mapper.map_batch_loinc(
            loinc_examples,
            stop_on_permission_error=False  # 测试时继续处理
        )
        
        print("\nLOINC映射结果:")
        for r in loinc_results:
            print(f"\n  {r.source_code} - {r.source_name}")
            if r.error_message:
                print(f"    ❌ 错误: {r.error_message}")
            else:
                if r.umls_cui:
                    print(f"    CUI: {r.umls_cui}")
                    print(f"    UMLS名称: {r.umls_name}")
                if r.snomed_code:
                    print(f"    SNOMED: {r.snomed_code} - {r.snomed_name}")
                print(f"    置信度: {r.confidence}")
    
    except UMLSPermissionError as e:
        print(f"\n❌ 权限错误: {e}")
        print("请检查UMLS API key的权限设置")
    
    # 打印错误统计
    mapper.print_error_summary()
    
    print("\n✅ 测试完成！")
    
    # # 保存结果
    # output_dir = Path('/mnt/user-data/outputs')
    # output_dir.mkdir(exist_ok=True)
    
    # mapper.save_mapping_results(
    #     loinc_results, 
    #     output_dir / 'loinc_mappings_test.csv'
    # )
    
    # mapper.save_mapping_results(
    #     icd9_results,
    #     output_dir / 'icd9_mappings_test.csv'
    # )
    
    print("\n✅ 测试完成！")


if __name__ == '__main__':
    main()
