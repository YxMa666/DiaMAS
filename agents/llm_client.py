"""
LLM 客户端
封装 Deepseek API 调用，提供医学推理能力
"""
import requests
import json
from typing import List, Dict, Optional
import warnings
warnings.filterwarnings('ignore')  # 忽略SSL警告

# 导入配置
from agents.config import APIConfig, ModelConfig
import logging
logger = logging.getLogger(__name__)

class LLMResponseError(RuntimeError):
    """LLM 调用失败或响应不可解析。

    历史教训:本类替代了旧的 `return "[错误] ..."` 哨兵值。
    哨兵值会被下游 `re.search(r"\\[.*\\]")` 匹配成 JSON 数组,
    经 json_repair 修成 ["错误"],逐项 isinstance(dict) 失败后 continue,
    最终返回 [] —— 恒 404 的 Judge 因此表现为"零冲突"达 6,951 例。
    """

def model_thinking(data: Dict, model: str) -> Dict:
    if model == "deepseek-v3.2":
        data["thinking"] = {
            "type": "enabled"  # 是否开启深度思考模式，默认关闭
        }
    elif model == "qwen3-30b-a3b":
        data["chat_template_kwargs"] = {
            "enable_thinking": True  # 是否开启深度思考模式，默认开启
        }
    elif model == "mimo-v2.5-pro":
        data["thinking"] = {
            "type": "enabled"  # 是否开启深度思考模式，默认关闭
        }
    elif model == "deepseek-v4-flash":
        data["thinking"] = {
            "type": "enabled"  # 是否开启深度思考模式，默认关闭
        }
    elif model == "deepseek-v4-pro":
        data["thinking"] = {
            "type": "enabled"  # 是否开启深度思考模式，默认关闭
        }
    else:
        data = data
    return data

class DeepseekClient:
    """
    Deepseek API 客户端
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None
    ):
        """
        初始化客户端
        
        Args:
            api_key: API密钥（默认从配置读取）
            base_url: API地址（默认从配置读取）
        """
        self.api_key = api_key or APIConfig.API_KEY
        self.base_url = base_url or APIConfig.BASE_URL
        self.timeout = APIConfig.TIMEOUT
        self.max_retries = APIConfig.MAX_RETRIES
        
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
        stream: bool = False,
        deep_thinking: bool = False
    ) -> Dict:
        """
        发送对话请求
        
        Args:
            messages: 消息列表
            model: 模型名称（默认使用对话模型）
            temperature: 温度参数
            max_tokens: 最大token数
            stream: 是否流式输出
            deep_thinking: 是否启用深度思考
        
        Returns:
            API响应字典
        """
   
        data = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        
        if deep_thinking:
            data = model_thinking(data, model)
        
        # 重试逻辑
        for attempt in range(self.max_retries):
            try:
                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    data=json.dumps(data),
                    verify=APIConfig.SSL_VERIFY,
                    timeout=self.timeout
                )
                
                if response.status_code == 200:
                    return response.json()
                else:
                    if attempt < self.max_retries - 1:
                        print(f"   ⚠️  请求失败 (状态码: {response.status_code})，重试中...")
                        import time
                        time.sleep(APIConfig.RETRY_DELAY)
                        continue
                    else:
                        return {
                            "error": True,
                            "status_code": response.status_code,
                            "message": response.text
                        }
            
            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    print(f"   ⚠️  网络错误，重试中...")
                    import time
                    time.sleep(APIConfig.RETRY_DELAY)
                    continue
                else:
                    return {
                        "error": True,
                        "message": f"请求失败: {str(e)}"
                    }
        
        return {"error": True, "message": "超过最大重试次数"}
    
    def chat_for_task(
        self,
        task: str,
        messages: List[Dict[str, str]],
        stream: bool = False
    ) -> Dict:
        """
        ✅ 新增：根据任务类型自动选择配置并调用LLM
        
        Args:
            task: 任务类型 ("parser", "intent", "diagnosis", "therapy", "chat", "quick")
            messages: 消息列表
            stream: 是否流式输出
        
        Returns:
            API响应字典
        
        Example:
            >>> client = DeepseekClient()
            >>> messages = [{"role": "user", "content": "解析病历..."}]
            >>> response = client.chat_for_task("parser", messages)
        """
        config = ModelConfig.get_config_for_task(task)
        
        return self.chat(
            messages=messages,
            model=config["model"],
            temperature=config["temperature"],
            max_tokens=config["max_tokens"],
            stream=stream,
            deep_thinking=config["deep_thinking"]
        )
    
    def get_response_text(self, response: Dict) -> str:
        """提取响应文本。任何失败都 raise，不返回哨兵值。

        先 logger.error 再 raise:即使调用方吞了异常,日志里也一定留痕。
        """
        if response.get("error"):
            msg = str(response.get("message", "未知错误"))
            logger.error("[LLM-FAIL] api error: %s", msg)
            raise LLMResponseError(msg)
        try:
            content = response["choices"][0]["message"].get("content")
        except (KeyError, IndexError) as e:
            logger.error("[LLM-FAIL] 响应结构异常: %.300r", response)
            raise LLMResponseError("无法解析响应内容") from e
        if not content:
            # 注意:reasoning_content 是思考链,不是答案,不可作为兜底
            logger.error("[LLM-FAIL] content 为空(finish_reason=%s)",
                         (response.get("choices") or [{}])[0].get("finish_reason"))
            raise LLMResponseError("content 为空")
        return content
# ==========================================
# 医学专用提示词模板
# ==========================================

class MedicalPrompts:
    """医学推理提示词库"""
    
    @staticmethod
    def diagnosis_prompt(profile: Dict) -> str:
        """诊断推理提示词"""
        return f"""你是一位内分泌科专家医生。请根据以下患者信息进行诊断分析：

患者档案：
- 年龄: {profile.get('age', '未知')} 岁
- 性别: {profile.get('gender', '未知')}
- BMI: {profile.get('BMI', '未知')}
- 糖化血红蛋白: {profile.get('HbA1c', '未测量')}%
- 空腹血糖: {profile.get('fasting_glucose', '未测量')} mmol/L
- C肽释放: {profile.get('c_peptide_fasting', '未知')} → {profile.get('c_peptide_120min', '未知')} pmol/L
- 症状: {', '.join(profile.get('symptoms', ['无']))}
- 并发症线索: {', '.join(profile.get('known_complications', ['无']))}
- 当前用药: {', '.join([m['name'] for m in profile.get('current_medications', [])])}

请按以下格式回答：
1. **诊断结论**: 是否为糖尿病？如果是，属于哪种类型？
2. **诊断依据**: 列出3-5条关键诊断依据
3. **并发症评估**: 识别已存在或高危的并发症
4. **需要补充的检查**: 如果信息不足，列出需要的检查

要求：
- 遵循中国糖尿病指南
- 使用医学术语但保持清晰
- 如果数据缺失，明确指出"""

    @staticmethod
    def therapy_prompt(profile: Dict, diagnosis: Dict) -> str:
        """治疗方案提示词"""
        return f"""你是一位内分泌科临床药师。请根据诊断结果制定个性化治疗方案：

诊断信息：
- 诊断: {diagnosis.get('subtype', '未确诊')}
- 并发症: {', '.join(diagnosis.get('complications', ['无']))}

患者信息：
- 年龄: {profile.get('age')} 岁
- 肾功能指标: 尿蛋白 {profile.get('urine_protein_24h', '未测')} mg/day
- 肝功能: 白蛋白 {profile.get('albumin', '未测')} g/L
- 心血管: 血压 {profile.get('blood_pressure_systolic', '?')}/{profile.get('blood_pressure_diastolic', '?')} mmHg
- 当前用药: {', '.join([m['name'] for m in profile.get('current_medications', [])])}

请按以下格式提供方案：
1. **推荐药物**: 列出所有必需药物，包括：
   - 药物名称
   - 作用机制
   - 推荐理由
   
2. **用药注意事项**: 针对患者特点的安全提示

3. **需要调整的现有用药**: 分析当前用药的合理性

4. **非药物治疗**: 饮食、运动建议

要求：
- 考虑器官功能
- 避免药物相互作用
- 个性化剂量调整"""

    @staticmethod
    def clarification_prompt(state: Dict) -> str:
        """信息澄清提示词"""
        missing = state.get('missing_info', [])
        return f"""你是一位医生助手。当前诊疗流程中发现以下信息缺失：

缺失信息: {', '.join(missing)}

请生成一段简洁、专业的问询语句，询问医生或患者如何补充这些信息。
要求：
- 一次只询问1-2个最关键的信息
- 说明为什么需要这些信息
- 语气友好、专业"""

    @staticmethod
    def conversation_prompt(user_input: str, context: Dict) -> str:
        """多轮对话提示词"""
        return f"""你是一位医疗AI助手，正在协助医生进行糖尿病患者的诊疗。

当前诊疗状态：
- 患者ID: {context.get('patient_id', '未知')}
- 已完成诊断: {'是' if context.get('diagnosis_summary') else '否'}
- 已制定方案: {'是' if context.get('therapy_plan') else '否'}

医生提问: {user_input}

请根据上下文回答医生的问题。如果问题与当前患者相关，结合患者信息回答。
保持专业、简洁，并提供医学依据。"""

# ==========================================
# 医学推理引擎
# ==========================================

class MedicalReasoningEngine:
    """
    基于LLM的医学推理引擎
    """
    
    def __init__(self, llm_client: DeepseekClient):
        self.llm = llm_client
        self.prompts = MedicalPrompts()
    
    def diagnose_with_llm(self, profile: Dict) -> Dict:
        """
        使用LLM进行诊断推理
        
        Returns:
            诊断结果字典
        """
        prompt = self.prompts.diagnosis_prompt(profile)
        
        messages = [
            {"role": "system", "content": "你是一位经验丰富的内分泌科医生，擅长糖尿病诊断。"},
            {"role": "user", "content": prompt}
        ]
        
        response = self.llm.chat(
            messages,
            model=ModelConfig.DIAGNOSIS_MODEL,
            temperature=ModelConfig.DIAGNOSIS_TEMPERATURE,
            max_tokens=ModelConfig.DIAGNOSIS_MAX_TOKENS,
            deep_thinking=ModelConfig.DIAGNOSIS_DEEP_THINKING
        )
        diagnosis_text = self.llm.get_response_text(response)
        
        # 解析LLM输出（简化版，实际可用结构化输出）
        result = {
            "raw_diagnosis": diagnosis_text,
            "is_diabetic": "糖尿病" in diagnosis_text and "不是" not in diagnosis_text,
            "llm_reasoning": diagnosis_text
        }
        
        return result
    
    def recommend_therapy_with_llm(self, profile: Dict, diagnosis: Dict) -> Dict:
        """
        使用LLM生成治疗方案
        
        Returns:
            治疗方案字典
        """
        prompt = self.prompts.therapy_prompt(profile, diagnosis)
        
        messages = [
            {"role": "system", "content": "你是一位临床药师，精通糖尿病药物治疗。"},
            {"role": "user", "content": prompt}
        ]
        
        # 使用治疗专用配置
        response = self.llm.chat(
            messages,
            model=ModelConfig.THERAPY_MODEL,
            temperature=ModelConfig.THERAPY_TEMPERATURE,
            max_tokens=ModelConfig.THERAPY_MAX_TOKENS,
            deep_thinking=ModelConfig.THERAPY_DEEP_THINKING
        )
        therapy_text = self.llm.get_response_text(response)
        
        result = {
            "raw_therapy": therapy_text,
            "llm_recommendation": therapy_text
        }
        
        return result
    
    def chat_with_doctor(self, user_input: str, context: Dict) -> str:
        """
        与医生进行多轮对话
        
        Args:
            user_input: 医生的输入
            context: 当前诊疗上下文
        
        Returns:
            AI的回复
        """
        prompt = self.prompts.conversation_prompt(user_input, context)
        
        messages = [
            {"role": "system", "content": "你是Dia-MAS智能诊疗助手，协助医生进行决策。"},
            {"role": "user", "content": prompt}
        ]
        
        # 使用对话专用配置
        response = self.llm.chat(
            messages,
            model=ModelConfig.CHAT_MODEL,
            temperature=ModelConfig.CHAT_TEMPERATURE,
            max_tokens=ModelConfig.CHAT_MAX_TOKENS,
            deep_thinking=ModelConfig.CHAT_DEEP_THINKING
        )
        
        return self.llm.get_response_text(response)

# ==========================================
# RAG专用提示词模板
# ==========================================

class RAGPrompts:
    """RAG系统提示词库"""
    
    @staticmethod
    def metadata_extraction_prompt(filename: str, content: str) -> str:
        """指南元数据提取提示词"""
        return f"""从以下医疗指南文本中提取元数据：

**文件名：** {filename}

**文本内容（前3000字符）：**
{content}

**请提取以下信息（JSON格式）：**
````json
{{
  "full_name": "指南完整名称",
  "short_name": "简称",
  "version": "版本/年份",
  "authority": "发布机构（如：中华医学会、ADA等）",
  "publish_date": "发布日期（YYYY-MM格式）",
  "language": "zh-CN 或 en-US",
  "country": "China 或 USA等",
  "specialty": "专科（如：内分泌科）",
  "disease_focus": ["主要疾病1", "主要疾病2"],
  "document_type": "指南 或 共识 或 标准"
}}
````

注意：
1. 如果某些信息无法从文本中获取，请设为null
2. 严格按JSON格式输出，不要有其他文字
3. full_name应包含版本信息"""
    
    @staticmethod
    def semantic_boundary_prompt(text_segment: str, guideline_name: str) -> str:
        """语义边界识别提示词"""
        return f"""分析以下医疗指南片段，识别所有**语义完整的单元**。

**指南名称：** {guideline_name}

**文本片段：**
{text_segment[:1500]}

**语义单元类型：**
1. drug_entry - 药物条目（必须包含：适应症+禁忌症+剂量）
2. diagnostic_criteria - 诊断标准（必须包含所有诊断条件）
3. treatment_algorithm - 治疗流程
4. table - 表格（包括表格标题+说明）
5. general - 一般性内容

**输出格式（JSON）：**
````json
{{
  "semantic_units": [
    {{
      "type": "drug_entry",
      "start_marker": "## 3.2.1 二甲双胍",
      "end_marker": "## 3.2.2",
      "description": "二甲双胍的完整用药信息",
      "complete": true
    }}
  ]
}}
````

注意：
1. 如果文本片段不包含明显的语义单元，返回type为"general"
2. start_marker和end_marker应该是文本中实际存在的标记
3. 严格按JSON格式输出"""
    
    @staticmethod
    def chunk_metadata_prompt(chunk_text: str) -> str:
        """Chunk元数据提取提示词（改进版：强调返回格式）"""
        return f"""从以下医疗指南片段中提取结构化信息：

**文本：**
{chunk_text[:1000]}

**提取字段（JSON格式）：**
```json
{{
  "drug_name": "药物名称（如果是药物条目，否则为null）",
  "indications": ["适应症列表"],
  "contraindications": ["禁忌症列表"],
  "dosage": "剂量方案描述",
  "monitoring": ["监测指标列表"],
  "evidence_level": "证据等级（A级/B级/C级推荐，如果有）",
  "key_numbers": {{"关键数值名称": "数值+单位"}},
  "related_diseases": ["相关疾病列表"],
  "section_title": "章节标题"
}}
```

**CRITICAL - 返回格式要求：**
1. ✅ **必须返回单个JSON对象**，如：{{"drug_name": "二甲双胍", ...}}
2. ❌ **不要返回数组**，如：[{{"drug_name": ...}}] ← 这是错误的！
3. ❌ **不要在JSON外添加任何文字**，包括"这是提取结果："等说明
4. 如果某些字段不存在，设为null或空列表[]
5. key_numbers应提取所有关键数值，如"eGFR <30"、"HbA1c ≥7.0%"

**正确示例：**
{{"drug_name": "二甲双胍", "dosage": "500mg bid", "indications": ["2型糖尿病"]}}

**错误示例：**
[{{"drug_name": "二甲双胍"}}]  ← 这是数组，不要这样！"""
    
    @staticmethod
    def patient_parsing_prompt(patient_case: str) -> str:
        """病例解析提示词"""
        return f"""从以下患者描述中提取结构化信息：

**患者病例：**
{patient_case}

**请提取以下字段（JSON格式）：**
````json
{{
  "age": 年龄（数字）,
  "gender": "性别",
  "primary_diagnosis": "主要诊断",
  "complications": ["并发症1", "并发症2"],
  "lab_values": {{"HbA1c": "8.5%", "eGFR": "55 ml/min"}},
  "contraindication_conditions": ["肾功能不全", "心力衰竭"],
  "embedding_text": "优化后的检索query（整合关键信息）"
}}
````

注意：
1. embedding_text应该是适合向量检索的自然语句
2. 例如："55岁2型糖尿病患者，HbA1c 8.5%，eGFR 55，需要降糖药物推荐"
3. 严格JSON格式"""
    
    @staticmethod
    def rerank_prompt(patient_info: Dict, candidates_text: str, top_k: int) -> str:
        """检索重排序提示词"""
        return f"""你是内分泌科主治医师。现有一个患者病例，请从以下候选指南片段中选出最相关的{top_k}个，并按临床相关性排序。

**患者信息：**
- 年龄：{patient_info.get('age')}岁
- 诊断：{patient_info.get('primary_diagnosis')}
- 实验室：{patient_info.get('lab_values', {})}
- 禁忌条件：{', '.join(patient_info.get('contraindication_conditions', []))}

**候选指南片段：**
{candidates_text}

**请输出JSON：**
````json
{{
  "top_k_indices": [3, 1, 7, 2, 9],
  "reasoning": "为什么这样排序的临床理由"
}}
````

排序原则：
1. 禁忌症优先（有禁忌的直接排除）
2. 肾功能适配性
3. 证据等级（A级>B级>C级）
4. 指南权威性"""
    
    @staticmethod
    def conflict_resolution_prompt(chunk_a_info: Dict, chunk_b_info: Dict) -> str:
        """冲突解决提示词"""
        return f"""你是临床药学专家。以下两份指南对同一药物有不同推荐，请分析原因并给出临床建议。

**指南A：{chunk_a_info['guideline_name']}**
{chunk_a_info['content'][:500]}
推荐剂量：{chunk_a_info['dosage']}

**指南B：{chunk_b_info['guideline_name']}**
{chunk_b_info['content'][:500]}
推荐剂量：{chunk_b_info['dosage']}

**请回答以下问题（JSON格式）：**
````json
{{
  "conflict_reason": "为什么两份指南不一致？",
  "recommendation": "临床应该遵循哪个？",
  "context_dependency": "是否与患者特征有关（如地域、病程）？",
  "priority_guideline": "应优先采纳的指南名称",
  "explanation": "推荐理由"
}}
```"""

# ==========================================
# RAG推理引擎
# ==========================================

class RAGReasoningEngine:
    """
    基于LLM的RAG推理引擎
    """
    
    def __init__(self, llm_client: DeepseekClient):
        self.llm = llm_client
        self.prompts = RAGPrompts()
    
    def extract_guideline_metadata(self, filename: str, content: str) -> Dict:
        """
        提取指南元数据
        
        Returns:
            元数据字典
        """
        prompt = self.prompts.metadata_extraction_prompt(filename, content)
        
        messages = [
            {"role": "system", "content": "你是医疗文献专家，擅长识别医疗指南的元信息。"},
            {"role": "user", "content": prompt}
        ]
        
        response = self.llm.chat_for_task("metadata", messages)
        result_text = self.llm.get_response_text(response)
        
        # 清理并解析JSON
        try:
            cleaned = result_text.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            print(f"⚠️  JSON解析失败: {str(e)}")
            return {"extraction_error": result_text}
    
    def identify_semantic_boundaries(self, text_segment: str, guideline_name: str) -> List[Dict]:
        """
        识别语义边界
        
        Returns:
            语义单元列表
        """
        prompt = self.prompts.semantic_boundary_prompt(text_segment, guideline_name)
        
        messages = [
            {"role": "system", "content": "你是医疗指南处理专家，擅长识别语义完整的内容单元。"},
            {"role": "user", "content": prompt}
        ]
        
        response = self.llm.chat_for_task("chunking", messages)
        result_text = self.llm.get_response_text(response)
        
        try:
            # 清理JSON
            cleaned = result_text.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            
            result = json.loads(cleaned)
            return result.get('semantic_units', [])
        except json.JSONDecodeError:
            # 降级方案
            return [{"type": "general", "start_marker": text_segment[:30], "end_marker": None, "complete": True}]
    
    def extract_chunk_metadata(self, chunk_text: str) -> Dict:
        """
        从chunk提取元数据（改进版：处理数组返回）
        
        Returns:
            元数据字典
        """
        prompt = self.prompts.chunk_metadata_prompt(chunk_text)
        
        messages = [
            {"role": "system", "content": "你是医疗信息提取专家。"},
            {"role": "user", "content": prompt}
        ]
        
        response = self.llm.chat_for_task("extraction", messages)
        result_text = self.llm.get_response_text(response)
        
        try:
            cleaned = result_text.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            
            result = json.loads(cleaned)
            
            # ✅ 新增：处理LLM返回数组的情况
            if isinstance(result, list):
                if len(result) == 0:
                    return {}
                
                # 如果是list of dict，合并所有dict
                if all(isinstance(item, dict) for item in result):
                    merged = {}
                    for item in result:
                        merged.update(item)
                    return merged
                
                # 如果只有一个元素且是dict，直接返回
                if len(result) == 1 and isinstance(result[0], dict):
                    return result[0]
                
                # 其他情况，返回空dict并记录警告
                print(f"⚠️  LLM返回了非预期的数组格式: {type(result[0])}")
                return {}
            
            # 正常情况：返回dict
            return result
            
        except json.JSONDecodeError:
            return {"extraction_error": result_text}
    
    def parse_patient_case(self, patient_case: str) -> Dict:
        """
        解析患者病例
        
        Returns:
            结构化患者信息
        """
        prompt = self.prompts.patient_parsing_prompt(patient_case)
        
        messages = [
            {"role": "system", "content": "你是医疗信息提取专家。"},
            {"role": "user", "content": prompt}
        ]
        
        response = self.llm.chat_for_task("parser", messages)
        result_text = self.llm.get_response_text(response)
        
        try:
            cleaned = result_text.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {"parse_error": result_text}
    
    def rerank_candidates(self, patient_info: Dict, candidates: List[Dict], top_k: int = 10) -> List[Dict]:
        """
        LLM重排序候选chunks
        
        Returns:
            重排序后的chunks
        """
        # 构建候选文本
        candidates_text = ""
        for i, chunk in enumerate(candidates[:20], 1):
            guideline_name = chunk['metadata']['guideline_meta'].get('full_name', '未知指南')
            candidates_text += f"""
[候选{i}]
指南：{guideline_name}
内容：{chunk['content'][:200]}
---
"""
        
        prompt = self.prompts.rerank_prompt(patient_info, candidates_text, top_k)
        
        messages = [
            {"role": "system", "content": "你是内分泌科主治医师。"},
            {"role": "user", "content": prompt}
        ]
        
        response = self.llm.chat_for_task("rerank", messages)
        result_text = self.llm.get_response_text(response)
        
        try:
            cleaned = result_text.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            
            result = json.loads(cleaned)
            indices = result.get('top_k_indices', list(range(1, top_k+1)))
            
            return [candidates[i-1] for i in indices if 0 < i <= len(candidates)]
        except (json.JSONDecodeError, IndexError):
            # 降级：返回原始顺序
            return candidates[:top_k]
    
    def resolve_conflict(self, chunk_a: Dict, chunk_b: Dict) -> Dict:
        """
        解决指南冲突
        
        Returns:
            冲突解决结果
        """
        chunk_a_info = {
            'guideline_name': chunk_a['metadata']['guideline_meta'].get('full_name'),
            'content': chunk_a['content'],
            'dosage': chunk_a['metadata']['extracted_fields'].get('dosage', '未知')
        }
        
        chunk_b_info = {
            'guideline_name': chunk_b['metadata']['guideline_meta'].get('full_name'),
            'content': chunk_b['content'],
            'dosage': chunk_b['metadata']['extracted_fields'].get('dosage', '未知')
        }
        
        prompt = self.prompts.conflict_resolution_prompt(chunk_a_info, chunk_b_info)
        
        messages = [
            {"role": "system", "content": "你是临床药学专家。"},
            {"role": "user", "content": prompt}
        ]
        
        response = self.llm.chat_for_task("conflict", messages)
        result_text = self.llm.get_response_text(response)
        
        try:
            cleaned = result_text.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return {"resolution_error": result_text}

# ==========================================
# 测试代码
# ==========================================

def test_client():
    """测试LLM客户端"""
    from agents.config import ConfigManager
    
    # 验证配置
    if not ConfigManager.validate_all():
        print("❌ 配置验证失败")
        return
    
    # 打印配置
    ConfigManager.print_config_summary()
    
    print("\n" + "="*70)
    print("测试便捷方法 chat_for_task()")
    print("="*70)
    
    client = DeepseekClient()
    
    # 测试不同任务
    tasks = ["parser", "intent", "diagnosis", "therapy", "chat"]
    
    for task in tasks:
        config = ModelConfig.get_config_for_task(task)
        print(f"\n📋 {task.upper()} 任务配置:")
        print(f"   模型: {config['model']}")
        print(f"   温度: {config['temperature']}")
        print(f"   最大tokens: {config['max_tokens']}")
        print(f"   深度思考: {config['deep_thinking']}")
    
    print("\n" + "="*70)
    print("测试实际调用（诊断任务）")
    print("="*70)
    
    engine = MedicalReasoningEngine(client)
    
    test_profile = {
        'age': 72,
        'gender': 'Female',
        'HbA1c': 11.7,
        'BMI': 20.77
    }
    
    print(f"\n使用模型: {ModelConfig.DIAGNOSIS_MODEL}")
    print(f"温度: {ModelConfig.DIAGNOSIS_TEMPERATURE}")
    print(f"深度思考: {ModelConfig.DIAGNOSIS_DEEP_THINKING}")
    
    result = engine.diagnose_with_llm(test_profile)
    print(f"\n诊断结果（摘要）:\n{result['llm_reasoning'][:300]}...")

if __name__ == "__main__":
    test_client()
