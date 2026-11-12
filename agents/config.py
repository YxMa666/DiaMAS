"""
config.py — Dia-MAS 系统配置中心（精简版）
==========================================
"""

from typing import Dict, Any, Tuple, List, Set, Optional
import logging
import os
import re
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════════
# 一、Baseline 模型实验矩阵
# ═══════════════════════════════════════════════════════════════════════════════

BASELINE_MODELS: Dict[str, Dict] = {
    "deepseek-r1-250528":       {"model_str": "deepseek-r1-250528",       "temperature": 0.0, "max_tokens": 8000, "deep_thinking": True,  "family": "deepseek_r"},
    "deepseek-v3":              {"model_str": "deepseek-v3",              "temperature": 0.0, "max_tokens": 8000, "deep_thinking": True, "family": "deepseek_v"},
    "deepseek-v3.1-terminus":   {"model_str": "deepseek-v3.1-terminus",   "temperature": 0.0, "max_tokens": 8000, "deep_thinking": True, "family": "deepseek_v"},
    "deepseek-v3.2":            {"model_str": "deepseek-v3.2",            "temperature": 0.0, "max_tokens": 8000, "deep_thinking": True, "family": "deepseek_v"},
    "deepseek-v4-flash":        {"model_str": "deepseek-v4-flash",        "temperature": 0.0, "max_tokens": 8000, "deep_thinking": True, "family": "deepseek_v"},
    "deepseek-v4-pro":          {"model_str": "deepseek-v4-pro",        "temperature": 0.0, "max_tokens": 8000, "deep_thinking": True, "family": "deepseek_v"},
    "qwen3-30b-a3b":            {"model_str": "qwen3-30b-a3b",            "temperature": 0.0, "max_tokens": 8000, "deep_thinking": True, "family": "qwen"},
    "qwen3-32b":    {"model_str": "qwen3-32b",            "temperature": 0.0, "max_tokens": 8000, "deep_thinking": True, "family": "qwen"},
    "qwen3-235b-a22b":          {"model_str": "qwen3-235b-a22b",          "temperature": 0.0, "max_tokens": 8000, "deep_thinking": True, "family": "qwen"},
    "qwen3-coder-480b-a35b-instruct":          {"model_str": "qwen3-coder-480b-a35b-instruct",          "temperature": 0.0, "max_tokens": 8000, "deep_thinking": True, "family": "qwen"},
    "Kimi-K2":                  {"model_str": "Kimi-K2",                  "temperature": 0.0, "max_tokens": 8000, "deep_thinking": True, "family": "kimi"},
    "kimi-k2.6":                  {"model_str": "kimi-k2.6",                  "temperature": 0.0, "max_tokens": 8000, "deep_thinking": True, "family": "kimi"},
    "glm-5":                    {"model_str": "glm-5",                    "temperature": 0.0, "max_tokens": 8000, "deep_thinking": True,  "family": "glm"},
    "glm-5.1":                    {"model_str": "glm-5.1",                    "temperature": 0.0, "max_tokens": 8000, "deep_thinking": True,  "family": "glm"},
    "glm-5.2":                    {"model_str": "glm-5.2",                    "temperature": 0.0, "max_tokens": 8000, "deep_thinking": True,  "family": "glm"},
    # "mimo-v2.5-pro":                    {"model_str": "mimo-v2.5-pro",                    "temperature": 0.0, "max_tokens": 8000, "deep_thinking": True,  "family": "xiaomi"},
}


# ═══════════════════════════════════════════════════════════════════════════════
# 二、路径配置
# ═══════════════════════════════════════════════════════════════════════════════

class PathConfig:
    """路径配置 — 全局唯一真相来源。所有脚本从此处读取路径，禁止各自硬编码。"""
    PROJECT_ROOT = Path("")
    DATA_DIR     = PROJECT_ROOT / "data"
 
    # 指南相关
    GUIDELINES_DIR = DATA_DIR / "processed" / "guidelines"
    CHUNKS_DIR     = DATA_DIR / "chunks"
    INDEX_DIR      = DATA_DIR / "indexes"
    METADATA_DIR   = DATA_DIR / "metadata"
 
    # 原始数据
    MIMIC3_PATH = Path("ori_data_zip/mimic-iii-clinical-database-carevue-subset-1.4")
    MIMIC4_PATH = Path("ori_data_zip/mimic-iv-3.1")
    EICU_PATH   = Path("ori_data_zip/eicu-collaborative-research-database-2.0")
 
    # 患者列表 & 结局 CSV
    PATIENCE_OUTPUT_BASE         = Path("./diabetes_subsets")
    MIMIC3_PATIENCE_PATH         = PATIENCE_OUTPUT_BASE / "mimiciii_patient_list.csv"
    MIMIC3_PATIENCE_OUTCOME_PATH = PATIENCE_OUTPUT_BASE / "mimic3_outcomes.csv"
    MIMIC4_PATIENCE_PATH         = PATIENCE_OUTPUT_BASE / "mimiciv_patient_list.csv"
    MIMIC4_PATIENCE_OUTCOME_PATH = PATIENCE_OUTPUT_BASE / "mimic4_outcomes.csv"
    EICU_PATIENCE_PATH           = PATIENCE_OUTPUT_BASE / "eicu_patient_list.csv"
    EICU_PATIENCE_OUTCOME_PATH   = PATIENCE_OUTPUT_BASE / "eicu_outcomes.csv"
 
    # 患者 JSON 缓存（run_extraction.py 输出）
    PATIENT_CACHE_BASE = DATA_DIR / "patient_cache_v1"
    OUTCOMES_DIR       = PATIENCE_OUTPUT_BASE
    COHORT_DIR         = DATA_DIR / "cohort_v4"
    EVAL_DIR           = DATA_DIR / "evaluation_results"
    UMLS_CACHE_BASE    = DATA_DIR / "umls"
 
    @classmethod
    def cache_dir(cls, database: str) -> Path:
        return cls.PATIENT_CACHE_BASE / database.lower()
 
    @classmethod
    def create_dirs(cls):
        for attr_name in dir(cls):
            if attr_name.endswith("_DIR"):
                d = getattr(cls, attr_name)
                if isinstance(d, Path):
                    d.mkdir(parents=True, exist_ok=True)
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# 三、Era 分层配置
# ═══════════════════════════════════════════════════════════════════════════════
# ── GAS 评分全局开关 ──
GAS_EQUAL_WEIGHTS = True   # True=所有规则等权(grade 一律按 1.0)。论文主口径:去除权重自由度。
 
# ── GAS-结局分析的数据集排除 ──
# eICU 是纯 ICU 库,无"出院回家"事件,discharge_medications 仅是"ICU 转出时在用药"的近似,
# 不是真·出院带药 → 无目标变量,排除出 GAS/处方评估(仍可用于队列/结局/严重度等非处方分析)。
GAS_EXCLUDE_DATABASES = {"eicu"}
class EraConfig:
    """
    时代（Era）分层定义 — 全局唯一真相来源。
 
    Era 定义：
      Era-1: ≤2013  pre-CVOT
      Era-2: 2014–2019  CVOT 证据积累
      Era-3: ≥2020  心肾全面保护
    """
    ERA1_END   = 2013
    ERA2_START = 2014
    ERA2_END   = 2019
    ERA3_START = 2020
    ERA_LABELS = ["Era-1", "Era-2", "Era-3"]
 
    # 每个 Era 的代表性年份，供 OSE/GAS 规则条件判断使用（如 yr >= 2016）。
    # 取各 Era 末年，确保：
    #   Era-1(2013): yr<2016 → EMPA-REG/LEADER 规则不触发（正确）
    #   Era-2(2019): yr>=2016 → SGLT2i/GLP-1RA CVD 规则触发（正确）
    #   Era-3(2022): yr>=2020 → CREDENCE/DAPA-CKD/KDIGO 2022 规则全部触发（正确）
    ERA_TO_ADA_YEAR: Dict[str, int] = {
        "Era-1": 2013,   # ERA1_END
        "Era-2": 2019,   # ERA2_END
        "Era-3": 2022,   # KDIGO 2022 / DAPA-CKD era
    }
 
    ERA_METADATA: Dict[str, Dict] = {
        "Era-1": {"label": "Pre-CVOT (≤2013)",                               "year_range": "≤2013",    "databases": ["mimic3", "mimic4"]},
        # "Era-2": {"label": "CVOT Evidence Accumulation (2014–2019)",          "year_range": "2014–2019", "databases": ["mimic4", "eicu"]},
        "Era-2": {"label": "CVOT Evidence Accumulation (2014–2019)", "year_range": "2014–2019", "databases": ["mimic4"]},  # 原 ["mimic4","eicu"]
        "Era-3": {"label": "Comprehensive Cardio-Renal Protection (≥2020)",   "year_range": "≥2020",    "databases": ["mimic4"]},
    }
 
    @classmethod
    def year_to_era(cls, year: int) -> str:
        if year <= cls.ERA1_END:
            return "Era-1"
        elif year <= cls.ERA2_END:
            return "Era-2"
        return "Era-3"
 
    MIMIC3_YEAR = 2008          # MIMIC-III CareVue subset (2001-2008): end of the data window
    MIMIC4_YEARS = (2008, 2022)
    _warned_no_anchor = False

    @staticmethod
    def _db(patient_data: dict) -> str:
        db = str(patient_data.get("database") or patient_data.get("db_source") or "").lower().replace("-", "_")
        if "eicu" in db:
            return "eicu"
        if "mimic3" in db or "mimic_iii" in db or "mimiciii" in db:
            return "mimic3"
        if "mimic4" in db or "mimic_iv" in db or "mimiciv" in db:
            return "mimic4"
        return db

    @staticmethod
    def _plausible(v) -> Optional[int]:
        try:
            y = int(str(v)[:4])
        except (TypeError, ValueError):
            return None
        return y if 1990 <= y <= 2030 else None

    @classmethod
    def assigned_year(cls, patient_data: dict) -> Optional[int]:
        """The single year assigned to an admission. Prompt year, guideline retrieval ceiling, medication list,
        period label, GAS rules and hard constraints all use it.

        MIMIC-III CareVue: 2008. eICU: recorded discharge year. MIMIC-IV: mimic4_admission_year (per admission).
        Other sources: a plausible calendar year of the admission. None if no year can be assigned.
        """
        cm = patient_data.get("clinical_metrics") or {}
        for v in (patient_data.get("admission_year"), cm.get("admission_year")):
            y = cls._plausible(v)
            if y is not None:
                return y
        db = cls._db(patient_data)
        if db == "mimic3":
            return cls.MIMIC3_YEAR
        if db == "eicu":
            return cls._extract_eicu_year(patient_data)
        if db == "mimic4" or patient_data.get("anchor_year_group") or cm.get("anchor_year_group"):
            y = cls.mimic4_admission_year(patient_data)
            if y is not None:
                return y
        for key in ("admission_time", "admittime", "discharge_time"):
            y = cls._plausible(str(patient_data.get(key) or cm.get(key) or "")[:4])
            if y is not None:
                return y
        return None

    @classmethod
    def mimic4_admission_year(cls, patient_data: dict) -> Optional[int]:
        """Real year of a MIMIC-IV admission.

        MIMIC-IV shifts all dates of a patient by one offset and gives, per patient, a shifted anchor_year and
        the real three-year anchor_year_group that contains it. An admission is placed by its distance from
        anchor_year: end year of anchor_year_group + (shifted admission year - anchor_year), clipped to the
        MIMIC-IV span 2008-2022. Example: group 2008-2010, anchor_year 2153, admission in 2159 -> 2016.
        """
        cm = patient_data.get("clinical_metrics") or {}
        get = lambda k: patient_data.get(k) if patient_data.get(k) not in (None, "") else cm.get(k)  # noqa: E731
        years = [int(y) for y in re.findall(r"(?:19|20)\d{2}", str(get("anchor_year_group") or ""))]
        if not years:
            return None
        year = max(years)
        try:
            year += int(str(get("admission_time") or get("admittime"))[:4]) - int(str(get("anchor_year"))[:4])
        except (TypeError, ValueError):
            if not cls._warned_no_anchor:
                cls._warned_no_anchor = True
                logging.getLogger(__name__).warning(
                    "MIMIC-IV record without anchor_year or admission_time: using the end year of "
                    "anchor_year_group, which is exact only for admissions in the anchor year. "
                    "Re-extract with data_prep/run_extraction.py to store anchor_year.")
        lo, hi = cls.MIMIC4_YEARS
        return min(max(year, lo), hi)

    @classmethod
    def classify_era(cls, patient_data: dict) -> tuple:
        """返回 (era, db_source)。Era 由 assigned_year 决定（全局唯一赋年）。"""
        db = cls._db(patient_data)
        src = db if db in ("mimic3", "mimic4", "eicu") else "unknown"
        y = cls.assigned_year(patient_data)
        if y is not None:
            return cls.year_to_era(y), src
        if src == "eicu":
            return "Era-2", "eicu"
        if src == "mimic4":
            logging.getLogger(__name__).warning(
                f"[classify_era] MIMIC-IV patient missing anchor_year_group "
                f"(subject_id={patient_data.get('subject_id', '?')}); returning Unknown.")
        return "Unknown", src

    @staticmethod
    def _extract_eicu_year(patient_data: dict):
        for field in ("hospitaldischargeyear", "hospital_discharge_year", "discharge_year"):
            val = patient_data.get(field)
            if val is not None:
                try:
                    y = int(float(str(val)))
                    if 2000 <= y <= 2030:
                        return y
                except (ValueError, TypeError):
                    pass
        adm = str(patient_data.get("admission_time", "") or "")
        if adm and len(adm) >= 4 and adm[:4].isdigit():
            try:
                y = int(adm[:4])
                if 2000 <= y <= 2030:
                    return y
            except ValueError:
                pass
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# 四、API 配置
# ═══════════════════════════════════════════════════════════════════════════════

class APIConfig:
    """API 连接配置。"""
    API_KEY    = os.environ.get("DIAMAS_API_KEY", "")
    BASE_URL   = os.environ.get("DIAMAS_BASE_URL", "http://localhost:8000/v1/chat/completions")
    SSL_VERIFY = os.environ.get("DIAMAS_SSL_VERIFY", "1") != "0"
    TIMEOUT    = 240    # 秒
    MAX_RETRIES = 5
    RETRY_DELAY = 2     # 秒
    NUMBER_WORKERS = 4
    UMLS_URL   = "https://uts-ws.nlm.nih.gov/rest"
    UMLS_KEY   = os.environ.get("UMLS_KEY", "")
    NCBI_API_KEY = os.environ.get("NCBI_API_KEY", "")
    NCBI_EMAIL   = os.environ.get("NCBI_EMAIL",   "")
    NCBI_TOOL = "DiaMAS"


# ═══════════════════════════════════════════════════════════════════════════════
# 五、模型配置（仅保留实际被主流程引用的任务）
# ═══════════════════════════════════════════════════════════════════════════════
 
class ModelConfig:
    """
    任务模型配置。
 
    实际引用位置：
      THERAPY_*       → llm_inference_stage.py（baseline 推荐默认模型）
      DIAGNOSIS_*     → llm_client.py（兼容旧 diagnose() 接口）
      CHAT_*          → llm_client.py（兼容旧 chat_with_expert() 接口）
      QUICK_QUERY_*   → drug_normalizer.py（LLM 辅助药物名映射）
      EMBEDDING_MODEL → ada_guideline_processor.py / rebuild_ada_index.py
      PARSER_*        → 历史兼容，保留供 llm_client.py 引用
    """
    EMBEDDING_MODEL = "BAAI/bge-m3"
 
    # ── Baseline 推荐（llm_inference_stage 默认值） ──────────────────────────
    THERAPY_MODEL         = "deepseek-v3.1-terminus"
    THERAPY_TEMPERATURE   = 0.0
    THERAPY_MAX_TOKENS    = 8000
    THERAPY_DEEP_THINKING = False
 
    # ── 诊断推理（llm_client 兼容接口） ─────────────────────────────────────
    DIAGNOSIS_MODEL         = "deepseek-r1-250528"
    DIAGNOSIS_TEMPERATURE   = 0.3
    DIAGNOSIS_MAX_TOKENS    = 4000
    DIAGNOSIS_DEEP_THINKING = True
 
    # ── 医学对话（llm_client 兼容接口） ─────────────────────────────────────
    CHAT_MODEL         = "qwen3-30b-a3b"
    CHAT_TEMPERATURE   = 0.7
    CHAT_MAX_TOKENS    = 4000
    CHAT_DEEP_THINKING = True
 
    # ── 快速查询（DrugNormalizer LLM 辅助映射） ─────────────────────────────
    QUICK_QUERY_MODEL         = "qwen3-30b-a3b"
    QUICK_QUERY_TEMPERATURE   = 0.5
    QUICK_QUERY_MAX_TOKENS    = 1000
    QUICK_QUERY_DEEP_THINKING = False
 
    # ── 病历解析（llm_client 兼容接口） ─────────────────────────────────────
    PARSER_MODEL         = "deepseek-v3.1-terminus"
    PARSER_TEMPERATURE   = 0.1
    PARSER_MAX_TOKENS    = 4000
    PARSER_DEEP_THINKING = False
    
    JUDGE_MODEL_DEFAULT = "deepseek-r1-250528"     # 推理侧用 deepseek-v3.2
 
    @classmethod
    def get_config_for_task(cls, task: str) -> Dict[str, Any]:
        """根据任务类型返回配置字典。"""
        configs = {
            "therapy":     {"model": cls.THERAPY_MODEL,     "temperature": cls.THERAPY_TEMPERATURE,     "max_tokens": cls.THERAPY_MAX_TOKENS,     "deep_thinking": cls.THERAPY_DEEP_THINKING},
            "diagnosis":   {"model": cls.DIAGNOSIS_MODEL,   "temperature": cls.DIAGNOSIS_TEMPERATURE,   "max_tokens": cls.DIAGNOSIS_MAX_TOKENS,   "deep_thinking": cls.DIAGNOSIS_DEEP_THINKING},
            "chat":        {"model": cls.CHAT_MODEL,         "temperature": cls.CHAT_TEMPERATURE,         "max_tokens": cls.CHAT_MAX_TOKENS,         "deep_thinking": cls.CHAT_DEEP_THINKING},
            "quick_query": {"model": cls.QUICK_QUERY_MODEL,  "temperature": cls.QUICK_QUERY_TEMPERATURE,  "max_tokens": cls.QUICK_QUERY_MAX_TOKENS,  "deep_thinking": cls.QUICK_QUERY_DEEP_THINKING},
            "parser":      {"model": cls.PARSER_MODEL,       "temperature": cls.PARSER_TEMPERATURE,       "max_tokens": cls.PARSER_MAX_TOKENS,       "deep_thinking": cls.PARSER_DEEP_THINKING},
        }
        if task not in configs:
            raise ValueError(f"未知任务类型: {task}。可用: {list(configs.keys())}")
        return configs[task]
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# 六、药物分类列表
# ═══════════════════════════════════════════════════════════════════════════════
 
DIABETES_MEDICATIONS: Dict[str, List[str]] = {
    "Biguanides":                ["Metformin"],
    "SGLT2i":                    ["Empagliflozin", "Dapagliflozin", "Canagliflozin", "Ertugliflozin", "Sotagliflozin", "Ipragliflozin", "Tofogliflozin", "Bexagliflozin", "Henagliflozin", "Rongliflozin", "Janagliflozin", "Remogliflozin"],
    "GLP-1RA":                   ["Exenatide", "Lixisenatide", "Liraglutide", "Semaglutide", "Dulaglutide", "Albiglutide", "Beinaglutide", "Loxenatide", "Polyethylene glycol loxenatide"],
    "DPP-4i":                    ["Sitagliptin", "Vildagliptin", "Saxagliptin", "Linagliptin", "Alogliptin", "Teneligliptin", "Gemigliptin", "Trelagliptin", "Omarigliptin", "Anagliptin", "Evogliptin", "Gosogliptin", "Retagliptin"],
    "GIP_GLP1_RA":               ["Tirzepatide"],
    "GCG_GLP1_RA":               ["Mazdutide"],
    "GLP1_Multimer_RA":          ["Efpeglenatide"],
    "Sulfonylureas":             ["Glipizide", "Glyburide", "Glimepiride", "Gliclazide", "Chlorpropamide", "Tolbutamide", "Tolazamide", "Acetohexamide"],
    "Meglitinides":              ["Repaglinide", "Nateglinide", "Mitiglinide"],
    "Thiazolidinediones":        ["Pioglitazone", "Rosiglitazone"],
    "Insulins_Basal":            ["NPH insulin", "Insulin human", "Insulin glargine", "Insulin detemir", "Insulin degludec", "Insulin icodec"],
    "Insulins_Rapid":            ["Regular insulin", "Insulin lispro", "Insulin aspart", "Insulin glulisine", "Faster insulin aspart"],
    # 未指明剂型的泛称胰岛素（MIMIC 大量出院药仅记作 "Insulin"）。
    # 计入降糖药口径（n_dm_drugs / DR-07 / Path-DM），但刻意不进 basal/rapid 成员集，
    # 避免凭未知剂型误满足/误违反剂型专属规则（PR-01 基础胰岛素 / PR-02 餐时胰岛素 等）。
    "Insulins_Unspecified":      ["Insulin (unspecified)"],
    "Alpha_Glucosidase_Inhibitors": ["Acarbose", "Miglitol", "Voglibose"],
    "Glucokinase_Activators":    ["Dorzagliatin"],
    "Amylin_Analogues":           ["Pramlintide"],
    "Bile_Acid_Sequestrants_DM":  ["Colesevelam"],
    "Dopamine_Agonists_DM":       ["Bromocriptine"],
}
 
# 并发症用药分类表（模块级常量，与 DIABETES_MEDICATIONS 并列）
# 设计原则：
#   - 仅覆盖与糖尿病并发症直接相关的药物类别
#   - Spironolactone 归入 Hypertension_Diuretics（保钾利尿），不重复计入 CKD_MRA
#   - Antiplatelet 仅含抗血小板药（Aspirin/P2Y12），已移除 Heparin/Enoxaparin（无 DM 特异性适应症）
#   - Anticoagulant 整类已移除（Warfarin/NOAC 无 DM 特异性适应症，LLM 审计 NOT_RELEVANT）
#   - 此常量必须与 EVAL_COMPLICATION_ONLY_CATEGORIES 保持同步
COMPLICATION_DRUGS: Dict[str, List[str]] = {
    # ── 降压药 ─────────────────────────────────────────────────────────────────
    # ACEi / ARB：DM 肾病、高血压一线（ADA 强烈推荐，Grade A）
    "Hypertension_ACEi":             ["Lisinopril", "Enalapril", "Ramipril", "Benazepril", "Perindopril", "Fosinopril", "Captopril", "Quinapril"],
    "Hypertension_ARB":              ["Losartan", "Valsartan", "Telmisartan", "Irbesartan", "Candesartan", "Olmesartan", "Azilsartan", "Eprosartan"],
 
    # CCB：仅保留 DM 高血压有明确指南地位的成员
    #   保留：Amlodipine（ADA 首选 CCB）、Levamlodipine（中国市场 S-Amlodipine）、
    #          Nitrendipine（SYST-EUR 证据）、Diltiazem（CKD 心率控制 BORDERLINE，保留）
    #   移除：Nifedipine、Felodipine（非 DM 一线）、Verapamil（负性肌力，避免 DM 心衰）、
    #          Nimodipine（脑血管痉挛用，非 DM 并发症）
    # v3 移除 Diltiazem：两个数据集 LLM 一致 NOT_RELEVANT（n=317/1234）
    # DM+AF 心率控制有争议，但 ADA 无明确 DM 特异性推荐
    "Hypertension_CCB":              [],  # ⛔ 与 DrugNormalizer 一致:CCB 非糖尿病特异(同 BB/噻嗪),NOT_RELEVANT 排除
 
    # 利尿剂：保留有 HF/CKD 作用的袢利尿剂和醛固酮拮抗剂
    #   保留：袢利尿剂（Furosemide/Torsemide/Bumetanide）+ Spironolactone（HF/CKD-MRA 前体）
    #         + Indapamide（ESC 指南 DM 高血压 B 类）
    #   移除：HCTZ（ADA 不推荐 DM 一线；加重胰岛素抵抗）、Chlorthalidone（同理）
    # v2 移除 Indapamide：BORDERLINE 两个数据集，无 DM 并发症特异性适应症
    "Hypertension_Diuretics":        ["Bumetanide", "Furosemide", "Torsemide", "Spironolactone"],
 
    # β-blocker：只保留有 DM 并发症（HF/CVD）指南支持的成员
    #   保留：Metoprolol/Bisoprolol/Carvedilol（HF 三大 β-blocker，ADA 推荐 DM+HF）、
    #          Nebivolol（HF 证据，vasodilatory，对糖代谢影响小）
    #   移除：Labetalol（仅高血压急症 IV 用，无 DM 并发症适应症）、
    #          Propranolol（掩盖低血糖症状，ADA 不推荐 DM）、
    #          Atenolol（不优于其他 β-blocker，ASCOT 证据不支持 DM 优先使用）
    # v4 移除 Nebivolol：两数据集 NOT_RELEVANT（test n=13），无 DM HF 特异性指南推荐
    "Hypertension_BetaBlockers":     ["Metoprolol", "Bisoprolol", "Carvedilol"],
 
    # ── ARNI（HFrEF 一线，非 ARB 同类） ───────────────────────────────────────
    # Sacubitril/Valsartan 是 ARNI（脑啡肽酶抑制剂 + ARB 复方），主要适应症 HFrEF
    # ESC HF 2021 / ACC-AHA 2022 强烈推荐用于 HFrEF；不与 ARB 同类（机制不同）
    "Hypertension_ARNI":             ["Sacubitril/valsartan"],
 
    # ── 调脂药 ─────────────────────────────────────────────────────────────────
    "Lipid_Statins":                 ["Atorvastatin", "Rosuvastatin", "Simvastatin", "Pravastatin", "Lovastatin", "Fluvastatin", "Pitavastatin"],
    # NonStatins：移除 Gemfibrozil（ADA 不推荐；与他汀联用风险；已被 Fenofibrate 取代）
    # 注意：PCSK9 通路药物（Evolocumab/Alirocumab/Inclisiran）保留在此以维持现有规则
    # 向后兼容；下方 Lipid_PCSK9_Pathway 作为更 specific 的独立类别供新规则使用
    "Lipid_NonStatins":              ["Ezetimibe", "Fenofibrate", "Evolocumab", "Alirocumab", "Inclisiran", "Icosapent ethyl", "Bempedoic acid"],
    # PCSK9 通路（mAb + siRNA）独立类别：针对 PCSK9-specific 推荐的规则使用
    # 与 Lipid_NonStatins 有意保留交叉；现有引用 Lipid_NonStatins 的规则不受影响
    "Lipid_PCSK9_Pathway":           ["Evolocumab", "Alirocumab", "Inclisiran"],
 
    # ── 抗血小板 ───────────────────────────────────────────────────────────────
    # 保留：ADA 明确推荐 DM+ASCVD 使用抗血小板药
    # 移除：Heparin、Enoxaparin（抗凝剂，非抗血小板；无 DM 特异性适应症；LLM 审计 NOT_RELEVANT）
    "Antiplatelet":                  ["Aspirin", "Clopidogrel", "Ticagrelor", "Prasugrel", "Cilostazol"],
 
    # ── 移除整类：Anticoagulant ────────────────────────────────────────────────
    # Warfarin/Apixaban/Rivaroxaban/Dabigatran/Edoxaban 均无 DM 特异性指南适应症
    # LLM 审计两个数据集一致判为 NOT_RELEVANT；GAS Path-C 中若 COMPLICATION_DRUG_GRADES
    # 有相关规则，其 condition lambda 应已通过 has_afib/has_dvt 等门控，不受此影响
 
    # ── CKD 专项 ───────────────────────────────────────────────────────────────
    "CKD_MRA":                       ["Finerenone", "Eplerenone"],
 
    # ── 神经病变 ───────────────────────────────────────────────────────────────
    "Neuropathy_Gabapentinoids":     ["Pregabalin", "Gabapentin", "Mirogabalin"],
    # SNRI：移除 Venlafaxine（ADA 仅推荐 Duloxetine 为神经病变一线；Venlafaxine 为 off-label）
    # v4 移除 Desvenlafaxine：ADA 神经病变指南只推荐 Duloxetine，Desvenlafaxine 无独立推荐
    "Neuropathy_SNRI":               ["Duloxetine"],
    "Neuropathy_TCA":                ["Amitriptyline", "Nortriptyline", "Imipramine"],
    # Neuropathy_Others：移除 Alpha-lipoic acid（不在 ADA/AAN 一线推荐，仅德国/中国指南）
    #   保留：Mecobalamin（B12，中国/亚洲指南广泛推荐）、Epalrestat（AACE/亚洲指南）、
    #          Mexiletine（BORDERLINE，AAN 指南 B 类）、Capsaicin（ADA 认可局部用）
    # v2 移除 Mexiletine：NOT_RELEVANT（seed 数据集），AAN B 类证据，非 ADA 推荐一线
    "Neuropathy_Others":             ["Mecobalamin", "Epalrestat", "Capsaicin"],
 
    # ── 微循环 / 视网膜 / 胃轻瘫 / 低血糖急救 ─────────────────────────────────
    # v3 移除 Pentoxifylline：NOT_RELEVANT(seed)/BORDERLINE(test)，非 ADA/ESC 一线推荐
    "Microcirculation_Vasodilators": ["Alprostadil"],
    "Retinopathy_AntiVEGF":          ["Ranibizumab", "Aflibercept", "Bevacizumab", "Brolucizumab", "Faricimab", "Conbercept"],
    "Retinopathy_Oral":              ["Calcium dobesilate"],
    "Gastroparesis":                 ["Metoclopramide", "Domperidone"],
    "Hypoglycemia_Rescue":           ["Glucagon", "Dasiglucagon", "Oral glucose"],
}
 
 
def create_diabetes_drug_list() -> List[Tuple[str, str]]:
    """
    创建糖尿病相关药物列表（降糖药 + 并发症药），返回 [(drug_name, category), ...]。
 
    注意：本函数直接引用模块级 DIABETES_MEDICATIONS 和 COMPLICATION_DRUGS，
    请勿在此处重新定义这两个 dict（会导致 config.py 的单一真相原则失效）。
    """
    result: List[Tuple[str, str]] = []
    for cat, drugs in DIABETES_MEDICATIONS.items():
        for d in drugs:
            result.append((d, cat))
    for cat, drugs in COMPLICATION_DRUGS.items():
        for d in drugs:
            result.append((d, cat))
    return result
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# 七、药物时代可用性配置
# ═══════════════════════════════════════════════════════════════════════════════
 
class DrugEraConfig:
    """
    药物时代可用性配置（C2 时序保真核心）。
 
    get_ada_year_ceiling(admission_year) 是修复"Era-level 截断 bug"的核心函数：
      旧行为：Era-2 患者统一用 ada_year=2016（误差最大2年）
      新行为：2014年患者 → ada_year=2014，2019年患者 → ada_year=2019
    """
    # RAG 检索专用：Era-X → 该 Era 对应的 ADA 指南文档版本年份。
    # 语义：告诉检索器"去查哪一年的 ADA 文档"，与 EraConfig.ERA_TO_ADA_YEAR 不同。
    # EraConfig.ERA_TO_ADA_YEAR 服务于 GAS 规则条件判断（yr >= 2016 等），
    # 本字段（ERA_TO_RAG_YEAR）服务于 ERA-aware RAG 的文档版本选择。
    ERA_TO_RAG_YEAR: Dict[str, int] = {
        "Era-1":   2008,   # MIMIC-III CareVue subset 数据窗口上界（2001–2008）
        "Era-2":   2016,   # Era-2 中段代表值，eICU 年份不可知时的兜底
        "Era-3":   2020,   # Era-3 起始年，anchor_year_group 缺失时的兜底
        "Unknown": 2016,   # 降级默认（取 Era-2 中段，比 Era-3 更保守）
    }
 
    # ADA 指南库覆盖 2006–2026 全年份，get_ada_year_ceiling() 精确匹配就诊年份。
    ADA_PUBLICATION_YEARS: List[int] = list(range(2006, 2027))
 
    # FDA 批准年份（drug-level 时序检查，精确到季度）
    FDA_APPROVAL_YEAR: Dict[str, float] = {
        "metformin": 1994.0, "glipizide": 1984.0, "glyburide": 1984.0,
        "glimepiride": 1995.0, "pioglitazone": 1999.0, "rosiglitazone": 1999.0,
        "sitagliptin": 2006.75, "saxagliptin": 2009.5, "linagliptin": 2011.25,
        "alogliptin": 2013.0, "vildagliptin": 2008.0,
        "exenatide": 2005.25, "liraglutide": 2010.0, "dulaglutide": 2014.75,
        "albiglutide": 2014.25, "semaglutide_sc": 2017.75, "semaglutide_po": 2019.75,
        "tirzepatide": 2022.4, "canagliflozin": 2013.25, "dapagliflozin": 2014.0,
        "empagliflozin": 2014.67, "ertugliflozin": 2017.75,
        "insulin": 1921.0, "glargine": 2000.5, "detemir": 2005.5,
         "degludec": 2015.75, "lispro": 1996.5, "aspart": 2000.75, "glulisine": 2004.25,
        # ── Additional entries for temporal-filter ablation coverage ─────────────
        "semaglutide": 2017.75,       # SC form (Ozempic) FDA Dec 2017
        "lixisenatide": 2016.5,
        "finerenone": 2021.5,          # FDA Jul 2021 — key Era-3 CKD/HF drug
        "eplerenone": 2003.75,
        "evolocumab": 2015.67,
        "alirocumab": 2015.58,
        "inclisiran": 2021.92,
        "bempedoic_acid": 2020.17,
        "icosapent_ethyl": 2012.67,
        "ticagrelor": 2011.75,
        "prasugrel": 2009.67,
        "dasiglucagon": 2021.25,
        "sotagliflozin": 2023.42,
        "bexagliflozin": 2023.08,
        "faster_insulin_aspart": 2017.58,
        "insulin_icodec": 2024.42,
 
        # ════════════════════════════════════════════════════════════════════
        # Formulary 全覆盖补录（temporal-filter ablation）
        # ────────────────────────────────────────────────────────────────────
        # 背景：drug_available_at() 对不在本表的药物保守放行（approval=None→True），
        #   导致大量未登记的较新药（多为中/日/韩/欧药）泄漏到 Era-1 prompt，
        #   削弱 --temporal_filter 的 Direction-A 防御。此处补齐 DIABETES_MEDICATIONS
        #   + COMPLICATION_DRUGS 中此前缺失的全部 96 个条目。
        # 语义：键值为"首个主要市场获批/上市年"。本表名义为 FDA_APPROVAL_YEAR，
        #   但含中/日/韩/欧专属药（无 FDA 批准），对这些药取其首个国家级监管批准年，
        #   以满足 era-appropriate 过滤的时序下界。键名遵循 drug_available_at 的
        #   规范化：name.lower().replace(" ","_")（括号/斜杠原样保留）。
        # ⚠ 标 [APPROX] 的为近似年（多为中国新药，公开获批时间不一致），
        #   进 Nature Medicine supplementary 前需逐项核实。
        # ════════════════════════════════════════════════════════════════════
 
        # ── SGLT2i 尾部（Direction-A 主要泄漏源；务必为 post-2012 年）──────────
        "ipragliflozin": 2014.0,        # 日本 2014-01
        "tofogliflozin": 2014.25,       # 日本 2014-03
        "remogliflozin": 2019.0,        # 印度 2019
        "henagliflozin": 2021.92,       # 中国 2021-12 (恒格列净)
        "rongliflozin": 2023.0,         # [APPROX] 中国 ~2023 (荣格列净)
        "janagliflozin": 2023.0,        # [APPROX] 中国 ~2023 (迦格列净)
 
        # ── GLP-1RA 尾部 ─────────────────────────────────────────────────────
        "beinaglutide": 2016.0,         # 中国 2016 (贝那鲁肽)
        "loxenatide": 2019.0,           # 中国 2019 (洛塞那肽；stem 同时覆盖 "Polyethylene glycol loxenatide")
 
        # ── DPP-4i 尾部 ──────────────────────────────────────────────────────
        "teneligliptin": 2012.67,       # 日本 2012-09
        "gemigliptin": 2012.5,          # 韩国 2012-06
        "anagliptin": 2012.75,          # 日本 2012-09
        "trelagliptin": 2015.25,        # 日本 2015-03 (周制剂)
        "omarigliptin": 2015.75,        # 日本 2015-09 (周制剂)
        "evogliptin": 2015.0,           # 韩国 2015
        "gosogliptin": 2016.0,          # 俄罗斯 2016
        "retagliptin": 2021.0,          # [APPROX] 中国 ~2021 (瑞格列汀)
 
        # ── 其他降糖新机制 ───────────────────────────────────────────────────
        "mazdutide": 2025.0,            # [APPROX] 中国 2025 (玛仕度肽, GCG/GLP-1)
        "efpeglenatide": 2025.0,        # [APPROX] 未获 FDA 批准；保守置于历史窗口之外
        "dorzagliatin": 2022.75,        # 中国 2022-10 (多格列艾汀, GKA)
 
        # ── 老降糖药（年份只需 ≤2001，对过滤无影响，取保守早值）──────────────
        "gliclazide": 1972.0,
        "chlorpropamide": 1958.0,
        "tolbutamide": 1956.0,
        "tolazamide": 1966.0,
        "acetohexamide": 1963.0,
        "repaglinide": 1997.92,         # FDA 1997-12
        "nateglinide": 2000.92,         # FDA 2000-12
        "mitiglinide": 2004.0,          # 日本 2004
        "acarbose": 1995.0,             # FDA 1995
        "miglitol": 1996.92,            # FDA 1996-12
        "voglibose": 1994.0,            # 日本 1994
        "pramlintide": 2005.25,         # FDA 2005-03
        "colesevelam": 2000.0,          # FDA 2000 (DM 适应症 2008，分子 2000 即可得)
        "bromocriptine": 1978.0,        # 分子 1978 (DM 专用 Cycloset FDA 2009)
        "insulin_human": 1982.0,        # 重组人胰岛素 1982
        "insulin_(unspecified)": 1921.0,
 
        # ── 并发症：降压（ACEi/ARB/CCB/利尿/β-阻断/ARNI）──────────────────────
        "lisinopril": 1987.0, "enalapril": 1985.0, "ramipril": 1991.0,
        "benazepril": 1991.0, "perindopril": 1988.0, "fosinopril": 1991.0,
        "captopril": 1981.0, "quinapril": 1991.0,
        "losartan": 1995.0, "valsartan": 1996.0, "telmisartan": 1998.0,
        "irbesartan": 1997.0, "candesartan": 1998.0, "olmesartan": 2002.25,
        "azilsartan": 2011.17,          # FDA 2011-02（>2001，影响 Era-1 排除）
        "eprosartan": 1997.0,
        "levamlodipine": 2003.0,        # 中国 左旋氨氯地平 ~2003
        "nitrendipine": 1985.0,
        "bumetanide": 1983.0, "furosemide": 1966.0, "torsemide": 1993.0,
        "spironolactone": 1960.0,
        "metoprolol": 1978.0, "bisoprolol": 1992.0, "carvedilol": 1995.0,
        "sacubitril/valsartan": 2015.5, # FDA 2015-07 (ARNI；关键 post-Era-1 药)
 
        # ── 并发症：调脂 ─────────────────────────────────────────────────────
        "atorvastatin": 1996.92, "rosuvastatin": 2003.58, "simvastatin": 1991.92,
        "pravastatin": 1991.83, "lovastatin": 1987.67, "fluvastatin": 1993.92,
        "pitavastatin": 2009.58,        # FDA 2009-08（>2001，影响 Era-1 排除）
        "ezetimibe": 2002.83,           # FDA 2002-10
        "fenofibrate": 1998.0,          # US 1998
 
        # ── 并发症：抗血小板 ─────────────────────────────────────────────────
        "aspirin": 1899.0,
        "clopidogrel": 1997.92,         # FDA 1997-11
        "cilostazol": 1999.0,           # FDA 1999
 
        # ── 并发症：神经病变 ─────────────────────────────────────────────────
        "pregabalin": 2004.92,          # FDA 2004-12
        "gabapentin": 1993.92,          # FDA 1993-12
        "mirogabalin": 2019.0,          # 日本 2019-01（关键 post-Era-1/2 药）
        "duloxetine": 2004.58,          # FDA 2004-08
        "amitriptyline": 1961.0, "nortriptyline": 1964.0, "imipramine": 1959.0,
        "mecobalamin": 1985.0,          # 日本 1980s (甲钴胺)
        "epalrestat": 1992.0,           # 日本 1992
        "capsaicin": 1990.0,            # 局部用分子（8% 贴片 Qutenza FDA 2009）
 
        # ── 并发症：微循环 / 视网膜（AntiVEGF 为主要泄漏源）──────────────────
        "alprostadil": 1981.0,
        "bevacizumab": 2004.17,         # FDA 2004-02（眼科为 off-label）
        "ranibizumab": 2006.5,          # FDA 2006-06
        "aflibercept": 2011.92,         # FDA 2011-11 (Eylea)
        "conbercept": 2013.92,          # 中国 2013-12 (康柏西普)
        "brolucizumab": 2019.83,        # FDA 2019-10
        "faricimab": 2022.08,           # FDA 2022-01
        "calcium_dobesilate": 1974.0,   # 欧洲 1970s
 
        # ── 并发症：胃轻瘫 / 低血糖急救 ──────────────────────────────────────
        "metoclopramide": 1979.0,       # FDA 1979
        "domperidone": 1979.0,          # 欧洲 1979（未获 FDA 批准）
        "glucagon": 1960.0,             # FDA 1960
        "oral_glucose": 1950.0,
    }
 
    # Era-aware 药物类别表（class_key, display_label），按 ada_year bucket
    _ERA_DRUG_CLASS_TABLE: Dict[int, List[Tuple[str, str]]] = {
        2008: [
            ("Biguanides",         "Biguanides"),
            ("Sulfonylureas",      "Sulfonylureas"),
            ("Thiazolidinediones", "Thiazolidinediones (TZD)"),
            ("Insulins_Basal",     "Insulin (basal / bolus)"),
            ("DPP-4i",             "DPP-4 inhibitors"),
            ("GLP-1RA",            "GLP-1 receptor agonists"),
        ],
        2010: [
            ("Biguanides",         "Biguanides"),
            ("DPP-4i",             "DPP-4 inhibitors"),
            ("GLP-1RA",            "GLP-1 receptor agonists"),
            ("Sulfonylureas",      "Sulfonylureas"),
            ("Thiazolidinediones", "Thiazolidinediones (TZD)"),
            ("Insulins_Basal",     "Insulin (basal / bolus)"),
        ],
        2014: [
            ("Biguanides",         "Biguanides"),
            ("SGLT2i",             "SGLT2 inhibitors"),
            ("GLP-1RA",            "GLP-1 receptor agonists"),
            ("DPP-4i",             "DPP-4 inhibitors"),
            ("Sulfonylureas",      "Sulfonylureas"),
            ("Thiazolidinediones", "Thiazolidinediones (TZD)"),
            ("Insulins_Basal",     "Insulin (basal / bolus)"),
        ],
        2016: [
            ("Biguanides",         "Biguanides"),
            ("SGLT2i",             "SGLT2 inhibitors"),
            ("GLP-1RA",            "GLP-1 receptor agonists"),
            ("DPP-4i",             "DPP-4 inhibitors"),
            ("Sulfonylureas",      "Sulfonylureas"),
            ("Thiazolidinediones", "Thiazolidinediones (TZD)"),
            ("Insulins_Basal",     "Insulin (basal / bolus)"),
        ],
        2020: [
            ("Biguanides",         "Biguanides"),
            ("SGLT2i",             "SGLT2 inhibitors"),
            ("GLP-1RA",            "GLP-1 receptor agonists"),
            ("DPP-4i",             "DPP-4 inhibitors"),
            ("Sulfonylureas",      "Sulfonylureas"),
            ("Thiazolidinediones", "Thiazolidinediones (TZD)"),
            ("Insulins_Basal",     "Insulin (basal / bolus)"),
        ],
        2024: [
            ("Biguanides",         "Biguanides"),
            ("SGLT2i",             "SGLT2 inhibitors"),
            ("GLP-1RA",            "GLP-1 receptor agonists"),
            ("GIP_GLP1_RA",        "GIP/GLP-1 dual agonists"),
            ("DPP-4i",             "DPP-4 inhibitors"),
            ("Sulfonylureas",      "Sulfonylureas"),
            ("Thiazolidinediones", "Thiazolidinediones (TZD)"),
            ("Insulins_Basal",     "Insulin (basal / bolus)"),
        ],
    }
 
    @classmethod
    def get_ada_year_ceiling(cls, admission_year: int) -> int:
        """将就诊年精确映射到可引用的最新 ADA 指南年份。"""
        available = [y for y in cls.ADA_PUBLICATION_YEARS if y <= admission_year]
        return max(available) if available else cls.ADA_PUBLICATION_YEARS[0]
 
    @classmethod
    def get_era_drug_classes(cls, ada_year: int) -> List[Tuple[str, str]]:
        """返回 ada_year 对应的药物类别列表（不超过该年的最近 bucket）。"""
        buckets = sorted(cls._ERA_DRUG_CLASS_TABLE.keys())
        bucket = buckets[0]
        for yr in buckets:
            if ada_year >= yr:
                bucket = yr
        return cls._ERA_DRUG_CLASS_TABLE[bucket]
 
    @classmethod
    def era_label_for_year(cls, ada_year: int) -> str:
        for era, yr in cls.ERA_TO_RAG_YEAR.items():
            if yr == ada_year:
                return era
        return f"ADA-{ada_year}"
 
    # [TEMPORAL-CONSISTENCY-V1] single temporal-registry lookup.
    @classmethod
    def get_drug_approval_year(cls, drug_inn: str) -> "Optional[float]":
        """Return registered first-availability year, else None.

        Matching is intentionally identical to the historical drug_available_at():
          1) full display name normalized with spaces -> underscores
          2) last-token INN stem fallback
        """
        if not drug_inn:
            return None
        key_full = str(drug_inn).strip().lower().replace(" ", "_")
        approval = cls.FDA_APPROVAL_YEAR.get(key_full)
        if approval is None:
            tokens = str(drug_inn).strip().lower().split()
            if len(tokens) > 1:
                approval = cls.FDA_APPROVAL_YEAR.get(tokens[-1])
        return float(approval) if approval is not None else None

    @classmethod
    def drug_available_at(cls, drug_inn: str, admission_year: int) -> bool:
        """Check whether the drug is available by the encounter year.

        Unknown/unregistered drugs remain conservative pass-through (True),
        matching the pre-patch behavior.
        """
        approval = cls.get_drug_approval_year(drug_inn)
        return True if approval is None else int(approval) <= int(admission_year)

    def filter_formulary_by_era(
        admission_year: int,
        dm_meds: "Dict[str, List[str]]" = None,
        complication_meds: "Dict[str, List[str]]" = None,
    ) -> "Tuple[Dict[str, List[str]], Dict[str, List[str]]]":
        """
        Sensitivity analysis helper: filter drug formulary to medications whose FDA
        approval year is ≤ admission_year.
        Drugs NOT in FDA_APPROVAL_YEAR pass through unchanged (conservative).
        Empty categories are dropped.
        """
        if dm_meds is None:
            dm_meds = DIABETES_MEDICATIONS
        if complication_meds is None:
            complication_meds = COMPLICATION_DRUGS
 
        def _filter(drug_dict):
            out = {}
            for cat, drugs in drug_dict.items():
                available = [d for d in drugs if DrugEraConfig.drug_available_at(d, admission_year)]
                if available:
                    out[cat] = available
            return out
 
        return _filter(dm_meds), _filter(complication_meds)
    
    @classmethod
    def get_available_dm_drugs(cls, ada_year: int) -> Dict[str, List[str]]:
        """
        返回 ada_year 时代可用的 DM 药物 dict（按 class_key 分组）。
        temporal_drug_filter=True 时由 prompt_builder 调用。
        """
        from agents.config import DIABETES_MEDICATIONS
        era_classes = cls.get_era_drug_classes(ada_year)
        available_keys = {ck for ck, _ in era_classes}
        # Insulins_Basal 同时覆盖 Insulins_Rapid
        if "Insulins_Basal" in available_keys:
            available_keys.add("Insulins_Rapid")
        result: Dict[str, List[str]] = {}
        for cat_key in available_keys:
            drugs = DIABETES_MEDICATIONS.get(cat_key, [])
            filtered = [
                d for d in drugs
                if cls.drug_available_at(d, ada_year)
            ]
            if filtered:
                result[cat_key] = filtered
        return result
 
    @classmethod
    def get_available_comp_drugs(cls, ada_year: int) -> Dict[str, List[str]]:
        """
        返回 ada_year 时代可用的并发症药 dict。
        temporal_drug_filter=True 时由 prompt_builder 调用。
        """
        from agents.config import COMPLICATION_DRUGS
        result: Dict[str, List[str]] = {}
        for cat_key, drugs in COMPLICATION_DRUGS.items():
            filtered = [
                d for d in drugs
                if cls.drug_available_at(d, ada_year)
            ]
            if filtered:
                result[cat_key] = filtered
        return result
 
# ═══════════════════════════════════════════════════════════════════════════════
# 八、临床编码与专科配置
# ═══════════════════════════════════════════════════════════════════════════════
 
HEMODYNAMIC_ICD_PREFIXES = ("458", "785", "I95", "R57")   # ICD-9 hypotension, shock; ICD-10 hypotension, shock


def hemodynamic_instability(patient_data: dict) -> bool:
    """Vasopressor use or a hypotension/shock diagnosis. Used for the clinical states at inference and for the
    hard constraints in evaluation."""
    comp = ((patient_data.get("clinical_metrics") or {}).get("dm_complications") or {})
    v = comp.get("vasopressor_use")
    if v is not None and not (isinstance(v, float) and v != v) and bool(v):
        return True
    return any(str((d or {}).get("icd_code", "") or "").strip().upper()[:3] in HEMODYNAMIC_ICD_PREFIXES
               for d in patient_data.get("all_diagnoses") or [] if isinstance(d, dict))


class ClinicalCodeConfig:
    """临床编码集合、专科触发词、失败簇标签等静态医学常量。"""
 
    ASCVD_ICD9:  Set[str] = {"410","411","412","413","414","430","431","432","433","434","435","436","437","438","440","441","443"}
    ASCVD_ICD10: Set[str] = {"I21","I22","I23","I24","I25","I63","I64","G45","I70","I71","I73","I74"}
    HF_ICD9:     Set[str] = {"428"}
    HF_ICD10:    Set[str] = {"I50"}
 
    SPECIALTY_TRIGGER_TAGS: Dict[str, List[str]] = {
        "nephrology":    ["CKD", "AKI", "ESRD", "肾", "renal", "kidney"],
        "cardiology":    ["ASCVD", "HF", "cardiovascular", "心", "cardiac"],
        "endocrinology": [],
    }
 
    CONFLICT_DRUGS: Set[str] = {
        "sglt2i", "metformin", "glp-1", "glp1", "insulin",
        "finerenone", "acei", "arb", "tzd", "pioglitazone",
        "saxagliptin", "alogliptin",
    }
 
    FAILURE_CLUSTER_LABELS: Dict[str, str] = {
        "A": "ICU暂停逻辑缺失（住院/NPO/AKI期间推荐了应暂停的口服药）",
        "B": "Era-1时代穿越（推荐了患者就诊年份尚未上市的药物）",
        "C": "肾功能临界误判（eGFR在阈值附近时分层错误）",
        "D": "过度保守（安全考虑过头，遗漏了有明确证据的有益药物）",
        "E": "MDT虚假冲突（跨族模型语义偏差导致的非真实临床分歧）",
        "F": "临床惰性复制（模型复现了真实处方中的惰性而非指南推荐）",
        "G": "RAG证据稀疏（ADA 2010 + Era-1查询返回空，退化为参数记忆推理）",
        "H": "其他/未分类",
    }
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# 九、住院临床上下文枚举
# ═══════════════════════════════════════════════════════════════════════════════
 
class InpatientContextConfig:
    """住院临床上下文字段枚举（v3 新增字段，影响 R3/R8/R9 规则）。"""
    INPATIENT_PHASE: Dict[str, str] = {
        "acute_unstable": "急性不稳定期（手术后24h内/血流动力学不稳定/机械通气）",
        "acute_stable":   "急性稳定期（已离开ICU/撤机/循环稳定，仍住院）",
        "pre_discharge":  "出院前准备期（病情稳定，计划出院，以出院处方为主）",
        "unknown":        "未知（默认保守策略）",
    }
    NUTRITION_STATUS: Dict[str, str] = {
        "npo":            "禁食（NPO）",
        "clear_liquid":   "流质（Clear Liquid）",
        "full_oral_diet": "经口正常进食",
        "tube_feeding":   "肠内营养（鼻饲/胃管）",
        "tpn":            "全肠外营养（TPN）",
        "unknown":        "未知",
    }
    GLUCOSE_PATTERN: Dict[str, str] = {
        "fasting_dominant":      "空腹血糖升高为主 → 倾向基础胰岛素",
        "postprandial_dominant": "餐后血糖升高为主 → 倾向速效胰岛素/DPP-4i",
        "mixed":                 "空腹+餐后均升高 → 基础+餐时方案或强化治疗",
        "corrective_only":       "仅偶发高血糖，需纠正剂量 → sliding scale",
        "unknown":               "未知",
    }
    PRESCRIPTION_TIMING: Dict[str, str] = {
        "admission_day_1":  "入院第1天（急性期，R3严格适用）",
        "inpatient_stable": "住院稳定期（R3部分放宽，可恢复谨慎口服药）",
        "discharge":        "出院处方（恢复家庭用药方案）",
        "unknown":          "未知（默认按住院急性期处理）",
    }
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# 十、VirtualMDT 专家激活配置（精简：仅保留激活逻辑，Prompt 模板移至 virtual_mdt_stage.py）
# ═══════════════════════════════════════════════════════════════════════════════
 
class VirtualMDTConfig:
    """Virtual MDT 激活逻辑配置。Prompt 模板见 virtual_mdt_stage.py。"""
 
    MDT_WORKFLOW: Dict[str, Any] = {
        "enable_parallel_consultation": True,
        "require_all_experts":          False,
        "conflict_threshold":           0.5,
        "enable_debate":                True,
        "max_debate_rounds":            2,
    }
 
    # 专家角色（轻量版：仅保留激活所需字段）
    EXPERTS: Dict[str, Dict] = {
        "endocrinologist": {"role_en": "Endocrinologist",  "model": "deepseek-r1-250528", "temperature": 0.3, "max_tokens": 4000, "deep_thinking": True,  "style": "aggressive"},
        "cardiologist":    {"role_en": "Cardiologist",     "model": "deepseek-r1-250528", "temperature": 0.2, "max_tokens": 4000, "deep_thinking": True,  "style": "organ_protective"},
        "nephrologist":    {"role_en": "Nephrologist",     "model": "deepseek-r1-250528", "temperature": 0.2, "max_tokens": 4000, "deep_thinking": True,  "style": "organ_protective"},
        "geriatrician":    {"role_en": "Geriatrician",     "model": "qwen3-235b-a22b",    "temperature": 0.3, "max_tokens": 3000, "deep_thinking": False, "style": "patient_centered"},
        "judge":           {"role_en": "MDT Chairperson",  "model": "Kimi-K2",            "temperature": 0.2, "max_tokens": 6000, "deep_thinking": True,  "style": "integrative"},
    }
 
    @classmethod
    def select_experts_for_patient(cls, patient_features: Dict) -> List[str]:
        selected = ["endocrinologist"]
        if patient_features.get("has_heart_failure") or patient_features.get("has_cardiovascular"):
            selected.append("cardiologist")
        if (patient_features.get("egfr") or 90) < 60:
            selected.append("nephrologist")
        if (patient_features.get("age") or 0) > 70:
            selected.append("geriatrician")
        return selected
 
    @classmethod
    def get_expert_config(cls, expert_name: str) -> Dict:
        return cls.EXPERTS.get(expert_name, {})
 
    @classmethod
    def get_all_expert_names(cls) -> List[str]:
        return [n for n in cls.EXPERTS if n != "judge"]
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# 十一、评估范围常量
# ═══════════════════════════════════════════════════════════════════════════════
 
EVAL_DM_ONLY_CATEGORIES: frozenset = frozenset(DIABETES_MEDICATIONS.keys())
 
# 并发症专属类别集合（= COMPLICATION_DRUGS.keys()，与模块级常量保持同步）
# 用于 complication_only_category_f1 计算（并发症用药拟合度独立评估）
# 注意：此集合与 EVAL_DM_ONLY_CATEGORIES 不相交，两者并集 = EVAL_FULL_DM_CATEGORIES
EVAL_COMPLICATION_ONLY_CATEGORIES: frozenset = frozenset(COMPLICATION_DRUGS.keys())
 
# 全量 DM 相关类别（降糖药 + 全部19个并发症类别）
# 修复：原版仅硬编码10个并发症类别，漏掉：
#   Hypertension_Diuretics, Microcirculation_Vasodilators, Anticoagulant,
#   Neuropathy_TCA, Neuropathy_Others, Retinopathy_AntiVEGF,
#   Retinopathy_Oral, Gastroparesis, Hypoglycemia_Rescue
EVAL_FULL_DM_CATEGORIES: frozenset = frozenset(
    EVAL_DM_ONLY_CATEGORIES | EVAL_COMPLICATION_ONLY_CATEGORIES
)
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# 十二、时代穿越检测 & GAS 关键词
# ═══════════════════════════════════════════════════════════════════════════════
 
DRUG_APPROVAL_YEAR: Dict[str, int] = {
    "dapagliflozin": 2012, "canagliflozin": 2013, "empagliflozin": 2014,
    "ertugliflozin": 2017, "sotagliflozin": 2023, "bexagliflozin": 2023,
    "exenatide": 2005, "liraglutide": 2010, "albiglutide": 2014, "dulaglutide": 2014,
    "semaglutide": 2017, "lixisenatide": 2016, "tirzepatide": 2022,
    "sitagliptin": 2006, "saxagliptin": 2009, "linagliptin": 2011,
    "alogliptin": 2013, "vildagliptin": 2007,
    "pioglitazone": 1999, "rosiglitazone": 1999, "metformin": 1994, "finerenone": 2021,
}
 
# GAS_GRADE_A_KEYWORDS: List[str] = [
#     "level a", "grade a", "recommendation a", "evidence a",
#     "strong recommendation", "high quality evidence",
#     "randomized controlled trial", "rct", "meta-analysis",
#     "empagliflozin", "dapagliflozin", "liraglutide", "semaglutide",
#     "canagliflozin", "dulaglutide",
#     "sglt2 inhibitor cardiovascular", "glp-1 cardiovascular",
#     "sglt2i heart failure", "sglt-2i ckd",
# ]
 
# GAS_GRADE_B_KEYWORDS: List[str] = [
#     "level b", "grade b", "recommendation b", "evidence b",
#     "moderate evidence", "well-conducted cohort",
#     "metformin first-line", "sglt2 renal protection",
#     "generally recommended", "preferred in patients",
# ]
 
 
# ═══════════════════════════════════════════════════════════════════════════════
# 十三、ConfigManager（精简版）
# ═══════════════════════════════════════════════════════════════════════════════
 
class ConfigManager:
    """配置验证工具（精简版，移除 export/print 调试方法）。"""
 
    @staticmethod
    def validate_api_config() -> Tuple[bool, str]:
        if not APIConfig.API_KEY:
            return False, "❌ API Key 未配置（设置环境变量 DIAMAS_API_KEY）"
        if not APIConfig.BASE_URL.startswith("http"):
            return False, "❌ BASE_URL 格式错误"
        if APIConfig.TIMEOUT <= 0:
            return False, "❌ TIMEOUT 必须大于 0"
        return True, "✅ API 配置有效"
 
    @staticmethod
    def validate_all() -> bool:
        ok, msg = ConfigManager.validate_api_config()
        print(msg)
        return ok
    
# from typing import Dict, Optional, Tuple, List
 
INDICATIONS: List[str] = [
    "glycemic",          # 单纯降糖
    "ascvd",             # 动脉粥样硬化性心血管病(一/二级预防)
    "heart_failure",     # 心衰
    "renoprotection",    # 糖尿病肾病/CKD 肾保护
    "hypertension",      # 高血压
    "dyslipidemia",      # 血脂
    "antiplatelet",      # 抗血小板
    "hypoglycemia_safety",  # 低血糖安全性(多为 AVOID 语义)
    "general",           # 无特异心肾适应症
]
 
# (drug_class, indication) -> (first_SoC_year, landmark, note)
LANDMARK_SOC_YEAR: Dict[Tuple[str, str], Tuple[int, str, str]] = {
    # ── SGLT2i ───────────────────────────────────────────────────────────────
    ("SGLT2i", "glycemic"):        (2013, "Canagliflozin FDA 2013",         "类首药上市即降糖标准"),
    ("SGLT2i", "ascvd"):           (2016, "EMPA-REG OUTCOME 2015",          "2016 ADA 起纳入 CV 获益"),
    ("SGLT2i", "heart_failure"):   (2020, "DAPA-HF 2019 / EMPEROR-Reduced 2020", "2020 起 HF 一线"),
    ("SGLT2i", "renoprotection"):  (2020, "CREDENCE 2019 / DAPA-CKD 2020",  "KDIGO 2022 强推；SoC≈2020"),
    # ── GLP-1RA ──────────────────────────────────────────────────────────────
    ("GLP-1RA", "glycemic"):       (2005, "Exenatide FDA 2005",            "类首药上市"),
    ("GLP-1RA", "ascvd"):          (2016, "LEADER / SUSTAIN-6 2016",       "2016 起 CV 获益"),
    ("GLP-1RA", "renoprotection"): (2020, "ADA 2020 / KDIGO 2022: CKD 优先或 SGLT2i 替代", "FLOW 2024 为肾结局硬证据, 但 CKD 优先用法 SoC≈2020"),
    ("GLP-1RA", "general"):        (2016, "心代谢获益证据 2016",            ""),
    # ── GIP/GLP-1 (tirzepatide) ──────────────────────────────────────────────
    ("GIP_GLP1_RA", "glycemic"):   (2022, "Tirzepatide FDA 2022",          ""),
    ("GIP_GLP1_RA", "ascvd"):      (2024, "SURPASS-CVOT 2024",             "MIMIC 期外"),
    # ── Finerenone / CKD_MRA ─────────────────────────────────────────────────
    ("CKD_MRA", "renoprotection"): (2021, "FIDELIO-DKD 2020 / FDA 2021",   "KDIGO 2022；finerenone 2021.5 可得"),
    ("CKD_MRA", "heart_failure"):  (2003, "EPHESUS 2003 (eplerenone post-MI HFrEF) / RALES 1999", "甾体MRA post-MI HFrEF 老标准"),
    ("Finerenone", "renoprotection"): (2021, "FIDELIO-DKD 2020 / FDA 2021", ""),
    # ── ACEi / ARB ───────────────────────────────────────────────────────────
    ("Hypertension_ACEi", "renoprotection"): (2001, "RENAAL/IDNT 2001 era",  "ADA 2003 起强推"),
    ("Hypertension_ACEi", "hypertension"):   (1994, "长期标准",              ""),
    ("Hypertension_ACEi", "heart_failure"):  (2000, "SOLVD 1991 / CONSENSUS 1987", "ACEi HFrEF 老标准, 取保守 2000"),
    ("Hypertension_ARB", "renoprotection"):  (2001, "RENAAL/IDNT 2001",      "ARB 蛋白尿肾保护"),
    ("Hypertension_ARB", "hypertension"):    (1995, "长期标准",              ""),
    ("Hypertension_ARB", "heart_failure"):   (2001, "Val-HeFT 2001 / CHARM 2003", "ARB 用于 HFrEF"),
    # ── 他汀 ──────────────────────────────────────────────────────────────────
    ("Lipid_Statins", "ascvd"):        (2004, "CARDS 2004",                "ADA 2006/2008 起强推"),
    ("Lipid_Statins", "dyslipidemia"): (2004, "4S 1994 / CARDS 2004",      "他汀治血脂长期标准"),
    # ── 非他汀降脂 (ezetimibe / PCSK9) ────────────────────────────────────────
    ("Lipid_NonStatins", "dyslipidemia"): (2015, "IMPROVE-IT 2015 / PCSK9 2015", "非他汀获益证据 2015 起"),
    ("Lipid_NonStatins", "ascvd"):        (2015, "IMPROVE-IT 2015 / PCSK9 2015", ""),
    # ── 抗血小板 ──────────────────────────────────────────────────────────────
    ("Antiplatelet", "ascvd"):        (2002, "二级预防长期标准 / ATT 2009", ""),
    # ── TZD (pioglitazone) ───────────────────────────────────────────────────
    ("Thiazolidinediones", "ascvd"):  (2016, "IRIS 2016",                  "吡格列酮减 ASCVD(无 HF)"),
    ("Thiazolidinediones", "glycemic"):(1999, "Pioglitazone FDA 1999",     ""),
    # ── β受体阻滞剂 ───────────────────────────────────────────────────────────
    ("Hypertension_BetaBlockers", "heart_failure"): (1999, "CIBIS-II/MERIT-HF 1999", ""),
    # ── 胰岛素 / 双胍 / DPP4(纯降糖，回落药物可得年也可) ───────────────────────
    ("Insulins_Basal", "glycemic"):   (1922, "Insulin 1922",               ""),
    ("Insulins_Rapid", "glycemic"):   (1996, "Lispro 1996",                ""),
    ("Biguanides", "glycemic"):       (1995, "Metformin US 1995",          ""),
    ("DPP-4i", "glycemic"):           (2006, "Sitagliptin FDA 2006",       ""),
}
 
def lookup(any_of_cats: List[str], indication: str) -> Tuple[Optional[int], str, bool]:
    """返回 (landmark_year, justification, hit). 多 cat 取最早(最早可循证标准化年)。"""
    hits = []
    for cat in any_of_cats:
        ent = LANDMARK_SOC_YEAR.get((cat, indication))
        if ent:
            hits.append(ent)
    if not hits:
        return None, f"landmark_miss:({any_of_cats},{indication})", False
    y, lm, note = min(hits, key=lambda e: e[0])
    return y, f"{lm}" + (f"; {note}" if note else ""), True
