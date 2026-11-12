#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evaluation/drug_normalizer.py
==============================
药物名标准化器（从 baseline_evaluation.py 抽出）

核心修复（v2）：
  - 新增 filter_dm_relevant_drugs() 方法
    将 ground truth 过滤到 DrugNormalizer 可映射的范围内，
    使 GT 范围与 LLM 推荐范围完全一致，消除 ICU 非相关药物（抗生素/镇静剂等）的噪声。

v3 修复：
  - DRUG_ALIAS_MAP 补全：覆盖 drug_taxonomy_scanner 发现的 78 个 DM 相关 UNMAPPED 药物
    包括 eICU 原始格式、反序写法、品牌+Insulin 模式、预混制剂、SymlinPen 变体等

v4 修复：
  - DRUG_ALIAS_MAP 最终补全：剩余 24 个 1x UNMAPPED（拼写错误、旧制剂、协议名等）
  - _preprocess() 增强：与 scanner._clean() 同步增加 *NF*、预混比例、tab、FlexPen粘连等清洗
  - _alias() partial 循环 None 守卫加固

四级标准化策略（优先级顺序）：
  1. 精确匹配（大小写不敏感）
  2. 别名表查找（DRUG_ALIAS_MAP）
  3. 模糊匹配（difflib cutoff=0.82）
  4. LLM 辅助（最后手段，可禁用）
  [注] RxNorm API 已移除（需要网络，ICU 离线环境不可用）
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass
from difflib import get_close_matches
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ── 需要从 config 导入药物列表 ───────────────────────────────────────────────
# 避免循环导入：仅在函数内延迟导入
def _load_drug_list() -> List[Tuple[str, str]]:
    """延迟加载标准药物列表，避免启动时循环导入。"""
    from agents.config import create_diabetes_drug_list
    return create_diabetes_drug_list()


# ═══════════════════════════════════════════════════════════════════════════════
# 一、别名映射表
# ═══════════════════════════════════════════════════════════════════════════════
# 设计原则：
#   - 覆盖 MIMIC EHR 中常见的非标准写法（品牌名、缩写、带剂量字符串）
#   - key 全小写，value 为 create_diabetes_drug_list() 中的标准名
#   - 新增品牌名时，确保 value 在标准列表中存在

DRUG_ALIAS_MAP: Dict[str, str] = {
    # ── Insulin 家族 ──────────────────────────────────────────────────────────
    "insulin regular":              "Regular insulin",
    "regular insulin":              "Regular insulin",
    "insulin human regular":        "Regular insulin",
    "humulin r":                    "Regular insulin",
    "novolin r":                    "Regular insulin",
    "insulin r":                    "Regular insulin",
    "nph":                          "NPH insulin",
    "insulin nph":                  "NPH insulin",
    "humulin n":                    "NPH insulin",
    "novolin n":                    "NPH insulin",
    "glargine":                     "Insulin glargine",
    "lantus":                       "Insulin glargine",
    "basaglar":                     "Insulin glargine",
    "toujeo":                       "Insulin glargine",
    "insulin glargine u-100":       "Insulin glargine",
    "insulin glargine u-300":       "Insulin glargine",
    "detemir":                      "Insulin detemir",
    "levemir":                      "Insulin detemir",
    "degludec":                     "Insulin degludec",
    "tresiba":                      "Insulin degludec",
    "lispro":                       "Insulin lispro",
    "humalog":                      "Insulin lispro",
    "insulin lispro u-100":         "Insulin lispro",
    "admelog":                      "Insulin lispro",
    "lyumjev":                      "Insulin lispro",
    "aspart":                       "Insulin aspart",
    "novolog":                      "Insulin aspart",
    "novorapid":                    "Insulin aspart",
    "fiasp":                        "Faster insulin aspart",
    "faster aspart":                "Faster insulin aspart",
    "glulisine":                    "Insulin glulisine",
    "apidra":                       "Insulin glulisine",
    "insulin human":                "Insulin human",
    "humulin 70/30":                "NPH insulin",     # 70/30 → 归入 NPH（临床常见）
    "novolin 70/30":                "NPH insulin",
    "insulin 70/30":                "NPH insulin",
    "icodec":                       "Insulin icodec",
    "awiqli":                       "Insulin icodec",
    # ── Metformin ─────────────────────────────────────────────────────────────
    "glucophage":                   "Metformin",
    "metformin hcl":                "Metformin",
    "metformin hydrochloride":      "Metformin",
    "fortamet":                     "Metformin",
    "glumetza":                     "Metformin",
    "riomet":                       "Metformin",
    "metformin er":                 "Metformin",
    "metformin xr":                 "Metformin",
    # ── SGLT-2i ───────────────────────────────────────────────────────────────
    "jardiance":                    "Empagliflozin",
    "empagliflozin 10mg":           "Empagliflozin",
    "empagliflozin 25mg":           "Empagliflozin",
    "farxiga":                      "Dapagliflozin",
    "forxiga":                      "Dapagliflozin",
    "dapagliflozin 10mg":           "Dapagliflozin",
    "invokana":                     "Canagliflozin",
    "canagliflozin 100mg":          "Canagliflozin",
    "canagliflozin 300mg":          "Canagliflozin",
    "steglatro":                    "Ertugliflozin",
    "brenzavvy":                    "Bexagliflozin",
    "inpefa":                       "Sotagliflozin",
    # ── GLP-1RA ───────────────────────────────────────────────────────────────
    "ozempic":                      "Semaglutide",
    "rybelsus":                     "Semaglutide",
    "wegovy":                       "Semaglutide",
    "semaglutide sc":               "Semaglutide",
    "semaglutide oral":             "Semaglutide",
    "victoza":                      "Liraglutide",
    "saxenda":                      "Liraglutide",
    "trulicity":                    "Dulaglutide",
    "byetta":                       "Exenatide",
    "bydureon":                     "Exenatide",
    "adlyxin":                      "Lixisenatide",
    "lyxumia":                      "Lixisenatide",
    "eperzan":                      "Albiglutide",
    "tanzeum":                      "Albiglutide",
    # ── GIP/GLP-1 双靶点 ──────────────────────────────────────────────────────
    "mounjaro":                     "Tirzepatide",
    "zepbound":                     "Tirzepatide",
    "tirzepatide 2.5mg":            "Tirzepatide",
    "tirzepatide 5mg":              "Tirzepatide",
    # ── DPP-4i ────────────────────────────────────────────────────────────────
    "januvia":                      "Sitagliptin",
    "sitagliptin 100mg":            "Sitagliptin",
    "tradjenta":                    "Linagliptin",
    "trajenta":                     "Linagliptin",
    "onglyza":                      "Saxagliptin",
    "nesina":                       "Alogliptin",
    "galvus":                       "Vildagliptin",
    "vipidia":                      "Alogliptin",
    # ── Sulfonylureas ─────────────────────────────────────────────────────────
    "glucotrol":                    "Glipizide",
    "glucotrol xl":                 "Glipizide",
    "glipizide xl":                 "Glipizide",
    "diabeta":                      "Glyburide",
    "glynase":                      "Glyburide",
    "micronase":                    "Glyburide",
    "amaryl":                       "Glimepiride",
    "diamicron":                    "Gliclazide",
    "diamicron mr":                 "Gliclazide",
    # ── Thiazolidinediones ────────────────────────────────────────────────────
    "actos":                        "Pioglitazone",
    "avandia":                      "Rosiglitazone",
    # ── Meglitinides ──────────────────────────────────────────────────────────
    "prandin":                      "Repaglinide",
    "starlix":                      "Nateglinide",
    "glufast":                      "Mitiglinide",
    # ── Alpha-glucosidase inhibitors ──────────────────────────────────────────
    "glucobay":                     "Acarbose",
    "precose":                      "Acarbose",
    "glyset":                       "Miglitol",
    # ── CKD-MRA ───────────────────────────────────────────────────────────────
    "kerendia":                     "Finerenone",
    "aldactone":                    "Spironolactone",
    "inspra":                       "Eplerenone",
    # ── 降压药（常见品牌名）───────────────────────────────────────────────────
    "zestril":                      "Lisinopril",
    "prinivil":                     "Lisinopril",
    "vasotec":                      "Enalapril",
    "altace":                       "Ramipril",
    "cozaar":                       "Losartan",
    "diovan":                       "Valsartan",
    "micardis":                     "Telmisartan",
    "avapro":                       "Irbesartan",
    "norvasc":                      None,          # ⛔ CCB 非糖尿病特异,与 BB/噻嗪一致排除(NOT_RELEVANT)
    "lasix":                        "Furosemide",
    # ── ARNI（HFrEF 一线，独立于 ARB） ─────────────────────────────────────────
    "entresto":                     "Sacubitril/valsartan",
    "sacubitril/valsartan":         "Sacubitril/valsartan",
    "sacubitril valsartan":         "Sacubitril/valsartan",
    "sacubitril-valsartan":         "Sacubitril/valsartan",
    "sacubitril":                   "Sacubitril/valsartan",   # combination only marketed
    # ── 他汀类 ────────────────────────────────────────────────────────────────
    "lipitor":                      "Atorvastatin",
    "crestor":                      "Rosuvastatin",
    "zocor":                        "Simvastatin",
    "pravachol":                    "Pravastatin",
    # ── Lipid_NonStatins（ezetimibe + fibrate）─────────────────────────────────
    "zetia":                        "Ezetimibe",
    "ezetimibe":                    "Ezetimibe",
    "tricor":                       "Fenofibrate",
    "fenofibrate":                  "Fenofibrate",
    "trilipix":                     "Fenofibrate",
    # ── Lipid_PCSK9_Pathway（PCSK9 mAb + siRNA）────────────────────────────────
    "repatha":                      "Evolocumab",
    "evolocumab":                   "Evolocumab",
    "praluent":                     "Alirocumab",
    "alirocumab":                   "Alirocumab",
    "leqvio":                       "Inclisiran",
    "inclisiran":                   "Inclisiran",
    # ── Lipid_NonStatins（IPE + bempedoic acid）────────────────────────────────
    "vascepa":                      "Icosapent ethyl",
    "icosapent":                    "Icosapent ethyl",
    "icosapent ethyl":              "Icosapent ethyl",
    "ethyl icosapentate":           "Icosapent ethyl",
    "nexletol":                     "Bempedoic acid",
    "bempedoic":                    "Bempedoic acid",
    "bempedoic acid":               "Bempedoic acid",
    "nexlizet":                     "Bempedoic acid",        # combo with ezetimibe (we normalize to primary)
    # ── 抗血小板 ──────────────────────────────────────────────────────────────
    "plavix":                       "Clopidogrel",
    "brilinta":                     "Ticagrelor",
    "effient":                      "Prasugrel",
    # ── 神经病变用药 ───────────────────────────────────────────────────────────
    "lyrica":                       "Pregabalin",
    "neurontin":                    "Gabapentin",
    "cymbalta":                     "Duloxetine",
    "elavil":                       "Amitriptyline",
    # ── 已从 COMPLICATION_DRUGS 移除的药物 → 显式排除，防止 fuzzy/rxnorm 重新命中 ──
    # （即使持久化缓存被清除后重建，也不会再被纳入）
    "amlodipine":                   None,          # ⛔ CCB 非糖尿病特异,与 BB/噻嗪一致排除(NOT_RELEVANT)
    "amlodipine besylate":          None,          # ⛔
    "levamlodipine":                None,          # ⛔
    "s-amlodipine":                 None,          # ⛔
    "nitrendipine":                 None,          # ⛔
    "norvasc 5mg":                  None,          # ⛔
    "norvasc 10mg":                 None,          # ⛔
    "verapamil":                    None,          # ⛔ 从 CCB 移除
    "verapamil hcl":                None,          # ⛔ 覆盖 rxnorm 路径
    "verapamil sr":                 None,          # ⛔
    "verapamil hydrochloride":      None,          # ⛔
    "calan":                        None,          # ⛔ Verapamil 品牌名
    "isoptin":                      None,          # ⛔
    "nifedipine":                   None,          # ⛔ 从 CCB 移除
    "felodipine":                   None,          # ⛔
    "nimodipine":                   None,          # ⛔
    "hydrochlorothiazide":          None,          # ⛔ 从 Diuretics 移除
    "hctz":                         None,          # ⛔
    "chlorthalidone":               None,          # ⛔
    "chlorothiazide":               None,          # ⛔
    "indapamide":                   None,          # ⛔ 从 Diuretics 移除
    "labetalol":                    None,          # ⛔ 从 BetaBlockers 移除
    "labetalol hcl":                None,          # ⛔
    "propranolol":                  None,          # ⛔ 从 BetaBlockers 移除
    "propranolol hcl":              None,          # ⛔
    "propranolol la":               None,          # ⛔
    "atenolol":                     None,          # ⛔ 从 BetaBlockers 移除
    "warfarin":                     None,          # ⛔ Anticoagulant 整类已移除
    "warfarin sodium":              None,          # ⛔
    "coumadin":                     None,          # ⛔ Warfarin 品牌名
    "apixaban":                     None,          # ⛔
    "eliquis":                      None,          # ⛔ Apixaban 品牌名
    "rivaroxaban":                  None,          # ⛔
    "xarelto":                      None,          # ⛔
    "dabigatran":                   None,          # ⛔
    "pradaxa":                      None,          # ⛔
    "edoxaban":                     None,          # ⛔
    "savaysa":                      None,          # ⛔
    "enoxaparin":                   None,          # ⛔ 从 Antiplatelet 移除
    "enoxaparin sodium":            None,          # ⛔
    "lovenox":                      None,          # ⛔ Enoxaparin 品牌名
    "heparin":                      None,          # ⛔ 从 Antiplatelet 移除（所有 Heparin 排除）
    "gemfibrozil":                  None,          # ⛔ 从 Lipid_NonStatins 移除
    "lopid":                        None,          # ⛔ Gemfibrozil 品牌名
    "venlafaxine":                  None,          # ⛔ 从 Neuropathy_SNRI 移除
    "venlafaxin":                   None,          # ⛔ 拼写变体
    "effexor":                      None,          # ⛔ Venlafaxine 品牌名
    "alpha lipoic acid":            None,          # ⛔ 从 Neuropathy_Others 移除
    "alpha-lipoic acid":            None,          # ⛔
    "mexiletine":                   None,          # ⛔ 从 Neuropathy_Others 移除
    "mexitil":                      None,          # ⛔ Mexiletine 品牌名
    # v3 新增排除
    "diltiazem":                    None,          # ⛔ 从 CCB 移除（两数据集 NOT_RELEVANT）
    "cardizem":                     None,          # ⛔ Diltiazem 品牌名
    "diltiazem hcl":                None,          # ⛔
    "diltiazem cd":                 None,          # ⛔ 缓释剂型
    "diltiazem er":                 None,          # ⛔
    "diltiazem sr":                 None,          # ⛔
    "pentoxifylline":               None,          # ⛔ 从 Microcirculation 移除
    "trental":                      None,          # ⛔ Pentoxifylline 品牌名
    # 特殊处方写法排除（非治疗性用途）
    # 注意：_preprocess 会把 "desensitization" 词去掉，所以必须在 raw name 层面拦截
    # normalize() 的 raw_lower 检查会匹配这些条目（在 _preprocess 之前）
    "aspirin desensitization":                 None,          # ⛔ 脱敏方案，非 DM 用途
    "aspirin desensitization (angioedema)":    None,          # ⛔ 原始写法带括号
    "aspirin desensitization protocol":        None,          # ⛔ 协议名变体
    # v4 新增排除
    "nebivolol":                    None,          # ⛔ 从 BetaBlockers 移除（两数据集 NOT_RELEVANT）
    "nebivolol hcl":                None,          # ⛔
    "bystolic":                     None,          # ⛔ Nebivolol 品牌名
    "desvenlafaxine":               None,          # ⛔ 从 Neuropathy_SNRI 移除（ADA 无独立推荐）
    "desvenlafaxine succinate":     None,          # ⛔
    "pristiq":                      None,          # ⛔ Desvenlafaxine 品牌名
    "khedezla":                     None,          # ⛔
    # Diltiazem 模糊匹配兜底（"Diltiaz" fuzzy → Diltiazem 残留）
    "diltiaz":                      None,          # ⛔ 截断拼写，fuzzy 前被 exact/alias 拦截
    # ── Heparin flush/infusion → 排除（详细变体见上方显式排除区块）──────────
    "heparin sodium lock flush":        None,          # ⛔ 排除：IV 管路维护
    "heparin flush":                    None,          # ⛔ 排除（精确 + partial anchor）
    "heparin flush cvl":                None,          # ⛔
    "heparin flush picc":               None,          # ⛔
    "heparin flush port":               None,          # ⛔ Port 维护
    "heparin flush midline":            None,          # ⛔ Midline 维护
    "heparin flush hickman":            None,          # ⛔ Hickman 维护
    "heparin flush crrt":               None,          # ⛔ CRRT 抗凝（非 DM）
    "heparin flush art":                None,          # ⛔ 动脉管路
    "heparin flush cvl":                None,          # ⛔
    "heparin dwell":                    None,          # ⛔
    "heparin lock flush":               None,          # ⛔
    "heparin lock":                     None,          # ⛔ 管路锁定
    # Heparin infusion 变体（alias_partial exclusion anchor）
    "heparin in d5w":                   None,          # ⛔ 覆盖 "HEPARIN IN D5W 50-5 ..."
    "heparin in nacl":                  None,          # ⛔ 覆盖 "HEPARIN IN NACL ..."
    "heparin porcine":                  None,          # ⛔ 覆盖 "HEPARIN, PORCINE (PF) ..."
    "heparin/dextrose":                 None,          # ⛔
    "heparin dextrose":                 None,          # ⛔ 清洗后形式
    # ⚠ None 值表示"已知药物但应排除"——需要在 DrugNormalizer 中处理
    
    # ── Enoxaparin / Beta Blockers / Diuretics ──
    # Enoxaparin、Atenolol、Propranolol 已从 COMPLICATION_DRUGS 移除，
    # 显式 None 排除条目在上方"已移除药物"区块，此处不再正向映射。

    # ── Diuretics ──
    "bumetanide":                       "Bumetanide",
    
    # ── 旧磺脲类 ──
    "chlorpropamide":                   "Chlorpropamide",
    "tolbutamide":                      "Tolbutamide",
    "tolazamide":                       "Tolazamide",
    "acetohexamide":                    "Acetohexamide",
    
    # ── 新类别药物 ──
    "pramlintide":                      "Pramlintide",
    "symlin":                           "Pramlintide",
    "colesevelam":                      "Colesevelam",
    "welchol":                          "Colesevelam",
    "bromocriptine":                    "Bromocriptine",
    "bromocriptine mesylate":           "Bromocriptine",
    "cycloset":                         "Bromocriptine",
    
    # ── eICU 特殊格式（前缀冒号 + 剂型后缀） ──
    "insulin regular human":            "Regular insulin",
    "insulin reg":                      "Regular insulin",
    "insulin regular, human":           "Regular insulin",
    "insulin-lispro":                   "Insulin lispro",
    "insulin-glargine":                 "Insulin glargine",
    "insulin-aspart":                   "Insulin aspart",
    "humulin-r insulin":                "Regular insulin",
    "humulin-r":                        "Regular insulin",
    "insulin pump":                     None,          # ⛔ 排除：设备，不是药物
    
    # ── Tresiba 变体 ──
    "tresiba u-200":                    "Insulin degludec",
    "tresiba u-100":                    "Insulin degludec",
    
    # ── Victoza 变体 ──
    "victoza 2-pak":                    "Liraglutide",
    "victoza 3-pak":                    "Liraglutide",

    # ══════════════════════════════════════════════════════════════════════════
    # v3 新增：覆盖 drug_taxonomy_scanner 发现的 DM 相关 UNMAPPED 药物
    # ══════════════════════════════════════════════════════════════════════════

    # ── eICU 原始格式（前缀冒号 + 剂型缩写） ─────────────────────────────────
    ": insulin regular human /ml ij soln":   "Regular insulin",
    "insulin regular human /ml ij soln":     "Regular insulin",
    "insulin regular human /ml inj soln":    "Regular insulin",
    "insulin reg 100 units":                 "Regular insulin",
    "insulin 100 units in 1":               "Regular insulin",
    "insulin 100 units":                     "Regular insulin",
    "insulin 100":                           "Regular insulin",
    "bag custom ndc : insulin infusion":     "Regular insulin",
    "bag custom ndc insulin infusion":       "Regular insulin",
    "insulin infusion":                      "Regular insulin",
    "insulin-lispro *unit* inj":             "Insulin lispro",
    "insulin-glargine *unit* inj":           "Insulin glargine",
    "insulin-regular *unit* inj":            "Regular insulin",
    "insulin-regular":                       "Regular insulin",
    "insulin hum-reg":                       "Regular insulin",
    "insulin hum reg":                       "Regular insulin",

    # ── 反序写法（eICU / MIMIC-IV） ──────────────────────────────────────────
    "aspart insulin":                        "Insulin aspart",
    "lispro insulin":                        "Insulin lispro",
    "glargine insulin":                      "Insulin glargine",
    "detemir insulin":                       "Insulin detemir",
    "glulisine insulin":                     "Insulin glulisine",
    "degludec insulin":                      "Insulin degludec",

    # ── "品牌名 + Insulin" 模式（MIMIC-IV） ──────────────────────────────────
    "lantus insulin":                        "Insulin glargine",
    "insulin lantus":                        "Insulin glargine",
    "levemir insulin":                       "Insulin detemir",
    "insulin levemir":                       "Insulin detemir",
    "levemir insuln":                        "Insulin detemir",    # 拼写错误
    "levemir flex":                          "Insulin detemir",
    "levemir /ml":                           "Insulin detemir",
    "tresiba insulin":                       "Insulin degludec",
    "toujeo insulin":                        "Insulin glargine",
    "toujeo max":                            "Insulin glargine",   # Toujeo Max SoloStar (U-300)
    "basaglar insulin":                      "Insulin glargine",
    "fiasp insulin":                         "Faster insulin aspart",
    "humalog insulin":                       "Insulin lispro",
    "insulin humalog":                       "Insulin lispro",
    "humalog junior":                        "Insulin lispro",
    "novolog insulin":                       "Insulin aspart",
    "insulin novolog":                       "Insulin aspart",
    "apidra insulin":                        "Insulin glulisine",
    "insulin reg human":                     "Regular insulin",

    # ── 预混胰岛素（Mix 制剂） ───────────────────────────────────────────────
    "novolog mix 70-30":                     "Insulin aspart",    # 预混 → 归入速效成分
    "novolog mix 70/30":                     "Insulin aspart",
    "novolog mix":                           "Insulin aspart",
    "humalog mix 75-25":                     "Insulin lispro",
    "humalog mix 75/25":                     "Insulin lispro",
    "humalog mix":                           "Insulin lispro",

    # ── Pramlintide 设备变体 ─────────────────────────────────────────────────
    "symlinpen 60":                          "Pramlintide",
    "symlinpen 120":                         "Pramlintide",
    "symlinpen":                             "Pramlintide",
    "symlin pen":                            "Pramlintide",

    # ── 特殊用途胰岛素 ──────────────────────────────────────────────────────
    "insulin for hyperkalemia":              None,                # ⛔ 高血钾急救用 RI，非 DM 管理
    "insulin hyperkalemia":                  None,                # ⛔
    "insulin intraperitoneal":               None,                # ⛔ 排除：腹膜透析泵，极罕见

    # ══════════════════════════════════════════════════════════════════════════
    # v4 新增：剩余 24 个 1x UNMAPPED 的最终清扫
    # ══════════════════════════════════════════════════════════════════════════

    # ── 裸品牌名（无型号后缀） ───────────────────────────────────────────────
    "humulin":                               "Insulin human",     # 裸品牌 → 归入 human insulin

    # ── 拼写错误 / 变体 ─────────────────────────────────────────────────────
    "insulin-levimir":                       "Insulin detemir",   # levemir typo
    "insulin levimir":                       "Insulin detemir",
    "insulin-levmir":                        "Insulin detemir",
    "insulin tuojeo":                        "Insulin glargine",  # toujeo typo
    "insulin tuoujeo":                       "Insulin glargine",
    "stoujeo insulin":                       "Insulin glargine",  # sToujeo typo（首字母 s 残留）
    "stoujeo":                               "Insulin glargine",
    "tresiba nf":                            "Insulin degludec",  # NF = non-formulary

    # ── 预混变体（清洗后残留形式） ───────────────────────────────────────────
    "humalog 75/25":                         "Insulin lispro",
    "humalog 75-25":                         "Insulin lispro",
    "humalog mix 50-50":                     "Insulin lispro",    # 50/50 预混
    "humalog mix 50/50":                     "Insulin lispro",
    "novolog mix 70-30flexpen":              "Insulin aspart",    # FlexPen 粘连

    # ── 旧制剂（已停产） ────────────────────────────────────────────────────
    "ultralente":                            "Insulin human",     # 超长效人胰岛素，已停产
    "ultralente insulin":                    "Insulin human",
    "ultralente humulin insulin":            "Insulin human",
    "insulin ultralente":                    "Insulin human",
    "humulin ultralente":                    "Insulin human",

    # ── 协议/方案名（不是药物） → 排除 ──────────────────────────────────────
    "insulin - sliding scale":               None,                # ⛔ 滑动剂量方案
    "insulin sliding scale":                 None,                # ⛔
    "insulin-symlin":                        None,                # ⛔ 联用标注，不是单一药物
    "insulin symlin":                        None,                # ⛔

    # ── 浓缩胰岛素 ──────────────────────────────────────────────────────────
    "insulin regular hum conc":              "Regular insulin",   # concentrated human regular
    "insulin regular human conc":            "Regular insulin",

    # ── NPH 变体 ────────────────────────────────────────────────────────────
    "insulin nph human recomb":              "NPH insulin",
    "insulin nph human":                     "NPH insulin",

    # ── 品牌名 + Tab/NF 等清洗残留 ──────────────────────────────────────────
    "januvia tab":                           "Sitagliptin",
    "tradjenta tab":                         "Linagliptin",
    "fiasp 100u":                            "Faster insulin aspart",
    "fiasp insulin":                         "Faster insulin aspart",

    # ── Hypoglycemia_Rescue 口服葡萄糖制剂 ────────────────────────────────────
    # 口服葡萄糖凝胶/片：清醒患者轻中度低血糖一线 rescue (ADA 15-15 rule)
    # 不替代Glucagon（严重低血糖需注射）；同类别but不同临床场景
    "glucose gel":                  "Oral glucose",
    "oral glucose":                 "Oral glucose",
    "oral glucose gel":             "Oral glucose",
    "glucose gel 40%":              "Oral glucose",
    "glucose oral":                 "Oral glucose",
    "glutose":                      "Oral glucose",
    "glutose 15":                   "Oral glucose",
    "insta-glucose":                "Oral glucose",
    "instaglucose":                 "Oral glucose",
    "dex4":                         "Oral glucose",
    "bd glucose":                   "Oral glucose",
    "glucose tablet":               "Oral glucose",
    "glucose tablets":              "Oral glucose",
    
    # ── *NF* 前缀清洗残留 ───────────────────────────────────────────────────
    "nf novolog":                            "Insulin aspart",    # *NF* 清洗后残留 "NF"
    "nf humalog":                            "Insulin lispro",
    "nf lantus":                             "Insulin glargine",
    "novolin":                               "Insulin human",     # 裸品牌名
    "insulin regular hum":                   "Regular insulin",   # human 缩写
}


# ═══════════════════════════════════════════════════════════════════════════════
# 二、DrugMapping 数据结构
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DrugMapping:
    """药物名标准化结果"""
    original_name:  str
    mapped_name:    Optional[str]   # 标准名（来自 create_diabetes_drug_list）
    drug_category:  Optional[str]   # 药物类别
    confidence:     float           # 0.0–1.0
    method:         str             # exact / alias / alias_partial / fuzzy / llm / unmapped


# ═══════════════════════════════════════════════════════════════════════════════
# 三、DrugNormalizer 类
# ═══════════════════════════════════════════════════════════════════════════════

class DrugNormalizer:
    """
    药物名标准化器（三级策略 + 持久化缓存）

    策略优先级：
      1. 精确匹配 + 别名表（静态规则，O(1)）
      2. 模糊匹配（difflib, cutoff=0.82）
      3. LLM 辅助（可选，最后手段）

    核心修复（v2）：
      - filter_dm_relevant_drugs(): 新增 GT 过滤方法，
        只保留能映射到系统 category 体系内的药物。
        这确保 ground truth 范围 == LLM 推荐范围，F1 计算有意义。

    线程安全：_cache 和 _persistent_cache 使用 Lock 保护。
    """

    def __init__(
        self,
        use_llm:    bool = False,
        cache_path: str  = "data/local_drug_mapping.json",
    ):
        self.use_llm    = use_llm
        self.cache_path = Path(cache_path)

        # 构建标准药物字典（延迟加载）
        all_drugs = _load_drug_list()
        self.drug_to_category: Dict[str, str] = {d.lower(): c for d, c in all_drugs}
        self.standard_names:   List[str]      = [d for d, _ in all_drugs]
        self.standard_lower:   Set[str]       = set(self.drug_to_category.keys())

        # 别名字典（小写 key）
        self.alias_map: Dict[str, str] = {k.lower(): v for k, v in DRUG_ALIAS_MAP.items()}

        # 内存缓存（单次运行高频词）
        self._cache: Dict[str, DrugMapping] = {}
        self._lock  = Lock()
        self._llm   = None  # 延迟初始化

        # 持久化缓存（跨运行共享，atomic write）
        self._persistent_cache: Dict[str, Dict] = self._load_persistent_cache()

        logger.info(
            f"DrugNormalizer 初始化: 标准药物={len(self.standard_names)}, "
            f"持久化缓存={len(self._persistent_cache)}条, LLM={use_llm}"
        )

    # ── 主接口 ───────────────────────────────────────────────────────────────

    def normalize(self, name: str) -> DrugMapping:
        """标准化单个药物名（线程安全）。"""
        if not name or not name.strip():
            return DrugMapping(name, None, None, 0.0, "unmapped")

        name_raw = name.strip()

        # 0. 预处理：去剂量/剂型/eICU 前缀等噪声
        name_clean = self._preprocess(name_raw)
        if not name_clean:
            return DrugMapping(name_raw, None, None, 0.0, "unmapped")
        name_lower = name_clean.lower()

        # 0.5 排除已知非治疗性药物（如 Heparin Flush）
        #     别名表中映射到 None 的药物视为"已知但应排除"
        if name_lower in self.alias_map and self.alias_map[name_lower] is None:
            return DrugMapping(name_raw, None, None, 0.0, "excluded")

        # 也检查原始名（预处理前）是否应排除
        raw_lower = name_raw.lower().strip()
        if raw_lower in self.alias_map and self.alias_map[raw_lower] is None:
            return DrugMapping(name_raw, None, None, 0.0, "excluded")

        # 1. 内存缓存
        # [CACHEKEY-FIX] 键改用 name_raw：判定逻辑同时依赖 name_lower 与 raw_lower
        # (:660)，而 _preprocess 是有损投影（'Insulin (intravenous infusion)'、
        # 'Insulin Sliding Scale'、'Insulin' 全部塌缩为 'Insulin'）。用 clean 作键
        # 会让一个输入的失败结果覆盖另一个输入的正确结果。
        with self._lock:
            if name_raw in self._cache:
                return self._cache[name_raw]

        # 2. 持久化 JSON 缓存
        with self._lock:
            cached_data = self._persistent_cache.get(name_lower)
            if cached_data:
                result = DrugMapping(**cached_data)
                self._cache[name_raw] = result      # [CACHEKEY-FIX]
                return result

        # 3. 静态规则（精确 + 别名）
        result = self._exact(name_clean) or self._alias(name_clean)

        # 3.5 预处理前的原始名也尝试匹配（兜底：别名表可能覆盖原始写法）
        if not result and name_raw != name_clean:
            result = self._exact(name_raw) or self._alias(name_raw)

        # 3.6 盐型/酯型回退:仅在前面全部未命中时才尝试,避免污染正常匹配
        if not result:
            _SALT = (r'\b(calcium|sodium|potassium|magnesium|tartrate|succinate|maleate|'
                    r'fumarate|besylate|bisulfate|mesylate|phosphate|sulfate|oxalate|'
                    r'hydrochloride|hcl|bromide|citrate|acetate|cilexetil|medoxomil|'
                    r'trihydrate|monohydrate|micronized|prestab)\b')
            import re as _re
            _base = _re.sub(r'\s+', ' ', _re.sub(_SALT, '', name_clean, flags=_re.I)).strip()
            if _base and _base.lower() != name_lower and len(_base) >= 4:
                result = self._exact(_base) or self._alias(_base)
        
        # 4. 模糊匹配
        if not result:
            result = self._fuzzy(name_clean)

        # 5. LLM 辅助（可选）
        if not result and self.use_llm:
            result = self._llm_match(name_clean)

        # 6. 兜底
        # if not result:
        #     result = DrugMapping(name_raw, None, None, 0.0, "unmapped")
        # 6. 兜底
        if not result:
            # 6a. 泛称胰岛素兜底：MIMIC 中大量出院药仅记作 "Insulin"（未指明剂型）。
            #     这类是明确的降糖药，必须计入 n_dm_drugs / DR-07 / Path-DM；因剂型未知，
            #     归入 Insulins_Unspecified（不属于 basal/rapid 成员集，不会误满足/误违反
            #     剂型专属规则）。仅在前面所有策略均未命中、且 step 0.5 未判 excluded 时触发。
            #     排除明显的装置/非降糖用途（注射器、针头、笔、泵、冲管、补液、降钾用胰岛素等）。
            _INSULIN_NEG = (
                "flush", "pump", "syringe", "needle", "cartridge", "pen",
                "lock", "dwell", "infusion", "device", "hyperkal", "intraperiton",
            )
            if ("insulin" in name_lower or "insulin" in raw_lower) and not any(
                _neg in name_lower or _neg in raw_lower for _neg in _INSULIN_NEG
            ):
                result = DrugMapping(
                    name_raw, "Insulin (unspecified)", "Insulins_Unspecified",
                    0.6, "insulin_generic",
                )
            else:
                result = DrugMapping(name_raw, None, None, 0.0, "unmapped")

        # 缓存结果
        with self._lock:
            self._cache[name_raw] = result          # [CACHEKEY-FIX]
            # 持久化（仅缓存有价值的映射）
            if result.method in ("fuzzy", "llm"):
                self._persistent_cache[name_lower] = {
                    "original_name": result.original_name,
                    "mapped_name":   result.mapped_name,
                    "drug_category": result.drug_category,
                    "confidence":    result.confidence,
                    "method":        result.method,
                }
                self._save_persistent_cache()

        return result

    def normalize_list(self, names: List[str]) -> List[DrugMapping]:
        """批量标准化。"""
        return [self.normalize(n) for n in names if n]

    def filter_dm_relevant_drugs(self, drug_names: List[str]) -> List[str]:
        """
        【核心修复 v2】过滤出糖尿病管理相关药物。

        原理：
          只保留能被 DrugNormalizer 映射到系统 category 体系的药物名。
          这确保 ground truth 范围 == LLM 推荐范围，避免 ICU 非相关药物
          （止痛药、抗生素、镇静剂等）拉低 F1。

        副作用：
          对无法映射的品牌名，此过滤会将其丢弃。需要先扩充 DRUG_ALIAS_MAP
          覆盖 MIMIC 中的常见品牌名（已在本文件中完成）。

        Args:
            drug_names: 从 patient_data 提取的出院药物名列表（未过滤）

        Returns:
            仅包含系统 category 体系内药物的名称列表
        """
        result: List[str] = []
        for name in drug_names:
            if not name:
                continue
            mapping = self.normalize(name)
            if mapping.drug_category is not None:
                result.append(name)
        return result

    def get_dm_only_drugs(self, drug_names: List[str]) -> List[str]:
        """
        进一步过滤：只保留纯降糖药（不含并发症用药）。
        用于 paper 主表的 category_f1 计算（使用 EVAL_DM_ONLY_CATEGORIES）。
        """
        from agents.config import EVAL_DM_ONLY_CATEGORIES
        result: List[str] = []
        for name in drug_names:
            if not name:
                continue
            mapping = self.normalize(name)
            if mapping.drug_category in EVAL_DM_ONLY_CATEGORIES:
                result.append(name)
        return result

    # ── 内部策略 ─────────────────────────────────────────────────────────────

    @staticmethod
    def _preprocess(raw: str) -> str:
        """
        清洗原始药物名，去除剂量/剂型/路径/浓度等后缀，保留核心药名。

        主要处理：
          1. eICU 前缀冒号（如 ": INSULIN REGULAR HUMAN /ML IJ SOLN"）
          2. 星号标记（如 "*NF*", "*UNIT*", "NEO**"）
          3. 剂量数字+单位（如 "500mg", "100 UNITS", "5,000", "100u"）
          4. 预混比例（如 "70/30", "75-25", "50-50"）
          5. 剂型/路径关键词（如 "INJ SOLN", "/ML", "tab", "conc"）
          6. 品牌后缀/设备名（如 "Solostar", "KwikPen", "FlexPen"）
          7. 浓度标记（如 "U-100", "U-300"）
          8. 协议名（如 "sliding scale", "desensitization"）
        """
        import re
        s = raw.strip()
        if not s:
            return s

        # 1. eICU 前缀冒号 + 空格
        s = re.sub(r'^[:\s]+', '', s)

        # 2. 去除 BAG CUSTOM NDC 前缀（eICU）
        s = re.sub(r'\bBAG CUSTOM NDC\s*:?\s*', '', s, flags=re.IGNORECASE)

        # 3. 去 *NF* / *UNIT* / *XX* 等星号标记
        s = re.sub(r'\*\w+\*', '', s)
        # 去 NEO** 等垃圾前缀（字母+连续星号）
        s = re.sub(r'[A-Za-z]+\*{2,}', '', s)

        # 4. 去括号内容
        s = re.sub(r'\(.*?\)', '', s)

        # 5. 去剂量数字+单位
        s = re.sub(
            r'\b\d+[,.]?\d*\s*'
            r'(mg|mcg|g|ml|units?|%|meq|mmol|iu|/ml|/hr)\b',
            '', s, flags=re.IGNORECASE
        )
        # 独立的 \d+u（胰岛素单位缩写，如 "100u"）
        s = re.sub(r'\b\d+u\b', '', s, flags=re.IGNORECASE)
        # "5,000" 或 "100 UNITS in 1" 等
        s = re.sub(r'\b\d+[,]\d+\b', '', s)
        s = re.sub(r'\bin\s*\d+\b', '', s, flags=re.IGNORECASE)

        # 6. 去预混胰岛素比例 70/30 75-25 50-50 等
        s = re.sub(r'\b\d+[-/]\d+\b', '', s)

        # 7. 去剂型/路径/制剂描述
        s = re.sub(
            r'\b(tablet|tab|capsule|cap|injection|solution|suspension|pen|vial|'
            r'tabs|caps|po|chew|chewable|tbec|tbcr|subl|aers|aepb|crys|neb|susp'
            r'syringe|patch|cream|ointment|inhaler|syrup|elixir|'
            r'suppository|er|xr|xl|sr|cr|dr|hfa|la|nf|'
            r'oral|iv|sc|im|subq|subcutaneous|intravenous|'
            r'inj|soln|syr|sdv|amp|prefilled|ij|'
            r'conc|concentrated|recomb|recombinant|'
            r'desensitization|sliding\s+scale|'
            r'u-100|u-200|u-300|u-500)\b',
            '', s, flags=re.IGNORECASE
        )

        # 8. 去设备名（不要求前导词界，处理 70-30FlexPen 粘连）
        s = re.sub(r'(?i)flexpen|flextouch|solostar|kwikpen|penfill|cartridge', '', s)

        # 9. 去 eICU 特殊标记
        s = re.sub(r'/ml|/sdv|/syr', '', s, flags=re.IGNORECASE)

        # 10. 去连字符格式变体（INSULIN-LISPRO → Insulin lispro）
        # 但保留已知的连字符药名（如 alpha-lipoic acid）
        if s.lower().startswith('insulin-'):
            s = s.replace('-', ' ', 1)

        # 11. 清理残余空格和符号
        s = re.sub(r'[,;/]', ' ', s)
        s = re.sub(r'\s+', ' ', s).strip(' .-:')

        return s
    
    def _exact(self, name: str) -> Optional[DrugMapping]:
        lower = name.lower()
        if lower in self.standard_lower:
            return DrugMapping(name, name, self.drug_to_category[lower], 1.0, "exact")
        return None

    def _alias(self, name: str) -> Optional[DrugMapping]:
        lower = name.lower()
        # 排除标记：别名映射到 None 的药物是"已知但应排除"的
        if lower in self.alias_map and self.alias_map[lower] is None:
            return DrugMapping(name, None, None, 0.0, "excluded")
        # 完整别名匹配
        std_name = self.alias_map.get(lower)
        if std_name:
            cat = self.drug_to_category.get(std_name.lower())
            return DrugMapping(name, std_name, cat, 0.95, "alias")
        # 部分别名匹配（处理 "metformin 500mg" 这类带剂量字符串）
        # 修复（v5）：
        #   1. 先检查 exclusion 模式子串匹配（None 映射），防止 Heparin flush 等漏网
        #      即：若任何 exclusion alias 是当前名字的子串，则排除
        #   2. 正向匹配要求 alias 长度 >= 6，避免 "er"/"la"/"nph" 等短词误匹配
        for alias_lower, std in self.alias_map.items():
            if std is None and alias_lower in lower:
                return DrugMapping(name, None, None, 0.0, "excluded")
        for alias_lower, std in self.alias_map.items():
            if std is not None and len(alias_lower) >= 6 and alias_lower in lower:
                cat = self.drug_to_category.get(std.lower())
                return DrugMapping(name, std, cat, 0.85, "alias_partial")
        return None

    def _fuzzy(self, name: str) -> Optional[DrugMapping]:
        try:
            matches = get_close_matches(name.lower(), self.standard_lower, n=1, cutoff=0.82)
            if matches:
                std_lower = matches[0]
                std = next((s for s in self.standard_names if s.lower() == std_lower), std_lower)
                return DrugMapping(name, std, self.drug_to_category[std_lower], 0.75, "fuzzy")
        except Exception:
            pass
        return None

    def _llm_match(self, name: str) -> Optional[DrugMapping]:
        try:
            if self._llm is None:
                from agents.llm_client import DeepseekClient
                self._llm = DeepseekClient()
            from agents.config import ModelConfig
            sample = ", ".join(list(self.standard_lower)[:60])
            prompt = (
                f"将以下药物名映射到标准列表中最接近的名称。\n"
                f"输入: {name}\n"
                f"标准列表（部分）: {sample}\n"
                f"以 JSON 返回: {{\"match\": \"标准名或null\", \"confidence\": 0.0-1.0}}"
            )
            resp = self._llm.chat(
                messages=[{"role": "user", "content": prompt}],
                model=ModelConfig.QUICK_QUERY_MODEL,
                temperature=0.0, max_tokens=100, stream=False, deep_thinking=False
            )
            text = self._llm.get_response_text(resp)
            data = json.loads(text)
            m = data.get("match")
            if m and m.lower() in self.standard_lower:
                return DrugMapping(
                    name, m, self.drug_to_category[m.lower()],
                    float(data.get("confidence", 0.5)), "llm"
                )
        except Exception:
            pass
        return None

    # ── 持久化缓存工具 ───────────────────────────────────────────────────────

    def _load_persistent_cache(self) -> Dict[str, Dict]:
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"读取持久化药物缓存失败: {e}")
        return {}

    def _save_persistent_cache(self) -> None:
        """原子写入（多进程安全）。"""
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        suffix   = f"{os.getpid()}_{uuid.uuid4().hex[:6]}"
        tmp_path = self.cache_path.with_name(f"{self.cache_path.name}.tmp.{suffix}")
        try:
            latest = self._load_persistent_cache()
            merged = {**latest, **self._persistent_cache}
            self._persistent_cache = merged
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(merged, f, ensure_ascii=False, indent=2)
            tmp_path.replace(self.cache_path)
        except Exception as e:
            logger.error(f"保存持久化药物缓存失败: {e}")
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except Exception:
                    pass
