#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ACD features, preprocessing and estimator families.

The 13 pre-branch features (Methods 3.3):
  numeric     n_specialties, n_states, is_ascvd, cardio_present, age, egfr, heart_failure, aki_suspected
  categorical era, specialty_combo, sex, dm_type, ckd_g_stage
Candidate estimators: Ridge (alpha 0.1, 1, 10, 100) and 16 gradient-boosted tree configurations.

Functions are copied unchanged from the implementation used for the published estimator.
Paths below are defaults; acd/run_acd.py overrides them from the command line.
"""

from __future__ import annotations
import ast, json, re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = Path("outputs/deliberation_value_v2/feature_audit_v2/rich_candidates_seed_v2.csv")

DEV  = Path("outputs/deliberation_value_v2/feature_audit_v2/rich_candidates_dev_v2.csv")

TEST = Path("outputs/e2_qwen/test_table_v3.csv/e0_table.csv")

CACHE = Path("data/patient_cache_v1")

HOSPITAL_MAP = Path("outputs/eicu_hospital_probe/eicu_hospital_map.csv")

TARGET = "d_gas"

RANDOM_STATE = 20260915

RICH_NUM = [
    "n_specialties","n_states","is_ascvd","cardio_present",
    "age","egfr","heart_failure","aki_suspected"
]

RICH_CAT = ["era","specialty_combo","sex","dm_type","ckd_g_stage"]

RIDGE_GRID=[{"alpha":a} for a in (0.1,1.0,10.0,100.0)]

GBDT_GRID=[
    dict(n_estimators=n,learning_rate=lr,max_depth=d,min_samples_leaf=leaf,subsample=1.0)
    for n in (100,200)
    for lr in (.03,.05)
    for d in (2,3)
    for leaf in (30,100)
]

def norm_db(x):
    s=str(x).strip().lower()
    if any(k in s for k in ("mimic4","mimiciv","mimic-iv")): return "mimic4"
    if any(k in s for k in ("mimic3","mimiciii","mimic-iii")): return "mimic3"
    if "eicu" in s: return "eicu"
    return s

def parse_specialties(v):
    if v is None or (isinstance(v,float) and np.isnan(v)): return []
    if isinstance(v,(list,tuple,set)): items=[str(x) for x in v]
    else:
        s=str(v).strip()
        if not s: return []
        try:
            o=ast.literal_eval(s)
            items=[str(x) for x in o] if isinstance(o,(list,tuple,set)) else [s]
        except Exception:
            items=re.split(r"[,|;+]+",s)
    aliases={
        "endocrinology":"endo","endocrine":"endo",
        "cardiology":"cardio","cardiometabolic":"cardio","cardio_metabolic":"cardio",
        "nephrology":"neph","renal":"neph",
        "geriatrics":"geriatric","geriatric":"geriatric",
        "pharmacy":"pharmacist","clinical_pharmacist":"pharmacist",
    }
    z=[]
    for x in items:
        x=re.sub(r"\s+","_",str(x).strip().lower())
        if x: z.append(aliases.get(x,x))
    return sorted(set(z))

def ensure_combo(df):
    z=df.copy()
    if "specialty_combo" not in z:
        z["specialty_combo"]=z["specialties"].map(
            lambda x:"+".join(parse_specialties(x)) if parse_specialties(x) else "none")
    return z

def to_binary(s):
    if pd.api.types.is_numeric_dtype(s): return pd.to_numeric(s,errors="coerce")
    m={"true":1.,"yes":1.,"y":1.,"1":1.,"false":0.,"no":0.,"n":0.,"0":0.}
    return s.astype(str).str.strip().str.lower().map(m)

def prepare(df):
    z=ensure_combo(df)
    X=z[RICH_NUM+RICH_CAT].copy()
    for c in RICH_NUM:
        X[c]=to_binary(X[c]) if c in {"is_ascvd","cardio_present","heart_failure","aki_suspected"} else pd.to_numeric(X[c],errors="coerce")
    for c in RICH_CAT:
        s=X[c].astype("object")
        X[c]=s.map(lambda v:np.nan if pd.isna(v) else str(v))
    return X

def prep():
    return ColumnTransformer([
        ("num",Pipeline([
            ("impute",SimpleImputer(strategy="median",add_indicator=True)),
            ("scale",StandardScaler()),
        ]),RICH_NUM),
        ("cat",Pipeline([
            ("impute",SimpleImputer(strategy="most_frequent")),
            ("onehot",OneHotEncoder(handle_unknown="ignore",sparse_output=False)),
        ]),RICH_CAT),
    ],remainder="drop",sparse_threshold=0.0)

def build(family,params):
    reg=Ridge(**params) if family=="ridge" else GradientBoostingRegressor(
        random_state=RANDOM_STATE,loss="squared_error",**params)
    return Pipeline([("prep",prep()),("reg",reg)])

def getp(d,path):
    cur=d
    for k in path.split("."):
        if not isinstance(cur,dict) or k not in cur:return None
        cur=cur[k]
    return cur

def fnum(x):
    try:
        v=float(x);return v if np.isfinite(v) else np.nan
    except:return np.nan

def fbool(x):
    if x is None:return np.nan
    if isinstance(x,bool):return int(x)
    if isinstance(x,(int,float,np.integer,np.floating)) and float(x) in (0.,1.):return int(x)
    s=str(x).strip().lower()
    if s in {"true","yes","y","1","present","positive"}:return 1
    if s in {"false","no","n","0","absent","negative"}:return 0
    return np.nan

def fcat(x):
    if x is None:return pd.NA
    s=str(x).strip()
    return pd.NA if not s or s.lower() in {"none","nan","null","unknown","na","n/a"} else s

def recursive_hospital_id(obj):
    aliases={"hospitalid","hospital_id","hospital_id_eicu"}
    hits=[]
    def walk(x,path=""):
        if isinstance(x,dict):
            for k,v in x.items():
                nk=re.sub(r"[^a-z0-9]+","_",str(k).lower()).strip("_")
                p=f"{path}.{k}" if path else str(k)
                if nk in aliases and not isinstance(v,(dict,list)) and v is not None:
                    hits.append((p,str(v)))
                walk(v,p)
        elif isinstance(x,list) and len(x)<=20:
            for i,v in enumerate(x):
                if isinstance(v,dict):walk(v,f"{path}[{i}]")
    walk(obj)
    return hits[0][1] if hits else None

def _norm_col(s):
    return re.sub(r"[^a-z0-9]+","_",str(s).strip().lower()).strip("_")

def attach_audited_hospital_map(df, path=HOSPITAL_MAP):
    """Attach the previously audited eICU hospital map."""
    if not path.exists():
        raise FileNotFoundError(path)

    hm = pd.read_csv(path, low_memory=False)
    norm = {_norm_col(c): c for c in hm.columns}

    hosp_col = next((norm[x] for x in ("hospitalid","hospital_id","hospital") if x in norm), None)
    if hosp_col is None:
        raise KeyError(f"No hospital column in {path}; columns={list(hm.columns)}")

    base = df.copy()
    base["patient_id"] = base["patient_id"].astype(str)
    base["split_key"] = base["split_key"].astype(str)

    if "patient_id" in norm:
        left_key, right_key = "patient_id", norm["patient_id"]
        left = base["patient_id"].astype(str)
        right = hm[right_key].astype(str)
    elif "split_key" in norm:
        left_key, right_key = "split_key", norm["split_key"]
        left = base["split_key"].astype(str)
        right = hm[right_key].astype(str)
    elif "uniquepid" in norm or "unique_pid" in norm:
        left_key = "split_key"
        right_key = norm.get("uniquepid", norm.get("unique_pid"))
        left = base["split_key"].astype(str)
        right = hm[right_key].astype(str)
    elif "patientunitstayid" in norm or "stay_id" in norm:
        left_key = "patient_id suffix"
        right_key = norm.get("patientunitstayid", norm.get("stay_id"))
        left = base["patient_id"].str.rsplit("_", n=1).str[-1].astype(str)
        right = hm[right_key].astype(str).str.replace(r"\.0$", "", regex=True)
    else:
        raise KeyError(f"No usable patient key in {path}; columns={list(hm.columns)}")

    mp = pd.DataFrame({"__join__": right, "hospital_id_map": hm[hosp_col].astype(str)})
    conflicts = mp.groupby("__join__")["hospital_id_map"].nunique()
    conflicts = conflicts[conflicts > 1]
    if len(conflicts):
        raise RuntimeError(f"{len(conflicts)} map keys link to >1 hospital")

    mp = mp.drop_duplicates("__join__")
    base["__join__"] = left
    base = base.merge(mp, on="__join__", how="left", validate="many_to_one")
    base["hospital_id"] = base["hospital_id_map"]
    base = base.drop(columns=["__join__","hospital_id_map"])

    cov = base["hospital_id"].notna().mean()
    nh = base["hospital_id"].dropna().astype(str).nunique()
    print(f"Audited hospital map: {path}")
    print(f"join: {left_key} -> {right_key}")
    print(f"coverage={cov:.1%}, hospitals={nh}")
    return base

def extract_target_features(base,cache_root):
    rows=[];missing=[]
    for _,r in base.iterrows():
        pid=str(r.patient_id);db=norm_db(r.db_source)
        p=cache_root/db/db/f"{pid}.json"
        if not p.exists():
            missing.append(str(p));continue
        obj=json.load(p.open())
        cm=obj.get("clinical_metrics",{}) or {}
        kid=cm.get("kidney",{}) or {}
        dmc=cm.get("dm_complications",{}) or {}
        dmcfg=cm.get("dm_classification",{}) or {}
        age=obj.get("age",getp(obj,"demographics.age"))
        sex=obj.get("gender",obj.get("sex",getp(obj,"demographics.gender")))
        z=r.to_dict()
        z.update(
            age=fnum(age),sex=fcat(sex),egfr=fnum(kid.get("egfr")),
            heart_failure=fbool(dmc.get("heart_failure")),
            aki_suspected=fbool(kid.get("aki_suspected")),
            ckd_g_stage=fcat(kid.get("ckd_g_stage")),
            dm_type=fcat(dmcfg.get("dm_type")),
            hospital_id=recursive_hospital_id(obj) if db=="eicu" else None
        )
        rows.append(z)
    if missing: raise RuntimeError(f"Missing cache rows={len(missing)} examples={missing[:5]}")
    return pd.DataFrame(rows)
