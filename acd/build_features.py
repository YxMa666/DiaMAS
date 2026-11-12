#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build ACD feature tables for the seed and development cohorts.
Reads the paired tables (scripts/build_paired_table.py) and the patient cache.
Does NOT access the test cohort and does NOT call any LLM.

Key correction vs v1:
- AKI is taken from clinical_metrics.kidney.aki_suspected
  (NOT data_quality.aki_suspected)
- CKD stage is taken from clinical_metrics.kidney.ckd_g_stage /
  ckd_a_stage / ckd_full_stage
"""

from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd

CACHE = Path("data/patient_cache_v1")

def norm_db(x):
    s = str(x).strip().lower()
    if any(k in s for k in ("mimic4","mimiciv","mimic-iv")): return "mimic4"
    if any(k in s for k in ("mimic3","mimiciii","mimic-iii")): return "mimic3"
    if "eicu" in s: return "eicu"
    return s

def getp(d, path):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur

def first_nonnull(*xs):
    for x in xs:
        if x is not None:
            if isinstance(x, str) and x.strip().lower() in {"", "none", "nan", "null", "unknown", "n/a", "na"}:
                continue
            return x
    return None

def fnum(x):
    if x is None: return np.nan
    try:
        v = float(x)
        return v if np.isfinite(v) else np.nan
    except Exception:
        return np.nan

def fbool(x):
    if x is None: return np.nan
    if isinstance(x, bool): return int(x)
    if isinstance(x, (int,float,np.integer,np.floating)) and float(x) in (0.0,1.0):
        return int(x)
    s = str(x).strip().lower()
    if s in {"true","yes","y","1","present","positive"}: return 1
    if s in {"false","no","n","0","absent","negative"}: return 0
    return np.nan

def fcat(x):
    if x is None: return pd.NA
    s = str(x).strip()
    if not s or s.lower() in {"none","nan","null","unknown","n/a","na"}:
        return pd.NA
    return s

def extract(obj):
    cm = obj.get("clinical_metrics", {}) or {}
    kid = cm.get("kidney", {}) or {}
    dmc = cm.get("dm_complications", {}) or {}
    dmcfg = cm.get("dm_classification", {}) or {}

    # Canonical, explicit, pre-branch paths only.
    out = {
        "age": fnum(first_nonnull(obj.get("age"), getp(obj, "demographics.age"))),
        "sex": fcat(first_nonnull(obj.get("gender"), obj.get("sex"), getp(obj, "demographics.gender"))),

        "bmi": fnum(first_nonnull(obj.get("bmi"), cm.get("bmi"))),
        "hba1c": fnum(cm.get("hba1c")),

        "egfr": fnum(kid.get("egfr")),
        "uacr": fnum(kid.get("uacr")),

        # Corrected renal-state fields.
        "aki_suspected": fbool(kid.get("aki_suspected")),
        "aki_kdigo_stage": fcat(kid.get("aki_kdigo_stage")),
        "ckd_g_stage": fcat(kid.get("ckd_g_stage")),
        "ckd_a_stage": fcat(kid.get("ckd_a_stage")),
        "ckd_full_stage": fcat(kid.get("ckd_full_stage")),

        "dm_type": fcat(dmcfg.get("dm_type")),
        "heart_failure": fbool(dmc.get("heart_failure")),
    }
    return out

def load_split(path, split):
    z = pd.read_csv(path, low_memory=False)
    z["split"] = split
    z["db_source"] = z["db_source"].map(norm_db)
    return z

def process(df):
    rows=[]
    miss=[]
    for _, r in df.iterrows():
        pid=str(r["patient_id"]); db=str(r["db_source"])
        p=CACHE/db/db/f"{pid}.json"
        base={
            "patient_id": pid,
            "split": r["split"],
            "db_source": db,
            "split_key": r.get("split_key", pd.NA),
            "d_gas": r["d_gas"],
        }
        for c in ["era","ada_year","n_specialties","specialties","n_states","is_ascvd","cardio_present"]:
            if c in df.columns: base[c]=r.get(c,pd.NA)

        if not p.exists():
            miss.append((pid,db,str(p)))
            base.update({k:np.nan for k in [
                "age","sex","bmi","hba1c","egfr","uacr","aki_suspected","aki_kdigo_stage",
                "ckd_g_stage","ckd_a_stage","ckd_full_stage","dm_type","heart_failure"
            ]})
        else:
            with open(p,"r",encoding="utf-8") as f:
                obj=json.load(f)
            base.update(extract(obj))
        rows.append(base)
    return pd.DataFrame(rows), pd.DataFrame(miss,columns=["patient_id","db_source","path"])

def coverage(df):
    feats=[
        "age","sex","bmi","hba1c","egfr","uacr",
        "aki_suspected","aki_kdigo_stage",
        "ckd_g_stage","ckd_a_stage","ckd_full_stage",
        "dm_type","heart_failure"
    ]
    rec=[]
    for (split,db),g in df.groupby(["split","db_source"]):
        for f in feats:
            rec.append({
                "split":split,"db_source":db,"feature":f,
                "N":len(g),"nonmissing":int(g[f].notna().sum()),
                "coverage":float(g[f].notna().mean())
            })
    for split,g in df.groupby("split"):
        for f in feats:
            rec.append({
                "split":split,"db_source":"ALL","feature":f,
                "N":len(g),"nonmissing":int(g[f].notna().sum()),
                "coverage":float(g[f].notna().mean())
            })
    return pd.DataFrame(rec)

def main():
    ap=argparse.ArgumentParser()
    global CACHE
    ap.add_argument("--seed",required=True,help="paired seed table")
    ap.add_argument("--dev",required=True,help="paired development table")
    ap.add_argument("--cache",default=str(CACHE))
    ap.add_argument("--outdir",required=True)
    args=ap.parse_args()
    CACHE=Path(args.cache)

    out=Path(args.outdir)
    out.mkdir(parents=True,exist_ok=True)

    seed=load_split(args.seed,"seed")
    dev=load_split(args.dev,"dev")

    sf,sm=process(seed)
    df,dm=process(dev)
    both=pd.concat([sf,df],ignore_index=True)
    cov=coverage(both)

    sf.to_csv(out/"rich_candidates_seed_v2.csv",index=False)
    df.to_csv(out/"rich_candidates_dev_v2.csv",index=False)
    cov.to_csv(out/"feature_coverage_v2.csv",index=False)
    pd.concat([sm,dm],ignore_index=True).to_csv(out/"cache_missing_v2.csv",index=False)

    print("="*100)
    print("CANONICAL RICH FEATURE RE-AUDIT — SEED/DEV ONLY")
    print("="*100)
    print(f"seed N={len(sf):,}  dev N={len(df):,}")
    print(f"cache missing={len(sm)+len(dm):,}")
    print("\nOVERALL COVERAGE (%)")
    p=(cov[cov.db_source=="ALL"]
       .pivot(index="feature",columns="split",values="coverage")*100)
    print(p.round(1).to_string())

    print("\nMINIMUM DB-SPECIFIC COVERAGE (%)")
    q=(cov[cov.db_source!="ALL"]
       .groupby(["feature","split"])["coverage"].min()
       .unstack("split")*100)
    print(q.round(1).to_string())

    print("\nCANONICAL PATHS")
    print("AKI        = clinical_metrics.kidney.aki_suspected")
    print("CKD G      = clinical_metrics.kidney.ckd_g_stage")
    print("CKD A      = clinical_metrics.kidney.ckd_a_stage")
    print("CKD full   = clinical_metrics.kidney.ckd_full_stage")
    print("eGFR       = clinical_metrics.kidney.egfr")
    print("HF         = clinical_metrics.dm_complications.heart_failure")
    print("DM type    = clinical_metrics.dm_classification.dm_type")
    print("\nHELD-OUT TEST ACCESSED: NO")
    print("LLM CALLS: NO")
    print(f"\nArtifacts -> {out}")

if __name__=="__main__":
    main()
