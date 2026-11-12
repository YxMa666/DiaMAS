#!/usr/bin/env python3
"""Study cohort and patient-level splits (run from the repository root).

  python data_prep/build_cohort.py --cache_dir data/patient_cache_v1 --outcomes_dir diabetes_subsets \
         --output_dir data/cohort

Inputs are the patient cache written by data_prep/run_extraction.py and the per-admission outcome
tables (<db>_outcomes.csv) written by data_prep/enhanced_outcome_extractor.py.

Exclusions, in order: no diabetes-related discharge medication; discharge against medical advice;
metastatic malignancy; hospital stay shorter than 48 hours; laboratory SOFA proxy >= 8; palliative or
hospice discharge.

Splits use the patient as the unit. Each patient is assigned the stratum (reference period x outcome
category x severity) of their most severe admission; within each stratum 20% of patients form the seed
pool and 80% the test cohort. The seed pool is divided 70/30 by patient into the seed cohort (ACD
fitting) and the development cohort (ACD model selection). MIMIC-III and MIMIC-IV admissions are written
to seed/, dev/ and test/; eICU admissions of the test cohort form the external cohort eicu/, and eICU
admissions of the seed pool are not used.

Output: <output_dir>/{seed,dev,test,eicu}/{<db>,all}/ (symlinks to the cache), cohort_manifest.csv,
cohort_summary.json
"""
import argparse
import json
import logging
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "data_prep")]
from agents.config import EVAL_FULL_DM_CATEGORIES, EraConfig  # noqa: E402
from enhanced_outcome_extractor import OutcomeLabelGenerator  # noqa: E402
from evaluation.drug_normalizer import DrugNormalizer  # noqa: E402

log = logging.getLogger("build_cohort")

DATABASES = ("mimic4", "eicu", "mimic3")  # scan order is part of the seeded split
SOFA_PROXY_MAX = 8
AMA = ("ama", "against medical advice", "left against", "elopement", "eloped")
PALLIATIVE = ("hospice", "comfort care", "palliative", "expired", "dead", "died", "dnr", "dni")
METASTATIC_ICD9, METASTATIC_ICD10 = ("196", "197", "198", "1990", "1991"), ("C77", "C78", "C79")
OUTCOME_RANK = {"Bad": 0, "Moderate": 1, "Good": 2}
SEVERITY_RANK = {"High": 0, "Medium": 1, "Low": 2}
ERA_RANK = {"Era-3": 0, "Era-2": 1, "Era-1": 2}


def fnum(v):
    try:
        return float(v) if v is not None and str(v) not in ("nan", "None", "") else None
    except (TypeError, ValueError):
        return None


def id_str(v):
    s = str(v).strip()
    return s[:-2] if s.endswith(".0") and s[:-2].isdigit() else s


def subject_of(pid):
    return pid.rsplit("_", 1)[0]


def discharge_destination(d):
    return str(d.get("discharge_destination", "") or (d.get("clinical_metrics") or {}).get("discharge_destination", "")
               or "").lower()


def metastatic(d):
    for dx in d.get("all_diagnoses") or []:
        if isinstance(dx, dict):
            code = str(dx.get("icd_code", "") or dx.get("icd9_code", "") or "").strip().upper().replace(".", "")
            if code and code != "NAN" and code.startswith(METASTATIC_ICD9 + METASTATIC_ICD10):
                return True
    return False


def medication_count(d, db, normalizer):
    """Discharge medications mapped to a diabetes or diabetes-complication class."""
    meds = d.get("discharge_medications", [])
    if not meds:
        meds = [m.get("drug_name", "") for m in d.get("all_medications") or []
                if isinstance(m, dict) and m.get("time_window") == "discharge" and m.get("drug_name")]
    if db == "mimic3" and not meds and isinstance(d.get("prior_medications"), list):
        meds = d["prior_medications"]
    return sum(1 for m in meds or [] if normalizer.normalize(str(m)).drug_category in EVAL_FULL_DM_CATEGORIES)


def load_outcomes(outcomes_dir):
    gen, index = OutcomeLabelGenerator(), {}
    for db in ("mimic3", "mimic4", "eicu"):
        path = next((p for p in (outcomes_dir / f"{db}_outcomes.csv", outcomes_dir / f"outcomes_{db}.csv") if p.exists()), None)
        if path is None:
            raise SystemExit(f"missing outcome table for {db} in {outcomes_dir}")
        for row in pd.read_csv(path, low_memory=False).to_dict("records"):
            sid, hid = row.get("subject_id"), row.get("hadm_id")
            if pd.isna(sid) or pd.isna(hid):
                continue
            label, tier = row.get("outcome_label"), row.get("severity_tier")
            index[f"{id_str(sid)}_{id_str(hid)}"] = {
                "outcome": str(label) if label and str(label) != "nan" else gen.generate(row),
                "severity": str(tier) if tier and str(tier) != "nan" else gen.get_severity_tier(row),
                "los": fnum(row.get("length_of_stay")), "sofa": fnum(row.get("lab_sofa_proxy")),
                "cci": fnum(row.get("charlson_cci")), "died": fnum(row.get("in_hospital_mortality")) == 1,
            }
    return index


def scan(cache_dir, outcomes, normalizer):
    records, excluded, short_stay = [], defaultdict(Counter), []
    for db in DATABASES:
        src = cache_dir / db / db if (cache_dir / db / db).exists() else cache_dir / db
        files = sorted(src.glob("*.json"))
        log.info("[%s] %d admissions in %s", db, len(files), src)
        for f in files:
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                excluded[db]["unreadable"] += 1
                continue
            n_meds = medication_count(d, db, normalizer)
            if n_meds == 0:
                excluded[db]["no_diabetes_related_medication"] += 1
                continue
            sid, hid = d.get("subject_id", ""), d.get("hadm_id", "")
            pid = f"{id_str(sid)}_{id_str(hid)}" if sid and hid else f.stem
            era, _ = EraConfig.classify_era(d)
            dest = discharge_destination(d)
            if any(k in dest for k in AMA):
                excluded[db]["against_medical_advice"] += 1
                continue
            if metastatic(d):
                excluded[db]["metastatic_malignancy"] += 1
                continue
            o = outcomes.get(pid, {})
            if o.get("los") is not None and o["los"] < 2.0:
                excluded[db]["stay_shorter_than_48h"] += 1
                short_stay.append(o)
                continue
            if o.get("sofa") is not None and o["sofa"] >= SOFA_PROXY_MAX:
                excluded[db]["lab_sofa_proxy_ge_8"] += 1
                continue
            if any(k in dest for k in PALLIATIVE):
                excluded[db]["palliative_or_hospice_discharge"] += 1
                continue
            records.append({"patient_id": pid, "subject_id": subject_of(pid), "db": db, "era": era,
                            "outcome": o.get("outcome", "Unknown"), "severity": o.get("severity", "Unknown"),
                            "n_medications": n_meds, "los_days": o.get("los"), "lab_sofa_proxy": o.get("sofa"),
                            "charlson_cci": o.get("cci"), "path": f})
    return records, excluded, short_stay


def split(records, seed, pool_ratio, fit_ratio):
    """Patient-level stratified seed-pool/test split, then a 70/30 seed/development split of the pool."""
    by_subject = defaultdict(list)
    for r in records:
        by_subject[r["subject_id"]].append(r)
    strata = defaultdict(list)
    for sid, recs in by_subject.items():
        w = min(recs, key=lambda r: (OUTCOME_RANK.get(r["outcome"], 3), SEVERITY_RANK.get(r["severity"], 3),
                                     ERA_RANK.get(r["era"], 3)))
        strata[(w["era"], w["outcome"], w["severity"])].append(sid)
    rng, pool = random.Random(seed), set()
    for sids in strata.values():
        sids = list(sids)
        rng.shuffle(sids)
        pool.update(sids[len(sids) - max(1, round(len(sids) * pool_ratio)):])
    pool_subjects = list(dict.fromkeys(r["subject_id"] for r in records if r["subject_id"] in pool))
    random.Random(seed + 1).shuffle(pool_subjects)
    fit = set(pool_subjects[:max(1, int(len(pool_subjects) * fit_ratio))])
    for r in records:
        eicu = r["db"] == "eicu"
        if r["subject_id"] not in pool:
            r["split"] = "eicu" if eicu else "test"
        else:
            r["split"] = "not_used" if eicu else ("seed" if r["subject_id"] in fit else "dev")


def link(src, dst, copy):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def write_split(name, recs, out, copy):
    seen = set()
    for r in recs:
        link(r["path"], out / name / r["db"] / r["path"].name, copy)
        flat = r["path"].name if r["path"].stem not in seen else f"{r['db']}__{r['path'].name}"
        seen.add(r["path"].stem)
        link(r["path"], out / name / "all" / flat, copy)


def describe(g):
    counts = lambda col: {str(k): int(v) for k, v in g[col].value_counts().items()}
    return {"admissions": len(g), "patients": int(g["subject_id"].nunique()),
            "by_database": counts("db"), "by_period": counts("era"), "by_outcome": counts("outcome")}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache_dir", type=Path, default=Path("data/patient_cache_v1"))
    ap.add_argument("--outcomes_dir", type=Path, default=Path("diabetes_subsets"))
    ap.add_argument("--output_dir", type=Path, default=Path("data/cohort"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pool_ratio", type=float, default=0.20, help="share of patients in the seed pool")
    ap.add_argument("--fit_ratio", type=float, default=0.70, help="share of seed-pool patients in the seed cohort")
    ap.add_argument("--copy", action="store_true", help="copy admission files instead of symlinking")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    records, excluded, short_stay = scan(a.cache_dir, load_outcomes(a.outcomes_dir), DrugNormalizer(use_llm=False))
    split(records, a.seed, a.pool_ratio, a.fit_ratio)
    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    for name in ("seed", "dev", "test", "eicu"):
        write_split(name, [r for r in records if r["split"] == name], out, a.copy)

    man = pd.DataFrame(records).assign(path=lambda x: x["path"].astype(str))
    man.to_csv(out / "cohort_manifest.csv", index=False)
    splits = {s: describe(man[man["split"] == s]) for s in ("seed", "dev", "test", "eicu", "not_used")}
    summary = {
        "config": {k: str(v) for k, v in vars(a).items()},
        "excluded": {db: dict(c) for db, c in excluded.items()},
        "included": describe(man),
        "splits": splits,
        "stay_shorter_than_48h": {"n": len(short_stay),
                                  "in_hospital_mortality": round(sum(o["died"] for o in short_stay) / max(len(short_stay), 1), 4)},
    }
    (out / "cohort_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for db, c in excluded.items():
        log.info("excluded [%s] %s", db, dict(c))
    for s, v in splits.items():
        log.info("%-9s %6d admissions  %6d patients  %s", s, v["admissions"], v["patients"], v["by_database"])


if __name__ == "__main__":
    main()
