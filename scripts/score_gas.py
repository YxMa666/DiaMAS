#!/usr/bin/env python3
"""Score recommendation records with the Guideline Alignment Score (GAS).

Inputs
  --recs      directory of recommendation records (patient_<id>.json from run_diamas.py or a direct LLM)
  --patients  directory of admission JSON files used to generate them
Output
  one CSV row per admission: patient_id, gas_f (the reported GAS), concordance, appropriateness,
  crit, major, temporal, dm_f1, comp_f1, n_cat, n_drugs, era, db_source

The admission year is resolved inside the evaluator (evaluation/safety_checker.resolve_admission_year),
so recommendations are judged by the rules applicable in the assigned year.

Run from the repository root:  python scripts/score_gas.py --recs ... --patients ... --out scores.csv
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_STATE = {}


def _init():
    from agents.config import EraConfig
    from evaluation.drug_normalizer import DrugNormalizer
    from evaluation.metrics_calculator import MetricsCalculator
    from evaluation.ose_calculator import OSECalculator
    from evaluation.safety_checker import HardConstraintChecker

    norm = DrugNormalizer(use_llm=False)
    _STATE.update(era=EraConfig, calc=MetricsCalculator(norm, HardConstraintChecker(norm)), ose=OSECalculator(norm))


def patient_file(patient_dir, pid):
    for name in (f"{pid}.json", f"patient_{pid}.json",
                 f"mimic4__{pid}.json", f"mimic3__{pid}.json", f"eicu__{pid}.json"):
        p = patient_dir / name
        if p.exists():
            return p
    return None


def score_one(rec_path, patient_dir):
    if not _STATE:
        _init()
    rec = json.loads(Path(rec_path).read_text(encoding="utf-8"))
    pid = str(rec.get("patient_id") or Path(rec_path).stem.removeprefix("patient_"))
    pf = patient_file(Path(patient_dir), pid)
    if pf is None:
        return {"patient_id": pid, "error": "patient file not found"}
    pdata = json.loads(pf.read_text(encoding="utf-8"))
    era, db = _STATE["era"].classify_era(pdata)
    pm = _STATE["calc"].calculate(rec_result=rec, patient_data=pdata, era=era, db_source=db)
    o = _STATE["ose"].calculate(patient_metrics=pm, rec_result=rec, patient_data=pdata, outcome_entry=None)
    return {
        "patient_id": pid, "gas_f": o.gas_f, "gas": o.gas_score,
        "concordance": o.gas_concordance, "appropriateness": o.gas_appropriateness,
        "crit": pm.n_violations_critical, "major": pm.n_violations_major,
        "temporal": o.temporal_violation_count,
        "dm_f1": pm.dm_only_category_f1, "comp_f1": pm.complication_only_category_f1,
        "n_cat": len(pm.rec_categories), "n_drugs": pm.n_recommendations,
        "era": era, "db_source": db,
    }


def _chunk(args):
    files, patient_dir = args
    return [score_one(f, patient_dir) for f in files]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--recs", required=True)
    ap.add_argument("--patients", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids", help="optional file with one patient_id per line to restrict scoring")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    files = sorted(Path(args.recs).glob("patient_*.json"))
    if args.ids:
        keep = {l.strip() for l in Path(args.ids).read_text().splitlines() if l.strip()}
        files = [f for f in files if f.stem.removeprefix("patient_") in keep]
    chunks = [(files[i:i + 500], args.patients) for i in range(0, len(files), 500)]
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init) as ex:
        rows = [r for part in ex.map(_chunk, chunks) for r in part]

    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    n_err = int(df["error"].notna().sum()) if "error" in df else 0
    print(f"scored {len(df) - n_err} / {len(df)} records -> {args.out}")


if __name__ == "__main__":
    main()
