#!/usr/bin/env python3
"""Add the shifted anchor_year to MIMIC-IV records of a patient cache extracted without it.

  python data_prep/add_anchor_year.py --cache_dir data/patient_cache_v1 \
         --mimic4_dir ori_data_zip/mimic-iv-3.1 --out_dir data/patient_cache_v2

The cache is copied to --out_dir (the input is left unchanged); each MIMIC-IV record gets anchor_year from
hosp/patients.csv.gz, which the per-admission assigned year needs (agents.config.EraConfig.mimic4_admission_year).
"""
import argparse
import json
import shutil
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cache_dir", type=Path, required=True)
    ap.add_argument("--mimic4_dir", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    a = ap.parse_args()
    pat = pd.read_csv(a.mimic4_dir / "hosp" / "patients.csv.gz", usecols=["subject_id", "anchor_year"])
    anchor = dict(zip(pat["subject_id"].astype(str), pat["anchor_year"].astype(int)))
    n = missing = 0
    for f in sorted(a.cache_dir.rglob("*.json")):
        dst = a.out_dir / f.relative_to(a.cache_dir)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if "mimic4" not in f.relative_to(a.cache_dir).parts:
            shutil.copy2(f, dst)
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        sid = str(d.get("subject_id", "")).removesuffix(".0")
        if sid in anchor:
            d["anchor_year"] = anchor[sid]
        else:
            missing += 1
        dst.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
        n += 1
    print(f"MIMIC-IV records: {n}  without a patients.csv.gz match: {missing}  -> {a.out_dir}")


if __name__ == "__main__":
    main()
