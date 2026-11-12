#!/usr/bin/env python3
"""Build the paired Single/Collaborative table that defines collaboration value.

  d_gas = GAS(Collaborative) - GAS(Single)    for every admission scored under both programs

Pre-branch fields used by ACD are read from the Collaborative record; they are computed before the
programs diverge (clinical states, eligible specialists, assigned year), so they are identical in the
Single record.

Run from the repository root:
  python scripts/build_paired_table.py --single-scores S.csv --collab-scores C.csv \
      --collab-recs results/test_C/qwen3-32b_0.0 --out paired_test.csv [--ids cohort_ids.txt]
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def prebranch(meta):
    states = meta.get("phenotype_states") or []
    roles = list(meta.get("activated_specialties") or [])
    return {
        "era": meta.get("era_label"),
        "ada_year": meta.get("ada_year_used"),
        "mdt_activated": bool(meta.get("mdt_activated")),
        "n_specialties": len(roles),
        "specialties": str(roles),
        "n_states": len(states),
        "is_ascvd": int(any("ASCVD" in str(s.get("name", "")) for s in states)),
        "cardio_present": int(any("cardiology" in (s.get("specialty_tags") or []) for s in states)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--single-scores", required=True)
    ap.add_argument("--collab-scores", required=True)
    ap.add_argument("--collab-recs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids", help="optional cohort file with one patient_id per line")
    args = ap.parse_args()

    s = pd.read_csv(args.single_scores, dtype={"patient_id": str})
    c = pd.read_csv(args.collab_scores, dtype={"patient_id": str})
    df = s[["patient_id", "db_source", "gas_f"]].rename(columns={"gas_f": "gas_a1"}).merge(
        c[["patient_id", "gas_f"]].rename(columns={"gas_f": "gas_a2"}), on="patient_id", validate="one_to_one")
    if args.ids:
        keep = {l.strip() for l in Path(args.ids).read_text().splitlines() if l.strip()}
        df = df[df["patient_id"].isin(keep)]
    df = df.dropna(subset=["gas_a1", "gas_a2"])

    rows = []
    for pid in df["patient_id"]:
        rec = json.loads((Path(args.collab_recs) / f"patient_{pid}.json").read_text(encoding="utf-8"))
        rows.append({"patient_id": pid, **prebranch(rec.get("diamas_metadata") or {})})
    df = df.merge(pd.DataFrame(rows), on="patient_id")
    df.insert(1, "split_key", df["patient_id"].str.rsplit("_", n=1).str[0])
    df["d_gas"] = df["gas_a2"] - df["gas_a1"]
    cols = ["patient_id", "split_key", "db_source", "era", "ada_year", "mdt_activated", "n_specialties",
            "specialties", "n_states", "is_ascvd", "cardio_present", "gas_a1", "gas_a2", "d_gas"]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df[cols].to_csv(args.out, index=False)
    print(f"{len(df)} paired admissions -> {args.out}")


if __name__ == "__main__":
    main()
