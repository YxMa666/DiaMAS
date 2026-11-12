#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
probe_eicu_hospital_mapping_v3.py

Recover eICU hospitalid from the project's composite patient_id.

Observed current format:
    002-10050_211144
which is interpreted as:
    uniquepid = 002-10050
    patientunitstayid = 211144

This script performs an exact TWO-KEY join against raw eICU patient.csv.gz:
    (uniquepid, patientunitstayid)

No fuzzy matching is used.

Acceptance criteria:
- >=95% coverage
- no duplicate current patient_id
- no raw key mapping to multiple hospitals
- no current patient_id collision

Outputs:
outputs/eicu_hospital_probe/eicu_hospital_probe_v3.md
outputs/eicu_hospital_probe/eicu_hospital_map.csv   # if accepted
outputs/eicu_hospital_probe/unmatched_ids.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def norm_db(x):
    s = str(x).lower().replace("_", "").replace("-", "")
    return "eicu" if "eicu" in s else s


def clean_id(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if re.fullmatch(r"\d+\.0", s):
        s = s[:-2]
    return s


def split_current_id(x):
    """
    Expected:
      uniquepid_patientunitstayid
    e.g. 002-10050_211144 -> ("002-10050","211144")
    """
    s = clean_id(x)
    if "_" not in s:
        return "", ""
    left, right = s.rsplit("_", 1)
    return clean_id(left), clean_id(right)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--table",
        default="outputs/e2_qwen/test_table_v3.csv/e0_table.csv",
    )
    ap.add_argument(
        "--eicu_path",
        default="ori_data_zip/eicu-collaborative-research-database-2.0",
    )
    ap.add_argument(
        "--outdir",
        default="outputs/eicu_hospital_probe",
    )
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    table_path = Path(args.table)
    patient_path = Path(args.eicu_path) / "patient.csv.gz"

    if not table_path.exists():
        raise SystemExit(f"missing table: {table_path}")
    if not patient_path.exists():
        raise SystemExit(f"missing raw eICU patient table: {patient_path}")

    df = pd.read_csv(table_path, low_memory=False)
    dbcol = next(c for c in ["db_source", "database", "db", "source"] if c in df.columns)
    if "patient_id" not in df.columns:
        raise SystemExit("test table missing patient_id")

    e = df[df[dbcol].map(norm_db) == "eicu"][["patient_id"]].copy()
    e["patient_id"] = e["patient_id"].map(clean_id)
    e = e.drop_duplicates("patient_id").reset_index(drop=True)

    if e["patient_id"].duplicated().any():
        raise SystemExit("duplicated current patient_id")

    parsed = e["patient_id"].map(split_current_id)
    e["uniquepid_join"] = [x[0] for x in parsed]
    e["patientunitstayid_join"] = [x[1] for x in parsed]

    parse_ok = (
        e["uniquepid_join"].ne("")
        & e["patientunitstayid_join"].ne("")
    )
    print("eICU test IDs:", len(e))
    print("composite IDs parsed:", int(parse_ok.sum()), "/", len(e))
    print("sample parsed:")
    print(e.head(10).to_string(index=False))

    raw = pd.read_csv(
        patient_path,
        usecols=[
            "patientunitstayid",
            "patienthealthsystemstayid",
            "uniquepid",
            "hospitalid",
        ],
        low_memory=False,
    )

    for c in raw.columns:
        raw[c] = raw[c].map(clean_id)

    print("raw patient rows:", len(raw))
    print("raw hospitals:", raw["hospitalid"].nunique())

    # Check raw two-key uniqueness and hospital consistency.
    key_cols = ["uniquepid", "patientunitstayid"]
    key_hosp = (
        raw.groupby(key_cols, dropna=False)["hospitalid"]
        .nunique(dropna=True)
        .reset_index(name="n_hosp")
    )
    ambiguous = key_hosp[key_hosp["n_hosp"] > 1]
    print("raw ambiguous two-key rows:", len(ambiguous))
    if len(ambiguous):
        bad = set(map(tuple, ambiguous[key_cols].to_numpy()))
        raw["_bad"] = [
            (u, s) in bad
            for u, s in zip(raw["uniquepid"], raw["patientunitstayid"])
        ]
        raw = raw[~raw["_bad"]].drop(columns="_bad")

    raw = raw.drop_duplicates(key_cols)

    merged = e.merge(
        raw[
            [
                "uniquepid",
                "patientunitstayid",
                "patienthealthsystemstayid",
                "hospitalid",
            ]
        ],
        left_on=["uniquepid_join", "patientunitstayid_join"],
        right_on=["uniquepid", "patientunitstayid"],
        how="left",
        validate="one_to_one",
    )

    mapped = int(merged["hospitalid"].notna().sum())
    coverage = mapped / len(merged) if len(merged) else 0.0
    n_hospitals = int(merged["hospitalid"].dropna().nunique())

    print("=" * 96)
    print("TWO-KEY EXACT JOIN")
    print("=" * 96)
    print("mapped:", mapped, "/", len(merged))
    print("coverage:", f"{coverage:.1%}")
    print("hospitals represented:", n_hospitals)

    unmatched = merged[merged["hospitalid"].isna()][
        ["patient_id", "uniquepid_join", "patientunitstayid_join"]
    ].copy()
    unmatched.to_csv(
        outdir / "unmatched_ids.csv",
        index=False,
        encoding="utf-8-sig",
    )

    accepted = coverage >= 0.95 and len(ambiguous) == 0

    if accepted:
        hospital_map = merged.loc[
            merged["hospitalid"].notna(),
            [
                "patient_id",
                "hospitalid",
                "patienthealthsystemstayid",
                "uniquepid",
                "patientunitstayid",
            ],
        ].copy()

        if hospital_map["patient_id"].duplicated().any():
            raise SystemExit("unexpected duplicate patient_id after exact join")

        hospital_map.to_csv(
            outdir / "eicu_hospital_map.csv",
            index=False,
            encoding="utf-8-sig",
        )

    L = [
        "# eICU hospital mapping probe v3",
        "",
        f"- Current eICU test IDs: **{len(e):,}**",
        f"- Composite IDs parsed: **{int(parse_ok.sum()):,}/{len(e):,}**",
        f"- Raw patient rows: **{len(raw):,}**",
        f"- Raw hospitals: **{raw['hospitalid'].nunique():,}**",
        "",
        "## Exact mapping",
        "",
        "- Join rule: `current patient_id = uniquepid + '_' + patientunitstayid`",
        "- Join uses both raw keys simultaneously.",
        f"- Mapped: **{mapped:,}/{len(merged):,} ({coverage:.1%})**",
        f"- Hospitals represented in mapped test cohort: **{n_hospitals:,}**",
        f"- Raw ambiguous `(uniquepid, patientunitstayid)` keys: **{len(ambiguous):,}**",
        f"- Unmatched current IDs: **{len(unmatched):,}**",
        "",
        "## Decision",
        "",
    ]

    if accepted:
        L += [
            "- **ACCEPTED** for hospital-cluster analysis.",
            "- Criteria: ≥95% coverage and no ambiguous raw two-key mapping.",
            f"- Map: `{outdir / 'eicu_hospital_map.csv'}`",
        ]
    else:
        L += [
            "- **NOT ACCEPTED** for hospital-cluster analysis.",
            "- Inspect `unmatched_ids.csv`; do not use approximate/fuzzy matching.",
        ]

    (outdir / "eicu_hospital_probe_v3.md").write_text(
        "\n".join(L) + "\n",
        encoding="utf-8",
    )

    print("mapping success:", accepted)
    if accepted:
        print("WROTE:", outdir / "eicu_hospital_map.csv")
    print("WROTE:", outdir / "eicu_hospital_probe_v3.md")


if __name__ == "__main__":
    main()
