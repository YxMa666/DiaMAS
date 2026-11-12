#!/usr/bin/env python3
"""Offline smoke test on a synthetic admission (no LLM, no guideline index, no PhysioNet data).

Checks that a cohort is built and split from a synthetic patient cache, that the Single program runs
end to end with a stubbed model response, that the record is scored by GAS, that the paired table and
ACD features are built, and that an ACD estimator can be fitted. Run from the repository root:
  python tests/smoke_test.py
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

STUB = {"choices": [{"message": {"content": json.dumps({
    "recommended_drugs": [{"drug": "Insulin glargine", "category": "Insulin_Basal"},
                          {"drug": "Empagliflozin", "category": "SGLT2i"}],
    "complication_drugs": [{"drug": "Atorvastatin", "category": "Lipid_Statins"},
                           {"drug": "Lisinopril", "category": "Hypertension_ACEi"}],
    "excluded_classes": [{"class": "Biguanides", "reason": "eGFR 38 with heart failure; dose limits"}],
    "reasoning": "stub"})}}], "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}


def main():
    import run_diamas
    from scripts.score_gas import score_one

    work = Path(tempfile.mkdtemp(prefix="diamas_smoke_"))
    pdir, rdir = work / "patients", work / "recs"
    pdir.mkdir(); rdir.mkdir()
    patient = json.loads((ROOT / "examples/synthetic_admission.json").read_text())
    pid = f"{patient['subject_id']}_{patient['hadm_id']}"
    (pdir / f"{pid}.json").write_text(json.dumps(patient))

    # 00. Assigned year: one rule for era label, prompt year and GAS (MIMIC-IV documentation example)
    from agents.config import EraConfig
    from diamas_utils.era_retriever import infer_era_year
    from evaluation.safety_checker import resolve_admission_year
    for admit, year, era in (("2153", 2010, "Era-1"), ("2159", 2016, "Era-2"), ("2170", 2022, "Era-3"), ("2150", 2008, "Era-1")):
        rec = {"database": "mimic4", "anchor_year_group": "2008 - 2010", "anchor_year": 2153, "admission_time": f"{admit}-06-01 08:00:00"}
        assert resolve_admission_year(rec) == year and EraConfig.classify_era(rec)[0] == era, (admit, resolve_admission_year(rec))
        assert infer_era_year(rec, {})[0] == era
    print("[00] MIMIC-IV assigned year: group end + (admission year - anchor_year), clipped to 2008-2022")
    from evaluation.drug_normalizer import DrugNormalizer
    from evaluation.safety_checker import HardConstraintChecker, _GAS_PATIENT_GATES
    chk = HardConstraintChecker(DrugNormalizer(use_llm=False))
    hits = lambda drugs, **kw: {v.constraint_id for v in chk.check([{"drug": d} for d in drugs], patient["clinical_metrics"],  # noqa: E731
                                                                   {**patient, **kw})}
    assert {"HC-11", "HC-13"} <= hits(["tirzepatide", "semaglutide"], anchor_year_group="2014 - 2016", anchor_year=2160, admission_time="2160-01-01")
    assert "HC-20" in hits(["gemfibrozil"]) and "HC-20" not in hits(["gemfibrozil"], anchor_year_group="2014 - 2016")
    assert "HC-31" in hits(["empagliflozin"], all_diagnoses=patient["all_diagnoses"] + [{"icd_code": "R570"}])
    assert not any(g["drug_match"]("amlodipine", "Hypertension_CCB") for g in _GAS_PATIENT_GATES)
    print("[00] hard constraints: year-gated and haemodynamic constraints fire on the assigned year")

    # 0. Cohort construction and patient-level splits on a synthetic cache
    rows = {db: [] for db in ("mimic3", "mimic4", "eicu")}
    for i in range(60):
        db = ("mimic4", "mimic3", "eicu")[i % 3]
        p = dict(patient, subject_id=f"{i // 2:03d}-{i}" if db == "eicu" else 5000 + i // 2, hadm_id=7000 + i, database=db)
        d = work / "cohort_cache" / db / db
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{p['subject_id']}_{p['hadm_id']}.json").write_text(json.dumps(p))
        rows[db].append({"subject_id": p["subject_id"], "hadm_id": p["hadm_id"], "outcome_label": ("Good", "Bad")[i % 2],
                         "severity_tier": "Medium", "length_of_stay": 1.0 if i % 10 == 0 else 5.0, "lab_sofa_proxy": 2})
    (work / "outcomes").mkdir()
    for db, r in rows.items():
        pd.DataFrame(r).to_csv(work / "outcomes" / f"{db}_outcomes.csv", index=False)
    subprocess.run([sys.executable, str(ROOT / "data_prep/build_cohort.py"), "--cache_dir", str(work / "cohort_cache"),
                    "--outcomes_dir", str(work / "outcomes"), "--output_dir", str(work / "cohort")],
                   check=True, cwd=ROOT, capture_output=True)
    summ = json.loads((work / "cohort" / "cohort_summary.json").read_text())
    print(f"[0] cohort: {summ['included']['admissions']} admissions; "
          f"{ {k: v['admissions'] for k, v in summ['splits'].items()} }; excluded {summ['excluded']}")

    # 1. Single program end to end (stubbed LLM)
    eng = run_diamas.DiamasRecommendationEngine(model_name="qwen3-32b", temperature=0.0, max_tokens=8000,
                                                use_mdt=False, guidelines_dir=work / "none",
                                                index_dir=work / "none", guideline_sources=["ada"])
    prompts = []
    eng.llm_client.chat = lambda messages, **kw: prompts.append(messages) or STUB
    rec = eng.recommend(patient)
    assert rec["success"], rec.get("error_message")
    meta = rec["diamas_metadata"]
    user = prompts[0][1]["content"]
    assert "PubMed RCT Evidence: (none retrieved)" in user
    print(f"[1] Single: era={meta['era_label']} ada_year={meta['ada_year_used']} "
          f"states={[s['name'] for s in meta['phenotype_states']]}")
    print(f"    eligible specialists={meta['activated_specialties']} final={[d['drug'] for d in rec['recommended_drugs']]}")
    (rdir / f"patient_{pid}.json").write_text(json.dumps(rec, ensure_ascii=False, default=str))

    # 2. GAS
    row = score_one(rdir / f"patient_{pid}.json", pdir)
    assert "error" not in row, row
    print(f"[2] GAS: gas_f={row['gas_f']:.4f} concordance={row['concordance']:.4f} "
          f"appropriateness={row['appropriateness']:.4f} temporal={row['temporal']}")

    # 3. Paired table (the same record stands in for both programs here)
    pd.DataFrame([row]).to_csv(work / "s.csv", index=False)
    subprocess.run([sys.executable, str(ROOT / "scripts/build_paired_table.py"), "--single-scores", str(work / "s.csv"),
                    "--collab-scores", str(work / "s.csv"), "--collab-recs", str(rdir), "--out", str(work / "paired.csv")],
                   check=True, cwd=ROOT)
    paired = pd.read_csv(work / "paired.csv")
    print(f"[3] paired table: {paired.iloc[0][['n_specialties', 'n_states', 'is_ascvd', 'cardio_present', 'd_gas']].to_dict()}")

    # 4. ACD features from the patient cache layout and an estimator fit on simulated targets
    from acd import features as F
    cache = work / "cache" / "mimic4" / "mimic4"
    cache.mkdir(parents=True)
    shutil.copy(pdir / f"{pid}.json", cache / f"{pid}.json")
    df = F.extract_target_features(paired, work / "cache")
    X = pd.concat([F.prepare(df)] * 200, ignore_index=True)
    y = np.random.default_rng(0).normal(0.02, 0.1, len(X))
    X["egfr"] = X["egfr"] + np.random.default_rng(1).normal(0, 5, len(X))
    model = F.build("gbdt", F.GBDT_GRID[-1]).fit(X, y)
    print(f"[4] ACD: features={list(F.prepare(df).columns)}; fitted {type(model[-1]).__name__}")

    shutil.rmtree(work)
    print("SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
