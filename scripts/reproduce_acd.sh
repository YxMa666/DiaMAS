#!/usr/bin/env bash
# From program outputs to ACD results.
#   scripts/reproduce_acd.sh <runs_root> <cohort_dir> <cache_dir> <eicu_dir> <work_dir>
# <runs_root>/<split>/{single,collaborative}/qwen3-32b_0.0 and <cohort_dir>/<split>/all for split in seed, dev, test, eicu
# (data_prep/build_cohort.py, scripts/run_programs.sh); <eicu_dir> is the eICU-CRD 2.0 directory.
set -euo pipefail
RUNS=$1; COHORT=$2; CACHE=$3; EICU=$4; W=$5
M=qwen3-32b_0.0
mkdir -p "$W"
for s in seed dev test eicu; do
  python scripts/score_gas.py --recs "$RUNS/$s/single/$M"        --patients "$COHORT/$s/all" --out "$W/${s}_single_gas.csv"
  python scripts/score_gas.py --recs "$RUNS/$s/collaborative/$M" --patients "$COHORT/$s/all" --out "$W/${s}_collab_gas.csv"
  python scripts/build_paired_table.py --single-scores "$W/${s}_single_gas.csv" --collab-scores "$W/${s}_collab_gas.csv" \
         --collab-recs "$RUNS/$s/collaborative/$M" --out "$W/paired_$s.csv"
done
# eICU admission -> hospital (exact join on uniquepid and patientunitstayid) for the hospital-cluster bootstrap
python data_prep/eicu_hospital_map.py --table "$W/paired_eicu.csv" --eicu_path "$EICU" --outdir "$W/hospital_map"
python acd/build_features.py --seed "$W/paired_seed.csv" --dev "$W/paired_dev.csv" --cache "$CACHE" --outdir "$W/features"
ACD="python -m acd.run_acd --seed-table $W/features/rich_candidates_seed_v2.csv --dev-table $W/features/rich_candidates_dev_v2.csv
     --cache $CACHE --hospital-map $W/hospital_map/eicu_hospital_map.csv --out $W/acd"
$ACD --test-table "$W/paired_test.csv" fit
$ACD --test-table "$W/paired_test.csv" eval-primary
$ACD --test-table "$W/paired_eicu.csv" eval-eicu
python -m acd.shap_analysis --model-dir "$W/acd" --test-table "$W/paired_test.csv" --cache "$CACHE" --out "$W/shap"
