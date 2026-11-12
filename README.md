# DiaMAS

Code for **"Reconstructing clinical decision environments for medical AI across evolving standards of care"**.

DiaMAS reconstructs the clinical decision environment of each admission (patient state, the therapies
and guideline evidence available in the assigned year, and the executable rules applicable in that year)
and uses it both to generate medication recommendations and to evaluate them. It contains:

| Component | Where |
|---|---|
| Cohort extraction, outcomes, exclusions and patient-level splits | `data_prep/` |
| Guideline corpus preparation and year-tagged indexes | `data_prep/convert_guidelines.py`, `scripts/build_indexes.py` |
| Clinical states, assigned year, specialist eligibility | `diamas_utils/clinical_phenotyper.py`, `diamas_utils/era_retriever.py` |
| Year-bounded guideline retrieval routed by specialty | `diamas_utils/era_retriever.py`, `diamas_utils/ada_guideline_processor.py` |
| Single and Collaborative programs, prompts | `run_diamas.py`, `diamas_utils/prompt_builder.py`, `diamas_utils/virtual_mdt_stage.py` |
| Medication availability registry | `agents/config.py` (`DrugEraConfig`, `DRUG_APPROVAL_YEAR`) |
| Critical drug-interaction audit | `evaluation/safety_checker.py` |
| Guideline Alignment Score (GAS) and executable rule library | `evaluation/ose_calculator.py` (rules), `evaluation/gas_v5_domain_eval.py` (scoring) |
| Adaptive Clinical Deliberation (ACD) | `acd/` |

DiaMAS is research software. It is not a medical device and must not be used for clinical decisions.

## Installation

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # edit, then load into the shell:
set -a && source .env && set +a
```

The recommendation programs call an OpenAI-compatible chat endpoint (`DIAMAS_BASE_URL`). In the study
every agent used a self-hosted Qwen3-32B served with vLLM; the model name passed to the endpoint is `qwen3-32b`.
Converting guideline PDFs additionally needs [MinerU](https://github.com/opendatalab/MinerU) 2.

## Data

MIMIC-III CareVue subset v1.4, MIMIC-IV v3.1 and the eICU Collaborative Research Database v2.0 are available
to credentialed users through [PhysioNet](https://physionet.org). Place the downloads (or symlinks) under
`ori_data_zip/` in the repository root, as named in `agents/config.py` (`PathConfig`). No patient-level data,
model inputs or outputs are distributed with this repository.

Guideline documents are not redistributed; the editions used are listed in Supplementary Table 2.

## Pipeline

All commands run from the repository root.

1. **Guideline corpus and indexes.** One sub-directory per source family (`ada`, `ada_supplements`, `kdigo`,
   `esc`, `acc_aha`); the edition year is read from the file name (`ada/ada_standards_<year>.md`,
   `<source>/<name>_<year>.md`).
   ```bash
   python data_prep/convert_guidelines.py --input guideline_pdfs --output data/guidelines_md
   python scripts/build_indexes.py --guidelines_dir data/guidelines_md --index_dir data/guidelines_indexes
   ```
2. **Admissions and outcomes.** Diabetes screening (at least two of five evidence categories) and extraction
   of each admission into `data/patient_cache_v1/`, then per-admission outcomes and severity:
   ```bash
   python data_prep/run_extraction.py --database all
   python data_prep/enhanced_outcome_extractor.py
   ```
3. **Cohort and splits.** Exclusions and the patient-level seed, development, test and eICU cohorts:
   ```bash
   python data_prep/build_cohort.py --output_dir data/cohort
   ```
4. **Run the two programs** on each cohort:
   ```bash
   for s in seed dev test eicu; do scripts/run_programs.sh data/cohort/$s/all results/$s; done
   ```
   | Program | Agents | Guideline sources |
   |---|---|---|
   | Single | endocrinology | ADA |
   | Collaborative | endocrinology + eligible nephrology and cardiology agents, conflict detection, revision, adjudication | ADA, supplementary diabetes guidance, KDIGO, ESC, ACC/AHA |

   Final synthesis temperature 0.0 (8,000 tokens); specialist opinions 0.3; conflict detection 0.0;
   at most two specialist rounds (Supplementary Table 3).
5. **GAS, collaboration value and ACD.** Scores both programs, pairs them (`d_gas = GAS_C - GAS_S`), builds
   the pre-branch features, runs the prespecified model selection on seed and development, evaluates the
   selected estimator on the test cohort and on eICU without refitting, and computes Tree SHAP:
   ```bash
   scripts/reproduce_acd.sh results data/cohort data/patient_cache_v1 ori_data_zip/eicu-collaborative-research-database-2.0 work
   ```
   Single steps: `scripts/score_gas.py`, `scripts/build_paired_table.py`, `acd/build_features.py`,
   `python -m acd.run_acd {fit,eval-primary,eval-eicu}`, `python -m acd.shap_analysis`.

