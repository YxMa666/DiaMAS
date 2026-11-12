#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap

import argparse

from acd import features as B

ROOT = PRED = MODEL = OUT = None  # set from the command line in main()

class _AnyCount:
    """Equals any count. --expect-paper-counts replaces these with the study's test cohort size."""
    def __eq__(self, other):
        return True


PAPER_COUNTS = dict(EXPECTED_N=61337, EXPECTED_CLUSTERS=26160)
EXPECTED_N = EXPECTED_CLUSTERS = _AnyCount()

RAW_FEATURES = [
    "n_specialties",
    "n_states",
    "is_ascvd",
    "cardio_present",
    "age",
    "egfr",
    "heart_failure",
    "aki_suspected",
    "era",
    "specialty_combo",
    "sex",
    "dm_type",
    "ckd_g_stage",
]

DOMAINS = {
    "Clinical complexity": [
        "n_specialties",
        "n_states",
        "specialty_combo",
    ],
    "Temporal context": [
        "era",
    ],
    "Cardiovascular": [
        "is_ascvd",
        "cardio_present",
        "heart_failure",
    ],
    "Renal": [
        "egfr",
        "aki_suspected",
        "ckd_g_stage",
    ],
    "Demographic / metabolic": [
        "age",
        "sex",
        "dm_type",
    ],
}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(block)
    return h.hexdigest()


def transformed_to_raw(name):
    """
    Map ColumnTransformer output feature
    back to one of the 13 frozen raw features.
    """

    s = str(name)

    if "__" in s:
        s = s.split("__", 1)[1]

    if s.startswith("missingindicator_"):
        s = s[len("missingindicator_"):]

    # longest first, important for specialty_combo etc.
    for raw in sorted(
        RAW_FEATURES,
        key=len,
        reverse=True,
    ):
        if (
            s == raw
            or s.startswith(raw + "_")
        ):
            return raw

    raise RuntimeError(
        f"Cannot map transformed feature "
        f"to raw feature: {name}"
    )


def main():
    global ROOT, PRED, MODEL, OUT
    ap = argparse.ArgumentParser(description="Tree SHAP of the frozen ACD estimator on the MIMIC test cohort")
    ap.add_argument("--model-dir", required=True, help="directory with model.joblib and test_primary_non_eicu/predictions.csv")
    ap.add_argument("--test-table", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect-paper-counts", action="store_true")
    a = ap.parse_args()
    if a.expect_paper_counts:
        globals().update(PAPER_COUNTS)
    ROOT = Path(a.model_dir)
    PRED, MODEL, OUT = ROOT / "test_primary_non_eicu" / "predictions.csv", ROOT / "model.joblib", Path(a.out)
    B.TEST, B.CACHE = Path(a.test_table), Path(a.cache)

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not MODEL.exists():
        raise FileNotFoundError(MODEL)

    if not PRED.exists():
        raise FileNotFoundError(PRED)

    # ------------------------------------------------------------
    # Reconstruct exactly the same combined-MIMIC held-out cohort
    # as the latest canonical evaluation.
    # ------------------------------------------------------------

    raw = pd.read_csv(
        B.TEST,
        low_memory=False,
    )

    raw["db_source"] = (
        raw["db_source"]
        .map(B.norm_db)
    )

    raw = raw[
        raw["db_source"].isin(
            {"mimic3", "mimic4"}
        )
    ].copy()

    assert len(raw) == EXPECTED_N, len(raw)

    df = B.extract_target_features(
        raw,
        B.CACHE,
    )

    assert len(df) == EXPECTED_N
    assert (
        df["split_key"]
        .astype(str)
        .nunique()
        == EXPECTED_CLUSTERS
    )

    X = B.prepare(df)

    assert list(X.columns) == (
        B.RICH_NUM + B.RICH_CAT
    )

    assert set(X.columns) == set(
        RAW_FEATURES
    )

    # ------------------------------------------------------------
    # Frozen model identity
    # ------------------------------------------------------------

    model = joblib.load(MODEL)

    if not hasattr(model, "named_steps"):
        raise RuntimeError(
            "Expected sklearn Pipeline."
        )

    if (
        "prep" not in model.named_steps
        or "reg" not in model.named_steps
    ):
        raise RuntimeError(
            f"Unexpected pipeline steps: "
            f"{list(model.named_steps)}"
        )

    prep = model.named_steps["prep"]
    reg = model.named_steps["reg"]

    pred_full = np.asarray(
        model.predict(X),
        float,
    )

    Xt = prep.transform(X)

    pred_reg = np.asarray(
        reg.predict(Xt),
        float,
    )

    pipeline_identity_error = float(
        np.max(
            np.abs(
                pred_full
                - pred_reg
            )
        )
    )

    # Cross-check against canonical saved predictions.
    frozen = pd.read_csv(
        PRED,
        low_memory=False,
    )

    frozen["patient_id"] = (
        frozen["patient_id"]
        .astype(str)
    )

    chk = pd.DataFrame({
        "patient_id":
            df["patient_id"].astype(str),
        "pred_now":
            pred_full,
    }).merge(
        frozen[
            [
                "patient_id",
                "vhat",
            ]
        ],
        on="patient_id",
        how="inner",
        validate="one_to_one",
    )

    assert len(chk) == EXPECTED_N

    prediction_identity_error = float(
        np.max(
            np.abs(
                chk["pred_now"].to_numpy(float)
                -
                chk["vhat"].to_numpy(float)
            )
        )
    )

    if prediction_identity_error > 1e-10:
        raise RuntimeError(
            "Prediction identity failed: "
            f"{prediction_identity_error}"
        )

    if pipeline_identity_error > 1e-10:
        raise RuntimeError(
            "Pipeline/reg identity failed: "
            f"{pipeline_identity_error}"
        )

    # ------------------------------------------------------------
    # Transformed feature names
    # ------------------------------------------------------------

    transformed_names = (
        prep.get_feature_names_out()
    )

    transformed_names = np.asarray(
        transformed_names,
        dtype=str,
    )

    if Xt.shape[1] != len(
        transformed_names
    ):
        raise RuntimeError(
            "Feature-name / transformed-matrix "
            "dimension mismatch."
        )

    raw_map = [
        transformed_to_raw(x)
        for x in transformed_names
    ]

    if set(raw_map) != set(RAW_FEATURES):
        raise RuntimeError(
            "Not all 13 raw features represented.\n"
            f"mapped={sorted(set(raw_map))}"
        )

    # ------------------------------------------------------------
    # Tree SHAP on frozen GBDT
    # ------------------------------------------------------------

    explainer = shap.TreeExplainer(
        reg
    )

    sv = explainer.shap_values(
        Xt,
        check_additivity=False,
    )

    sv = np.asarray(
        sv,
        dtype=float,
    )

    if sv.ndim == 3:
        if sv.shape[-1] != 1:
            raise RuntimeError(
                f"Unexpected SHAP shape: "
                f"{sv.shape}"
            )
        sv = sv[..., 0]

    if sv.shape != Xt.shape:
        raise RuntimeError(
            f"SHAP shape mismatch: "
            f"{sv.shape} vs {Xt.shape}"
        )

    expected_value = np.asarray(
        explainer.expected_value
    ).reshape(-1)

    if len(expected_value) != 1:
        raise RuntimeError(
            f"Unexpected expected_value: "
            f"{expected_value}"
        )

    base_value = float(
        expected_value[0]
    )

    reconstructed = (
        base_value
        + sv.sum(axis=1)
    )

    shap_additivity_error = float(
        np.max(
            np.abs(
                reconstructed
                - pred_reg
            )
        )
    )

    if shap_additivity_error > 1e-8:
        raise RuntimeError(
            "SHAP additivity failed: "
            f"{shap_additivity_error}"
        )

    # ------------------------------------------------------------
    # Transformed feature importance
    # ------------------------------------------------------------

    transformed_tab = pd.DataFrame({
        "transformed_feature":
            transformed_names,
        "raw_feature":
            raw_map,
        "mean_abs_shap":
            np.mean(
                np.abs(sv),
                axis=0,
            ),
    }).sort_values(
        "mean_abs_shap",
        ascending=False,
    )

    transformed_tab.to_csv(
        OUT
        / "transformed_feature_importance.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # Aggregate transformed SHAP -> 13 RAW features
    #
    # IMPORTANT:
    # first sum SHAP values belonging to a raw feature
    # within each patient;
    # then take mean(abs(.)).
    # ------------------------------------------------------------

    raw_sv = np.zeros(
        (
            len(df),
            len(RAW_FEATURES),
        ),
        dtype=float,
    )

    raw_index = {
        f: i
        for i, f
        in enumerate(RAW_FEATURES)
    }

    for j, raw_name in enumerate(
        raw_map
    ):
        raw_sv[
            :,
            raw_index[raw_name]
        ] += sv[:, j]

    raw_importance = np.mean(
        np.abs(raw_sv),
        axis=0,
    )

    raw_tab = pd.DataFrame({
        "raw_feature":
            RAW_FEATURES,
        "mean_abs_shap":
            raw_importance,
    }).sort_values(
        "mean_abs_shap",
        ascending=False,
    )

    raw_tab.to_csv(
        OUT
        / "raw_feature_importance.csv",
        index=False,
    )

    # ------------------------------------------------------------
    # Domain attribution
    #
    # Domain importance:
    # E_i | SUM_{j in domain} phi_ij |
    #
    # Do NOT sum raw mean-|SHAP| values.
    # ------------------------------------------------------------

    domain_rows = []

    for domain, feats in DOMAINS.items():

        idx = [
            raw_index[f]
            for f in feats
        ]

        domain_shap = (
            raw_sv[:, idx]
            .sum(axis=1)
        )

        importance = float(
            np.mean(
                np.abs(
                    domain_shap
                )
            )
        )

        domain_rows.append({
            "domain":
                domain,
            "importance":
                importance,
        })

    domain_tab = pd.DataFrame(
        domain_rows
    )

    total_domain_importance = float(
        domain_tab[
            "importance"
        ].sum()
    )

    domain_tab[
        "importance_share"
    ] = (
        domain_tab["importance"]
        / total_domain_importance
    )

    domain_tab = domain_tab.sort_values(
        "importance",
        ascending=False,
    )

    domain_tab.to_csv(
        OUT
        / "domain_importance.csv",
        index=False,
    )

    # Save aggregated patient-level raw SHAP only.
    # More compact and directly useful for audit / plotting.
    np.savez_compressed(
        OUT / "raw_shap_values.npz",
        patient_id=(
            df["patient_id"]
            .astype(str)
            .to_numpy()
        ),
        raw_features=np.asarray(
            RAW_FEATURES,
            dtype=str,
        ),
        shap_values=raw_sv,
        expected_value=np.asarray(
            [base_value]
        ),
    )

    domain_share_sum = float(
        domain_tab[
            "importance_share"
        ].sum()
    )

    audit = {
        "status":
            "PASS",
        "source":
            "MIMIC-III + MIMIC-IV",
        "heldout_N":
            len(df),
        "subject_clusters":
            int(
                df["split_key"]
                .astype(str)
                .nunique()
            ),
        "raw_feature_count":
            len(RAW_FEATURES),
        "transformed_feature_count":
            int(Xt.shape[1]),
        "domain_count":
            len(DOMAINS),
        "prediction_identity_error":
            prediction_identity_error,
        "pipeline_identity_error":
            pipeline_identity_error,
        "shap_additivity_error":
            shap_additivity_error,
        "domain_share_sum":
            domain_share_sum,
        "model_sha256":
            sha256_file(MODEL),
        "predictions_sha256":
            sha256_file(PRED),
        "guardrail":
            (
                "SHAP attribution within frozen "
                "combined-MIMIC ACD estimator; "
                "not causal effect."
            ),
    }

    if abs(
        domain_share_sum - 1.0
    ) > 1e-12:
        raise RuntimeError(
            "Domain shares do not sum to 1."
        )

    (
        OUT / "audit.json"
    ).write_text(
        json.dumps(
            audit,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "=" * 100
    )
    print(
        "COMBINED-MIMIC HELD-OUT SHAP"
    )
    print(
        "=" * 100
    )

    print(
        json.dumps(
            audit,
            indent=2,
        )
    )

    print(
        "\nDOMAIN IMPORTANCE"
    )
    print(
        domain_tab.to_string(
            index=False
        )
    )

    print(
        "\nRAW FEATURE IMPORTANCE"
    )
    print(
        raw_tab.to_string(
            index=False
        )
    )

    print(
        "\nSaved ->",
        OUT,
    )


if __name__ == "__main__":
    main()
