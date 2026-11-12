#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Adaptive Clinical Deliberation (ACD): fit, select and evaluate the collaboration-value estimator.

  fit           fit Ridge/GBDT candidates on the seed cohort, select on the development cohort
                (mean population utility over budgets 10%-90%, ties broken by lower MAE)
  eval-primary  frozen estimator on the MIMIC-III + MIMIC-IV test cohort (patient-cluster bootstrap)
  eval-eicu     frozen estimator on eICU without refitting (hospital-cluster bootstrap)

No LLM calls. eICU is never used for fitting or selection.
Run from the repository root:  python -m acd.run_acd --out <dir> ... fit
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


from acd import features as B


# ================================================================
# Frozen design
# ================================================================

OUT = Path(
    "outputs/deliberation_value_v2/"
    "source_mimic34_rich_v1"
)

class _AnyCount:
    """Equals any count. --expect-paper-counts replaces these with the study's cohort sizes."""
    def __eq__(self, other):
        return True

    def __repr__(self):
        return "any"


PAPER_COUNTS = dict(EXPECTED_SEED=10881, EXPECTED_DEV=4516, EXPECTED_PRIMARY_N=61337,
                    EXPECTED_PRIMARY_CLUSTERS=26160, EXPECTED_EICU_N=10409, EXPECTED_EICU_HOSPITALS=129)
EXPECTED_SEED = EXPECTED_DEV = EXPECTED_PRIMARY_N = EXPECTED_PRIMARY_CLUSTERS = _AnyCount()
EXPECTED_EICU_N = EXPECTED_EICU_HOSPITALS = _AnyCount()

DEV_BUDGETS = tuple(
    round(x, 2)
    for x in np.arange(0.10, 1.00, 0.10)
)

GRID = tuple(
    round(x, 2)
    for x in np.arange(0.00, 1.001, 0.05)
)

BOOT_SEED = 20260921


def utcnow():
    return datetime.now(timezone.utc).isoformat()


# ================================================================
# Basic ranking utilities
# ================================================================

def topk(scores, k):
    scores = np.asarray(scores, float)
    n = len(scores)

    k = max(0, min(int(k), n))

    mask = np.zeros(n, bool)

    if k == 0:
        return mask

    if k == n:
        mask[:] = True
        return mask

    # deterministic tie break
    order = np.lexsort(
        (np.arange(n), -scores)
    )

    mask[order[:k]] = True
    return mask


def utility(delta, mask):
    delta = np.asarray(delta, float)
    mask = np.asarray(mask, bool)

    return float(
        delta[mask].sum()
        / len(delta)
    )


def pearson(a, b):
    a = np.asarray(a, float)
    b = np.asarray(b, float)

    if (
        len(a) < 2
        or np.std(a) == 0
        or np.std(b) == 0
    ):
        return np.nan

    return float(
        np.corrcoef(a, b)[0, 1]
    )


def spearman(a, b):
    ra = (
        pd.Series(a)
        .rank(method="average")
        .to_numpy(float)
    )

    rb = (
        pd.Series(b)
        .rank(method="average")
        .to_numpy(float)
    )

    return pearson(ra, rb)


# ================================================================
# Frontier
# ================================================================

def make_frontier(scores, delta):

    scores = np.asarray(scores, float)
    delta = np.asarray(delta, float)

    n = len(delta)

    allc = float(
        delta.mean()
    )

    rows = []

    for b in GRID:

        k = int(
            round(b * n)
        )

        m = topk(
            scores,
            k,
        )

        oracle = topk(
            delta,
            k,
        )

        rows.append(
            {
                "budget": b,

                "selected_n":
                    int(m.sum()),

                "branch_rate":
                    float(m.mean()),

                "acd_value":
                    utility(
                        delta,
                        m,
                    ),

                "random_expected":
                    float(m.mean())
                    * allc,

                "hindsight_value":
                    utility(
                        delta,
                        oracle,
                    ),
            }
        )

    return pd.DataFrame(rows)


# ================================================================
# 5%-shell audit
# ================================================================

def make_shells(scores, delta):

    scores = np.asarray(scores, float)
    delta = np.asarray(delta, float)

    n = len(delta)

    order = np.lexsort(
        (
            np.arange(n),
            -scores,
        )
    )

    rows = []

    for lo_pct in range(
        0,
        100,
        5,
    ):

        hi_pct = lo_pct + 5

        lo = int(
            round(
                lo_pct
                / 100
                * n
            )
        )

        hi = int(
            round(
                hi_pct
                / 100
                * n
            )
        )

        idx = order[lo:hi]

        x = delta[idx]

        rows.append(
            {
                "shell":
                    f"{lo_pct}-{hi_pct}",

                "N":
                    len(x),

                "mean_delta":
                    float(x.mean()),

                "utility_increment":
                    float(
                        x.sum()
                        / n
                    ),

                "positive_rate":
                    float(
                        (x > 0).mean()
                    ),

                "tie_rate":
                    float(
                        (x == 0).mean()
                    ),

                "negative_rate":
                    float(
                        (x < 0).mean()
                    ),
            }
        )

    return pd.DataFrame(rows)


# ================================================================
# Cumulative routing audit
# ================================================================

def make_cumulative(scores, delta):

    scores = np.asarray(scores, float)
    delta = np.asarray(delta, float)

    n = len(delta)

    total_positive = int(
        (delta > 0).sum()
    )

    rows = []

    for b in GRID:

        k = int(
            round(
                b * n
            )
        )

        m = topk(
            scores,
            k,
        )

        sel = delta[m]
        unsel = delta[~m]

        rows.append(
            {
                "budget":
                    b,

                "selected_n":
                    int(m.sum()),

                "utility":
                    float(
                        sel.sum()
                        / n
                    )
                    if len(sel)
                    else 0.0,

                "positive_contribution":
                    float(
                        sel[
                            sel > 0
                        ].sum()
                        / n
                    )
                    if len(sel)
                    else 0.0,

                "negative_contribution":
                    float(
                        sel[
                            sel < 0
                        ].sum()
                        / n
                    )
                    if len(sel)
                    else 0.0,

                "positive_recall":
                    float(
                        (sel > 0).sum()
                        / total_positive
                    )
                    if total_positive
                    else np.nan,

                "positive_share_selected":
                    float(
                        (sel > 0).mean()
                    )
                    if len(sel)
                    else np.nan,

                "negative_share_selected":
                    float(
                        (sel < 0).mean()
                    )
                    if len(sel)
                    else np.nan,

                "mean_delta_selected":
                    float(
                        sel.mean()
                    )
                    if len(sel)
                    else np.nan,

                "mean_delta_unselected":
                    float(
                        unsel.mean()
                    )
                    if len(unsel)
                    else np.nan,
            }
        )

    return pd.DataFrame(rows)


# ================================================================
# Deciles
# ================================================================

def make_decile_membership(df):

    z = df.sort_values(
        [
            "vhat",
            "patient_id",
        ],
        kind="mergesort",
    ).copy()

    n = len(z)

    z["decile"] = (
        np.floor(
            np.arange(n)
            * 10
            / n
        ).astype(int)
        + 1
    ).clip(
        1,
        10,
    )

    return z


def decile_bootstrap(
    ranked,
    cluster_col,
    n_boot,
):

    clusters = (
        ranked[
            cluster_col
        ]
        .astype("string")
        .fillna("__MISSING__")
        .to_numpy()
    )

    unique_clusters = (
        pd.unique(
            clusters
        )
    )

    by_cluster = {
        u:
        np.flatnonzero(
            clusters == u
        )
        for u
        in unique_clusters
    }

    dec = (
        ranked["decile"]
        .to_numpy(int)
    )

    y = (
        ranked["d_gas"]
        .to_numpy(float)
    )

    rng = (
        np.random
        .default_rng(
            BOOT_SEED
        )
    )

    boot = np.empty(
        (
            n_boot,
            10,
        ),
        float,
    )

    for b in range(
        n_boot
    ):

        sampled = rng.choice(
            unique_clusters,
            size=len(
                unique_clusters
            ),
            replace=True,
        )

        idx = np.concatenate(
            [
                by_cluster[u]
                for u
                in sampled
            ]
        )

        db = dec[idx]
        yb = y[idx]

        for d in range(
            1,
            11,
        ):

            m = (
                db == d
            )

            boot[
                b,
                d - 1
            ] = float(
                yb[m].mean()
            )

    rows = []

    for d in range(
        1,
        11,
    ):

        g = ranked[
            ranked[
                "decile"
            ] == d
        ]

        dg = (
            g["d_gas"]
            .to_numpy(float)
        )

        lo, hi = np.quantile(
            boot[
                :,
                d - 1
            ],
            [
                0.025,
                0.975,
            ],
        )

        rows.append(
            {
                "decile":
                    f"D{d}",

                "N":
                    len(g),

                "mean_vhat":
                    float(
                        g[
                            "vhat"
                        ].mean()
                    ),

                "observed_mean_delta":
                    float(
                        g[
                            "d_gas"
                        ].mean()
                    ),

                "ci_low":
                    float(lo),

                "ci_high":
                    float(hi),

                "negative_rate":
                    float(
                        (dg < 0).mean()
                    ),

                "tie_rate":
                    float(
                        (dg == 0).mean()
                    ),

                "positive_rate":
                    float(
                        (dg > 0).mean()
                    ),
            }
        )

    tab = pd.DataFrame(
        rows
    )

    diff_boot = (
        boot[:, 9]
        - boot[:, 0]
    )

    lo, hi = np.quantile(
        diff_boot,
        [
            0.025,
            0.975,
        ],
    )

    contrast = {
        "D10_minus_D1":
            float(
                tab.iloc[-1][
                    "observed_mean_delta"
                ]
                -
                tab.iloc[0][
                    "observed_mean_delta"
                ]
            ),

        "ci_low":
            float(lo),

        "ci_high":
            float(hi),

        "n_boot":
            n_boot,

        "n_clusters":
            len(
                unique_clusters
            ),
    }

    return (
        tab,
        contrast,
    )


# ================================================================
# FIT: MIMIC-III + MIMIC-IV only
# ================================================================

def fit_cmd(args):

    if (
        OUT.exists()
        and any(
            OUT.iterdir()
        )
        and not args.force
    ):
        raise RuntimeError(
            f"{OUT} already exists. "
            "Use --force only for "
            "an intentional rerun."
        )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    seed = pd.read_csv(
        B.SEED,
        low_memory=False,
    )

    dev = pd.read_csv(
        B.DEV,
        low_memory=False,
    )

    seed[
        "db_source"
    ] = seed[
        "db_source"
    ].map(
        B.norm_db
    )

    dev[
        "db_source"
    ] = dev[
        "db_source"
    ].map(
        B.norm_db
    )

    source_dbs = {
        "mimic3",
        "mimic4",
    }

    seed = seed[
        seed[
            "db_source"
        ].isin(
            source_dbs
        )
    ].copy()

    dev = dev[
        dev[
            "db_source"
        ].isin(
            source_dbs
        )
    ].copy()

    assert (
        len(seed)
        == EXPECTED_SEED
    ), len(seed)

    assert (
        len(dev)
        == EXPECTED_DEV
    ), len(dev)

    assert (
        "eicu"
        not in
        set(
            seed[
                "db_source"
            ]
        )
    )

    assert (
        "eicu"
        not in
        set(
            dev[
                "db_source"
            ]
        )
    )

    # Available-ID split checks
    for key in [
        "patient_id",
        "split_key",
    ]:

        a = set(
            seed[
                key
            ].astype(str)
        )

        b = set(
            dev[
                key
            ].astype(str)
        )

        assert not (
            a & b
        ), (
            key,
            len(a & b),
        )

    Xs = B.prepare(
        seed
    )

    Xd = B.prepare(
        dev
    )

    ys = pd.to_numeric(
        seed[
            B.TARGET
        ],
        errors="coerce",
    ).to_numpy(float)

    yd = pd.to_numeric(
        dev[
            B.TARGET
        ],
        errors="coerce",
    ).to_numpy(float)

    candidates = (
        [
            (
                "ridge",
                p,
            )
            for p
            in B.RIDGE_GRID
        ]
        +
        [
            (
                "gbdt",
                p,
            )
            for p
            in B.GBDT_GRID
        ]
    )

    records = []

    best = None
    best_model = None
    best_pred = None

    print(
        "=" * 108
    )

    print(
        "CANONICAL ACD FIT — "
        "MIMIC-III + MIMIC-IV"
    )

    print(
        "=" * 108
    )

    print(
        f"seed N={len(seed):,}"
        f" | dev N={len(dev):,}"
    )

    print(
        "eICU target: "
        "NOT ACCESSED"
    )

    print()

    for i, (
        family,
        params,
    ) in enumerate(
        candidates,
        1,
    ):

        model = B.build(
            family,
            params,
        )

        model.fit(
            Xs,
            ys,
        )

        pred = np.asarray(
            model.predict(
                Xd
            ),
            float,
        )

        fr = []

        for budget in DEV_BUDGETS:

            k = int(
                round(
                    budget
                    * len(yd)
                )
            )

            mask = topk(
                pred,
                k,
            )

            fr.append(
                utility(
                    yd,
                    mask,
                )
            )

        mean_value = float(
            np.mean(fr)
        )

        mae = float(
            mean_absolute_error(
                yd,
                pred,
            )
        )

        rmse = float(
            mean_squared_error(
                yd,
                pred,
            )
            ** 0.5
        )

        r = pearson(
            pred,
            yd,
        )

        row = {
            "candidate_id":
                i,

            "family":
                family,

            "params_json":
                json.dumps(
                    params,
                    sort_keys=True,
                ),

            "dev_allocation_value_mean_10_90":
                mean_value,

            "dev_mae":
                mae,

            "dev_rmse":
                rmse,

            "dev_pearson":
                r,
        }

        records.append(
            row
        )

        key = (
            mean_value,
            -mae,
        )

        if (
            best is None
            or key
            > best[
                "key"
            ]
        ):

            best = {
                "key":
                    key,

                **row,

                "params":
                    params,
            }

            best_model = model
            best_pred = pred

        print(
            f"[{i:02d}/"
            f"{len(candidates):02d}] "
            f"{family:<5} "
            f"value={mean_value:+.6f} "
            f"MAE={mae:.5f} "
            f"RMSE={rmse:.5f} "
            f"r={r:+.4f} "
            f"{params}"
        )

    pd.DataFrame(
        records
    ).sort_values(
        [
            "dev_allocation_value_mean_10_90",
            "dev_mae",
        ],
        ascending=[
            False,
            True,
        ],
    ).to_csv(
        OUT
        / "dev_model_search.csv",
        index=False,
    )

    make_frontier(
        best_pred,
        yd,
    ).to_csv(
        OUT
        / "dev_frontier.csv",
        index=False,
    )

    joblib.dump(
        best_model,
        OUT
        / "model.joblib",
    )

    manifest = {
        "schema":
            "source_mimic34_rich_v1",

        "created_utc":
            utcnow(),

        "source":
            "MIMIC-III + MIMIC-IV",

        "seed_n":
            len(seed),

        "dev_n":
            len(dev),

        "seed_counts":
            seed[
                "db_source"
            ]
            .value_counts()
            .sort_index()
            .to_dict(),

        "dev_counts":
            dev[
                "db_source"
            ]
            .value_counts()
            .sort_index()
            .to_dict(),

        "eicu_rows_used_for_fit":
            0,

        "eicu_rows_used_for_model_selection":
            0,

        "feature_schema": {
            "numeric":
                B.RICH_NUM,

            "categorical":
                B.RICH_CAT,
        },

        "selection_rule":
            (
                "maximize mean population "
                "incremental GAS over fixed "
                "combined-MIMIC development "
                "budgets 10%-90%; "
                "tie-break lower MAE"
            ),

        "winner": {
            "family":
                best[
                    "family"
                ],

            "params":
                best[
                    "params"
                ],

            "dev_mean_value":
                best[
                    "dev_allocation_value_mean_10_90"
                ],

            "dev_mae":
                best[
                    "dev_mae"
                ],

            "dev_rmse":
                best[
                    "dev_rmse"
                ],

            "dev_pearson":
                best[
                    "dev_pearson"
                ],
        },

        "main_operating_point":
            "NONE; full prespecified grid used",

        "external_target":
            "eICU only",
    }

    (
        OUT
        / "freeze_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()

    print(
        "WINNER:",
        best[
            "family"
        ],
        best[
            "params"
        ],
    )

    print(
        "Frozen ->",
        OUT,
    )


# ================================================================
# Shared evaluation
# ================================================================

def evaluate(
    target,
    n_boot,
    force,
):

    model_path = (
        OUT
        / "model.joblib"
    )

    manifest_path = (
        OUT
        / "freeze_manifest.json"
    )

    if (
        not model_path.exists()
        or not manifest_path.exists()
    ):
        raise FileNotFoundError(
            "Run fit first."
        )

    manifest = json.loads(
        manifest_path.read_text()
    )

    assert (
        manifest[
            "source"
        ]
        ==
        "MIMIC-III + MIMIC-IV"
    )

    raw = pd.read_csv(
        B.TEST,
        low_memory=False,
    )

    raw[
        "db_source"
    ] = raw[
        "db_source"
    ].map(
        B.norm_db
    )

    if target == "primary":

        raw = raw[
            raw[
                "db_source"
            ].isin(
                {
                    "mimic3",
                    "mimic4",
                }
            )
        ].copy()

        assert (
            len(raw)
            == EXPECTED_PRIMARY_N
        )

        out = (
            OUT
            / "test_primary_non_eicu"
        )

        cluster_col = (
            "split_key"
        )

        expected_clusters = (
            EXPECTED_PRIMARY_CLUSTERS
        )

    elif target == "eicu":

        raw = raw[
            raw[
                "db_source"
            ]
            == "eicu"
        ].copy()

        assert (
            len(raw)
            == EXPECTED_EICU_N
        )

        out = (
            OUT
            / "test_eicu"
        )

        cluster_col = (
            "hospital_id"
        )

        expected_clusters = (
            EXPECTED_EICU_HOSPITALS
        )

    else:
        raise ValueError(
            target
        )

    if (
        out.exists()
        and any(
            out.iterdir()
        )
        and not force
    ):
        raise RuntimeError(
            f"{out} already exists."
        )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = B.extract_target_features(
        raw,
        B.CACHE,
    )

    if target == "eicu":

        df = (
            B.attach_audited_hospital_map(
                df,
                B.HOSPITAL_MAP,
            )
        )

        assert not (
            df[
                "hospital_id"
            ].isna().any()
        )

    assert (
        df[
            cluster_col
        ]
        .astype(str)
        .nunique()
        == expected_clusters
    )

    model = joblib.load(
        model_path
    )

    pred = np.asarray(
        model.predict(
            B.prepare(
                df
            )
        ),
        float,
    )

    y = pd.to_numeric(
        df[
            B.TARGET
        ],
        errors="coerce",
    ).to_numpy(float)

    predout = pd.DataFrame(
        {
            "patient_id":
                df[
                    "patient_id"
                ].astype(str),

            "split_key":
                df[
                    "split_key"
                ].astype(str),

            "db_source":
                df[
                    "db_source"
                ].astype(str),

            "d_gas":
                y,

            "vhat":
                pred,
        }
    )

    if (
        target
        == "eicu"
    ):

        predout[
            "hospital_id"
        ] = (
            df[
                "hospital_id"
            ].astype(str)
        )

    fr = make_frontier(
        pred,
        y,
    )

    shells = make_shells(
        pred,
        y,
    )

    cumulative = (
        make_cumulative(
            pred,
            y,
        )
    )

    ranked = (
        make_decile_membership(
            predout
        )
    )

    deciles, contrast = (
        decile_bootstrap(
            ranked,
            cluster_col,
            n_boot,
        )
    )

    predout.to_csv(
        out
        / "predictions.csv",
        index=False,
    )

    fr.to_csv(
        out
        / "frontier.csv",
        index=False,
    )

    shells.to_csv(
        out
        / "routing_shells.csv",
        index=False,
    )

    cumulative.to_csv(
        out
        / "routing_cumulative.csv",
        index=False,
    )

    deciles.to_csv(
        out
        / "deciles.csv",
        index=False,
    )

    allc = float(
        y.mean()
    )

    positive = float(
        y[
            y > 0
        ].sum()
        / len(y)
    )

    max_row = (
        fr.loc[
            fr[
                "acd_value"
            ].idxmax()
        ]
    )

    pattern = {
        "D10_gt_D1":
            bool(
                contrast[
                    "D10_minus_D1"
                ]
                > 0
            ),

        "D10_D1_CI_positive":
            bool(
                contrast[
                    "ci_low"
                ]
                > 0
            ),

        "acd_gt_random_5_to_95":
            int(
                (
                    fr.loc[
                        (
                            fr[
                                "budget"
                            ] > 0
                        )
                        &
                        (
                            fr[
                                "budget"
                            ] < 1
                        ),
                        "acd_value",
                    ].to_numpy()
                    >
                    fr.loc[
                        (
                            fr[
                                "budget"
                            ] > 0
                        )
                        &
                        (
                            fr[
                                "budget"
                            ] < 1
                        ),
                        "random_expected",
                    ].to_numpy()
                ).sum()
            ),

        "grid_points_5_to_95":
            19,

        "first_shell_positive":
            bool(
                shells.iloc[
                    0
                ][
                    "mean_delta"
                ]
                > 0
            ),

        "last_shell_negative":
            bool(
                shells.iloc[
                    -1
                ][
                    "mean_delta"
                ]
                < 0
            ),

        "interior_max_above_all_c":
            bool(
                max_row[
                    "acd_value"
                ]
                > allc
            ),
    }

    summary = {
        "target":
            target,

        "N":
            len(df),

        "n_clusters":
            int(
                df[
                    cluster_col
                ]
                .astype(str)
                .nunique()
            ),

        "cluster_col":
            cluster_col,

        "pearson":
            pearson(
                pred,
                y,
            ),

        "all_c":
            allc,

        "positive_rate":
            float(
                (y > 0).mean()
            ),

        "tie_rate":
            float(
                (y == 0).mean()
            ),

        "negative_rate":
            float(
                (y < 0).mean()
            ),

        "hindsight_max":
            positive,

        "decile_contrast":
            contrast,

        "descriptive_grid_max":
            max_row.to_dict(),

        "pattern":
            pattern,
    }

    if (
        target
        == "eicu"
    ):

        hospital_rows = []

        hospital_summary = []

        for budget in GRID:

            k = int(
                round(
                    budget
                    * len(df)
                )
            )

            mask = topk(
                pred,
                k,
            )

            tmp = pd.DataFrame(
                {
                    "hospital_id":
                        df[
                            "hospital_id"
                        ].astype(str),

                    "d_gas":
                        y,

                    "selected":
                        mask.astype(int),
                }
            )

            tmp[
                "contrast"
            ] = (
                tmp[
                    "d_gas"
                ]
                *
                (
                    tmp[
                        "selected"
                    ]
                    - 1
                )
            )

            effects = []

            for hosp, g in (
                tmp.groupby(
                    "hospital_id"
                )
            ):

                effect = float(
                    g[
                        "contrast"
                    ].mean()
                )

                effects.append(
                    effect
                )

                hospital_rows.append(
                    {
                        "budget":
                            budget,

                        "hospital_id":
                            hosp,

                        "N":
                            len(g),

                        "acd_minus_all_c":
                            effect,
                    }
                )

            effects = np.asarray(
                effects,
                float,
            )

            hospital_summary.append(
                {
                    "budget":
                        budget,

                    "positive_hospitals":
                        int(
                            (
                                effects > 0
                            ).sum()
                        ),

                    "zero_hospitals":
                        int(
                            (
                                effects == 0
                            ).sum()
                        ),

                    "negative_hospitals":
                        int(
                            (
                                effects < 0
                            ).sum()
                        ),

                    "positive_hospital_rate":
                        float(
                            (
                                effects > 0
                            ).mean()
                        ),

                    "median_effect":
                        float(
                            np.median(
                                effects
                            )
                        ),
                }
            )

        pd.DataFrame(
            hospital_rows
        ).to_csv(
            out
            / "hospital_by_budget.csv",
            index=False,
        )

        pd.DataFrame(
            hospital_summary
        ).to_csv(
            out
            / "hospital_budget_summary.csv",
            index=False,
        )

    (
        out
        / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "=" * 108
    )

    print(
        target.upper()
    )

    print(
        "=" * 108
    )

    print(
        f"N={len(df):,} "
        f"clusters={summary['n_clusters']:,}"
    )

    print(
        f"Pearson="
        f"{summary['pearson']:+.4f}"
    )

    print(
        "D10-D1="
        f"{contrast['D10_minus_D1']:+.6f} "
        f"[{contrast['ci_low']:+.6f}, "
        f"{contrast['ci_high']:+.6f}]"
    )

    print(
        "All-C=",
        f"{allc:+.6f}",
    )

    print(
        "descriptive grid max=",
        f"{100*max_row['branch_rate']:.1f}% / "
        f"{max_row['acd_value']:+.6f}",
    )

    print(
        "pattern=",
        json.dumps(
            pattern,
            indent=2,
        ),
    )

    print(
        "Artifacts ->",
        out,
    )


# ================================================================
# CLI
# ================================================================

def main():
    global OUT

    ap = argparse.ArgumentParser(description="ACD fit / evaluation")
    ap.add_argument("--seed-table", required=True, help="seed features table (acd/build_features.py)")
    ap.add_argument("--dev-table", required=True, help="development features table")
    ap.add_argument("--test-table", required=True, help="paired test table (scripts/build_paired_table.py)")
    ap.add_argument("--cache", required=True, help="patient cache root: <cache>/<db>/<db>/<patient_id>.json")
    ap.add_argument("--hospital-map", required=True, help="eICU patient_id -> hospital_id (data_prep/eicu_hospital_map.py)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect-paper-counts", action="store_true",
                    help="assert the study's cohort sizes (seed 10,881; development 4,516; test 61,337; eICU 10,409)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("fit")
    p.add_argument("--force", action="store_true")
    for name in ("eval-primary", "eval-eicu"):
        p = sub.add_parser(name)
        p.add_argument("--n-boot", type=int, default=5000)
        p.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if args.expect_paper_counts:
        globals().update(PAPER_COUNTS)
    B.SEED, B.DEV, B.TEST = Path(args.seed_table), Path(args.dev_table), Path(args.test_table)
    B.CACHE, B.HOSPITAL_MAP = Path(args.cache), Path(args.hospital_map)
    OUT = Path(args.out)

    if args.cmd == "fit":
        fit_cmd(args)
    else:
        evaluate("primary" if args.cmd == "eval-primary" else "eicu", args.n_boot, args.force)


if __name__ == "__main__":
    main()
