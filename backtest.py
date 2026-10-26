
#!/usr/bin/env python3
"""Historical backtest and diagnostics for Track 2 forecasting samplers.

Examples:
    python backtest.py --n 300

    python backtest.py --n 300 --v3-sweep

    python backtest.py --n 300 --compare-v3-v4 --seed 0

    python backtest.py --fx-corr-sweep --all-dates \
        --start 2012-01-01 --end 2024-12-31 \
        --output fx_corr_full_results.csv

Each forecast uses only observations available at or before
its as-of date.

All models share the same cases, Monte Carlo seed, and baseline.
Lower normalized scores are better.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from qfbench2_common.scoring import crps
from qfbench2_track_forecasting.scoring import TAIL_METRICS
from qfbench2_track_forecasting.improved import (
    BASELINE,
    Config,
    sample_levels,
    variant,
)


PANELS = {
    "rates": "units/t2-F1-hawkish-cut-2024/rates_daily.parquet",
    "fx": "units/t2-F3-election-2024-joint/g10_fx_daily.parquet",
}

TASKS = {
    "ust_curve_1to4": (
        "rates",
        ["UST_2Y", "UST_5Y", "UST_10Y", "UST_30Y"],
        [1, 2, 3, 4],
    ),
    "ust10_1d": (
        "rates",
        ["UST_10Y"],
        [1],
    ),
    "fx_g10_1to4": (
        "fx",
        ["EUR", "GBP", "JPY", "CHF", "AUD", "CAD"],
        [1, 2, 3, 4],
    ),
    "eur_4d": (
        "fx",
        ["EUR"],
        [4],
    ),
}

TAIL = (0.01, 0.05, 0.95, 0.99)

FX_CORR_LAMBDAS = {
    "FX_Corr_A": 0.90,
    "FX_Corr_B": 0.97,
    "FX_Corr_C": 0.99,
}


def load(path):
    df = pd.read_parquet(path)

    col = next(
        c for c in ("asset", "asset_id", "series")
        if c in df.columns
    )

    df["date"] = pd.to_datetime(
        df["date"].astype(str).str[:10]
    )

    return df.pivot_table(
        index="date",
        columns=col,
        values="value",
    ).sort_index()


def components(samples, y):
    s = samples.reshape(samples.shape[0], -1)
    yy = y.reshape(-1)

    m = float(crps.crps_marginal(s, yy))

    t = float(
        TAIL_METRICS["pinball"](s, yy, TAIL)
    )

    j = (
        float(crps.variogram_score(s, yy, p=0.5))
        if yy.size > 1
        else 0.0
    )

    return m, j, t


def normalized(c, b):
    m, j, t = c
    mb, jb, tb = b

    if jb > 0:
        return (
            0.5 * m / mb
            + 0.3 * j / jb
            + 0.2 * t / tb
        )

    return (
        0.714 * m / mb
        + 0.286 * t / tb
    )


def volatility_features(steps):
    """Historical volatility diagnostics using past steps only."""

    if len(steps) < 2:
        return np.nan, np.nan, np.nan

    recent = np.std(
        steps[-20:],
        axis=0,
        ddof=0,
    )

    long = np.std(
        steps[-120:],
        axis=0,
        ddof=0,
    )

    valid = (
        np.isfinite(recent)
        & np.isfinite(long)
        & (long > 0)
    )

    ratio = (
        float(np.mean(recent[valid] / long[valid]))
        if np.any(valid)
        else np.nan
    )

    return (
        float(np.mean(recent)),
        float(np.mean(long)),
        ratio,
    )


def dynamic_scale(vol_ratio, upper):
    """Dynamic FX scale from 0.9 to upper."""

    if not np.isfinite(vol_ratio):
        return 0.9

    weight = float(
        np.clip(
            (vol_ratio - 1.0) / 0.4,
            0.0,
            1.0,
        )
    )

    return 0.9 + (upper - 0.9) * weight


def active_config(task, name, cfg, vol_ratio):
    """Apply experimental changes to G10 FX only."""

    if task != "fx_g10_1to4":
        return cfg

    if name in FX_CORR_LAMBDAS:
        return variant(
            cfg,
            corr="ewma",
            lam_corr=FX_CORR_LAMBDAS[name],
        )

    if name == "Dynamic_A":
        return variant(
            cfg,
            scale=dynamic_scale(vol_ratio, 1.0),
        )

    if name == "Dynamic_B":
        return variant(
            cfg,
            scale=dynamic_scale(vol_ratio, 0.95),
        )

    return cfg


def run(
    configs,
    n_cases,
    start="2012-01-01",
    end=None,
    n_draws=500,
    output=None,
    split_date=None,
    return_cases=False,
    seed=0,
):
    panels = {
        k: load(v)
        for k, v in PANELS.items()
    }

    table = {}
    records = []

    for task, (panel, assets, hs) in TASKS.items():

        p = panels[panel][assets].dropna()
        vals = p.to_numpy()

        lo = int(
            np.searchsorted(
                p.index.values,
                np.datetime64(start),
            )
        )

        # Every forecast horizon must have a realized value.
        hi = len(p) - max(hs)

        if end is not None:
            hi = min(
                hi,
                int(
                    np.searchsorted(
                        p.index.values,
                        np.datetime64(end),
                        side="right",
                    )
                ),
            )

        # Keep the original minimum-history convention.
        lo = max(lo, 2)

        if hi <= lo:
            raise ValueError(
                f"No backtest cases for {task} "
                f"between {start} and {end}"
            )

        # All models use exactly the same dates.
        if n_cases is None:
            idx = np.arange(lo, hi)
        else:
            rng = np.random.default_rng(seed)

            idx = np.sort(
                rng.choice(
                    np.arange(lo, hi),
                    size=min(n_cases, hi - lo),
                    replace=False,
                )
            )

        scores = {
            name: []
            for name in configs
        }

        for i in idx:

            steps = np.diff(
                vals[:i + 1],
                axis=0,
            )

            last = vals[i]

            y = np.stack(
                [vals[i + h] for h in hs],
                axis=-1,
            )

            # Common random numbers for fair comparison.
            base = components(
                sample_levels(
                    steps,
                    last,
                    hs,
                    n_draws,
                    int(i),
                    BASELINE,
                ),
                y,
            )

            recent_vol, long_vol, vol_ratio = (
                volatility_features(steps)
            )

            date = p.index[i]

            split = (
                "validation"
                if split_date
                and date >= pd.Timestamp(split_date)
                else "development"
            )

            for name, cfg in configs.items():

                actual_cfg = active_config(
                    task,
                    name,
                    cfg,
                    vol_ratio,
                )

                if actual_cfg == BASELINE:
                    c = base
                else:
                    c = components(
                        sample_levels(
                            steps,
                            last,
                            hs,
                            n_draws,
                            int(i),
                            actual_cfg,
                        ),
                        y,
                    )

                score = normalized(c, base)

                scores[name].append(score)

                records.append({
                    "task": task,
                    "date": date.strftime("%Y-%m-%d"),
                    "split": split,
                    "seed": seed,
                    "model": name,
                    "score": score,
                    "active_scale": actual_cfg.scale,
                    "active_corr": actual_cfg.corr,
                    "active_lam_corr": actual_cfg.lam_corr,
                    "baseline_crps": base[0],
                    "baseline_variogram": base[1],
                    "baseline_pinball": base[2],
                    "model_crps": c[0],
                    "model_variogram": c[1],
                    "model_pinball": c[2],
                    "recent_volatility": recent_vol,
                    "long_volatility": long_vol,
                    "vol_ratio_20_120": vol_ratio,
                })

        table[task] = {
            name: float(np.mean(values))
            for name, values in scores.items()
        }

        print(
            f"done {task} "
            f"({len(idx)} cases, seed={seed})",
            flush=True,
        )

    result = pd.DataFrame(table)

    result["MEAN"] = result.mean(axis=1)

    result = result.sort_values("MEAN")

    cases = pd.DataFrame.from_records(records)

    if output:
        destination = Path(output)

        destination.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        cases.to_csv(
            destination,
            index=False,
        )

        print(
            f"Saved {len(cases)} model-case records "
            f"to {destination}"
        )

    if split_date:
        print(
            f"\nTime split: development < {split_date}; "
            f"validation >= {split_date}"
        )

        for label in ("development", "validation"):

            subset = cases[
                cases["split"] == label
            ]

            if subset.empty:
                print(f"{label}: no cases")
                continue

            split_table = (
                subset.groupby(["model", "task"])["score"]
                .mean()
                .unstack()
            )

            split_table["MEAN"] = (
                split_table.mean(axis=1)
            )

            print(f"\n{label.upper()}")
            print(
                split_table.sort_values("MEAN").round(6)
            )

    if return_cases:
        return result, cases

    return result


def compare_v3_v4(cases):
    relevant = cases[
        cases["model"].isin(
            ["V3_original", "V4_scale0.9"]
        )
    ]

    if relevant["model"].nunique() < 2:
        return

    wide = relevant.pivot(
        index=[
            "task",
            "date",
            "split",
            "vol_ratio_20_120",
        ],
        columns="model",
        values="score",
    ).dropna(
        subset=["V3_original", "V4_scale0.9"]
    )

    if wide.empty:
        return

    wide = wide.reset_index()

    wide["improvement"] = (
        wide["V3_original"]
        - wide["V4_scale0.9"]
    )

    wide["v4_wins"] = (
        wide["improvement"] > 0
    )

    print(
        "\nV3 vs V4 "
        "(positive improvement means V4 is better):"
    )

    summary = wide.groupby("task").agg(
        v3_mean=("V3_original", "mean"),
        v4_mean=("V4_scale0.9", "mean"),
        mean_improvement=("improvement", "mean"),
        v4_win_rate=("v4_wins", "mean"),
        cases=("improvement", "size"),
    )

    print(summary.round(6).to_string())

    valid = wide[
        np.isfinite(wide["vol_ratio_20_120"])
    ].copy()

    if len(valid) >= 3:

        valid["vol_regime"] = pd.cut(
            valid["vol_ratio_20_120"],
            bins=[
                -np.inf,
                0.8,
                1.2,
                np.inf,
            ],
            labels=[
                "low",
                "normal",
                "high",
            ],
        )

        regime = valid.groupby(
            "vol_regime",
            observed=True,
        ).agg(
            mean_improvement=("improvement", "mean"),
            v4_win_rate=("v4_wins", "mean"),
            cases=("improvement", "size"),
        )

        print("\nVolatility regime diagnostics:")
        print(regime.round(6).to_string())

    print(
        "\nNote: overlapping horizons and nearby dates "
        "are not statistically independent."
    )


def compare_fx_correlations(cases):
    """Diagnostics for the FX-only correlation experiment."""

    fx = cases[
        cases["task"] == "fx_g10_1to4"
    ].copy()

    if fx.empty:
        return

    models = [
        "V4_scale0.9",
        "FX_Corr_A",
        "FX_Corr_B",
        "FX_Corr_C",
    ]

    fx = fx[
        fx["model"].isin(models)
    ].copy()

    if fx["model"].nunique() < 2:
        return

    for metric in (
        "crps",
        "variogram",
        "pinball",
    ):
        denominator = fx[f"baseline_{metric}"]

        fx[f"{metric}_ratio"] = np.where(
            denominator > 1e-12,
            fx[f"model_{metric}"] / denominator,
            np.nan,
        )

    columns = [
        "crps_ratio",
        "variogram_ratio",
        "pinball_ratio",
        "score",
    ]

    print("\n=== FX correlation component scores ===")
    print("Lower is better.")

    print(
        fx.groupby("model")[columns]
        .mean()
        .sort_values("score")
        .round(6)
        .to_string()
    )

    fx["year"] = pd.to_datetime(
        fx["date"]
    ).dt.year

    print("\n=== FX yearly scores ===")

    yearly = fx.pivot_table(
        index="year",
        columns="model",
        values="score",
        aggfunc="mean",
    )

    print(
        yearly.reindex(columns=models)
        .round(6)
        .to_string()
    )

    # Paired differences against fixed v4.
    wide = fx.pivot(
        index="date",
        columns="model",
        values="score",
    )

    if "V4_scale0.9" not in wide:
        return

    print("\n=== Paired improvements over fixed v4 ===")
    print("Positive means the candidate is better.")

    for name in models[1:]:

        if name not in wide:
            continue

        paired = wide[
            ["V4_scale0.9", name]
        ].dropna()

        improvement = (
            paired["V4_scale0.9"]
            - paired[name]
        )

        print(
            f"{name}: "
            f"mean={improvement.mean():+.6f}, "
            f"win_rate={(improvement > 0).mean():.2%}, "
            f"cases={len(improvement)}"
        )

    print(
        "\nCaution: yearly results are descriptive. "
        "Adjacent forecasts have overlapping horizons."
    )


def build_configs(a):
    A = BASELINE
    B = variant(A, vol="ewma")
    C = variant(B, dist="t")
    D = variant(C, path=True)
    E = variant(D, corr="ewma")

    configs = {
        "A_official": A,
        "B_+ewma_vol": B,
        "C_+fat_tails": C,
        "D_+path": D,
        "E_+ewma_corr": E,
    }

    if a.sweep:
        for lam in (0.90, 0.97, 0.99):
            configs[f"D_lam{lam}"] = variant(
                D,
                lam=lam,
            )

        for df_ in (4.0, 8.0):
            configs[f"D_df{df_:g}"] = variant(
                D,
                df=df_,
            )

        for scale in (0.9, 1.1):
            configs[f"D_scale{scale}"] = variant(
                D,
                scale=scale,
            )

    if a.blend:
        for blend in (0.7, 0.5, 0.3):
            configs[f"D_blend{blend}"] = variant(
                D,
                blend=blend,
            )

    if a.v3_sweep or a.compare_v3_v4:

        v3 = Config(
            vol="ewma",
            lam=0.94,
            corr="full",
            dist="t",
            df=5.0,
            path=True,
            scale=1.0,
            blend=0.3,
        )

        configs = {
            "V3_original": v3,
        }

        if a.v3_sweep:

            for blend in (0.1, 0.2, 0.4):
                configs[f"V4_blend{blend}"] = variant(
                    v3,
                    blend=blend,
                )

            for lam in (0.90, 0.97):
                configs[f"V4_lam{lam}"] = variant(
                    v3,
                    lam=lam,
                )

            for df_ in (4.0, 6.0, 8.0):
                configs[f"V4_df{df_:g}"] = variant(
                    v3,
                    df=df_,
                )

            for scale in (0.90, 0.95, 1.05):
                configs[f"V4_scale{scale}"] = variant(
                    v3,
                    scale=scale,
                )

        if a.compare_v3_v4:
            v4 = variant(
                v3,
                scale=0.9,
            )

            configs["V4_scale0.9"] = v4
            configs["Dynamic_A"] = v4
            configs["Dynamic_B"] = v4

    if a.fx_corr_sweep:

        v4 = Config(
            vol="ewma",
            lam=0.94,
            corr="full",
            dist="t",
            df=5.0,
            path=True,
            scale=0.9,
            blend=0.3,
        )

        configs = {
            "V4_scale0.9": v4,
            "FX_Corr_A": v4,
            "FX_Corr_B": v4,
            "FX_Corr_C": v4,
        }

    return configs


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--n",
        type=int,
        default=300,
    )

    ap.add_argument(
        "--sweep",
        action="store_true",
    )

    ap.add_argument(
        "--blend",
        action="store_true",
    )

    ap.add_argument(
        "--v3-sweep",
        action="store_true",
    )

    ap.add_argument(
        "--compare-v3-v4",
        action="store_true",
    )

    ap.add_argument(
        "--fx-corr-sweep",
        action="store_true",
        help="Compare full vs EWMA FX correlations",
    )

    ap.add_argument(
        "--all-dates",
        action="store_true",
        help="Evaluate every eligible date",
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for sampling dates",
    )

    ap.add_argument(
        "--start",
        default="2012-01-01",
    )

    ap.add_argument(
        "--end",
        default=None,
    )

    ap.add_argument(
        "--split-date",
        default=None,
        help="First validation date",
    )

    ap.add_argument(
        "--output",
        default=None,
        help="Save per-case CSV",
    )

    ap.add_argument(
        "--draws",
        type=int,
        default=500,
    )

    a = ap.parse_args()

    if not a.all_dates and a.n <= 0:
        ap.error("--n must be positive")

    if a.draws <= 0:
        ap.error("--draws must be positive")

    if (
        a.split_date is not None
        and pd.Timestamp(a.split_date)
        <= pd.Timestamp(a.start)
    ):
        ap.error(
            "--split-date must be later than --start"
        )

    if (
        a.end is not None
        and pd.Timestamp(a.end)
        < pd.Timestamp(a.start)
    ):
        ap.error(
            "--end must not precede --start"
        )

    pd.set_option(
        "display.width",
        180,
    )

    configs = build_configs(a)

    n_cases = (
        None if a.all_dates else a.n
    )

    result, cases = run(
        configs,
        n_cases,
        start=a.start,
        end=a.end,
        n_draws=a.draws,
        output=a.output,
        split_date=a.split_date,
        return_cases=True,
        seed=a.seed,
    )

    print("\n=== Overall scores (lower is better) ===")
    print(result.round(6).to_string())

    if a.compare_v3_v4 or a.v3_sweep:
        compare_v3_v4(cases)

    if a.fx_corr_sweep:
        compare_fx_correlations(cases)


if __name__ == "__main__":
    main()