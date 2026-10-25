
#!/usr/bin/env python3
"""Backtest: compare samplers on 1-4 business-day UST and G10 FX forecasts.

Each case uses only data up to an as-of date and is scored against the realized
values with the official Track 2 components. Scores are normalized by the
official baseline on the same case (1.0 = baseline; lower is better).

Usage:
    python backtest.py --n 300
    python backtest.py --n 300 --blend
    python backtest.py --n 300 --sweep
    python backtest.py --n 300 --v3-sweep
"""

import argparse

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


def load(path):
    df = pd.read_parquet(path)
    col = next(
        c for c in ("asset", "asset_id", "series")
        if c in df.columns
    )
    df["date"] = pd.to_datetime(df["date"].astype(str).str[:10])

    return df.pivot_table(
        index="date",
        columns=col,
        values="value",
    ).sort_index()


def components(samples, y):
    s = samples.reshape(samples.shape[0], -1)
    yy = y.reshape(-1)

    m = float(crps.crps_marginal(s, yy))
    t = float(TAIL_METRICS["pinball"](s, yy, TAIL))

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


def run(
    configs,
    n_cases,
    start="2012-01-01",
    end=None,
    n_draws=500,
):
    panels = {
        k: load(v)
        for k, v in PANELS.items()
    }

    table = {}

    for task, (panel, assets, hs) in TASKS.items():
        p = panels[panel][assets].dropna()
        vals = p.to_numpy()

        lo = int(
            np.searchsorted(
                p.index.values,
                np.datetime64(start),
            )
        )

        hi = len(p) - max(hs) - 1

        if end is not None:
            hi = min(
                hi,
                int(
                    np.searchsorted(
                        p.index.values,
                        np.datetime64(end),
                    )
                ),
            )

        if hi <= lo:
            raise ValueError(
                f"No backtest cases for {task} "
                f"between {start} and {end}"
            )

        rng = np.random.default_rng(0)

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
            steps = np.diff(vals[: i + 1], axis=0)
            last = vals[i]

            y = np.stack(
                [vals[i + h] for h in hs],
                axis=-1,
            )

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

            for name, cfg in configs.items():
                if cfg == BASELINE:
                    c = base
                else:
                    c = components(
                        sample_levels(
                            steps,
                            last,
                            hs,
                            n_draws,
                            int(i),
                            cfg,
                        ),
                        y,
                    )

                scores[name].append(
                    normalized(c, base)
                )

        table[task] = {
            k: float(np.mean(v))
            for k, v in scores.items()
        }

        print(
            f"done {task} ({len(idx)} cases)",
            flush=True,
        )

    df = pd.DataFrame(table)
    df["MEAN"] = df.mean(axis=1)

    return df.sort_values("MEAN")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--blend", action="store_true")
    ap.add_argument("--v3-sweep", action="store_true")
    ap.add_argument("--start", default="2012-01-01")
    ap.add_argument("--end", default=None)

    a = ap.parse_args()

    if a.n <= 0:
        ap.error("--n must be positive")

    # --------------------------------------------------
    # Original baseline and Tong's model variants
    # --------------------------------------------------

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

    # --------------------------------------------------
    # Original parameter sweep
    # --------------------------------------------------

    if a.sweep:
        for lam in (0.90, 0.97, 0.99):
            configs[f"D_lam{lam}"] = variant(
                D, lam=lam
            )

        for df_ in (4.0, 8.0):
            configs[f"D_df{df_:g}"] = variant(
                D, df=df_
            )

        for s in (0.9, 1.1):
            configs[f"D_scale{s}"] = variant(
                D, scale=s
            )

    # --------------------------------------------------
    # Original blend sweep
    # --------------------------------------------------

    if a.blend:
        for w in (0.7, 0.5, 0.3):
            configs[f"D_blend{w}"] = variant(
                D, blend=w
            )

    # --------------------------------------------------
    # NEW: v3-based parameter sweep
    # --------------------------------------------------

    if a.v3_sweep:
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

        # Use v3 as the reference for all new variants.
        configs = {
            "V3_original": v3,
        }

        # Experiment 1: EWMA variance blending
        for blend in (0.1, 0.2, 0.4):
            configs[f"V4_blend{blend}"] = variant(
                v3,
                blend=blend,
            )

        # Experiment 2: EWMA decay
        for lam in (0.90, 0.97):
            configs[f"V4_lam{lam}"] = variant(
                v3,
                lam=lam,
            )

        # Experiment 3: Student-t degrees of freedom
        for df_ in (4.0, 6.0, 8.0):
            configs[f"V4_df{df_:g}"] = variant(
                v3,
                df=df_,
            )

        # Experiment 4: Forecast volatility scale
        for scale in (0.90, 0.95, 1.05):
            configs[f"V4_scale{scale}"] = variant(
                v3,
                scale=scale,
            )

    pd.set_option("display.width", 160)

    results = run(
        configs,
        a.n,
        a.start,
        a.end,
    )

    print(results.round(4))


if __name__ == "__main__":
    main()