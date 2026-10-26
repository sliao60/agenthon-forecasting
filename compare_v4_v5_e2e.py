import json
import subprocess
import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

CARDS = [
    "t2-F3-cpi-release-cross-2022",
    "t2-F3-rates-into-factors-2022",
    "t2-F3-readout-calendar-factors-2020",
]

ROOT = Path.cwd()
V4_ROOT = ROOT / "v4_e2e_outputs"
V5_ROOT = ROOT / "v5_e2e_outputs"

for name in CARDS:
    folder = ROOT / "units" / name

    with (folder / "card.toml").open("rb") as f:
        card = tomllib.load(f)

    assets = list(card["targets"]["asset_ids"])
    horizons = list(map(int, card["targets"]["horizons"]))
    asof = str(card["provenance"]["data_cutoff"])

    out_dir = V4_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable, "-m", "qfbench2_track_forecasting.cli",
        "--panels", str(folder),
        "--text", str(folder / "text"),
        "--asof", asof,
        "--card", str(folder / "card.toml"),
        "--out", str(out_dir / "forecast.parquet"),
        "--n-draws", "500",
        "--seed", "2026",
    ]

    subprocess.run(command, check=True, capture_output=True, text=True)

    v4 = pd.read_parquet(out_dir / "forecast.parquet")
    v5 = pd.read_parquet(V5_ROOT / name / "forecast.parquet")

    assert list(v4.columns) == list(v5.columns)
    assert v4.shape == v5.shape

    key = ["draw", "asset", "horizon"]

    assert not v4.duplicated(key).any()
    assert not v5.duplicated(key).any()

    a = v4.set_index(key).sort_index()
    b = v5.set_index(key).sort_index()

    assert a.index.equals(b.index)
    assert np.isfinite(a["value"].to_numpy()).all()
    assert np.isfinite(b["value"].to_numpy()).all()

    for asset in assets:
        for h in horizons:
            av = v4.loc[
                (v4["asset"] == asset) &
                (v4["horizon"] == h),
                "value",
            ].to_numpy()

            bv = v5.loc[
                (v5["asset"] == asset) &
                (v5["horizon"] == h),
                "value",
            ].to_numpy()

            np.testing.assert_array_equal(
                np.sort(av), np.sort(bv)
            )

    for h in horizons:
        av = (
            v4[v4["horizon"] == h]
            .pivot(index="draw", columns="asset", values="value")
            .reindex(columns=assets)
            .to_numpy()
        )

        bv = (
            v5[v5["horizon"] == h]
            .pivot(index="draw", columns="asset", values="value")
            .reindex(columns=assets)
            .to_numpy()
        )

        assert sorted(map(tuple, av)) == sorted(map(tuple, bv))

    changed = int(
        np.count_nonzero(
            a["value"].to_numpy() != b["value"].to_numpy()
        )
    )

    assert changed > 0, "Expected OT to change cross-horizon pairing"

    print(f"\n{name}")
    print("  Output schema: PASS")
    print("  Marginal distributions: PASS")
    print("  Same-horizon joint samples: PASS")
    print("  OT changed values by draw:", changed)

print("\nALL V4/V5 END-TO-END COMPARISONS PASSED")
