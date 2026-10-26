from pathlib import Path
import tomllib
import numpy as np

from qfbench2_track_forecasting.cli import _draw, _read_panels
from smoke_test_log_return_path import gaussian_path_candidate
from test_multivariate_ot import multivariate_ot_candidate

N_DRAWS = 500
SEED = 2026

root = Path("units")
results = []

for folder in sorted(root.iterdir()):
    card_file = folder / "card.toml"
    if not card_file.exists():
        continue

    with card_file.open("rb") as f:
        card = tomllib.load(f)

    targets = card.get("targets", {})
    if targets.get("target_type") != "log_return":
        continue

    assets = list(targets["asset_ids"])
    horizons = list(map(int, targets["horizons"]))
    asof = str(card["provenance"]["data_cutoff"])

    panels = _read_panels(folder)

    independent, info = _draw(
        panels,
        assets,
        horizons,
        asof,
        N_DRAWS,
        SEED,
        target_type="log_return",
    )

    # The CLI's log-return covariance is estimated from
    # its full aligned historical return matrix.
    from qfbench2_track_forecasting.targets import log_return_steps
    import pandas as pd

    frame = pd.read_parquet(folder / "factors_daily.parquet")
    wide = (
        frame[frame["asset"].isin(assets)]
        .pivot(index="date", columns="asset", values="value")
        .sort_index()
        .reindex(columns=assets)
        .dropna()
    )

    dates = pd.to_datetime(wide.index)
    mask = dates <= pd.Timestamp(asof)
    wide = wide.loc[mask]

    steps = log_return_steps(wide.to_numpy())
    cov = np.atleast_2d(
        np.cov(steps, rowvar=False, ddof=1)
    )

    if len(horizons) == 1:
        candidate = independent.copy()
        method = "unchanged"
    else:
        candidate = multivariate_ot_candidate(
            independent, cov, horizons, SEED + 5000000
        )
        method = "multivariate_ot"

    assert candidate.shape == independent.shape
    assert np.isfinite(candidate).all()

    # Exact marginal preservation.
    np.testing.assert_array_equal(
        np.sort(candidate, axis=0),
        np.sort(independent, axis=0),
    )

    # Preserve each horizon's complete multivariate rows.
    for hi in range(len(horizons)):
        original_rows = sorted(
            map(tuple, independent[:, :, hi])
        )
        candidate_rows = sorted(
            map(tuple, candidate[:, :, hi])
        )
        assert original_rows == candidate_rows

    results.append((folder.name, method, "PASS"))
    print(f"{folder.name}: {method} PASS")

print(f"\nPassed: {len(results)}/{len(results)}")
assert len(results) == 16, (
    f"Expected 16 log-return cards, found {len(results)}"
)
