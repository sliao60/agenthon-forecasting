from pathlib import Path
import tomllib

import numpy as np

from qfbench2_track_forecasting.cli import _draw, _read_panels
from v5_candidate import draw_v5


N_DRAWS = 500
SEED = 2026

counts = {
    "level": 0,
    "single_log_return": 0,
    "multi_log_return": 0,
}

failures = []

for folder in sorted(Path("units").iterdir()):
    card_file = folder / "card.toml"

    if not card_file.exists():
        continue

    try:
        with card_file.open("rb") as f:
            card = tomllib.load(f)

        targets = card["targets"]
        assets = list(targets["asset_ids"])
        horizons = list(map(int, targets["horizons"]))
        target_type = targets["target_type"]
        asof = str(card["provenance"]["data_cutoff"])

        panels = _read_panels(folder)

        kwargs = dict(
            panels=panels,
            assets=assets,
            horizons=horizons,
            asof=asof,
            n_draws=N_DRAWS,
            seed=SEED,
            target_type=target_type,
        )

        original, original_info = _draw(**kwargs)
        candidate, candidate_info = draw_v5(**kwargs)

        assert original.shape == candidate.shape
        assert np.isfinite(candidate).all()

        if target_type == "level":
            np.testing.assert_array_equal(
                original, candidate
            )
            counts["level"] += 1

        elif len(horizons) == 1:
            np.testing.assert_array_equal(
                original, candidate
            )
            counts["single_log_return"] += 1

        else:
            np.testing.assert_array_equal(
                np.sort(original, axis=0),
                np.sort(candidate, axis=0),
            )

            for hi in range(len(horizons)):
                original_rows = sorted(
                    map(tuple, original[:, :, hi])
                )
                candidate_rows = sorted(
                    map(tuple, candidate[:, :, hi])
                )
                assert original_rows == candidate_rows

            counts["multi_log_return"] += 1

        print(f"PASS {folder.name}")

    except Exception as exc:
        failures.append((folder.name, repr(exc)))
        print(f"FAIL {folder.name}: {exc}")

print("\n=== Summary ===")
print("Level:", counts["level"])
print("Single-horizon log-return:", counts["single_log_return"])
print("Multi-horizon log-return:", counts["multi_log_return"])
print("Failures:", len(failures))

for name, error in failures:
    print(" ", name, error)

assert counts == {
    "level": 110,
    "single_log_return": 13,
    "multi_log_return": 3,
}, "Unexpected card coverage"

assert not failures, "Regression test failed"

print("\nALL 126 CARDS PASSED")
