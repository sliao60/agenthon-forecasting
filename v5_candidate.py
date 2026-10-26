"""Experimental V5 forecasting candidate.

Keeps official V4 unchanged.
Applies multivariate OT coupling only to multi-horizon log returns.
"""

import numpy as np

from qfbench2_track_forecasting.cli import _draw, _series
from qfbench2_track_forecasting.targets import log_return_steps
from test_multivariate_ot import multivariate_ot_candidate


def draw_v5(
    panels,
    assets,
    horizons,
    asof,
    n_draws,
    seed,
    *,
    target_type="level",
    panel_steps=None,
):
    """Return (samples, info) using V4 plus optional OT coupling."""

    samples, info = _draw(
        panels,
        assets,
        horizons,
        asof,
        n_draws,
        seed,
        target_type=target_type,
        panel_steps=panel_steps,
    )

    if target_type != "log_return" or len(horizons) <= 1:
        return samples, info

    # Match V4's asset-level history extraction and date alignment.
    import pandas as pd

    hist = {
        asset: _series(panels, asset, asof)
        for asset in assets
    }

    steps = pd.DataFrame({
        asset: pd.Series(
            log_return_steps(hist[asset]),
            index=hist[asset].index,
        )
        for asset in assets
    }).dropna()

    if len(steps) < 30:
        raise ValueError("Insufficient aligned historical observations")

    sd = steps.std().to_numpy(dtype=float)
    corr = steps.corr().to_numpy(dtype=float)

    corr = np.nan_to_num(corr, nan=0.0)
    np.fill_diagonal(corr, 1.0)

    eigenvalues, eigenvectors = np.linalg.eigh(corr)
    corr = (
        eigenvectors
        @ np.diag(np.clip(eigenvalues, 1e-8, None))
        @ eigenvectors.T
    )

    cov = np.diag(sd) @ corr @ np.diag(sd)

    coupled = multivariate_ot_candidate(
        samples,
        cov,
        horizons,
        int(seed) + 5000000,
    )

    if coupled.shape != samples.shape:
        raise RuntimeError("OT changed output shape")

    if not np.isfinite(coupled).all():
        raise RuntimeError("OT produced nonfinite samples")

    result_info = dict(info)
    result_info["candidate"] = "v5_ot"
    result_info["coupling"] = "multivariate_ot"

    return coupled, result_info
