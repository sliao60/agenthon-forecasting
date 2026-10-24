"""Improved Track 2 sampler: EWMA volatility/correlation, Student-t shocks, cumulative paths.

sample_levels() takes a history of one-step changes and the current level, and returns
draws of the level at each requested horizon. Every improvement is a switch on Config, and
BASELINE reproduces the official reference sampler (full-history Gaussian, sqrt(h) scaling,
an independent shock per horizon) so the two can be compared on equal terms.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class Config:
    vol: str = "ewma"       # "full" | "ewma"  -- how per-asset volatility is estimated
    lam: float = 0.94       # EWMA decay for volatility (closer to 1 = longer memory)
    corr: str = "ewma"      # "full" | "ewma"  -- how cross-asset correlation is estimated
    lam_corr: float = 0.97  # EWMA decay for correlation
    dist: str = "t"         # "normal" | "t"   -- shape of the daily shock
    df: float = 5.0         # Student-t degrees of freedom (smaller = fatter tails)
    path: bool = True       # True: cumulative daily path; False: independent draw per horizon
    scale: float = 1.0      # overall width multiplier
    window: int = 1500      # most recent rows used by the EWMA estimates
    blend: float = 1.0      # weight on EWMA variance (1.0 = pure EWMA, 0.0 = full history)


BASELINE = Config(vol="full", corr="full", dist="normal", path=False, scale=1.0)


def variant(base: Config = Config(), **kw) -> Config:
    return replace(base, **kw)


def _ewma_cov(x: np.ndarray, lam: float) -> np.ndarray:
    n = len(x)
    w = lam ** np.arange(n - 1, -1, -1, dtype=float)
    w /= w.sum()
    return (x * w[:, None]).T @ x


def _to_corr(c: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.clip(np.diag(c), 1e-300, None))
    r = c / np.outer(d, d)
    r = np.nan_to_num(r, nan=0.0)
    np.fill_diagonal(r, 1.0)
    w, v = np.linalg.eigh(r)
    r = v @ np.diag(np.clip(w, 1e-8, None)) @ v.T
    d = np.sqrt(np.diag(r))
    return r / np.outer(d, d)


def _shocks(rng: np.random.Generator, n: int, k: int, cfg: Config) -> np.ndarray:
    if cfg.dist == "t":
        return rng.standard_t(cfg.df, size=(n, k)) * np.sqrt((cfg.df - 2.0) / cfg.df)
    return rng.standard_normal((n, k))


def sample_levels(
    steps: np.ndarray,
    last: np.ndarray,
    horizons: list[int],
    n_draws: int,
    seed: int,
    cfg: Config = Config(),
) -> np.ndarray:
    """steps: (T, A) one-step changes, no NaN. last: (A,) current level.
    Returns (n_draws, A, H): the level at each horizon (in steps)."""
    steps = np.asarray(steps, dtype=float)
    last = np.asarray(last, dtype=float)
    a = steps.shape[1]
    rng = np.random.default_rng(seed)

    full_sd = steps.std(axis=0, ddof=1)
    recent = steps[-cfg.window:]
    if cfg.vol == "ewma":
        sd = np.sqrt(np.diag(_ewma_cov(recent, cfg.lam)))
        sd = np.where(np.isfinite(sd) & (sd > 0), sd, full_sd)
        sd = np.sqrt(cfg.blend * sd**2 + (1.0 - cfg.blend) * full_sd**2)
    else:
        sd = full_sd
    sd = sd * cfg.scale

    if a == 1:
        corr = np.ones((1, 1))
    elif cfg.corr == "ewma":
        corr = _to_corr(_ewma_cov(recent, cfg.lam_corr))
    else:
        corr = _to_corr(np.atleast_2d(np.cov(steps, rowvar=False)))
    chol = np.linalg.cholesky(corr)

    out = np.empty((n_draws, a, len(horizons)), dtype=float)
    if cfg.path:
        cum = np.zeros((n_draws, a))
        for day in range(1, max(horizons) + 1):
            cum += (_shocks(rng, n_draws, a, cfg) @ chol.T) * sd
            for hi, h in enumerate(horizons):
                if h == day:
                    out[:, :, hi] = last + cum
    else:
        for hi, h in enumerate(horizons):
            z = _shocks(rng, n_draws, a, cfg) @ chol.T
            out[:, :, hi] = last + z * sd * np.sqrt(h)
    return out