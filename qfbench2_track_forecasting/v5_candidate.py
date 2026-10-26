"""Experimental Hybrid B configuration selector.

Only exact matches to backtested asset/horizon grids
receive task-specific Student-t degrees of freedom.

All other tasks retain the official V4 configuration.
"""

from dataclasses import replace

from .improved import Config


V4_CONFIG = Config(
    vol="ewma",
    lam=0.94,
    corr="full",
    dist="t",
    df=5.0,
    path=True,
    scale=0.9,
    blend=0.3,
)


BACKTESTED_TASKS = {
    (
        ("EUR",),
        (4,),
    ): 3.0,

    (
        ("UST_10Y",),
        (1,),
    ): 4.0,

    (
        ("UST_2Y", "UST_5Y", "UST_10Y", "UST_30Y"),
        (1, 2, 3, 4),
    ): 4.0,

    (
        ("EUR", "GBP", "JPY", "CHF", "AUD", "CAD"),
        (1, 2, 3, 4),
    ): 5.0,
}


def config_for_task(
    assets,
    horizons,
    target_type="level",
    target_frequency="daily",
):
    """Select Hybrid B only for exact backtested tasks."""

    if target_type != "level":
        return V4_CONFIG

    if target_frequency != "daily":
        return V4_CONFIG

    key = (
        tuple(assets),
        tuple(int(h) for h in horizons),
    )

    df = BACKTESTED_TASKS.get(key, 5.0)

    return replace(V4_CONFIG, df=df)
