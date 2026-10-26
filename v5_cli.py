"""Standalone experimental V5 CLI.

Reuses the official CLI's argument parsing, input handling,
output formatting, and metadata logic without editing cli.py.
"""

from qfbench2_track_forecasting import cli
from v5_candidate import draw_v5


def main(argv=None):
    original_draw = cli._draw

    try:
        cli._draw = draw_v5
        return cli.main(argv)
    finally:
        cli._draw = original_draw


if __name__ == "__main__":
    raise SystemExit(main())
