"""End-to-end V5 CLI smoke test for three multi-horizon cards."""

import subprocess
import sys
import tomllib
from pathlib import Path

CARDS = [
    "t2-F3-cpi-release-cross-2022",
    "t2-F3-rates-into-factors-2022",
    "t2-F3-readout-calendar-factors-2020",
]

ROOT = Path.cwd()
OUTPUT_ROOT = ROOT / "v5_e2e_outputs"

for name in CARDS:
    folder = ROOT / "units" / name

    with (folder / "card.toml").open("rb") as f:
        card = tomllib.load(f)

    asof = str(card["provenance"]["data_cutoff"])
    output_dir = OUTPUT_ROOT / name
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "forecast.parquet"

    command = [
        sys.executable,
        "-m",
        "v5_cli",
        "--panels",
        str(folder),
        "--text",
        str(folder / "text"),
        "--asof",
        asof,
        "--card",
        str(folder / "card.toml"),
        "--out",
        str(output_file),
        "--n-draws",
        "500",
        "--seed",
        "2026",
    ]

    print(f"\n=== {name} ===", flush=True)

    subprocess.run(command, check=True)

    if not output_file.exists():
        raise RuntimeError(f"Missing output: {output_file}")

    print(f"PASS: {output_file}", flush=True)

print("\nALL THREE V5 CLI RUNS PASSED")
