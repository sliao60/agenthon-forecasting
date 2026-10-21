#!/usr/bin/env python3
"""Run the Docker image on every unit, check admissibility, record timing.
Usage: python run_all.py [image]   (default t2-agent:v1)
Outputs: runs/<image>/<unit>/ holds ONLY what the container wrote;
         logs and score reports go to runs/<image>_logs/."""
import json, shutil, subprocess, sys, time, tomllib
from pathlib import Path

IMAGE = sys.argv[1] if len(sys.argv) > 1 else "t2-agent:v1"
ROOT = Path(__file__).resolve().parent
UNITS = sorted(p for p in (ROOT / "units").iterdir() if (p / "card.toml").exists())
TAG = IMAGE.replace("/", "_").replace(":", "_")
RUNS = ROOT / "runs" / TAG
LOGS = ROOT / "runs" / f"{TAG}_logs"
LOGS.mkdir(parents=True, exist_ok=True)

def asof_of(card):
    for sec, key in (("provenance", "data_cutoff"), ("text", "cutoff")):
        v = card.get(sec, {}).get(key)
        if v:
            return str(v)
    raise KeyError("no as-of date in card")

results = []
for i, u in enumerate(UNITS, 1):
    out = RUNS / u.name
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    rec = {"unit": u.name, "status": "run_failed", "seconds": None, "gates": None}
    try:
        asof = asof_of(tomllib.loads((u / "card.toml").read_text()))
    except Exception as e:
        rec["status"] = f"card_error: {e}"
        results.append(rec); print(f"[{i}/{len(UNITS)}] {u.name}: {rec['status']}"); continue
    cmd = ["docker", "run", "--rm", "--network=none", "--platform", "linux/amd64",
           "-v", f"{u}:/input:ro", "-v", f"{out}:/output", IMAGE,
           "forecast", "--panels", "/input/panels", "--text", "/input/text",
           "--asof", asof, "--out", "/output/forecast.parquet"]
    t0 = time.time()
    r = subprocess.run(cmd, capture_output=True, text=True)
    rec["seconds"] = round(time.time() - t0, 1)
    (LOGS / f"{u.name}.docker.log").write_text(r.stdout + r.stderr)
    if r.returncode == 0 and (out / "forecast.parquet").exists():
        s = subprocess.run([sys.executable, str(ROOT / "scoring" / "scoring.py"), "score",
                            "--card", str(u / "card.toml"),
                            "--forecast", str(out / "forecast.parquet")],
                           capture_output=True, text=True)
        (LOGS / f"{u.name}.score.json").write_text(s.stdout + s.stderr)
        try:
            rep = json.loads(s.stdout)
            rec["gates"] = rep.get("gates")
            rec["status"] = "ok" if rep.get("admissible") else "inadmissible"
        except json.JSONDecodeError:
            rec["status"] = "score_error"
    results.append(rec)
    print(f"[{i}/{len(UNITS)}] {u.name}: {rec['status']} ({rec['seconds']}s)", flush=True)

(LOGS / "summary.json").write_text(json.dumps(results, indent=2))
ok = [r for r in results if r["status"] == "ok"]
bad = [r for r in results if r["status"] != "ok"]
secs = [r["seconds"] for r in results if r["seconds"] is not None]
print(f"\n{len(ok)}/{len(results)} admissible")
if secs:
    print(f"time: total {sum(secs):.0f}s, max {max(secs):.0f}s, mean {sum(secs)/len(secs):.1f}s")
for r in bad:
    print(f"  FAIL {r['unit']}: {r['status']}  -> runs/{TAG}_logs/{r['unit']}.docker.log / .score.json")
sys.exit(0 if not bad else 1)
