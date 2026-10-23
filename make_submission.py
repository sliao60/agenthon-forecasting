#!/usr/bin/env python3
"""Write a Track 2 Final submission.json from the toolkit's own fixture.
Usage: python make_submission.py <dockerhub-user>/<repo> sha256:<64hex>"""
import json, re, sys, pathlib, qfbench2_common
from qfbench2_common.contracts.descriptor import seal_descriptor_digest

repo, digest = sys.argv[1], sys.argv[2]
assert re.fullmatch(r"sha256:[0-9a-f]{64}", digest), "digest must look like sha256:<64 hex>"
fx = pathlib.Path(qfbench2_common.__file__).parent / "contracts/fixtures/c5/forecasting_final.json"
d = json.loads(fx.read_text())
d.pop("descriptor_digest", None)
d.update({
    "schema_version": "1.1.0",
    "image": {"registry": "docker.io", "repository": repo, "digest": digest},
    "image_access": "public",
    "models": [],                           # reference forecaster calls no model
    "license": "MIT",
})
d.pop("team_id", None)                  # pack derives and fills team_id
d = seal_descriptor_digest(d)
pathlib.Path("submission.json").write_text(json.dumps(d, indent=2) + "\n")
print(json.dumps(d, indent=2))
