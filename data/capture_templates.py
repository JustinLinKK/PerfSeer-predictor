#!/usr/bin/env python3
"""Release-building utility: capture label-era executable templates, not data rows.

All 6000 historical model configurations must share exactly one template per
family, after replacing only literal workload parameters and the header ID.
The resulting templates are ordinary executable-source specifications consumed
by the standalone generator, which never needs this historical source directory.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
from generate import HERE, sha

LITERALS = ("MODEL_ID", "PROFILE_POINT_ID", "MODEL_CONFIG", "BATCH_SIZE", "PRECISION_CONFIG", "OPTIMIZER_NAME", "INPUT_SPECS")


def template(text):
    for name in LITERALS:
        text, count = re.subn(r"^" + name + r" = .*$", name + " = @" + name + "@", text, flags=re.M)
        if count != 1:
            raise ValueError("expected one workload literal: " + name)
    return re.sub(r"architecture for model_\d+_\d+", "architecture for @MODEL_ID_TEXT@", text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant-root", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(x) for x in (args.variant_root / "model_config.jsonl").open()]
    templates, checked, seen = {}, Counter(), set()
    for row in rows:
        # Hash every variant; structural template check is per source config.
        path = args.variant_root / row["model_file"]
        if sha(path) != row["source_sha256"]:
            raise ValueError("historical variant hash mismatch")
        text = template(path.read_text())
        family = row["model_id"].split("_")[1]
        if family in templates and templates[family] != text:
            raise ValueError("multiple templates within a family: " + family)
        templates[family] = text
        checked[family] += 1
    out = HERE / "specs/variant_templates"
    out.mkdir(parents=True, exist_ok=True)
    for family, text in templates.items():
        (out / (family + ".py.in")).write_text(text)
    receipt = {"passed": True, "original_variant_config_sha256": sha(args.variant_root / "model_config.jsonl"),
               "variant_sources_checked": sum(checked.values()), "per_family": dict(checked),
               "templates": {p.name: sha(p) for p in out.iterdir()},
               "lineage_note": "A10/A100 standalone variant templates differ from original A10G shared runtime. Both lineages are emitted and hash-verified separately."}
    (HERE / "receipts/variant_templates.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"passed": True, "variants_checked": sum(checked.values()), "templates": len(templates)}))


if __name__ == "__main__":
    main()
