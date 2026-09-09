#!/usr/bin/env python3
"""Execute and optionally write the complete local Phase 3 golden matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from perfseer_v3.dataset_pack.model_golden import run_model_factory_audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--optimization-steps", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    audit = run_model_factory_audit(optimization_steps=args.optimization_steps)
    payload = audit.to_dict()
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(
        json.dumps(
            {
                "audit_sha256": audit.sha256,
                "model_count": len(audit.model_results),
                "adapter_count": len(audit.adapter_results),
                "specialized_step_count": len(audit.specialized_step_results),
                "operation_cell_count": len(audit.observed_operation_cells),
                "mixed_precision_verified_count": sum(
                    result.mixed_precision_status == "verified"
                    for result in audit.model_results
                ),
                "mixed_precision_unsupported_local_count": sum(
                    result.mixed_precision_status == "unsupported_local_cpu"
                    for result in audit.model_results
                ),
                "training_approved": audit.training_approved,
                "output": str(args.output) if args.output else None,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
