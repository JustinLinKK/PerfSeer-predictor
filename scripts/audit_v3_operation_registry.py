#!/usr/bin/env python3
"""Execute the registry-derived local operation/composite golden audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from perfseer_v3.dataset_pack.composite_blocks import (  # noqa: E402
    build_composite_block,
    verify_composite_block,
)
from perfseer_v3.dataset_pack.composite_sampler import (  # noqa: E402
    build_composite_block_registry,
    plan_composite_corpus,
)
from perfseer_v3.dataset_pack.operation_benchmarks import (  # noqa: E402
    build_operation_benchmark,
    run_p1_fixture_suite,
    verify_operation_benchmark,
)
from perfseer_v3.dataset_pack.operation_sampler import (  # noqa: E402
    build_operation_generator_registry,
    plan_operation_corpus,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="optional JSON audit output")
    parser.add_argument(
        "--skip-composites",
        action="store_true",
        help="run operation/P1 goldens without the slower 45-block audit",
    )
    args = parser.parse_args()

    generators = build_operation_generator_registry()
    operation_plan = plan_operation_corpus(generator_registry=generators)
    float32_by_generator = {}
    for candidate in operation_plan.candidates:
        if candidate.dtype == "float32":
            float32_by_generator.setdefault(candidate.generator_id, candidate)

    verified_operations = []
    gated_operations = []
    for generator in generators.generators:
        candidate = float32_by_generator[generator.generator_id]
        result = verify_operation_benchmark(
            build_operation_benchmark(generator, candidate),
            generators=generators,
        )
        if result.status == "verified":
            verified_operations.append(generator.canonical_operation_id)
        elif result.status == "environment_unsupported":
            gated_operations.append(generator.canonical_operation_id)
        else:
            raise SystemExit(
                f"unexpected operation verification state: {generator.canonical_operation_id}={result.status}"
            )

    block_registry = build_composite_block_registry()
    verified_blocks = []
    if not args.skip_composites:
        composite_plan = plan_composite_corpus(registry=block_registry)
        candidate_by_block = {}
        for candidate in composite_plan.candidates:
            if candidate.dtype == "float32":
                candidate_by_block.setdefault(candidate.block_id, candidate)
        for block in block_registry.blocks:
            result = verify_composite_block(
                build_composite_block(
                    block,
                    candidate_by_block[block.block_id],
                    generators=generators,
                ),
                generators=generators,
            )
            covered = set(result.verified_operation_ids) | set(
                result.environment_unsupported_operation_ids
            )
            if covered != set(block.canonical_operation_ids):
                raise SystemExit(f"composite block failed exact coverage: {block.block_id}")
            verified_blocks.append(block.block_id)

    p1 = run_p1_fixture_suite()
    report = {
        "version": "perfseer_v3_a10g_local_operation_audit_v1",
        "target_hardware_id": "local_cpu_structural_and_identity_smoke",
        "training_approved": False,
        "generator_registry_sha256": generators.sha256,
        "composite_registry_sha256": block_registry.sha256,
        "verified_operation_count": len(verified_operations),
        "environment_gated_operation_count": len(gated_operations),
        "verified_composite_block_count": len(verified_blocks),
        "verified_operation_ids": verified_operations,
        "environment_gated_operation_ids": gated_operations,
        "verified_composite_block_ids": verified_blocks,
        "p1_fixtures": [row.__dict__ for row in p1],
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
