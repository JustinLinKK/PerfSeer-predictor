from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from perfseer_v3.capture_export import capture_export
from perfseer_v3.splits import (
    EVALUATION_SLICE_NAMES,
    evaluation_slice_manifest_payload,
    grouped_split,
    split_manifest_payload,
    validate_group_isolation,
)
from perfseer_v3.workloads import (
    SHAPE_REGIMES,
    WorkloadDescriptor,
    build_workload,
    default_composites,
    default_microbenchmarks,
    default_source_workloads,
    manifest_payload,
    select_coverage_gaps,
    validate_declared_operations,
)


class WorkloadTests(unittest.TestCase):
    def test_bootstrap_exact_operations_have_five_shape_regimes(self) -> None:
        descriptors = default_microbenchmarks()
        by_operation: dict[str, dict[str, set]] = {}
        for descriptor in descriptors:
            operation = descriptor.declared_operations[0]
            matrix = by_operation.setdefault(
                operation,
                {
                    "shape": set(),
                    "dtype": set(),
                    "phase": set(),
                    "batch": set(),
                    "optimizer": set(),
                },
            )
            matrix["shape"].add(descriptor.shape_regime)
            matrix["dtype"].add(descriptor.dtype)
            matrix["phase"].add(descriptor.phase)
            matrix["batch"].add(descriptor.batch_size)
            if descriptor.optimizer is not None:
                matrix["optimizer"].add(descriptor.optimizer)
        self.assertGreaterEqual(len(by_operation), 15)
        for operation, matrix in by_operation.items():
            with self.subTest(operation=operation):
                self.assertEqual(matrix["shape"], set(SHAPE_REGIMES))
                self.assertEqual(matrix["dtype"], {"float32", "float16", "bfloat16"})
                self.assertEqual(matrix["phase"], {"forward", "training"})
                self.assertEqual(matrix["batch"], {1, 2, 8})
        parameterized = {
            "aten::linear",
            "aten::conv2d",
            "aten::batch_norm",
            "aten::layer_norm",
            "aten::embedding",
        }
        for operation in parameterized:
            self.assertEqual(
                by_operation[operation]["optimizer"],
                {"sgd", "adam", "adamw"},
            )
        self.assertEqual(len({row.workload_id for row in descriptors}), len(descriptors))

    def test_training_microbenchmarks_execute_backward_and_matching_optimizer(self) -> None:
        descriptors = default_microbenchmarks(
            dtypes=("float32",),
            phases=("training",),
            batch_sizes=(2,),
        )
        for operation in ("aten::linear", "aten::matmul"):
            descriptor = next(
                row
                for row in descriptors
                if row.declared_operations == (operation,)
                and row.shape_regime == "tiny"
                and (row.optimizer == "adam" if operation == "aten::linear" else row.optimizer is None)
            )
            instance = build_workload(descriptor)
            output = instance.model(*instance.args, **instance.kwargs)
            assert instance.loss_fn is not None
            loss = instance.loss_fn(output, instance.target)
            loss.backward()
            gradients = [
                value.grad
                for value in (*instance.model.parameters(), *instance.args)
                if value.is_floating_point() and value.requires_grad
            ]
            self.assertTrue(any(gradient is not None for gradient in gradients))
            if operation == "aten::linear":
                assert instance.optimizer_factory is not None
                optimizer = instance.optimizer_factory(instance.model.parameters())
                self.assertIsInstance(optimizer, torch.optim.Adam)
            else:
                self.assertIsNone(instance.optimizer_factory)

    def test_microbenchmarks_execute_declared_operations_not_substitutes(self) -> None:
        wanted = {
            "aten::matmul",
            "aten::bmm",
            "aten::transpose.int",
            "aten::reshape",
            "aten::scaled_dot_product_attention",
        }
        candidates = list(default_microbenchmarks())
        # P0 long-tail semantics that are not bootstrap exact IDs still use the
        # same exact-operation factory.
        prototype = candidates[0]
        candidates.extend(
            [
                WorkloadDescriptor(
                    **{
                        **prototype.to_dict(),
                        "workload_id": f"probe_{operation.replace(':', '_').replace('.', '_')}",
                        "source_group": f"probe:{operation}",
                        "source_fingerprint": "f" * 64,
                        "declared_operations": (operation,),
                        "config": {"operation": operation, "size": 8},
                    }
                )
                for operation in wanted - {row.declared_operations[0] for row in candidates}
            ]
        )
        selected = {}
        for descriptor in candidates:
            operation = descriptor.declared_operations[0]
            if operation in wanted and operation not in selected and descriptor.shape_regime == "tiny":
                selected[operation] = descriptor
        # Probes inherited "tiny"; include any explicitly added rows.
        for descriptor in candidates:
            selected.setdefault(descriptor.declared_operations[0], descriptor)
        for operation in wanted:
            descriptor = selected[operation]
            instance = build_workload(descriptor)
            result = capture_export(instance.model, instance.args, instance.kwargs)
            self.assertTrue(result.success, (operation, result.failures))
            assert result.graph is not None
            if operation == "aten::reshape":
                # torch.export canonicalizes a legal reshape to its ATen view.
                self.assertTrue(
                    {"aten::reshape", "aten::view"} & {node.raw_target for node in result.graph.nodes}
                )
            else:
                validate_declared_operations(result.graph, descriptor)

    def test_manifest_and_gap_selection_are_deterministic(self) -> None:
        descriptors = default_microbenchmarks()
        first = manifest_payload(descriptors)
        second = manifest_payload(reversed(descriptors))
        self.assertEqual(first, second)
        observed = tuple(
            descriptor
            for descriptor in descriptors
            if descriptor.declared_operations[0] != "aten::topk"
        )
        selected = select_coverage_gaps(
            descriptors,
            observed,
            limit=5,
            error_by_operation={"aten::topk": 10.0},
        )
        self.assertTrue(all(row.declared_operations == ("aten::topk",) for row in selected))

    def test_every_bootstrap_exact_operation_has_three_composite_contexts(self) -> None:
        micro_operations = {
            descriptor.declared_operations[0]
            for descriptor in default_microbenchmarks()
        }
        contexts: dict[str, set[str]] = {
            operation: set()
            for operation in micro_operations
        }
        composites = default_composites()
        for descriptor in composites:
            for operation in descriptor.declared_operations:
                if operation in contexts:
                    contexts[operation].add(descriptor.source_group)
        self.assertEqual(
            {operation: names for operation, names in contexts.items() if len(names) < 3},
            {},
        )

        small_by_context = {}
        for descriptor in composites:
            if descriptor.shape_regime == "small":
                small_by_context.setdefault(descriptor.source_group, descriptor)
        for source_group, descriptor in small_by_context.items():
            with self.subTest(source_group=source_group):
                instance = build_workload(descriptor)
                result = capture_export(instance.model, instance.args, instance.kwargs)
                self.assertTrue(result.success, result.failures)
                assert result.graph is not None
                validate_declared_operations(result.graph, descriptor)

    def test_real_library_workloads_execute_their_declared_source_semantics(self) -> None:
        descriptors = default_source_workloads()
        self.assertEqual(len(descriptors), 4)
        self.assertEqual({row.data_layer for row in descriptors}, {"real"})
        self.assertEqual(
            {row.modality for row in descriptors},
            {"image", "text", "graph"},
        )
        for descriptor in descriptors:
            with self.subTest(workload=descriptor.workload_id):
                instance = build_workload(descriptor)
                result = capture_export(instance.model, instance.args, instance.kwargs)
                self.assertTrue(result.success, result.failures)
                assert result.graph is not None
                validate_declared_operations(result.graph, descriptor)

    def test_grouped_split_never_leaks_source_variants(self) -> None:
        descriptors = (
            *default_microbenchmarks(),
            *default_composites(),
            *default_source_workloads(),
        )
        splits = grouped_split(descriptors, seed=7)
        validate_group_isolation(splits)
        again = grouped_split(reversed(descriptors), seed=7)
        self.assertEqual(splits, again)
        self.assertEqual(
            split_manifest_payload(splits),
            split_manifest_payload(again),
        )
        ownership = {}
        for split_name, rows in splits.items():
            for row in rows:
                ownership.setdefault(row.source_group, set()).add(split_name)
        self.assertTrue(all(len(names) == 1 for names in ownership.values()))
        for split_name in splits:
            self.assertEqual(
                {row.data_layer for row in splits[split_name]},
                {"microbenchmark", "composite", "real"},
            )

    def test_evaluation_slice_manifest_is_explicit_and_fail_closed(self) -> None:
        descriptors = (
            *default_microbenchmarks(),
            *default_composites(),
            *default_source_workloads(),
        )
        splits = grouped_split(descriptors, seed=7)
        first = evaluation_slice_manifest_payload(splits)
        second = evaluation_slice_manifest_payload(
            {
                name: tuple(reversed(rows))
                for name, rows in reversed(tuple(splits.items()))
            }
        )
        self.assertEqual(first, second)
        self.assertEqual(tuple(first["required_slices"]), EVALUATION_SLICE_NAMES)
        self.assertFalse(first["complete"])
        self.assertEqual(
            set(first["missing_required_slices"]),
            {
                "generated_code_robustness",
                "custom_oov_suite",
                "v2_compatible_matched_test",
            },
        )
        train_groups = {row.source_group for row in splits["train"]}
        for name, details in first["slices"].items():
            with self.subTest(evaluation_slice=name):
                self.assertFalse(train_groups & set(details["source_groups"]))
        for name in (
            "in_distribution_validation",
            "architecture_source_family_held_out",
            "operation_combination_held_out",
            "dynamic_shape_extrapolation",
            "precision_optimizer_held_out",
        ):
            self.assertTrue(first["slices"][name]["available"])


if __name__ == "__main__":
    unittest.main()
