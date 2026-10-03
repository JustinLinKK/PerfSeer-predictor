from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

from perfseer_v3.features import batch_graph_features
from perfseer_v3.op_registry import OperationRegistry
from perfseer_v31.io import file_sha256, fingerprint
from perfseer_v32.capture import capture
from perfseer_v32.features import build_features
from perfseer_v32.model import SeerNetV32, SeerNetV32Config
from perfseer_v32.runner import fit_training_normalization
from perfseer_v32.training import checkpoint_payload, restore_model as restore_v32_model
from perfseer_v4.conversion import convert_file, convert_v32_checkpoint
from perfseer_v4.model import SeerNetV4, SeerNetV4Config
from perfseer_v4.version import (TARGET_NAMES, HEAD_GROUPS, NORMALIZATION_VERSION,
                                FEATURE_VERSION, NUMERIC_FEATURE_VERSION, validate_targets)


@pytest.fixture
def source():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    training = dict(version="perfseer_v31_training_config_v1", precision="fp32", microbatch_size=3,
                    gradient_accumulation_steps=1, optimizer={"name": "adam", "learning_rate": .001},
                    scheduler={"name": "none"}, backend="cuda_eager", loss="mse_to_zero", steps_per_epoch=12)
    design, _ = capture(torch.nn.Linear(4, 2), (torch.randn(3, 4),), None, None, training, architecture_key="v4-test")
    features = build_features(design)
    target = torch.tensor([1., .9, 12., 20., 200., 210., 50., .5, .4, 15., 100., 110.])
    scales = target.clone()
    scales[[3, 9]] = 100
    config = SeerNetV32Config.from_registry(OperationRegistry.load(), features.layout,
                                          hidden=16, num_blocks=1, dropout=0.)
    model = SeerNetV32(config, scales, target).eval()
    with torch.no_grad():
        for head in model.prediction_heads:
            head[-1].weight.uniform_(-.02, .02)
    normalization = asdict(fit_training_normalization([{"features": features}], "source-dataset"))
    payload = checkpoint_payload(model, role="student", epoch=17, normalization=normalization,
                                 dataset_fingerprint="source-dataset", validation={"gate_passed": True},
                                 next_epoch=18, best_key=(0.,), teacher_checkpoint_sha256="old-teacher")
    return model, features, target, payload


def test_training_only_heads_and_single_graph_pass(source):
    original, features, target, _ = source
    config = SeerNetV4Config(**{**original.config.to_dict(), "num_outputs": 7})
    model = SeerNetV4(config, original.target_scales[:7], target[:7])
    calls = []
    encode = model._encode

    def counted(mode, *args):
        calls.append(mode)
        return encode(mode, *args)

    model._encode = counted
    batch = batch_graph_features([features.training] * 2)
    output = model.predict_batch(SimpleNamespace(training=batch))
    assert calls == [0]
    assert len(model.phase_encoder) == len(model.phase_fusion) == len(model.phase_fusion_norm) == 1
    assert len(model.prediction_heads) == len(HEAD_GROUPS) == 3
    assert output.prediction.shape == (2, 7)
    assert output.graph_embedding.shape == (2, 1, 16)
    assert output.phase_embedding.shape == (2, 1, config.num_phases, 16)
    assert output.phase_presence.shape == (2, 1, config.num_phases)
    torch.testing.assert_close(output.prediction, target[:7].expand(2, -1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    for _ in range(3):
        optimizer.zero_grad()
        loss = (model(batch).prediction / target[:7] - 1.1).square().mean()
        loss.backward()
        optimizer.step()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
               for head in model.prediction_heads for parameter in head.parameters())


def test_conversion_retains_predictions_and_resets_run_state(source):
    original, features, _, payload = source
    converted = convert_v32_checkpoint(payload, source_sha256="a" * 64)
    model = SeerNetV4(SeerNetV4Config(**converted["model_config"]), converted["target_scales"]).eval()
    model.load_state_dict(converted["model_state_dict"], strict=True)
    training = batch_graph_features([features.training] * 2)
    inference = batch_graph_features([features.inference] * 2)
    with torch.no_grad():
        expected = original(training, inference)
        actual = model(training)
    torch.testing.assert_close(actual.prediction, expected.prediction[:, :7], rtol=0, atol=0)
    torch.testing.assert_close(actual.graph_embedding, expected.graph_embedding[:, :1], rtol=0, atol=0)
    assert sum(p.numel() for p in model.parameters()) < sum(p.numel() for p in original.parameters())
    assert tuple(converted["target_names"]) == TARGET_NAMES
    assert converted["epoch"] == 0 and converted["model_variant"] == "v4.0"
    assert not {"optimizers", "schedulers", "rng", "next_epoch", "validation", "best_key", "teacher_checkpoint_sha256"} & converted.keys()
    assert set(converted["normalization"]) == {"training", "version"}
    assert converted["normalization"]["training"] == {**payload["normalization"]["training"], "normalization_version": NORMALIZATION_VERSION}
    assert converted["normalization_sha256"] == fingerprint(converted["normalization"])
    assert converted["feature_version"] == FEATURE_VERSION
    assert features.layout.feature_schema_version == NUMERIC_FEATURE_VERSION
    assert converted["conversion"]["source_checkpoint_sha256"] == "a" * 64
    assert converted["conversion"]["initialization_only"]
    provenance = converted["conversion"]["source_preprocessing"]
    assert provenance["feature_version"] == NUMERIC_FEATURE_VERSION
    assert provenance["model_config_sha256"] == fingerprint(payload["model_config"])
    assert provenance["conversion_runtime_files"] and not provenance["historical_runtime_verified"]
    assert converted["conversion"]["source_preprocessing_sha256"] == fingerprint(provenance)
    with pytest.raises(ValueError, match="v3.2"):
        restore_v32_model(converted)


def test_conversion_verifies_source_and_saved_artifact(source, tmp_path):
    _, _, _, payload = source
    checkpoint, output = tmp_path / "source.pt", tmp_path / "v4.pt"
    torch.save(payload, checkpoint)
    converted = convert_file(checkpoint, output)
    assert converted["conversion"]["source_checkpoint_sha256"] == file_sha256(checkpoint)
    assert torch.load(output, weights_only=False)["target_names"] == TARGET_NAMES
    with pytest.raises(ValueError, match="new output"):
        convert_file(checkpoint, output)
    with pytest.raises(ValueError, match="new output"):
        convert_file(checkpoint, checkpoint)
    with pytest.raises(ValueError, match="v3.2"):
        convert_v32_checkpoint({**payload, "version": "unknown"}, source_sha256="a" * 64)
    with pytest.raises(ValueError, match="SHA-256"):
        convert_v32_checkpoint(payload, source_sha256="unknown")
    with pytest.raises(ValueError, match="normalization dataset"):
        convert_v32_checkpoint({**payload, "dataset_fingerprint": "other"}, source_sha256="a" * 64)
    assert payload["normalization"]["version"] != NORMALIZATION_VERSION


def test_seven_target_contract_rejects_inference_outputs_and_invalid_ranges(source):
    original, _, target, _ = source
    validate_targets(target[:7].tolist())
    validate_targets([*target[:3].tolist(), 0., *target[4:7].tolist()])
    with pytest.raises(ValueError, match="seven"):
        validate_targets(target.tolist())
    with pytest.raises(ValueError, match="physical range"):
        validate_targets([*target[:3].tolist(), 101., *target[4:7].tolist()])
    with pytest.raises(ValueError, match="seven"):
        SeerNetV4Config(**original.config.to_dict())
