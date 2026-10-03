import copy
from dataclasses import replace
import math
from types import SimpleNamespace

import pytest
import torch

from perfseer_v3.features import batch_graph_features
from perfseer_v3.op_registry import OperationRegistry
from perfseer_v4.adapters import HardwareAdaptedV4, TrainingHeadAdapter
from perfseer_v4.capture import capture, graph_from_design
from perfseer_v4.features import build_features
from perfseer_v4.model import SeerNetV4, SeerNetV4Config
from perfseer_v4.resource_model import RESOURCE_NAMES, ResourceMLP, ResourceMLPConfig, resource_values


@pytest.fixture
def sample():
    torch.set_num_threads(1)
    torch.manual_seed(42)
    training = dict(version="perfseer_v31_training_config_v1", precision="fp32", microbatch_size=3,
                    gradient_accumulation_steps=1, optimizer={"name": "adam", "learning_rate": .001},
                    scheduler={"name": "none"}, backend="cuda_eager", loss="mse_to_zero", steps_per_epoch=12)
    design, _ = capture(torch.nn.Linear(4, 2), (torch.randn(3, 4),), None, None, training, architecture_key="variant-test")
    features = build_features(design)
    config = SeerNetV4Config.from_registry(OperationRegistry.load(), features.layout,
                                         hidden=16, num_blocks=1, dropout=.2)
    target = torch.tensor([1., .9, 12., 20., 200., 210., 50.])
    scales = target.clone()
    scales[3] = 100
    base = SeerNetV4(config, scales, target)
    return base, design, features, target


def trained_heads(base):
    with torch.no_grad():
        for head in base.prediction_heads:
            head[-1].weight.uniform_(-.02, .02)
    return base


def test_hardware_adapter_identity_gradients_and_frozen_eval_source(sample):
    base, _, features, target = sample
    trained_heads(base).eval()
    batch = batch_graph_features([features.training] * 2)
    expected = base(batch).prediction.detach().clone()
    original = copy.deepcopy(base.state_dict())
    model = HardwareAdaptedV4(base).train()
    assert model.model_variant == "v4.2" and model.adapter_config == {"rank": 8, "hardware_hidden": 32}
    assert all(not module.training for module in base.modules())
    assert all(not value.requires_grad for value in base.parameters())
    torch.testing.assert_close(model(batch).prediction, expected, rtol=0, atol=0)
    torch.testing.assert_close(model.predict_batch(SimpleNamespace(training=batch)).prediction, expected, rtol=0, atol=0)
    assert model.displacement_penalty().item() == 0.
    optimizer = torch.optim.AdamW([value for value in model.parameters() if value.requires_grad], lr=.01, weight_decay=0.)
    loss = ((model(batch).prediction / target) - 1.2).square().mean() + model.displacement_penalty()
    loss.backward()
    assert all(value.grad is None for value in base.parameters())
    for adapter in model.adapters:
        assert adapter.up.weight.grad.abs().sum() > 0
        assert adapter.down.weight.grad.abs().sum() == 0
        assert adapter.film[-1].weight.grad.abs().sum() > 0
        assert adapter.film[0].weight.grad.abs().sum() == 0
    optimizer.step()
    assert model.displacement_penalty() > 0
    assert not torch.equal(model(batch).prediction, expected)
    for name, value in base.state_dict().items():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)


def test_adapter_formula_and_initialization_anchored_penalty_reload(sample):
    base, _, features, _ = sample
    adapter = TrainingHeadAdapter(16, 4, rank=8)
    graph, hardware = torch.randn(3, 16), torch.randn(3, 4)
    with torch.no_grad():
        adapter.up.weight.fill_(.05)
        adapter.film[-1].bias.fill_(.1)
    expected = 1.1 * graph + .1 + (graph @ adapter.down.weight.T @ adapter.up.weight.T) / 8
    torch.testing.assert_close(adapter(graph, hardware), expected)
    model = HardwareAdaptedV4(trained_heads(base))
    assert model.displacement_penalty().item() == 0. and model.adapters[0].down.weight.abs().sum() > 0
    count = model.initial_parameters.numel()
    with torch.no_grad():
        model.adapters[0].down.weight[0, 0] += 2
    assert model.displacement_penalty().item() == pytest.approx(1e-3 * 4 / count)
    restored_base = SeerNetV4(base.config, base.target_scales)
    restored = HardwareAdaptedV4(restored_base, require_trained_source=False)
    restored.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(restored.displacement_penalty(), model.displacement_penalty(), rtol=0, atol=0)
    batch = batch_graph_features([features.training])
    torch.testing.assert_close(restored(batch).prediction, model(batch).prediction, rtol=0, atol=0)


def test_adapter_rejects_untrained_source_and_invalid_dimensions(sample):
    base, _, _, _ = sample
    with pytest.raises(ValueError, match="trained source"):
        HardwareAdaptedV4(base)
    with pytest.raises(ValueError, match="positive"):
        TrainingHeadAdapter(16, 4, rank=0)


def test_resource_registry_matches_static_graph_and_preserves_unknown_epoch(sample):
    _, design, _, _ = sample
    values = resource_values(design)
    assert tuple(values) == RESOURCE_NAMES and len(set(RESOURCE_NAMES)) == 17
    assert all(math.isfinite(value) and value >= 0 for value in values.values())
    assert values["log_steps_per_epoch"] == math.log1p(12) and values["epoch_length_unknown"] == 0
    assert values["amp"] == values["tf32"] == values["compiled"] == values["checkpointing"] == 0
    assert values["log_microbatch"] == math.log1p(3)
    assert values["log_parameter_bytes"] == math.log1p(graph_from_design(design).global_features.total_parameter_bytes)
    changed = {**design, "training": {**design["training"], "TRAINING_CONFIG": {
        **design["training"]["TRAINING_CONFIG"], "steps_per_epoch": None}}}
    result = resource_values(changed)
    assert result["epoch_length_unknown"] == 1. and result["log_steps_per_epoch"] == 0.
    assert {name: value for name, value in result.items() if name not in RESOURCE_NAMES[-2:]} == {
        name: value for name, value in values.items() if name not in RESOURCE_NAMES[-2:]}


def test_resource_mlp_training_heads_hardware_and_serialization(sample):
    base, design, features, target = sample
    config = ResourceMLPConfig()
    assert config.hidden == 128 and config.resource_dim == 17 and config.hardware_dim == 124
    model = ResourceMLP(config, base.target_scales, torch.zeros(17), torch.ones(17), target)
    resources = torch.tensor([list(resource_values(design).values())] * 2)
    batch = SimpleNamespace(training=batch_graph_features([features.training] * 2), resources=resources)
    output = model.predict_batch(batch)
    torch.testing.assert_close(output.prediction, target.expand(2, -1))
    assert output.graph_embedding.shape == (2, 1, 128)
    assert output.phase_embedding.shape == (2, 1, 0, 128) and output.phase_presence.shape == (2, 1, 0)
    assert len(model.prediction_heads) == 3 and not any(isinstance(module, torch.nn.Dropout) for module in model.modules())
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-4)
    for _ in range(3):
        optimizer.zero_grad()
        loss = (model.predict_batch(batch).prediction / target - 1.2).square().mean()
        loss.backward()
        optimizer.step()
    assert all(value.grad is not None and torch.isfinite(value.grad).all() and value.grad.abs().sum() > 0
               for head in model.prediction_heads for value in head.parameters())
    output = model.predict_batch(batch).prediction
    assert (output[:, [0, 1, 2, 4, 5, 6]] > 0).all() and ((output[:, 3] >= 0) & (output[:, 3] <= 100)).all()
    changed = SimpleNamespace(resources=resources, training=replace(batch.training, hardware_cont=batch.training.hardware_cont + .2))
    assert not torch.equal(output, model.predict_batch(changed).prediction)
    restored = ResourceMLP(ResourceMLPConfig(**config.to_dict()), base.target_scales, torch.zeros(17), torch.ones(17))
    restored.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(restored.predict_batch(batch).prediction, output, rtol=0, atol=0)


def test_resource_mlp_rejects_missing_descriptors_and_invalid_normalization(sample):
    base, _, features, _ = sample
    with pytest.raises(ValueError, match="normalization"):
        ResourceMLP(ResourceMLPConfig(), base.target_scales, torch.zeros(17), torch.zeros(17))
    model = ResourceMLP(ResourceMLPConfig(), base.target_scales, torch.zeros(17), torch.ones(17))
    with pytest.raises(ValueError, match="raw resource"):
        model.predict_batch(SimpleNamespace(training=batch_graph_features([features.training])))


@pytest.mark.parametrize("variant", ["v4.2", "v4.3"])
def test_variant_checkpoint_dispatch_preserves_predictions_and_buffers(sample, variant):
    from perfseer_v4.training import checkpoint_payload, restore_model

    base, design, features, target = sample
    batch = SimpleNamespace(training=batch_graph_features([features.training]),
                            resources=torch.tensor([list(resource_values(design).values())]))
    model = (HardwareAdaptedV4(trained_heads(base)) if variant == "v4.2" else
             ResourceMLP(ResourceMLPConfig(), base.target_scales, torch.ones(17), torch.full((17,), 2.), target))
    model.eval()
    payload = checkpoint_payload(model, role="student", epoch=1, normalization={}, dataset_fingerprint="variant-test")
    restored = restore_model(payload).eval()
    assert restored.model_variant == variant
    torch.testing.assert_close(restored.predict_batch(batch).prediction, model.predict_batch(batch).prediction, rtol=0, atol=0)
    if variant == "v4.2":
        torch.testing.assert_close(restored.displacement_penalty(), model.displacement_penalty(), rtol=0, atol=0)
        assert all(not value.requires_grad for value in restored.base.parameters())
    else:
        torch.testing.assert_close(restored.resource_mean, model.resource_mean, rtol=0, atol=0)
        torch.testing.assert_close(restored.resource_scale, model.resource_scale, rtol=0, atol=0)
