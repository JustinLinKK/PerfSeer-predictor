"""Portable continuation state and optional physical timing prior for ABA-3.

All costs inferred from the static forward graph remain estimates. The GPU busy-time
label is not used as a compute-throughput target. This module never reads labels.
"""
import datetime as dt
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class TrainingStream:
    """Save the actual shuffle and next position, not just an epoch counter."""
    def __init__(self, indices, state=None):
        self.indices = indices.cpu()
        self.order = torch.empty(0, dtype=torch.long)
        self.cursor = self.epoch = self.presentations = 0
        if state is not None:
            self.order = state['order'].cpu()
            self.cursor = int(state['cursor'])
            self.epoch = int(state['epoch'])
            self.presentations = int(state['presentations'])
            if sorted(self.order.tolist()) != sorted(self.indices.tolist()):
                raise ValueError('continuation training split differs')
            if not 0 <= self.cursor <= len(self.order):
                raise ValueError('invalid continuation cursor')

    def next(self, batch_size):
        if self.cursor == len(self.order):
            self.order = self.indices[torch.randperm(len(self.indices))]
            self.cursor = 0
            self.epoch += 1
        return self.order[self.cursor:self.cursor + batch_size]

    def commit(self, count):
        self.cursor += count
        self.presentations += count

    def state_dict(self):
        return dict(order=self.order.cpu(), cursor=self.cursor, epoch=self.epoch,
                    presentations=self.presentations)


def capture_rng():
    return dict(torch=torch.get_rng_state(), numpy=np.random.get_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [])


def restore_rng(state):
    torch.set_rng_state(state['torch'].cpu())
    np.random.set_state(state['numpy'])
    if state['cuda']:
        torch.cuda.set_rng_state_all([v.cpu() for v in state['cuda']])


def campaign_window(task, now=None):
    value = json.loads((Path(task) / 'launch-window.json').read_text())
    experiment = json.loads((Path(task) / 'experiment.json').read_text())
    if (value.get('budget_minutes') != experiment['budget_minutes'] or
            value.get('deadline_resets_allowed') is not False):
        raise ValueError('public window differs from verified schedule; never reset')
    start, end = [dt.datetime.fromisoformat(value[k]) for k in ('started_at', 'deadline')]
    if start.tzinfo is None or end.tzinfo is None or (end-start).total_seconds() != 86400:
        raise ValueError('expected an immutable twenty-four-hour campaign window')
    now = now or dt.datetime.now(dt.timezone.utc)
    if not start <= now < end:
        raise RuntimeError('outside original campaign window; never reset its deadline')
    return start, end


class HorusFeatures(nn.Module):
    """Input-only workload summaries; operator families are registry semantics, not model IDs.

    Existing global FLOPs/activation sizes already include captured batch dimensions.
    Never multiply them by batch size a second time or infer utilization from FLOPs.
    Source-fitted normalization stays unchanged. Count features use fixed log1p scaling.
    """
    def __init__(self, config):
        super().__init__()
        self.register_buffer('fields', torch.tensor(config['horus_global_indices'], dtype=torch.long))
        self.register_buffer('op_groups', torch.tensor(config['horus_op_groups'], dtype=torch.long))
        self.group_count = len(config['horus_group_names'])
        self.width = len(config['horus_global_indices']) + 2 * self.group_count + 1

    def forward(self, batch):
        rows = batch.u_cont.shape[0]
        counts = batch.u_cont.new_zeros((rows, self.group_count), dtype=torch.float32)
        flat = batch.batch * self.group_count + self.op_groups[batch.op_exact_id]
        counts.view(-1).index_add_(0, flat, torch.ones_like(flat, dtype=torch.float32))
        total = counts.sum(-1, keepdim=True)
        return torch.cat((batch.u_cont.float()[:, self.fields], torch.log1p(counts),
                          counts / total.clamp_min(1), torch.log1p(total)), dim=-1)


class HorusSM(nn.Module):
    """Independent hurdle/magnitude residual using Horus-inspired workload summaries."""
    def __init__(self, legacy, config):
        super().__init__()
        self.legacy = legacy
        self.features = HorusFeatures(config)
        self.adapter = nn.Linear(self.features.width, self.features.width)
        nn.init.eye_(self.adapter.weight); nn.init.zeros_(self.adapter.bias)
        self.head = nn.Sequential(nn.Linear(self.features.width, 64), nn.SiLU(), nn.Linear(64, 2))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)

    def forward(self, batch):
        _, logit, positive, penalty = self.legacy(batch)
        residual = self.head(self.adapter(self.features(batch))).float()
        logit = logit + residual[:, 0]
        # Bounded odds adjustment preserves the exact inherited magnitude at residual=0.
        odds = torch.exp(4 * torch.tanh(residual[:, 1]))
        positive = positive * odds / (1 + (positive / 100) * (odds - 1))
        positive = positive.clamp(0, 100)
        prediction = torch.where(logit >= 0, positive, torch.zeros_like(positive))
        return prediction, logit, positive, penalty + 1e-5 * residual.square().mean()


class PhysicalTiming(nn.Module):
    """Warm-started timing branch with explicit, weakly supervised cost priors.

The legacy timing model stays inside this timing branch. A zero-initialized mixing
coefficient introduces the prior gradually; no VRAM/SM parameters are referenced.
Verified nominal hardware ceilings are separate from unknown runtime specifications.
"""
    def __init__(self, legacy, config):
        super().__init__()
        self.legacy = legacy
        self.config = config
        self.register_buffer('node_mean', torch.tensor(config['node_mean']))
        self.register_buffer('node_std', torch.tensor(config['node_std']))
        self.register_buffer('global_mean', torch.tensor(config['global_mean']))
        self.register_buffer('global_std', torch.tensor(config['global_std']))
        self.register_buffer('op_types', torch.tensor(config['op_types'], dtype=torch.long))
        width = len(config['node_mean'])
        self.experts = nn.ModuleList(nn.Sequential(nn.Linear(width, 64), nn.SiLU(),
                                                 nn.Linear(64, 3)) for _ in range(3))
        self.backward_factor = nn.Parameter(torch.tensor(math.log(math.expm1(2.))))
        self.optimizer_bytes = nn.Embedding(config['num_optimizers'], 1)
        nn.init.constant_(self.optimizer_bytes.weight, math.log(math.expm1(16.)))
        self.mix = nn.Parameter(torch.tensor(0.))
        self.correction = nn.Sequential(nn.Linear(legacy.width, 64), nn.SiLU(), nn.Linear(64, 1))
        nn.init.zeros_(self.correction[-1].weight)
        nn.init.zeros_(self.correction[-1].bias)
        self.host = nn.Sequential(nn.Linear(legacy.width, 32), nn.SiLU(), nn.Linear(32, 1))
        nn.init.zeros_(self.host[-1].weight)
        nn.init.constant_(self.host[-1].bias, -12.)
        if config['mode'] == 'horus':
            self.features = HorusFeatures(config)
            self.adapter = nn.Linear(self.features.width, self.features.width)
            nn.init.eye_(self.adapter.weight); nn.init.zeros_(self.adapter.bias)
            self.workload_head = nn.Sequential(nn.Linear(self.features.width, 64), nn.SiLU(), nn.Linear(64, 2))
            nn.init.zeros_(self.workload_head[-1].weight); nn.init.zeros_(self.workload_head[-1].bias)

    @staticmethod
    def unlog(value):
        # These selected physical fields use log1p_nonnegative in the pinned schema.
        return torch.expm1(value.double().clamp(0, 70)).float()

    def forward(self, batch):
        if not bool((batch.hardware_id == self.config.get('hardware_id', 105)).all()):
            raise ValueError('A10 physical prior needs an explicit target hardware adapter before transfer')
        prediction, old_correction, penalty = self.legacy(batch)
        _, _, graph = self.legacy.core(batch)
        raw = batch.x_cont.float() * self.node_std + self.node_mean
        global_raw = batch.u_cont.float() * self.global_std + self.global_mean
        fields = self.config['node_fields']
        flops = self.unlog(raw[:, fields.index('flops')])
        traffic = self.unlog(raw[:, fields.index('bytes_read')]) + self.unlog(raw[:, fields.index('bytes_written')])
        types = self.op_types[batch.op_exact_id]
        experts = torch.stack([net(batch.x_cont.float()) for net in self.experts], dim=1)
        learned = experts[torch.arange(len(types), device=types.device), types]
        efficiency = .01 + .99 * torch.sigmoid(learned[:, :2])
        hw = self.config['hardware']
        precision = batch.precision_id[batch.batch]
        tensor_ops = types == 0
        low_precision = (precision == 9) | (precision == 10)
        ceilings = torch.full_like(flops, hw['fp32_tflops'] * 1e12)
        ceilings = torch.where(tensor_ops & low_precision,
                               torch.full_like(ceilings, hw['fp16_bf16_dense_tflops'] * 1e12), ceilings)
        # Nominal tensor-core availability is an upper bound, not evidence of actual use.
        compute = flops / (ceilings * efficiency[:, 0])
        memory = traffic / (hw['memory_bandwidth_gb_s'] * 1e9 * efficiency[:, 1])
        forward_ms = 1000 * torch.maximum(compute, memory) + .001 * F.softplus(learned[:, 2])
        sums = prediction.new_zeros(len(prediction))
        sums.index_add_(0, batch.batch, forward_ms)
        global_fields = self.config['global_fields']
        accumulation = self.unlog(global_raw[:, global_fields.index('gradient_accumulation_steps')]).clamp_min(1)
        params = self.unlog(global_raw[:, global_fields.index('total_parameter_numel')])
        optimizer_ms = 1000 * params * F.softplus(self.optimizer_bytes(batch.optimizer_id)[:, 0]) / (hw['memory_bandwidth_gb_s'] * 1e9)
        prior = (sums * (1 + F.softplus(self.backward_factor)) * accumulation + optimizer_ms).clamp_min(1e-6)
        graph_correction = (.5 * torch.tanh(self.correction(graph)[:, 0])
                            if self.config['mode'] in {'physics_graph', 'horus'} else torch.zeros_like(prior))
        prior = prior * torch.exp(graph_correction)
        # Bounded residual around the trained A10 estimate; the initial GPU prediction is unchanged.
        adjustment = torch.tanh(self.mix) * torch.tanh(torch.log(prior / prediction[:, 1].clamp_min(1e-6)))
        workload = prediction.new_zeros((len(prediction), 2))
        if self.config['mode'] == 'horus':
            workload = 2 * torch.tanh(self.workload_head(self.adapter(self.features(batch))).float())
            adjustment = adjustment + workload[:, 0]
        gpu = prediction[:, 1] * torch.exp(adjustment)
        overhead = (prediction[:, 0] - prediction[:, 1]).clamp_min(0) + F.softplus(self.host(graph)[:, 0])
        overhead = overhead * torch.exp(workload[:, 1])
        wall = gpu + overhead
        correction = old_correction + torch.stack((adjustment, adjustment), dim=-1)
        return torch.stack((wall, gpu), dim=-1), correction, penalty + 1e-5 * graph_correction.square().mean()


def attach_physics(model, config):
    model.timing = PhysicalTiming(model.timing, config)
    if config['mode'] == 'horus':
        model.sm = HorusSM(model.sm, config)
    model.physics_config = config
    model._verify_disjoint_parameters()
    return model


def physics_config(task, mode, model):
    from perfseer_v3.op_registry import OperationRegistry
    task = Path(task)
    data = torch.load(task/'data/A10/smoke-inputs.pt', weights_only=False, map_location='cpu')
    normalizer = data['normalization']
    sample = data['features'][0]
    registry = OperationRegistry.load()
    # Semantic operator categories are derived from exact operation identity, never workload family.
    op_types = [2] * model.vram.config.num_exact_ops
    group_names = sorted({'unknown'} | {str(r.family) for r in registry.rules})
    op_groups = [group_names.index('unknown')] * len(op_types)
    for rule in registry.rules:
        if isinstance(rule, dict):
            exact, family = rule.get('id'), rule.get('family')
        else:
            exact, family = getattr(rule, 'exact_id', None), getattr(rule, 'family', None)
        if exact is not None and 0 < int(exact) < len(op_types):
            op_types[int(exact)] = 0 if family in {'dense_matrix', 'convolution', 'attention'} else 1 if family in {'elementwise', 'reduction', 'normalization', 'activation_unary'} else 2
            op_groups[int(exact)] = group_names.index(str(family))
    global_fields = list(sample.layout.global_continuous_fields)
    horus_fields = ['total_flops', 'total_parameter_bytes', 'total_activation_bytes',
                    'batch_size', 'model_input_numel', 'model_input_bytes', 'operation_nodes',
                    'tensor_edges', 'critical_path_length', 'gradient_accumulation_steps',
                    'unknown_operation_fraction', 'unknown_cost_fraction', 'unknown_byte_fraction',
                    'forward_bytes_read', 'forward_bytes_written', 'execution_cuda_eager',
                    'execution_torch_compile', 'execution_unknown', 'precision_tf32']
    return dict(mode=mode, node_mean=list(normalizer.node.mean), node_std=list(normalizer.node.std),
                global_mean=list(normalizer.global_features.mean), global_std=list(normalizer.global_features.std),
                node_fields=list(sample.layout.node_continuous_fields),
                global_fields=list(sample.layout.global_continuous_fields), op_types=op_types,
                num_optimizers=model.vram.config.num_optimizers,
                hardware_id=105, horus_global_fields=horus_fields,
                horus_global_indices=[global_fields.index(n) for n in horus_fields],
                horus_group_names=group_names, horus_op_groups=op_groups,
                hardware=json.loads((task/'hardware-priors.json').read_text())['A10'],
                backward_optimizer_costs='estimated_not_captured', nominal_amp_only=True)
