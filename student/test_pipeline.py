"""End-to-end tests for the consecutive converter -> predictor pipeline.

Writes several pure PyTorch model files to a temp dir, runs SeerPredictor over
every (architecture x precision x backend) combination, and checks:
  1. pipeline completes and returns all 6 targets, finite and positive-bounded
  2. torch and onnx-int8 backends agree within quantization tolerance
  3. predictions are inside plausible A10 label ranges
  4. repeated calls are deterministic

Array shapes: see predict.py. Tolerances: relative |t - o| / max(t, eps) <= 25%
for int8-vs-fp32 backend agreement (quantized weights), determinism exact.

Usage:
  python student/test_pipeline.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from predict import SeerPredictor  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

MODELS = {
    "cnn": (
        """
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(32)
        self.act = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.act(self.bn(self.conv1(x)))
        x = self.act(self.conv2(x))
        return self.fc(self.flat(self.pool(x)))
""",
        "Net", [[8, 3, 224, 224]], ["float32"]),
    "resnet_block": (
        """
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.c1 = nn.Conv2d(32, 32, 3, padding=1)
        self.c2 = nn.Conv2d(32, 32, 3, padding=1)
        self.act = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(32, 100)

    def forward(self, x):
        x = self.act(self.stem(x))
        y = self.act(self.c1(x))
        y = self.c2(y)
        x = self.act(x + y)
        return self.fc(self.flat(self.pool(x)))
""",
        "Net", [[8, 3, 128, 128]], ["float32"]),
    "depthwise": (
        """
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(3, 24, 3, stride=2, padding=1)
        self.dw = nn.Conv2d(24, 24, 3, padding=1, groups=24)
        self.pw = nn.Conv2d(24, 48, 1)
        self.act = nn.SiLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flat = nn.Flatten()
        self.fc = nn.Linear(48, 10)

    def forward(self, x):
        x = self.act(self.stem(x))
        x = self.act(self.dw(x))
        x = self.act(self.pw(x))
        return self.fc(self.flat(self.pool(x)))
""",
        "Net", [[8, 3, 160, 160]], ["float32"]),
    "mlp": (
        """
import torch.nn as nn


def make():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512),
        nn.LayerNorm(512),
        nn.GELU(),
        nn.Linear(512, 128),
        nn.GELU(),
        nn.Linear(128, 10),
    )
""",
        "make", [[8, 1, 28, 28]], ["float32"]),
    "lstm_head": (
        """
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(1000, 64)
        self.rnn = nn.LSTM(64, 128, batch_first=True)
        self.fc = nn.Linear(128, 5)

    def forward(self, tokens):
        x = self.emb(tokens)
        out, _ = self.rnn(x)
        return self.fc(out[:, -1])
""",
        "Net", [[8, 32]], ["int64"]),
}

PRECISIONS = ["fp32_ieee", "tf32", "bf16_amp", "fp16_amp"]

RANGES = {  # plausible A10 bounds per target (label_v3 population)
    "train_util": (0.0, 100.0), "train_mem": (100.0, 24000.0), "train_time": (0.01, 10000.0),
    "infer_util": (0.0, 100.0), "infer_mem": (100.0, 24000.0), "infer_time": (0.005, 10000.0),
}


def main():
    torch_p = SeerPredictor(ROOT / "student/student_A10.pt")
    onnx_p = SeerPredictor(ROOT / "student/student_a10_cpu_int8.onnx")
    tmp = Path(tempfile.mkdtemp(prefix="seer_pipe_test_"))
    failures = []
    n_cases = 0

    for name, (src, entry, shapes, dtypes) in MODELS.items():
        f = tmp / f"{name}.py"
        f.write_text(src)
        for prec in PRECISIONS:
            n_cases += 1
            case = f"{name}/{prec}"
            try:
                t_out = torch_p.predict_source(str(f), entry, shapes, prec, input_dtypes=dtypes)
                o_out = onnx_p.predict_source(str(f), entry, shapes, prec, input_dtypes=dtypes)
                t2 = torch_p.predict_source(str(f), entry, shapes, prec, input_dtypes=dtypes)
            except Exception as exc:
                failures.append(f"{case}: pipeline raised {type(exc).__name__}: {exc}")
                continue
            for k, v in t_out.items():
                if not np.isfinite(v):
                    failures.append(f"{case}: non-finite {k}={v}")
                lo, hi = RANGES[k]
                if not (lo <= v <= hi):
                    failures.append(f"{case}: {k}={v:.3f} outside [{lo}, {hi}]")
            for k in t_out:
                rel = abs(t_out[k] - o_out[k]) / max(abs(t_out[k]), 1e-6)
                if rel > 0.25:
                    failures.append(f"{case}: backend disagreement {k} torch={t_out[k]:.3f} onnx={o_out[k]:.3f} rel={rel:.2%}")
            if any(t_out[k] != t2[k] for k in t_out):
                failures.append(f"{case}: non-deterministic torch outputs")
            print(f"ok {case}: train_util={t_out['train_util']:.2f}% train_mem={t_out['train_mem']:.0f}MiB "
                  f"train_time={t_out['train_time']:.3f} (onnx {o_out['train_util']:.2f}%)", flush=True)

    print(f"\ncases={n_cases} failures={len(failures)}")
    for msg in failures:
        print("FAIL", msg)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
