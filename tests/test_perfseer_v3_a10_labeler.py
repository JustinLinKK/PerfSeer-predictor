from __future__ import annotations

import fcntl
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
import importlib.util

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def _native_script(source: str, *, timeout: int = 300) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PERFSEER_LABELER_PROFILE"] = "native_a10"
    environment["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(
        [sys.executable, "-c", source],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_exact_reference_native_crosswalk_and_campaign_contract() -> None:
    result = _native_script(
        """
import json
from perfseer_v3.dataset_pack.a10_crosswalk import build_crosswalk
from perfseer_v3.dataset_pack.a10_campaign import build_campaign_contract, pilot_candidates
from perfseer_v3.dataset_pack.sampler import build_target_manifest
m=build_target_manifest(); x=build_crosswalk(m); c=build_campaign_contract(); p=pilot_candidates(m)
print(json.dumps({
 'manifest':m.sha256,
 'reference':x.summary['reference_manifest_sha256'],
 'rows':len(x.rows),
 'precisions':m.to_summary()['precision_counts'],
 'tasks':len({r.task_id for r in p}),
 'families':len({r.family_id for r in p}),
 'pilot':len(p),
 'chunks':c['production_chunk_sizes'],
}))
"""
    )
    value = json.loads(result.stdout)
    assert value["reference"] == "bf805655d2a9fe978ce2ad4d8bb1f0c0efa400b013e2d1f1fa258ffc83cbeb8e"
    assert value["rows"] == 18_000
    assert value["precisions"] == {
        "bf16": 5_232,
        "fp16_grad_scaler": 4_171,
        "fp32_tf32": 5_193,
        "mixed_structured": 3_404,
    }
    assert (value["pilot"], value["tasks"], value["families"]) == (96, 22, 35)
    assert value["chunks"] == [*([256] * 69), 240]


def test_legacy_reference_hashes_are_exact() -> None:
    environment = os.environ.copy()
    environment["PERFSEER_LABELER_PROFILE"] = "legacy_a10g"
    environment["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from perfseer_v3.dataset_pack.sampler import build_target_manifest; "
            "from perfseer_v3.dataset_pack.task_registry import load_task_registry; "
            "from perfseer_v3.dataset_pack.model_registry import load_model_registry; "
            "print(build_target_manifest().sha256,load_task_registry().sha256,load_model_registry().sha256)",
        ],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.stdout.split() == [
        "bf805655d2a9fe978ce2ad4d8bb1f0c0efa400b013e2d1f1fa258ffc83cbeb8e",
        "781b93ddc020d7cbc77d816458fc213065088a27b9e72d42290ddd8e656de049",
        "ad1027aa906ff8dbc23c24b921ea85496b473b7eeb88fcce10628d142a35c9e2",
    ]


def test_a10_hardware_acceptance_and_rejection() -> None:
    result = _native_script(
        """
from perfseer_v3.dataset_pack.supervisor import SupervisorError, validate_a10_gpu_identity
validate_a10_gpu_identity('NVIDIA A10',(8,6),24*1024**3)
for value in [('NVIDIA A10G',(8,6),24*1024**3),('NVIDIA A10',(8,0),24*1024**3),('NVIDIA A10',(8,6),32*1024**3),('RTX A10',(8,6),24*1024**3)]:
 try: validate_a10_gpu_identity(*value)
 except SupervisorError: pass
 else: raise AssertionError(value)
print('passed')
"""
    )
    assert result.stdout.strip() == "passed"


def test_precision_contexts_and_scaler_contract(monkeypatch) -> None:
    result = _native_script(
        """
from contextlib import nullcontext
from perfseer_v3.dataset_pack.local_runtime import _precision_context
from perfseer_v3.dataset_pack.sampler import build_target_manifest
import torch
m=build_target_manifest()
for precision, expected_dtype, scaler in [('fp32_tf32',None,False),('bf16',torch.bfloat16,False),('fp16_grad_scaler',torch.float16,True),('mixed_structured',torch.bfloat16,False)]:
 c=next(r for r in m.candidates if r.precision_policy['policy_id']==precision)
 ctx=_precision_context(c,torch.device('cuda'))
 assert c.precision_policy['gradient_scaler'] is scaler
 if expected_dtype is None: assert isinstance(ctx,type(nullcontext()))
 else: assert ctx.fast_dtype==expected_dtype
print('passed')
"""
    )
    assert result.stdout.strip() == "passed"


def _hold_lock(path: str, ready: multiprocessing.Queue) -> None:
    root = Path(path)
    lock = root / "state/campaign.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(lock, os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(descriptor, fcntl.LOCK_EX)
    ready.put(True)
    time.sleep(2)
    os.close(descriptor)


def test_exclusive_workspace_lock_rejects_second_writer(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment["PERFSEER_LABELER_PROFILE"] = "native_a10"
    environment["PYTHONPATH"] = str(ROOT / "src")
    ready: multiprocessing.Queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=_hold_lock, args=(str(tmp_path), ready))
    process.start()
    assert ready.get(timeout=5)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from perfseer_v3.dataset_pack.a10_campaign import exclusive_workspace_lock; "
            f"\nwith exclusive_workspace_lock({str(tmp_path)!r}): pass",
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
    )
    process.join(timeout=5)
    assert result.returncode != 0
    assert "another writer" in result.stderr


def test_chunk_mode_requires_pilot_and_strict_sequence(tmp_path: Path) -> None:
    result = _native_script(
        f"""
from pathlib import Path
from perfseer_v3.dataset_pack.a10_campaign import A10CampaignError, run_campaign
root=Path({str(tmp_path)!r})
try:
 run_campaign(workspace=root,repository_root='.',mlebench_checkout='.',repository_revision='a'*40,image_digest='registry/repo@sha256:'+'b'*64,chunk_index=0,executor=lambda rows:None)
except A10CampaignError as e:
 assert 'pilot receipt' in str(e)
else: raise AssertionError('chunk ran without pilot')
print('passed')
""",
        timeout=300,
    )
    assert result.stdout.strip() == "passed"


def test_rendered_pilot_chunk_and_pvc_contract(tmp_path: Path) -> None:
    path = ROOT / "scripts/render_a10_nautilus_job.py"
    spec = importlib.util.spec_from_file_location("render_a10_nautilus_job", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _load, main, verify_job = module._load, module.main, module.verify_job

    image = "gitlab-registry.nrp-nautilus.io/group/perfseer-a10-labeler@sha256:" + "b" * 64
    revision = "a" * 40
    pilot = tmp_path / "pilot.yaml"
    chunk = tmp_path / "chunk.yaml"
    assert main(["--output", str(pilot), "--namespace", "test-ns", "--image", image, "--source-revision", revision, "--mode", "pilot"]) == 0
    assert main(["--output", str(chunk), "--namespace", "test-ns", "--image", image, "--source-revision", revision, "--mode", "chunk", "--chunk-index", "69"]) == 0
    verify_job(_load(pilot), mode="pilot")
    verify_job(_load(chunk), mode="chunk", chunk_index=69)
    pvc = yaml.safe_load((ROOT / "k8s/a10-labeler-pvc.yaml").read_text())
    assert pvc["spec"]["accessModes"] == ["ReadWriteMany"]
    assert pvc["spec"]["resources"]["requests"]["storage"] == "700Gi"


def test_kaggle_rate_limit_retries_then_succeeds(monkeypatch, tmp_path: Path) -> None:
    from perfseer_v3.dataset_pack.kaggle import KaggleCliClient

    executable = tmp_path / "kaggle"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o755)
    outcomes = [
        SimpleNamespace(returncode=1, stderr="HTTP 429", stdout=""),
        SimpleNamespace(returncode=1, stderr="too many requests", stdout=""),
        SimpleNamespace(returncode=0, stderr="", stdout="ok"),
    ]
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: outcomes.pop(0))
    monkeypatch.setattr(time, "sleep", lambda _: None)
    client = KaggleCliClient(executable=str(executable), maximum_attempts=4, initial_backoff_seconds=0)
    assert client._run(["competitions", "list"]).stdout == "ok"
    assert not outcomes


def test_image_lock_contains_hashed_full_runtime() -> None:
    lock = (ROOT / "containers/a10-labeler/requirements.lock").read_text()
    for package in (
        "torch==2.10.0+cu128",
        "torchaudio==2.10.0+cu128",
        "torchvision==0.25.0+cu128",
        "torch-geometric==2.7.0",
        "tensorflow-cpu==2.20.0",
        "transformers==5.7.0",
        "kaggle==2.2.2",
    ):
        assert package in lock
    assert "--hash=sha256:" in lock
    assert "kaggle.json" not in lock and "git+" not in lock and "/home/" not in lock


def test_offline_image_preflight_does_not_import_kaggle_cli() -> None:
    script = (ROOT / "scripts" / "run_perfseer_v3_a10_labeling.py").read_text(
        encoding="utf-8"
    )
    assert '"kaggle": None' in script
