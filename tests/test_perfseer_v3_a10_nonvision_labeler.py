from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
PROFILE = "native_a10_nonvision_4gpu_v1"


def _profile_script(source: str, *, timeout: int = 600) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PERFSEER_LABELER_PROFILE"] = PROFILE
    environment["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run(
        [sys.executable, "-c", source],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=timeout,
    )


def test_nonvision_manifest_counts_and_balanced_effective_batches() -> None:
    result = _profile_script(
        """
import json
from collections import Counter
from perfseer_v3.dataset_pack.sampler import build_target_manifest
m=build_target_manifest()
print(json.dumps({
 'hash':m.sha256,
 'count':len(m.candidates),
 'epochs':len(m.candidates)*3,
 'tasks':len({r.task_id for r in m.candidates}),
 'families':len({r.family_id for r in m.candidates}),
 'modalities':Counter(r.quota_modality for r in m.candidates),
 'precisions':Counter(r.precision_policy['policy_id'] for r in m.candidates),
 'batches':Counter(r.microbatch_size*r.gradient_accumulation_steps for r in m.candidates),
 'vision':sum(r.source_modality=='vision' for r in m.candidates),
}))
"""
    )
    import json

    value = json.loads(result.stdout)
    assert value == {
        "hash": "05e4d98b94c55d787a3d325b9c35f8ab3ff5dfbbb211be22844ff49bd881515b",
        "count": 11_200,
        "epochs": 33_600,
        "tasks": 12,
        "families": 22,
        "modalities": {
            "audio": 1_300,
            "generated": 2_600,
            "graph": 1_300,
            "nlp": 5_050,
            "tabular": 950,
        },
        "precisions": {
            "bf16": 3_183,
            "fp16_grad_scaler": 2_541,
            "fp32_tf32": 3_232,
            "mixed_structured": 2_244,
        },
        "batches": {"32": 2_240, "64": 2_240, "128": 2_240, "256": 2_240, "512": 2_240},
        "vision": 0,
    }


def test_frozen_pilot_has_exact_category_and_route_coverage() -> None:
    result = _profile_script(
        """
import json
from collections import Counter
from perfseer_v3.dataset_pack.a10_campaign import pilot_candidates
rows=pilot_candidates()
print(json.dumps({
 'count':len(rows),
 'categories':Counter(r.quota_modality for r in rows),
 'families':len({r.family_id for r in rows}),
 'tasks':len({r.task_id for r in rows}),
 'precisions':len({r.precision_policy['policy_id'] for r in rows}),
 'execution':len({r.execution['mode'] for r in rows}),
 'regimes':len({r.regime for r in rows}),
 'checkpoint':len({r.activation_checkpointing['enabled'] for r in rows}),
 'batches':Counter(r.microbatch_size*r.gradient_accumulation_steps for r in rows),
}))
"""
    )
    import json

    value = json.loads(result.stdout)
    assert value["count"] == 32
    assert value["categories"] == {
        "audio": 5,
        "generated": 6,
        "graph": 5,
        "nlp": 11,
        "tabular": 5,
    }
    assert (value["families"], value["tasks"]) == (22, 12)
    assert (value["precisions"], value["execution"], value["regimes"], value["checkpoint"]) == (4, 2, 3, 2)
    assert all(value["batches"].get(str(batch), 0) >= 5 for batch in (32, 64, 128, 256, 512))


def test_crosswalk_contract_declares_exact_retention_and_exclusion() -> None:
    result = _profile_script(
        """
import json
from perfseer_v3.dataset_pack.a10_nonvision_crosswalk import build_crosswalk
x=build_crosswalk()
print(json.dumps({
 'rows':len(x.rows),
 'retained':x.summary['retained_count'],
 'excluded':x.summary['excluded_count'],
 'reasons':x.summary['exclusion_counts'],
 'hash':x.summary['crosswalk_sha256'],
 'changed':sum(r['disposition']=='retained' and r['predecessor_candidate_id']==r['nonvision_candidate_id'] for r in x.rows),
}))
""",
        timeout=900,
    )
    import json

    value = json.loads(result.stdout)
    assert value == {
        "rows": 18_000,
        "retained": 11_200,
        "excluded": 6_800,
        "reasons": {"generated_vision_source": 650, "vision_family": 6_150},
        "hash": "6556f56f799c83772eb9fbac8639aa4c7be7476fdf3616b519cbcfd89f1be999",
        "changed": 0,
    }


def test_four_worker_batch_is_concurrent_and_isolates_one_candidate() -> None:
    environment = os.environ.copy()
    environment["PERFSEER_LABELER_PROFILE"] = PROFILE
    os.environ["PERFSEER_LABELER_PROFILE"] = PROFILE
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    from perfseer_v3.dataset_pack.a10_campaign import run_isolated_worker_batch
    from perfseer_v3.dataset_pack.workflow import SlotExhaustedError

    probes = tuple(SimpleNamespace(gpu_uuid=f"GPU-{index}") for index in range(4))
    started = time.monotonic()

    def worker(root: int, probe: SimpleNamespace) -> tuple[int, str]:
        time.sleep(0.15)
        if root == 1:
            raise SlotExhaustedError("fixture exhausted")
        return root, probe.gpu_uuid

    successes, isolated = run_isolated_worker_batch(
        (0, 1, 2, 3),
        probes,
        worker,
        isolated_exceptions=(SlotExhaustedError,),
    )
    assert time.monotonic() - started < 0.5
    assert {row[0] for row in successes} == {0, 2, 3}
    assert len(isolated) == 1 and isolated[0][0] == 1


def test_four_worker_batch_rejects_duplicate_gpu_and_observes_siblings() -> None:
    os.environ["PERFSEER_LABELER_PROFILE"] = PROFILE
    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    from perfseer_v3.dataset_pack.a10_campaign import A10CampaignError, run_isolated_worker_batch

    duplicate = tuple(SimpleNamespace(gpu_uuid="GPU-same") for _ in range(4))
    with pytest.raises(A10CampaignError, match="more than once"):
        run_isolated_worker_batch((0, 1, 2, 3), duplicate, lambda root, probe: root)

    completed: list[int] = []
    probes = tuple(SimpleNamespace(gpu_uuid=f"GPU-{index}") for index in range(4))

    def worker(root: int, probe: SimpleNamespace) -> int:
        time.sleep(0.05)
        completed.append(root)
        if root == 0:
            raise RuntimeError("global fixture failure")
        return root

    with pytest.raises(RuntimeError, match="global fixture"):
        run_isolated_worker_batch((0, 1, 2, 3), probes, worker)
    assert set(completed) == {0, 1, 2, 3}


def test_hardware_identity_and_rtx_separation() -> None:
    result = _profile_script(
        """
from perfseer_v3.dataset_pack.supervisor import SupervisorError, validate_a10_gpu_identity, validate_rtx5090_gpu_identity
validate_a10_gpu_identity('NVIDIA A10',(8,6),24*1024**3)
validate_rtx5090_gpu_identity('NVIDIA GeForce RTX 5090',(12,0),32*1024**3)
for fn,value in ((validate_a10_gpu_identity,('NVIDIA GeForce RTX 5090',(12,0),32*1024**3)),(validate_rtx5090_gpu_identity,('NVIDIA A10',(8,6),24*1024**3))):
 try: fn(*value)
 except SupervisorError: pass
 else: raise AssertionError('mismatch accepted')
print('passed')
"""
    )
    assert result.stdout.strip() == "passed"


def test_oom_descent_preserves_effective_batch_until_one() -> None:
    result = _profile_script(
        """
from perfseer_v3.dataset_pack.contracts import AttemptStatus,FailureStage,FingerprintBundle,GpuCleanupEvidence,LABEL_RUN_RECORD_VERSION,LabelRunRecord
from perfseer_v3.dataset_pack.fingerprints import canonical_sha256
from perfseer_v3.dataset_pack.repair import next_oom_repair
from perfseer_v3.dataset_pack.sampler import build_target_manifest
from perfseer_v3.dataset_pack.task_registry import load_task_registry
c=next(r for r in build_target_manifest().candidates if r.microbatch_size>=64)
t=next(r for r in load_task_registry().entries if r.task_id==c.task_id)
root=c.candidate_id; effective=c.microbatch_size*c.gradient_accumulation_steps; index=0
while c.microbatch_size>1:
 f=FingerprintBundle(c.source_sha256,'1'*64,'2'*64,'3'*64,'4'*64,'5'*64)
 cleanup=GpuCleanupEvidence(0.0,0.0,64.0,2.0,2.0,True,0,True)
 record=LabelRunRecord(LABEL_RUN_RECORD_VERSION,canonical_sha256({'c':c.candidate_id}),c.candidate_id,AttemptStatus.OOM,FailureStage.ALLOCATOR,f,'GPU-test',c.target_hardware_id,c.candidate_id,c.candidate_id,c.coverage_cell_ids,c.task_id,c.execution['mode'],False,t.expected_train_examples,c.microbatch_size,c.gradient_accumulation_steps,5,(1,2),(3,4,5),(),(),(),(),cleanup)
 record.validate_against_configuration(c,t)
 repair=next_oom_repair(c,record,repair_index=index,root_candidate_id=root)
 c=repair.candidate; index+=1
 assert c.microbatch_size*c.gradient_accumulation_steps==effective
assert c.microbatch_size==1 and c.gradient_accumulation_steps==effective
print(index,effective)
"""
    )
    depth, effective = map(int, result.stdout.split())
    assert depth >= 6 and effective in {32, 64, 128, 256, 512}


def test_supervisor_persists_model_error_crash_and_timeout_locally() -> None:
    result = _profile_script(
        """
from dataclasses import replace
import json, pathlib, signal, subprocess, tempfile
from types import SimpleNamespace
from unittest import mock
from perfseer_v3.dataset_pack.contracts import GpuCleanupEvidence
from perfseer_v3.dataset_pack.fingerprints import canonical_sha256
from perfseer_v3.dataset_pack.label_worker import WORKER_ENVELOPE_VERSION
from perfseer_v3.dataset_pack.prepared_view import PREPARED_VIEW_VERSION,PreparedViewManifest
from perfseer_v3.dataset_pack.sampler import build_target_manifest
from perfseer_v3.dataset_pack.storage import atomic_write_json
from perfseer_v3.dataset_pack.supervisor import AttemptSupervisor
from perfseer_v3.dataset_pack.task_registry import MLEBENCH_METADATA_REVISION,load_task_registry
from perfseer_v3.dataset_pack.v100_runner import TelemetryReading
c=build_target_manifest().candidates[0]; t=next(r for r in load_task_registry().entries if r.task_id==c.task_id)
payload={'version':PREPARED_VIEW_VERSION,'task_id':t.task_id,'kaggle_slug':t.kaggle_slug,'modality':t.modality,'mlebench_revision':MLEBENCH_METADATA_REVISION,'task_schema_sha256':c.task_schema_sha256,'archive_sha256':'a'*64,'source_example_count':4096,'prepared_example_count':4096,'sampling_with_replacement':False,'recipe_sha256':'b'*64,'samples_sha256':'c'*64}
v=PreparedViewManifest(**payload,dataset_fingerprint=canonical_sha256(payload)); v.validate()
cleanup=GpuCleanupEvidence(0.0,0.0,64.0,2.0,2.0,True,0,True)
class Probe:
 physical_index=0; gpu_uuid='GPU-fixture'; hardware_fingerprint='f'*64; hardware_provenance={}
 def read(self): return TelemetryReading(0.0,0.0,0.0,0,())
class Finished:
 pid=2000000000
 def __init__(self,code): self.returncode=code
 def poll(self): return self.returncode
 def wait(self,timeout=None): return self.returncode
mode={'value':'failed','calls':0}
def popen(command,**kwargs):
 mode['calls']+=1; output=pathlib.Path(command[command.index('--output')+1])
 if mode['value']!='crash':
  status=mode['value']; body={'failure_stage':'forward','reason_code':'fixture.DeterministicModelError','global_integrity_failure':False}
  envelope={'version':WORKER_ENVELOPE_VERSION,'candidate_id':c.candidate_id,'status':status,'payload':body}
  atomic_write_json(output,{**envelope,'envelope_sha256':canonical_sha256(envelope)})
 return Finished(21)
class Hanging(Finished):
 def __init__(self): super().__init__(-15)
 def poll(self): return None
def killpg(pid,sig):
 if sig==0: raise ProcessLookupError
with tempfile.TemporaryDirectory() as d:
 root=pathlib.Path(d); public=root/'public'; prepared=root/'prepared'; public.mkdir(); prepared.mkdir()
 with mock.patch('perfseer_v3.dataset_pack.supervisor.subprocess.Popen',side_effect=popen),mock.patch('perfseer_v3.dataset_pack.supervisor.os.killpg',side_effect=killpg),mock.patch('perfseer_v3.dataset_pack.supervisor.wait_for_gpu_cleanup',return_value=cleanup),mock.patch('perfseer_v3.dataset_pack.supervisor._bind_native_a10_provenance',side_effect=lambda record,*args,**kwargs: replace(record,production_eligible=kwargs.get('production_eligible',True))):
  model=AttemptSupervisor(root).run(c,t,v,public_directory=public,prepared_directory=prepared,archive_sha256='a'*64,probe=Probe(),attempt_index=0)
  assert model.status.value=='quarantined'
  mode['value']='crash'; before=mode['calls']
  crash=AttemptSupervisor(root).run(c,t,v,public_directory=public,prepared_directory=prepared,archive_sha256='a'*64,probe=Probe(),attempt_index=10)
  assert crash.status.value=='interrupted' and mode['calls']-before==2
 with mock.patch('perfseer_v3.dataset_pack.supervisor.subprocess.Popen',side_effect=lambda *a,**k:Hanging()),mock.patch('perfseer_v3.dataset_pack.supervisor.os.killpg',side_effect=killpg),mock.patch('perfseer_v3.dataset_pack.supervisor.wait_for_gpu_cleanup',return_value=cleanup),mock.patch('perfseer_v3.dataset_pack.supervisor._bind_native_a10_provenance',side_effect=lambda record,*args,**kwargs: replace(record,production_eligible=kwargs.get('production_eligible',True))):
  timeout=AttemptSupervisor(root,timeout_seconds=.05,no_progress_seconds=.05).run(c,t,v,public_directory=public,prepared_directory=prepared,archive_sha256='a'*64,probe=Probe(),attempt_index=20)
  assert timeout.status.value=='timed_out'
 print(model.status.value,crash.status.value,timeout.status.value,len(tuple((root/'attempts/failed').glob('*.json'))))
"""
    )
    assert result.stdout.split() == ["quarantined", "interrupted", "timed_out", "3"]


def test_digest_only_four_a10_yaml_and_pvc(tmp_path: Path) -> None:
    path = ROOT / "scripts/render_a10_nonvision_nautilus_job.py"
    spec = importlib.util.spec_from_file_location("render_nonvision", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    image = "gitlab-registry.nrp-nautilus.io/group/project@sha256:" + "b" * 64
    revision = "a" * 40
    pilot = tmp_path / "pilot.yaml"
    final = tmp_path / "chunk43.yaml"
    export = tmp_path / "export.yaml"
    assert module.main(["--output", str(pilot), "--namespace", "test-ns", "--image", image, "--source-revision", revision, "--mode", "pilot"]) == 0
    assert module.main(["--output", str(final), "--namespace", "test-ns", "--image", image, "--source-revision", revision, "--mode", "chunk", "--chunk-index", "43"]) == 0
    assert module.main(["--output", str(export), "--namespace", "test-ns", "--image", image, "--source-revision", revision, "--mode", "export"]) == 0
    module.verify_job(yaml.safe_load(pilot.read_text()), mode="pilot")
    module.verify_job(yaml.safe_load(final.read_text()), mode="chunk", chunk_index=43)
    module.verify_job(yaml.safe_load(export.read_text()), mode="export")
    pvc = yaml.safe_load((ROOT / "k8s/a10-nonvision-4gpu-labeler-pvc.yaml").read_text())
    assert pvc["spec"]["accessModes"] == ["ReadWriteMany"]
    assert pvc["spec"]["resources"]["requests"]["storage"] == "700Gi"


def test_partial_export_joins_label_config_source_and_reconstructs() -> None:
    result = _profile_script(
        """
from dataclasses import asdict,replace
import json,pathlib,tempfile
from unittest import mock
from perfseer_v3.dataset_pack.a10_export import export_release,verify_release
from perfseer_v3.dataset_pack.contracts import AttemptStatus,EPOCH_MEASUREMENT_VERSION,EpochMeasurement,FailureStage,FingerprintBundle,GpuCleanupEvidence,LABEL_RUN_RECORD_VERSION,LabelRunRecord,TelemetrySample
from perfseer_v3.dataset_pack.fingerprints import canonical_sha256
from perfseer_v3.dataset_pack.sampler import build_target_manifest
from perfseer_v3.dataset_pack.storage import atomic_write_json
from perfseer_v3.dataset_pack.task_registry import load_task_registry
from perfseer_v3.dataset_pack.workflow import _new_slot,_save_slot
root=next(r for r in build_target_manifest().candidates if r.family_id=='selu_mlp' and r.execution['mode']=='eager')
task=next(r for r in load_task_registry().entries if r.task_id==root.task_id)
batches=(task.expected_train_examples+root.microbatch_size-1)//root.microbatch_size
steps=(batches+root.gradient_accumulation_steps-1)//root.gradient_accumulation_steps
sample=TelemetrySample(0.0,1.0,50.0,40.0,1024.0,0)
epochs=tuple(EpochMeasurement(EPOCH_MEASUREMENT_VERSION,e,True,100.0+e,task.expected_train_examples,batches,batches,steps,True,True,True,(sample,),900.0,root.execution['backend_id'],root.execution['backend_id'],False) for e in (3,4,5))
fingerprints=FingerprintBundle(root.source_sha256,'1'*64,'2'*64,'3'*64,'4'*64,'5'*64)
cleanup=GpuCleanupEvidence(0.0,0.0,64.0,2.0,2.0,True,0,True)
draft=LabelRunRecord(LABEL_RUN_RECORD_VERSION,canonical_sha256({'candidate':root.candidate_id}),root.candidate_id,AttemptStatus.ACCEPTED,FailureStage.NONE,fingerprints,'GPU-local',root.target_hardware_id,root.candidate_id,root.candidate_id,root.coverage_cell_ids,root.task_id,root.execution['mode'],False,task.expected_train_examples,root.microbatch_size,root.gradient_accumulation_steps,5,(1,2),(3,4,5),(1,2,3,4,5),(1,2,3,4,5),(1,2,3,4,5),epochs,cleanup,None,False)
record=replace(draft,targets=draft.aggregate_targets()); record.validate_against_configuration(root,task)
with tempfile.TemporaryDirectory() as directory:
 workspace=pathlib.Path(directory)/'workspace'; output=pathlib.Path(directory)/'output'
 _save_slot(workspace/'state/slots'/f'{root.candidate_id}.json',_new_slot(root))
 atomic_write_json(workspace/'attempts/accepted'/f'{root.candidate_id}.json',asdict(record))
 with mock.patch('perfseer_v3.dataset_pack.a10_export.build_campaign_contract',return_value={'contract_sha256':'6'*64}):
  release=export_release(workspace,pathlib.Path.cwd(),output,complete=False)
 checked=verify_release(release['archive'])
 print(json.dumps({'labels':checked['label_count'],'reconstructed':checked['reconstructed_configuration_count'],'suffix':release['archive'].endswith('.tar.zst')}))
""",
        timeout=900,
    )
    import json

    assert json.loads(result.stdout) == {"labels": 1, "reconstructed": 1, "suffix": True}


def test_image_and_runbook_are_nonvision_and_no_cluster_mutation_was_scripted() -> None:
    dockerfile = (ROOT / "containers/a10-nonvision-4gpu-labeler/Dockerfile").read_text()
    runbook = (ROOT / "docs/PerfSeer_V3_Native_Nautilus_A10_NonVision_4GPU_Labeling_Runbook.md").read_text()
    monitor = (ROOT / "scripts/monitor_a10_nonvision_job.sh").read_text()
    assert "native_a10_nonvision_4gpu_v1" in dockerfile
    assert "pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime@sha256:" in dockerfile
    assert "COPY kaggle.json" not in dockerfile
    assert "kubectl exec" not in monitor and "kubectl cp" not in monitor
    for phrase in ("docker login", "docker buildx build", "docker push", "kubectl apply", "kubectl get job", "kubectl describe pod", "nohup", "verify-export"):
        assert phrase in runbook
