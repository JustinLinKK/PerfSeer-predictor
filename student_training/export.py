"""Winner deployment export with eager, isolated interpreter and batch-independence checks."""
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import torch
from support import atomic_save, packed, sha


def export_verified(model, base, data, rows, output, metadata):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    rows = torch.as_tensor(rows[:8])
    inputs = packed(base, data, rows, "cpu")
    temporary, target = output / "student.ts.tmp", output / "student.ts"
    script = torch.jit.script(model)
    torch.jit.save(script, str(temporary), _extra_files={"student_config.json": json.dumps(metadata, sort_keys=True)})
    extra = {"student_config.json": ""}
    loaded = torch.jit.load(str(temporary), map_location="cpu", _extra_files=extra).eval()
    if json.loads(extra["student_config.json"]) != metadata:
        raise AssertionError("Embedded metadata changed")
    with torch.no_grad():
        eager, deployed = model(*inputs), loaded(*inputs)
        if eager.keys() != deployed.keys():
            raise AssertionError("Deployment output contract changed")
        for name in eager:
            torch.testing.assert_close(eager[name], deployed[name], rtol=1e-5, atol=1e-5)
        singles = torch.cat([loaded(*packed(base, data, row.reshape(1), "cpu"))["prediction"] for row in rows])
    prediction = deployed["prediction"]
    torch.testing.assert_close(singles, prediction, rtol=1e-5, atol=1e-5)
    if not (torch.isfinite(prediction).all() and (prediction[:, [0, 1, 2, 4, 5]] > 0).all()
            and (prediction[:, 3] >= 0).all() and (prediction[:, 3] <= 100).all()
            and torch.isfinite(deployed["memory_upper"]).all()
            and (deployed["memory_upper"] >= prediction[:, 4:6]).all()):
        raise AssertionError("Invalid physical predictions")
    torch.testing.assert_close(prediction[:, 2], prediction[:, 0] * inputs[-1].float())
    with tempfile.TemporaryDirectory(prefix="isolated-export-", dir=output) as scratch:
        scratch = Path(scratch)
        atomic_save(inputs, scratch / "inputs.pt")
        code = ("import sys,torch;torch.set_num_threads(1);"
                "m=torch.jit.load(sys.argv[1],map_location='cpu').eval();"
                "x=torch.load(sys.argv[2],weights_only=True,map_location='cpu');"
                "torch.save(m(*x)['prediction'].detach(),sys.argv[3]);"
                "print([k for k in sys.modules if k.startswith('perfseer')])")
        process = subprocess.run([sys.executable, "-I", "-c", code, str(temporary.resolve()),
                                  str(scratch / "inputs.pt"), str(scratch / "prediction.pt")],
                                 cwd=scratch, capture_output=True, text=True, timeout=120)
        if process.returncode or process.stdout.strip() != "[]":
            raise AssertionError(f"Package-free deployment failed: {process.stdout} {process.stderr}")
        torch.testing.assert_close(torch.load(scratch / "prediction.pt", weights_only=True), prediction,
                                   rtol=1e-5, atol=1e-5)
    temporary.replace(target)
    return {"scripted_eager_verified": True, "batch_independence_verified": True,
            "clean_interpreter_verified": True, "physical_outputs_verified": True, "fp32": True,
            "student_sha256": sha(target)}
