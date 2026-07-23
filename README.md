# Information

## A10 Student Deployment

`src/perfseer_student` owns the production student graph featurizer, source
encoder, model definition, export tool, registry loader, and strict CPU
TorchScript runtime. Source conversion reuses `perfseer_source_converter`.

The deployment contract is:

- raw node features: 53 (`23` operation categories + `30` continuous values)
- raw edge features: 3
- raw global features: 40 (`36` graph values + `4` precision categories)
- outputs: six training/inference targets; `train_mem` at index `1` is average
  used GPU VRAM in MiB
- artifact: `models/nvidia_a10/student_a10_cpu.torchscript.pt`
- device: CPU only

The artifact embeds input normalization and output de-normalization. Hardware
aliases, compute capability, VRAM range, schema, output index, path, and SHA-256
are recorded in `models/registry.json`.

Run the focused deployment verification with:

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

- Model size depends on GPU, optimizer, dtype

- FLOPs depends on model type, size, optimizer

| Available GPU |
| --- |
| NVIDIA A10 |
| NVIDIA A40 |
| NVIDIA A100 |
| NVIDIA L4 |
| NVIDIA L40 |
| NVIDIA L40S |
| NVIDIA RTX A4000 |
| NVIDIA RTX A5000 |
| NVIDIA RTX A6000 |
| NVIDIA RTX 4000 Ada Generation |
| NVIDIA RTX 5000 Ada Generation |
| NVIDIA RTX PRO 6000 Blackwell |
| NVIDIA Quadro RTX 6000 |
| NVIDIA Quadro RTX 8000 |
| NVIDIA Tesla T4 |
| NVIDIA Tesla V100 |

# TODO

## Dataset Construction - Not For Me

### Problems

> Muon is not used in dataset

> How to create all kinds of representative variants for Transformer, MLP, CNN, LSTM, RNN, Mamba...

- Focus 1: Variants of each layer and make combinations of these layers.

- Focus 2: Model branches

- Focus 3: Selecting Representative Models

- Focus 4: Data type

### Approaches

- Input: Pytorch Model Source Code

- Labels

| Labels | Training | Inference |
|--------|----------|-----------|
| Avg Training/Inference time of 1 epoch | Avg Training time of 1 epoch | Avg Inference time of 1 epoch |
| Peak/Avg SM Occupancy | Peak/Avg SM Occupancy | Peak/Avg SM Occupancy |
| GPU RAM Usage | GPU RAM Usage | GPU RAM Usage |
| Host Memory Usage | Host Memory Usage | Host Memory Usage |
| Data type for forward/backward propagation | Data type for forward/backward propagation | Data type for forward/backward propagation |
| Compile/Warmup Time |   | Compile/Warmup Time |
| Optimizer Type | Optimizer Type | Optimizer Type |
| Other labels: GPU type, CUDA Version |   |   |

## Others

- Converter for jax etc. (Optional)

# Done

## Dataset Sample

### Problems

- Nautilus is not available in many cases, so keep on refreshing requests all the time

### Approaches

> Sample dataset from available GPUs listed above on Nautilus Server

- Warmup for 1 epoch and get labels from 2nd epoch

- Skip devices with hardware failure and CUDA initialization failure

### Script Usage

> Input model file requirements

- Each `.py` file must define `make_model()`.
- `MODEL_ID` is optional; the file stem is used when it is missing.
- `INPUT_SHAPE` is optional; `--default-input-shape` is used when it is missing.

> Dataflow:

```text
local model folder
-> build manifest
-> upload models, manifest, profiler, and verifier to Nautilus PVC
-> submit GPU jobs with switching
-> generate labels
-> verify labels
-> copy remote labels back to local labels/<run_id>/
```

> Usage

```bash
python3 scripts/run_nautilus_folder_label_sampling.py \
  --models-dir /path/to/pytorch_model_files \
  --local-labels-dir labels \
  --namespace ecepxie \
  --pvc test-pvc \
  --gpus all-readme \
  --active-gpus 4 \
  --pending-timeout-seconds 300 \
  --min-successful-gpus 1
```

> Verify copied labels locally

```bash
python3 scripts/verify_sampled_labels.py labels/<run_id>
```
