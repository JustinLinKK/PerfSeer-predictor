# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE=pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime@sha256:b85566342b86d13a67712e9315d40cdc2dad7f8d86df1aff3831f80835edbcca

FROM --platform=linux/amd64 ${BASE_IMAGE}
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/opt/perfseer/repository/src
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates=20260601~24.04.1 \
       tini=0.19.0-1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/perfseer/repository
COPY pyproject.toml ./
COPY src ./src
COPY scripts/train_perfseer_v3_combined_dev.py ./scripts/train_perfseer_v3_combined_dev.py
COPY containers/perfseer-v3-combined-training/build-manifest.json /opt/perfseer/build-manifest.json
RUN python -c 'import numpy,torch,yaml; from perfseer_v3.model import SeerNetV3' \
    && python -c 'import gzip,hashlib,json,pathlib; root=pathlib.Path("src/perfseer_v3/dataset_with_label/ready_for_train"); manifest=json.loads((root/"dataset_manifest.json").read_text()); assert manifest["total_rows"]==36409; assert manifest["split_files"]["train"]["rows"]==32768; assert manifest["split_files"]["validation"]["rows"]==3641; [(lambda p,expected: (_ for _ in ()).throw(AssertionError(p)) if hashlib.sha256(p.read_bytes()).hexdigest()!=expected else None)(root/value["path"],value["sha256"]) for value in manifest["split_files"].values()]; assert sum(1 for _ in gzip.open(root/manifest["split_files"]["train"]["path"],"rt"))==32768; assert sum(1 for _ in gzip.open(root/manifest["split_files"]["validation"]["path"],"rt"))==3641' \
    && chmod -R a-w /opt/perfseer/repository \
    && chmod 0555 scripts/train_perfseer_v3_combined_dev.py \
    && test -z "$(find /opt/perfseer/repository -type f \( -name 'kaggle.json' -o -name '*.pem' -o -name '*.key' -o -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' -o -name '*.safetensors' \) -print -quit)"
ENTRYPOINT ["/usr/bin/tini", "--", "python", "/opt/perfseer/repository/scripts/train_perfseer_v3_combined_dev.py"]
CMD ["--output-dir", "/outputs", "--epochs", "3", "--train-samples", "12", "--validation-samples", "4", "--batch-size", "4", "--families", "selu_mlp,tabtransformer,mixture_density_network", "--capacity", "T1", "--device", "cuda", "--amp", "bfloat16"]
