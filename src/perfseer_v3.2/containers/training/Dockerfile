# syntax=docker/dockerfile:1.7
FROM --platform=linux/amd64 pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime@sha256:b85566342b86d13a67712e9315d40cdc2dad7f8d86df1aff3831f80835edbcca
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/opt/perfseer-packages
WORKDIR /opt/perfseer
COPY pyproject.toml ./
COPY src/perfseer_v3 ./src/perfseer_v3
COPY src/perfseer_v3.1/*.py src/perfseer_v3.1/*.json ./src/perfseer_v3.1/
COPY src/perfseer_v3.2/*.py src/perfseer_v3.2/*.md ./src/perfseer_v3.2/
COPY src/perfseer_v3.2/dataset_with_label/ready_for_train_12 ./src/perfseer_v3.2/dataset_with_label/ready_for_train_12
RUN python -m pip install --no-deps --no-build-isolation --target /opt/perfseer-packages . \
    && python -c 'import torch, numpy, yaml; from perfseer_v32.model import SeerNetV32; assert hasattr(torch.optim, "Muon")' \
    && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m perfseer_v32.dataset verify --workers 4 --output /opt/perfseer/src/perfseer_v3.2/dataset_with_label/ready_for_train_12 \
    && chmod -R a-w,a+rX /opt/perfseer /opt/perfseer-packages
USER 1000:1000
ENTRYPOINT ["python", "-m", "perfseer_v32.runner"]
CMD ["--dataset", "/opt/perfseer/src/perfseer_v3.2/dataset_with_label/ready_for_train_12", "--output", "/outputs/perfseer-v32-a100", "--resume"]
