# syntax=docker/dockerfile:1.7
FROM --platform=linux/amd64 pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime@sha256:b85566342b86d13a67712e9315d40cdc2dad7f8d86df1aff3831f80835edbcca
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/opt/perfseer-packages
WORKDIR /opt/perfseer
COPY pyproject.toml ./
COPY src ./src
COPY tests/test_perfseer_v31.py ./tests/test_perfseer_v31.py
RUN python -m pip install --no-deps --no-build-isolation --target /opt/perfseer-packages . \
    && python -c 'import torch; from perfseer_v31.model import SeerNetV31; assert hasattr(torch.optim, "Muon")' \
    && python -m perfseer_v31.dataset verify --output /opt/perfseer/src/perfseer_v3.1/dataset_with_label/ready_for_train \
    && chmod -R a-w,a+rX /opt/perfseer /opt/perfseer-packages
ENTRYPOINT ["python", "-m", "perfseer_v31.runner"]
CMD ["--dataset", "/opt/perfseer/src/perfseer_v3.1/dataset_with_label/ready_for_train", "--output", "/outputs/perfseer-v31-a100", "--resume"]
