# syntax=docker/dockerfile:1.7
FROM --platform=linux/amd64 pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime@sha256:b85566342b86d13a67712e9315d40cdc2dad7f8d86df1aff3831f80835edbcca
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/opt/perfseer-packages OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
WORKDIR /opt/perfseer
COPY . .
RUN python verify_package.py \
    && python -m pip install --no-deps --no-build-isolation --target /opt/perfseer-packages . \
    && python verify_package.py --runtime \
    && chmod -R a-w,a+rX /opt/perfseer /opt/perfseer-packages
USER 1000:1000
ENTRYPOINT ["bash", "/opt/perfseer/train.sh"]
