ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.03-py3@sha256:3c90e38f5ec51e51d1c73bd7eb3d83674a254f451147c5cadc4344314258a112
FROM ${BASE_IMAGE}

WORKDIR /workspace/PerfSeer-predictor
RUN python -m pip install --no-cache-dir networkx numpy scikit-learn pyyaml nvidia-ml-py tqdm torch_geometric ogb
COPY README.md pyproject.toml ./
COPY nrp_calibration_pack ./nrp_calibration_pack
COPY scripts ./scripts
COPY src ./src
ENV PYTHONPATH=/workspace/PerfSeer-predictor/src:/workspace/PerfSeer-predictor

CMD ["python", "nrp_calibration_pack/profile/run_profile.py", "--help"]
