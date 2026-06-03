ARG BASE_IMAGE=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
FROM ${BASE_IMAGE}

WORKDIR /workspace
RUN python -m pip install --no-cache-dir networkx numpy scikit-learn pyyaml pynvml
COPY nrp_calibration_pack /workspace/nrp_calibration_pack
ENV PYTHONPATH=/workspace

CMD ["python", "/workspace/nrp_calibration_pack/profile/run_profile.py", "--help"]
