# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE=pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime@sha256:b85566342b86d13a67712e9315d40cdc2dad7f8d86df1aff3831f80835edbcca

FROM --platform=linux/amd64 ${BASE_IMAGE} AS mlebench
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates=20260601~24.04.1 \
       git=1:2.43.0-1ubuntu7.3 \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --filter=blob:none https://github.com/openai/mle-bench.git /opt/mle-bench \
    && git -C /opt/mle-bench checkout --detach 507f92e1138bb6e40dac5c6ee7a6758e6424bf97 \
    && test "$(git -C /opt/mle-bench rev-parse HEAD)" = 507f92e1138bb6e40dac5c6ee7a6758e6424bf97 \
    && test -z "$(git -C /opt/mle-bench status --porcelain --untracked-files=all)"
RUN python -c 'import hashlib,json,pathlib; r=pathlib.Path("/opt/mle-bench"); f=[p for p in sorted(r.rglob("*")) if p.is_file() and ".git" not in p.parts and ".perfseer-revision.json" not in p.parts]; d={p.relative_to(r).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in f}; h=hashlib.sha256(json.dumps(d,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest(); (r/".perfseer-revision.json").write_text(json.dumps({"revision":"507f92e1138bb6e40dac5c6ee7a6758e6424bf97","tree_sha256":h},sort_keys=True)+"\n")' \
    && rm -rf /opt/mle-bench/.git

FROM --platform=linux/amd64 ${BASE_IMAGE} AS runtime
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/opt/perfseer/repository/src:/opt/mle-bench \
    PERFSEER_A10_IMAGE_PROFILE=native_a10_speech_v2 \
    PERFSEER_LABELER_PROFILE=native_a10_speech_v2 \
    KAGGLE_CONFIG_DIR=/run/secrets/kaggle
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates=20260601~24.04.1 \
       ffmpeg=7:6.1.1-3ubuntu5 \
       libgomp1=14.2.0-4ubuntu2~24.04.1 \
       libsndfile1=1.2.2-1ubuntu5.24.04.1 \
       tini=0.19.0-1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/perfseer/repository
COPY containers/a10-labeler/requirements.lock /tmp/requirements.lock
RUN python -m pip install --break-system-packages --no-cache-dir --require-hashes \
      --extra-index-url https://download.pytorch.org/whl/cu128 \
      -r /tmp/requirements.lock \
    && python -m pip check \
    && rm /tmp/requirements.lock
COPY --from=mlebench /opt/mle-bench /opt/mle-bench
COPY pyproject.toml ./
COPY src ./src
COPY scripts/run_perfseer_v3_a10_labeling.py ./scripts/run_perfseer_v3_a10_labeling.py
COPY containers ./containers
COPY containers/a10-speech-v2-labeler/build-manifest.json /opt/perfseer/build-manifest.json
RUN python -m pip install --break-system-packages --no-cache-dir --no-deps . \
    && chmod 0555 scripts/run_perfseer_v3_a10_labeling.py \
    && test ! -e /opt/perfseer/repository/kaggle.json \
    && test ! -e /opt/perfseer/kaggle.json
ENTRYPOINT ["/usr/bin/tini", "--", "python", "/opt/perfseer/repository/scripts/run_perfseer_v3_a10_labeling.py"]
CMD ["analyze"]
