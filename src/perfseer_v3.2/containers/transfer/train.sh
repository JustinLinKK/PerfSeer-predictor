#!/usr/bin/env bash
set -euo pipefail
transfer_output=${TRANSFER_OUTPUT:-/outputs/run}
mkdir -p "$transfer_output"
exec > >(tee -a "$transfer_output/training.log") 2>&1
cd /opt/perfseer
python verify_package.py
exec python -m perfseer_v32.transfer_training train --dataset /opt/perfseer/dataset \
  --base-checkpoint /opt/perfseer/weights/base-teacher.pt --output "$transfer_output" --resume "$@"
