#!/usr/bin/env bash
set -euo pipefail
export PERFSEER_MODALITY_WRAPPER=generated
exec "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/scripts/nautilus_a10_modality_controller.sh" "$@"
