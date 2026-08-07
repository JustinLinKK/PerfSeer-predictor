#!/usr/bin/env bash
set -euo pipefail
export PERFSEER_MODALITY_WRAPPER=tabular
exec "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/scripts/nautilus_a10_modality_controller.sh" "$@"
