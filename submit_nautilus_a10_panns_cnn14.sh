#!/usr/bin/env bash
set -euo pipefail
export PERFSEER_MODALITY_WRAPPER=audio
export PERFSEER_FAMILY_WRAPPER=panns_cnn14
exec "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)/scripts/nautilus_a10_modality_controller.sh" "$@"
