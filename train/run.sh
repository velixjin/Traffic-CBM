#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python -u train/train.py --config train/config.json --data_root "${TRAFFIC_CBM_FEATURE_ROOT:-data}" "$@"
