#!/usr/bin/env bash
set -euo pipefail
exec python -u "$(dirname "${BASH_SOURCE[0]}")/prepare_dataset.py" "$@"
