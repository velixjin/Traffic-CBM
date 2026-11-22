#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
dataset=${1:?Usage: bash data_process/run.sh DATASET [WORKERS]}
workers=${2:-8}
flow_dir="${TRAFFIC_CBM_FLOW_ROOT:-dataset}/$dataset"
feature_dir="${TRAFFIC_CBM_FEATURE_ROOT:-data}/$dataset"
config=data_process/config.json
python -u data_process/build_stats.py --config "$config" --flows_dir "$flow_dir/flows_filtered" --out "$feature_dir/stats" --workers "$workers"
python -u data_process/build_seq.py --config "$config" --flows_dir "$flow_dir/flows_filtered" --out_dir "$feature_dir/seq" --workers "$workers"
python -u data_process/build_bytes.py --config "$config" --flows_dir "$flow_dir/flows_filtered" --out_dir "$feature_dir/bytes" --workers "$workers" --sep
python -u data_process/build_manifest.py --dataset_dir "$flow_dir" --data_dir "$feature_dir" --out "$feature_dir/manifest.csv"
