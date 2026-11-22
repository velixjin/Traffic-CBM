# Traffic-CBM

Traffic-CBM classifies encrypted traffic using flow statistics, packet sequences,
and byte bigrams. Its 18-dimensional concept bottleneck has separate evidence
paths for these inputs. The GA2M head exposes the main effects and pairwise
interactions that make up a class score.

See the [architecture guide](docs/architecture.md) for the evidence paths
and how to read the prediction scores.

## Setup

The reference training environment uses Linux, Python 3.10, PyTorch 2.1.1,
and CUDA 12.1. Run the commands below from the repository root.

```bash
conda env create -f environment.yml
conda activate traffic-cbm
```

On other platforms, create a Python 3.10 environment and install
PyTorch 2.1.1 for that platform, then run
`python -m pip install -r requirements.txt`. The standard training and
evaluation commands below support CPU execution. The revised experiment
training and pretraining runners require CUDA.

Paths default to `dataset/` for prepared flows and `data/` for feature
tables. To override these paths, copy `.env.example` to `.env`, edit it,
and load it from the repository root with `set -a; source .env; set +a`.

## Prepare data

Obtain the datasets from their original sources. For captures already split
into individual flows, put one file per flow under a class-named directory.

```bash
python data_process/_prepare/prepare_dataset.py \
  --pre_sliced raw/ISCX-VPN-App \
  --out "${TRAFFIC_CBM_FLOW_ROOT:-dataset}/ISCX-VPN-App" --min_bytes 256

bash data_process/run.sh ISCX-VPN-App
```

For raw captures, pass `--raw` and your own `--splitcap_exe` path.
For CipherSpectrum, mask ClientHello hostnames before extracting features.
The [capture preparation instructions](data_process/_prepare/README.md) cover
both cases. The feature runner writes the manifest and the aligned
`train_ids.txt`, `val_ids.txt`, and `test_ids.txt`. See the
[feature pipeline](data_process/README.md) for the output layout.

## Train and evaluate

Use the supervised runner for a new dataset. Supply `--pretrain_ckpt` to
initialize the encoders from a compatible checkpoint.

```bash
python train/train.py --config train/config.json \
  --data_dir "${TRAFFIC_CBM_FEATURE_ROOT:-data}/ISCX-VPN-App" \
  --output outputs/ISCX-VPN-App --batch_size 32

python eval/eval_ckpt.py --ckpt_dir outputs/ISCX-VPN-App \
  --data_dir "${TRAFFIC_CBM_FEATURE_ROOT:-data}/ISCX-VPN-App" --split test
```

The [revised experiment instructions](reproduction/README.md) explain the
link-type-aware flow cache, three-branch pretraining, and aligned fine-tuning.
