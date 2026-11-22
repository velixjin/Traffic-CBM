# Revised experiment protocol

The cache builder accepts an optional directory of ordered split ID files.
To reproduce the reported Traffic-CBM scores, supply the registered traffic
flows and labels in their original order. Dataset files, processed arrays, and
saved weights are separate inputs.

```bash
python reproduction/prepare_cache.py --dataset ISCX-VPN-App --corrected \
  --aligned-ids splits/ISCX-VPN-App --output outputs/reproduction/cache

python reproduction/train_revision.py \
  --cache outputs/reproduction/cache/ISCX-VPN-App_corrected_full.pt \
  --model ga2m --seed 40 --epochs 80 --patience 10 --batch-size 32 \
  --pretrain-ckpt outputs/reproduction/pretrain/best_pretrain.pt \
  --output outputs/ISCX-VPN-App/seed40
```

The cache builder uses flow statistics and byte tokens from feature
extraction. The corrected sequence path reads each capture using its link
type. Statistics and continuous sequence features are scaled from training
flows. The trainer selects the checkpoint by validation macro-F1 and reports
held-out performance. Run seeds 40, 41, and 42 with separate output directories.
The reference batch sizes are 32 on ISCX-VPN(App), 64 on ISCX-VPN(Service),
and 256 on the other four benchmarks.

## Encoder initialization

Build the corrected cache for each benchmark into
`outputs/reproduction/cache` before assembling the pretraining data. Set
`TRAFFIC_CBM_VISQUIC_ROOT` and `TRAFFIC_CBM_CICIDS_ROOT` to your local
directories. The CIC-IDS2017 reader expects classic PCAP files named
`*.classic.pcap`. Use dataset directories consistent with the feature tables
and split IDs used by the downstream experiments.

```bash
python reproduction/audit_sources.py --output outputs/reproduction/source_audit
python reproduction/prepare_clean_pretrain.py
python reproduction/build_visquic_cache.py
python reproduction/build_cicids_cache.py
python reproduction/build_pretrain_all.py
python reproduction/build_pretrain_all3.py
python reproduction/pretrain_all_branches.py \
  --data outputs/reproduction/pretrain_all3.pt \
  --meta outputs/reproduction/cache/ISCX-VPN-App_corrected_full.pt \
  --epochs 10 --batch-size 96 --output outputs/reproduction/pretrain
```

The audit verifies candidate duplicates against full packet bytes.
The in-domain builder excludes held-out duplicates and assembles the
remaining training and validation streams. The pretrainer reconstructs
masked statistics, sequence features, and byte tokens.
