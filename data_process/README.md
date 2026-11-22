# Feature extraction

Run from the repository root after [capture preparation](_prepare/README.md).
The runner reads `dataset/<name>/flows_filtered/` and writes to `data/<name>/`.
Set `TRAFFIC_CBM_FLOW_ROOT` and `TRAFFIC_CBM_FEATURE_ROOT` to override the roots.
The optional second argument sets the number of worker processes.

```bash
bash data_process/run.sh USTC-TFC2016 4
```

| Output under `data/<name>/` | Contents |
| --- | --- |
| `stats/stats.parquet` | CICFlowMeter statistics keyed by flow ID |
| `seq/seq.parquet` | Per-packet length, inter-arrival time, direction, TCP flags, payload length, header length, and window size |
| `bytes/bytes.parquet` | Byte bigram tokens keyed by flow ID |
| `manifest.csv` | Class, split, and modality availability for each flow |
| `train_ids.txt`, `val_ids.txt`, `test_ids.txt` | Ordered splits restricted to flows with all three feature sources |
| `label2id.json` | The class mapping from capture preparation |

The feature directory contains the inputs for supervised training and evaluation.
The revised cache builder can additionally read the prepared PCAPs to extract
packet sequences using each file's link type.

The runner uses `data_process/config.json`, adds a separator between packets
for byte tokenization, and records the effective settings in each modality
directory. The configuration uses up to 200 packets for flow statistics and
100 for packet sequences. It extracts up to 64 bytes from offset 38 in each of
the first five packets for byte bigrams. The byte extractor's `--sanitize_mode`
option can zero or randomize connection fields. Use the separate capture-level
masker for TLS ClientHello hostnames.

To extract a single split, pass its ID file to the corresponding extractor.
Use a dedicated output directory for each extractor. Rerunning an extractor
replaces its existing outputs. The sequence extractor also supports `--resume`.

```bash
python data_process/build_seq.py --config data_process/config.json \
  --flows_dir dataset/USTC-TFC2016/flows_filtered \
  --ids dataset/USTC-TFC2016/train_ids.txt \
  --out_dir outputs/train_sequences --workers 4
```

Run `python data_process/build_seq.py --help`, or the other extractor's help
command, to see the available overrides.
