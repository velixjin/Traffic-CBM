# Flow preparation

Use class-named subdirectories under a raw or pre-sliced dataset directory.
For raw captures, install Mono and obtain SplitCap from its distributor.
Pass its file path to `--splitcap_exe` or set `TRAFFIC_CBM_SPLITCAP`.
Use a dedicated output directory separate from the input captures.

```bash
python data_process/_prepare/prepare_dataset.py \
  --raw raw/ISCX-VPN-App --splitcap_exe /path/to/SplitCap.exe \
  --out dataset/ISCX-VPN-App --min_bytes 256
```

For pre-sliced CipherSpectrum flows, mask only the ClientHello hostname
bytes, then prepare the masked PCAPs.

```bash
python data_process/_prepare/mask_sni_dataset.py \
  --input_root raw/CipherSpectrum-AES128 --output_root raw/CipherSpectrum-AES128-masked \
  --mode zero

python data_process/_prepare/prepare_dataset.py \
  --pre_sliced raw/CipherSpectrum-AES128-masked \
  --out dataset/CipherSpectrum-AES128 --min_bytes 512
```

The preparation command writes one classic PCAP per retained flow to
`flows_filtered/<class>/`, the class labels to `label2id.json`, and three
ordered flow-ID files. Each ID has the form `class/filename_stem`. By default,
the command uses a minimum of three packets per flow, 30 flows per class,
and a class-stratified 80/10/10 split with seed 42.
Use 256 bytes as the PCAP file threshold for the two ISCX tasks and 512 for
the remaining four benchmarks. See `prepare_dataset.py --help` for options.
