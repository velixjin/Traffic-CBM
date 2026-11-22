# Traffic-CBM architecture

Traffic-CBM maps statistics, packet sequences, and byte tokens to 18
learned scalars. Each scalar receives input from a defined part of the
flow representation. This identifies the source of the evidence used by
the scalar. Its value is learned from data.

| Evidence | Concepts | Model input |
| --- | ---: | --- |
| Flow statistics | 6 | Counts, packet lengths, byte volumes, timing, TCP flags, and header/window statistics, extracted from up to 200 packets |
| Packet sequences | 4 | Separate sequences for volume, timing, direction, and TCP behavior, using up to 100 packets |
| Packet bytes | 5 | One scalar from the byte window of each of the first five packets |
| Cross-packet bytes | 3 | One scalar per aligned window segment across those five packets |

The default byte window contains 64 bytes, starting at offset 38. Adjacent
byte pairs produce 63 tokens. The cross-packet encoder divides the token
window into three segments of 21 tokens and combines the same segment
across packets.

The [model](../model/traffic_cbm.py) concatenates the scalars in the order
shown above. Its GA2M prediction head computes each class logit as the sum
of a bias, one learned term per concept, and one learned term per pair of
concepts. Inspect a term for the class of interest to see its contribution
to that class logit. Use [the analysis runner](../eval/analyze_contrib.py)
to examine individual predictions and aggregate contribution patterns.

The default model configuration is in [train/config.json](../train/config.json).
See the [reproduction guide](../reproduction/README.md) for the paper's
pretraining and fine-tuning commands.
