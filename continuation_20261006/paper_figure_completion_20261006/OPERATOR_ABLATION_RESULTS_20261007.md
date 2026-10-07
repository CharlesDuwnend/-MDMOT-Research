# Frozen SRC Operator Addendum Results (2026-10-07)

This is a post hoc, inference-only diagnostic against frozen F00 on VisDrone
test-dev. It is not a new trained method, a test-dev winner selection, or an
automatic manuscript result. The original five-arm outputs were not modified.

## Completion and contracts

- Addendum receipt: `/home/chenhc/leaf_v5_evidence_completion_20261006_v1/experiment/operator_ablation_20261007_v1/artifacts/receipt.json`
- Receipt status: `PASS_OPERATOR_ADDENDUM_COMPLETE`; launcher exit: `0`
- Both arms: 17 sequences and 6,635 frames; evaluator ground-truth contract matched.
- Both arms ran under the same verified physical A100 UUID: `GPU-1b297aba-ae7e-e326-5903-476f1bb683d7` (physical index 1).
- Shared-GPU execution was explicitly authorized; both hardware receipts recorded at least 8 GiB free at launch. Physical GPU2 was not eligible or used.
- No training, refit, calibration, threshold search, significance claim, or winner selection was performed.

## Paired metrics against F00

| Arm | MOTA | IDF1 | FP | FN | IDs | Delta MOTA (pp) | Delta IDF1 (pp) |
|---|---:|---:|---:|---:|---:|---:|---:|
| F00 control | 58.0251 | 71.3737 | 26,047 | 69,497 | 791 | 0.0000 | 0.0000 |
| `pair_shuffle` | 57.9571 | 71.2868 | 27,250 | 68,364 | 877 | -0.0680 | -0.0869 |
| `hard_update` | 58.0251 | 71.3737 | 26,047 | 69,497 | 791 | 0.0000 | 0.0000 |

`pair_shuffle` changed all 17 prediction files. Its operator contract recorded
4,925 birth events, 4,405 eligible multi-source events, 4,405 non-identity
permutations, and 209,606 changed source assignments. The paired output shows a
small aggregate degradation, with fewer misses but more false positives and ID
switches.

`hard_update` recorded 221,039 committed updates, and all 221,039 had unit
responsibility. All 17 prediction files were byte-identical to F00, so the
hard-update intervention had no observable trajectory effect under this frozen
configuration.

## Artifacts

- Paired sequence deltas: `operator_ablation_20261007_v1/artifacts/paired_sequence_deltas.csv`
- Paired flight-prefix deltas: `operator_ablation_20261007_v1/artifacts/paired_flight_deltas.csv`
- Paired summary: `operator_ablation_20261007_v1/artifacts/paired_summary.json`
- Addendum receipt SHA-256: `2dcab6e9467562e32f3e11cead4a175eae576c5288fd9d9cc49c2cc19d2127f5`
- Paired summary SHA-256: `e812f4ec9d64f85fed9d3d2e576ce54fa223360f1d87f914f733a3109bdf849c`

The results support a mechanism diagnostic about source ownership and the
responsibility update, but do not by themselves justify changing the method or
adding a new main-paper table. Inclusion remains an author decision.
