# MDMT-COD paper tables (train-only evidence)

Source: frozen `/home/chenhc/mdmt_cod_revision_audit_20260926/artifacts_v2/summary.json`; no new split access. The tables are protocol diagnostics, not official MOT results.

| quantity | value | interpretation |
|---|---:|---|
| paired train sequences | 25 | independent pair units for aggregation |
| event ledger rows | 8,311 | one prefix event per confirmed local track |
| candidate-present events | 3,045 | source candidate pool was observable |
| candidate-missing events | 30 | candidate coverage failure |
| candidate recall, pair macro | 99.1815% | conditional observable-positive coverage |
| given-candidate Top-1, pair macro | 9.2708% | identity evidence/ranking bottleneck |
| given-candidate MRR, pair macro | 21.4933% | same bottleneck under a softer rank statistic |
| new false commits | 792 | accepted owner decisions later contradicted by train-only prefix outcome |
| persistent false commits | 740 | false commits with future wrong co-observation |
| pairs with persistent false commits | 25/25 | error is not isolated to one pair |
| future wrong co-observed edge frames | 78,624 | contamination diagnostic; not a MOT metric |
| causal-lag-4 pair-macro precision / recall / F1 | 13.6918% / 3.3168% / 4.9247% | fixed B0 paired-link diagnostic |
| formal IDF1/AssA/HOTA/MOTA | not run | no legal global-ID/test claim |

The decomposition supports the protocol claim: conditional candidate coverage is high, while ranking and owner commit are the dominant observable failure stages. It does **not** prove that an arbitrary ranking or state-repair module will improve full tracking.

## Label and causality contract

Runtime replay uses frozen detector/local-tracker streams, candidate sets, five-frame evidence windows, one-to-one assignment, and ready-frame ordering. XML labels are only opened in the post-replay diagnostic pass. Physical cross-device identity, synchronization, calibration, and global namespace are not inferred from the files.

P32 attempted to turn the commit-error observation into a method. Its matched-control result is recorded separately in [`continuation_20261005/p32_cod_state_repair/P32_DECISION.md`](continuation_20261005/p32_cod_state_repair/P32_DECISION.md) and did not pass the method gate.
