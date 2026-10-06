# MDMOT 2026-10-06 continuation status

## Latest continuation: P54 v2 invalidated as a method test

The resumed thread `01a10a58-06da-7631-9ae1-a772f6d29e20` had stopped while waiting
for P54. That background run subsequently completed 1,200 updates with exit 0.
Its conditional calibration Recall@1 was 0.06589844 versus frozen 0.09284853,
delta -0.02695008, with 2/5 pair wins. This does not establish method failure:
the new independent 3-round x 3-check audit found a different model in the CPU
contract, missing declared margin loss, non-trimmed row-based future aggregation,
padding-dependent dustbin, missing no-match evaluation, and a candidate-copy
shortcut in the teacher objective. The archived CUDA name also identifies the
prohibited DGX Display/physical GPU2; the claimed A100 UUID was only an environment
label. Historical artifacts remain unchanged and must be read with the new audit.

See `p54_future_evidence_distill/SUPERSEDING_AUDIT.md` and `DECISION.json`.
The current vector ranker is stopped as a paper module; retraining is held pending
a representation/supervision redesign. Future distillation in general is not
declared ineffective. The research objective remains a substantive trainable
improvement of P26; the COD protocol route below is historical diagnostic work,
not a replacement of that objective. No new learned paper method is validated.

This continuation added a same-process GPU UUID/name/memory verification launcher,
passed nine GPU-policy regression tests and a live permitted-A100 probe, and
verified all 658 sealed P26 files. No training, raw official-val/test read, or
artifact deletion occurred in this continuation. P54's audit and run are a new
phase after the CCFI/CCSI initial archive.

## What was actually discovered and corrected

LCSCF v3's negative short gate was invalid. Independent counterexamples found incorrect local reverse-plan sampling, cross-view reference comparison, unknown-candidate ranking/probability handling, and detector native image/RoI geometry. The v3 decision was superseded; it must not be cited as an effectiveness failure.

The corrected implementation passed `lcscf_b4_gate/LCSCF_IMPLEMENTATION_AUDIT_V4.json` (three rounds, three checks each): displaced round-trip, destination-view consistency, extreme-temperature/border gradients; exact MMCV native pixels/boxes/RoI extent and label masks; strict AutoAssign 258-tensor load, K=0/1/64 step-zero, real episode backward and optimizer finiteness.

The corrected 600-update B4 versus LCSCF-B4 block then passed an independent metric replay but failed the predeclared advancement gate. LCSCF gained Recall@1 by +0.001653 macro, but lost MRR by -0.012632 and margin by -0.021692; it strictly won 2/4 pairs and tied 1. It is stopped as a paper method and retained as a negative control. No official val/test was accessed.

## Current method status

P26 remains the protected mature MIA-Net baseline. P52 and CVSGM remain stopped by collision/non-incrementality audits. LCSCF is stopped only after the corrected run; its earlier v3 stop record is superseded.

The next screened candidate, Cross-View Temporal Differential Co-Movement Field, passed train-only observability and a CPU mechanism contract, but failed the frozen-feature signal gate before training: static cosine was positive on 8/8 primary pairs, while first-order differential cosine was positive on 3/8 and had macro delta -0.0015705. It is stopped before any B4 adapter or training. Missing support rows are explicitly masked; 95.1607% of support rows are addressable overall and all block1 primary pairs are complete.

The query-token and unsupervised UAV-role calibration screen also stopped before training. On four primary held-out train pairs, frozen visual features reached macro Recall@1 0.2047869, query-token 0.1941831, equal fusion 0.1982479, and role z-score 0.2124508 (only 2/4 pair wins). The query branch won 0/4 and the role normalization overlaps an established camera-aware normalization family. These are diagnostics, not method results.

P50 fragment-teacher map transfer was independently audited after 3,200 total updates (3.013 effective epochs). Checkpoint/gradient/teacher isolation/finite replay all passed, but map-level Recall@1 remained 0.0474585 versus frozen 0.1164549 with 0/5 pair wins. The negative result is therefore not explained by a short schedule or an implementation failure.

The current paper route is MDMT-COD as a protocol/diagnosis contribution. Its independent three-round audit is in `cod_paper_route/PAPER_READINESS_AUDIT.json`: 25 train pairs, 8,311 events, candidate recall macro 0.9918146, given-candidate Top-1 macro 0.0927080, MRR 0.2149330, and persistent false commits on 25/25 pairs. It used no training, GPU, val/test, or formal MOT metrics. The route is kept for error decomposition and auditability, not presented as a new learned module or performance improvement.

## Do not claim yet

There is no validated new learned paper method and no official-val/test improvement. No stopped branch is authorized for P26 attachment. The next algorithmic step, if pursued, must add a genuinely new identity representation or supervision target on top of the COD bottleneck, then pass novelty, legal observability, CPU contract, train-free signal, and train-only causal repair gates before GPU training or official evaluation.

## CCFI/CCSI full-coverage outcome (2026-10-06)

The CCFI branch was trained through the frozen full schedule (55,236 steps, 18,412 eligible groups, 3 epochs) on an A100-SXM4-40GB and evaluated on the frozen 20-pair train-only read. Its short run reached calibration Recall@1 0.4540345 and +0.0026682 versus P38 on 4/5 pairs; full coverage fell to 0.4369484 and -0.0144179 versus P38 on 0/5, with fit-to-calibration drop 0.0652501. The 3x3 full audit passed implementation and integrity checks, then classified cross-pair overfit and stopped CCFI.

CCSI adds counterfactual history/change swapping to CCFI. Its CPU mechanism contract and train-free fixed-history signal passed. The repaired 1,200-step short run reached calibration Recall@1 0.4534309 and +0.0020646 versus P38 on 4/5 pairs, below the preregistered absolute gate 0.4641. The complete 55,236-step, 3-epoch run used the same legal frozen support and A100-SXM4-40GB; it fell to 0.4401437 and -0.0112226 versus P38 on 1/5 pairs, while fit was 0.9095778 and fit drop versus P1 was 0.0619518. `p53_ccsi/FULL_FAILURE_AUDIT.json` passed 3 rounds × 3 checks and classifies the result as cross-pair generalization failure/overfit. CCSI is stopped after full coverage; no host attachment, dev read, official-val/test, or MOT claim is authorized.

The retained artifacts are the short and full checkpoints as local diagnostic evidence, the method specification, CPU contract, train-free signal, implementation audits, frozen evaluation manifests, and full failure audit. Checkpoints and raw embeddings remain local and are indexed by SHA-256 rather than stored in Git.

## P57-P59 current continuation outcomes (2026-10-06)

P57 tested frozen bidirectional top-16 DINO patch correspondence after fixing an initial per-token normalization defect. The corrected five-pair signal was 0.2351224 versus P23 target-only 0.4905144, delta -0.2553920, with 0/5 pair wins; it is stopped before training.

P58 tested a strict-past temporal anchor on the P23 head. The sealed calibration cache samples every 11 frames while the preregistered history window is at most 8 frames, so legal temporal support was zero on all five pairs. It is an observability hold, not a scientific negative; no training is authorized until a continuous all-frame P23 feature cache exists.

P59 screened and trained a cross-representation causal bridge: frozen P23 current identity plus frozen P1 strict-past local-track history, a gated residual, and cross-view masked identity loss with causal consistency. Fit support was 96.37%; the corrected v2 run completed 1,200 updates on physical GPU3. Calibration5 was 0.4645404 versus target-only 0.4791246, delta -0.0145842, with 0/5 pair wins. Its three-round implementation audit passed and classified a valid cross-pair generalization failure; the branch is stopped before P26 host attachment.

The protected P26 baseline remains unchanged. No P57-P59 branch supports an official-val/test or formal MOT improvement claim. The next candidate must use a different identity target or operator, pass a fresh mechanism-level prior-art screen, and clear the same legal observability, CPU, train-only, and calibration gates.

## P60 setwise transport outcome (2026-10-06)

P60 screened a fixed balanced Sinkhorn transport operator over each same-frame,
same-class opposite-view candidate set, followed by a zero-initialized trainable
residual (SEIT). The frozen operator signal passed on calibration5: target-only
`0.4791246` to fixed transport `0.5158327`, delta `+0.0367081`, 4/5 pair wins.
The accepted 1,200-update run used physical GPU3 A100-SXM4-40GB and produced
trained SEIT `0.5169382`, delta `+0.0378137`, 4/5 wins. The independent audit passed
three rounds with three checks each, including transport marginals, set permutation,
zero initialization, receipt/hash, split closure, and independent metric replay.

The attribution gate is decisive: trained SEIT adds only `+0.0011055` over the
fixed transport operator. Balanced optimal transport/differentiable assignment is
established prior art, so P60 is retained as a validated engineering control and
stopped for paper novelty and P26 host attachment. No official val/test or formal
MOT result was read.

## P61 cross-pair invariant identity outcome (2026-10-06)

P61 (CPIA) treated each fit pair as an environment and aligned second-order
gradient directions from two distinct pair-local identity losses. The repaired CPU
contract covered all 15 fit pairs and 1,200 groups. The physical GPU3 run completed
1,200 updates, but calibration fell from P23 `0.4791246` to `0.4608756`, delta
`-0.0182490`, with 0/5 pair wins. The independent audit passed three rounds with
three checks each, including split/receipt closure, zero-init, unknown-label masking,
finite/unit embeddings, and independent scorer replay. This is a valid negative
cross-pair generalization result, not an implementation failure; P61 is stopped
before host attachment and official evaluation.

## P62 causal owner-state transition screening (2026-10-06)

P62 starts from the COD finding that high candidate coverage coexists with
persistent false ownership commits. The candidate is a prefix-causal owner-state
transition policy: candidate evidence and the current provisional partition produce
a link/no-link action and an explicit state update. The repaired data/CPU contract
covers 20 legal train pairs, 179,949 candidate rows, and 2,552 prefix-positive
rows; all source cache/event hashes, finite gradients, and provisional/revoke/final
state transitions pass. A PyTorch positional-dtype typo was fixed before acceptance
and produced no scientific result. P62 is still before model training and formal
metrics; the next gate is held-out prefix replay with same-information/delayed and
log-only controls.

## Git phase archival

The workspace is initialized on branch `main` with the authorized remote `git@github.com:CharlesDuwnend/-MDMOT-Research.git`. Each major phase is archived with `scripts/stage_snapshot.py`; it records local large-artifact paths and checkpoint hashes, commits compact source/spec/audit/result evidence, pushes `origin/main`, and verifies the exact remote commit. The initial phase will include the screened CCFI/CCSI evidence and the stop decisions above.
