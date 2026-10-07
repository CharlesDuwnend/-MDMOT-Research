# Internal SRC mechanism evidence

The user clarified that the desired figures must explain and test internal SRC
behavior, like the existing cost analysis. Sequence-level MOTA/IDF1 and aggregate
error reductions do not answer this request. These figures therefore examine
the learned semantic input and its effect on actual candidate costs.

## Main recommendation: selective cost modulation

`src_cost_selectivity.pdf` compares empirical cost distributions for the same
candidate edges before and after the SRC operator. The reference is the recorded
masked geometry/appearance base cost `C`; the intervention is
`C_tilde = C + (1-C) clip(1 - b @ (z/pi), 0, 1)`.

On all eight frozen calibration sequences, the first association stage provides
234,011 initially legal edges with known foreground endpoints. Class-agnostic
geometry labels identify three categories:

| Edge category | n | Median base cost | Median SRC cost | Still eligible |
|---|---:|---:|---:|---:|
| Same owner | 191,731 | 0.048521 | 0.048572 | 99.4972% |
| Different owner, same coarse group | 18,330 | 0.728330 | 0.729251 | 99.0616% |
| Different owner, cross coarse group | 23,950 | 0.636576 | 0.993078 | 1.8288% |

Thus 98.1712% of initially legal cross-group mismatches exceed the fixed 0.80
threshold after modulation, while 99.4972% of same-owner edges remain eligible.
The same-group mismatch distribution is retained in both panels and changes
little. This gives a specific, positive and limited mechanism claim: **SRC's
semantic cost selectively removes cross-group incompatibility while preserving
most same-owner candidates.** It does not claim fine-grained identity separation.

This is a stronger complement to the existing analytic cost map than sequence
effects: the analytic map explains the operator, and this figure checks where
the penalty falls on labeled real candidate edges. It is appropriate in the SRC
mechanism analysis subsection.

## Input evidence: learned semantic probabilities

`src_semantic_evidence_reliability.pdf` compares the frozen learned observation
head with the predeclared raw coarse-group confidence construction on 26,088
GT-free-thinned calibration observations. Ten fixed bins retain every nonempty
bin; marker area is `5 + 0.35 sqrt(n)`. The diagonal indicates perfect empirical
calibration. Sparse bins are descriptive and have no independence-based error
bars or significance claim.

| Probability source | NLL | Two-class Brier | Accuracy |
|---|---:|---:|---:|
| Raw confidence | 0.108452 | 0.032101 | 98.9382% |
| Learned SRC evidence | 0.038011 | 0.018090 | 98.9459% |

The proper probability scores improve; classification accuracy changes little.
The claim is **more reliable semantic probabilities**, rather than improved
coarse-group classification accuracy. Some sparse bins remain imperfect and
are visible. This figure supports the input used by the cost operator.

## Data and operator boundary

- Current F00 observation and responsibility head parameters, scaler moments,
  classes and prior are exactly equal to the frozen `train_belief_v2` heads.
  The historical bundle's whole-method gate status is not used as evidence of
  success; parameters, data and scores are checked directly.
- Fit and calibration flights are disjoint. No test-dev/test20 data, model fit,
  threshold selection, full tracker rerun or GPU use is introduced.
- Beliefs are reconstructed from recorded birth observations and updated with
  the exact current responsibility-weighted operator on the recorded H2
  committed matches. All 1,390 birth states are observed; there are zero
  left-censored states. All 11,274 association records are consumed, including
  other stages needed for causal state maintenance. The displayed cost sample
  uses only stage 0 and excludes already illegal base edges, unknown endpoints
  and clutter.
- This is an **operator counterfactual on frozen H2 visited states**. SRC costs
  do not change the saved assignment or trajectory. It is internal development
  evidence, not a new tracking result or a measured online SRC error rate.
- The reference cost is the pre-semantic base `C`, rather than the historical
  host's separate continuous-semantic `native_cost`. The percentage suppressed
  is not an incremental correction rate relative to that historical policy.
- Owner labels come from the last actually accepted observation with a unique
  class-agnostic GT geometry match. Known detections can share a GT owner;
  eligibility is not proof of correctness of a global one-to-one assignment.
- Groups are human/vehicle. The method cannot separate every pair of different
  targets within one coarse group. The control distribution shows this limit.
- Responsibility weighting is used to reproduce the operator state. Its
  independent temporal benefit is not established by these figures.

## Independent verification and deliverables

`INDEPENDENT_AUDIT.json` records nine checks: unchanged source hashes; all eight
sequences; all association records; birth-state completeness; current original
GT hash agreement; 264 systematically sampled label reconstructions; independent
manual head-probability reconstruction; scalar cost reconstruction; and CDF
threshold/count agreement. `ANALYSIS_RECEIPT.json` retains all 67 source hashes,
full descriptive summaries, causal-state counters and interpretation limits.

Exports are in
`/home/chenhc/src_egia_icme_paper/figures/src_internal_mechanism_v3/`.
Each independent figure has fixed single-column PDF/SVG/600-DPI PNG versions,
color/grayscale previews, a standalone caption and SHA-256 verification. The
existing analytic cost figure and manuscript remain unchanged.

```bash
/home/chenhc/.conda/envs/u2mot/bin/python continuation_20261007/src_internal_mechanism/analyze_internal.py
/home/chenhc/.conda/envs/u2mot/bin/python continuation_20261007/src_internal_mechanism/audit_internal.py
python3 continuation_20261007/src_internal_mechanism/plot_internal.py
python3 continuation_20261007/src_internal_mechanism/plot_internal.py --export
```
