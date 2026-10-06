# P54 selection audit: problem -> module -> test

## Problem selected

The protected P26 MIA baseline already has a comparable complete-output score. The
train-only MDMT-COD replay shows a different failure signature: candidate coverage
is high (`candidate recall` macro `0.9918146`), while ranking among supplied
candidates is low (Top-1 macro `0.0927080`) and every checked pair has persistent
false commits. Improving a threshold, candidate cap, or owner tie break would only
move an operating point and is excluded.

The representation problem is that a causal target observation may not contain a
stable identity cue at the commit frame, while a later source-view fragment of the
same training identity can reveal a cue that was absent or occluded at the commit.
The method must transfer that identity direction into a student that sees only the
causal prefix at inference. It must not predict future boxes or use future features
at inference.

## Chosen module

P54 uses a candidate-conditioned residual identity field over the *whole candidate
set*. For each candidate, the leave-one-out set context prevents the module from
being a pairwise cosine reranker. The residual latent is scored jointly with a
no-match output. A training-only future source-view prototype supplies a detached
identity target only when a legal future donor exists; events without a donor still
contribute assignment/no-match loss. The inference graph is exactly
`current target + current same-frame candidates + mask + strict prefix summary`.

This differs from the stopped branches as follows:

* P38/CCSI read strict-past history at inference and used cross-view CE/consistency;
  P54's new supervision is an acausal training target, while its inference graph is
  prefix-only and candidate-set conditional.
* P50 used strict-past fragment teachers and selected only donor-eligible positive
  events; P54 uses strictly future source fragments and reports the full positive
  candidate population separately from teacher-eligible rows.
* P52/SCI-ID's positive-removal/cardinality/dustbin intervention is a collision
  boundary, not P54's claimed contribution. P54 stops if its gain disappears when
  those terms are removed.
* P29/P28 are detector feature-fusion controls, not identity-ranking modules.

## Falsification and stage gates

1. **Data gate:** all teacher frames satisfy `frame > target frame`; no same-frame,
   past, validation, or official data enters the teacher. Report future-donor
   coverage and full-event coverage separately.
2. **Mechanism gate:** CPU contract passes candidate permutation equivariance,
   finite gradients, detached teacher, no-match output, and inference-field audit.
3. **Train-free signal gate:** future-prototype ranking is only a ceiling diagnostic;
   it must be reported without calling it method effectiveness. If the future
   prototype does not improve the frozen baseline on at least 3/5 pairs, stop before
   GPU training.
4. **Short train gate:** fixed 1,200-step, pair-grouped train-only fit; evaluate
   all positive-candidate events on five held-out train pairs. Require at least
   +1.0 percentage point macro Recall@1 and 3/5 pair wins over frozen cosine, with
   finite outputs and no teacher at inference. This authorizes only a second
   representation audit, never P26 attachment.
5. **Sufficiency gate:** if the short gate passes, repeat the same checkpoint to a
   predeclared full schedule (55,236 steps / 3 epochs over the eligible fit groups)
   or a clearly justified grouped schedule. A short-to-full regression is a method
   failure, not a reason to tune the gate.
6. **Host gate:** only a full-coverage representation pass can authorize an
   isolated P26 host replay. Official val/test remains closed until host evidence
   is complete and independently audited.

Any failure receives three rounds with two implementation checks per the current
session requirement: (R1) data/lineage, (R2) operator/gradient, and (R3)
evaluation/split wiring. The failure audit must explicitly classify an
implementation defect versus a weak representation signal or generalization
failure.

## Feasibility and risk

The first experiment is vector-level to isolate the identity target from expensive
FPN/map transport. Candidate sets are at most 64 and the student is under one
million parameters, so an A100-40GB run is practical. It cannot prove a complete MOT
improvement. The main risks are future-teacher selection bias, LUPI collision with
MVCD/PKD, and fit-to-calibration overfit; these are why the full-event denominator,
five-pair group split, and sufficiency gate are fixed before training.

**Current decision:** HOLD pending data-audit receipt and train-free signal. Do not
attach to P26, read dev/test, or claim novelty/effectiveness yet.
