# Frozen cost-factorial source audit, 2026-10-08

Read-only review of the frozen runtime, online adapter, canonical host and 15
synthetic CPU contracts. No GPU was launched and no frozen file, model or
manifest was changed. The five reviewed source files still match their frozen
hashes; tests are sealed indirectly by the manifest SHA of `cost_cpu_gate.json`.
The 15 current CPU contracts were rerun successfully. See
`COST_FACTORIAL_FROZEN_AUDIT_RECEIPT.json` for commands, output and hashes.

## Actual intervention

| Cell | Runtime arm | Assignment coarse-class mask | Learned semantic penalty |
|---|---|---:|---:|
| C11 | fc_full | on | on |
| C00 | fc_control | off | off |
| C10 | fc_mask_only | on | off |
| C01 | fc_penalty_only | off | on |

All four manifest entries use the identical F00 bundle hash
`ee903030e48f73fe87411b6e9aa60d52bbf2ee2d895aebfa8085d260779702ac`.
Constructor branches retain SRC birth initialization, responsibility-weighted
recurrent updating, EGIA source scoring, summary fusion and selective birth.
Extra cost policies, alternate category maps and host cost settings are rejected.

`_base_cost` computes the original masked B_ref. `_assignment_base` either copies
it or reconstructs the same geometry/appearance fusion while omitting only
`gate_cost_matrix_by_cls`. The mask replaces cross-coarse-class appearance cost
with 1, before the frozen proximity mask and min(IoU, appearance) fusion. It is
an appearance-channel mask, not a hard prohibition of a complete candidate edge;
geometry can retain such an edge. The learned penalty separately uses human /
vehicle belief and z/pi and changes cost by `(1 - assignment_base) * u`.

The same-state synthetic matrix check exercises different committed pairs:
C00 retains crossed low-appearance pairs; C10/C01/C11 select diagonals. Their
costs are respectively `[[.05,.01],[.01,.05]]`, `[[.05,.20],[.20,.05]]`,
`[[.05,.7228],[.7228,.05]]`, and `[[.05,.776],[.776,.05]]`.
This verifies a real operator difference without relying on a saved PASS label.

## Memory and online-state contract

`associate` keeps B_ref, raw appearance and geometry outside the factorial cost
matrices and passes those original arrays to `responsibility_features` for the
actually committed pairs. The responsibility-feature contract test compares
every returned head input numerically against B_ref in all four cells, including
the crossed C00 assignment. `_flush_committed` requires the host frame commit
before writing belief, and the posterior formula and initialization remain
unchanged. Uncommitted candidate edges do not update semantic memory.

Fixed B_ref means the calculation, head and feature definition are fixed for a
given state. It does not mean B_ref values, responsibilities, beliefs, track pools
or EGIA evidence are identical across closed-loop arms after trajectories differ.
C00 uses actual native association while retaining learned memory and EGIA;
it is not a standalone native-tracker run.

## Candidate, ordinal and parity boundaries

Every nonempty factorial seam checks nondecreasing cost and forbids a native
threshold-rejected edge becoming legal. It also checks that every committed LAP
edge is finite and below the unchanged seam threshold. Existing candidate lists
and detector outputs are retained; mask/penalty can reweight or reject legal
edges. Empty seams call the actual host and are counted separately.

C00 returns `super().associate` directly. In all nonempty cells the actual host
assignment is compared with the independent unmasked reconstruction; a mismatch
stops. Tests exercise both the actual C00 result and a deliberately mismatched
native seam. This is stronger than deriving native solely from a flag. Whole-run
C11 parity with canonical F00 and C00 parity with the old update-only reference
remain online-runner gates, not consequences of synthetic tests.

Ordinal lookup requires an unused geometry-matching detector row; arbitrary
fallback was removed, and `det_record` further checks exact stored box equality.
The tests verify unresolved boxes stop before model calls. A remaining untested
edge case is two same-frame identical boxes carrying different class/confidence
metadata, especially across high/low pools: lookup chooses the first unused box
match and checks geometry only. No actual duplicate-box error is demonstrated
by this review. Input hashes do not by themselves prove that such ambiguity
never occurs; do not claim a metadata-uniqueness audit from these tests.

The online ledger hashes exact detector and sampled-embedding values, dtype,
shape, row order and frame metadata before host update. Its tests prove value,
order, dtype and shape sensitivity and no mutation of host inputs/output order.
The runner records skipped empty frames, checks full frame continuity, and
compares sequence rolling hashes against C11 before accepting other full arms.
It hashes the sampled tracker embeddings, not the entire feature map or images.

## What eventual results would support

Use conditional paired contrasts: mask at penalty off = C10-C00; mask at penalty
on = C11-C01; penalty at mask off = C01-C00; penalty at mask on = C11-C10.
Interaction is `(C11-C10) - (C01-C00)` for a consistently oriented metric.

Positive conditional mask effects support the frozen coarse-class appearance
operator in that context. Positive conditional penalty effects support the
learned belief-conditioned cost path beyond the corresponding mask control.
An interaction supports context dependence, not additive attribution to both
operators. Negative effects indicate a deployment tradeoff or harm at this
frozen operating point; zero effects, after coverage and parity gates, indicate
no demonstrated necessity in this comparison. They do not by themselves show
an operator was unexecuted, or prove that no different protocol could benefit.

These cost cells hold memory update and EGIA enabled. They cannot establish
necessity of the reliability head, recurrent update, EGIA factorization,
coherence, fusion or selective admission. MOTA/IDF1 disagreements should retain
FP/FN/IDs and paired sequence/flight detail. Report all predeclared cells and
avoid test-dev winner selection, independent-frame significance or claims that
synthetic contracts establish a formal MOT gain.
