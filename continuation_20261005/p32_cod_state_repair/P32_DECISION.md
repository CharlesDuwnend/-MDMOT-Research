# P32 decision: HOLD_NOT_PAPER_READY

Date: 2026-10-05

P32 implemented a causal owner-state revision candidate and audited it on the frozen **train-only** MDMT-COD replay. The implementation gate passed, but the method gate did not.

## Evidence

Primary `W=4` run: [`runs_v3/W4/AGGREGATE.json`](runs_v3/W4/AGGREGATE.json)

* 25 independent train pairs and 1,942 selected real assignments were replayed.
* 32 revoke/revision operations occurred, touching 10/25 pairs.
* Same-information/same-delay ended with 954 edges: 57 correct, 871 false, 26 unknown.
* Repair ended with 954 edges: 57 correct, 870 false, 27 unknown.
* Future wrong co-observed frames were 87,290 for the delay control and 87,028 for repair (−262, about −0.30%); correct edges did not increase.
* The structural sensitivity run found zero revisions for `W=0`, 32 for `W=4`, and 44 for `W=8`; the change is therefore a window-triggered conflict replacement, not a new identity representation.

The edge labels and frame counts are train-only diagnostics. They are not IDF1, AssA, HOTA, MOTA, validation, or test results.

## Implementation/failure audit

[`IMPLEMENTATION_AUDIT.json`](IMPLEMENTATION_AUDIT.json) is PASS for the representative pair 23:

* zero evidence rows cross the ready boundary;
* zero non-finite costs and duplicate batch edges;
* repeated replay is deterministic;
* zero causal violations (no future proposal enters an earlier transition);
* state-transition functions receive proposals and `W` only, never GT.

The initial full-run diagnostic had an invalid global end-frame cutoff. That was identified and corrected to a per-edge ready-frame cutoff before the reported v3 run; this correction is part of the audit trail.

## Research decision

`HOLD_NOT_PAPER_READY`. The observed change is one false edge over 871 and no additional correct edge. The candidate is ordinary bounded deferred conflict replacement and does not yet supply a defensible method-level contribution for a paper. Do not train, tune, or open official val/test for P32.

The next paper step must add a new observable identity signal or a protocol contribution with a falsifiable mechanism gate. Changing `W`, thresholds, or the owner tie-breaker would be parameter/ranker tuning and is not an acceptable rescue of P32.
