# P62 Counterfactual Owner-State Transition (COST)

## Problem

MDMT-COD shows high candidate coverage but persistent false cross-view ownership
commits. A wrong accepted edge changes the owner partition, so later decisions see
a contaminated state. A descriptor-only repair cannot represent this dependency.

## Input -> operator -> objective -> output

`prefix candidate evidence + current owner partition -> state-transition policy -> provisional link/no-link action and updated owner partition`

For each event, the policy receives only evidence through `ready_frame`, candidate
edge features (distance trajectory, support, and rank), endpoint local-track state,
and the current provisional owner partition. A gated state cell updates endpoint
states only when a provisional action is published. Finalization freezes the
partition; later XML/future frames are never policy inputs.

Training uses legal fit-pair prefix labels: masked cross-view link/no-link loss plus
an owner-transition consistency loss against the deterministic `OwnerState` contract.
The model is evaluated on held-out train pairs by prefix replay. Formal MOT metrics
require a separate host integration and are not authorized by this phase.

This is a substantive state-transition hypothesis, not a fixed-lag reranker or an
ID relabeling pass. The initial phase only builds the data/contract audit and CPU
mechanism gate; no calibration, val, or official result is inferred from it.
