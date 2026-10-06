# P32: Prefix-Causal Owner-State Revision

## Intended claim

P32 tests whether a paired cross-view owner can publish a cross-view edge provisionally, revise it inside a bounded evidence frontier, and freeze it after the frontier. The candidate is deliberately restricted to the legal MDMT-COD setting: one camera pair, frozen detector/tracker features, frozen candidate events, and no timestamp/GPS/calibration/global namespace assumption.

This document specifies a mechanism gate. It does not authorize training, official validation, test evaluation, or a formal MOT claim.

## Runtime contract

Each frozen Hungarian batch yields a real proposal
`p=(a,b,c,t_r)`, where `a` is a camera-1 local track, `b` is a camera-2 local track, `c` is the frozen aggregate distance, and `t_r` is its ready frame. The state contains active edges `A`; an edge is either provisional or final.

For a new proposal at frame `t`:

1. If neither endpoint has a different active mate, publish `(a,b)` provisionally.
2. Before publication, freeze an older edge `e` when `t - t_r(e) > W`.
3. If there are conflicts, revoke all conflicting edges only when every conflict is still provisional and `c < c_e` for every edge being revoked. Publish the lower-cost proposal and record the revision.
4. Otherwise reject the proposal. No GT label, future frame, or future candidate is read by this transition.

The primary window is `W=4`, inherited from the already frozen decision lag. `W=0` and `W=8` are structural sensitivity checks; they are not selected from labels.

## Controls

* **B0**: the frozen stage-15 owner-accepted edges.
* **Same-information/same-delay**: the same selected real proposals are held for `W` frames and then accepted in arrival/cost order without revocation.
* **Repair**: the state machine above, with the same proposal stream and the same `W`.

The only difference between the latter two controls is bounded state revision. Candidate scores, candidate sets, one-to-one Hungarian outputs, and decision lag are held fixed.

## Evaluation boundary

The state transition is replayed on the 25 train pairs only. XML is opened after replay solely to label final edges and count future co-observed frames as an offline diagnostic. The run writes no val/test prediction, no official MOT metric, and no trainable checkpoint.

## Decision rule

The candidate is KEEP only if revision changes a substantive identity outcome against the same-information/same-delay control across independent pairs, with no implementation or causality failure. A reduction caused only by censoring later frames, a change in candidate ranking, or a synthetic contract pass does not satisfy the gate.
