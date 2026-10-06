# CCSI: Causal Change-Swap Identity

CCSI is a follow-up to the stopped CCFI experiment. CCFI separated identity
core, temporal change and uncertainty, but full fit training overfit the fit
pairs. CCSI adds a falsifiable anti-shortcut operator: it swaps strictly-past
change evidence between two same-view rows and requires the identity core for
the current observation to remain stable.

`current frozen P1 embedding + strict local-track history -> identity/change/
uncertainty factorization -> counterfactual history/change swap consistency +
cross-view masked matching + causal change prediction -> identity core`.

The swap is made only among available past histories in the current frame/class
batch. It uses no labels, future rows, pair IDs, or official data. The swapped
history is a training-only counterfactual; inference receives the normal causal
history and emits the identity core. This directly tests whether the prior
full-fit regression came from encoding track/pair-specific temporal change in
identity, rather than adding another candidate ranker.

The train-only gate is inherited and frozen: calibration macro Recall@1 >=
0.4641, at least 3/5 calibration pair wins over P38, and no fit15 drop greater
than 0.05. A pass authorizes only another representation audit, never direct
P26 attachment or official evaluation.
