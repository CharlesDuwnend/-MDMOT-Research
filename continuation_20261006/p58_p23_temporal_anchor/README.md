P58 is a train-free transfer gate for a causal temporal anchor on top of the
protected P23 head-only identity representation. The output is a compact signal
readout, not an MOT result. Check `SIGNAL.json` and `DECISION.json` after the run.

The first readout is an observability stop: P23 calibration rows are sampled at
11-frame intervals, so the strict <=8-frame history has zero legal support on all
five pairs. The equal scores are therefore a current-embedding fallback, not a
negative temporal result. See `OBSERVABILITY_AUDIT.json` and `DECISION.json`.
