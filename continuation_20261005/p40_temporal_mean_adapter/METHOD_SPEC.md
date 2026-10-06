# P40 causal temporal-mean anchored adapter

P40 is the single predeclared trained follow-up to P39. The input is the frozen P1 independent128 embedding plus a causal same-view history indexed by local-track metadata. The history is exactly P1C: the last 8 feature-present observations including the current embedding, with class-drift and >30-frame gap resets. The anchor is the unit-normalized causal mean. A zero-initialized residual adapter predicts a delta from `[current, anchor, current-anchor]` and adds it only when history exists; observations without history remain the current P1 embedding.

The loss is the frozen P7R cross-view masked identity CE plus 0.2 causal consistency to the anchor. It uses the same 1,200-step P7R schedule, seed 42, learning rate and weight decay as P38. Fit15 is the only gradient role. Calibration is read once after all label-free fit/calibration embeddings are frozen. This is a research candidate, not a novelty claim or official MOT result.
