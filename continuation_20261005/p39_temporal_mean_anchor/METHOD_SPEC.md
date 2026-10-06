# P39 fixed causal temporal-mean anchor

This is a predeclared diagnostic/control for P38. It uses the sealed P1-128 embedding and the existing P1C causal rule exactly: per view, local-track metadata indexes a FIFO of the last 8 feature-present embeddings, including the current observation; class drift and a gap greater than 30 frames reset the FIFO; missing features are never imputed. No labels, IDs, or future frames enter the output. There is no training, threshold search, or host transfer in this diagnostic.

The purpose is to determine whether the below-gate P38 result is caused by a weak current-state initialization or whether a fixed temporal anchor is itself insufficient on the P4 fit15/cal5 retrieval contract. This is an established control, not a novelty claim.
