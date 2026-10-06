# P54 — Prefix-Causal Future-Evidence Distillation (screening candidate)

P54 addresses the MDMT-COD bottleneck that candidate coverage is high while
candidate ranking and owner commit are poor. During **training only**, a teacher
may inspect strictly future source-view fragments of the same annotated identity.
At inference the student receives only the current target feature, the current
same-frame source candidate set (`K<=64`), an optional strict-prefix target-view
history summary, and masks. Future fragments, XML IDs, pair IDs, and teacher
vectors are absent from the student graph.

The proposed representation is a candidate-conditioned identity residual. For
candidate `k`, the student computes a leave-one-out candidate-set context and a
residual latent `z_k = f(target, candidate_k, other_candidates, prefix_summary)`;
logits and a no-match logit are produced from `z_k`. The future teacher is a
trimmed, detached prototype of source-view features after the target event. The
training objective combines multi-positive candidate assignment, teacher
alignment only on events with a future donor, and a cardinality-preserving
hard-negative margin. No threshold or checkpoint is selected from the readout.

This is a screening candidate, not a novelty claim. The narrow possible boundary
is a legal prefix-causal MDMT candidate-ranking target that transfers future
cross-view identity evidence into an inference-time candidate-set residual. It
must be stopped if it reduces to ordinary ReID distillation, a post-pooling
ranker, candidate-composition/dustbin intervention, or a temporal-memory rename.
No host attachment or official-val/test read is authorized by the screening gate.
