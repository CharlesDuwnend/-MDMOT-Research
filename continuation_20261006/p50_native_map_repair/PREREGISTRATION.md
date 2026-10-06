# Gate preregistration — 2026-10-06

- Protected baseline: `continuation_20261005/p26_mia_baseline`.
- Fit pairs: 23,25,27,28,29,30,32,39,42,44,45,50,51,53,54.
- Held-out train calibration pairs: 69,70,74,76,78. Official val/test are locked.
- Unit of replication: MDMT pair; pair-macro summary, not frame-level pseudo-replication.
- Arms: relational residual × teacher off/on; pooled equal-capacity control × teacher off/on.
- Seeds: 7,17. Updates: 2400 per arm/seed. Batch/order identical.
- Native cache: frozen stage3 ROI features, detector checkpoint SHA is recorded by the existing cache receipt; target/candidates resolved by frame path, class, and IoU rather than ambiguous class-local/global detector indices.
- Teacher: strict source-frame prefix only; fixed raw feature mean; no trainable teacher projector; absent on rows without a prefix.
- Population: every valid train episode, including zero-candidate and no-positive rows. Missing feature rows are recorded as integrity failures and never silently converted to negatives.
- Selection rule: no official-val/test result is used. Keep the direction only under the METHOD_SPEC.md gate; otherwise stop or return to design audit.
- Evidence required after run: exit code, per-pair counts, equal denominators, checkpoint SHA256, finite/reload audit, baseline step-zero check, and explicit teacher-absence inference check.
