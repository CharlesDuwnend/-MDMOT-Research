# Further pure EGIA ablation

Continue the user's Native + pure v5 EGIA internal reductions. SRC cost,
updates and state, the extra class-confidence birth gate 0.7, and H2 fusion
are all disabled. Native detector-score birth eligibility stays at 0.6.
Preserve all current fitted heads; no refitting or test-dev threshold selection.
Three new full runs: coverage only with selective; foreground plus coverage
without source cues and without selective; full EGIA without selective.
Reuse the completed full and two-head selective references explicitly, never
present those as new inference. Missing a geometry or appearance cue makes
co-reference evidence identically zero; audit this algebraically rather than
spend GPUs on redundant controls. Report every prescribed result and tradeoff.
Only GPU UUIDs belonging to physical 0, 1, 3 may be used, with same-process
verification and >=8192 MiB free; physical GPU2 is forbidden. Keep raw streams,
weights and datasets local; parent manifests and P26 remain immutable.
No manuscript edits, new figures, or messages to others. Parent Git snapshot
instructions apply, preserving unrelated staged changes.
