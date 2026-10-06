# P54 continuation: v2 audited, retraining held

This directory contains historical P54 screening source and results, plus the
independent continuation audit. Read `SUPERSEDING_AUDIT.md` and `DECISION.json`
before interpreting any older `PASS_*` status. The old CPU test, data audit, and
short-gate completion flags do not establish implementation or method validity.

`src/train_short.py` is preserved for forensic replay of the old checkpoint. Do
not relaunch it: the unguarded ordinal-based run used a prohibited display GPU,
and unresolved method/operator/loss/scorer contracts remain. The broader future
supervision idea is not proven ineffective by this run.

The authoritative next action is to resolve the representation and supervision
contract in `SUPERSEDING_AUDIT.md`, including the candidate-copy counterexample,
before proposing another training run. The objective remains a substantive
trainable improvement on P26, not a diagnostic-only paper substitution.

Reproduce the CPU audit:

```bash
CUDA_VISIBLE_DEVICES='' /home/chenhc/mdmt_phase0_reproduction_20260829_202909/mdmt_env/bin/python continuation_20261006/p54_future_evidence_distill/src/audit_short_v2_independent.py
```

The audit deliberately reports failed contracts. It independently checks a
128-event train-pair sample and recomputes stored calibration aggregates; it does
not claim a complete replay of all five calibration prediction populations.

Future authorized GPU work must verify the physical UUID in the actual process:

```bash
/home/chenhc/mdmt_phase0_reproduction_20260829_202909/mdmt_env/bin/python scripts/verified_gpu.py --uuid GPU-aaa79a66-b5c4-e0a0-6e2f-28c1ac0e3e6a
```

The same launcher accepts `-- path/to/training_script.py [arguments]` only for a
separately justified training step. It verifies physical index, driver UUID,
PyTorch name and total memory, then executes the script in the verified process.
This command is not authorization to rerun the historical P54 trainer.
