# MDMT research workspace conventions

The user authorized an independent private remote Git repository on 2026-10-06
and ordinary commits/pushes after each substantial research phase. Do not request
that authorization again. If origin is unavailable, commit locally and report
the precise remote/authentication dependency; do not claim a remote backup.

Use `python3 scripts/stage_snapshot.py --stage UNIQUE_STAGE --message "Outcome"`
at phase boundaries, after recording completion, failure, or unresolved audit
findings. Use `--local-only` only while origin is unavailable. Never force push.
Preserve protected P26 baseline files and experiment lineage. This repository
does not authorize deleting datasets, large artifacts, or unrelated projects.

Keep source, method specifications, preregistrations, compact evidence and
artifact hashes in Git. Keep checkpoints, full prediction streams, datasets,
authentication files and large caches local. Verify `.gitignore` and staged
files before changing this policy. Do not commit live partial results as final.

Research remains aimed at a substantive trainable improvement of the mature
MDMT baseline. Train/calibration diagnostics, protocol findings and synthetic
contracts are not formal MOT improvements. Audit code, data, operator, loss and
scorer contracts independently; a saved PASS flag alone is insufficient. Failures
require at least three rounds with three implementation checks per round.
All non-4GB large GPUs may be used when available; never use physical GPU2.
