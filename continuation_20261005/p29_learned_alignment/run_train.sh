#!/usr/bin/env bash
set -uo pipefail
P29_DIR=/home/chenhc/claude_try_MDMOT/continuation_20261005/p29_learned_alignment
P29_PY=/home/chenhc/mdmt_phase0_reproduction_20260829_202909/mdmt_env/bin/python
P29_MODE=train
P29_ARGS=()
if [[ $# -eq 1 && "$1" == --smoke-only ]]; then
  P29_MODE=smoke
  P29_ARGS=(--smoke-only)
elif [[ $# -ne 0 ]]; then
  printf '%s\n' 'Usage: run_train.sh [--smoke-only]' >&2
  exit 2
fi
cd "$P29_DIR" || exit 2
if [[ -e "$P29_MODE.log" || -e "$P29_MODE.exit" ]]; then
  printf '%s\n' 'Existing log or exit receipt. Preserve the previous attempt.' >&2
  exit 3
fi
env -u SUPPL_DUMP -u DET_OVERRIDE_DIR -u CKPT_OVERRIDE -u CFG_OVERRIDE \
 PYTHONDONTWRITEBYTECODE=1 SUPPL_NO_VIS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-aaa79a66-b5c4-e0a0-6e2f-28c1ac0e3e6a \
 "$P29_PY" -B -u "$P29_DIR/train_residual.py" "${P29_ARGS[@]}" > "$P29_MODE.log" 2>&1
P29_EXIT=$?
printf '%s\n' "$P29_EXIT" > "$P29_MODE.exit"
exit "$P29_EXIT"
