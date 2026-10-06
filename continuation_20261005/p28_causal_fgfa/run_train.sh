#!/usr/bin/env bash
set -uo pipefail
P28_DIR=/home/chenhc/claude_try_MDMOT/continuation_20261005/p28_causal_fgfa
P28_PY=/home/chenhc/mdmt_phase0_reproduction_20260829_202909/mdmt_env/bin/python
cd "$P28_DIR"
if [[ -e train.log || -e train.exit ]]; then
  printf '%s\n' 'Existing train log or receipt. Preserve the previous attempt.' >&2
  exit 3
fi
env -u SUPPL_DUMP -u DET_OVERRIDE_DIR -u CKPT_OVERRIDE -u CFG_OVERRIDE \
 PYTHONDONTWRITEBYTECODE=1 SUPPL_NO_VIS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-aaa79a66-b5c4-e0a0-6e2f-28c1ac0e3e6a \
 "$P28_PY" -B -u train_adapter.py > train.log 2>&1
P28_EXIT=$?
printf '%s\n' "$P28_EXIT" > train.exit
exit "$P28_EXIT"
