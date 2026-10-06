#!/usr/bin/env bash
set -uo pipefail
DIR=/home/chenhc/claude_try_MDMOT/continuation_20261005/p29_repair_20261005
PY=/home/chenhc/mdmt_phase0_reproduction_20260829_202909/mdmt_env/bin/python
MODE=train; ARGS=()
if [[ $# -eq 1 && "$1" == --smoke-only ]]; then MODE=smoke; ARGS=(--smoke-only); elif [[ $# -ne 0 ]]; then exit 2; fi
cd "$DIR" || exit 2
if [[ -e "$MODE.log" || -e "$MODE.exit" ]]; then exit 3; fi
env -u SUPPL_DUMP -u DET_OVERRIDE_DIR -u CKPT_OVERRIDE -u CFG_OVERRIDE MDMT_GPU_UUID=GPU-aaa79a66-b5c4-e0a0-6e2f-28c1ac0e3e6a PYTHONDONTWRITEBYTECODE=1 SUPPL_NO_VIS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-aaa79a66-b5c4-e0a0-6e2f-28c1ac0e3e6a "$PY" -B -u "$DIR/train_residual.py" "${ARGS[@]}" > "$MODE.log" 2>&1
status=$?; printf '%s\n' "$status" > "$MODE.exit"; exit "$status"
