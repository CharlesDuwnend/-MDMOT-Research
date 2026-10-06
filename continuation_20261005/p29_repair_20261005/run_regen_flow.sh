#!/usr/bin/env bash
set -euo pipefail
DIR=/home/chenhc/claude_try_MDMOT/continuation_20261005/p29_repair_202605  # overwritten below
DIR=/home/chenhc/claude_try_MDMOT/continuation_20261005/p29_repair_20261005
PY=/home/chenhc/.conda/envs/remdet_paper/bin/python
ROLE=${1:-fit}; [[ "$ROLE" == fit || "$ROLE" == calibration ]] || exit 2
cd "$DIR"; [[ ! -e "$ROLE/flow.log" && ! -e "$ROLE/flow.exit" ]] || exit 3
set +e
env -u SUPPL_DUMP -u DET_OVERRIDE_DIR -u CKPT_OVERRIDE -u CFG_OVERRIDE MDMT_GPU_UUID=GPU-aaa79a66-b5c4-e0a0-6e2f-28c1ac0e3e6a MDMT_CACHE_ROOT=/raid/datasets/chc_data/claude_try_MDMOT_p29_repair_20261005 PYTHONDONTWRITEBYTECODE=1 SUPPL_NO_VIS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-aaa79a66-b5c4-e0a0-6e2f-28c1ac0e3e6a "$PY" -B "$DIR/export_flow.py" --role "$ROLE" > "$DIR/$ROLE/flow.log" 2>&1
rc=$?; echo "$rc" > "$DIR/$ROLE/flow.exit"; exit "$rc"
