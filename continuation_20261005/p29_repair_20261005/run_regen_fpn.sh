#!/usr/bin/env bash
set -euo pipefail
DIR=/home/chenhc/claude_try_MDMOT/continuation_20261005/p29_repair_20261005
PY=/home/chenhc/mdmt_phase0_reproduction_20260829_202909/mdmt_env/bin/python
ROLE=${1:-fit}; [[ "$ROLE" == fit || "$ROLE" == calibration ]] || exit 2
ARGS=(); [[ "$ROLE" == calibration ]] && { [[ "${2:-}" == --calibration-authorized ]] || exit 2; ARGS+=(--calibration-authorized); }
cd "$DIR"; [[ ! -e "$ROLE/fpn.log" && ! -e "$ROLE/fpn.exit" ]] || exit 3
set +e
env -u SUPPL_DUMP -u DET_OVERRIDE_DIR -u CKPT_OVERRIDE -u CFG_OVERRIDE MDMT_GPU_UUID=GPU-aaa79a66-b5c4-e0a0-6e2f-28c1ac0e3e6a MDMT_CACHE_ROOT=/raid/datasets/chc_data/claude_try_MDMOT_p29_repair_20261005 PYTHONDONTWRITEBYTECODE=1 SUPPL_NO_VIS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-aaa79a66-b5c4-e0a0-6e2f-28c1ac0e3e6a PYTHONPATH="$DIR/../p26_mia_baseline/adapters:$DIR/../p26_mia_baseline/source:$DIR/../p26_mia_baseline/source/demo:$DIR/../p26_mia_baseline/source/demo/utils" "$PY" -B "$DIR/export_fpn.py" --role "$ROLE" "${ARGS[@]}" > "$DIR/$ROLE/fpn.log" 2>&1
rc=$?; echo "$rc" > "$DIR/$ROLE/fpn.exit"; exit "$rc"
