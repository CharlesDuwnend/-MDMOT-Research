#!/usr/bin/env bash
set -euo pipefail
P28_DIR=/home/chenhc/claude_try_MDMOT/continuation_20261005/p28_causal_fgfa
P28_PY=/home/chenhc/mdmt_phase0_reproduction_20260829_202909/mdmt_env/bin/python
cd "$P28_DIR"
test "$(cat train.exit)" = 0
test -f training/TRAIN_RECEIPT.json
test ! -e calibration_pipeline.exit
test ! -e PREDICTION_RECEIPT.json
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=GPU-aaa79a66-b5c4-e0a0-6e2f-28c1ac0e3e6a
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
run_steps() {
  bash run_fpn.sh calibration --calibration-authorized || return $?
  "$P28_PY" -B verify_fpn.py --role calibration > calibration/verify.log 2>&1 || return $?
  bash run_flow.sh calibration || return $?
  "$P28_PY" -B -u predict_calibration.py > calibration/predict.log 2>&1 || return $?
}
set +e
run_steps
P28_EXIT=$?
printf '%s\n' "$P28_EXIT" > calibration_pipeline.exit
exit "$P28_EXIT"
