#!/usr/bin/env bash
set -uo pipefail
cd /home/chenhc/claude_try_MDMOT/continuation_20261005/p30_full_host
role=${1:-fit}
export CUDA_VISIBLE_DEVICES=GPU-aaa79a66-b5c4-e0a0-6e2f-28c1ac0e3e6a
export CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
/home/chenhc/.conda/envs/remdet_paper/bin/python -u export_flow.py --role "$role" > "flow_${role}.log" 2>&1
code=$?
printf '%s\n' "$code" > "flow_${role}.exit"
exit "$code"
