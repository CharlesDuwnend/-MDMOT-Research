#!/usr/bin/env bash
set -uo pipefail
cd /home/chenhc/claude_try_MDMOT/continuation_20261005/p27_strong_host_temporal
export CUDA_VISIBLE_DEVICES=GPU-aaa79a66-b5c4-e0a0-6e2f-28c1ac0e3e6a
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
/home/chenhc/mdmt_phase0_reproduction_20260829_202909/mdmt_env/bin/python -u detector_parity.py > parity.log 2>&1
code=$?
printf '%s\n' "$code" > parity.exit
exit "$code"
