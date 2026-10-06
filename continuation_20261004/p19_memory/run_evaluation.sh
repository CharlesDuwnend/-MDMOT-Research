#!/bin/bash
set -uo pipefail
cd /home/chenhc/claude_try_MDMOT/continuation_20261004/p19_memory
export CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
/home/chenhc/.conda/envs/remdet_paper/bin/python -B evaluate_frozen_cal5.py > evaluation.log 2>&1
p19_eval_status=$?
printf '%s\n' "$p19_eval_status" > evaluation.exit
exit "$p19_eval_status"
