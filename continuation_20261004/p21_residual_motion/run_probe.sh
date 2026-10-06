#!/usr/bin/env bash
set -u
cd /home/chenhc/claude_try_MDMOT/continuation_20261004/p21_residual_motion
export CUDA_VISIBLE_DEVICES=''
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
/home/chenhc/.conda/envs/remdet_paper/bin/python -B probe.py --run > run.log 2>&1
result=$?
printf '%s\n' "$result" > launcher.exit
exit "$result"
