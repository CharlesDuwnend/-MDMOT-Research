#!/usr/bin/env bash
set -u
cd /home/chenhc/leaf_egia_detailed_20261008_v2 || exit 90
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
bash tools/launch_smoke.sh
task_exit=$?
if [ "$task_exit" -ne 0 ]; then
    printf '%s\n' "$task_exit" > artifacts/pipeline.exit
    exit "$task_exit"
fi
CUDA_VISIBLE_DEVICES='' /home/chenhc/.conda/envs/u2mot/bin/python tools/seal_corrected_smoke.py > logs/identity_seal.log 2>&1
task_exit=$?
if [ "$task_exit" -ne 0 ]; then
    printf '%s\n' "$task_exit" > artifacts/pipeline.exit
    exit "$task_exit"
fi
bash tools/launch_all.sh
task_exit=$?
printf '%s\n' "$task_exit" > artifacts/pipeline.exit
exit "$task_exit"
