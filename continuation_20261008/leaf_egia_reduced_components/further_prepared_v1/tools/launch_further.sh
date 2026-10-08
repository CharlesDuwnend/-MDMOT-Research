#!/usr/bin/env bash
set -u
cd /home/chenhc/leaf_egia_further_pure_20261008_v1 || exit 90
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
/home/chenhc/.conda/envs/u2mot/bin/python tools/run_further_grid.py --smoke-only > logs/further_smoke_launcher.log 2>&1
task_exit=$?
printf '%s\n' "$task_exit" > artifacts/smoke.exit
if [ "$task_exit" -ne 0 ]; then
    printf '%s\n' "$task_exit" > artifacts/pipeline.exit
    exit "$task_exit"
fi
/home/chenhc/.conda/envs/u2mot/bin/python tools/run_further_grid.py > logs/further_full_launcher.log 2>&1
task_exit=$?
printf '%s\n' "$task_exit" > artifacts/full.exit
printf '%s\n' "$task_exit" > artifacts/pipeline.exit
exit "$task_exit"
