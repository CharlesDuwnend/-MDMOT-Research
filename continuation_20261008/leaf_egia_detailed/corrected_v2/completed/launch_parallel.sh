#!/usr/bin/env bash
set -u
cd /home/chenhc/leaf_egia_detailed_20261008_v2 || exit 90
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
/home/chenhc/.conda/envs/u2mot/bin/python review/parallel_coordinator.py > review/parallel_coordinator.log 2>&1
task_exit=$?
printf '%s\n' "$task_exit" > artifacts/launcher.exit
printf '%s\n' "$task_exit" > artifacts/pipeline.exit
if [ "$task_exit" -eq 0 ]; then
    /home/chenhc/.conda/envs/u2mot/bin/python review/finalize_when_complete.py > review/parallel_finalization.log 2>&1
fi
exit "$task_exit"
