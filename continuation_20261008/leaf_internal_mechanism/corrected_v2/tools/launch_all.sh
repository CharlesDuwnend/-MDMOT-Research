#!/usr/bin/env bash
set +e
cd /home/chenhc/leaf_internal_mechanism_20261008_v2_metadata || exit 90
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
/home/chenhc/.conda/envs/u2mot/bin/python tools/run_mechanism.py > logs/launcher.log 2>&1
task_exit=$?
printf '%s\n' "$task_exit" > artifacts/launcher.exit
exit "$task_exit"
