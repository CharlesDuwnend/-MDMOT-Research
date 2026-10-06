#!/usr/bin/env bash
set -u
cd /home/chenhc/claude_try_MDMOT/continuation_20261004/p22_dense_motion
export PYTHONDONTWRITEBYTECODE=1
/home/chenhc/.conda/envs/remdet_paper/bin/python -B extract_dense.py --run > run.log 2>&1
result=$?
printf '%s\n' "$result" > launcher.exit
exit "$result"
