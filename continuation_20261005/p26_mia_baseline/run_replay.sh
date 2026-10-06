#!/usr/bin/env bash
set -uo pipefail
cd /home/chenhc/claude_try_MDMOT/continuation_20261005/p26_mia_baseline
/home/chenhc/.conda/envs/remdet_paper/bin/python -u replay_fit23.py > replay.log 2>&1
code=$?
printf '%s\n' "$code" > replay.exit
exit "$code"
