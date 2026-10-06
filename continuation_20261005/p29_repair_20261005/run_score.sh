#!/usr/bin/env bash
set -uo pipefail
D=/home/chenhc/claude_try_MDMOT/continuation_20261005/p29_repair_20261005
cd "$D" || exit 2
if [[ -e score.log || -e score.exit ]]; then exit 3; fi
/home/chenhc/mdmt_phase0_reproduction_20260829_202909/mdmt_env/bin/python -B -u "$D/score_detection.py" > score.log 2>&1
s=$?; printf '%s\n' "$s" > score.exit; exit "$s"
