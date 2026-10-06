#!/bin/bash
set -u
cd /home/chenhc/claude_try_MDMOT/continuation_20261004/p23_visual
P=/home/chenhc/.conda/envs/remdet_paper/bin/python
for A in head_only visual_ft; do
  echo "=== arm $A start $(date -Is)"
  $P train_visual.py --arm $A
  echo "=== arm $A exit=$? $(date -Is)"
done
echo ARMS_DONE
