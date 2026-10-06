#!/usr/bin/env bash
set -u
P26=/home/chenhc/claude_try_MDMOT/continuation_20261005/p26_mia_baseline
P48=/home/chenhc/claude_try_MDMOT/continuation_20261005/p48_cross_view_kinematic_transport
MDMT_PY=/home/chenhc/mdmt_phase0_reproduction_20260829_202909/mdmt_env/bin/python
OUT="$P48/runs/trace_pair23_v1"
mkdir -p "$OUT"
nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used --format=csv,noheader,nounits > "$OUT/GPU_BEFORE.txt"
env -u SUPPL_DUMP -u DET_OVERRIDE_DIR -u CKPT_OVERRIDE -u CFG_OVERRIDE \
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-1b297aba-ae7e-e326-5903-476f1bb683d7 \
  OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 SUPPL_NO_VIS=1 \
  P48_TRACE_PATH="$OUT/trace.jsonl" \
  PYTHONPATH="$P26/adapters:$P26/source:$P26/source/demo:$P26/source/demo/utils" \
  "$MDMT_PY" "$P48/trace_baseline.py" \
  --config "$P26/source/configs/mot/bytetrack/bytetrack_autoassign_full_mdmt-private-half.py" \
  --checkpoint /raid/datasets/chc_data/MDMT/checkpoints/autoassign_r50_fpn_8x2_1x_full_mdmt/epoch_60.pth \
  --input "$P26/inputs/fit23/1/" --xml_dir "$P26/xml/" \
  --result_dir "$OUT/json" --method traced_baseline \
  --output "$OUT/vis1" --output2 "$OUT/vis2" --device cuda:0 > "$OUT/inference.log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$OUT/launcher.exit"
exit "$rc"
