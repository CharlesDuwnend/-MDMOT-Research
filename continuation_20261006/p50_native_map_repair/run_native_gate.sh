#!/usr/bin/env bash
set -euo pipefail
cd /home/chenhc/claude_try_MDMOT
OUT=continuation_20261006/p50_native_map_repair
exec env CUDA_VISIBLE_DEVICES=GPU-aaa79a66-b5c4-e0a0-6e2f-28c1ac0e3e6a P50_STEPS=2400 /home/chenhc/mdmt_phase0_reproduction_20260829_202909/mdmt_env/bin/python "$OUT/src/native_set_residual_gate.py"
