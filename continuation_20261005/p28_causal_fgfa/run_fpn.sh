#!/usr/bin/env bash
set -euo pipefail
P28_DIR=/home/chenhc/claude_try_MDMOT/continuation_20261005/p28_causal_fgfa
P26_DIR=/home/chenhc/claude_try_MDMOT/continuation_20261005/p26_mia_baseline
P28_PY=/home/chenhc/mdmt_phase0_reproduction_20260829_202909/mdmt_env/bin/python
P28_ROLE=${1:-fit}
if [[ "$P28_ROLE" != fit && "$P28_ROLE" != calibration ]]; then
  printf '%s\n' 'Role must be fit or calibration.' >&2
  exit 2
fi
P28_EXTRA=()
if [[ "$P28_ROLE" == calibration ]]; then
  if [[ "${2:-}" != --calibration-authorized ]]; then
    printf '%s\n' 'Calibration has not been enabled in this invocation.' >&2
    exit 2
  fi
  P28_EXTRA+=(--calibration-authorized)
fi
mkdir -p "$P28_DIR/$P28_ROLE"
if [[ -e "$P28_DIR/$P28_ROLE/fpn.log" || -e "$P28_DIR/$P28_ROLE/fpn.exit" ]]; then
  printf '%s\n' 'Existing log/exit target; preserve and inspect before any new run.' >&2
  exit 3
fi
cd "$P28_DIR"
set +e
env -u SUPPL_DUMP -u DET_OVERRIDE_DIR -u CKPT_OVERRIDE -u CFG_OVERRIDE \
  PYTHONDONTWRITEBYTECODE=1 SUPPL_NO_VIS=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=GPU-aaa79a66-b5c4-e0a0-6e2f-28c1ac0e3e6a \
  PYTHONPATH="$P26_DIR/adapters:$P26_DIR/source:$P26_DIR/source/demo:$P26_DIR/source/demo/utils" \
  "$P28_PY" -B "$P28_DIR/export_fpn.py" --role "$P28_ROLE" "${P28_EXTRA[@]}" \
  > "$P28_DIR/$P28_ROLE/fpn.log" 2>&1
P28_EXIT=$?
printf '%s\n' "$P28_EXIT" > "$P28_DIR/$P28_ROLE/fpn.exit"
exit "$P28_EXIT"
