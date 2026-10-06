#!/usr/bin/env python3
"""Create detector override arrays from the sealed P30 cache.

This stage reads only frozen FPN/flow archives and the P29 checkpoint. It does
not read XML or labels. Arrays use the original-image xyxy+score convention
expected by the existing MIA override runtime.
"""
import argparse
import copy
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
P29 = HERE.parent / "p29_learned_alignment"
P28 = HERE.parent / "p28_causal_fgfa"
RAID = Path("/raid/datasets/chc_data/claude_try_MDMOT_p30_full_host_20261005")
UUID = "GPU-aaa79a66-b5c4-e0a0-6e2f-28c1ac0e3e6a"

import sys
sys.path.insert(0, str(P29))
sys.path.insert(0, str(P28))
from train_residual import apply_native_capacity, decode, make_adapter, model_init, sha, state_sha


def cat_dets(arrays):
    nonempty = [np.asarray(x, dtype=np.float32) for x in arrays if len(x)]
    return np.concatenate(nonempty, axis=0) if nonempty else np.empty((0, 5), dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("native", "past_fixed_offsets"), required=True)
    args = ap.parse_args()
    out = RAID / "calibration" / "overrides" / args.mode
    assert not out.exists(), "preserve existing override attempt"
    out.mkdir(parents=True)
    index = json.loads((HERE / "calibration" / "FEATURE_INDEX.json").read_text())
    flow_index = json.loads((HERE / "calibration" / "FLOW_INDEX.json").read_text())
    entries = {x["key"]: x for x in index["entries"]}
    flows = {x["key"]: x for x in flow_index["entries"]}
    frames = json.loads((HERE / "FRAMES.json").read_text())
    records = [x for x in frames if x["role"] == "calibration"]
    model, runtime = model_init()
    frozen = state_sha(model)
    head = model.detector.bbox_head
    expected = dict(nms_pre=2000, max_per_img=300, score_thr=.05, nms_iou=.6)
    assert apply_native_capacity(head, json.loads((P29 / "TRAIN_PROTOCOL.json").read_text())) == expected
    adapter = None
    checkpoint = None
    if args.mode == "past_fixed_offsets":
        tr = json.loads((P29 / "training" / "TRAIN_RECEIPT.json").read_text())
        checkpoint = tr["checkpoints"][args.mode]
        adapter = make_adapter(args.mode, 42)
        loaded = torch.load(checkpoint["path"], map_location="cpu")
        adapter.load_state_dict(loaded["state_dict"], strict=True)
        adapter.eval().requires_grad_(False)
        assert state_sha(adapter) == checkpoint["final_state_sha256"]
    by_seq = {}
    for r in records:
        by_seq.setdefault((str(r["pair"]), int(r["view"])), []).append(r)
    start = time.monotonic(); rows = 0
    with torch.no_grad():
        for seq, seq_records in sorted(by_seq.items()):
            seq_dir = out / f"{seq[0]}-{seq[1]}"; seq_dir.mkdir()
            ordered = sorted(seq_records, key=lambda x: int(x["frame"]))
            first_frame = int(ordered[0]["frame"])
            frame_map = {int(x["frame"]): x for x in ordered}
            for ordinal, rec in enumerate(ordered):
                key = rec["key"]; entry = entries[key]
                meta = copy.deepcopy(entry["metadata"])
                meta["scale_factor"] = np.asarray(meta["scale_factor"], dtype=np.float32)
                meta["batch_input_shape"] = tuple(meta["batch_input_shape"][:2])
                with np.load(entry["archive_path"], allow_pickle=False) as z:
                    current = [torch.from_numpy(z[f"f{i}"]).cuda() for i in range(5)]
                    if args.mode == "native" or int(rec["frame"]) - first_frame < 8:
                        det = decode(head, current, meta)
                    else:
                        past = []
                        for level in range(5):
                            supports = []
                            for lag in (1, 4, 8):
                                support_rec = frame_map[int(rec["frame"]) - lag]
                                with np.load(entries[support_rec["key"]]["archive_path"], allow_pickle=False) as sz:
                                    supports.append(torch.from_numpy(sz[f"f{level}"]).cuda())
                            past.append(torch.cat(supports, dim=0))
                        flow_rec = flows[key]
                        with np.load(flow_rec["path"], allow_pickle=False) as fz:
                            flow = torch.from_numpy(fz["flow"]).cuda()
                        geometry = dict(original_hw=tuple(meta["ori_shape"][:2]),
                                        detector_img_hw=tuple(meta["img_shape"][:2]),
                                        pad_hw=tuple(meta["pad_shape"][:2]))
                        features = adapter(current, past, flow, **geometry)
                        det = decode(head, features, meta)
                    arr = cat_dets(det)
                np.save(seq_dir / f"{ordinal:06d}.npy", arr)
                rows += len(arr)
            print(json.dumps(dict(event="OVERRIDE_SEQUENCE_COMPLETE", mode=args.mode,
                                  sequence=f"{seq[0]}-{seq[1]}", frames=len(ordered), rows=rows,
                                  seconds=time.monotonic() - start)), flush=True)
    receipt = dict(status="COMPLETE_P30_DETECTOR_OVERRIDES", mode=args.mode,
                   sequence_count=len(by_seq), frame_count=len(records), output_root=str(out),
                   total_rows=rows, native_capacity=expected, p29_checkpoint=checkpoint,
                   detector_state_sha256=frozen, model_runtime=runtime,
                   feature_receipt_sha256=sha(HERE / "calibration" / "FEATURE_RECEIPT.json"),
                   flow_receipt_sha256=sha(HERE / "calibration" / "FLOW_RECEIPT.json"),
                   source_sha256=sha(__file__), gt_or_labels_read=False,
                   elapsed_seconds=time.monotonic() - start)
    (out / "RECEIPT.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
