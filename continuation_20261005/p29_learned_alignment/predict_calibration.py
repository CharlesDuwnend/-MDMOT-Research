#!/usr/bin/env python3
"""Freeze P29 cap300 predictions before calibration labels are read."""
import copy
import json
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
P28 = HERE.parent / "p28_causal_fgfa"
P29_ARMS = ("native", "current_residual", "past_fixed_offsets", "past_learned_offsets")
CKPT = Path("/raid/datasets/chc_data/MDMT/checkpoints/autoassign_r50_fpn_8x2_1x_full_mdmt/epoch_60.pth")
CKPT_SHA = "8894ea5ffe8309017d78e2dac359d405b38aa66725e1b465a8dce30beb12c901"

from train_residual import (CachedGroups, apply_native_capacity, decode, dump, make_adapter,
                            model_init, sha, state_sha)


def main():
    protocol = json.loads((HERE / "TRAIN_PROTOCOL.json").read_text())
    training_path = HERE / "training" / "TRAIN_RECEIPT.json"
    training = json.loads(training_path.read_text())
    assert training["status"] == "COMPLETE_THREE_ARM_FIT"
    assert training["protocol_sha256"] == sha(HERE / "TRAIN_PROTOCOL.json")
    assert set(training["checkpoints"]) == set(protocol["arms"])
    for path, digest in training["inputs"].items():
        assert sha(path) == digest, path
    out = HERE / "predictions"
    assert not out.exists(), "preserve any existing prediction attempt"
    out.mkdir()
    groups_all = json.loads((HERE / "GROUPS.json").read_text())
    groups = [g for g in groups_all if g["role"] == "calibration"]
    dataset = CachedGroups("calibration", verify_archives=True)
    assert [g["key"] for g in dataset.groups] == [g["key"] for g in groups]
    assert len(groups) == 40
    model, runtime = model_init()
    frozen = state_sha(model)
    head = model.detector.bbox_head
    assert apply_native_capacity(head, protocol) == dict(nms_pre=2000, max_per_img=300,
                                                         score_thr=.05, nms_iou=.6)
    models = {}
    checkpoint_artifacts = {}
    for arm, mode in protocol["arms"].items():
        c = training["checkpoints"][arm]
        path = Path(c["path"])
        assert sha(path) == c["sha256"]
        checkpoint = torch.load(str(path), map_location="cpu")
        assert checkpoint["arm"] == arm and checkpoint["mode"] == mode
        assert checkpoint["protocol_sha256"] == sha(HERE / "TRAIN_PROTOCOL.json")
        adapter = make_adapter(mode, protocol["seed"])
        adapter.load_state_dict(checkpoint["state_dict"], strict=True)
        adapter.eval().requires_grad_(False)
        assert state_sha(adapter) == c["final_state_sha256"]
        models[arm] = adapter
        checkpoint_artifacts[str(path.resolve())] = c["sha256"]
    predictions = {arm: {} for arm in P29_ARMS}
    start = time.monotonic()
    from mmdet.core import bbox2result
    with torch.no_grad():
        for index, group in enumerate(groups, 1):
            key = group["key"]
            current, past, flow, meta, geometry, _ = dataset.load(key, past=True)
            native = decode(head, current, meta)
            predictions["native"][key] = {str(cls): arr.tolist() for cls, arr in enumerate(native)}
            for arm, adapter in models.items():
                features = adapter(current, past, flow, **geometry)
                det = head.simple_test(features, [copy.deepcopy(meta)], rescale=True)[0]
                arrays = bbox2result(det[0], det[1], 3)
                predictions[arm][key] = {str(cls): arr.tolist() for cls, arr in enumerate(arrays)}
            if index % 10 == 0:
                print(json.dumps({"event": "P29_CAL_PREDICTION_PROGRESS", "groups": index,
                                  "total": 40, "seconds": time.monotonic() - start}), flush=True)
    artifacts = dict(checkpoint_artifacts)
    for arm in P29_ARMS:
        path = out / (arm + ".json")
        dump(path, predictions[arm])
        artifacts[str(path.resolve())] = sha(path)
    for path in (HERE / "GROUPS.json", HERE / "CONFIG.json", HERE / "TRAIN_PROTOCOL.json",
                 training_path, HERE / "predict_calibration.py", HERE / "train_residual.py",
                 HERE / "residual_deformable.py"):
        artifacts[str(path.resolve())] = sha(path)
    artifacts[str(CKPT.resolve())] = CKPT_SHA
    inputs = {str(training_path.resolve()): sha(training_path),
              str(HERE.joinpath("GROUPS.json").resolve()): sha(HERE / "GROUPS.json"),
              str(HERE.joinpath("CONFIG.json").resolve()): sha(HERE / "CONFIG.json"),
              str(HERE.joinpath("TRAIN_PROTOCOL.json").resolve()): sha(HERE / "TRAIN_PROTOCOL.json"),
              str(CKPT.resolve()): CKPT_SHA}
    receipt = dict(status="COMPLETE_CALIBRATION_PREDICTIONS_FROZEN", group_count=40,
                   arms=list(P29_ARMS), artifacts=artifacts, predictions_by_arm={
                       arm: {"path": str((out / (arm + ".json")).resolve()),
                             "sha256": artifacts[str((out / (arm + ".json")).resolve())]}
                       for arm in P29_ARMS},
                   checkpoints_by_arm={arm: {"path": str(Path(c["path"]).resolve()),
                                             "sha256": c["sha256"]}
                                       for arm, c in training["checkpoints"].items()},
                   native_checkpoint={"path": str(CKPT.resolve()), "sha256": CKPT_SHA},
                   inputs=inputs, labels_read=False, seconds=time.monotonic() - start,
                   runtime=runtime, frozen_model_state_sha256=frozen,
                   original_output_semantics="cap300 native score_thr=.05, NMS=.6, no added postfilter",
                   training_receipt_sha256=sha(training_path),
                   protocol_sha256=sha(HERE / "TRAIN_PROTOCOL.json"))
    dump(HERE / "PREDICTION_RECEIPT.json", receipt)
    assert state_sha(model) == frozen
    print(json.dumps({"event": "P29_CALIBRATION_PREDICTIONS_COMPLETE", "groups": 40,
                      "arms": list(P29_ARMS)}), flush=True)


if __name__ == "__main__":
    main()
