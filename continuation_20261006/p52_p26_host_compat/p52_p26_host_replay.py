#!/usr/bin/env python3
"""Inject the audited P52 scorer into the protected P26 fit23 host.

The P26 wrapper is imported unchanged.  Runtime hooks replace only the three
cross-view ID refresh functions.  Detection, local tracking, initialization,
homography construction, and collision handling stay in the original wrapper.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
P26 = ROOT / "continuation_20261005/p26_mia_baseline"
FEATURE_ROOT = Path("/home/chenhc/cross_uav_query_mamba_20260830/features/train")
P52_SRC = ROOT / "continuation_20261006/p50_native_map_repair/src"
P52_CKPT = ROOT / "continuation_20261006/p50_native_map_repair/counterfactual_cf_on_seed7.pt"
SEQ = "23"
MAX_FEATURE_IOU = 0.50

sys.path.insert(0, str(P52_SRC))
sys.path[:0] = [
    str(P26 / "adapters"), str(P26 / "source"),
    str(P26 / "source/demo"), str(P26 / "source/demo/utils"),
]
import native_set_residual_gate as p52  # noqa: E402


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def iou(a, b):
    x1, y1 = max(float(a[0]), float(b[0])), max(float(a[1]), float(b[1]))
    x2, y2 = min(float(a[2]), float(b[2])), min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    aa = max(0.0, float(a[2]) - float(a[0])) * max(0.0, float(a[3]) - float(a[1]))
    bb = max(0.0, float(b[2]) - float(b[0])) * max(0.0, float(b[3]) - float(b[1]))
    return inter / max(aa + bb - inter, 1e-9)


def center(row):
    return np.asarray([(float(row[1]) + float(row[3])) / 2.0,
                       (float(row[2]) + float(row[4])) / 2.0], dtype=np.float32)


class FeatureStore:
    def __init__(self, sequence=SEQ):
        self.by_view_frame = {}
        self.receipt = {}
        for view in (1, 2):
            index_path = FEATURE_ROOT / f"{sequence}-{view}.jsonl"
            shard_path = FEATURE_ROOT / f"{sequence}-{view}.npz"
            rows = [json.loads(line) for line in index_path.read_text().splitlines() if line.strip()]
            visual = np.load(shard_path, allow_pickle=False)["visual_feature"].astype(np.float32)
            if len(rows) != len(visual):
                raise RuntimeError(f"feature index/shard mismatch for view {view}")
            grouped = defaultdict(list)
            for row in rows:
                grouped[int(row["frame_id"])].append((row["bbox"], visual[int(row["feature_row"])]))
            self.by_view_frame[view] = dict(grouped)
            self.receipt[str(view)] = {
                "index": str(index_path), "index_sha256": sha(index_path),
                "shard": str(shard_path), "shard_sha256": sha(shard_path),
                "rows": len(rows), "frames": len(grouped),
            }

    def resolve(self, view, frame, row):
        candidates = self.by_view_frame.get(int(view), {}).get(int(frame), [])
        if not candidates:
            return None, 0.0
        best_iou, best = max(((iou(row[1:5], box), feature) for box, feature in candidates), key=lambda x: x[0])
        if best_iou < MAX_FEATURE_IOU:
            return None, best_iou
        return np.asarray(best, dtype=np.float32), best_iou


class HookState:
    def __init__(self, store, checkpoint=P52_CKPT, device="cpu"):
        self.store = store
        self.device = torch.device(device)
        self.model = p52.NativeSetResidual(relational=True).to(self.device).eval()
        payload = torch.load(checkpoint, map_location=self.device)
        self.model.load_state_dict(payload["state_dict"])
        self.checkpoint = checkpoint
        self.frame = None
        self.stats = defaultdict(int)
        self.samples = []

    def score(self, target_view, source_view, target_row, source_rows, threshold):
        """Return the original geometry candidate order with learned scores."""
        if self.frame is None:
            self.stats["missing_frame_context"] += 1
            return None
        target_feature, target_iou = self.store.resolve(target_view, self.frame, target_row)
        candidate_features = []
        candidate_indices = []
        candidate_ious = []
        for j, row in enumerate(source_rows):
            feature, match_iou = self.store.resolve(source_view, self.frame, row)
            if feature is not None:
                candidate_indices.append(j)
                candidate_features.append(feature)
                candidate_ious.append(match_iou)
        if target_feature is None or not candidate_features:
            self.stats["fallback_missing_feature"] += 1
            self.stats["target_feature_missing"] += int(target_feature is None)
            self.stats["candidate_feature_set_empty"] += int(not candidate_features)
            return None
        a = torch.from_numpy(target_feature[None]).to(self.device)
        c = torch.from_numpy(np.stack(candidate_features, axis=0)[None]).to(self.device)
        mask = torch.ones((1, len(candidate_features)), dtype=torch.bool, device=self.device)
        with torch.no_grad():
            logits, dust, _ = self.model(a, c, mask)
        order = torch.argsort(logits[0], descending=True).tolist()
        best_pos = int(order[0])
        best_j = int(candidate_indices[best_pos])
        best_logit = float(logits[0, best_pos].item())
        dust_value = float(dust[0].item())
        self.stats["scored_events"] += 1
        self.stats["candidate_rows_scored"] += len(candidate_indices)
        self.stats["dustbin_selected"] += int(dust_value > best_logit)
        if len(self.samples) < 24:
            self.samples.append({
                "frame": int(self.frame), "target_view": int(target_view),
                "source_view": int(source_view), "target_iou": float(target_iou),
                "candidate_count": len(candidate_indices),
                "candidate_ious": [float(x) for x in candidate_ious],
                "selected_source_index": best_j, "selected_logit": best_logit,
                "dustbin": dust_value, "dustbin_wins": bool(dust_value > best_logit),
            })
        if dust_value > best_logit:
            return {"index": None, "scores": [float(logits[0, q].item()) for q in range(len(candidate_indices))],
                    "dustbin": dust_value, "candidate_indices": candidate_indices}
        return {"index": best_j, "scores": [float(logits[0, q].item()) for q in range(len(candidate_indices))],
                "dustbin": dust_value, "candidate_indices": candidate_indices}


def _project(H, point):
    if H is None or not np.isfinite(H).all():
        return None
    out = cv2.perspectiveTransform(np.asarray(point, dtype=np.float32).reshape(1, 1, 2), H)
    if out is None or not np.isfinite(out).all():
        return None
    return out.reshape(2)


def _geometry_candidates(points, rows, H, threshold):
    projected = _project(H, points)
    if projected is None:
        return [], None
    distances = np.asarray([float(np.linalg.norm(projected - center(row))) for row in rows], dtype=np.float32)
    return [int(x) for x in np.where(distances < float(threshold))[0]], distances


def make_refresh(state, direction):
    """Build a drop-in P26 refresh function for A->B or B->A."""
    def refresh(new_ids, new_points, new_corners, H, target_centers, rows_a, rows_b,
                matched_ids, det_a, det_b, image, co_ids, thres=100):
        if not new_points:
            return rows_a, rows_b, matched_ids, 0, co_ids
        # P26 has no class field in track_bboxes; the wrapper's selected class
        # is therefore preserved and the existing geometric gate defines the
        # candidate pool.  Feature resolution is bbox/IoU based and independent
        # of labels or GT.
        # ``new_ids`` and ``new_points`` belong to the owner side.  The other
        # side supplies the geometry-gated candidates that may inherit the ID.
        if direction == "A_TO_B":
            owner_view, candidate_view = 1, 2
            owner_rows, candidate_rows_all = rows_a, rows_b
        else:
            owner_view, candidate_view = 2, 1
            owner_rows, candidate_rows_all = rows_b, rows_a
        for idx, source_id in enumerate(new_ids):
            source_point = np.asarray(new_points[idx], dtype=np.float32)
            geom_indices, distances = _geometry_candidates(source_point, candidate_rows_all, H, thres)
            if not geom_indices:
                continue
            candidate_rows = [candidate_rows_all[j] for j in geom_indices]
            target_matches = np.where(owner_rows[:, 0] == source_id)[0] if len(owner_rows) else []
            if len(target_matches) == 0:
                continue
            target_row = owner_rows[int(target_matches[0])]
            scored = state.score(owner_view, candidate_view, target_row, candidate_rows, thres)
            if scored is None:
                chosen_local = int(np.argmin(distances[geom_indices]))
                chosen = int(geom_indices[chosen_local])
                state.stats["geometry_fallback_choice"] += 1
            else:
                if scored["index"] is None:
                    state.stats["learned_dustbin_skip"] += 1
                    continue
                # ``score`` returns an index within the geometry-gated
                # candidate list; convert exactly once to the owner-side row.
                chosen = int(geom_indices[int(scored["index"])])
                state.stats["learned_choice"] += 1
            chosen_global = int(chosen)
            if candidate_rows_all[chosen_global, 0] in matched_ids:
                state.stats["collision_skip"] += 1
                continue
            min_id = int(min(candidate_rows_all[chosen_global, 0], source_id))
            a_index = int(target_matches[0])
            owner_rows[a_index, 0] = min_id
            candidate_rows_all[chosen_global, 0] = min_id
            co_ids.append(min_id)
            matched_ids.append(min_id)
            state.stats["relabels"] += 1
        return rows_a, rows_b, matched_ids, 0, co_ids
    return refresh


def make_same_refresh(state):
    # The old-unmatched path has the same contract as A refresh; direction is
    # inferred from the argument names at the call site by using the A path.
    return make_refresh(state, "A_TO_B")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("baseline", "p52"), default="p52")
    parser.add_argument("--steps", type=int, default=None, help="optional frame limit for smoke audits")
    args, passthrough = parser.parse_known_args()
    if args.steps is not None and args.steps <= 0:
        raise ValueError("--steps must be positive")

    store = FeatureStore()
    state = HookState(store)
    if args.mode == "baseline":
        # Baseline mode disables learned hooks while keeping this process's
        # feature and frame accounting available for a step-zero comparison.
        state.score = lambda *a, **k: None

    import importlib.util
    spec = importlib.util.spec_from_file_location("p26_supplement_mia", P26 / "wrappers/supplement_mia.py")
    p26 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p26)
    original_inference = p26.inference_mot

    def wrapped_inference(model, img, frame_id, *a, **kw):
        state.frame = int(frame_id)
        return original_inference(model, img, frame_id, *a, **kw)

    p26.inference_mot = wrapped_inference
    p26.A_same_target_refresh_same_ID = make_refresh(state, "A_TO_B")
    p26.B_same_target_refresh_same_ID = make_refresh(state, "B_TO_A")
    p26.same_target_refresh_same_ID = make_same_refresh(state)

    old_argv = sys.argv
    sys.argv = [str(P26 / "wrappers/supplement_mia.py")] + passthrough
    try:
        p26.main()
    finally:
        sys.argv = old_argv
    out = HERE / f"HOST_REPLAY_{args.mode}.json"
    receipt = {
        "status": "COMPLETE_P52_P26_HOST_REPLAY",
        "mode": args.mode,
        "sequence": SEQ,
        "frame_limit": args.steps,
        "checkpoint": str(P52_CKPT),
        "checkpoint_sha256": sha(P52_CKPT),
        "feature_store": state.store.receipt,
        "stats": dict(state.stats),
        "samples": state.samples,
        "official_val_test_access": False,
        "p26_source_sha256": sha(P26 / "wrappers/supplement_mia.py"),
    }
    out.write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
