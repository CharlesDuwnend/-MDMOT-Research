#!/usr/bin/env python3
"""Frozen AutoAssign visual/query-token signal audit on train episodes only.

This is deliberately parameter-free.  It does not train, read val/test, or
write predictions.  The episode labels are used only to score candidate rank
after all feature rows have been resolved by the native Stage-3 IoU contract.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path("/home/chenhc/cross_uav_query_mamba_20260830")
SCI = Path("/home/chenhc/mdmot_sci_id_20260906")
BASE = ROOT / "features" / "train"
OUT = Path(__file__).resolve().parent
KMAX = 64
FIT = ("23", "25", "27", "28", "29", "30", "32", "39", "42", "44", "45", "50", "51", "53", "54")
EVAL = ("29", "30", "32", "39")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(aa + bb - inter, 1e-12)


def load_store(pair: str, uav: str):
    rows = [json.loads(line) for line in (BASE / f"{pair}-{uav}.jsonl").open() if line.strip()]
    with np.load(BASE / f"{pair}-{uav}.npz", allow_pickle=False) as z:
        visual = z["visual_feature"].astype(np.float32)
        query = z["query_token"].astype(np.float32)
    if len(rows) != len(visual) or len(rows) != len(query):
        raise RuntimeError(f"row/feature mismatch for {pair}-{uav}")
    by_frame = defaultdict(list)
    for row in rows:
        j = int(row["feature_row"])
        by_frame[(int(row["frame_id"]), int(row["class_id"]))].append(
            (row["bbox"], visual[j], query[j], float(row.get("score", 0.0)), j)
        )
    return by_frame


def frame_maps(pair: str):
    # The episode image paths are the same paths used by Stage-3 frame records.
    import sys
    sys.path.insert(0, str(ROOT))
    import stage3_runner as s3
    maps = {}
    for record in s3.records("train", (pair,)):
        dev = str(record["device"])
        paths = s3.frame_paths(Path(record["image_dir"]))
        maps[dev] = {str(path.resolve()): i for i, path in enumerate(paths)}
    return maps


def resolve(store, frame: int, cls: int, box: Sequence[float]):
    choices = store.get((int(frame), int(cls)), [])
    if not choices:
        return None
    best = max(choices, key=lambda x: iou(box, x[0]))
    return best if iou(box, best[0]) >= 0.5 else None


def build_pair(pair: str):
    stores = {uav: load_store(pair, uav) for uav in ("1", "2")}
    paths = frame_maps(pair)
    events, dropped = [], []
    path = SCI / "data" / "train_episodes_v1" / f"{pair}.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            row = json.loads(line)
            target_path = str(Path(row["target_image"]).resolve())
            source_path = str(Path(row["source_image"]).resolve())
            td, sd = str(row["target_uav"]), str(row["source_uav"])
            if target_path not in paths.get(td, {}) or source_path not in paths.get(sd, {}):
                dropped.append({"line": line_no, "reason": "image_path_unresolved"})
                continue
            tf, sf = paths[td][target_path], paths[sd][source_path]
            if tf != sf:
                dropped.append({"line": line_no, "reason": "paired_frame_mismatch"})
                continue
            t = resolve(stores[td], tf, int(row["class"]), row["target_bbox_xyxy"])
            if t is None:
                dropped.append({"line": line_no, "reason": "target_feature_missing"})
                continue
            candidates = []
            missing = False
            for candidate in row.get("candidates", [])[:KMAX]:
                c = resolve(stores[sd], sf, int(candidate["class"]), candidate["bbox_xyxy"])
                if c is None:
                    missing = True
                    break
                candidates.append(c)
            if missing:
                dropped.append({"line": line_no, "reason": "candidate_feature_missing"})
                continue
            positives = [int(j) for j in row.get("same_id_candidate_indices", []) if int(j) < len(candidates)]
            if not positives:
                continue
            events.append({
                "pair": pair,
                "frame": int(tf),
                "target_uav": td,
                "source_uav": sd,
                "a_visual": t[1],
                "a_query": t[2],
                "c_visual": np.stack([c[1] for c in candidates], axis=0) if candidates else np.empty((0, 256), np.float32),
                "c_query": np.stack([c[2] for c in candidates], axis=0) if candidates else np.empty((0, 285), np.float32),
                "positive": positives,
                "candidate_count": len(candidates),
            })
    return events, {"pair": pair, "events": len(events), "dropped": dropped}


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.maximum(np.linalg.norm(a), 1e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    return b @ a


def rank_of(scores: np.ndarray, positives: Sequence[int]) -> Optional[int]:
    if not len(scores):
        return None
    order = np.argsort(-scores, kind="stable")
    p = set(int(i) for i in positives)
    for rank, idx in enumerate(order, 1):
        if int(idx) in p:
            return rank
    return None


def score(events: Sequence[dict], mode: str, alpha: float = 0.5):
    ranks, mrr = [], []
    for event in events:
        sv = cosine(event["a_visual"], event["c_visual"])
        sq = cosine(event["a_query"], event["c_query"])
        if mode == "visual":
            s = sv
        elif mode == "query":
            s = sq
        elif mode == "equal_fusion":
            # Equal-weight fusion is pre-registered and parameter-free; each
            # branch is cosine-normalized, so neither dimension dominates.
            s = alpha * sv + (1.0 - alpha) * sq
        elif mode == "query_residual":
            # A fixed residual tests whether query tokens add complementary
            # evidence without turning this audit into a learned ranker.
            s = sv + alpha * (sq - sv)
        else:
            raise ValueError(mode)
        r = rank_of(s, event["positive"])
        if r is not None:
            ranks.append(r)
            mrr.append(1.0 / r)
    return {
        "events": len(ranks),
        "recall1": float(np.mean(np.asarray(ranks) == 1)) if ranks else None,
        "mrr": float(np.mean(mrr)) if mrr else None,
        "median_rank": float(np.median(ranks)) if ranks else None,
    }


def main() -> None:
    data, audits = {}, {}
    for pair in tuple(dict.fromkeys(FIT + EVAL)):
        data[pair], audits[pair] = build_pair(pair)
        print(json.dumps({"pair": pair, **{k: v for k, v in audits[pair].items() if k != "dropped"}, "dropped": len(audits[pair]["dropped"])}, ensure_ascii=False), flush=True)
    modes = ("visual", "query", "equal_fusion", "query_residual")
    per_pair = {mode: {pair: score(data[pair], mode) for pair in EVAL} for mode in modes}
    macro = {
        mode: {
            key: float(np.mean([per_pair[mode][pair][key] for pair in EVAL]))
            for key in ("recall1", "mrr", "median_rank")
        }
        for mode in modes
    }
    wins_vs_visual = {
        mode: sum(per_pair[mode][pair]["recall1"] > per_pair["visual"][pair]["recall1"] for pair in EVAL)
        for mode in modes if mode != "visual"
    }
    result = {
        "status": "COMPLETE_QUERY_TOKEN_TRAIN_FREE_SIGNAL_AUDIT",
        "split": "train_only",
        "fit_pairs": list(FIT),
        "eval_pairs": list(EVAL),
        "modes": modes,
        "alpha": 0.5,
        "per_pair": per_pair,
        "macro": macro,
        "wins_vs_visual": wins_vs_visual,
        "controls": {
            "no_parameter_update": True,
            "official_val_test_access": False,
            "gt_used_only_for_positive_rank_labels": True,
            "candidate_set_and_denominator_identical": True,
            "visual_cache": str(BASE),
            "visual_cache_sha256_23_1": sha(BASE / "23-1.npz"),
            "query_token_dim": 285,
            "visual_dim": 256,
        },
        "audits": audits,
    }
    (OUT / "QUERY_SIGNAL_AUDIT.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "macro": macro, "wins_vs_visual": wins_vs_visual}, ensure_ascii=False))


if __name__ == "__main__":
    main()
