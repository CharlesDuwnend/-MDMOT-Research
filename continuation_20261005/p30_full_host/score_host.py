#!/usr/bin/env python3
"""Score frozen P30 MIA-host outputs on the five calibration pairs.

The detector arrays and tracker JSON are frozen before this script reads XML.
Scores are calibration diagnostics: they are not official val/test evidence.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import os
import statistics
import sys
import time
from pathlib import Path

import motmetrics as mm

HERE = Path(__file__).resolve().parent
RAID = Path("/raid/datasets/chc_data/claude_try_MDMOT_p30_full_host_20261005")
PAIRS = ["27", "32", "42", "64", "65"]
P24 = Path("/home/chenhc/mdmot_p24_test_20261004/official_demo")
STAGE15 = Path("/home/chenhc/mdmot_global_host_stage15_20260905/scripts/stage15_global_host.py")
MANGO = Path("/home/chenhc/mdmot_strong_host_stage1_20260905/source_compat/demo/eval/mango_eval.py")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_stage15():
    spec = importlib.util.spec_from_file_location("p30_stage15_helper", STAGE15)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def write_mot(json_path: Path, mot_path: Path):
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    keys = list(payload)
    expected = [f"frame={i}" for i in range(len(keys))]
    if keys != expected:
        raise ValueError(f"non-contiguous frame keys: {json_path}")
    rows = 0
    duplicates = 0
    bad = 0
    with mot_path.open("w", encoding="utf-8") as f:
        for key, objects in payload.items():
            frame = int(key.split("=", 1)[1]) + 1
            ids = []
            for obj in objects:
                if len(obj) < 5:
                    raise ValueError(f"short detection row in {json_path}: {obj}")
                gid = int(float(obj[0]))
                x1, y1, x2, y2 = map(float, obj[1:5])
                if not all(map(lambda x: x == x and abs(x) < 1e9, (x1, y1, x2, y2))):
                    raise ValueError(f"non-finite detection row in {json_path}")
                bad += int(x2 <= x1 or y2 <= y1)
                ids.append(gid)
                f.write(f"{frame},{gid},{x1:.9g},{y1:.9g},{x2-x1:.9g},{y2-y1:.9g},1,1,1\n")
                rows += 1
            duplicates += len(ids) - len(set(ids))
    return {"frames": len(keys), "rows": rows, "duplicate_ids_per_frame": duplicates,
            "nonpositive_boxes": bad}


def score_arm(mod, mode: str):
    host_mode = "fixed" if mode == "past_fixed_offsets" else mode
    method = "p30_fixed" if mode == "past_fixed_offsets" else "p30_native"
    pred_dir = RAID / "host" / host_mode / "json" / method
    score_dir = RAID / "host" / host_mode / "score"
    score_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for pair in PAIRS:
        gt_paths = {}
        pred_paths = {}
        structure = {}
        for view in (1, 2):
            gt = score_dir / f"gt-{pair}-{view}.txt"
            mod.xml_to_mot(pair, str(view), gt)
            gt_paths[view] = gt
            pred_json = pred_dir / f"{pair}-{view}.json"
            if not pred_json.is_file():
                raise FileNotFoundError(pred_json)
            pred = score_dir / f"{pair}-{view}.txt"
            structure[view] = write_mot(pred_json, pred)
            pred_paths[view] = pred
        # Official MDA reads the JSON frame order and attaches XML GT by frame.
        capture = io.StringIO()
        frames = mod.load_official_mda()[0].read_result(
            str(pred_dir / f"{pair}-1.json"), str(pred_dir / f"{pair}-2.json"))
        mango = mod.load_official_mda()[0]
        gt_a = mango.read_gt(str(gt_paths[1]), 1)
        gt_b = mango.read_gt(str(gt_paths[2]), 2)
        mango.assign_gt_to_frames(frames, gt_a, gt_b)
        with contextlib.redirect_stdout(capture):
            mda = float(mango.calAAS(frames))
        view_metrics = {}
        metrics = ["idf1", "idp", "idr", "mota", "motp", "num_switches",
                   "idtp", "idfp", "idfn", "num_objects", "num_predictions",
                   "num_false_positives", "num_misses"]
        for view in (1, 2):
            gt = mm.io.loadtxt(str(gt_paths[view]), fmt="mot15-2D", min_confidence=1)
            tr = mm.io.loadtxt(str(pred_paths[view]), fmt="mot15-2D")
            acc = mm.utils.compare_to_groundtruth(gt, tr, "iou", distth=0.5)
            vals = mm.metrics.create().compute(acc, metrics=metrics).iloc[0].to_dict()
            view_metrics[str(view)] = {k: float(vals[k]) for k in metrics}
        row = {"pair_id": pair, "MDA": mda,
               **{k: statistics.mean(view_metrics[str(v)][k] for v in (1, 2))
                  for k in ("idf1", "idp", "idr", "mota", "motp")},
               "num_switches": sum(view_metrics[str(v)]["num_switches"] for v in (1, 2)),
               "views": view_metrics, "structure": structure,
               "official_stdout": capture.getvalue()}
        rows.append(row)
        print(json.dumps({"mode": mode, "pair": pair, "MDA": row["MDA"],
                          "idf1": row["idf1"], "mota": row["mota"]}), flush=True)
    macro = {k: statistics.mean(row[k] for row in rows)
             for k in ("MDA", "idf1", "idp", "idr", "mota", "motp")}
    out = {"status": "COMPLETE_P30_HOST_CALIBRATION_SCORE", "mode": mode,
           "method": method, "pairs": PAIRS, "pair_count": len(rows),
           "macro": macro, "macro_percent": {k: 100 * v for k, v in macro.items()},
           "aggregation": "equal mean over five pair rows; MOT first mean over two views",
           "protocol": "calibration pairs 27/32/42/64/65; first-frame XML MIA initialization; no official val/test",
           "rows": rows, "script_sha256": sha(HERE / "score_host.py"),
           "mango_sha256": sha(MANGO)}
    (score_dir / "SUMMARY.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    return out


def main():
    started = time.monotonic()
    mod = load_stage15()
    modes = sys.argv[1:] or ["native", "past_fixed_offsets"]
    results = [score_arm(mod, mode) for mode in modes]
    if len(results) == 2 and {x["mode"] for x in results} == {"native", "past_fixed_offsets"}:
        by = {x["mode"]: x for x in results}
        deltas = {k: by["past_fixed_offsets"]["macro"][k] - by["native"]["macro"][k]
                  for k in ("MDA", "idf1", "mota")}
        comparison = {"fixed_minus_native": deltas,
                      "fixed_minus_native_percentage_points": {k: 100 * v for k, v in deltas.items()},
                      "pair_deltas": {pair: {
                          k: next(r for r in by["past_fixed_offsets"]["rows"] if r["pair_id"] == pair)[k]
                           - next(r for r in by["native"]["rows"] if r["pair_id"] == pair)[k]
                          for k in ("MDA", "idf1", "mota")} for pair in PAIRS}}
    else:
        comparison = None
    report = {"status": "COMPLETE_P30_HOST_COMPARISON", "modes": [x["mode"] for x in results],
              "results": results, "comparison": comparison,
              "elapsed_seconds": time.monotonic() - started}
    (RAID / "host" / "HOST_SCORE.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": report["status"], "comparison": comparison,
                      "elapsed_seconds": report["elapsed_seconds"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
