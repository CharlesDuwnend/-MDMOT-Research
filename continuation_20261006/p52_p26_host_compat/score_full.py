#!/usr/bin/env python3
"""Development-only MOT diagnostics for pair23 P52/P26 host outputs."""
import importlib.util
import json
import statistics
from pathlib import Path
import motmetrics as mm

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
P26 = ROOT / "continuation_20261005/p26_mia_baseline"
P52 = HERE / "full_json/p52_full"
OUT = HERE / "FULL_SCORE.json"
STAGE15 = Path("/home/chenhc/mdmot_global_host_stage15_20260905/scripts/stage15_global_host.py")


def load_stage15():
    spec = importlib.util.spec_from_file_location("stage15_pair23", STAGE15)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def mot_json(path, out):
    data = json.loads(path.read_text())
    with out.open("w") as f:
        for key, rows in data.items():
            frame = int(key.split("=", 1)[1]) + 1
            for row in rows:
                x1, y1, x2, y2 = map(float, row[1:5])
                f.write(f"{frame},{int(round(float(row[0])))},{x1},{y1},{x2-x1},{y2-y1},1,1,1\n")


def score(mod, name, source):
    score_dir = HERE / "scores"
    score_dir.mkdir(exist_ok=True)
    gt = {}
    pred = {}
    for view in (1, 2):
        gt[view] = score_dir / f"gt-23-{view}.txt"
        mod.xml_to_mot("23", str(view), gt[view])
        pred[view] = score_dir / f"{name}-23-{view}.txt"
        mot_json(source / f"23-{view}.json", pred[view])
    mango, _ = mod.load_official_mda()
    mda = float(mango.calAAS(mango.assign_gt_to_frames(
        mango.read_result(str(source / "23-1.json"), str(source / "23-2.json")),
        mango.read_gt(str(gt[1]), 1), mango.read_gt(str(gt[2]), 2))))
    rows = []
    for view in (1, 2):
        g = mm.io.loadtxt(str(gt[view]), fmt="mot15-2D", min_confidence=1)
        t = mm.io.loadtxt(str(pred[view]), fmt="mot15-2D")
        acc = mm.utils.compare_to_groundtruth(g, t, "iou", distth=0.5)
        vals = mm.metrics.create().compute(acc, metrics=["idf1", "mota", "num_switches", "num_false_positives", "num_misses"]).iloc[0].to_dict()
        rows.append({k: float(v) for k, v in vals.items()})
    return {"method": name, "MDA": mda,
            "IDF1": statistics.mean(x["idf1"] for x in rows),
            "MOTA": statistics.mean(x["mota"] for x in rows),
            "num_switches": sum(x["num_switches"] for x in rows), "views": rows}


def main():
    mod = load_stage15()
    baseline = score(mod, "P26_baseline", P26 / "runs/fit23/json/mia_baseline")
    p52 = score(mod, "P52_host", P52)
    result = {
        "status": "COMPLETE_P52_P26_PAIR23_DEVELOPMENT_DIAGNOSTIC",
        "protocol": "pair23 fit development only; same XML MOT scorer as P42 diagnostic",
        "baseline": baseline, "p52_host": p52,
        "delta_p52_minus_p26": {k: p52[k] - baseline[k] for k in ("MDA", "IDF1", "MOTA", "num_switches")},
        "official_val_test_access": False,
        "formal_mot_score_authorized": False,
        "training_authorized": False,
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
