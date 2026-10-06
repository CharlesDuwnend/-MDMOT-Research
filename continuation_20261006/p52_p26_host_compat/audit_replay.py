#!/usr/bin/env python3
"""Audit P52/P26 fit23 replay outputs without reading official validation/test."""
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
P26 = ROOT / "continuation_20261005/p26_mia_baseline"
P52 = HERE / "full_json/p52_full"
BASE = P26 / "runs/fit23/json/mia_baseline"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text())


def inspect(data):
    duplicate_frames = 0
    rows = 0
    frame_ids = set()
    for key, values in data.items():
        if not key.startswith("frame="):
            raise AssertionError(f"unexpected frame key: {key}")
        frame_ids.add(int(key.split("=", 1)[1]))
        ids = [int(round(float(row[0]))) for row in values]
        duplicate_frames += int(len(ids) != len(set(ids)))
        rows += len(values)
    return {"frames": len(data), "min_frame": min(frame_ids), "max_frame": max(frame_ids),
            "rows": rows, "duplicate_id_frames": duplicate_frames}


def main():
    views = {}
    changed_frames = {}
    for view in (1, 2):
        p52 = P52 / f"23-{view}.json"
        base = BASE / f"23-{view}.json"
        a = read(p52); b = read(base)
        ia, ib = inspect(a), inspect(b)
        assert ia["frames"] == ib["frames"] == 700
        assert ia["min_frame"] == ib["min_frame"] == 0
        assert ia["max_frame"] == ib["max_frame"] == 699
        assert ia["duplicate_id_frames"] == 0
        changed = [key for key in a if a[key] != b[key]]
        views[str(view)] = {"p52": ia, "p26": ib,
                            "p52_sha256": sha(p52), "p26_sha256": sha(base)}
        changed_frames[str(view)] = {"count": len(changed), "first": changed[:20]}
    receipt = read(HERE / "HOST_REPLAY_p52.json")
    assert receipt["official_val_test_access"] is False
    result = {
        "status": "PASS_P52_P26_FULL_REPLAY_STRUCTURE_AUDIT",
        "sequence": "23", "views": views, "changed_frames": changed_frames,
        "hook_stats": receipt["stats"],
        "checkpoint_sha256": receipt["checkpoint_sha256"],
        "official_val_test_access": False,
        "formal_mot_score_authorized": False,
    }
    out = HERE / "FULL_REPLAY_AUDIT.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
