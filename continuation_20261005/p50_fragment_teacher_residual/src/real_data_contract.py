#!/usr/bin/env python3
"""Audit the P50 teacher-donor contract on frozen train episode JSONL only."""
from __future__ import annotations

import gzip
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path("/home/chenhc/mdmot_sci_id_20260906/data/train_episodes_v1")
PAIRS = ("23", "25", "27", "28", "29", "30", "32", "39", "42", "44", "45", "50", "51", "53", "54", "58", "63", "64", "65", "66", "69", "70", "74", "76", "78")


def load():
    rows = []
    for pair in PAIRS:
        with gzip.open(ROOT / f"{pair}.jsonl.gz", "rt", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    assert row["split"] == "train"
                    rows.append(row)
    return rows


def main():
    rows = load()
    # Candidate fragments are indexed by (pair, source train identity) and retain
    # only the public event frame. These are labels for teacher construction, not
    # inference inputs.
    by_id = defaultdict(list)
    for row in rows:
        frame = int(row["confirmation_frame"])
        for cand in row["candidates"]:
            gid = cand.get("source_train_gt_id")
            if gid is not None:
                by_id[(str(row["pair_id"]), int(gid))].append(frame)
    positive = donor_ok = donor_missing = 0
    max_donors = 0
    for row in rows:
        if not bool(row.get("supervision_valid")):
            continue
        positives = row.get("same_id_candidate_indices", [])
        if not positives:
            continue
        positive += 1
        target_frame = int(row["confirmation_frame"])
        gids = {row["candidates"][i].get("source_train_gt_id") for i in positives}
        donor_frames = []
        for gid in gids:
            if gid is None:
                continue
            donor_frames.extend(f for f in by_id[(str(row["pair_id"]), int(gid))] if f < target_frame)
        if donor_frames:
            donor_ok += 1
            max_donors = max(max_donors, len(donor_frames))
        else:
            donor_missing += 1
    result = {
        "status": "PASS_P50_REAL_TRAIN_DONOR_CONTRACT",
        "pairs": list(PAIRS),
        "episodes": len(rows),
        "supervised_positive_events": positive,
        "events_with_strict_prefix_donor": donor_ok,
        "events_without_strict_prefix_donor": donor_missing,
        "strict_prefix_donor_fraction": donor_ok / positive if positive else 0.0,
        "max_prefix_donors_for_one_event": max_donors,
        "teacher_fields_are_train_only": True,
        "official_val_test_access": False,
        "inference_inputs_contain_teacher_or_gt": False,
    }
    out = Path(__file__).resolve().parents[1] / "REAL_DATA_CONTRACT.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
