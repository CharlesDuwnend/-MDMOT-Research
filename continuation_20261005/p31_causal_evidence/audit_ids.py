#!/usr/bin/env python3
"""Audit train-only XML ID overlap used to define P31 supervision boundaries.

This reads only MDMT/new_xml/{1,2} train annotations.  The overlap is an
annotation convention audit, not evidence of physical identity, calibration,
or cross-view synchronization.  It must never be used as an official metric.
"""
from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path


ROOT = Path("/raid/datasets/chc_data/MDMT/new_xml")
PAIRS = [23, 25, 27, 28, 29, 32, 39, 42, 44, 51, 53, 63, 64, 65, 66, 69, 70, 74, 78]


def read_view(view: int, seq: int) -> dict[int, set[int]]:
    path = ROOT / str(view) / f"{seq}-{view}.xml"
    if not path.exists():
        raise FileNotFoundError(path)
    root = ET.parse(path).getroot()
    tracks: dict[int, set[int]] = {}
    for track in root.findall("track"):
        tid = int(track.attrib["id"])
        # Track span is taken from declared frame anchors, including outside
        # markers.  This is an annotation bookkeeping audit; visibility is not
        # being asserted here.
        frames = {int(box.attrib["frame"]) for box in track.findall("box")}
        tracks[tid] = frames
    return tracks


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    records = []
    source_files = []
    for seq in PAIRS:
        left_path = ROOT / "1" / f"{seq}-1.xml"
        right_path = ROOT / "2" / f"{seq}-2.xml"
        left, right = read_view(1, seq), read_view(2, seq)
        shared = sorted(set(left) & set(right))
        coframe = [tid for tid in shared if left[tid] & right[tid]]
        union = len(set(left) | set(right))
        records.append({
            "sequence": seq,
            "left_track_count": len(left),
            "right_track_count": len(right),
            "shared_raw_ids": len(shared),
            "coframe_shared_raw_ids": len(coframe),
            "raw_id_jaccard": round(len(shared) / union, 6) if union else 0.0,
            "coframe_rate_among_shared": round(len(coframe) / len(shared), 6) if shared else 0.0,
        })
        source_files.extend([str(left_path), str(right_path)])

    payload = {
        "status": "PASS_TRAIN_ONLY_ID_CONVENTION_AUDIT",
        "source_root": str(ROOT),
        "source_split": "train_only_inferred_from_MDMT_new_xml_path; official_val_test_not_read",
        "sequences": PAIRS,
        "records": records,
        "source_files": [
            {"path": p, "sha256": sha256(Path(p))} for p in sorted(source_files)
        ],
        "interpretation": {
            "raw_id_overlap_is_annotation_convention_not_physical_global_identity": True,
            "coframe_overlap_is_not_cross_view_correspondence_proof": True,
            "pose_calibration_timestamp_and_global_id_assumptions": "unsupported",
            "legal_supervision": "FIT/train grouped pairs only; no official val/test reads",
        },
    }
    out = Path(__file__).with_name("ID_AUDIT.json")
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"status": payload["status"], "pairs": len(records), "output": str(out)}, indent=2))


if __name__ == "__main__":
    main()
