#!/usr/bin/env python3
"""CPU-only diagnostic for cross-view kinematic transport.

This uses XML identities only to measure separability. It is a diagnostic, not
an official MOT score and not a training authorization.
"""

import json
import math
import os
import statistics
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np


ROOT = Path('/home/chenhc/claude_try_MDMOT')
XML = ROOT / 'continuation_20261005/p26_mia_baseline/xml'
OUT = ROOT / 'continuation_20261005/p48_cross_view_kinematic_transport/runs/pair23'


def read_xml(path):
    root = ET.parse(path).getroot()
    out = {}
    for tr in root.findall('track'):
        tid = int(tr.attrib['id'])
        for box in tr.findall('box'):
            f = int(box.attrib['frame'])
            out.setdefault(f, {})[tid] = np.array([
                float(box.attrib['xtl']), float(box.attrib['ytl']),
                float(box.attrib['xbr']), float(box.attrib['ybr'])], dtype=np.float64)
    return out


def center(b):
    return np.array([(b[0] + b[2]) * 0.5, (b[1] + b[3]) * 0.5], dtype=np.float64)


def diag(b):
    return max(float(np.hypot(b[2] - b[0], b[3] - b[1])), 1.0)


def project(H, p):
    x = np.asarray([[[float(p[0]), float(p[1])]]], dtype=np.float64)
    y = cv2.perspectiveTransform(x, H).reshape(2)
    return y


def finite(v):
    return np.isfinite(v).all()


def main():
    a, b = read_xml(XML / '23-1.xml'), read_xml(XML / '23-2.xml')
    common0 = sorted(set(a[0]) & set(b[0]))
    src = np.asarray([center(a[0][i]) for i in common0], np.float32)
    dst = np.asarray([center(b[0][i]) for i in common0], np.float32)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None or not finite(H):
        raise RuntimeError('could not estimate frame-zero homography')

    records = []
    pos_ranks, kin_ranks = [], []
    pos_true, pos_wrong, kin_true, kin_wrong = [], [], [], []
    for f in range(2, 700):
        H_use = H
        # Diagnostic upper bound only: estimate a fresh H from XML identities at
        # the current frame. This mode is never legal for training/inference.
        if os.environ.get('P48_DYNAMIC_H') == '1':
            common_f = sorted(set(a.get(f, {})) & set(b.get(f, {})))
            if len(common_f) >= 4:
                hs, hd = (np.asarray([center(a[f][i]) for i in common_f], np.float32),
                          np.asarray([center(b[f][i]) for i in common_f], np.float32))
                H_candidate, _ = cv2.findHomography(hs, hd, cv2.RANSAC, 5.0)
                if H_candidate is not None and finite(H_candidate):
                    H_use = H_candidate
        ids = sorted(set(a.get(f, {})) & set(b.get(f, {})))
        for sid in ids:
            # Strictly past source and target states; no future or label is an input.
            if sid not in a.get(f - 1, {}) or sid not in b.get(f - 1, {}):
                continue
            p_now = center(a[f][sid]); p_prev = center(a[f - 1][sid])
            q_now = center(b[f][sid]); q_prev = center(b[f - 1][sid])
            p_proj = project(H_use, p_now)
            p_prev_proj = project(H_use, p_prev)
            if not finite(p_proj) or not finite(p_prev_proj):
                continue
            pred_qv = p_proj - p_prev_proj
            true_qv = q_now - q_prev
            pos_costs, kin_costs = [], []
            cand_ids = sorted(b.get(f, {}))
            for tid in cand_ids:
                q = center(b[f][tid])
                pcost = float(np.linalg.norm(p_proj - q) / diag(b[f][tid]))
                if tid in b.get(f - 1, {}):
                    qv = q - center(b[f - 1][tid])
                    vcost = float(np.linalg.norm(pred_qv - qv) / diag(b[f][tid]))
                else:
                    vcost = 2.0
                # Fixed diagnostic combination; no tuning or official selection.
                kcost = 0.65 * pcost + 0.35 * vcost
                pos_costs.append((pcost, tid)); kin_costs.append((kcost, tid))
            pos_costs.sort(); kin_costs.sort()
            if sid not in [x[1] for x in pos_costs] or sid not in [x[1] for x in kin_costs]:
                continue
            pr = 1 + [x[1] for x in pos_costs].index(sid)
            kr = 1 + [x[1] for x in kin_costs].index(sid)
            pos_ranks.append(pr); kin_ranks.append(kr)
            for val, tid in pos_costs:
                (pos_true if tid == sid else pos_wrong).append(val)
            for val, tid in kin_costs:
                (kin_true if tid == sid else kin_wrong).append(val)

    def summary(ranks, true, wrong):
        return {
            'queries': len(ranks), 'rank1': float(np.mean(np.asarray(ranks) == 1)),
            'rank_le_3': float(np.mean(np.asarray(ranks) <= 3)),
            'median_rank': float(np.median(ranks)),
            'true_median_cost': float(np.median(true)),
            'wrong_median_cost': float(np.median(wrong)),
            'median_wrong_minus_true': float(np.median(wrong) - np.median(true)),
        }

    result = {
        'status': 'PASS_P48_CPU_KINEMATIC_DIAGNOSTIC',
        'scope': 'fit pair23 XML diagnostic; XML labels used only for measurement',
        'homography_mode': 'current-frame XML oracle' if os.environ.get('P48_DYNAMIC_H') == '1' else 'frame-zero estimate',
        'input_contract': 'strict prefix positions; no timestamps; no future frames; no official score',
        'homography_inliers_frame0': int(mask.sum()) if mask is not None else None,
        'position': summary(pos_ranks, pos_true, pos_wrong),
        'position_plus_kinematics': summary(kin_ranks, kin_true, kin_wrong),
        'rank1_delta': float(np.mean(np.asarray(kin_ranks) == 1) - np.mean(np.asarray(pos_ranks) == 1)),
        'queries_with_rank_change': int(sum(a != b for a, b in zip(pos_ranks, kin_ranks))),
        'no_training': True,
        'no_official_val_test': True,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'RESULT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
