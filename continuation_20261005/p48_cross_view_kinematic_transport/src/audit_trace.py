#!/usr/bin/env python3
"""Audit prefix kinematic evidence from the read-only native MIA trace."""
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
import cv2
import numpy as np

ROOT = Path('/home/chenhc/claude_try_MDMOT')
P48 = ROOT / 'continuation_20261005/p48_cross_view_kinematic_transport'
P26 = ROOT / 'continuation_20261005/p26_mia_baseline'
TRACE = P48 / 'runs/trace_pair23_v1/trace.jsonl'
OUT = P48 / 'runs/trace_pair23_v1/TRACE_AUDIT.json'

def center(r):
    return np.asarray([(float(r[1])+float(r[3]))/2, (float(r[2])+float(r[4]))/2], dtype=np.float64)

def scale(r):
    return max(float(np.hypot(float(r[3])-float(r[1]), float(r[4])-float(r[2]))), 1.0)

def iou(a, b):
    x1, y1 = max(float(a[1]), float(b[0])), max(float(a[2]), float(b[1]))
    x2, y2 = min(float(a[3]), float(b[2])), min(float(a[4]), float(b[3]))
    inter = max(0., x2-x1) * max(0., y2-y1)
    aa = max(0., float(a[3])-float(a[1])) * max(0., float(a[4])-float(a[2]))
    bb = max(0., float(b[2])-float(b[0])) * max(0., float(b[3])-float(b[1]))
    return inter / max(aa+bb-inter, 1e-12)

def xml_rows(path):
    out = {}
    for tr in ET.parse(path).getroot().findall('track'):
        tid = int(tr.attrib['id'])
        for box in tr.findall('box'):
            f = int(box.attrib['frame'])
            out.setdefault(f, []).append((tid, [float(box.attrib['xtl']), float(box.attrib['ytl']),
                                                float(box.attrib['xbr']), float(box.attrib['ybr'])]))
    return out

def row_gt(row, gt):
    best_iou, best_id = max(((iou(row, box), tid) for tid, box in gt), default=(0., None))
    return (best_id if best_iou >= .25 else None), best_iou

def project(H, p):
    return cv2.perspectiveTransform(np.asarray([[p]], dtype=np.float64),
                                    np.asarray(H, dtype=np.float64)).reshape(2)

def finite(H):
    return H is not None and np.isfinite(np.asarray(H)).all()

def summary(ranks, true, wrong):
    if not ranks:
        return {'queries': 0}
    return {'queries': len(ranks), 'rank1': float(np.mean(np.asarray(ranks)==1)),
            'rank_le_3': float(np.mean(np.asarray(ranks)<=3)),
            'median_rank': float(np.median(ranks)),
            'true_median_cost': float(np.median(true)),
            'wrong_median_cost': float(np.median(wrong)),
            'wrong_minus_true': float(np.median(wrong)-np.median(true))}

def direction(rows, src_key, dst_key, src_xml, dst_xml, stage):
    pre = {int(x['frame']): x for x in rows if x['stage'] == 'pre_mia'}
    stages = {int(x['frame']): x for x in rows if x['stage'] == stage}
    pos_ranks, kin_ranks, vel_ranks = [], [], []
    pos_true, pos_wrong, kin_true, kin_wrong, vel_true, vel_wrong = [], [], [], [], [], []
    dropped = Counter()
    for f in range(2, 700):
        if f not in stages or f-1 not in stages or f-1 not in pre:
            dropped['missing_frame'] += 1
            continue
        H = stages[f]['H']
        H_prev = stages[f-1]['H']
        if not finite(H) or not finite(H_prev):
            dropped['invalid_H'] += 1
            continue
        src, dst, old_src, old_dst = stages[f][src_key], stages[f][dst_key], pre[f-1][src_key], pre[f-1][dst_key]
        old_src_by_id = {int(round(float(r[0]))): r for r in old_src}
        old_dst_by_id = {int(round(float(r[0]))): r for r in old_dst}
        dst_gt = [(r, row_gt(r, dst_xml[f])[0]) for r in dst]
        for sr in src:
            sid = int(round(float(sr[0])))
            if sid not in old_src_by_id:
                dropped['source_no_history'] += 1
                continue
            gid, _ = row_gt(sr, src_xml[f])
            if gid is None:
                dropped['source_no_gt'] += 1
                continue
            try:
                p = project(H, center(sr))
                pp = project(H_prev, center(old_src_by_id[sid]))
            except Exception:
                dropped['projection_error'] += 1
                continue
            if not np.isfinite(p).all() or not np.isfinite(pp).all():
                dropped['projection_nonfinite'] += 1
                continue
            pv = p - pp
            pos, kin, vel = [], [], []
            for dr, did in dst_gt:
                if did is None:
                    continue
                pc = float(np.linalg.norm(p-center(dr))/scale(dr))
                did_local = int(round(float(dr[0])))
                if did_local in old_dst_by_id:
                    qv = center(dr)-center(old_dst_by_id[did_local])
                    vc = float(np.linalg.norm(pv-qv)/scale(dr))
                else:
                    vc = 2.0
                pos.append((pc, did))
                kin.append((.65*pc+.35*vc, did))
                vel.append((vc, did))
            if not pos or gid not in [x[1] for x in pos]:
                dropped['target_unusable'] += 1
                continue
            pos.sort(); kin.sort(); vel.sort()
            pos_ids, kin_ids, vel_ids = [x[1] for x in pos], [x[1] for x in kin], [x[1] for x in vel]
            pr, kr, vr = 1+pos_ids.index(gid), 1+kin_ids.index(gid), 1+vel_ids.index(gid)
            pos_ranks.append(pr); kin_ranks.append(kr); vel_ranks.append(vr)
            pos_true.append(next(x[0] for x in pos if x[1] == gid))
            pos_wrong.extend(x[0] for x in pos if x[1] != gid)
            kin_true.append(next(x[0] for x in kin if x[1] == gid))
            kin_wrong.extend(x[0] for x in kin if x[1] != gid)
            vel_true.append(next(x[0] for x in vel if x[1] == gid))
            vel_wrong.extend(x[0] for x in vel if x[1] != gid)
    hard = [i for i, x in enumerate(pos_ranks) if x > 1]
    return {'stage': stage, 'position': summary(pos_ranks,pos_true,pos_wrong),
            'position_plus_kinematic': summary(kin_ranks,kin_true,kin_wrong),
            'kinematic_only': summary(vel_ranks,vel_true,vel_wrong),
            'rank1_delta': float(np.mean(np.asarray(kin_ranks)==1)-np.mean(np.asarray(pos_ranks)==1)) if pos_ranks else None,
            'hard_geometry_queries': len(hard),
            'hard_geometry_kinematic_rank1': float(np.mean(np.asarray([kin_ranks[i] for i in hard])==1)) if hard else None,
            'hard_geometry_position_rank1': float(np.mean(np.asarray([pos_ranks[i] for i in hard])==1)) if hard else None,
            'rank_changes': int(sum(a != b for a,b in zip(pos_ranks,kin_ranks))),
            'dropped': dict(dropped)}

def main():
    rows = [json.loads(x) for x in TRACE.open()]
    out = {'trace_lines': len(rows), 'stage_counts': dict(Counter(x['stage'] for x in rows))}
    out['A_to_B'] = direction(rows, 'view1', 'view2', xml_rows(P26/'xml/23-1.xml'),
                               xml_rows(P26/'xml/23-2.xml'), 'before_A_new')
    out['B_to_A'] = direction(rows, 'view2', 'view1', xml_rows(P26/'xml/23-2.xml'),
                               xml_rows(P26/'xml/23-1.xml'), 'before_B_new')
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True)+'\n')
    print(json.dumps(out, indent=2, sort_keys=True))

if __name__ == '__main__':
    main()
