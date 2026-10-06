#!/usr/bin/env python3
"""Implementation and output audit for the P47 front-door replay."""

import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path('/home/chenhc/claude_try_MDMOT')
P26 = ROOT / 'continuation_20261005/p26_mia_baseline/runs/fit23/json/mia_baseline'
P47 = ROOT / 'continuation_20261005/p47_true_mia_p14_frontdoor'


def load(path):
    return json.loads(Path(path).read_text())


def audit_json(path):
    data = load(path)
    keys = list(data)
    expected = [f'frame={i}' for i in range(700)]
    assert keys == expected, (path, keys[:2], keys[-2:], len(keys))
    duplicates = []
    bad = []
    rows = 0
    for frame, entries in data.items():
        ids = [int(round(float(r[0]))) for r in entries]
        counts = Counter(ids)
        duplicates.extend((int(frame.split('=')[1]), i, n) for i, n in counts.items() if n > 1)
        for r in entries:
            rows += 1
            if len(r) < 5 or not all(math.isfinite(float(x)) for x in r[:5]):
                bad.append((frame, r))
            elif float(r[3]) <= float(r[1]) or float(r[4]) <= float(r[2]):
                bad.append((frame, r))
    return {'frames': len(data), 'rows': rows, 'duplicate_ids': duplicates,
            'bad_rows': bad, 'row_count_min': min(map(len, data.values())),
            'row_count_max': max(map(len, data.values()))}


def audit_receipt(path):
    r = load(path)
    align = r['descriptor_alignment']
    assert len(align) == 700, len(align)
    alignment = {}
    for view in ('view1', 'view2'):
        vals = [x[view] for x in align]
        alignment[view] = {
            'frames': len(vals),
            'coverage_min': min(float(x['coverage']) for x in vals),
            'coverage_mean': statistics.mean(float(x['coverage']) for x in vals),
            'coverage_lt_1': sum(float(x['coverage']) < 1.0 for x in vals),
            'coverage_lt_0_99': sum(float(x['coverage']) < 0.99 for x in vals),
            'mean_iou_min': min(float(x['mean_iou']) for x in vals if x['mean_iou'] is not None),
            'min_iou_min': min(float(x['min_iou']) for x in vals if x['min_iou'] is not None),
        }
    event_stats = {}
    for stage in ('A_new', 'B_new', 'A_old'):
        evs = [x for x in r['events'] if x['stage'] == stage]
        props = [p for x in evs for p in x['proposals']]
        acc = [p for x in evs for p in x['accepted']]
        event_stats[stage] = {
            'frames': len(evs), 'proposal_count': len(props), 'accepted_count': len(acc),
            'skipped_count': sum(int(x['skipped']) for x in evs),
            'accepted_unique_target_indices': len({(x['frame'], p['target_index']) for x in evs for p in x['accepted']}),
            'accepted_unique_new_ids': len({p['new_id'] for p in acc}),
            'cost_min': min((float(p['cost']) for p in props), default=None),
            'cost_max': max((float(p['cost']) for p in props), default=None),
        }
    return {'status': r.get('status'), 'mode': r.get('mode'), 'sequences': r.get('sequences'),
            'labels_read': r.get('labels_read'), 'official_test_read': r.get('official_test_read'),
            'alignment': alignment, 'events': event_stats,
            'event_total_proposals': sum(x['proposal_count'] for x in event_stats.values()),
            'event_total_accepted': sum(x['accepted_count'] for x in event_stats.values()),
            'event_total_skipped': sum(x['skipped_count'] for x in event_stats.values())}


def compare_ids(p26_path, p47_path):
    a, b = load(p26_path), load(p47_path)
    changed = []
    for frame in a:
        ra, rb = a[frame], b[frame]
        if len(ra) != len(rb):
            changed.append({'frame': frame, 'kind': 'row_count', 'p26': len(ra), 'p47': len(rb)})
            continue
        # Rows preserve detector/tracker order in this host. Compare geometry and IDs separately.
        geom_changed = 0
        id_changed = 0
        for x, y in zip(ra, rb):
            if any(abs(float(x[k]) - float(y[k])) > 1e-5 for k in range(1, 5)):
                geom_changed += 1
            if int(round(float(x[0]))) != int(round(float(y[0]))):
                id_changed += 1
        if geom_changed or id_changed:
            changed.append({'frame': frame, 'kind': 'rows', 'geometry_changed': geom_changed,
                            'id_changed': id_changed, 'p26_rows': len(ra), 'p47_rows': len(rb)})
    return {'frames_with_any_change': len(changed),
            'frames_with_geometry_change': sum(x.get('geometry_changed', 0) > 0 for x in changed),
            'frames_with_id_change': sum(x.get('id_changed', 0) > 0 for x in changed),
            'rows_geometry_changed': sum(x.get('geometry_changed', 0) for x in changed),
            'rows_id_changed': sum(x.get('id_changed', 0) for x in changed),
            'row_count_change_frames': sum(x['kind'] == 'row_count' for x in changed),
            'first_changes': changed[:20]}


def main():
    variant_dir = Path(__import__('os').environ.get('P47_VARIANT_DIR', str(P47 / 'runs/true_mia_p14_v3')))
    method = __import__('os').environ.get('P47_VARIANT_METHOD', 'p14_frontdoor')
    pred = variant_dir / 'json' / method
    receipt = pred / 'P47_RECEIPT.json'
    result = {
        'variant_dir': str(variant_dir), 'method': method,
        'view1': audit_json(pred / '23-1.json'), 'view2': audit_json(pred / '23-2.json'),
        'receipt': audit_receipt(receipt),
        'vs_p26': compare_ids(P26 / '23-1.json', pred / '23-1.json'),
    }
    out = variant_dir / 'AUDIT.json'
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
