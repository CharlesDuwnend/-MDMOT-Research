#!/usr/bin/env python3
"""Official MDA + motmetrics score for P47 and its frozen P26 output."""

import importlib.util
import json
import os
import statistics
from pathlib import Path

import motmetrics as mm

ROOT = Path('/home/chenhc/claude_try_MDMOT')
P26 = ROOT / 'continuation_20261005/p26_mia_baseline'
P47 = ROOT / 'continuation_20261005/p47_true_mia_p14_frontdoor'
VARIANT_DIR = Path(os.environ.get('P47_VARIANT_DIR', str(P47 / 'runs/true_mia_p14_v3')))
VARIANT_TAG = os.environ.get('P47_VARIANT_TAG', 'P47_true_mia_p14')
VARIANT_METHOD = os.environ.get('P47_VARIANT_METHOD', 'p14_frontdoor')
OUT = Path(os.environ.get('P47_SCORE_DIR', str(VARIANT_DIR / 'score')))
P47_JSON = VARIANT_DIR / 'json' / VARIANT_METHOD
P26_JSON = P26 / 'runs/fit23/json/mia_baseline'
STAGE15 = Path('/home/chenhc/mdmot_global_host_stage15_20260905/scripts/stage15_global_host.py')


def load_stage15():
    spec = importlib.util.spec_from_file_location('p47_stage15_helper', STAGE15)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def mot_rows(mod, pred_dir, tag):
    paths = {}
    for v in (1, 2):
        gt = OUT / f'gt-23-{v}.txt'
        if not gt.exists():
            mod.xml_to_mot('23', str(v), gt)
        pred = pred_dir / f'23-{v}.json'
        txt = OUT / f'{tag}-23-{v}.txt'
        js = json.loads(pred.read_text())
        with txt.open('w') as f:
            for key, rows in js.items():
                frame = int(key.split('=')[1]) + 1
                for r in rows:
                    gid, x1, y1, x2, y2 = r[:5]
                    f.write(f'{frame},{int(float(gid))},{x1},{y1},{float(x2)-float(x1)},{float(y2)-float(y1)},1,1,1\n')
        g = mm.io.loadtxt(str(gt), fmt='mot15-2D', min_confidence=1)
        t = mm.io.loadtxt(str(txt), fmt='mot15-2D')
        acc = mm.utils.compare_to_groundtruth(g, t, 'iou', distth=0.5)
        keys = ['idf1', 'idp', 'idr', 'mota', 'motp', 'num_switches', 'num_false_positives', 'num_misses']
        vals = mm.metrics.create().compute(acc, metrics=keys).iloc[0].to_dict()
        paths[str(v)] = {'txt': str(txt), 'metrics': {k: float(vals[k]) for k in keys}}
    return paths


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    mod = load_stage15()
    mango, _ = mod.load_official_mda()
    rows = {}
    for tag, pred in [('P26_baseline', P26_JSON), (VARIANT_TAG, P47_JSON)]:
        gt1 = OUT / 'gt-23-1.txt'; gt2 = OUT / 'gt-23-2.txt'
        if not gt1.exists(): mod.xml_to_mot('23', '1', gt1)
        if not gt2.exists(): mod.xml_to_mot('23', '2', gt2)
        mda, stdout = mod.official_mda(mango, pred / '23-1.json', pred / '23-2.json', gt1, gt2)
        views = mot_rows(mod, pred, tag)
        rows[tag] = {
            'MDA': float(mda),
            'IDF1': float(statistics.mean(views[v]['metrics']['idf1'] for v in ('1', '2'))),
            'MOTA': float(statistics.mean(views[v]['metrics']['mota'] for v in ('1', '2'))),
            'switches': int(sum(views[v]['metrics']['num_switches'] for v in ('1', '2'))),
            'views': views,
            'official_stdout': stdout,
        }
    delta = {k: rows[VARIANT_TAG][k] - rows['P26_baseline'][k] for k in ('MDA', 'IDF1', 'MOTA', 'switches')}
    result = {'status': 'COMPLETE_P47_PAIR23_SCORE', 'variant_tag': VARIANT_TAG, 'rows': rows, 'delta_variant_minus_P26': delta,
              'gt_scope': 'fit pair23 XML only', 'official_test_read': False}
    (OUT / 'SCORES.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
