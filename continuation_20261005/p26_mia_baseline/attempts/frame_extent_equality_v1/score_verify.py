#!/usr/bin/env python3
"""Independently re-score frozen MIA reference predictions; no inference/GT XML.

Copies exactly the sealed 8+6 test-pair JSON outputs and the existing evaluation
GT TXT snapshots into this stage. Runs unchanged official calAAS and motmetrics,
checks an independent vectorized calAAS transcription, and binds every input.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
import sys
sys.dont_write_bytecode = True
import contextlib
import hashlib
import importlib.util
import io
import json
import shutil
import statistics
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import motmetrics as mm

HERE = Path(__file__).resolve().parent
REF = Path('/home/chenhc/mdmot_p24_test_20261004/official_demo')
SRC = Path('/home/chenhc/mdmot_strong_host_stage1_20260905/source_compat')
SEAL = REF / 'SEAL_mia_net_full_20261005.json'
METHOD_PAIRS = {'official_suppl_test': ['56','57','59','61','62','68','71','73'],
                'official_suppl_test6': ['26','31','34','48','52','55']}
PAIRS = sorted(sum(METHOD_PAIRS.values(), []), key=int)
SNAP = HERE / 'reference_outputs'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + '\n')


def snapshot():
    if SNAP.exists():
        raise FileExistsError('Reference snapshot already exists; use --rescore to verify/reuse it')
    seal = json.loads(SEAL.read_text())
    seal_records = {}
    mismatches = []
    for rec in seal['files']:
        p = Path(rec['file'])
        if not p.is_absolute():
            p = REF / p
        actual = sha(p) if p.is_file() else None
        seal_records[str(p)] = {'expected_sha256': rec['sha256'], 'actual_sha256': actual,
                                'match': actual == rec['sha256']}
        if actual != rec['sha256']:
            mismatches.append(str(p))
    dump(HERE/'REFERENCE_SEAL_CHECK.json', {'seal': str(SEAL), 'seal_sha256': sha(SEAL),
        'files': seal_records, 'checked': len(seal_records), 'mismatches': mismatches})
    SNAP.mkdir()
    deps = {}
    files = []

    def copy(p, dest, kind, seal_required=False):
        before = sha(p)
        if seal_required and (str(p) not in seal_records or not seal_records[str(p)]['match']):
            raise ValueError('Selected prediction does not match its original seal: ' + str(p))
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
        if sha(p) != before or sha(dest) != before:
            raise ValueError('Snapshot source changed during copy: ' + str(p))
        deps[str(dest)] = before
        files.append({'kind': kind, 'source': str(p), 'copy': str(dest), 'sha256': before,
                      'bytes': dest.stat().st_size, 'original_seal_bound': seal_required})

    original_scores = {}
    for method, pairs in METHOD_PAIRS.items():
        pred_files = {p.name for p in (REF/'json'/method).glob('*.json')}
        expected_names = {f'{p}-{v}.json' for p in pairs for v in (1,2)}
        if pred_files != expected_names:
            raise ValueError('Reference method JSON set is incomplete or has unexpected files: ' + method)
        summary = REF/'score'/method/'summary.json'
        original_scores.update({r['pair_id']: r for r in json.loads(summary.read_text())['rows']})
        copy(summary, SNAP/'metadata'/f'{method}_summary.json', 'previous_summary')
        for pair in pairs:
            for v in (1,2):
                copy(REF/'json'/method/f'{pair}-{v}.json', SNAP/'predictions'/f'{pair}-{v}.json', 'prediction_json', True)
                copy(REF/'score'/method/f'gt-{pair}-{v}.txt', SNAP/'gt'/f'{pair}-{v}.txt', 'existing_GT_TXT_snapshot')
                copy(REF/'score'/method/f'{pair}-{v}.txt', SNAP/'reference_mot'/f'{pair}-{v}.txt', 'previous_prediction_MOT_conversion')
    for p, name, kind in [
        (SRC/'demo/eval/mango_eval.py', 'mango_eval.py', 'official_evaluator'),
        (SRC/'README.md', 'README.md', 'repository_primary_source'),
        (Path('/home/chenhc/mdmot_research_20261002/sources/direct_stca_pdf.txt'), 'STCA_primary_text.txt', 'published_comparison_primary_source'),
        (REF.parent/'scripts/score_official_demo.py', 'reference_score_wrapper.py', 'previous_evaluation_wrapper'),
        (Path('/home/chenhc/mdmot_global_host_stage15_20260905/scripts/stage15_global_host.py'), 'reference_stage15_helper.py', 'previous_GT_conversion_helper'),
        (SEAL, 'SEAL_mia_net_full_20261005.json', 'original_reference_seal')]:
        copy(p, SNAP/'metadata'/name, kind)
    receipt = {'status': 'FROZEN_REFERENCE_SCORE_INPUTS', 'pairs': PAIRS, 'pair_count': 14,
        'method_pairs': METHOD_PAIRS, 'dependencies': deps, 'files': files,
        'original_scores_by_pair': original_scores, 'original_seal_mismatches': mismatches,
        'GT_scope': 'existing evaluation TXT snapshots only; no raw XML/image reads or new inference',
        'reference_prediction_JSON_count': 28, 'existing_GT_TXT_count': 28,
        'snapshot_gt_previously_sealed': False,
        'script_sha256': sha(__file__)}
    dump(HERE/'INDEPENDENT_SCORE_INPUTS.json', receipt)
    return receipt


def verify_snapshot():
    rec = json.loads((HERE/'INDEPENDENT_SCORE_INPUTS.json').read_text())
    for path, digest in rec['dependencies'].items():
        if sha(path) != digest:
            raise ValueError('Frozen scoring snapshot changed: ' + path)
    return rec


def load_mango():
    p = SNAP/'metadata/mango_eval.py'
    spec = importlib.util.spec_from_file_location('p26_unchanged_official_mango', p)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def iou_matrix(a, b):
    a = np.asarray(a, dtype=float).reshape(-1,4)
    b = np.asarray(b, dtype=float).reshape(-1,4)
    wh = np.maximum(np.minimum(a[:,None,2:], b[None,:,2:])-np.maximum(a[:,None,:2], b[None,:,:2]), 0)
    inter = wh.prod(axis=2)
    area_a = (a[:,2:]-a[:,:2]).prod(axis=1)
    area_b = (b[:,2:]-b[:,:2]).prod(axis=1)
    denom = area_a[:,None] + area_b[None,:] - inter
    return np.divide(inter, denom, out=np.zeros_like(inter), where=denom > 0)


def vectorized_mda(frames):
    """Preserves official duplicate-ID/last-match and unmatched-GT[-1] semantics."""
    ratios = []
    totals = {'GA': 0, 'RA': 0, 'TA': 0, 'FA': 0, 'MA': 0,
              'unshared_GT_A_candidates_with_A_IoU05': 0, 'unshared_GT_A_TA_via_negative_index': 0,
              'negative_FA_frames': 0, 'negative_MA_frames': 0}
    for fr in frames:
        a, b, ga, gb = fr.resultA, fr.resultB, fr.gtA, fr.gtB
        aid = np.array([x.ID for x in a]); bid = np.array([x.ID for x in b])
        gaid = np.array([x.ID for x in ga]); gbid = np.array([x.ID for x in gb])
        RA = int(np.count_nonzero(aid[:,None] == bid[None,:]))
        GA = int(np.count_nonzero(gaid[:,None] == gbid[None,:]))
        b_map = {x.ID: i for i,x in enumerate(b)}
        gb_map = {x.ID: i for i,x in enumerate(gb)}
        source_ids = [i for i,x in enumerate(a) if x.ID in b_map]
        ab = lambda x: [x.x1,x.y1,x.x2,x.y2]
        simA = iou_matrix([ab(a[i]) for i in source_ids], [ab(x) for x in ga])
        if source_ids and ga and not gb:
            raise ValueError('Official negative indexing would be undefined for empty GT-B')
        target_bs = [ab(b[b_map[a[i].ID]]) for i in source_ids]
        gt_bs = [ab(gb[gb_map.get(x.ID, -1)]) for x in ga] if gb else []
        simB = iou_matrix(target_bs, gt_bs)
        accepted_a = simA >= .5
        accepted = accepted_a & (simB >= .5)
        TA = int(accepted.sum())
        FA, MA = RA-TA, GA-TA
        denom = GA + FA + MA
        if denom == 0:
            raise ValueError('Official MDA would divide by zero')
        ratios.append(TA / denom)
        missing = np.array([x.ID not in gb_map for x in ga], dtype=bool)
        totals['unshared_GT_A_candidates_with_A_IoU05'] += int(accepted_a[:,missing].sum())
        totals['unshared_GT_A_TA_via_negative_index'] += int(accepted[:,missing].sum())
        for name,value in [('GA',GA),('RA',RA),('TA',TA),('FA',FA),('MA',MA)]:
            totals[name] += value
        totals['negative_FA_frames'] += int(FA < 0)
        totals['negative_MA_frames'] += int(MA < 0)
    return statistics.mean(ratios), totals


def score_pair(pair):
    started = time.monotonic()
    mango = load_mango()
    structural = {}
    jsons = {}
    mot_paths = {}
    scored_dir = HERE/'independent_score'
    scored_dir.mkdir(exist_ok=True)
    for v in (1,2):
        path = SNAP/'predictions'/f'{pair}-{v}.json'
        js = json.loads(path.read_text())
        expected = [f'frame={i}' for i in range(len(js))]
        if list(js) != expected:
            raise ValueError('Official evaluator enumerates insertion order; noncontiguous keys: ' + str(path))
        jsons[v] = js
        bad_boxes = duplicate_ids = observations = 0
        pred = scored_dir/f'{pair}-{v}.txt'
        with pred.open('w') as f:
            for k, rs in js.items():
                frame = int(k.split('=')[1]) + 1
                ids = []
                for r in rs:
                    if len(r) < 5 or not np.isfinite(np.asarray(r[:5],float)).all():
                        raise ValueError('Malformed prediction row')
                    gid = int(float(r[0])); ids.append(gid)
                    x1,y1,x2,y2 = r[1:5]
                    bad_boxes += int(float(x2)<=float(x1) or float(y2)<=float(y1))
                    f.write(f'{frame},{gid},{x1},{y1},{float(x2)-float(x1)},{float(y2)-float(y1)},1,1,1\n')
                observations += len(rs)
                duplicate_ids += len(ids)-len(set(ids))
        mot_paths[v] = pred
        gt_rows = np.loadtxt(SNAP/'gt'/f'{pair}-{v}.txt', delimiter=',', ndmin=2)
        if not len(gt_rows) or int(gt_rows[:,0].max()) != len(js) or int(gt_rows[:,0].min()) < 1:
            raise ValueError('Prediction frame extent differs from existing GT snapshot')
        if len(np.unique(gt_rows[:,:2],axis=0)) != len(gt_rows):
            raise ValueError('GT frame/ID is duplicated')
        structural[str(v)] = {'frames':len(js), 'prediction_rows': observations, 'GT_rows':len(gt_rows),
            'GT_frame_min':int(gt_rows[:,0].min()), 'GT_frame_max':int(gt_rows[:,0].max()),
            'GT_frames_without_annotation':len(js)-len(np.unique(gt_rows[:,0])),
            'duplicate_prediction_ids_per_frame':duplicate_ids, 'nonpositive_prediction_boxes':bad_boxes,
            'MOT_conversion_matches_previous_bytes':sha(pred)==sha(SNAP/'reference_mot'/f'{pair}-{v}.txt')}
    if list(jsons[1]) != list(jsons[2]):
        raise ValueError('Views do not cover identical frames')
    frames = mango.read_result(str(SNAP/'predictions'/f'{pair}-1.json'), str(SNAP/'predictions'/f'{pair}-2.json'))
    g1 = mango.read_gt(str(SNAP/'gt'/f'{pair}-1.txt'), 1)
    g2 = mango.read_gt(str(SNAP/'gt'/f'{pair}-2.txt'), 2)
    mango.assign_gt_to_frames(frames, g1, g2)
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        official = float(mango.calAAS(frames))
    independent, mda_counts = vectorized_mda(frames)
    if abs(official-independent) > 1e-12:
        raise AssertionError('Independent MDA differs from unmodified official evaluator')
    keys = ['idf1','idp','idr','mota','motp','num_switches','idtp','idfp','idfn',
            'num_objects','num_predictions','num_false_positives','num_misses']
    views = {}
    for v in (1,2):
        g = mm.io.loadtxt(str(SNAP/'gt'/f'{pair}-{v}.txt'), fmt='mot15-2D', min_confidence=1)
        t = mm.io.loadtxt(str(mot_paths[v]), fmt='mot15-2D')
        acc = mm.utils.compare_to_groundtruth(g,t,'iou',distth=.5)
        vals = mm.metrics.create().compute(acc, metrics=keys).iloc[0].to_dict()
        views[str(v)] = {k:float(vals[k]) for k in keys}
    result = {'pair_id':pair, 'MDA':official, 'independent_vectorized_MDA':independent,
        'MDA_counts_pooled_diagnostic_only':mda_counts,
        **{k:statistics.mean(r[k] for r in views.values()) for k in ('idf1','idp','idr','mota','motp')},
        'num_switches':sum(r['num_switches'] for r in views.values()), 'views':views,
        'structure':structural, 'official_stdout':capture.getvalue(), 'elapsed_seconds':time.monotonic()-started}
    dump(scored_dir/f'{pair}_metrics.json', result)
    return result


def main():
    t0 = time.monotonic()
    receipt = verify_snapshot() if '--rescore' in sys.argv else snapshot()
    rows = []
    with ProcessPoolExecutor(max_workers=4) as pool:
        pending = {pool.submit(score_pair,p):p for p in PAIRS}
        for future in as_completed(pending):
            row = future.result(); rows.append(row)
            print(json.dumps({'pair':row['pair_id'],'MDA':row['MDA'],'idf1':row['idf1'],
                              'mota':row['mota'],'seconds':row['elapsed_seconds']}),flush=True)
    rows.sort(key=lambda r:int(r['pair_id']))
    macro = {k:statistics.mean(r[k] for r in rows) for k in ('MDA','idf1','idp','idr','mota')}
    deltas = {r['pair_id']:{k:r[k]-receipt['original_scores_by_pair'][r['pair_id']][k]
                           for k in ('MDA','idf1','idp','idr','mota','motp','num_switches')} for r in rows}
    maximum_delta = max(abs(v) for rec in deltas.values() for v in rec.values())
    verify_snapshot()
    meta = {'status':'COMPLETE_INDEPENDENT_REFERENCE_RESCORE', 'pair_count':len(rows), 'pairs':PAIRS,
        'macro':macro, 'macro_percent':{k:100*v for k,v in macro.items()},
        'units':'raw values are fractions; percent=100*value; not percentage-point fractions',
        'aggregation':'MDA equal mean of per-frame ratios within pair, then equal mean over14 pairs; MOT equal mean over2 views within pair then14pairs',
        'not_aggregation':'not pooled-count MDA, not unweighted average of test8 and test6 summaries, not global-IDF1 across all sequences',
        'published_STCA_Table1_MIA':{'MDA_percent':41.72,'idf1_percent':68.24,'mota_percent':49.68},
        'delta_from_published_percentage_points':{'MDA':100*macro['MDA']-41.72,
                    'idf1':100*macro['idf1']-68.24,'mota':100*macro['mota']-49.68},
        'previous_summary_max_abs_difference':maximum_delta, 'previous_summary_differences':deltas,
        'official_and_independent_MDA_max_abs_difference':max(abs(r['MDA']-r['independent_vectorized_MDA']) for r in rows),
        'rows':rows, 'input_manifest_sha256':sha(HERE/'INDEPENDENT_SCORE_INPUTS.json'),
        'script_sha256':sha(__file__), 'python':sys.version, 'numpy':np.__version__, 'motmetrics':mm.__version__,
        'motmetrics_default_solver':mm.lap.default_solver, 'elapsed_seconds':time.monotonic()-t0,
        'new_inference':False,'training':False,'GPU_used':False,'raw_XML_or_image_read':False,
        'existing_test_evaluation_snapshots_rescored':True}
    dump(HERE/'INDEPENDENT_SCORES.json', meta)
    print(json.dumps({k:meta[k] for k in ('status','macro_percent','previous_summary_max_abs_difference',
                     'delta_from_published_percentage_points','elapsed_seconds')},indent=2),flush=True)


if __name__ == '__main__':
    main()
