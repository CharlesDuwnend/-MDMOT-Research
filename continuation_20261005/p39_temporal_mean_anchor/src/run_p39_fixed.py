#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json, sys
from collections import deque
from pathlib import Path
import numpy as np

ROOT = Path('/home/chenhc/mdmot_research_20261002')
WORK = Path('/home/chenhc/claude_try_MDMOT/continuation_20261005/p39_temporal_mean_anchor')
P4 = ROOT / 'p4_diagnosis'
sys.path.insert(0, str(ROOT / 'p2/src'))
sys.path.insert(0, str(P4))
from data import PairData, sha256
import compare_crop32 as cmp

FIT = ('23','25','28','29','39','44','45','51','53','63','66','69','70','74','78')
CAL = ('27','32','42','64','65')
WINDOW = 8
MAX_GAP = 30

def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n')

def unit(x):
    x = np.asarray(x, np.float32)
    n = np.linalg.norm(x)
    return x / n if n > 1e-12 else x

def temporal_mean_outputs(pair, selected_frames):
    """Replay both streams causally over all source frames, then return selected crop order."""
    d = PairData(pair, with_labels=False)
    by_key = {}
    reports = []
    for view, (arr, offsets) in enumerate(zip(d._views, d._offsets), start=1):
        out = np.zeros_like(arr['embedding'], dtype=np.float32)
        memory = {}
        counts = {'detector_rows': int(len(arr['frame'])), 'feature_present': 0,
                  'history_used': 0, 'missing': 0, 'untracked': 0,
                  'class_resets': 0, 'gap_resets': 0}
        for i in range(len(arr['frame'])):
            frame = int(arr['frame'][i]); local = int(arr['local_id'][i]); cls = int(arr['cls'][i])
            present = bool(arr['feature_present'][i])
            if present: counts['feature_present'] += 1
            else: counts['missing'] += 1
            if local < 0:
                counts['untracked'] += 1
                if present: out[i] = unit(arr['embedding'][i])
            else:
                old = memory.get(local)
                if old is not None:
                    if frame <= old['frame']:
                        raise ValueError(f'noncausal/duplicate row pair={pair} view={view} index={i}')
                    if cls != old['class']:
                        counts['class_resets'] += 1; old = None
                    elif frame - old['frame'] > MAX_GAP:
                        counts['gap_resets'] += 1; old = None
                if old is None:
                    old = {'frame': frame, 'class': cls, 'embeddings': deque(maxlen=WINDOW)}
                    memory[local] = old
                old['frame'] = frame; old['class'] = cls
                if present:
                    z = unit(arr['embedding'][i])
                    old['embeddings'].append(np.array(z, dtype=np.float32, copy=True))
                    if len(old['embeddings']) > 1: counts['history_used'] += 1
                    out[i] = unit(np.mean(np.stack(old['embeddings']), axis=0, dtype=np.float32))
            key = f'{pair}-{view}:{frame}:{int(arr["detector_index"][i])}'
            by_key[key] = (out[i].copy(), present)
        reports.append(counts)
    selected_keys = []
    selected_vals = []
    selected_present = []
    for frame in selected_frames:
        b = d.frame(int(frame))
        for row in b['rows']:
            key = row['key']; selected_keys.append(key)
            value, present = by_key[key]
            selected_vals.append(value if present else np.zeros(128, np.float32))
            selected_present.append(present)
    return np.asarray(selected_keys), np.asarray(selected_vals, np.float32), np.asarray(selected_present, bool), reports, d

def main():
    manifest = json.loads((P4 / 'crop32/FRAME_MANIFEST.json').read_text())
    records = {r['pair']: r for r in manifest['pairs']}
    all_pairs = FIT + CAL
    outdir = WORK / 'runs'
    outdir.mkdir(parents=True, exist_ok=False)
    frozen = {}
    audit = {'status': 'RUNNING', 'pairs': {}, 'labels_read_after_freeze': False,
             'window': WINDOW, 'max_gap': MAX_GAP, 'future_frames_used': False,
             'missing_feature_imputed': False}
    # First pass is strictly label-free. Freeze one embedding file per pair before labels are loaded.
    for pair in all_pairs:
        rec = records[pair]
        keys, emb, present, reports, d = temporal_mean_outputs(pair, rec['selected_frames'])
        with np.load(P4 / 'crop32' / f'{pair}.npz', allow_pickle=False) as crop:
            crop_keys = crop['keys']; common = crop['common_support']
        if not np.array_equal(keys, crop_keys):
            raise ValueError(f'crop key order mismatch: {pair}')
        if np.any(emb[~present] != 0):
            raise ValueError(f'missing feature was imputed: {pair}')
        if not np.isfinite(emb).all() or not np.allclose(np.linalg.norm(emb[common], axis=1), 1, atol=2e-5):
            raise ValueError(f'embedding contract failed: {pair}')
        path = outdir / f'{pair}_embeddings.npz'
        np.savez_compressed(path, row_key=keys, embedding=emb, valid=common)
        frozen[pair] = {'path': str(path), 'sha256': sha256(path), 'rows': int(len(keys)), 'common_support': int(common.sum())}
        audit['pairs'][pair] = {'role': rec['role'], 'selected_frames': len(rec['selected_frames']),
                                'reports': reports, 'rows': int(len(keys)), 'common_support': int(common.sum())}
    audit['labels_read_after_freeze'] = True
    dump(WORK / 'LABEL_FREEZE_AUDIT.json', audit)
    # Second pass uses only the frozen files and the established P4 scorer, with labels offline.
    old_methods = cmp.METHODS
    cmp.METHODS = ('p1_independent128', 'temporal_mean8')
    reports = []
    for pair in all_pairs:
        rec = records[pair]
        row, features, refs, cov = cmp.align_pair(rec, P4 / 'crop32',
                                                    {r['sequence']: r for r in json.loads((ROOT / 'p2/cache/MANIFEST.json').read_text())['sequences']})
        z = np.load(outdir / f'{pair}_embeddings.npz', allow_pickle=False)
        if not np.array_equal(row['row_key'], z['row_key']): raise ValueError(f'row alignment changed: {pair}')
        features['temporal_mean8'] = z['embedding']
        arr = cmp.evaluate_arrays(row, features)
        result = {'pair': pair, 'role': rec['role'], 'metrics': cmp.summaries(arr), 'coverage': cov,
                  'embedding': frozen[pair]}
        dump(outdir / f'{pair}_RESULTS.json', result)
        reports.append(result)
    by_role = {}
    for role in ('fit', 'calibration'):
        rs = [r for r in reports if r['role'] == role]
        macro = {m: float(np.mean([r['metrics']['all']['methods'][m]['all_candidates']['known_positive_r1_tie_averaged'] for r in rs])) for m in cmp.METHODS}
        wins = int(sum(r['metrics']['all']['methods']['temporal_mean8']['all_candidates']['known_positive_r1_tie_averaged'] > r['metrics']['all']['methods']['p1_independent128']['all_candidates']['known_positive_r1_tie_averaged'] for r in rs))
        by_role[role] = {'pair_macro_r1': macro, 'pairs': [r['pair'] for r in rs], 'wins': wins}
    fit = by_role['fit']['pair_macro_r1']; cal = by_role['calibration']['pair_macro_r1']
    gate = {'calibration_pair_macro_r1_min': 0.4641, 'relative_lift_min': 0.01,
            'calibration_pair_wins_min': 3, 'fit_drop_max': 0.05,
            'calibration_lift': cal['temporal_mean8'] - cal['p1_independent128'],
            'calibration_pair_wins': by_role['calibration']['wins'],
            'fit_drop': fit['temporal_mean8'] - fit['p1_independent128']}
    gate['pass'] = bool(cal['temporal_mean8'] >= gate['calibration_pair_macro_r1_min'] and
                        gate['calibration_lift'] >= gate['relative_lift_min'] and
                        gate['calibration_pair_wins'] >= gate['calibration_pair_wins_min'] and
                        gate['fit_drop'] >= -gate['fit_drop_max'])
    result = {'status': 'COMPLETE_P39_FIXED_MEAN_DIAGNOSTIC', 'scope': 'P4 crop32 fit15/cal5',
              'by_role': by_role, 'gate': gate, 'pairs': reports, 'dev_read': False,
              'official_val_test_read': False, 'labels_read_after_label_freeze': True}
    dump(outdir / 'RESULTS.json', result)
    decision = {'verdict': 'KEEP_P39_MEAN_ANCHOR_FOR_ONE_TRAINED_ADAPTER' if gate['pass'] else 'STOP_P39_FIXED_MEAN_BRANCH',
                'gate_pass': gate['pass'], 'reason': 'fixed rule only; no tuning', 'results': 'runs/RESULTS.json'}
    dump(WORK / 'DECISION.json', decision)
    cmp.METHODS = old_methods
    print(json.dumps({'fit': fit, 'calibration': cal, 'gate': gate, 'verdict': decision['verdict']}, indent=2))

if __name__ == '__main__': main()
