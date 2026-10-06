#!/usr/bin/env python3
"""Read historical fit/calibration caches; write only this audit's new JSON.

No fitting, inference, image decoding, official-val/test access, or old writes.
Tie-aware quantities are expected COUNTS under uniform boundary sampling;
they are not a new method evaluation or expected macro precision.
"""
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
ROOT = Path('/home/chenhc/mdmot_research_20261002')
OUT = Path(__file__).resolve().parent
GEOM = ROOT / 'p19_merge_geometry/tracklet_geometry'
NS = (5, 10, 20, 50, 100, 200)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def inspect_source(path, start, end):
    path = ROOT / path
    lines = path.read_text().splitlines()
    return {'path': str(path), 'sha256': digest(path), 'line_start': start,
            'line_end': end, 'excerpt': '\n'.join(lines[start - 1:end])}


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def subset_stats(d):
    n, y, k = d['n_co'], d['y'].astype(bool), d['known'].astype(bool)
    return {'candidates': len(n), 'known': int(k.sum()), 'unknown': int((~k).sum()),
            'positive': int(y.sum()), 'singleton_count': int((n == 1).sum()),
            'singleton_fraction': float((n == 1).mean()),
            'positive_singleton_count': int((n[y] == 1).sum()),
            'positive_singleton_fraction': float((n[y] == 1).mean()),
            'positive_median_nco': float(np.median(n[y])),
            'median_nco': float(np.median(n)),
            'known_negative_singleton_fraction': float((n[k & ~y] == 1).mean())}


def label_audit(pair, view):
    ip = ROOT / f'p2/cache/inputs/{pair}-{view}.npz'
    lp = ROOT / f'p2/cache/labels/{pair}-{view}.npz'
    with np.load(ip, allow_pickle=False) as z:
        lid, present, cls = z['local_id'], z['feature_present'], z['cls']
    with np.load(lp, allow_pickle=False) as z:
        pid = z['pid']
    inds = np.argsort(lid, kind='stable')
    uniq, starts = np.unique(lid[inds], return_index=True)
    counts = dict(tracks=len(uniq), known_tracks=0, mixed_pid_tracks=0,
                  vote_ties=0, majority_purity_below_0p9=0,
                  unknown_rows_in_majority_labeled_tracks=0,
                  present_rows_in_majority_labeled_tracks=0,
                  multi_class_majority_labeled_tracks=0)
    for ix in np.split(inds, starts[1:]):
        use = ix[present[ix]]
        votes, votes_n = np.unique(pid[use][pid[use] >= 0], return_counts=True)
        if not len(votes):
            continue
        counts['known_tracks'] += 1
        counts['mixed_pid_tracks'] += int(len(votes) > 1)
        counts['vote_ties'] += int((votes_n == votes_n.max()).sum() > 1)
        counts['majority_purity_below_0p9'] += int(votes_n.max() / votes_n.sum() < 0.9)
        counts['unknown_rows_in_majority_labeled_tracks'] += int((pid[use] < 0).sum())
        counts['present_rows_in_majority_labeled_tracks'] += len(use)
        counts['multi_class_majority_labeled_tracks'] += int(len(np.unique(cls[use])) > 1)
    counts['sources'] = [{'path': str(p), 'sha256': digest(p)} for p in (ip, lp)]
    return counts


def coverage(cal):
    arms = {
        'cnt_ge0.7': {p: d['ge'][:, list(d['grid']).index(0.7)] for p, d in cal.items()},
        'host_maxcos': {p: d['max_cos'] for p, d in cal.items()},
        'mean_cos': {p: d['sum_cos'] / np.maximum(d['n_co'], 1) for p, d in cal.items()},
        'n_co_only': {p: d['n_co'] for p, d in cal.items()},
    }
    out = {}
    for name, ps in arms.items():
        pool = np.concatenate(list(ps.values()))
        out[name] = []
        for n in NS:
            target = min(n * len(cal), len(pool))
            tau = float(np.sort(pool)[::-1][target - 1])
            strict = int((pool > tau).sum())
            tie = int((pool == tau).sum())
            take = target - strict
            fraction = take / tie
            per = {}
            macro, recall = [], []
            for p, d in cal.items():
                s, k, y = ps[p], d['known'].astype(bool), d['y'].astype(bool)
                acc, above, eq = s >= tau, s > tau, s == tau
                tp, accepted, known = int((acc & k & y).sum()), int(acc.sum()), int((acc & k).sum())
                macro.append(tp / max(accepted, 1))
                recall.append(tp / max(int((k & y).sum()), 1))
                per[p] = {
                    'legacy_accepted': accepted, 'legacy_tp': tp,
                    'legacy_known_accepted': known,
                    'legacy_unknown_accepted': int((acc & ~k).sum()),
                    'legacy_reported_precision_lower_bound': tp / max(accepted, 1),
                    'known_only_precision': tp / known if known else None,
                    'strictly_above': int(above.sum()), 'boundary_ties': int(eq.sum()),
                    'exact_budget_uniform_tie_expected_accepted': float(above.sum() + fraction * eq.sum()),
                    'exact_budget_uniform_tie_expected_unknown_accepted': float((above & ~k).sum() + fraction * (eq & ~k).sum()),
                }
            out[name].append({'nominal_N_per_pair': n, 'target_total': target, 'tau': tau,
                              'strictly_above_total': strict, 'boundary_tie_total': tie,
                              'needed_from_boundary_tie': take,
                              'boundary_inclusion_fraction_for_exact_budget': fraction,
                              'actual_legacy_total': sum(v['legacy_accepted'] for v in per.values()),
                              'legacy_macro_precision_lower_bound': float(np.mean(macro)),
                              'legacy_macro_recall_known': float(np.mean(recall)),
                              'per_pair': per})
    return out


def main():
    result_path = OUT / 'REASSESSMENT.json'
    if result_path.exists():
        raise FileExistsError('refuse to overwrite ' + str(result_path))
    split_path = ROOT / 'p0/data/ASSOCIATION_DIAGNOSTIC_SPLIT.json'
    split = json.loads(split_path.read_text())
    assignments = split['assignments']
    assert not (set(assignments['fit']) & set(assignments['calibration']))
    before = {}
    data, role_summary, per_pair = {}, {}, {}
    for role in ('fit', 'calibration'):
        data[role] = {}
        per_pair[role] = {}
        for pair in sorted(assignments[role], key=int):
            p = GEOM / f'{role}_{pair}.npz'
            before[str(p)] = digest(p)
            d = load(p)
            data[role][pair] = d
            per_pair[role][pair] = subset_stats(d)
        joined = {key: np.concatenate([d[key] for d in data[role].values()])
                  for key in ('n_co', 'y', 'known')}
        role_summary[role] = subset_stats(joined)
    cal = data['calibration']
    counts = coverage(cal)
    label_details = {p: {str(v): label_audit(p, v) for v in (1, 2)} for p in cal}
    label_totals = {k: sum(v[k] for d in label_details.values() for v in d.values())
                    for k in next(iter(label_details.values()))['1'] if k != 'sources'}
    screen_path = ROOT / 'p19_merge_geometry/P19A_SCREEN.json'
    original = json.loads(screen_path.read_text())
    reproduced = {}
    for arm, rows in counts.items():
        p = np.asarray([r['legacy_macro_precision_lower_bound'] for r in rows])
        r = np.asarray([r['legacy_macro_recall_known'] for r in rows])
        reproduced[arm] = {'precision_max_abs_error': float(np.max(np.abs(p - original['curve'][arm]['precision']))),
                           'recall_max_abs_error': float(np.max(np.abs(r - original['curve'][arm]['recall'])))}
    assert all(v['precision_max_abs_error'] < 1e-12 and v['recall_max_abs_error'] < 1e-12 for v in reproduced.values())
    differences = (np.asarray(original['curve']['cosine_geometry']['precision']) -
                   np.asarray(original['curve']['cosine_only']['precision'])) * 100
    p10_path = ROOT / 'p10_falsification/DECISION.json'
    p10 = json.loads(p10_path.read_text())
    evidence = {
        'motion_placeholder': inspect_source('p19_merge_geometry/build_tracklet_geometry.py', 69, 90),
        'raw_unregistered_coordinate_difference': inspect_source('p19_merge_geometry/build_tracklet_geometry.py', 79, 84),
        'whole_sequence_loop_and_majority_votes': inspect_source('p19_merge_geometry/build_tracklet_geometry.py', 31, 50),
        'majority_labels_and_future_extent': inspect_source('p19_merge_geometry/build_tracklet_geometry.py', 94, 115),
        'feature_definition': inspect_source('p19_merge_geometry/p19a_screen.py', 34, 52),
        'threshold_tie_policy': inspect_source('p19_merge_geometry/p19a_screen.py', 55, 59),
        'unknown_denominator': inspect_source('p19_merge_geometry/p19a_screen.py', 99, 109),
        'cached_without_source_fingerprint': inspect_source('p19_merge_geometry/build_tracklet_geometry.py', 118, 129),
        'linear_pooled_fit': inspect_source('p19_merge_geometry/p19a_screen.py', 64, 78),
        'declared_unknown_policy': inspect_source('p0/data/ASSOCIATION_DIAGNOSTIC_SPLIT.json', 89, 91),
        'split_and_identity_truth_boundary': inspect_source('p0/data/ASSOCIATION_DIAGNOSTIC_SPLIT.json', 35, 50),
    }
    for ev in evidence.values():
        before[ev['path']] = ev['sha256']
    for p in (screen_path, p10_path, ROOT / 'p19_merge_geometry/P19_PREREG.json',
              ROOT / 'p10_falsification/run_minimal_falsification_v1.py',
              ROOT / 'p17_link_verifier/build_tracklet_pairs.py',
              ROOT / 'p18_merge_verifier/coverage_curve.py', ROOT / 'p2/src/data.py'):
        before[str(p)] = digest(p)
    for d in label_details.values():
        for v in d.values():
            for src in v['sources']:
                before[src['path']] = src['sha256']
    unchanged = all(digest(p) == h for p, h in before.items())
    assert unchanged
    report = {
        'schema': 'P19A_INDEPENDENT_REASSESSMENT_V1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'INVALID_FOR_GENERAL_GEOMETRY_OR_MOTION_STOP',
        'decision': 'Retain this run as a limited retrospective handcrafted-feature screen; do not infer that non-cosine evidence adds nothing or that geometric/target-motion information is exhausted.',
        'scope': {'historical_files_read_only': True, 'official_val_test_read': False,
                  'train_cache_roles_read': ['fit', 'calibration'], 'dev_cache_read': False,
                  'training_executed': False, 'inference_executed': False, 'images_decoded': False,
                  'gpu_used': False, 'new_model_implemented': False},
        'reproduce_command': 'python3 ' + str(Path(__file__).resolve()),
        'script_sha256': digest(Path(__file__)),
        'original_result_source': {'path': str(screen_path), 'sha256': digest(screen_path)},
        'split_source': {'path': str(split_path), 'sha256': digest(split_path),
                         'fit': assignments['fit'], 'calibration': assignments['calibration'],
                         'fit_cal_pair_overlap': []},
        'empirical': {'role_summary': role_summary, 'per_pair': per_pair,
                      'coverage_and_ties': counts,
                      'original_nontrained_arm_reproduction': reproduced,
                      'reported_cosine_geometry_minus_cosine_only_precision_pp': differences.tolist(),
                      'calibration_track_label_purity_totals': label_totals,
                      'calibration_track_label_purity_per_pair_view': label_details},
        'source_evidence': evidence,
        'interpretations': [
            'Motion updating is a pass statement. absdrift_sum equals accumulated abs(scale_ratio - 1), not target or camera motion.',
            'Raw cross-view center subtraction is not calibrated geometry; no camera compensation or shared coordinate mapping is implemented.',
            'All six reported cosine_geometry macro precision values exceed cosine_only, but this is not proof of useful conditional information or online gain because per-pair coverage and recall differ and uncertainty was not assessed.',
            'Legacy precision counts unknown accepted candidates in the denominator. Interpret it as a conservative lower bound unless unknown negatives are verified; known precision and unknown acceptance must be reported separately.',
            'The count score N=200 accepts the complete calibration pool because kth score is zero; its recall=1 is not matched-coverage evidence.',
            'Whole-sequence features and majority labels are legitimate for a declared retrospective task but cannot support prefix-only online claims. No direct PID feature input or fit/cal pair overlap was found.',
            'Known-vote majorities ignore purity and ties; their meaning is whole-track majority-label compatibility, not instantaneous identity truth or verified physical cross-view identity.',
            'The split is internal official-train diagnostic isolation, with scene independence and physical global identities unverified. It is not untouched whole-system generalization.',
            'Most candidate tracklet pairs, including most positive pairs, have only one co-observation. No conclusion about multi-step target motion follows.',
            'P19_PREREG.json describes a memory-threshold sweep, not this geometry screen.',
            'Exact-budget uniform boundary sampling quantities above are expected counts only. They are not new model scores, expected macro precision, or evidence of improved tracking.',
            'P17/P18 tracklet builder and coverage curve share the majority-label and threshold-tie conventions; concerns are inherited rather than uniquely created by P19.',
        ],
        'retracted_prior_recommendation': {
            'reason': 'The proposed four-observation patch path gate was already executed in P10; the earlier suggestion mistakenly relied on a stale source assessment.',
            'source': str(p10_path), 'source_sha256': digest(p10_path),
            'decision': p10['status'], 'cal5_pair_macro': p10['cal5_pair_macro'],
            'implementation_findings': p10['implementation_findings'],
            'do_not_repeat': 'Do not relaunch patch route overlap / partial transport / wrong-history gating or entropy-temperature-null-bin-threshold rescue as an untested axis.',
            'runner_read_but_not_rerun': str(ROOT / 'p10_falsification/run_minimal_falsification_v1.py'),
        },
        'bounded_next_direction': 'See NEW_DIRECTION.md: only assess observability of target motion after causal camera-motion removal with target-excluded local differential transfer; no method or novelty claim.',
        'verification': {'historical_sources_unchanged_during_audit': unchanged,
                         'source_file_count': len(before), 'historical_source_sha256': before},
    }
    result_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps({'path': str(result_path), 'sha256': digest(result_path),
                      'calibration': role_summary['calibration'],
                      'vote_purity': label_totals, 'historical_sources_unchanged': unchanged,
                      'original_score_reproduction': reproduced}, ensure_ascii=False))


if __name__ == '__main__':
    main()
