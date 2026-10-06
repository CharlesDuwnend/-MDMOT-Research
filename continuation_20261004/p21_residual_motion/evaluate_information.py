"""Offline fit-only information assay over previously frozen motion features."""
import sys
sys.dont_write_bytecode = True
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np

HERE = Path(__file__).resolve().parent
OLD = Path('/home/chenhc/mdmot_research_20261002')
sys.path.insert(0, str(OLD / 'p2/src'))
from data import PairData


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def main():
    freeze = json.loads((HERE / 'FEATURE_RECEIPT.json').read_text())
    assert freeze['status'] == 'FEATURES_FROZEN_BEFORE_LABELS' and freeze['PID_read'] is False
    assert (HERE / 'runner.exit').read_text().strip() == '0'
    for item in freeze['files']:
        assert sha(item['path']) == item['sha256'], item['path']
    if (HERE / 'INFORMATION_RESULT.json').exists():
        raise ValueError('information assay already exists')
    protocol = {
        'written_utc': datetime.now(timezone.utc).isoformat(),
        'feature_freeze_sha256': sha(HERE / 'FEATURE_RECEIPT.json'),
        'code_sha256': sha(Path(__file__)),
        'scope': 'previously used fit5 only, feature-support-conditioned information assay, no performance claim',
        'query': 'known current and t-8 pseudo-label agree; at least one similarly stable known positive and negative',
        'positive_choice': 'smallest observation key if multiple; no score-based positive selection',
        'negative_choice': 'largest current P1 cosine among known nonmatching candidates on the same frozen support',
        'readouts': ['current P1 cosine', 'background static geometry', 'exact residual motion', 'linearized residual motion', 'shuffled right residual'],
        'ties': 'half credit at absolute tolerance 1e-12',
        'unknown_handling': 'retain feature candidate inventory; separate counts; no unknown-as-negative',
        'permutation': 'right residuals circularly permuted by sorted observation key within same frame/class, without labels',
        'minimum_support_for_next_learning_gate': '20 evaluable queries per pair in at least 3 of the 5 fixed fit pairs',
        'motion_gate': 'exact motion improves conditional discrimination over static geometry by .02 pair macro; improves >=3 pairs; target shuffle removes >=half lift',
        'cal_dev_val_test_read': False, 'train_weights': False,
    }
    dump(HERE / 'INFORMATION_PROTOCOL.json', protocol)
    per_pair, query_records, label_sources = {}, [], []
    for pair in freeze['config']['pairs']:
        data = PairData(pair, with_labels=True)
        assert data.role == 'fit'
        for view in ('1', '2'):
            path = OLD / 'p2/cache/labels' / (pair + '-' + view + '.npz')
            if path.exists():
                label_sources.append({'path': str(path), 'sha256': sha(path)})
        counts = Counter()
        metrics = defaultdict(list)
        noise = defaultdict(list)
        for path in sorted((HERE / ('pair' + pair)).glob('features_*.json')):
            frame = int(path.stem.split('_')[-1])
            features = json.loads(path.read_text())
            current, history = data.frame(frame), data.frame(frame - freeze['config']['lag'])
            current_info = {row['key']: {'pid': int(current['pid'][i]), 'embedding': current['emb'][i], 'row': row}
                            for i, row in enumerate(current['rows'])}
            old_info = {(row['camera'], row['local_track_id_index']): int(history['pid'][i])
                        for i, row in enumerate(history['rows'])}
            def stable(key):
                info = current_info[key]
                row = info['row']
                return info['pid'] >= 0 and old_info.get((row['camera'], row['local_track_id_index']), -1) == info['pid']
            groups, right_vectors = defaultdict(list), defaultdict(dict)
            for edge in features:
                groups[edge['left_key']].append(edge)
                right_vectors[edge['class']][edge['right_key']] = np.array(edge['residual_right'])
                left, right = edge['left_key'], edge['right_key']
                counts['all_feature_edges'] += 1
                if current_info[left]['pid'] < 0 or current_info[right]['pid'] < 0:
                    counts['unknown_current_label_edges'] += 1
                elif not stable(left) or not stable(right):
                    counts['unstable_or_unknown_history_label_edges'] += 1
                elif current_info[left]['pid'] == current_info[right]['pid']:
                    counts['stable_positive_edges'] += 1
                else:
                    counts['stable_negative_edges'] += 1
                noise['linearization_error_pixels'].append(edge['linearization_error_pixels'])
            permuted = {}
            for category, vectors in right_vectors.items():
                keys = sorted(vectors)
                for index, key in enumerate(keys):
                    permuted[key] = vectors[keys[(index + 1) % len(keys)]]
            for key, edges in groups.items():
                counts['all_supported_query_observations'] += 1
                if not stable(key):
                    counts['query_label_unknown_or_unstable'] += 1
                    continue
                positives, negatives = [], []
                for edge in edges:
                    right = edge['right_key']
                    if not stable(right):
                        continue
                    if current_info[key]['pid'] == current_info[right]['pid']:
                        positives.append(edge)
                    else:
                        negatives.append(edge)
                if not positives or not negatives:
                    counts['query_without_stable_positive_or_negative'] += 1
                    continue
                def cosine(edge):
                    return float(current_info[key]['embedding'] @ current_info[edge['right_key']]['embedding'])
                positive = min(positives, key=lambda x: x['right_key'])
                negative = max(negatives, key=lambda x: (cosine(x), x['right_key']))
                def scores(edge):
                    return {'appearance': cosine(edge),
                            'static_geometry': -edge['static_position_error_pixels'],
                            'exact_motion': -edge['exact_motion_error_pixels'],
                            'linearized_motion': -edge['motion_error_pixels'],
                            'shuffled_target_motion': -float(np.linalg.norm(np.array(edge['exact_transported_left']) - permuted[edge['right_key']]))}
                pos, neg = scores(positive), scores(negative)
                credits = {name: (0.5 if abs(pos[name] - neg[name]) <= 1e-12 else float(pos[name] > neg[name])) for name in pos}
                for name, credit in credits.items():
                    metrics[name].append(credit)
                counts['evaluable_queries'] += 1
                counts['geometry_wrong_motion_right'] += int(credits['static_geometry'] == 0 and credits['exact_motion'] == 1)
                counts['geometry_right_motion_wrong'] += int(credits['static_geometry'] == 1 and credits['exact_motion'] == 0)
                query_records.append({'pair': pair, 'frame': frame, 'query_key': key,
                                      'positive_key': positive['right_key'], 'negative_key': negative['right_key'],
                                      'positive_scores': pos, 'negative_scores': neg, 'credit': credits})
        per_pair[pair] = {'counts': dict(counts), 'discrimination': {name: statistics.mean(values) for name, values in metrics.items()},
                          'linearization_error_p90_pixels': float(np.percentile(noise['linearization_error_pixels'], 90)) if noise['linearization_error_pixels'] else None}
    active = [row for row in per_pair.values() if row['counts'].get('evaluable_queries', 0)]
    eligible_pairs = [pair for pair, row in per_pair.items() if row['counts'].get('evaluable_queries', 0) >= 20]
    macro = {name: statistics.mean(row['discrimination'][name] for row in active)
             for name in ['appearance', 'static_geometry', 'exact_motion', 'linearized_motion', 'shuffled_target_motion']} if active else {}
    result = {'status': 'FIT_ONLY_INFORMATION_ASSAY_COMPLETE', 'per_pair': per_pair, 'supported_pair_macro': macro,
              'pairs_with_20_evaluable_queries': eligible_pairs, 'minimum_support_gate': len(eligible_pairs) >= 3,
              'evaluable_query_count': len(query_records), 'denominator_warning': 'Only geometrically supported, stable-label fit queries; not population MOT or generalization evidence',
              'cal_dev_val_test_read': False, 'model_trained': False, 'method_confirmed': False,
              'label_sources': label_sources, 'feature_freeze_sha256': sha(HERE / 'FEATURE_RECEIPT.json'),
              'protocol_sha256': sha(HERE / 'INFORMATION_PROTOCOL.json')}
    if macro:
        result['motion_delta_vs_static'] = macro['exact_motion'] - macro['static_geometry']
        result['motion_wins_pairs'] = sum(row['discrimination']['exact_motion'] > row['discrimination']['static_geometry'] for row in active)
        result['motion_gate'] = (result['minimum_support_gate'] and result['motion_delta_vs_static'] >= .02
                                 and result['motion_wins_pairs'] >= 3
                                 and macro['exact_motion'] - macro['shuffled_target_motion'] >= result['motion_delta_vs_static'] / 2)
    else:
        result['motion_gate'] = False
    result['decision'] = 'HOLD_LEARNING_PENDING_BETTER_GEOMETRY_AND_INFORMATION' if not result['motion_gate'] else 'KEEP_FOR_BOUNDED_REPRESENTATION_LEARNING'
    dump(HERE / 'QUERY_DIAGNOSTICS.json', query_records)
    dump(HERE / 'INFORMATION_RESULT.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
