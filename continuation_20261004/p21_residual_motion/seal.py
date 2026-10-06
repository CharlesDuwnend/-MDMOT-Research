"""Seal the bounded P21 information gate without changing historical artifacts."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    feature = json.loads((HERE / 'FEATURE_RECEIPT.json').read_text())
    result = json.loads((HERE / 'INFORMATION_RESULT.json').read_text())
    records = json.loads((HERE / 'QUERY_DIAGNOSTICS.json').read_text())
    assert feature['status'] == 'FEATURES_FROZEN_BEFORE_LABELS'
    assert (HERE / 'launcher.exit').read_text().strip() == '0'
    assert result['evaluable_query_count'] == len(records)
    assert result['feature_freeze_sha256'] == sha(HERE / 'FEATURE_RECEIPT.json')
    for item in feature['files'] + feature['sources'] + result['label_sources']:
        assert sha(item['path']) == item['sha256'], item['path']
    listed_features = {str(Path(item['path']).resolve()) for item in feature['files']
                       if Path(item['path']).name.startswith('features_')}
    enumerated_features = {str(p.resolve()) for p in HERE.glob('pair*/features_*.json')}
    assert listed_features == enumerated_features, 'unsealed feature files were enumerated'
    for pair in feature['config']['pairs']:
        rows = [row for row in records if row['pair'] == pair]
        counts = result['per_pair'][pair]['counts']
        assert counts.get('evaluable_queries', 0) == len(rows)
        if rows:
            for name, reported in result['per_pair'][pair]['discrimination'].items():
                assert abs(reported - sum(row['credit'][name] for row in rows) / len(rows)) < 1e-12
    micro = {name: sum(row['credit'][name] for row in records) / len(records)
             for name in ['appearance', 'static_geometry', 'exact_motion', 'linearized_motion', 'shuffled_target_motion']}
    report = {
        'status': 'P21_FIT_INFORMATION_GATE_COMPLETE_NOT_A_METHOD',
        'decision': 'HOLD_DENSE_RESIDUAL_REPRESENTATION_LEARNING',
        'primary_failure': 'Only 2 of 5 pairs have at least 20 evaluable queries; cross-view geometry support is sparse.',
        'narrow_empirical_result': 'This sparse bottom-center residual readout underperforms static geometry on the same geometry-supported fit queries.',
        'not_inferred': ['motion contains no complementary identity information', 'dense target residual fields cannot help',
                         'all geometry estimators are unusable', 'a deep method has been validated', 'official MOT improvement'],
        'counts': feature['counts'], 'evaluable_queries': len(records),
        'fit_support_conditioned_query_micro_discrimination': micro,
        'geometry_wrong_motion_right': sum(row['credit']['static_geometry'] == 0 and row['credit']['exact_motion'] == 1 for row in records),
        'geometry_right_motion_wrong': sum(row['credit']['static_geometry'] == 1 and row['credit']['exact_motion'] == 0 for row in records),
        'implementation_lookback': [
            'Serialized geometry.valid as Python bool after a numpy scalar failure; initial failed attempt preserved.',
            'Bound image path to original stream.image_stem after pair78 offset failure; all ten train view mappings checked.',
            'Verified four-coordinate-domain composition and finite-difference Jacobian; recorded exact transport as a control.',
            'Exact and linearized residuals have identical discrimination on all 165 evaluated queries; linearization does not explain the narrow negative result.',
            'Feature extraction used no labels; stable current/history labels attached only after all five feature sets froze.',
            'Readout currently reduces each object to one bottom-center displacement; it does not implement dense flow or a learned residual-field model.',
            'Current gate compares standalone cues. Underperformance alone cannot falsify complementary information or a learned fusion hypothesis.',
            'Independent review checked that the nine feature files actually enumerated by the evaluator exactly match the frozen list. Future evaluators should enumerate from the receipt itself.',
        ],
        'standalone_lift_gate_validity': {
            'status': 'INVALID_AS_A_DIRECTION_FALSIFIER_DUE_TO_CEILING',
            'static_geometry_supported_pair_macro': result['supported_pair_macro']['static_geometry'],
            'maximum_possible_lift': 1 - result['supported_pair_macro']['static_geometry'],
            'predeclared_required_lift': 0.02,
            'consequence': 'The standalone +.02 criterion cannot be met on this support even by a perfect method. It is not re-tuned after seeing labels; report it as an unsuitable criterion. Independent sample-support failure still stands.',
        },
        'complementarity_limit': 'Motion rescues all four static-geometry errors but damages 44 correct choices. This is tiny fit-only oracle complementarity, not an executable learned selection rule or model gain.',
        'metric_scope': 'previously used fit subset, conventional pseudo-labels, positive-present given-candidate discrimination',
        'data_scope': 'MDMT official-train fit pairs 23/25/29/69/78 only; 8 fixed sampled times each',
        'training': False, 'GPU_used': False, 'calibration_dev_official_val_test_read': False,
        'next_research_boundary': 'A dense object-motion representation needs observable independent geometric support and a mechanism beyond GMC/homography/speed fusion. Do not promote this probe or tune its geometric thresholds into a method.',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'feature_receipt_sha256': sha(HERE / 'FEATURE_RECEIPT.json'),
        'information_result_sha256': sha(HERE / 'INFORMATION_RESULT.json'),
    }
    (HERE / 'DECISION.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    paths = [p for p in sorted(HERE.rglob('*')) if p.is_file() and p.name != 'SHA256SUMS'
             and p.suffix != '.log' and '__pycache__' not in p.parts]
    (HERE / 'SHA256SUMS').write_text(''.join(sha(p) + '  ' + str(p.relative_to(HERE)) + '\n' for p in paths))
    print(json.dumps({'status': report['status'], 'files_sealed': len(paths), 'decision_sha256': sha(HERE / 'DECISION.json'),
                      'query_micro': micro}), flush=True)


if __name__ == '__main__':
    main()
