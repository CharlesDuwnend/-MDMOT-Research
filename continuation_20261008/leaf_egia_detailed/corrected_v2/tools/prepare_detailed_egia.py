"""Create exactly declared frozen-head controls; never fit or select on test-dev."""
import copy
import datetime
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
import numpy as np
from belief.egia_ablation import ConstantBinaryPrior, NeutralCueSourceModel
from belief.hierarchical import class_balanced_weights
from fit_paired_anchor import (DATA, DATA_SHA, ORIGINAL_TRAIN, F00_SHA, TEACHER_SHA,
                               independent_base_weights, objsha, sha)
from train_coverage_egia_v4 import weights

CELLS = [
    ('D00_full_reference', 'full'), ('D01_EGIA_bypass', 'disabled'),
    ('D02_no_foreground_evidence', 'no_foreground'),
    ('D03_no_coverage_evidence', 'no_coverage'),
    ('D04_no_learned_context', 'no_context'),
    ('D05_no_source_coherence', 'no_coherence'),
    ('D06_broken_source_pairing', 'pair_shuffle'),
    ('D07_no_geometry_cue', 'no_geometry_cue'),
    ('D08_no_appearance_cue', 'no_appearance_cue')]


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def main():
    assert sha(DATA) == DATA_SHA
    assert sha(ROOT / 'models/F00_full.pkl') == F00_SHA
    assert sha(ROOT / 'models/summary_frozen.pkl') == TEACHER_SHA
    data = pickle.loads(DATA.read_bytes())
    fit = data['fit']
    prep = json.loads((DATA.parent / 'prepare_receipt.json').read_text())
    assert len(fit) == 8024 and len(prep['fit_sequences']) == 39
    assert sorted({r['sequence'] for r in fit}) == sorted(prep['fit_sequences'])
    assert not ({r['flight'] for r in fit} & {r['flight'] for r in data['calibration']})
    x = np.stack([r['legacy'] for r in fit]).astype(np.float64)
    y = np.asarray([r['label'] for r in fit], dtype=int)
    assert np.bincount(y, minlength=3).tolist() == [3985, 2108, 1931]
    base_weights = independent_base_weights(fit)
    np.testing.assert_array_equal(base_weights, weights(fit))
    fg = (y != 2).astype(int)
    covered = (y[y != 2] == 1).astype(int)
    fw = class_balanced_weights(base_weights, fg, .5)
    cw = class_balanced_weights(base_weights[y != 2], covered, .5)
    foreground_prior = float(np.average(fg, weights=fw))
    coverage_prior = float(np.average(covered, weights=cw))
    base = pickle.loads((ROOT / 'models/F00_full.pkl').read_bytes())
    original_receipt = json.loads((ORIGINAL_TRAIN / 'receipt.json').read_text())
    assert original_receipt['prepared_sha256'] == DATA_SHA
    assert sha(ORIGINAL_TRAIN / 'model.pkl') == original_receipt['model_sha256']
    original = pickle.loads((ORIGINAL_TRAIN / 'model.pkl').read_bytes())
    assert objsha(original['source_model']) == objsha(base['source_model'])
    for model, rows in [(base['source_model'].anchor.foreground_model, x),
                        (base['source_model'].anchor.coverage_model, x[y != 2])]:
        scaler = model.named_steps['standardscaler']
        assert int(scaler.n_samples_seen_) == len(rows)
        np.testing.assert_array_equal(scaler.mean_, rows.mean(axis=0))
        np.testing.assert_allclose(scaler.var_, rows.var(axis=0), rtol=1e-14)
    records = []
    for name, mode in CELLS:
        if mode == 'full':
            path = ROOT / 'models/F00_full.pkl'
            bundle = base
        else:
            bundle = copy.deepcopy(base)
            bundle['egia_ablation'] = mode
            source = bundle['source_model']
            if mode in ('no_foreground', 'no_context'):
                source.anchor.foreground_model = ConstantBinaryPrior(foreground_prior)
            if mode in ('no_coverage', 'no_context'):
                source.anchor.coverage_model = ConstantBinaryPrior(coverage_prior)
            if mode == 'no_coherence':
                source.weight = 0.
            if mode in ('no_geometry_cue', 'no_appearance_cue'):
                source.__class__ = NeutralCueSourceModel
                source.neutral_cue = 'geometry' if mode == 'no_geometry_cue' else 'appearance'
            path = ROOT / 'models' / (name + '.pkl')
            with path.open('xb') as stream:
                pickle.dump(bundle, stream, protocol=4)
        # Protect all SRC estimators and every deployment setting by structural hash.
        unchanged = [k for k in base if k != 'source_model']
        assert all(objsha(bundle[k]) == objsha(base[k]) for k in unchanged)
        records.append({'name': name, 'mode': mode, 'bundle': str(path.relative_to(ROOT)),
                        'bundle_sha256': sha(path),
                        'unchanged_nonsource_fields': unchanged,
                        'source_sha256': objsha(bundle['source_model'])})
    write(ROOT / 'artifacts/PRIOR_AND_BUNDLE_RECEIPT.json', {
        'status': 'PREPARED_FIT_ONLY_NEUTRAL_CONTROLS',
        'created_utc': datetime.datetime.utcnow().isoformat() + 'Z',
        'fit_rows': len(fit), 'fit_sequences': prep['fit_sequences'],
        'fit_flights': sorted({r['flight'] for r in fit}),
        'calibration_used_for_prior': False, 'testdev_selection': False, 'head_refit': False,
        'foreground_prior': foreground_prior, 'coverage_prior': coverage_prior,
        'base_weight_sha256': objsha(base_weights),
        'foreground_weight_sha256': objsha(fw), 'coverage_weight_sha256': objsha(cw),
        'fit_feature_sha256': objsha(x), 'fit_label_sha256': objsha(y),
        'data_sha256': DATA_SHA, 'original_model_sha256': original_receipt['model_sha256'],
        'source_files_sha256': {str(p): sha(p) for p in
            [DATA, ORIGINAL_TRAIN / 'model.pkl', ORIGINAL_TRAIN / 'receipt.json',
             DATA.parent / 'prepare_receipt.json', ROOT / 'models/F00_full.pkl',
             ROOT / 'models/summary_frozen.pkl', ROOT / 'belief/egia_ablation.py',
             ROOT / 'tools/prepare_detailed_egia.py']}, 'cells': records})
    print(json.dumps({'foreground_prior': foreground_prior, 'coverage_prior': coverage_prior,
                      'cells': len(records), 'head_refit': False}))


if __name__ == '__main__':
    main()
