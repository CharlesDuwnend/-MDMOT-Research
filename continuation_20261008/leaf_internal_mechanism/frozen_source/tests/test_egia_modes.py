"""Audit the missing fusion-on/selective-off cell against real frozen heads.

These contracts do not train a head or report MOT effectiveness.
"""
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
PREPARED = Path(
    '/home/chenhc/src_egia_belief_20260923/artifacts/'
    'prepare_coverage_v4_fit39_cal8_v2/prepared.pkl'
)
F00_SHA256 = 'ee903030e48f73fe87411b6e9aa60d52bbf2ee2d895aebfa8085d260779702ac'
PREPARED_SHA256 = 'e126f0ebcbb284985eb8bfd43c6bcaec4dc401fe7e507960518ea59fafde055d'


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical(value):
    """Include every estimator attribute, array value and policy field."""
    if isinstance(value, np.ndarray):
        data = canonical(value.tolist()) if value.dtype.hasobject else value.tobytes().hex()
        return ['array', str(value.dtype), list(value.shape), data]
    if isinstance(value, np.generic):
        return ['scalar', str(value.dtype), value.item()]
    if value is None or isinstance(value, (str, int, float, bool)):
        return [type(value).__name__, value]
    if isinstance(value, (list, tuple)):
        return [type(value).__name__, [canonical(item) for item in value]]
    if isinstance(value, dict):
        return ['dict', [[str(key), canonical(item)] for key, item in
                         sorted(value.items(), key=lambda pair: str(pair[0]))]]
    if hasattr(value, '__dict__'):
        return ['object', type(value).__module__ + '.' + type(value).__qualname__,
                canonical(value.__dict__)]
    raise TypeError('Unsupported audit value: ' + str(type(value)))


def object_sha256(value):
    encoded = json.dumps(canonical(value), sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(encoded.encode()).hexdigest()


def changed_fields(reference, candidate):
    return sorted(key for key in set(reference) | set(candidate)
                  if key not in reference or key not in candidate or
                  object_sha256(reference[key]) != object_sha256(candidate[key]))


@pytest.fixture(scope='module')
def bundles():
    reference_path = ROOT / 'models/F00_full.pkl'
    assert file_sha256(reference_path) == F00_SHA256
    return (pickle.loads(reference_path.read_bytes()),
            pickle.loads((ROOT / 'models/FusionOn_NoSelective.pkl').read_bytes()))


@pytest.fixture(scope='module')
def prepared():
    assert file_sha256(PREPARED) == PREPARED_SHA256
    return pickle.loads(PREPARED.read_bytes())


def test_missing_cell_changes_only_selective_policy(bundles):
    reference, candidate = bundles
    assert changed_fields(reference, candidate) == ['selective_birth_policy']
    assert reference['selective_birth_policy'] is True
    assert candidate['selective_birth_policy'] is False
    assert candidate['egia_fusion_policy'] is True
    assert candidate['egia_fusion_weight'] == 0.4
    assert candidate['selective_birth_margin'] == 0.2
    assert candidate['source_model'].weight == 0.6331914010259171
    assert candidate.get('safe_native_incumbent', False) is False
    for key in ['asymmetric_selective_birth_policy', 'owner_selective_birth_policy',
                'decision_specific_birth_policy']:
        assert not candidate.get(key, False)


def test_real_fitcal_source_posteriors_are_identical(bundles, prepared):
    reference, candidate = bundles
    for split in ['fit', 'calibration']:
        rows = prepared[split][::max(1, len(prepared[split]) // 64)]
        reference_p, reference_e = reference['source_model'].predict_rows(rows)
        candidate_p, candidate_e = candidate['source_model'].predict_rows(rows)
        np.testing.assert_array_equal(candidate_p, reference_p)
        np.testing.assert_array_equal(candidate_e, reference_e)
        assert np.isfinite(candidate_p).all()
        np.testing.assert_allclose(candidate_p.sum(axis=1), 1., atol=1e-12)


def test_actual_data_membership_and_fit_only_scaler_support(bundles, prepared):
    reference, _ = bundles
    fit, cal = prepared['fit'], prepared['calibration']
    assert (len(fit), len(cal)) == (8024, 8411)
    assert len({row['sequence'] for row in fit}) == 39
    assert len({row['sequence'] for row in cal}) == 8
    assert not ({row['flight'] for row in fit} & {row['flight'] for row in cal})
    assert set(row['split'] for row in fit) == {'fit'}
    assert set(row['split'] for row in cal) == {'calibration'}
    fit_x = np.stack([row['legacy'] for row in fit])
    fit_y = np.asarray([row['label'] for row in fit])
    assert fit_x.shape == (8024, 10) and np.isfinite(fit_x).all()
    assert np.bincount(fit_y, minlength=3).tolist() == [3985, 2108, 1931]
    anchor = reference['source_model'].anchor
    for pipeline, rows in [(anchor.foreground_model, fit_x),
                           (anchor.coverage_model, fit_x[fit_y != 2])]:
        scaler = pipeline.named_steps['standardscaler']
        assert int(scaler.n_samples_seen_) == len(rows)
        np.testing.assert_allclose(scaler.mean_, rows.astype(float).mean(axis=0),
                                   rtol=0, atol=1e-12)
