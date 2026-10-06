#!/usr/bin/env python3
"""Independent saved-row verification; never imports the producer or loads models."""
import csv
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def main():
    start = time.monotonic()
    summary = json.loads((HERE / 'SUMMARY.json').read_text())
    inputs = json.loads((HERE / 'INPUTS.json').read_text())
    with np.load(HERE / 'observations.npz', allow_pickle=False) as saved:
        rows = {k: saved[k] for k in saved.files}
    n = len(rows['lag'])
    checks = []
    assert n == summary['comparisons'] == 40420
    assert all(v.shape == (n,) for v in rows.values())
    assert all(np.isfinite(v).all() for v in rows.values())
    keys = ('pair', 'view', 'frame', 'lag', 'raw_id', 'class_id')
    assert len(set(zip(*(rows[k] for k in keys)))) == n
    assert np.array_equal(rows['past_frame'], rows['frame'] - rows['lag'])
    assert set(rows['lag']) == {1, 4, 8}
    assert set(rows['pair']) == set(map(int, summary['fit_pairs']))
    assert set(rows['view']) == {1, 2}
    assert set(rows['class_id']) <= {0, 1, 2}
    checks.append('finite_rows_unique_same_view_identity_and_lag_contract')

    dims = summary['original_and_flow_hw']
    assert dims == [[1080, 1920, 360, 640]]
    oh, ow, fh, fw = dims[0]
    current = np.column_stack((rows['current_cx'], rows['current_cy']))
    past = np.column_stack((rows['past_cx'], rows['past_cy']))
    displacement = np.column_stack((rows['original_dx'], rows['original_dy']))
    projected = np.column_stack((rows['projected_past_cx'], rows['projected_past_cy']))
    sampled = np.column_stack((rows['flow_dx'], rows['flow_dy']))
    index = np.column_stack((rows['flow_index_x'], rows['flow_index_y']))
    max_errors = {}

    def close(name, actual, expected):
        error = float(np.max(np.abs(np.asarray(actual) - expected)))
        max_errors[name] = error
        assert np.allclose(actual, expected, rtol=1e-12, atol=1e-10), (name, error)

    close('halfpixel_raster_index', index, current * [fw / ow, fh / oh] - 0.5)
    close('per_axis_displacement_scale', displacement, sampled * [ow / fw, oh / fh])
    close('current_plus_flow_projection', projected, current + displacement)
    # hypot is separate from the producer's per-row linalg.norm calculation.
    residual = projected - past
    raft_error = np.hypot(residual[:, 0], residual[:, 1])
    residual = current - past
    zero_error = np.hypot(residual[:, 0], residual[:, 1])
    close('raft_error_px', rows['raft_error_px'], raft_error)
    close('zero_error_px', rows['zero_error_px'], zero_error)
    for endpoint in ('current', 'past'):
        scale = rows[endpoint + '_sqrt_area']
        assert (scale > 0).all()
        assert (rows[endpoint + '_min_side'] > 0).all()
        assert (rows[endpoint + '_min_side'] <= scale + 1e-10).all()
        for method, error in [('raft', raft_error), ('zero', zero_error)]:
            field = method + '_error_over_' + endpoint + '_sqrt_area'
            close(field, rows[field], error / scale)
    checks.append('saved_geometry_errors_and_both_size_normalizations_recomputed')

    for field in ('current_min_side', 'current_sqrt_area'):
        assert np.array_equal(rows[field + '_lt16'], rows[field] < 16)
    for field, points in [('current_center_inside', current), ('past_center_inside', past),
                          ('raft_endpoint_inside', projected)]:
        expected = ((points >= 0) & (points < [ow, oh])).all(axis=1)
        assert np.array_equal(rows[field], expected)
    for field in ('current_occluded', 'past_occluded', 'path_any_occluded'):
        assert set(rows[field]) <= {0, 1}
    assert (rows['path_any_occluded'] >= rows['current_occluded']).all()
    assert (rows['path_any_occluded'] >= rows['past_occluded']).all()
    checks.append('size_occlusion_and_image_domain_flags_consistent')

    groups = summary['group_counts']
    expected_groups = [g for g in json.loads((HERE.parent / 'GROUPS.json').read_text()) if g['role'] == 'fit']
    assert len(groups) == len(expected_groups) == summary['groups'] == 240
    assert {g['key'] for g in groups} == {g['key'] for g in expected_groups}
    assert len({(g['pair'], g['view'], g['frame']) for g in groups}) == 240
    exclusions = Counter()
    current_count = 0
    for group in groups:
        counts = group['counts']
        current_count += counts['current_supported_annotations']
        mask = ((rows['pair'] == int(group['pair'])) & (rows['view'] == int(group['view'])) &
                (rows['frame'] == group['frame']))
        for lag in (1, 4, 8):
            eligible = counts.get(f'lag{lag}_eligible', 0)
            absent = counts.get(f'lag{lag}_past_endpoint_absent', 0)
            gap = counts.get(f'lag{lag}_interior_annotation_gap', 0)
            assert eligible == int((mask & (rows['lag'] == lag)).sum())
            assert eligible + absent + gap == counts['current_supported_annotations']
            if absent:
                exclusions[f'lag{lag}_past_endpoint_absent'] += absent
            if gap:
                exclusions[f'lag{lag}_interior_annotation_gap'] += gap
    assert dict(exclusions) == summary['exclusions']
    assert current_count == 13663
    checks.append('all_240_group_counts_and_exclusions_conserved')

    # Reconstruct every published stratum, without importing producer stats().
    strata = summary['strata']
    masks = [('overall', strata['overall'], np.ones(n, bool))]
    for lag_key, block in strata['by_lag'].items():
        base = rows['lag'] == int(lag_key)
        for field, values in block.items():
            if field == 'all':
                masks.append((f'lag/{lag_key}/all', values, base))
            else:
                subtotal = 0
                for label, value in values.items():
                    target = {'True': True, 'False': False}.get(label, label)
                    if isinstance(target, str):
                        target = int(target)
                    mask = base & (rows[field] == target)
                    masks.append((f'lag/{lag_key}/{field}/{label}', value, mask))
                    subtotal += int(mask.sum())
                assert subtotal == int(base.sum())
    for pair, block in strata['by_pair'].items():
        for lag, value in block.items():
            masks.append((f'pair/{pair}/lag/{lag}', value,
                          (rows['pair'] == int(pair)) & (rows['lag'] == int(lag))))
    for name, value in strata['lag_size_occlusion'].items():
        lag, size, occ = name.split(':')
        field, size_label = size.split('=')
        mask = ((rows['lag'] == int(lag)) & (rows[field] == (size_label == 'True')) &
                (rows['current_occluded'] == int(occ.split('=')[1])))
        masks.append(('intersection/' + name, value, mask))
    flat_stats = []
    numeric_metrics = [k for k in rows if k.startswith(('raft_error_', 'zero_error_'))]
    for name, expected, mask in masks:
        assert expected['n'] == int(mask.sum()), name
        actual = {'n': int(mask.sum())}
        if actual['n']:
            actual['raft_better_than_zero_fraction'] = float((raft_error[mask] < zero_error[mask]).mean())
            actual['mean_paired_error_reduction_px'] = float((zero_error[mask] - raft_error[mask]).mean())
            for field in ('current_center', 'past_center', 'raft_endpoint'):
                actual[field + '_outside_count'] = int((~rows[field + '_inside'][mask]).sum())
            for field in numeric_metrics:
                v = rows[field][mask]
                actual[field] = {'mean': float(np.average(v)), 'median': float(np.percentile(v, 50)),
                                 'p90': float(np.percentile(v, 90))}
        assert set(actual) == set(expected), name
        flat = {'stratum': name}
        for key, value in actual.items():
            if isinstance(value, dict):
                for metric, number in value.items():
                    assert np.isclose(number, expected[key][metric], rtol=1e-12, atol=1e-10), (name, key, metric)
                    flat[key + '_' + metric] = number
            else:
                assert np.isclose(value, expected[key], rtol=1e-12, atol=1e-10), (name, key)
                flat[key] = value
        flat_stats.append(flat)
    checks.append('all_published_strata_recomputed_independently')
    columns = list(dict.fromkeys(k for row in flat_stats for k in row))
    with (HERE / 'STRATA.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(flat_stats)

    tests = json.loads((HERE / 'TESTS.json').read_text())
    direction = json.loads((HERE / 'DIRECTION_AUDIT.json').read_text())
    assert tests['status'] == 'PASS' and tests['GPU_used'] is False
    assert direction['status'] == 'PASS_SOURCE_DIRECTION_CONTRACT'
    assert summary['max_numpy64_vs_grid_sample32_displacement_component_difference_original_px'] <= .01
    log_lines = [json.loads(x) for x in (HERE / 'run.log').read_text().splitlines()]
    assert log_lines[-2]['groups_done'] == 240
    assert log_lines[-1]['status'] == summary['status']
    assert log_lines[-1]['comparisons'] == n
    assert log_lines[-1]['seconds'] == summary['elapsed_seconds']
    checks.append('producer_tests_direction_and_completed_run_log_pass')

    for path, digest in inputs.items():
        assert sha(path) == digest, 'Changed input: ' + path
    checks.append('every_bound_input_sha256_unchanged')
    report = dict(status='PASS_INDEPENDENT_SAVED_ROW_VERIFICATION', rows=n, groups=len(groups),
                  current_supported_annotations=current_count, strata_checked=len(masks),
                  input_hashes_rechecked=len(inputs), checks=checks, max_abs_recompute_errors=max_errors,
                  source_sha256=sha(__file__), elapsed_seconds=round(time.monotonic() - start, 3),
                  scope='saved rows and FIT input hashes only; no images, model, GPU or non-FIT GT')
    (HERE / 'VERIFY.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps(report, sort_keys=True))


if __name__ == '__main__':
    main()
