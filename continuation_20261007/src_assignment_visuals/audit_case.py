#!/usr/bin/env python3
"""Verify the illustrated component and enumerate its assignment ambiguity."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import itertools
import json
from pathlib import Path
import sys
import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from extract_cases import assign, OLD, sha, rows


def enumerate_partial(cost, tau):
    m, n = cost.shape
    configurations = []
    for k in range(min(m, n) + 1):
        for rr in itertools.combinations(range(m), k):
            for cc in itertools.combinations(range(n), k):
                for perm in itertools.permutations(cc):
                    pairs = list(zip(rr, perm))
                    if any(cost[i, j] > tau for i, j in pairs):
                        continue
                    # Same objective as the host's cost-limit LAP extension.
                    objective = sum(cost[i, j] for i, j in pairs) + (m+n-2*k) * tau/2
                    configurations.append({'pairs': [list(p) for p in pairs], 'objective': float(objective)})
    configurations.sort(key=lambda r: (r['objective'], r['pairs']))
    best = configurations[0]['objective']
    optima = [r for r in configurations if abs(r['objective']-best) <= 1e-12]
    next_best = next((r['objective'] for r in configurations if r['objective'] > best + 1e-12), None)
    return {'legal_configuration_count': len(configurations), 'optima': optima,
        'optimal_objective': best, 'gap_to_next_distinct_objective': next_best-best if next_best is not None else None,
        'all_configurations': configurations}


def hungarian_extended(cost, tau):
    m, n = cost.shape
    extended = np.full((m+n, n+m), 1e6)
    extended[:m, :n] = np.where(cost <= tau, cost, 1e6)
    for i in range(m):
        extended[i, n+i] = tau/2
    for j in range(n):
        extended[m+j, j] = tau/2
    extended[m:, n:] = 0
    ri, ci = linear_sum_assignment(extended)
    pairs = {(int(i), int(j)) for i, j in zip(ri, ci) if i < m and j < n}
    return pairs, float(extended[ri, ci].sum())


def main():
    receipt = json.loads((HERE / 'CASE_SEARCH_RECEIPT.json').read_text())
    assert receipt['status'] == 'COMPLETE_ALL_CALIBRATION_CASE_SEARCH'
    case = receipt['cases']['3x3']
    assert case['closed_legal_component'] and case['standalone_and_full_pool_assignments_identical']
    assert case['base_owner_score'] == {'same_owner_pairs': 1, 'different_owner_pairs': 2}
    assert case['src_owner_score'] == {'same_owner_pairs': 3}
    base, src, b, z, u = map(np.array, [case['base_cost'], case['src_cost'], case['beliefs'],
        case['observation_probabilities'], case['semantic_penalty']])
    prior, tau = np.array(case['prior']), case['threshold']
    independent = np.array([[min(1., max(0., 1-sum(float(b[i, k]) * float(z[j, k])/float(prior[k])
        for k in range(2)))) for j in range(3)] for i in range(3)])
    assert np.allclose(u, independent, atol=1e-14)
    assert np.allclose(src, base + (1-base)*independent, atol=1e-14)
    assert np.all(src >= base) and np.all(src <= 1)
    event = next(r for r in rows(OLD / 'capture_fitcal_v1' / case['sequence'] / 'association.jsonl.gz')
                 if r['source_frame'] == case['frame'] and r['stage'] == 0)
    full_base = np.array(event['base_cost'])
    ti, di = case['global_row_indices'], case['global_column_indices']
    assert list(full_base.shape) == case['full_pool_shape']
    assert np.array_equal(base, full_base[np.ix_(ti, di)])
    assert not np.any(full_base[np.ix_(ti, [j for j in range(full_base.shape[1]) if j not in di])] <= tau)
    assert not np.any(full_base[np.ix_([i for i in range(full_base.shape[0]) if i not in ti], di)] <= tau)
    full_pairs = assign(full_base, tau)
    assert {(ti[i], di[j]) for i, j in case['base_assignment']} <= full_pairs
    for i, j in case['src_assignment']:
        assert case['track_labels'][i]['gt_id'] == case['detection_labels'][j]['gt_id']
    analysis = {}
    for name, cost, stored in [('base', base, case['base_assignment']), ('src', src, case['src_assignment'])]:
        enumerated = enumerate_partial(cost, tau)
        lap_pairs = assign(cost, tau)
        assert lap_pairs == set(map(tuple, stored))
        assert any(lap_pairs == set(map(tuple, r['pairs'])) for r in enumerated['optima'])
        scipy_pairs, scipy_objective = hungarian_extended(cost, tau)
        assert abs(scipy_objective-enumerated['optimal_objective']) < 1e-12
        assert any(scipy_pairs == set(map(tuple, r['pairs'])) for r in enumerated['optima'])
        enumerated['host_selected_pairs'] = stored
        enumerated['independent_hungarian_pairs'] = sorted(map(list, scipy_pairs))
        analysis[name] = enumerated
    assert len(analysis['base']['optima']) == 2
    assert len(analysis['src']['optima']) == 1
    selected = {'evidence_type': 'real frozen-calibration same-state operator and assignment example',
        'case': case, 'assignment_analysis': analysis,
        'interpretation': 'SRC resolves two tied base-cost optima into one semantically consistent optimum.',
        'tie_boundary': 'The crossed and owner-consistent base assignments have exactly equal cost; the base solver did not have a unique wrong optimum.',
        'search_counts': receipt['counts'],
        'case_search_receipt_sha256': sha(HERE / 'CASE_SEARCH_RECEIPT.json')}
    (HERE / 'SELECTED_CASE.json').write_text(json.dumps(selected, indent=2) + '\n')
    audit = {'status': 'PASS_INDEPENDENT_CLOSED_COMPONENT_AND_ASSIGNMENT_AUDIT',
        'checks': {'all_calibration_search_completed': True, 'raw_full_matrix_matches_slice_exactly': True,
            'no_legal_edges_connect_to_omitted_candidates': True, 'full_pool_base_assignment_contains_displayed_assignment': True,
            'scalar_semantic_penalty_and_cost_reconstructed': True, 'host_lap_results_reproduced': True,
            'independent_hungarian_objective_matches_exhaustive_enumeration': True,
            'base_tie_explicitly_preserved': True, 'src_unique_semantic_assignment_verified': True},
        'base_optimal_count': 2, 'src_optimal_count': 1,
        'case_search_receipt_sha256': sha(HERE / 'CASE_SEARCH_RECEIPT.json'),
        'selected_case_sha256': sha(HERE / 'SELECTED_CASE.json'), 'script_sha256': sha(Path(__file__)),
        'formal_mot_metrics': False}
    (HERE / 'INDEPENDENT_CASE_AUDIT.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps({'status': audit['status'], 'base_optima': analysis['base']['optima'],
        'src_optima': analysis['src']['optima'], 'base_hungarian': analysis['base']['independent_hungarian_pairs']}))


if __name__ == '__main__':
    main()
