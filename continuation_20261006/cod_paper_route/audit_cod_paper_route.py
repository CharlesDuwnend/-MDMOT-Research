#!/usr/bin/env python3
"""Independent, read-only audit of the train-only MDMT-COD paper route.

This audit intentionally does not rerun the source evaluator or read val/test.
It checks provenance/legal boundaries and recomputes aggregate diagnostics from
the frozen v2 pair summaries/per-pair table.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path('/home/chenhc/mdmt_cod_revision_audit_20260926/artifacts_v2')
OUT = Path('/home/chenhc/claude_try_MDMOT/continuation_20261006/cod_paper_route')
EXPECTED_PAIRS = ['23','25','27','28','29','30','32','39','42','44','45','50','51','53','54','58','63','64','65','66','69','70','74','76','78']
EXPECTED = {
    'events': 8311,
    'candidate_recall': 0.9918145908986111,
    'given_candidate_top1': 0.09270802006988602,
    'given_candidate_mrr': 0.2149330411381026,
    'new_false_commits': 792,
    'observed_persistent_false_commits': 740,
    'future_wrong_coobserved_edge_frames': 78624,
    'causal_lag4_precision': 0.13691840079393658,
    'causal_lag4_recall': 0.033168367056211186,
    'causal_lag4_f1': 0.04924721166535837,
}

def finite(x):
    return isinstance(x, (int, float)) and math.isfinite(float(x))

def close(a, b, tol=1e-12):
    return finite(a) and abs(float(a)-float(b)) <= tol

def sha256(p: Path):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    manifest = json.loads((ROOT/'manifest.json').read_text())
    summary = json.loads((ROOT/'summary.json').read_text())
    checks = []
    def check(name, value, detail):
        checks.append({'name': name, 'pass': bool(value), 'detail': detail})

    # Round 1: source, split, and claim legality.
    check('R1_manifest_complete', manifest.get('status') == 'complete', manifest.get('status'))
    check('R1_train_only_split', manifest.get('split') == 'train' and manifest.get('val_access') is False and manifest.get('test_access') is False,
          {'split': manifest.get('split'), 'val_access': manifest.get('val_access'), 'test_access': manifest.get('test_access')})
    check('R1_no_training_or_gpu', manifest.get('training') is False and manifest.get('gpu_used') is False,
          {'training': manifest.get('training'), 'gpu_used': manifest.get('gpu_used')})
    check('R1_source_unchanged', manifest.get('source_files_unchanged') is True, manifest.get('source_files_unchanged'))
    check('R1_all_25_pair_files', all((ROOT/'pair_summaries'/f'{p}.json').is_file() for p in EXPECTED_PAIRS) and len(manifest.get('pairs', [])) == 25,
          {'manifest_pairs': len(manifest.get('pairs', [])), 'missing': [p for p in EXPECTED_PAIRS if not (ROOT/'pair_summaries'/f'{p}.json').is_file()]})
    check('R1_no_formal_mot_metrics', summary.get('formal_oracle_IDF1_AssA_HOTA_MOTA') == 'NOT_RUN', summary.get('formal_oracle_IDF1_AssA_HOTA_MOTA'))

    # Round 2: independent aggregate replay from per-pair JSON files.
    rows = []
    for p in EXPECTED_PAIRS:
        d = json.loads((ROOT/'pair_summaries'/f'{p}.json').read_text())
        rows.append(d)
    events = sum(int(d['events']) for d in rows)
    def macro(key): return sum(float(d[key]) for d in rows) / len(rows)
    persistent_pairs = sum(int(d['observed_persistent_false_commits']) > 0 for d in rows)
    false_commits = sum(int(d['new_false_commits']) for d in rows)
    persistent = sum(int(d['observed_persistent_false_commits']) for d in rows)
    future_frames = sum(int(d['future_wrong_coobserved_edge_frames']) for d in rows)
    lag4 = [d['paired_link_diagnostics']['causal_lag4'] for d in rows]
    lag4_keys = ['precision', 'recall', 'f1']
    lag4_macro = {k: sum(float(d[k]) for d in lag4)/len(lag4) for k in lag4_keys}
    check('R2_events_recomputed', events == EXPECTED['events'], {'recomputed': events, 'summary': summary.get('events')})
    check('R2_pair_macro_candidate_recall', close(macro('candidate_recall'), EXPECTED['candidate_recall']), macro('candidate_recall'))
    check('R2_pair_macro_given_top1', close(macro('given_candidate_top1'), EXPECTED['given_candidate_top1']), macro('given_candidate_top1'))
    check('R2_pair_macro_given_mrr', close(macro('given_candidate_mrr'), EXPECTED['given_candidate_mrr']), macro('given_candidate_mrr'))
    check('R2_false_commit_totals', false_commits == EXPECTED['new_false_commits'] and persistent == EXPECTED['observed_persistent_false_commits'], {'new_false_commits': false_commits, 'persistent': persistent})
    check('R2_persistent_pair_coverage', persistent_pairs == 25, {'pairs_with_persistent_false_commits': persistent_pairs})
    check('R2_future_frames_recomputed', future_frames == EXPECTED['future_wrong_coobserved_edge_frames'], future_frames)
    check('R2_causal_link_macro', all(close(lag4_macro[k], EXPECTED[f'causal_lag4_{k}']) for k in lag4_keys), lag4_macro)
    check('R2_all_numeric_finite', all(finite(d[k]) for d in rows for k in ['candidate_recall','given_candidate_top1','given_candidate_mrr']), True)

    # Round 3: falsifiable paper boundary, not a performance claim.
    check('R3_candidate_coverage_not_primary_bottleneck', macro('candidate_recall') > 0.95 and macro('given_candidate_top1') < 0.20,
          {'candidate_recall': macro('candidate_recall'), 'given_candidate_top1': macro('given_candidate_top1')})
    check('R3_persistent_error_is_cross_pair', persistent_pairs == 25, persistent_pairs)
    check('R3_route_is_diagnostic_only', summary.get('candidate_method_status') == 'HOLD_NOT_AUTHORIZED_FOR_TRAINING' and summary.get('interpretation','').startswith('Error prevalence is descriptive'),
          {'candidate_method_status': summary.get('candidate_method_status'), 'interpretation': summary.get('interpretation')})
    check('R3_no_official_selection', manifest.get('val_access') is False and manifest.get('test_access') is False and summary.get('formal_oracle_IDF1_AssA_HOTA_MOTA') == 'NOT_RUN', True)
    check('R3_source_receipt_present', (ROOT/'PREDICTIONS_FROZEN_BEFORE_XML.json').is_file() and (ROOT/'per_pair.csv').is_file(), True)

    result = {
        'status': 'PASS_COD_PAPER_ROUTE_AUDIT_3X3',
        'route_status': 'KEEP_PROTOCOL_DIAGNOSIS_ROUTE_NO_NEW_ALGORITHM',
        'source_root': str(ROOT),
        'rounds': {
            'R1_source_legal': [c for c in checks if c['name'].startswith('R1_')],
            'R2_aggregate_replay': [c for c in checks if c['name'].startswith('R2_')],
            'R3_claim_boundary': [c for c in checks if c['name'].startswith('R3_')],
        },
        'recomputed': {'pairs': len(rows), 'events': events, 'candidate_recall': macro('candidate_recall'), 'given_candidate_top1': macro('given_candidate_top1'), 'given_candidate_mrr': macro('given_candidate_mrr'), 'new_false_commits': false_commits, 'observed_persistent_false_commits': persistent, 'pairs_with_persistent_false_commits': persistent_pairs, 'future_wrong_coobserved_edge_frames': future_frames, 'paired_link_causal_lag4': lag4_macro},
        'claim_boundary': ['train-only diagnostics', 'no learned module trained', 'no val/test access', 'no IDF1/AssA/HOTA/MOTA claim', 'no SOTA or first claim', 'no authorization for official evaluation'],
        'all_checks_pass': all(c['pass'] for c in checks),
    }
    if not result['all_checks_pass']:
        result['status'] = 'FAIL_COD_PAPER_ROUTE_AUDIT'
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'PAPER_READINESS_AUDIT.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    receipt = {'audit_sha256': sha256(OUT/'PAPER_READINESS_AUDIT.json'), 'source_summary_sha256': sha256(ROOT/'summary.json'), 'source_manifest_sha256': sha256(ROOT/'manifest.json')}
    (OUT/'AUDIT_RECEIPT.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
