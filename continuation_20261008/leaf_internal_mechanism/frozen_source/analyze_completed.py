"""Report all predeclared complete cells with paired clip/flight deltas."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
METRICS = ['mota', 'idf1', 'num_false_positives', 'num_misses',
           'num_switches', 'num_fragmentations']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    manifest = json.loads((ROOT/'manifest.json').read_text())
    receipt = json.loads((ROOT/'artifacts/receipt.json').read_text())
    assert receipt['status'] == 'PASS_PREDECLARED_ONLINE_MECHANISM_ADDENDUM_COMPLETE'
    assert len(receipt['arms']) == len(manifest['arms']) == 7
    assert (ROOT/'artifacts/launcher.exit').read_text().strip() == '0'
    metrics, hashes = {}, {}
    for arm in manifest['arms']:
        name = arm['name']
        output = ROOT/'YOLOX_outputs'/name
        r = json.loads((output/'receipt.json').read_text())
        assert r['status'] == 'PASS_SCORED_ARM_COMPLETE' and r['sequence_count'] == 17 and r['frames'] == 6635
        path = output/'tracking_metrics.json'
        m = json.loads(path.read_text())
        assert sha(path) == r['metrics_sha256']
        assert all(sha(output/'track_res'/(s+'.txt')) == h for s, h in m['result_files_sha256'].items())
        assert m['result_files_sha256'] == r['result_sha256']
        metrics[name] = m
        hashes[name] = sha(path)
    pairs = [
        ('mask_without_penalty', 'C10_mask_only', 'C00_nativecost_reference'),
        ('penalty_without_mask', 'C01_penalty_only', 'C00_nativecost_reference'),
        ('penalty_with_mask', 'C11_full_reference', 'C10_mask_only'),
        ('mask_with_penalty', 'C11_full_reference', 'C01_penalty_only'),
        ('selective_with_fusion', 'C11_full_reference', 'E10_fusion_no_selective'),
        ('paired_hierarchy_minus_flat', 'H_paired_hierarchical', 'H_paired_flat'),
    ]
    rows, flights, contrasts = [], defaultdict(list), {}
    for contrast, treatment, control in pairs:
        a, b = metrics[treatment]['metrics'], metrics[control]['metrics']
        contrasts[contrast] = {
            'treatment': treatment, 'control': control,
            'delta_mota_pp': 100*(a['OVERALL']['mota']-b['OVERALL']['mota']),
            'delta_idf1_pp': 100*(a['OVERALL']['idf1']-b['OVERALL']['idf1']),
            **{k: a['OVERALL'][k]-b['OVERALL'][k] for k in METRICS[2:]},
        }
        for sequence in manifest['sequence_frames']:
            row = {'contrast': contrast, 'sequence': sequence,
                   'flight_prefix': sequence.split('_')[0],
                   'delta_mota_pp': 100*(a[sequence]['mota']-b[sequence]['mota']),
                   'delta_idf1_pp': 100*(a[sequence]['idf1']-b[sequence]['idf1']),
                   **{k: a[sequence][k]-b[sequence][k] for k in METRICS[2:]}}
            rows.append(row)
            flights[(contrast, row['flight_prefix'])].append(row)
    with (ROOT/'artifacts/paired_sequence_deltas.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    flight_rows = []
    for (contrast, flight), group in sorted(flights.items()):
        flight_rows.append({'contrast': contrast, 'flight_prefix': flight, 'clips': len(group),
                           'mean_clip_delta_mota_pp': sum(x['delta_mota_pp'] for x in group)/len(group),
                           'mean_clip_delta_idf1_pp': sum(x['delta_idf1_pp'] for x in group)/len(group),
                           **{k: sum(x[k] for x in group) for k in METRICS[2:]}})
    with (ROOT/'artifacts/paired_flight_descriptive_deltas.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(flight_rows[0])); writer.writeheader(); writer.writerows(flight_rows)
    summary = {'status': 'VERIFIED_COMPLETE_RESULT_AND_PAIRED_CONTRASTS',
               'metric_sha256': hashes, 'contrasts': contrasts,
               'sequence_count': 17, 'flight_count': len(set(s.split('_')[0] for s in manifest['sequence_frames'])),
               'analysis_scope': 'Closed-loop paired conditional contrasts; flight table is mean clip deltas, not pooled flight IDF1',
               'frame_level_significance': False, 'testdev_selection': False,
               'current_primary_method_changed': False}
    (ROOT/'artifacts/paired_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    lines = ['# Completed predeclared internal mechanism results', '',
             'All seven online arms are complete. Current F00 and native-cost reference predictions pass 17-file parity. Detector/ReID input hashes and evaluation contracts match. No test-dev winner selection was performed.', '',
             '| Arm | MOTA | IDF1 | FP | FN | IDs | FM |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for arm in manifest['arms']:
        name = arm['name']; m = metrics[name]['metrics']['OVERALL']
        lines.append('| %s | %.4f | %.4f | %s | %s | %s | %s |' %
                     (name, 100*m['mota'], 100*m['idf1'], *[m[k] for k in METRICS[2:]]))
    lines += ['', '| Conditional contrast | MOTA delta (pp) | IDF1 delta (pp) | FP delta | FN delta | IDs delta |',
              '|---|---:|---:|---:|---:|---:|']
    for name, d in contrasts.items():
        lines.append('| %s | %+.4f | %+.4f | %+d | %+d | %+d |' %
                     (name, d['delta_mota_pp'], d['delta_idf1_pp'],
                      d['num_false_positives'], d['num_misses'], d['num_switches']))
    lines += ['', 'The paired refit hierarchy/flat comparison shares a fitting scaler and event weights; it is separate from canonical F00\'s original binary-specific recipe. Selected calibration figures and head NLL diagnostics are mechanism evidence, not new aggregate tracking scores. Zero and negative contrasts remain in the report. Existing hard-update/recurrence null controls and the birth-gate contribution remain part of the author-facing evidence inventory.', '']
    (ROOT/'artifacts/RESULTS_FOR_AUTHOR.md').write_text('\n'.join(lines))
    print(json.dumps(contrasts, indent=2))


if __name__ == '__main__':
    main()
