#!/usr/bin/env python3
"""Freeze P30 host comparison and write the progression decision."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAID = Path('/raid/datasets/chc_data/claude_try_MDMOT_p30_full_host_20261005')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def files(root, pattern='*'):
    out = {}
    for p in sorted(Path(root).glob(pattern)):
        if p.is_file():
            out[str(p)] = {'bytes': p.stat().st_size, 'sha256': sha(p)}
    return out


def main():
    host = json.loads((RAID / 'host/HOST_SCORE.json').read_text())
    by = {x['mode']: x for x in host['results']}
    cmp = host['comparison']
    native = by['native']['macro']
    fixed = by['past_fixed_offsets']['macro']
    detector_hash = json.loads((RAID / 'calibration/overrides/native/RECEIPT.json').read_text())['detector_state_sha256']
    fixed_receipt = json.loads((RAID / 'calibration/overrides/past_fixed_offsets/RECEIPT.json').read_text())
    valid_logs = {
        'native_exit': (HERE / 'host_native_v2.exit').read_text().strip(),
        'fixed_exit': (HERE / 'host_fixed_v2.exit').read_text().strip(),
        'native_model_mismatch_warning': 'mismatch' in (HERE / 'host_native_v2.log').read_text().lower(),
        'fixed_model_mismatch_warning': 'mismatch' in (HERE / 'host_fixed_v2.log').read_text().lower(),
    }
    assert valid_logs['native_exit'] == valid_logs['fixed_exit'] == '0'
    assert not valid_logs['native_model_mismatch_warning'] and not valid_logs['fixed_model_mismatch_warning']
    assert fixed_receipt['detector_state_sha256'] == detector_hash
    decision = {
        'status': 'STOP_P30_FIXED_HOST_NOT_COMPETITIVE',
        'decision_codes': ['STOP_FIXED_HOST_NOT_COMPETITIVE', 'NO_OFFICIAL_VAL_TEST_ACCESS', 'NO_NOVELTY_CLAIM'],
        'protocol': {
            'dataset': 'MDMT', 'pairs': ['27', '32', '42', '64', '65'],
            'role': 'calibration', 'full_continuous_frames': 3900, 'views': 10,
            'first_frame': 'official MIA XML bbox/ID initialization retained',
            'tracker': 'P26 MIA ByteTrack/Kalman/geometric cross-view host retained',
            'detector': 'frozen AutoAssign R50-FPN epoch60; cap300 native config',
            'official_val_or_test': False,
        },
        'native_macro': native,
        'past_fixed_offsets_macro': fixed,
        'fixed_minus_native': cmp['fixed_minus_native'],
        'fixed_minus_native_percentage_points': cmp['fixed_minus_native_percentage_points'],
        'pair_deltas': cmp['pair_deltas'],
        'critical_failure': 'fixed loses MDA by 2.083925 pp overall and by 11.009193 pp on pair 42; IDF1 and MOTA also decline.',
        'next_action': 'stop P29 fixed residual as a paper candidate; preserve the diagnostic artifacts and return to mechanism audit before any new training.',
        'valid_host_runs': valid_logs,
        'detector_state_sha256': detector_hash,
        'feature_receipt_sha256': sha(HERE / 'calibration/FEATURE_RECEIPT.json'),
        'flow_receipt_sha256': sha(HERE / 'calibration/FLOW_RECEIPT.json'),
        'override_receipts': {
            'native': sha(RAID / 'calibration/overrides/native/RECEIPT.json'),
            'past_fixed_offsets': sha(RAID / 'calibration/overrides/past_fixed_offsets/RECEIPT.json'),
        },
        'mango_eval_sha256': by['native']['mango_sha256'],
        'score_script_sha256': by['native']['script_sha256'],
    }
    (RAID / 'host/DECISION.json').write_text(json.dumps(decision, indent=2, ensure_ascii=False) + '\n')
    manifest = {
        'status': 'FROZEN_P30_HOST_COMPARISON',
        'decision_sha256': sha(RAID / 'host/DECISION.json'),
        'host_score_sha256': sha(RAID / 'host/HOST_SCORE.json'),
        'valid_native_outputs': files(RAID / 'host/native/json/p30_native', '*.json'),
        'valid_fixed_outputs': files(RAID / 'host/fixed/json/p30_fixed', '*.json'),
        'score_summaries': {
            'native': files(RAID / 'host/native/score', '*.json'),
            'fixed': files(RAID / 'host/fixed/score', '*.json'),
        },
        'operator_diagnostic': {
            'path': str(RAID / 'host/P30_OPERATOR_DIAGNOSTIC.json'),
            'sha256': sha(RAID / 'host/P30_OPERATOR_DIAGNOSTIC.json'),
        },
    }
    (RAID / 'host/HOST_MANIFEST.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    report = f'''# P30 full continuous host result

P30 sealed 3,900 frames across MDMT calibration pairs 27/32/42/64/65, generated
native FPN and 3-lag causal flow caches, and injected frozen detector arrays into
the official P26 MIA host. The two valid host runs used the AutoAssign config,
epoch-60 checkpoint, cap300 detector protocol, first-frame XML initialization,
ByteTrack/Kalman state, and MIA cross-view update code. They ran on the frozen
arrays through the strict P30 runtime; both exit markers are zero and neither log
contains a model-mismatch warning.

| arm | MDA (%) | IDF1 (%) | MOTA (%) |
|---|---:|---:|---:|
| native | {100*native['MDA']:.6f} | {100*native['idf1']:.6f} | {100*native['mota']:.6f} |
| P29 past fixed offsets | {100*fixed['MDA']:.6f} | {100*fixed['idf1']:.6f} | {100*fixed['mota']:.6f} |
| fixed - native (pp) | {100*cmp['fixed_minus_native']['MDA']:.6f} | {100*cmp['fixed_minus_native']['idf1']:.6f} | {100*cmp['fixed_minus_native']['mota']:.6f} |

The fixed residual loses 2.083925 percentage points MDA overall. Pair 27 improves
by 2.680949 points, but pair 42 collapses by 11.009193 points; IDF1 and MOTA also
decline in the macro mean. This falsifies the P29 host candidate under the frozen
protocol. The earlier CARAFE-configured attempt is archived under
`host/attempts/carafe_config_v1/REJECTED.json` and is excluded from all scores.

The array-only operator diagnostic confirms that all first eight warm-up frames are
bit-identical to native. On pair 42, the later fixed arrays change the mean detection
count by +3.36/+1.73 per view, with mean matched IoU 0.959/0.952 and mean score
changes 0.027/0.032. These modest detector perturbations are enough to destabilize
the MIA cross-view association on that pair; the diagnostic reads no XML or labels.
It supports stopping the operator rather than attributing the failure to a missing
training seed.

This is calibration evidence only. It does not authorize official-val/test claims,
additional training, or a novelty claim. The detailed machine-readable decision is
`host/DECISION.json`; hashes and every valid JSON output are in `host/HOST_MANIFEST.json`.
'''
    (HERE / 'P30_REPORT.md').write_text(report)
    print(json.dumps({'status': decision['status'], 'fixed_minus_native_pp': decision['fixed_minus_native_percentage_points']}, indent=2))


if __name__ == '__main__':
    main()
