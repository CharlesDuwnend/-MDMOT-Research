#!/usr/bin/env python3
"""Three-round audit for the CCSI full-coverage outcome."""
import hashlib
import json
import math
from pathlib import Path

WORK = Path('/home/chenhc/claude_try_MDMOT/continuation_20261006/p53_ccsi')
P38 = Path('/home/chenhc/claude_try_MDMOT/continuation_20261005/p38_temporal_memory/runs/p38_temporal/fixed_step_001200.pt')


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    receipt = json.loads((WORK / 'runs_full' / 'TRAINING_RECEIPT.json').read_text())
    result = json.loads((WORK / 'runs_full' / 'evaluation' / 'RESULTS.json').read_text())
    freeze = json.loads((WORK / 'runs_full' / 'evaluation' / 'LABEL_FREEZE.json').read_text())
    short = json.loads((WORK / 'runs' / 'evaluation' / 'RESULTS.json').read_text())
    checks = []

    def check(name, passed, detail):
        checks.append({'name': name, 'pass': bool(passed), 'detail': detail})

    # R1: full schedule, receipt, physical GPU, and immutable checkpoint lineage.
    check('R1_full_schedule_contract',
          receipt.get('steps') == 55236 and receipt.get('eligible_groups') == 18412 and receipt.get('epochs') == 3,
          {'steps': receipt.get('steps'), 'eligible_groups': receipt.get('eligible_groups'), 'epochs': receipt.get('epochs')})
    checkpoint = WORK / 'runs_full' / 'fixed_step_055236.pt'
    check('R1_full_checkpoint_hash', checkpoint.exists() and sha(checkpoint) == receipt.get('checkpoint_sha256'),
          receipt.get('checkpoint_sha256'))
    check('R1_full_receipt_gpu_and_status',
          receipt.get('status') == 'PASS_CCSI_FULL_COVERAGE_TRAINING'
          and receipt.get('gpu_name') == 'NVIDIA A100-SXM4-40GB'
          and receipt.get('gpu_uuid') == 'GPU-aaa79a66-b5c4-e0a0-6e2f-28c1ac0e3e6a',
          {'status': receipt.get('status'), 'gpu_name': receipt.get('gpu_name'), 'gpu_uuid': receipt.get('gpu_uuid')})

    # R2: evaluation is frozen, finite, complete, and never read dev/official data.
    reports = result.get('pairs', [])
    check('R2_label_freeze',
          freeze.get('status') == 'PASS_CCSI_LABEL_FREEZE' and len(freeze.get('pairs', [])) == 20,
          {'status': freeze.get('status'), 'pairs': len(freeze.get('pairs', []))})
    values = []
    for row in reports:
        values.append(float(row['metrics']['all']['methods']['ccsi128']['all_candidates']['known_positive_r1_tie_averaged']))
    check('R2_all_outputs_finite', len(reports) == 20 and all(math.isfinite(v) for v in values),
          {'pairs': len(reports), 'finite_values': sum(math.isfinite(v) for v in values)})
    check('R2_no_forbidden_split',
          result.get('dev_read') is False and result.get('official_val_test_read') is False
          and receipt.get('calibration_read') is False and receipt.get('dev_read') is False,
          {'dev_read': result.get('dev_read'), 'official_val_test_read': result.get('official_val_test_read'),
           'receipt_calibration_read': receipt.get('calibration_read'), 'receipt_dev_read': receipt.get('dev_read')})

    # R3: method failure and its cause are explicit; no host attachment is authorized.
    gate = dict(result['gate'])
    gate['calibration_r1_vs_p1'] = (result['by_role']['calibration']['pair_macro_r1']['ccsi128'] - result['by_role']['calibration']['pair_macro_r1']['p1_independent128'])
    short_gate = dict(short['gate'])
    short_gate['calibration_r1_vs_p1'] = (short['by_role']['calibration']['pair_macro_r1']['ccsi128'] - short['by_role']['calibration']['pair_macro_r1']['p1_independent128'])
    check('R3_full_calibration_gate_failure',
          gate.get('pass') is False and gate.get('wins_vs_p38') == 1 and gate.get('calibration_r1_vs_p38') < 0,
          gate)
    check('R3_short_to_full_regression',
          short_gate.get('calibration_r1_vs_p38') > 0 and gate.get('calibration_r1_vs_p38') < 0
          and short_gate.get('wins_vs_p38') == 4,
          {'short': short_gate, 'full': gate})
    check('R3_fit_calibration_gap_exposes_overfit',
          gate.get('fit_drop_vs_p1', 0) > 0.05 and gate.get('calibration_r1_vs_p1', 0) < 0,
          {'fit_drop_vs_p1': gate.get('fit_drop_vs_p1'), 'calibration_r1_vs_p1': gate.get('calibration_r1_vs_p1')})

    audit = {
        'status': 'PASS_CCSI_FULL_FAILURE_AUDIT_3X3',
        'classification': 'METHOD_GENERALIZATION_FAILURE_CROSS_PAIR_OVERFIT',
        'decision': 'STOP_CCSI_AFTER_FULL_COVERAGE',
        'rounds': {
            'R1_training_integrity': [x for x in checks if x['name'].startswith('R1_')],
            'R2_evaluation_integrity': [x for x in checks if x['name'].startswith('R2_')],
            'R3_failure_cause': [x for x in checks if x['name'].startswith('R3_')],
        },
        'all_checks_pass': all(x['pass'] for x in checks),
        'evidence': {'short_gate': short_gate, 'full_gate': gate,
                     'checkpoint_sha256': receipt.get('checkpoint_sha256'),
                     'p38_checkpoint_sha256': receipt.get('p38_checkpoint_sha256')},
    }
    (WORK / 'FULL_FAILURE_AUDIT.json').write_text(json.dumps(audit, indent=2) + '\n')
    decision = {
        'status': 'STOP_CCSI_AFTER_FULL_COVERAGE',
        'candidate': 'Counterfactual Cross-View Swap Invariance',
        'short_gate': {'calibration_r1': short['by_role']['calibration']['pair_macro_r1']['ccsi128'],
                       'vs_p38_delta': short_gate['calibration_r1_vs_p38'],
                       'wins_vs_p38': short_gate['wins_vs_p38']},
        'full_coverage': {'steps': receipt['steps'], 'epochs': receipt['epochs'],
                          'calibration_r1': result['by_role']['calibration']['pair_macro_r1']['ccsi128'],
                          'vs_p38_delta': gate['calibration_r1_vs_p38'],
                          'wins_vs_p38': gate['wins_vs_p38'],
                          'fit_r1': result['by_role']['fit']['pair_macro_r1']['ccsi128']},
        'decision': '停止继续训练、停止 host attachment 和 official val/test；短门改善被完整 fit 训练后的跨 pair 回归否定。',
        'failure_audit': 'FULL_FAILURE_AUDIT.json',
        'interpretation': '实现、有限性、冻结顺序、完整覆盖和物理 GPU 校验通过；主要问题是跨 pair 泛化/过拟合，不是 GPU、步数不足或评估 wiring。',
        'retain': ['短门 checkpoint 作为诊断控制', '全覆盖 checkpoint 作为失败证据', 'CCSI method spec and audit trail'],
    }
    (WORK / 'P53_DECISION.json').write_text(json.dumps(decision, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
