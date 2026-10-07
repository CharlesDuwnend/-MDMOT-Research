"""Rehash actual completed smoke evidence before allowing full inference."""
import datetime
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'tools'))

from analyze_completed import verify_input_ledger
from run_mechanism import audit_runtime, check_frozen, sha, write


def main():
    manifest = json.loads((ROOT/'manifest.json').read_text())
    check_frozen(manifest)
    assert len(manifest['arms']) == 9 and manifest['smoke_frames'] == 200
    identity_path = ROOT/'artifacts/INPUT_IDENTITY_REPAIR_CPU.json'
    identity = json.loads(identity_path.read_text())
    assert identity['status'] == 'CPU_INPUT_IDENTITY_REPAIR_GATE_PASS'
    assert identity['metadata_contracts']['exit_code'] == 0
    assert identity['metadata_contracts']['tests_run'] == 9
    assert identity['existing_cpu_contracts']['exit_code'] == 0
    assert identity['existing_cpu_contracts']['tests_run'] == 29
    assert identity['real_frame_identity_recheck']['frames_checked'] == 9
    assert identity['real_frame_identity_recheck']['v2_wrong_metadata_detections'] == 0
    # The identity tests explicitly do not validate the oracle. Check the
    # separate real-head reference tests and their actual provider hashes.
    reference_path = ROOT/'artifacts/CORRECTED_REFERENCE_CPU.json'
    reference = json.loads(reference_path.read_text())
    assert reference['status'] == 'PASS_NINE_CORRECTED_REFERENCE_CPU_CONTRACTS'
    assert len(reference['checks']) == 9 and all(c['status'] == 'PASS' for c in reference['checks'])
    assert not reference['cuda_initialized'] and not reference['formal_online_accuracy_evidence']
    for p, expected in reference['source_sha256'].items():
        assert sha(p) == expected, 'REFERENCE_CPU_SOURCE_CHANGED: ' + p
    smoke_path = ROOT/'artifacts/smoke_receipt.json'
    smoke = json.loads(smoke_path.read_text())
    assert smoke['status'] == 'PASS_CORRECTED_200_FRAME_ENGINEERING_SMOKE'
    assert smoke['manifest_sha256'] == sha(ROOT/'manifest.json')
    exit_path = ROOT/'artifacts/smoke_launcher.exit'
    assert exit_path.read_text().strip() == '0'
    assert len(smoke['arms']) == 9
    evidence = {str(identity_path): sha(identity_path), str(reference_path): sha(reference_path),
                str(smoke_path): sha(smoke_path), str(exit_path): sha(exit_path)}
    baseline_inputs = None
    for arm, record in zip(manifest['arms'], smoke['arms']):
        name = 'SMOKE_' + arm['name']
        assert record['name'] == name and record['status'] == 'PASS_ENGINEERING_SMOKE'
        assert record['sequence_count'] == 1 and record['frames'] == 200
        assert record['gpu_uuid'] == record['same_process_gpu_verification']['cuda_uuid'] == manifest['gpu_uuid']
        assert record['gt_guard']['status'] == 'PASS_NO_GT_READ'
        assert record['gt_guard']['blocked_gt_attempts'] == 0
        sequence = manifest['smoke_sequence']
        output = ROOT/'YOLOX_outputs'/name
        path = output/'track_res'/(sequence+'.txt')
        assert sha(path) == record['result_sha256'][sequence]
        report_path = Path(str(path)+'.runtime.json')
        report = json.loads(report_path.read_text())
        audit_runtime({sequence: report}, arm, smoke=True)
        ledger = verify_input_ledger(report, sequence, 200)
        assert report['input_stream_sha256'] == record['input_sha256'][sequence]
        if baseline_inputs is None:
            baseline_inputs = ledger
        else:
            assert ledger == baseline_inputs, 'SMOKE_INPUT_LEDGER_CHANGED'
        assert ledger[185]['detector_sha256'] == '15d69e475607041a541b2d1245bcc7ebb722993f2bf19dbadbeffa824dafa585', \
            'CONFIRMED_DUPLICATE_FRAME_NOT_REPLAYED'
        evidence[str(report_path)] = sha(report_path)
        evidence[str(output/'receipt.json')] = sha(output/'receipt.json')
    for relative in ['belief/runtime.py', 'belief/reference_oracle.py',
                     'tools/capture_belief_evidence.py', 'belief/online.py',
                     'tests/test_input_identity.py', 'tests/test_corrected_reference.py']:
        evidence[str(ROOT/relative)] = sha(ROOT/relative)
    gate = {'status': 'RESOLVED_DETECTION_METADATA_IDENTITY', 'sealed': True,
            'manifest_sha256': sha(ROOT/'manifest.json'),
            'sealed_utc': datetime.datetime.utcnow().isoformat()+'Z',
            'evidence_sha256': evidence, 'real_problem_frame_replayed': 186,
            'smoke_sequence': manifest['smoke_sequence'], 'smoke_arms': 9,
            'smoke_frames_per_arm': 200, 'formal_MOT_scores_produced': False,
            'limits': 'CPU and engineering smoke validity; full 17-sequence arm validity is checked during execution and final analysis.'}
    assert not (ROOT/'artifacts/INPUT_IDENTITY_AUDIT_GATE.json').exists()
    write(ROOT/'artifacts/INPUT_IDENTITY_AUDIT_GATE.json', gate)
    print(json.dumps({'status': gate['status'], 'manifest_sha256': gate['manifest_sha256'],
                      'evidence_files': len(evidence)}), flush=True)


if __name__ == '__main__':
    main()
