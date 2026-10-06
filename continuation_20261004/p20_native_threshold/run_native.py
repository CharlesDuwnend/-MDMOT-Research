"""Reuse the verified P18 input reader; run only the isolated native host."""
import sys
sys.dont_write_bytecode = True
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import multiprocessing
import os
from pathlib import Path
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
HERE = Path(__file__).resolve().parent
OLD = Path('/home/chenhc/mdmot_research_20261002')
RUN_ROOT = Path('/raid/datasets/chc_data/mdmot_continuation_20261004/p20_runs')
PAIRS = ['27', '32', '42', '64', '65']
sys.path.insert(0, str(HERE))
from native_current_future_runtime import CurrentFutureRuntime


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def upstream():
    spec = importlib.util.spec_from_file_location('p20_upstream', OLD / 'p18_merge_verifier/run_host_merge.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_pair(pair, run_root, threshold):
    module = upstream()
    def factory(memory_threshold, memory_age, scorer=None):
        return CurrentFutureRuntime(memory_threshold, memory_age, same_frame_threshold=0.70)
    module.CurrentFutureRuntime = factory
    return module.run_pair(pair, run_root, 'anchor_p1', threshold, 0.70, 0, None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--memory-threshold', type=float, choices=[0.8918, 0.75], required=True)
    args = parser.parse_args()
    protocol = json.loads((HERE / 'PROTOCOL.json').read_text())
    for path, digest in protocol['frozen_sources'].items():
        assert sha(path) == digest
    tests = json.loads((HERE / 'TESTS.json').read_text())
    assert tests['status'] == 'PASS'
    threshold = args.memory_threshold
    tag = 'native_mem' + str(threshold) + '_sf0.70'
    run = RUN_ROOT / tag
    run.mkdir(parents=True, exist_ok=False)
    module = upstream()
    references, imported = module.check_frozen_references()
    module.load_encoder('anchor_p1')
    started = time.monotonic()
    results = {}
    with ProcessPoolExecutor(max_workers=2, mp_context=multiprocessing.get_context('spawn')) as pool:
        jobs = {pool.submit(run_pair, pair, str(run), threshold): pair for pair in PAIRS}
        for job in as_completed(jobs):
            results[jobs[job]] = job.result()
    assert set(results) == set(PAIRS)
    counts = {'pair_results': {p: results[p]['summary'] for p in PAIRS},
              'elapsed_seconds': time.monotonic() - started}
    dump(run / 'COUNTS.json', counts)
    code_paths = [Path(__file__), HERE / 'native_prefix_runtime.py', HERE / 'native_current_future_runtime.py',
                  OLD / 'p18_merge_verifier/run_host_merge.py']
    receipt = {'status': 'PREDICTIONS_FROZEN_BEFORE_GT', 'tag': tag, 'pairs': PAIRS,
               'arms': ['current_future_merge'], 'encoder_arm': 'anchor_p1',
               'encoder_checkpoint': {'path': str(module.CHECKPOINT), 'sha256': sha(module.CHECKPOINT)},
               'threshold': threshold, 'memory_threshold': threshold, 'same_frame_threshold': 0.70,
               'same_frame_implementation': 'native_raw_cosine', 'score_shift': 0,
               'gt_read': False, 'dev_access': False, 'official_val_access': False, 'test_access': False,
               'gpu_used': False, 'encoder_training': False, 'threshold_search': False,
               'scope': 'fixed 2x2 implementation correction on reused internal cal5; no novelty claim',
               'visual_memory_age': 30, 'lookahead': 0, 'output_delay': 0,
               'past_output_rewritten': False, 'counts': counts,
               'protocol_sha256': sha(HERE / 'PROTOCOL.json'), 'tests_sha256': sha(HERE / 'TESTS.json'),
               'code': [{'path': str(p), 'sha256': sha(p)} for p in code_paths],
               'upstream_references': references, 'upstream_imports': imported,
               'sources': [item for p in PAIRS for item in results[p]['sources']],
               'reader_auxiliary_dino_files': [item for p in PAIRS for item in results[p]['dino']],
               'files': [{'path': str(p), 'sha256': sha(p)} for p in sorted(run.rglob('*')) if p.is_file()],
               'completed_utc': datetime.now(timezone.utc).isoformat()}
    dump(run / 'PREDICTIONS_FROZEN.json', receipt)
    (run / 'runner.exit').write_text('0\n')
    print(json.dumps({'status': receipt['status'], 'tag': tag, 'seconds': counts['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
