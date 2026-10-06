"""Create an isolated, reviewable native-threshold patch from frozen sources."""
import difflib
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = Path('/home/chenhc/mdmot_research_20261002')


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('source contract drift: ' + old)
    return text.replace(old, new, 1)


def main():
    sources = {}
    for name, source in [('native_prefix_runtime.py', OLD / 'p1b/src/prefix_runtime.py'),
                         ('native_current_future_runtime.py', OLD / 'p1d/src/current_future_runtime.py')]:
        original = source.read_text()
        patched = replace_once(original,
            'def __init__(self, threshold, memory_age=30, scorer=None):',
            'def __init__(self, threshold, memory_age=30, scorer=None, same_frame_threshold=None):')
        if name == 'native_prefix_runtime.py':
            patched = replace_once(patched, '        self.scorer = scorer or CosineScorer()',
                '        self.scorer = scorer or CosineScorer()\n'
                '        self.same_frame_threshold = (float(threshold) if same_frame_threshold is None\n'
                '                                     else float(same_frame_threshold))')
            patched = replace_once(patched, "aa[0] != bb[0] and weights[i, j] >= self.threshold:",
                "aa[0] != bb[0] and weights[i, j] >= self.same_frame_threshold:")
            patched = replace_once(patched,
                '            for i, j, score in matched(weights, self.threshold):\n                a, b = left[i], right[j]',
                '            for i, j, score in matched(weights, self.same_frame_threshold):\n                a, b = left[i], right[j]')
        else:
            patched = replace_once(patched, 'from prefix_runtime import PrefixRuntime',
                                   'from native_prefix_runtime import PrefixRuntime')
            patched = replace_once(patched, 'super().__init__(threshold, memory_age, scorer)',
                                   'super().__init__(threshold, memory_age, scorer, same_frame_threshold)')
            patched = replace_once(patched,
                "from pathlib import Path\nimport sys\nsys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'p1b/src'))\n", '')
        target = HERE / name
        if target.exists():
            raise ValueError('refuse overwrite: ' + str(target))
        target.write_text(patched)
        (HERE / (name + '.patch')).write_text(''.join(difflib.unified_diff(
            original.splitlines(True), patched.splitlines(True), fromfile=str(source), tofile=str(target))))
        sources[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
    protocol = {
        'id': 'P20_NATIVE_THRESHOLD_IMPLEMENTATION_2X2',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'question': 'Does score shifting contaminate the isolated memory-threshold comparison?',
        'scope': 'MDMT official-train internal cal5 engineering correction, not a novel method',
        'pairs': ['27', '32', '42', '64', '65'],
        'factors': {'same_frame_implementation': ['shifted', 'native'], 'memory_threshold': [0.8918, 0.75]},
        'same_frame_threshold': 0.70,
        'encoder': 'anchor_p1 independent seed42 step4000',
        'native_operator': 'raw cosine weights; separate admissibility threshold; original max-weight matching',
        'reuse_shifted_runs': ['p19_cal_mem0.8918_sf0.70', 'p19_cal_mem0.75_sf0.70'],
        'native_execution_order': ['native_mem0.75_sf0.70', 'native_mem0.8918_sf0.70'],
        'unit': 'paired sequence; frames and detections are repeated observations, not independent replicates',
        'endpoints': ['emitted identity disagreement', 'route/count disagreement', 'CA_IDF1_conventional_pseudo', 'strict crosslinks'],
        'no_new_threshold_search': True, 'dev5_read': False, 'official_val_or_test_read': False,
        'pass_meaning': 'correctly isolates gates; neither numerical improvement nor method novelty is required',
        'frozen_sources': sources,
    }
    (HERE / 'PROTOCOL.json').write_text(json.dumps(protocol, indent=2) + '\n')


if __name__ == '__main__':
    main()
