"""Preregister three frozen-head interventions before any new inference."""
import copy, datetime, json, os, pickle, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['BELIEF_HOST'] = os.environ['UAVDT_ONLINE_HOST'] = str(ROOT / 'host')
from fit_paired_anchor import sha, objsha
from belief.structural import DeletedHeadAnchor, WithoutSourceCues
PARENT = Path('/home/chenhc/leaf_egia_reduced_pure_20261008_v1')

def exclusive(name, value):
    with (ROOT / name).open('x') as f:
        json.dump(value, f, indent=2, sort_keys=True); f.write('\n')

parent = json.loads((PARENT / 'manifest.json').read_text())
for p, h in parent['frozen_files_sha256'].items():
    assert sha(p) == h
assert json.loads((PARENT/'review/results/FINAL_AUDIT.json').read_text())['new_full_runs'] == 4
base = pickle.loads((ROOT/'models/R00_full.pkl').read_bytes())
assert not base['egia_fusion_policy'] and base['selective_birth_policy']
for key in ['safe_native_incumbent', 'owner_selective_birth_policy',
            'decision_specific_birth_policy', 'asymmetric_selective_birth_policy']:
    assert not base.get(key, False)
GPU = {0: 'GPU-2a55197e-2c19-45ff-cf33-13031951f9d0',
       1: 'GPU-1b297aba-ae7e-e326-5903-476f1bb683d7',
       3: 'GPU-aaa79a66-b5c4-e0a0-6e2f-28c1ac0e3e6a'}
specs = [('A04_coverage_only_selective', 'coverage_only', True, 0, (2,)),
         ('A05_two_heads_argmax', 'delete_source_cues', False, 1, ()),
         ('A06_full_argmax', 'full', False, 3, ())]
arms = []
for name, mode, selective, index, forbidden in specs:
    b = copy.deepcopy(base)
    if mode != 'full':
        s = WithoutSourceCues(b['source_model'])
        if mode == 'coverage_only':
            s.anchor = DeletedHeadAnchor(s.anchor, 'foreground')
            s.forbidden_classes = forbidden
            s.structural_mode = mode
        b['source_model'] = s
    b['selective_birth_policy'] = selective
    allowed = {'source_model', 'selective_birth_policy'}
    assert all(objsha(b[k]) == objsha(base[k]) for k in base if k not in allowed)
    p = ROOT/'models'/(name+'.pkl')
    with p.open('xb') as f: pickle.dump(b, f, protocol=4)
    arms.append(dict(name=name, structural_mode=mode, forbidden_classes=list(forbidden),
        control=mode, mode='full', runtime_arm='egia', src=False, mask=False, penalty=False,
        gate=False, birth_class_gate=0., use_egia=True, fusion=False, selective=selective,
        corrected_reference_oracle=False, bundle=str(p.relative_to(ROOT)), bundle_sha256=sha(p),
        gpu_physical_index=index, gpu_uuid=GPU[index], historical_reference_metrics=None))
refmetrics = str(PARENT/'YOLOX_outputs/R00_full/tracking_metrics.json')
reference = dict(parent['arms'][0], bundle='models/R00_full.pkl',
    name='REFERENCE_full_smoke', historical_reference_metrics=refmetrics)
reg = dict(status='PREREGISTERED_BEFORE_ANY_NEW_FURTHER_INFERENCE',
    created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    new_scores_seen=0, arms=arms, smoke_reference=reference,
    fixed=dict(src=False, birth_class_gate=0., birth_detector_score=.6, fusion=False),
    no_new_training=True, no_prior_replacement=True, no_testdev_selection=True,
    all_results_retained=True, no_new_figures=True, current_paper_modified=False,
    parent_root=str(PARENT), parent_manifest_sha256=sha(PARENT/'manifest.json'),
    reused_references=['R00_full', 'R03_no_source_cues'],
    paired_units='17 sequences, 14 flights; no independent-frame significance claim',
    outcomes=['MOTA','IDF1','FP','FN','IDs','FM','birth activations','actual head calls'],
    purpose='Complete coherence x selective 2x2 and isolate FG effect without coherence',
    redundant_controls='Single cue removed => zero coherence; covered class removed => no E-only authority',
    inference_GT_reads_forbidden=True)
exclusive('PREREGISTRATION.json', reg)
design = {k: parent[k] for k in ['data_root','gt_root','detector','fixed','sequence_frames',
    'smoke_sequence','smoke_frames','corrected_reference_sources']}
design.update(status='FURTHER_PURE_DESIGN_NOT_YET_FROZEN', parent_root=str(PARENT),
    arms=arms, smoke_reference=reference, full_reference_metrics=refmetrics,
    input_reference_receipt=str(PARENT/'YOLOX_outputs/R00_full/receipt.json'),
    smoke_input_reference_receipt=str(PARENT/'YOLOX_outputs/SMOKE_R00_full/receipt.json'),
    gpu_uuid=GPU[0], gpu_physical_index=0, testdev_parameter_selection=False)
exclusive('design.json', design)
print(json.dumps(dict(new_full_arms=3, new_scores_seen=0, SRC=False, gate=0., fusion=False)))
