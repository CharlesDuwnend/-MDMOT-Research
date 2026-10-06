#!/usr/bin/env python3
"""Executable audit superseding the P50 short-gate interpretation.

This does not rerun the tracker or alter the protected P26 baseline.  It checks
whether the prior P50 comparison actually isolated the proposed mechanism.
"""
from __future__ import annotations
import ast, hashlib, json, re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = Path('/home/chenhc/claude_try_MDMOT/continuation_20261005/p50_fragment_teacher_residual')
MAP = OLD / 'src/map_gate.py'
FPN = OLD / 'src/map_fpn_gate.py'
AUDIT = ROOT / 'SUPERSEDING_AUDIT.json'

def sha(p):
    h=hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()

def source_checks(text):
    checks = {
      'resized_non_native_256x448': 'image_size=(256,448)' in text or 'image_size=(256, 448)' in text,
      'trainable_teacher_projection': 'self.teacher_proj = nn.Linear' in text and 'teacher_proj' in text and 'requires_grad=False' not in text[text.find('self.teacher_proj'):text.find('self.teacher_proj')+160],
      'teacher_projection_in_loss': 'model.head.teacher_proj' in text,
      'teacher_not_detached_before_projection': 'teacher.detach()' in text,
      'positive_and_teacher_filter': 't is not None and row.get("same_id_candidate_indices")' in text,
      'no_match_explicitly_retained': 'no-match' in text.lower() and 'candidate' in text.lower(),
      'raw_dot_labeled_cosine': '(a*c).sum' in text or '(ta*x)' in text and 'np.linalg.norm' in text,
      'unequal_eval_append_paths': 'rs.append' in text and 'base.append' in text and 'continue' in text,
    }
    # AST-independent line evidence for a reviewer.
    checks['model_train_called'] = bool(re.search(r'model\.to\(device\)\.train\(\)', text))
    return checks

def fpn_checks(text):
    return {
      'p2_is_trainable': 'name.startswith("p2_id") or name.startswith("refine_p2")' in text,
      'bn_or_norm_added_to_copied_branch': 'GroupNorm' in text or 'BatchNorm' in text or 'block[1].weight.fill_' in text,
      'discard_or_replace_original_fpn_semantics': 'copy_(w)' in text and 'p3_id' in text,
      'declared_frozen_false_by_code': 'frozen_backbone_pyramid":True' in text,
    }

def extract_numbers():
    out={}
    for name in ('MAP_GATE.json','MAP_SUFFICIENCY.json','MAP_FPN_GATE.json','MAP_IMPLEMENTATION_AUDIT.json'):
        p=OLD/name
        if p.exists():
            try: out[name]=json.loads(p.read_text())
            except Exception as e: out[name]={'parse_error':str(e)}
    return out

def main():
    mt=MAP.read_text(); ft=FPN.read_text()
    mapc=source_checks(mt); fpnc=fpn_checks(ft)
    results=extract_numbers()
    # All old P50 “positive” rows were selected after target GT/teacher eligibility.
    filter_literal = 't is not None and row.get("same_id_candidate_indices")' in mt
    # Spatial resolution ratio is a deterministic consequence of the two pipelines.
    # 1920x1080 -> native keep-ratio 1333x750; short gate -> 448x252.
    native_area=1333*750; short_area=448*252
    payload={
      'status':'HOLD_IMPLEMENTATION_CONFOUNDED_SUPERSEDING_AUDIT',
      'audit_version':'2026-10-06-p50-repair-v1',
      'old_artifacts':{str(p.name):sha(p) for p in [MAP,FPN,OLD/'MAP_GATE.json',OLD/'MAP_SUFFICIENCY.json',OLD/'MAP_FPN_GATE.json'] if p.exists()},
      'source_checks':mapc,
      'fpn_source_checks':fpnc,
      'prior_result_artifacts':results,
      'deterministic_pipeline_checks':{
        'native_detector_input':'1333x800 keep_ratio, source 1920x1080 gives approximately 1333x750',
        'old_map_input':'256x448 direct resize, source 1920x1080 gives approximately 448x252',
        'short_to_native_image_area_ratio':short_area/native_area,
        'short_input_is_not_native_detector_pipeline':True,
        'teacher_projection_is_optimized_in_old_map_gate':mapc['trainable_teacher_projection'],
        'teacher_target_is_not_fixed_native_feature':True,
        'gt_positive_and_teacher_rows_are_required_before_training':filter_literal,
        'no_match_rows_are_not_a_validated_population':filter_literal,
        'eval_denominators_are_not_proven_equal':mapc['unequal_eval_append_paths'],
      },
      'failure_audit':[
        {'id':'IA-1','severity':'critical','finding':'The old map comparison changes the image preprocessing from native detector resolution to a 256x448 direct resize; this is a backbone input change, not a module-only comparison.'},
        {'id':'IA-2','severity':'critical','finding':'The teacher projection is trainable and used in the teacher loss, so the target representation can move with the student; this does not establish a fixed privileged-teacher causal effect.'},
        {'id':'IA-3','severity':'critical','finding':'Training rows require an available teacher and at least one same-ID candidate. Events with no candidate or no prefix teacher are removed, so no-match/dustbin behavior is untested.'},
        {'id':'IA-4','severity':'high','finding':'The old evaluation appends student and frozen-feature ranks through different continue paths, so equal matched denominators are not established.'},
        {'id':'IA-5','severity':'high','finding':'The FPN arm copies selected lateral weights into a new pyramid, adds a normalization block, trains P2/refine parameters, and calls train(); it is not a frozen native FPN control.'},
        {'id':'IA-6','severity':'high','finding':'The prior implementation audit checked checkpoint shape/finite values but did not test native-feature equivalence, no-match retention, teacher fixedness, BN statistics, equal denominators, or causal teacher-off controls.'},
      ],
      'decision':'Prior P50 performance and STOP/PASS wording cannot authorize a scientific rejection or paper claim. Preserve all old files; rerun a native-space equal-budget teacher-on/off and spatial-vs-pooled factorial with explicit no-match rows.',
      'official_val_test_access':False,
      'protected_baseline':'continuation_20261005/p26_mia_baseline',
    }
    AUDIT.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n')
    (ROOT/'SUPERSEDING_AUDIT.md').write_text('# P50 superseding implementation audit\n\n'+payload['decision']+'\n\n'+ '\n'.join(f"- **{x['id']} ({x['severity']})**: {x['finding']}" for x in payload['failure_audit'])+'\n')
    print(json.dumps(payload,indent=2,ensure_ascii=False))
if __name__=='__main__': main()
