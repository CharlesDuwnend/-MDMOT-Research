"""Audit SRC-off, shared birth gate and explicit EGIA interventions."""
def audit_grid_runtime(reports,arm,smoke=False):
    for sequence,r in reports.items():
        c=r['counts']
        if arm.get('structural_mode'):
            d=r['structural_audit']
            assert d['mode']==arm['structural_mode']
            assert d['forbidden_classes']==arm['forbidden_classes']
            assert d['source_calls']['events']==c['birth_seams']
            assert d['source_calls']['forbidden_nonzero']==0
            if arm['structural_mode'] in ('delete_foreground','delete_coverage'):
                removed='foreground' if arm['structural_mode']=='delete_foreground' else 'coverage'
                retained='coverage' if removed=='foreground' else 'foreground'
                assert d[removed+'_head_deleted']
                assert d['head_calls'][removed]==0
                assert d['head_calls'][retained]>=c['birth_seams']>0
            if arm['structural_mode'] in ('delete_source_cues','coverage_only'):
                assert d['geometry_head_deleted'] and d['appearance_head_deleted']
                assert d['coherence_weight']==0. and d['cue_estimator_calls']==0
                assert c['coherence_nonzero_events']==0
            if arm['structural_mode']=='coverage_only':
                assert d['foreground_head_deleted'] and not d['coverage_head_deleted']
                assert d['head_calls']['foreground']==0
                assert d['head_calls']['coverage']>=c['birth_seams']>0
        assert r['arm']==arm['runtime_arm']
        assert r['src_cost_enabled']==r['src_update_enabled']==r['src_state_enabled']==arm['src']
        assert r['birth_class_confidence_threshold']==arm['birth_class_gate']
        assert r['birth_score_threshold']==.6
        assert r['use_egia']==arm['use_egia'] and r['egia_ablation']==arm['mode']
        assert r['egia_fusion_policy']==arm['fusion']
        assert r['pure_v5_policy'] is True and r['legacy_egia_loaded'] is False
        assert r['selective_birth_policy']==arm['selective']
        assert r['selective_birth_margin']==.2
        assert r['gt_read'] is False
        assert r['input_frames_hashed']==r['frames_seen']
        assert [v['frame'] for v in r['input_records']]==list(range(1,r['frames_seen']+1))
        assert c.get('final_birth_activations',0)>0
        assert c['native_birth_gate_calls']>=c['native_birth_gate_admitted']>0
        if arm['gate']:
            assert c['native_birth_gate_low_class_admitted']==0
        if arm['src']:
            assert c['belief_initializations']==c['final_birth_activations']
            assert c['belief_updates']>0 and c['src_responsibility_calls']>0
            assert c['corrected_reference_seams_verified']==c['association_seams']
            assert r['cost_factorial']['assignment_mask_enabled']
            assert r['cost_factorial']['semantic_penalty_enabled']
        else:
            assert r['cost_factorial'] is None and r['stored_beliefs']==0
            assert r['assignment_source']=='native'
            for name in ['belief_initializations','belief_updates','src_responsibility_calls',
                    'src_semantic_penalty_calls','src_observation_probability_calls','src_applied_edges']:
                assert c.get(name,0)==0,(sequence,name)
            assert c['native_reference_seams_verified']==c['association_seams']
        if arm['use_egia']:
            assert c['egia_source_probability_calls']==c['birth_seams']==c['native_birth_gate_admitted']
            assert c['admitted']==c['final_birth_activations']
            assert c.get('egia_fusion_events',0)==(c['birth_seams'] if arm['fusion'] else 0)
        else:
            assert c.get('egia_source_probability_calls',0)==c.get('egia_fusion_events',0)==0
            assert c['native_birth_gate_admitted']==c['final_birth_activations']
        if not arm['selective']:
            assert all(v==0 for k,v in c.items() if k.startswith('selective_'))
        if arm['mode']=='pair_shuffle':
            assert r['operator_audit']['pair_shuffle']['birth_events']==c['birth_seams']
        if not smoke:assert len(r['input_stream_sha256'])==64
