```text
...........................EEEEEEEEE........................FFFFFFFFFFFF [ 84%]
FFFFFFFF.....                                                            [100%]
==================================== ERRORS ====================================
__ ERROR at setup of test_round1_exact_preregistered_sources_and_no_test_fit ___

    @pytest.fixture(scope='module')
    def audit():
>       prereg = json.loads((ROOT / 'PHASE2_HEAD_PREREGISTRATION.json').read_text())

tests/test_paired_anchor_contract.py:18: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
../.conda/envs/u2mot/lib/python3.8/pathlib.py:1236: in read_text
    with self.open(mode='r', encoding=encoding, errors=errors) as f:
../.conda/envs/u2mot/lib/python3.8/pathlib.py:1222: in open
    return io.open(self, mode, buffering, encoding, errors, newline,
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = PosixPath('/home/chenhc/leaf_egia_src_off_gate_20261008_v1/PHASE2_HEAD_PREREGISTRATION.json')
name = '/home/chenhc/leaf_egia_src_off_gate_20261008_v1/PHASE2_HEAD_PREREGISTRATION.json'
flags = 524288, mode = 438

    def _opener(self, name, flags, mode=0o666):
        # A stub for the opener argument to built-in open()
>       return self._accessor.open(self, flags, mode)
E       FileNotFoundError: [Errno 2] No such file or directory: '/home/chenhc/leaf_egia_src_off_gate_20261008_v1/PHASE2_HEAD_PREREGISTRATION.json'

../.conda/envs/u2mot/lib/python3.8/pathlib.py:1078: FileNotFoundError
___ ERROR at setup of test_round1_actual_training_and_calibration_membership ___

    @pytest.fixture(scope='module')
    def audit():
>       prereg = json.loads((ROOT / 'PHASE2_HEAD_PREREGISTRATION.json').read_text())

tests/test_paired_anchor_contract.py:18: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
../.conda/envs/u2mot/lib/python3.8/pathlib.py:1236: in read_text
    with self.open(mode='r', encoding=encoding, errors=errors) as f:
../.conda/envs/u2mot/lib/python3.8/pathlib.py:1222: in open
    return io.open(self, mode, buffering, encoding, errors, newline,
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = PosixPath('/home/chenhc/leaf_egia_src_off_gate_20261008_v1/PHASE2_HEAD_PREREGISTRATION.json')
name = '/home/chenhc/leaf_egia_src_off_gate_20261008_v1/PHASE2_HEAD_PREREGISTRATION.json'
flags = 524288, mode = 438

    def _opener(self, name, flags, mode=0o666):
        # A stub for the opener argument to built-in open()
>       return self._accessor.open(self, flags, mode)
E       FileNotFoundError: [Errno 2] No such file or directory: '/home/chenhc/leaf_egia_src_off_gate_20261008_v1/PHASE2_HEAD_PREREGISTRATION.json'

../.conda/envs/u2mot/lib/python3.8/pathlib.py:1078: FileNotFoundError
___________ ERROR at setup of test_round1_actual_features_and_labels ___________

    @pytest.fixture(scope='module')
    def audit():
>       prereg = json.loads((ROOT / 'PHASE2_HEAD_PREREGISTRATION.json').read_text())

tests/test_paired_anchor_contract.py:18: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
../.conda/envs/u2mot/lib/python3.8/pathlib.py:1236: in read_text
    with self.open(mode='r', encoding=encoding, errors=errors) as f:
../.conda/envs/u2mot/lib/python3.8/pathlib.py:1222: in open
    return io.open(self, mode, buffering, encoding, errors, newline,
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = PosixPath('/home/chenhc/leaf_egia_src_off_gate_20261008_v1/PHASE2_HEAD_PREREGISTRATION.json')
name = '/home/chenhc/leaf_egia_src_off_gate_20261008_v1/PHASE2_HEAD_PREREGISTRATION.json'
flags = 524288, mode = 438

    def _opener(self, name, flags, mode=0o666):
        # A stub for the opener argument to built-in open()
>       return self._accessor.open(self, flags, mode)
E       FileNotFoundError: [Errno 2] No such file or directory: '/home/chenhc/leaf_egia_src_off_gate_20261008_v1/PHASE2_HEAD_PREREGISTRATION.json'

../.conda/envs/u2mot/lib/python3.8/pathlib.py:1078: FileNotFoundError
_ ERROR at setup of test_round2_independent_causal_and_common_weight_construction _

    @pytest.fixture(scope='module')
    def audit():
>       prereg = json.loads((ROOT / 'PHASE2_HEAD_PREREGISTRATION.json').read_text())

tests/test_paired_anchor_contract.py:18: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
../.conda/envs/u2mot/lib/python3.8/pathlib.py:1236: in read_text
    with self.open(mode='r', encoding=encoding, errors=errors) as f:
../.conda/envs/u2mot/lib/python3.8/pathlib.py:1222: in open
    return io.open(self, mode, buffering, encoding, errors, newline,
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = PosixPath('/home/chenhc/leaf_egia_src_off_gate_20261008_v1/PHASE2_HEAD_PREREGISTRATION.json')
name = '/home/chenhc/leaf_egia_src_off_gate_20261008_v1/PHASE2_HEAD_PREREGISTRATION.json'
flags = 524288, mode = 438

    def _opener(self, name, flags, mode=0o666):
        # A stub for the opener argument to built-in open()
>       return self._accessor.open(self, flags, mode)
E       FileNotFoundError: [Errno 2] No such file or directory: '/home/chenhc/leaf_egia_src_off_gate_20261008_v1/PHASE2_HEAD_PREREGISTRATION.json'

../.conda/envs/u2mot/lib/python3.8/pathlib.py:1078: FileNotFoundError
______ ERROR at setup of test_round2_common_scalers_are_full_fit_float64 _______

    @pytest.fixture(scope='module')
    def audit():
>       prereg = json.loads((ROOT / 'PHASE2_HEAD_PREREGISTRATION.json').read_text())

tests/test_paired_anchor_contract.py:18: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
../.conda/envs/u2mot/lib/python3.8/pathlib.py:1236: in read_text
    with self.open(mode='r', encoding=encoding, errors=errors) as f:
../.conda/envs/u2mot/lib/python3.8/pathlib.py:1222: in open
    return io.open(self, mode, buffering, encoding, errors, newline,
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = PosixPath('/home/chenhc/leaf_egia_src_off_gate_20261008_v1/PHASE2_HEAD_PREREGISTRATION.json')
name = '/home/chenhc/leaf_egia_src_off_gate_20261008_v1/PHASE2_HEAD_PREREGISTRATION.json'
flags = 524288, mode = 438

    def _opener(self, name, flags, mode=0o666):
        # A stub for the opener argument to built-in open()
>       return self._accessor.open(self, flags, mode)
E       FileNotFoundError: [Errno 2] No such file or directory: '/home/chenhc/leaf_egia_src_off_gate_20261008_v1/PHASE2_HEAD_PREREGISTRATION.json'

../.conda/envs/u2mot/lib/python3.8/pathlib.py:1078: FileNotFoundError
_ ERROR at setup of test_round2_common_fit_parameters_and_posterior_factorization _

    @pytest.fixture(scope='module')
    def audit():
>       prereg = json.loads((ROOT / 'PHASE2_HEAD_PREREGISTRATION.json').read_text())

tests/test_paired_anchor_contract.py:18: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
../.conda/envs/u2mot/lib/python3.8/pathlib.py:1236: in read_text
    with self.open(mode='r', encoding=encoding, errors=errors) as f:
../.conda/envs/u2mot/lib/python3.8/pathlib.py:1222: in open
    return io.open(self, mode, buffering, encoding, errors, newline,
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = PosixPath('/home/chenhc/leaf_egia_src_off_gate_20261008_v1/PHASE2_HEAD_PREREGISTRATION.json')
name = '/home/chenhc/leaf_egia_src_off_gate_20261008_v1/PHASE2_HEAD_PREREGISTRATION.json'
flags = 524288, mode = 438

    def _opener(self, name, flags, mode=0o666):
        # A stub for the opener argument to built-in open()
>       return self._accessor.open(self, flags, mode)
E       FileNotFoundError: [Errno 2] No such file or directory: '/home/chenhc/leaf_egia_src_off_gate_20261008_v1/PHASE2_HEAD_PREREGISTRATION.json'

../.conda/envs/u2mot/lib/python3.8/pathlib.py:1078: FileNotFoundError
___ ERROR at setup of test_round3_only_nested_anchor_differs_from_frozen_f00 ___

    @pytest.fixture(scope='module')
    def audit():
>       prereg = json.loads((ROOT / 'PHASE2_HEAD_PREREGISTRATION.json').read_text())

tests/test_paired_anchor_contract.py:18: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
../.conda/envs/u2mot/lib/python3.8/pathlib.py:1236: in read_text
    with self.open(mode='r', encoding=encoding, errors=errors) as f:
../.conda/envs/u2mot/lib/python3.8/pathlib.py:1222: in open
    return io.open(self, mode, buffering, encoding, errors, newline,
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = PosixPath('/home/chenhc/leaf_egia_src_off_gate_20261008_v1/PHASE2_HEAD_PREREGISTRATION.json')
name = '/home/chenhc/leaf_egia_src_off_gate_20261008_v1/PHASE2_HEAD_PREREGISTRATION.json'
flags = 524288, mode = 438

    def _opener(self, name, flags, mode=0o666):
        # A stub for the opener argument to built-in open()
>       return self._accessor.open(self, flags, mode)
E       FileNotFoundError: [Errno 2] No such file or directory: '/home/chenhc/leaf_egia_src_off_gate_20261008_v1/PHASE2_HEAD_PREREGISTRATION.json'

../.conda/envs/u2mot/lib/python3.8/pathlib.py:1078: FileNotFoundError
_________ ERROR at setup of test_round3_gt_free_online_event_contract __________

    @pytest.fixture(scope='module')
    def audit():
>       prereg = json.loads((ROOT / 'PHASE2_HEAD_PREREGISTRATION.json').read_text())

tests/test_paired_anchor_contract.py:18: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
../.conda/envs/u2mot/lib/python3.8/pathlib.py:1236: in read_text
    with self.open(mode='r', encoding=encoding, errors=errors) as f:
../.conda/envs/u2mot/lib/python3.8/pathlib.py:1222: in open
    return io.open(self, mode, buffering, encoding, errors, newline,
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = PosixPath('/home/chenhc/leaf_egia_src_off_gate_20261008_v1/PHASE2_HEAD_PREREGISTRATION.json')
name = '/home/chenhc/leaf_egia_src_off_gate_20261008_v1/PHASE2_HEAD_PREREGISTRATION.json'
flags = 524288, mode = 438

    def _opener(self, name, flags, mode=0o666):
        # A stub for the opener argument to built-in open()
>       return self._accessor.open(self, flags, mode)
E       FileNotFoundError: [Errno 2] No such file or directory: '/home/chenhc/leaf_egia_src_off_gate_20261008_v1/PHASE2_HEAD_PREREGISTRATION.json'

../.conda/envs/u2mot/lib/python3.8/pathlib.py:1078: FileNotFoundError
_ ERROR at setup of test_round3_readback_receipt_models_and_diagnostic_boundary _

    @pytest.fixture(scope='module')
    def audit():
>       prereg = json.loads((ROOT / 'PHASE2_HEAD_PREREGISTRATION.json').read_text())

tests/test_paired_anchor_contract.py:18: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
../.conda/envs/u2mot/lib/python3.8/pathlib.py:1236: in read_text
    with self.open(mode='r', encoding=encoding, errors=errors) as f:
../.conda/envs/u2mot/lib/python3.8/pathlib.py:1222: in open
    return io.open(self, mode, buffering, encoding, errors, newline,
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = PosixPath('/home/chenhc/leaf_egia_src_off_gate_20261008_v1/PHASE2_HEAD_PREREGISTRATION.json')
name = '/home/chenhc/leaf_egia_src_off_gate_20261008_v1/PHASE2_HEAD_PREREGISTRATION.json'
flags = 524288, mode = 438

    def _opener(self, name, flags, mode=0o666):
        # A stub for the opener argument to built-in open()
>       return self._accessor.open(self, flags, mode)
E       FileNotFoundError: [Errno 2] No such file or directory: '/home/chenhc/leaf_egia_src_off_gate_20261008_v1/PHASE2_HEAD_PREREGISTRATION.json'

../.conda/envs/u2mot/lib/python3.8/pathlib.py:1078: FileNotFoundError
=================================== FAILURES ===================================
________ test_declared_runtime_flags_and_actual_stream[S0_G1_disabled] _________

arm = {'birth_class_gate': 0.7, 'bundle': 'models/D01_EGIA_bypass.pkl', 'bundle_sha256': '05161442dd036004d531a405c7d4bc2f9be2799e91215be65e80d95da4cd85a6', 'control': 'disabled', ...}

    @pytest.mark.parametrize('arm',DESIGN['arms'],ids=lambda a:a['name'])
    def test_declared_runtime_flags_and_actual_stream(arm):
        t=tracker(arm)
        for frame in range(1,4):
            raw=np.asarray([[10+frame,20,30+frame,60,.9,.8,0],
                            [100,20,120,60,.9,.65,3]],np.float32)
            t.update_online(raw,(200,300),(200,300),np.eye(4,dtype=np.float32)[:2],frame)
        r=t.runtime_report();c=r['counts']
        assert r['src_state_enabled']==r['src_update_enabled']==r['src_cost_enabled']==arm['src']
        assert r['birth_class_confidence_threshold']==arm['birth_class_gate']
        assert r['use_egia']==arm['use_egia'] and r['egia_fusion_policy']==arm['fusion']
        assert c['association_seams']==9
        if not arm['src']:
            assert c['native_reference_seams_verified']==9 and r['stored_beliefs']==0
            for k in ['belief_initializations','belief_updates','src_responsibility_calls',
                      'src_observation_probability_calls','src_semantic_penalty_calls']:
>               assert c[k]==0
E               KeyError: 'belief_initializations'

tests/test_gate_grid.py:43: KeyError
________ test_declared_runtime_flags_and_actual_stream[S0_G0_disabled] _________

arm = {'birth_class_gate': 0.0, 'bundle': 'models/D01_EGIA_bypass.pkl', 'bundle_sha256': '05161442dd036004d531a405c7d4bc2f9be2799e91215be65e80d95da4cd85a6', 'control': 'disabled', ...}

    @pytest.mark.parametrize('arm',DESIGN['arms'],ids=lambda a:a['name'])
    def test_declared_runtime_flags_and_actual_stream(arm):
        t=tracker(arm)
        for frame in range(1,4):
            raw=np.asarray([[10+frame,20,30+frame,60,.9,.8,0],
                            [100,20,120,60,.9,.65,3]],np.float32)
            t.update_online(raw,(200,300),(200,300),np.eye(4,dtype=np.float32)[:2],frame)
        r=t.runtime_report();c=r['counts']
        assert r['src_state_enabled']==r['src_update_enabled']==r['src_cost_enabled']==arm['src']
        assert r['birth_class_confidence_threshold']==arm['birth_class_gate']
        assert r['use_egia']==arm['use_egia'] and r['egia_fusion_policy']==arm['fusion']
        assert c['association_seams']==9
        if not arm['src']:
            assert c['native_reference_seams_verified']==9 and r['stored_beliefs']==0
            for k in ['belief_initializations','belief_updates','src_responsibility_calls',
                      'src_observation_probability_calls','src_semantic_penalty_calls']:
>               assert c[k]==0
E               KeyError: 'belief_initializations'

tests/test_gate_grid.py:43: KeyError
__________ test_declared_runtime_flags_and_actual_stream[S0_G1_full] ___________

arm = {'birth_class_gate': 0.7, 'bundle': 'models/F00_full.pkl', 'bundle_sha256': 'ee903030e48f73fe87411b6e9aa60d52bbf2ee2d895aebfa8085d260779702ac', 'control': 'full', ...}

    @pytest.mark.parametrize('arm',DESIGN['arms'],ids=lambda a:a['name'])
    def test_declared_runtime_flags_and_actual_stream(arm):
        t=tracker(arm)
        for frame in range(1,4):
            raw=np.asarray([[10+frame,20,30+frame,60,.9,.8,0],
                            [100,20,120,60,.9,.65,3]],np.float32)
            t.update_online(raw,(200,300),(200,300),np.eye(4,dtype=np.float32)[:2],frame)
        r=t.runtime_report();c=r['counts']
        assert r['src_state_enabled']==r['src_update_enabled']==r['src_cost_enabled']==arm['src']
        assert r['birth_class_confidence_threshold']==arm['birth_class_gate']
        assert r['use_egia']==arm['use_egia'] and r['egia_fusion_policy']==arm['fusion']
        assert c['association_seams']==9
        if not arm['src']:
            assert c['native_reference_seams_verified']==9 and r['stored_beliefs']==0
            for k in ['belief_initializations','belief_updates','src_responsibility_calls',
                      'src_observation_probability_calls','src_semantic_penalty_calls']:
>               assert c[k]==0
E               KeyError: 'belief_initializations'

tests/test_gate_grid.py:43: KeyError
__________ test_declared_runtime_flags_and_actual_stream[S0_G0_full] ___________

arm = {'birth_class_gate': 0.0, 'bundle': 'models/F00_full.pkl', 'bundle_sha256': 'ee903030e48f73fe87411b6e9aa60d52bbf2ee2d895aebfa8085d260779702ac', 'control': 'full', ...}

    @pytest.mark.parametrize('arm',DESIGN['arms'],ids=lambda a:a['name'])
    def test_declared_runtime_flags_and_actual_stream(arm):
        t=tracker(arm)
        for frame in range(1,4):
            raw=np.asarray([[10+frame,20,30+frame,60,.9,.8,0],
                            [100,20,120,60,.9,.65,3]],np.float32)
            t.update_online(raw,(200,300),(200,300),np.eye(4,dtype=np.float32)[:2],frame)
        r=t.runtime_report();c=r['counts']
        assert r['src_state_enabled']==r['src_update_enabled']==r['src_cost_enabled']==arm['src']
        assert r['birth_class_confidence_threshold']==arm['birth_class_gate']
        assert r['use_egia']==arm['use_egia'] and r['egia_fusion_policy']==arm['fusion']
        assert c['association_seams']==9
        if not arm['src']:
            assert c['native_reference_seams_verified']==9 and r['stored_beliefs']==0
            for k in ['belief_initializations','belief_updates','src_responsibility_calls',
                      'src_observation_probability_calls','src_semantic_penalty_calls']:
>               assert c[k]==0
E               KeyError: 'belief_initializations'

tests/test_gate_grid.py:43: KeyError
______ test_declared_runtime_flags_and_actual_stream[S0_G1_no_foreground] ______

arm = {'birth_class_gate': 0.7, 'bundle': 'models/D02_no_foreground_evidence.pkl', 'bundle_sha256': '1098de1acbbb46db99abebcd1b2ed761a9dec1a83f16088ffabed5ded56015d6', 'control': 'no_foreground', ...}

    @pytest.mark.parametrize('arm',DESIGN['arms'],ids=lambda a:a['name'])
    def test_declared_runtime_flags_and_actual_stream(arm):
        t=tracker(arm)
        for frame in range(1,4):
            raw=np.asarray([[10+frame,20,30+frame,60,.9,.8,0],
                            [100,20,120,60,.9,.65,3]],np.float32)
            t.update_online(raw,(200,300),(200,300),np.eye(4,dtype=np.float32)[:2],frame)
        r=t.runtime_report();c=r['counts']
        assert r['src_state_enabled']==r['src_update_enabled']==r['src_cost_enabled']==arm['src']
        assert r['birth_class_confidence_threshold']==arm['birth_class_gate']
        assert r['use_egia']==arm['use_egia'] and r['egia_fusion_policy']==arm['fusion']
        assert c['association_seams']==9
        if not arm['src']:
            assert c['native_reference_seams_verified']==9 and r['stored_beliefs']==0
            for k in ['belief_initializations','belief_updates','src_responsibility_calls',
                      'src_observation_probability_calls','src_semantic_penalty_calls']:
>               assert c[k]==0
E               KeyError: 'belief_initializations'

tests/test_gate_grid.py:43: KeyError
______ test_declared_runtime_flags_and_actual_stream[S0_G0_no_foreground] ______

arm = {'birth_class_gate': 0.0, 'bundle': 'models/D02_no_foreground_evidence.pkl', 'bundle_sha256': '1098de1acbbb46db99abebcd1b2ed761a9dec1a83f16088ffabed5ded56015d6', 'control': 'no_foreground', ...}

    @pytest.mark.parametrize('arm',DESIGN['arms'],ids=lambda a:a['name'])
    def test_declared_runtime_flags_and_actual_stream(arm):
        t=tracker(arm)
        for frame in range(1,4):
            raw=np.asarray([[10+frame,20,30+frame,60,.9,.8,0],
                            [100,20,120,60,.9,.65,3]],np.float32)
            t.update_online(raw,(200,300),(200,300),np.eye(4,dtype=np.float32)[:2],frame)
        r=t.runtime_report();c=r['counts']
        assert r['src_state_enabled']==r['src_update_enabled']==r['src_cost_enabled']==arm['src']
        assert r['birth_class_confidence_threshold']==arm['birth_class_gate']
        assert r['use_egia']==arm['use_egia'] and r['egia_fusion_policy']==arm['fusion']
        assert c['association_seams']==9
        if not arm['src']:
            assert c['native_reference_seams_verified']==9 and r['stored_beliefs']==0
            for k in ['belief_initializations','belief_updates','src_responsibility_calls',
                      'src_observation_probability_calls','src_semantic_penalty_calls']:
>               assert c[k]==0
E               KeyError: 'belief_initializations'

tests/test_gate_grid.py:43: KeyError
_______ test_declared_runtime_flags_and_actual_stream[S0_G1_no_coverage] _______

arm = {'birth_class_gate': 0.7, 'bundle': 'models/D03_no_coverage_evidence.pkl', 'bundle_sha256': 'ecf9e44a98c7cb51ad4266114000cf947bbdcd569d9875fef71b308c0203d4b5', 'control': 'no_coverage', ...}

    @pytest.mark.parametrize('arm',DESIGN['arms'],ids=lambda a:a['name'])
    def test_declared_runtime_flags_and_actual_stream(arm):
        t=tracker(arm)
        for frame in range(1,4):
            raw=np.asarray([[10+frame,20,30+frame,60,.9,.8,0],
                            [100,20,120,60,.9,.65,3]],np.float32)
            t.update_online(raw,(200,300),(200,300),np.eye(4,dtype=np.float32)[:2],frame)
        r=t.runtime_report();c=r['counts']
        assert r['src_state_enabled']==r['src_update_enabled']==r['src_cost_enabled']==arm['src']
        assert r['birth_class_confidence_threshold']==arm['birth_class_gate']
        assert r['use_egia']==arm['use_egia'] and r['egia_fusion_policy']==arm['fusion']
        assert c['association_seams']==9
        if not arm['src']:
            assert c['native_reference_seams_verified']==9 and r['stored_beliefs']==0
            for k in ['belief_initializations','belief_updates','src_responsibility_calls',
                      'src_observation_probability_calls','src_semantic_penalty_calls']:
>               assert c[k]==0
E               KeyError: 'belief_initializations'

tests/test_gate_grid.py:43: KeyError
_______ test_declared_runtime_flags_and_actual_stream[S0_G0_no_coverage] _______

arm = {'birth_class_gate': 0.0, 'bundle': 'models/D03_no_coverage_evidence.pkl', 'bundle_sha256': 'ecf9e44a98c7cb51ad4266114000cf947bbdcd569d9875fef71b308c0203d4b5', 'control': 'no_coverage', ...}

    @pytest.mark.parametrize('arm',DESIGN['arms'],ids=lambda a:a['name'])
    def test_declared_runtime_flags_and_actual_stream(arm):
        t=tracker(arm)
        for frame in range(1,4):
            raw=np.asarray([[10+frame,20,30+frame,60,.9,.8,0],
                            [100,20,120,60,.9,.65,3]],np.float32)
            t.update_online(raw,(200,300),(200,300),np.eye(4,dtype=np.float32)[:2],frame)
        r=t.runtime_report();c=r['counts']
        assert r['src_state_enabled']==r['src_update_enabled']==r['src_cost_enabled']==arm['src']
        assert r['birth_class_confidence_threshold']==arm['birth_class_gate']
        assert r['use_egia']==arm['use_egia'] and r['egia_fusion_policy']==arm['fusion']
        assert c['association_seams']==9
        if not arm['src']:
            assert c['native_reference_seams_verified']==9 and r['stored_beliefs']==0
            for k in ['belief_initializations','belief_updates','src_responsibility_calls',
                      'src_observation_probability_calls','src_semantic_penalty_calls']:
>               assert c[k]==0
E               KeyError: 'belief_initializations'

tests/test_gate_grid.py:43: KeyError
_______ test_declared_runtime_flags_and_actual_stream[S0_G1_no_context] ________

arm = {'birth_class_gate': 0.7, 'bundle': 'models/D04_no_learned_context.pkl', 'bundle_sha256': '754cbc2572c5a44f42d2312078bdd10e272d941956735ce2fa9496a20c2b97ab', 'control': 'no_context', ...}

    @pytest.mark.parametrize('arm',DESIGN['arms'],ids=lambda a:a['name'])
    def test_declared_runtime_flags_and_actual_stream(arm):
        t=tracker(arm)
        for frame in range(1,4):
            raw=np.asarray([[10+frame,20,30+frame,60,.9,.8,0],
                            [100,20,120,60,.9,.65,3]],np.float32)
            t.update_online(raw,(200,300),(200,300),np.eye(4,dtype=np.float32)[:2],frame)
        r=t.runtime_report();c=r['counts']
        assert r['src_state_enabled']==r['src_update_enabled']==r['src_cost_enabled']==arm['src']
        assert r['birth_class_confidence_threshold']==arm['birth_class_gate']
        assert r['use_egia']==arm['use_egia'] and r['egia_fusion_policy']==arm['fusion']
        assert c['association_seams']==9
        if not arm['src']:
            assert c['native_reference_seams_verified']==9 and r['stored_beliefs']==0
            for k in ['belief_initializations','belief_updates','src_responsibility_calls',
                      'src_observation_probability_calls','src_semantic_penalty_calls']:
>               assert c[k]==0
E               KeyError: 'belief_initializations'

tests/test_gate_grid.py:43: KeyError
_______ test_declared_runtime_flags_and_actual_stream[S0_G0_no_context] ________

arm = {'birth_class_gate': 0.0, 'bundle': 'models/D04_no_learned_context.pkl', 'bundle_sha256': '754cbc2572c5a44f42d2312078bdd10e272d941956735ce2fa9496a20c2b97ab', 'control': 'no_context', ...}

    @pytest.mark.parametrize('arm',DESIGN['arms'],ids=lambda a:a['name'])
    def test_declared_runtime_flags_and_actual_stream(arm):
        t=tracker(arm)
        for frame in range(1,4):
            raw=np.asarray([[10+frame,20,30+frame,60,.9,.8,0],
                            [100,20,120,60,.9,.65,3]],np.float32)
            t.update_online(raw,(200,300),(200,300),np.eye(4,dtype=np.float32)[:2],frame)
        r=t.runtime_report();c=r['counts']
        assert r['src_state_enabled']==r['src_update_enabled']==r['src_cost_enabled']==arm['src']
        assert r['birth_class_confidence_threshold']==arm['birth_class_gate']
        assert r['use_egia']==arm['use_egia'] and r['egia_fusion_policy']==arm['fusion']
        assert c['association_seams']==9
        if not arm['src']:
            assert c['native_reference_seams_verified']==9 and r['stored_beliefs']==0
            for k in ['belief_initializations','belief_updates','src_responsibility_calls',
                      'src_observation_probability_calls','src_semantic_penalty_calls']:
>               assert c[k]==0
E               KeyError: 'belief_initializations'

tests/test_gate_grid.py:43: KeyError
______ test_declared_runtime_flags_and_actual_stream[S0_G1_no_coherence] _______

arm = {'birth_class_gate': 0.7, 'bundle': 'models/D05_no_source_coherence.pkl', 'bundle_sha256': '68a6b7102ada81d9a8ab5ae614f7937c02db03586175afbd232d12e3ea624a00', 'control': 'no_coherence', ...}

    @pytest.mark.parametrize('arm',DESIGN['arms'],ids=lambda a:a['name'])
    def test_declared_runtime_flags_and_actual_stream(arm):
        t=tracker(arm)
        for frame in range(1,4):
            raw=np.asarray([[10+frame,20,30+frame,60,.9,.8,0],
                            [100,20,120,60,.9,.65,3]],np.float32)
            t.update_online(raw,(200,300),(200,300),np.eye(4,dtype=np.float32)[:2],frame)
        r=t.runtime_report();c=r['counts']
        assert r['src_state_enabled']==r['src_update_enabled']==r['src_cost_enabled']==arm['src']
        assert r['birth_class_confidence_threshold']==arm['birth_class_gate']
        assert r['use_egia']==arm['use_egia'] and r['egia_fusion_policy']==arm['fusion']
        assert c['association_seams']==9
        if not arm['src']:
            assert c['native_reference_seams_verified']==9 and r['stored_beliefs']==0
            for k in ['belief_initializations','belief_updates','src_responsibility_calls',
                      'src_observation_probability_calls','src_semantic_penalty_calls']:
>               assert c[k]==0
E               KeyError: 'belief_initializations'

tests/test_gate_grid.py:43: KeyError
______ test_declared_runtime_flags_and_actual_stream[S0_G0_no_coherence] _______

arm = {'birth_class_gate': 0.0, 'bundle': 'models/D05_no_source_coherence.pkl', 'bundle_sha256': '68a6b7102ada81d9a8ab5ae614f7937c02db03586175afbd232d12e3ea624a00', 'control': 'no_coherence', ...}

    @pytest.mark.parametrize('arm',DESIGN['arms'],ids=lambda a:a['name'])
    def test_declared_runtime_flags_and_actual_stream(arm):
        t=tracker(arm)
        for frame in range(1,4):
            raw=np.asarray([[10+frame,20,30+frame,60,.9,.8,0],
                            [100,20,120,60,.9,.65,3]],np.float32)
            t.update_online(raw,(200,300),(200,300),np.eye(4,dtype=np.float32)[:2],frame)
        r=t.runtime_report();c=r['counts']
        assert r['src_state_enabled']==r['src_update_enabled']==r['src_cost_enabled']==arm['src']
        assert r['birth_class_confidence_threshold']==arm['birth_class_gate']
        assert r['use_egia']==arm['use_egia'] and r['egia_fusion_policy']==arm['fusion']
        assert c['association_seams']==9
        if not arm['src']:
            assert c['native_reference_seams_verified']==9 and r['stored_beliefs']==0
            for k in ['belief_initializations','belief_updates','src_responsibility_calls',
                      'src_observation_probability_calls','src_semantic_penalty_calls']:
>               assert c[k]==0
E               KeyError: 'belief_initializations'

tests/test_gate_grid.py:43: KeyError
______ test_declared_runtime_flags_and_actual_stream[S0_G1_pair_shuffle] _______

arm = {'birth_class_gate': 0.7, 'bundle': 'models/D06_broken_source_pairing.pkl', 'bundle_sha256': 'fc491a93c408050bbe0a0b4570e020abbe26812871a9b58cfa4c9d1a9b6cb4b5', 'control': 'pair_shuffle', ...}

    @pytest.mark.parametrize('arm',DESIGN['arms'],ids=lambda a:a['name'])
    def test_declared_runtime_flags_and_actual_stream(arm):
        t=tracker(arm)
        for frame in range(1,4):
            raw=np.asarray([[10+frame,20,30+frame,60,.9,.8,0],
                            [100,20,120,60,.9,.65,3]],np.float32)
            t.update_online(raw,(200,300),(200,300),np.eye(4,dtype=np.float32)[:2],frame)
        r=t.runtime_report();c=r['counts']
        assert r['src_state_enabled']==r['src_update_enabled']==r['src_cost_enabled']==arm['src']
        assert r['birth_class_confidence_threshold']==arm['birth_class_gate']
        assert r['use_egia']==arm['use_egia'] and r['egia_fusion_policy']==arm['fusion']
        assert c['association_seams']==9
        if not arm['src']:
            assert c['native_reference_seams_verified']==9 and r['stored_beliefs']==0
            for k in ['belief_initializations','belief_updates','src_responsibility_calls',
                      'src_observation_probability_calls','src_semantic_penalty_calls']:
>               assert c[k]==0
E               KeyError: 'belief_initializations'

tests/test_gate_grid.py:43: KeyError
______ test_declared_runtime_flags_and_actual_stream[S0_G0_pair_shuffle] _______

arm = {'birth_class_gate': 0.0, 'bundle': 'models/D06_broken_source_pairing.pkl', 'bundle_sha256': 'fc491a93c408050bbe0a0b4570e020abbe26812871a9b58cfa4c9d1a9b6cb4b5', 'control': 'pair_shuffle', ...}

    @pytest.mark.parametrize('arm',DESIGN['arms'],ids=lambda a:a['name'])
    def test_declared_runtime_flags_and_actual_stream(arm):
        t=tracker(arm)
        for frame in range(1,4):
            raw=np.asarray([[10+frame,20,30+frame,60,.9,.8,0],
                            [100,20,120,60,.9,.65,3]],np.float32)
            t.update_online(raw,(200,300),(200,300),np.eye(4,dtype=np.float32)[:2],frame)
        r=t.runtime_report();c=r['counts']
        assert r['src_state_enabled']==r['src_update_enabled']==r['src_cost_enabled']==arm['src']
        assert r['birth_class_confidence_threshold']==arm['birth_class_gate']
        assert r['use_egia']==arm['use_egia'] and r['egia_fusion_policy']==arm['fusion']
        assert c['association_seams']==9
        if not arm['src']:
            assert c['native_reference_seams_verified']==9 and r['stored_beliefs']==0
            for k in ['belief_initializations','belief_updates','src_responsibility_calls',
                      'src_observation_probability_calls','src_semantic_penalty_calls']:
>               assert c[k]==0
E               KeyError: 'belief_initializations'

tests/test_gate_grid.py:43: KeyError
_______ test_declared_runtime_flags_and_actual_stream[S0_G1_fusion_off] ________

arm = {'birth_class_gate': 0.7, 'bundle': 'models/GRID_fusion_off.pkl', 'bundle_sha256': '4165da0896b5246d7a0c494b17c38625fbbb8e55fdee10c6fb0c848e3887ddd9', 'control': 'fusion_off', ...}

    @pytest.mark.parametrize('arm',DESIGN['arms'],ids=lambda a:a['name'])
    def test_declared_runtime_flags_and_actual_stream(arm):
        t=tracker(arm)
        for frame in range(1,4):
            raw=np.asarray([[10+frame,20,30+frame,60,.9,.8,0],
                            [100,20,120,60,.9,.65,3]],np.float32)
            t.update_online(raw,(200,300),(200,300),np.eye(4,dtype=np.float32)[:2],frame)
        r=t.runtime_report();c=r['counts']
        assert r['src_state_enabled']==r['src_update_enabled']==r['src_cost_enabled']==arm['src']
        assert r['birth_class_confidence_threshold']==arm['birth_class_gate']
        assert r['use_egia']==arm['use_egia'] and r['egia_fusion_policy']==arm['fusion']
        assert c['association_seams']==9
        if not arm['src']:
            assert c['native_reference_seams_verified']==9 and r['stored_beliefs']==0
            for k in ['belief_initializations','belief_updates','src_responsibility_calls',
                      'src_observation_probability_calls','src_semantic_penalty_calls']:
>               assert c[k]==0
E               KeyError: 'belief_initializations'

tests/test_gate_grid.py:43: KeyError
_______ test_declared_runtime_flags_and_actual_stream[S0_G0_fusion_off] ________

arm = {'birth_class_gate': 0.0, 'bundle': 'models/GRID_fusion_off.pkl', 'bundle_sha256': '4165da0896b5246d7a0c494b17c38625fbbb8e55fdee10c6fb0c848e3887ddd9', 'control': 'fusion_off', ...}

    @pytest.mark.parametrize('arm',DESIGN['arms'],ids=lambda a:a['name'])
    def test_declared_runtime_flags_and_actual_stream(arm):
        t=tracker(arm)
        for frame in range(1,4):
            raw=np.asarray([[10+frame,20,30+frame,60,.9,.8,0],
                            [100,20,120,60,.9,.65,3]],np.float32)
            t.update_online(raw,(200,300),(200,300),np.eye(4,dtype=np.float32)[:2],frame)
        r=t.runtime_report();c=r['counts']
        assert r['src_state_enabled']==r['src_update_enabled']==r['src_cost_enabled']==arm['src']
        assert r['birth_class_confidence_threshold']==arm['birth_class_gate']
        assert r['use_egia']==arm['use_egia'] and r['egia_fusion_policy']==arm['fusion']
        assert c['association_seams']==9
        if not arm['src']:
            assert c['native_reference_seams_verified']==9 and r['stored_beliefs']==0
            for k in ['belief_initializations','belief_updates','src_responsibility_calls',
                      'src_observation_probability_calls','src_semantic_penalty_calls']:
>               assert c[k]==0
E               KeyError: 'belief_initializations'

tests/test_gate_grid.py:43: KeyError
______ test_declared_runtime_flags_and_actual_stream[S0_G1_selective_off] ______

arm = {'birth_class_gate': 0.7, 'bundle': 'models/GRID_selective_off.pkl', 'bundle_sha256': 'e58043b014e851b84440851449c5a52e10bf29af900428e56b2567445f72511c', 'control': 'selective_off', ...}

    @pytest.mark.parametrize('arm',DESIGN['arms'],ids=lambda a:a['name'])
    def test_declared_runtime_flags_and_actual_stream(arm):
        t=tracker(arm)
        for frame in range(1,4):
            raw=np.asarray([[10+frame,20,30+frame,60,.9,.8,0],
                            [100,20,120,60,.9,.65,3]],np.float32)
            t.update_online(raw,(200,300),(200,300),np.eye(4,dtype=np.float32)[:2],frame)
        r=t.runtime_report();c=r['counts']
        assert r['src_state_enabled']==r['src_update_enabled']==r['src_cost_enabled']==arm['src']
        assert r['birth_class_confidence_threshold']==arm['birth_class_gate']
        assert r['use_egia']==arm['use_egia'] and r['egia_fusion_policy']==arm['fusion']
        assert c['association_seams']==9
        if not arm['src']:
            assert c['native_reference_seams_verified']==9 and r['stored_beliefs']==0
            for k in ['belief_initializations','belief_updates','src_responsibility_calls',
                      'src_observation_probability_calls','src_semantic_penalty_calls']:
>               assert c[k]==0
E               KeyError: 'belief_initializations'

tests/test_gate_grid.py:43: KeyError
______ test_declared_runtime_flags_and_actual_stream[S0_G0_selective_off] ______

arm = {'birth_class_gate': 0.0, 'bundle': 'models/GRID_selective_off.pkl', 'bundle_sha256': 'e58043b014e851b84440851449c5a52e10bf29af900428e56b2567445f72511c', 'control': 'selective_off', ...}

    @pytest.mark.parametrize('arm',DESIGN['arms'],ids=lambda a:a['name'])
    def test_declared_runtime_flags_and_actual_stream(arm):
        t=tracker(arm)
        for frame in range(1,4):
            raw=np.asarray([[10+frame,20,30+frame,60,.9,.8,0],
                            [100,20,120,60,.9,.65,3]],np.float32)
            t.update_online(raw,(200,300),(200,300),np.eye(4,dtype=np.float32)[:2],frame)
        r=t.runtime_report();c=r['counts']
        assert r['src_state_enabled']==r['src_update_enabled']==r['src_cost_enabled']==arm['src']
        assert r['birth_class_confidence_threshold']==arm['birth_class_gate']
        assert r['use_egia']==arm['use_egia'] and r['egia_fusion_policy']==arm['fusion']
        assert c['association_seams']==9
        if not arm['src']:
            assert c['native_reference_seams_verified']==9 and r['stored_beliefs']==0
            for k in ['belief_initializations','belief_updates','src_responsibility_calls',
                      'src_observation_probability_calls','src_semantic_penalty_calls']:
>               assert c[k]==0
E               KeyError: 'belief_initializations'

tests/test_gate_grid.py:43: KeyError
__ test_declared_runtime_flags_and_actual_stream[S0_G1_fusion_selective_off] ___

arm = {'birth_class_gate': 0.7, 'bundle': 'models/GRID_fusion_selective_off.pkl', 'bundle_sha256': 'bbd97ac63539083accaf655d93cd3dd3ca442330af89b411ecfc0162d0f8cd21', 'control': 'fusion_selective_off', ...}

    @pytest.mark.parametrize('arm',DESIGN['arms'],ids=lambda a:a['name'])
    def test_declared_runtime_flags_and_actual_stream(arm):
        t=tracker(arm)
        for frame in range(1,4):
            raw=np.asarray([[10+frame,20,30+frame,60,.9,.8,0],
                            [100,20,120,60,.9,.65,3]],np.float32)
            t.update_online(raw,(200,300),(200,300),np.eye(4,dtype=np.float32)[:2],frame)
        r=t.runtime_report();c=r['counts']
        assert r['src_state_enabled']==r['src_update_enabled']==r['src_cost_enabled']==arm['src']
        assert r['birth_class_confidence_threshold']==arm['birth_class_gate']
        assert r['use_egia']==arm['use_egia'] and r['egia_fusion_policy']==arm['fusion']
        assert c['association_seams']==9
        if not arm['src']:
            assert c['native_reference_seams_verified']==9 and r['stored_beliefs']==0
            for k in ['belief_initializations','belief_updates','src_responsibility_calls',
                      'src_observation_probability_calls','src_semantic_penalty_calls']:
>               assert c[k]==0
E               KeyError: 'belief_initializations'

tests/test_gate_grid.py:43: KeyError
__ test_declared_runtime_flags_and_actual_stream[S0_G0_fusion_selective_off] ___

arm = {'birth_class_gate': 0.0, 'bundle': 'models/GRID_fusion_selective_off.pkl', 'bundle_sha256': 'bbd97ac63539083accaf655d93cd3dd3ca442330af89b411ecfc0162d0f8cd21', 'control': 'fusion_selective_off', ...}

    @pytest.mark.parametrize('arm',DESIGN['arms'],ids=lambda a:a['name'])
    def test_declared_runtime_flags_and_actual_stream(arm):
        t=tracker(arm)
        for frame in range(1,4):
            raw=np.asarray([[10+frame,20,30+frame,60,.9,.8,0],
                            [100,20,120,60,.9,.65,3]],np.float32)
            t.update_online(raw,(200,300),(200,300),np.eye(4,dtype=np.float32)[:2],frame)
        r=t.runtime_report();c=r['counts']
        assert r['src_state_enabled']==r['src_update_enabled']==r['src_cost_enabled']==arm['src']
        assert r['birth_class_confidence_threshold']==arm['birth_class_gate']
        assert r['use_egia']==arm['use_egia'] and r['egia_fusion_policy']==arm['fusion']
        assert c['association_seams']==9
        if not arm['src']:
            assert c['native_reference_seams_verified']==9 and r['stored_beliefs']==0
            for k in ['belief_initializations','belief_updates','src_responsibility_calls',
                      'src_observation_probability_calls','src_semantic_penalty_calls']:
>               assert c[k]==0
E               KeyError: 'belief_initializations'

tests/test_gate_grid.py:43: KeyError
=========================== short test summary info ============================
FAILED tests/test_gate_grid.py::test_declared_runtime_flags_and_actual_stream[S0_G1_disabled]
FAILED tests/test_gate_grid.py::test_declared_runtime_flags_and_actual_stream[S0_G0_disabled]
FAILED tests/test_gate_grid.py::test_declared_runtime_flags_and_actual_stream[S0_G1_full]
FAILED tests/test_gate_grid.py::test_declared_runtime_flags_and_actual_stream[S0_G0_full]
FAILED tests/test_gate_grid.py::test_declared_runtime_flags_and_actual_stream[S0_G1_no_foreground]
FAILED tests/test_gate_grid.py::test_declared_runtime_flags_and_actual_stream[S0_G0_no_foreground]
FAILED tests/test_gate_grid.py::test_declared_runtime_flags_and_actual_stream[S0_G1_no_coverage]
FAILED tests/test_gate_grid.py::test_declared_runtime_flags_and_actual_stream[S0_G0_no_coverage]
FAILED tests/test_gate_grid.py::test_declared_runtime_flags_and_actual_stream[S0_G1_no_context]
FAILED tests/test_gate_grid.py::test_declared_runtime_flags_and_actual_stream[S0_G0_no_context]
FAILED tests/test_gate_grid.py::test_declared_runtime_flags_and_actual_stream[S0_G1_no_coherence]
FAILED tests/test_gate_grid.py::test_declared_runtime_flags_and_actual_stream[S0_G0_no_coherence]
FAILED tests/test_gate_grid.py::test_declared_runtime_flags_and_actual_stream[S0_G1_pair_shuffle]
FAILED tests/test_gate_grid.py::test_declared_runtime_flags_and_actual_stream[S0_G0_pair_shuffle]
FAILED tests/test_gate_grid.py::test_declared_runtime_flags_and_actual_stream[S0_G1_fusion_off]
FAILED tests/test_gate_grid.py::test_declared_runtime_flags_and_actual_stream[S0_G0_fusion_off]
FAILED tests/test_gate_grid.py::test_declared_runtime_flags_and_actual_stream[S0_G1_selective_off]
FAILED tests/test_gate_grid.py::test_declared_runtime_flags_and_actual_stream[S0_G0_selective_off]
FAILED tests/test_gate_grid.py::test_declared_runtime_flags_and_actual_stream[S0_G1_fusion_selective_off]
FAILED tests/test_gate_grid.py::test_declared_runtime_flags_and_actual_stream[S0_G0_fusion_selective_off]
ERROR tests/test_paired_anchor_contract.py::test_round1_exact_preregistered_sources_and_no_test_fit
ERROR tests/test_paired_anchor_contract.py::test_round1_actual_training_and_calibration_membership
ERROR tests/test_paired_anchor_contract.py::test_round1_actual_features_and_labels
ERROR tests/test_paired_anchor_contract.py::test_round2_independent_causal_and_common_weight_construction
ERROR tests/test_paired_anchor_contract.py::test_round2_common_scalers_are_full_fit_float64
ERROR tests/test_paired_anchor_contract.py::test_round2_common_fit_parameters_and_posterior_factorization
ERROR tests/test_paired_anchor_contract.py::test_round3_only_nested_anchor_differs_from_frozen_f00
ERROR tests/test_paired_anchor_contract.py::test_round3_gt_free_online_event_contract
ERROR tests/test_paired_anchor_contract.py::test_round3_readback_receipt_models_and_diagnostic_boundary
20 failed, 56 passed, 9 errors in 4.31s

```
