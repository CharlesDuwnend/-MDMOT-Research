..FF.....FFF.                                                            [100%]
=================================== FAILURES ===================================
___________ test_round1_native_algorithms_and_birth_policy_unchanged ___________

    def test_round1_native_algorithms_and_birth_policy_unchanged():
        for rel in ['host/yolox/tracker/u2mot_tracker.py','belief/online.py','belief/coherence.py','belief/structural.py']:
            assert sha(ROOT/rel)==sha(PARENT/rel)
        def method(p,name):
            tree=ast.parse(p.read_text())
            return ast.dump(next(x for x in ast.walk(tree) if isinstance(x,ast.FunctionDef) and x.name==name),include_attributes=False)
        for name in ['_egia_allows_birth','_allow_new_track_activation','_solve_assignment','_source_event']:
>           assert method(ROOT/'belief/runtime.py',name)==method(PARENT/'belief/runtime.py',name)

tests/test_further_components.py:61: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

p = PosixPath('/raid/datasets/chc_data/research_artifact_storage/leaf_egia_20261008/further_pure_runtime_v1/belief/runtime.py')
name = '_solve_assignment'

    def method(p,name):
        tree=ast.parse(p.read_text())
>       return ast.dump(next(x for x in ast.walk(tree) if isinstance(x,ast.FunctionDef) and x.name==name),include_attributes=False)
E       StopIteration

tests/test_further_components.py:59: StopIteration

The above exception was the direct cause of the following exception:

cls = <class '_pytest.runner.CallInfo'>
func = <function call_and_report.<locals>.<lambda> at 0x7f56d2bf2ee0>
when = 'call'
reraise = (<class '_pytest.outcomes.Exit'>, <class 'KeyboardInterrupt'>)

    @classmethod
    def from_call(
        cls,
        func: Callable[[], TResult],
        when: Literal["collect", "setup", "call", "teardown"],
        reraise: type[BaseException] | tuple[type[BaseException], ...] | None = None,
    ) -> CallInfo[TResult]:
        """Call func, wrapping the result in a CallInfo.
    
        :param func:
            The function to call. Called without arguments.
        :type func: Callable[[], _pytest.runner.TResult]
        :param when:
            The phase in which the function is called.
        :param reraise:
            Exception or exceptions that shall propagate if raised by the
            function, instead of being wrapped in the CallInfo.
        """
        excinfo = None
        start = timing.time()
        precise_start = timing.perf_counter()
        try:
>           result: TResult | None = func()

/home/chenhc/.local/lib/python3.8/site-packages/_pytest/runner.py:341: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
/home/chenhc/.local/lib/python3.8/site-packages/_pytest/runner.py:242: in <lambda>
    lambda: runtest_hook(item=item, **kwds), when=when, reraise=reraise
/home/chenhc/.local/lib/python3.8/site-packages/pluggy/_hooks.py:513: in __call__
    return self._hookexec(self.name, self._hookimpls.copy(), kwargs, firstresult)
/home/chenhc/.local/lib/python3.8/site-packages/pluggy/_manager.py:120: in _hookexec
    return self._inner_hookexec(hook_name, methods, kwargs, firstresult)
/home/chenhc/.local/lib/python3.8/site-packages/_pytest/threadexception.py:92: in pytest_runtest_call
    yield from thread_exception_runtest_hook()
/home/chenhc/.local/lib/python3.8/site-packages/_pytest/threadexception.py:68: in thread_exception_runtest_hook
    yield
/home/chenhc/.local/lib/python3.8/site-packages/_pytest/unraisableexception.py:95: in pytest_runtest_call
    yield from unraisable_exception_runtest_hook()
/home/chenhc/.local/lib/python3.8/site-packages/_pytest/unraisableexception.py:70: in unraisable_exception_runtest_hook
    yield
/home/chenhc/.local/lib/python3.8/site-packages/_pytest/logging.py:846: in pytest_runtest_call
    yield from self._runtest_for(item, "call")
/home/chenhc/.local/lib/python3.8/site-packages/_pytest/logging.py:829: in _runtest_for
    yield
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <CaptureManager _method='fd' _global_capturing=<MultiCapture out=<FDCapture 1 oldfd=5 _state='suspended' tmpfile=<_io....xtIOWrapper name='/dev/null' mode='r' encoding='utf-8'>> _state='suspended' _in_suspended=False> _capture_fixture=None>
item = <Function test_round1_native_algorithms_and_birth_policy_unchanged>

    @hookimpl(wrapper=True)
    def pytest_runtest_call(self, item: Item) -> Generator[None]:
        with self.item_capture("call", item):
>           return (yield)
E           RuntimeError: generator raised StopIteration

/home/chenhc/.local/lib/python3.8/site-packages/_pytest/capture.py:898: RuntimeError
________ test_round2_only_coverage_probabilities_match_real_classifier _________

    def test_round2_only_coverage_probabilities_match_real_classifier():
        a=next(a for a in DESIGN['arms'] if a['structural_mode']=='coverage_only')
        s=bundle(a)['source_model']
        rows=pickle.loads(DATA.read_bytes())['calibration'][::41]
        x=np.stack([r['legacy'] for r in rows])
        model=s.anchor.coverage_model;q=model.predict_proba(x)[:,list(model.classes_).index(1)]
>       np.testing.assert_array_equal(s.anchor.predict_proba(x),np.column_stack((1-q,q,np.zeros(len(q)))))

tests/test_further_components.py:69: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

args = (<built-in function eq>, array([[0.94848062, 0.05151938, 0.        ],
       [0.6237014 , 0.3762986 , 0.        ],
   ...5, 0.33608495, 0.        ],
       [0.60816563, 0.39183437, 0.        ],
       [0.58939777, 0.41060223, 0.        ]]))
kwds = {'err_msg': '', 'header': 'Arrays are not equal', 'strict': False, 'verbose': True}

    @wraps(func)
    def inner(*args, **kwds):
        with self._recreate_cm():
>           return func(*args, **kwds)
E           AssertionError: 
E           Arrays are not equal
E           
E           Mismatched elements: 412 / 618 (66.7%)
E           Max absolute difference: 6.24212815e-09
E           Max relative difference: 1.63562673e-08
E            x: array([[0.948481, 0.051519, 0.      ],
E                  [0.623701, 0.376299, 0.      ],
E                  [0.738402, 0.261598, 0.      ],...
E            y: array([[0.948481, 0.051519, 0.      ],
E                  [0.623701, 0.376299, 0.      ],
E                  [0.738402, 0.261598, 0.      ],...

/home/chenhc/.conda/envs/u2mot/lib/python3.8/contextlib.py:75: AssertionError
_ test_round3_outer_and_inner_gate_disabled_with_real_candidate[A04_coverage_only_selective] _

arm = {'birth_class_gate': 0.0, 'bundle': 'models/A04_coverage_only_selective.pkl', 'bundle_sha256': '60e8d0ec1083223342120e322db336e7a7d049a4aaf8bc551692e7914ef196e7', 'control': 'coverage_only', ...}

    @pytest.mark.parametrize('arm',DESIGN['arms'],ids=lambda a:a['name'])
    def test_round3_outer_and_inner_gate_disabled_with_real_candidate(arm):
        t=track(arm);t.frame_id=1;t.source_frame=1
        candidate=STrack(np.asarray([10.,20.,20.,40.]),.9,0,semantic_score=.65)
        t.frame_observations={0:{'frame':1,'source_frame':1,'detection_ordinal':0,
            'bbox_tlwh':[10.,20.,20.,40.],'score':.9,'class_confidence':.65,
            'predicted_class':0,'frame_width':300,'frame_height':200}}
        t.det_ordinals[id(candidate)]=0
        # Native eligibility must admit below 0.7 before EGIA makes its own decision.
        assert t._allow_new_track_activation(candidate)
        t._egia_allows_birth(candidate,[],[])
>       assert t.counts['local_h2_admitted']==1
E       assert 2 == 1

tests/test_further_components.py:127: AssertionError
_ test_round3_outer_and_inner_gate_disabled_with_real_candidate[A05_two_heads_argmax] _

arm = {'birth_class_gate': 0.0, 'bundle': 'models/A05_two_heads_argmax.pkl', 'bundle_sha256': 'b1992a866b8031ef56ba9ed2b3a18a086dfec06c3107cc2e3f2d4d5b24ebbb66', 'control': 'delete_source_cues', ...}

    @pytest.mark.parametrize('arm',DESIGN['arms'],ids=lambda a:a['name'])
    def test_round3_outer_and_inner_gate_disabled_with_real_candidate(arm):
        t=track(arm);t.frame_id=1;t.source_frame=1
        candidate=STrack(np.asarray([10.,20.,20.,40.]),.9,0,semantic_score=.65)
        t.frame_observations={0:{'frame':1,'source_frame':1,'detection_ordinal':0,
            'bbox_tlwh':[10.,20.,20.,40.],'score':.9,'class_confidence':.65,
            'predicted_class':0,'frame_width':300,'frame_height':200}}
        t.det_ordinals[id(candidate)]=0
        # Native eligibility must admit below 0.7 before EGIA makes its own decision.
>       assert t._allow_new_track_activation(candidate)
E       assert False
E        +  where False = _allow_new_track_activation(OT_0_(0-0))
E        +    where _allow_new_track_activation = <belief.online.OnlineBeliefTracker object at 0x7f56cbeab940>._allow_new_track_activation

tests/test_further_components.py:125: AssertionError
_ test_round3_outer_and_inner_gate_disabled_with_real_candidate[A06_full_argmax] _

arm = {'birth_class_gate': 0.0, 'bundle': 'models/A06_full_argmax.pkl', 'bundle_sha256': '36f76b833162c426f9c998b2f6ff81d715c902300e7d0f9598ca196fba99de44', 'control': 'full', ...}

    @pytest.mark.parametrize('arm',DESIGN['arms'],ids=lambda a:a['name'])
    def test_round3_outer_and_inner_gate_disabled_with_real_candidate(arm):
        t=track(arm);t.frame_id=1;t.source_frame=1
        candidate=STrack(np.asarray([10.,20.,20.,40.]),.9,0,semantic_score=.65)
        t.frame_observations={0:{'frame':1,'source_frame':1,'detection_ordinal':0,
            'bbox_tlwh':[10.,20.,20.,40.],'score':.9,'class_confidence':.65,
            'predicted_class':0,'frame_width':300,'frame_height':200}}
        t.det_ordinals[id(candidate)]=0
        # Native eligibility must admit below 0.7 before EGIA makes its own decision.
>       assert t._allow_new_track_activation(candidate)
E       assert False
E        +  where False = _allow_new_track_activation(OT_0_(0-0))
E        +    where _allow_new_track_activation = <belief.online.OnlineBeliefTracker object at 0x7f56cba47460>._allow_new_track_activation

tests/test_further_components.py:125: AssertionError
=========================== short test summary info ============================
FAILED tests/test_further_components.py::test_round1_native_algorithms_and_birth_policy_unchanged
FAILED tests/test_further_components.py::test_round2_only_coverage_probabilities_match_real_classifier
FAILED tests/test_further_components.py::test_round3_outer_and_inner_gate_disabled_with_real_candidate[A04_coverage_only_selective]
FAILED tests/test_further_components.py::test_round3_outer_and_inner_gate_disabled_with_real_candidate[A05_two_heads_argmax]
FAILED tests/test_further_components.py::test_round3_outer_and_inner_gate_disabled_with_real_candidate[A06_full_argmax]
5 failed, 8 passed in 3.38s
