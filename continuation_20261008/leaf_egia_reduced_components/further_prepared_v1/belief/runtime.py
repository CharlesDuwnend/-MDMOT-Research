"""GT-free SRC/EGIA replacement at the immutable host's real runtime seams.

The wrapper keeps semantic beliefs and observation provenance outside STrack.
An association returned by LAP is queued, then committed only after the host
has actually updated that track.  Existing host capture is explicitly disabled.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools'))
from belief_data_common import HOST
from capture_belief_evidence import (CaptureTracker, U2MOTTracker, matching,
                                     render, finite_list)
from collections import Counter
import pickle
import time
import copy
import numpy as np
import torch
from belief.features import (semantic_features, responsibility_features,
                             source_vectors, legacy_vector, coverage_event, CLASS_NAMES)
from belief.association import features_from_runtime, pair_features_from_runtime
from belief.models import SourceModel, semantic_penalty, semantic_update


ARMS = {
    'native': (False, False, False, False),
    'h2': (False, False, False, False),
    'src': (True, False, False, False),
    'egia': (False, True, False, False),
    'both': (True, True, False, False),
    'pair_shuffle': (True, True, True, False),
    'hard_update': (True, True, False, True),
    'summary_control': (True, True, False, False),
}

# Explicit SRC pathway controls used by the internal closed-loop ablation.
# The historical arms above retain their original tuple semantics.
PATH_ARMS = {
    'path_no_src': dict(src_cost_enabled=False, src_update_enabled=False,
                        use_egia=True, src_shadow_only=False),
    'path_update_only': dict(src_cost_enabled=False, src_update_enabled=True,
                             use_egia=True, src_shadow_only=False),
    'path_cost_only': dict(src_cost_enabled=True, src_update_enabled=False,
                           use_egia=True, src_shadow_only=False),
    'path_full': dict(src_cost_enabled=True, src_update_enabled=True,
                      use_egia=True, src_shadow_only=False),
}

# Frozen LEAF-context factorial: only the assignment mask and semantic
# penalty vary.  Every cell keeps SRC initialization/soft updates and EGIA.
FACTORIAL_ARMS = {
    'fc_control': dict(assignment_mask_enabled=False, semantic_penalty_enabled=False),
    'fc_mask_only': dict(assignment_mask_enabled=True, semantic_penalty_enabled=False),
    'fc_penalty_only': dict(assignment_mask_enabled=False, semantic_penalty_enabled=True),
    'fc_full': dict(assignment_mask_enabled=True, semantic_penalty_enabled=True),
}


def load_bundle(path):
    """Load a local training artifact; no labels or GT are loaded at runtime."""
    with Path(path).open('rb') as stream:
        return pickle.load(stream)


class BundleAdapter:
    """The explicit inference-only portion of a fit-only training bundle."""
    def __init__(self, bundle):
        self.bundle = bundle
        self.observation = bundle.get('observation_model')
        self.responsibility_model = bundle.get('responsibility_model')
        self.prior = np.asarray(bundle.get('group_prior', [.5, .5]), float)
        if self.prior.shape != (2,) or not np.isfinite(self.prior).all() or (self.prior <= 0).any():
            raise ValueError('invalid group training prior')
        self.prior = self.prior / self.prior.sum()
        self.source = bundle.get('source_model')
        self.coherent_source = bundle.get('source_architecture') in (
            'coherence_ratio_v3', 'coverage_coherence_v4',
            'coverage_hierarchical_coherence_v5')
        self.coverage_semantics = bundle.get('source_architecture') == 'coverage_coherence_v4'
        # This explicit diagnostic fixes the v5 training/deployment source-pool
        # contract. The historical default remains byte-compatible; the summary
        # feature vector and frozen EGIA fusion input are never re-rendered.
        self.source_pool_mode = bundle.get('source_pool_mode', 'legacy_all')
        if self.source_pool_mode not in ('legacy_all', 'coverage_current'):
            raise ValueError('unsupported source_pool_mode')
        if (self.source_pool_mode == 'coverage_current' and
                bundle.get('source_architecture') != 'coverage_hierarchical_coherence_v5'):
            raise ValueError('coverage_current source pool is an explicit v5 diagnostic')
        self.coverage_render = bundle.get('coverage_render', {})
        self.vehicle_only_semantics = bool(bundle.get('vehicle_only_semantics', False))
        self.association_foreground_model = bundle.get('association_foreground_model')
        self.association_foreground_feature_names = bundle.get(
            'association_foreground_feature_names', [])
        self.association_edge_model = bundle.get('association_edge_model')
        self.association_edge_feature_names = bundle.get(
            'association_edge_feature_names', [])
        self.association_edge_lambda = float(
            bundle.get('association_edge_lambda', 0.15))
        self.last_coherence = None
        if self.source is None and 'source_state_dict' in bundle:
            architecture = bundle.get('source_architecture', 'candidate_null_v1')
            if architecture not in ('candidate_null_v1', 'contextual_null_v2'):
                raise ValueError('unsupported source architecture')
            self.source = SourceModel(contextual_null=architecture == 'contextual_null_v2')
            self.source.load_state_dict(bundle['source_state_dict'])
        if self.source is not None and not self.coherent_source:
            self.source = self.source.cpu().eval()
        self.candidate_mean = np.asarray(bundle.get('candidate_mean', np.zeros(16)), float)
        self.candidate_scale = np.asarray(bundle.get('candidate_scale', np.ones(16)), float)
        self.owner_mean = np.asarray(bundle.get('owner_mean', np.zeros(16)), float)
        self.owner_scale = np.asarray(bundle.get('owner_scale', np.ones(16)), float)
        for mean, scale in ((self.candidate_mean, self.candidate_scale),
                            (self.owner_mean, self.owner_scale)):
            if mean.shape != (16,) or scale.shape != (16,) or not np.isfinite(mean).all() or not np.isfinite(scale).all() or (scale <= 0).any():
                raise ValueError('invalid source feature normalization')

    def observation_probabilities(self, observations):
        if self.observation is None:
            raise ValueError('bundle is missing observation_model')
        x = np.asarray([semantic_features(o) for o in observations], float)
        if not len(x):
            return np.empty((0, 2), float)
        classes = list(self.observation.classes_)
        if set(classes) != {0, 1}:
            raise ValueError('observation classes must be human=0, vehicle=1')
        p = np.asarray(self.observation.predict_proba(x), float)[:, [classes.index(0), classes.index(1)]]
        if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1.):
            raise ValueError('invalid calibrated observation probabilities')
        return p

    def responsibilities(self, features):
        if self.responsibility_model is None:
            raise ValueError('bundle is missing responsibility_model')
        classes = list(self.responsibility_model.classes_)
        if 1 not in classes:
            raise ValueError('responsibility model has no positive class')
        p = np.asarray(self.responsibility_model.predict_proba(features), float)[:, classes.index(1)]
        if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise ValueError('invalid responsibilities')
        return p

    def source_probabilities(self, event, shuffle_pairs=False, summary_only=False,
                             intervention_counts=None):
        if self.source is None:
            raise ValueError('bundle is missing source_model')
        if self.coherent_source:
            if self.vehicle_only_semantics:
                event = copy.deepcopy(event)
                for source in event.get('sources', []):
                    evidence = np.asarray(source.get('semantic_evidence', [0., 0.]), float)
                    total = float(evidence.sum())
                    source['semantic_evidence'] = [0., total] if total > 0 else [0., 0.]
            source_event = (coverage_event(event, **self.coverage_render)
                            if (self.coverage_semantics or
                                self.source_pool_mode == 'coverage_current') else event)
            probabilities, ids, evidence = self.source.predict_event(
                source_event, shuffled=shuffle_pairs,
                intervention_counts=intervention_counts)
            self.last_coherence = evidence
            if summary_only:
                probabilities = self.source.anchor.predict_proba(legacy_vector(event)[None])[0]
            return probabilities, ids
        if summary_only:
            raise ValueError('summary_control requires the coherent-source bundle')
        candidate, owners, ids = source_vectors(
            event, shuffle_pairs=shuffle_pairs,
            intervention_counts=intervention_counts)
        candidate = (candidate - self.candidate_mean) / self.candidate_scale
        owners = (owners - self.owner_mean) / self.owner_scale
        # One padded masked node also supports a completely empty pool.
        padded = np.zeros((1, max(1, len(ids)), 16), np.float32)
        mask = np.zeros((1, max(1, len(ids))), bool)
        padded[0, :len(ids)] = owners
        mask[0, :len(ids)] = True
        with torch.no_grad():
            logits, energies = self.source(torch.as_tensor(candidate[None], dtype=torch.float32),
                                           torch.from_numpy(padded), torch.from_numpy(mask))
            probabilities = logits.softmax(-1)[0].cpu().numpy().astype(float)
        if not np.isfinite(probabilities).all() or not np.isclose(probabilities.sum(), 1.):
            raise ValueError('invalid source probabilities')
        return probabilities, ids


class BeliefTracker(U2MOTTracker):
    det_record = CaptureTracker.det_record
    detection_identity_matches = staticmethod(CaptureTracker.detection_identity_matches)
    track_record = CaptureTracker.track_record

    def __init__(self, args, frame_rate=30, *, arm='h2', bundle=None):
        if arm not in ARMS and arm not in PATH_ARMS and arm not in FACTORIAL_ARMS:
            raise ValueError('unknown arm: ' + str(arm))
        if getattr(args, 'egia_capture_dir', ''):
            raise ValueError('HOST_CAPTURE_MUST_BE_DISABLED')
        self.pure_v5_policy = bool(
            arm != 'h2' and bundle is not None and
            bundle.get('source_architecture') == 'coverage_hierarchical_coherence_v5' and
            not bundle.get('egia_fusion_policy', False)
        )
        if self.pure_v5_policy:
            args = copy.copy(args)
            args.egia_model = ''
            args.continuous_semantic_reliability = False
        super().__init__(args, frame_rate)
        # The immutable capture host exposes these fields directly.  The
        # canonical U2MOT host keeps the same tracker behavior but does not
        # expose the EGIA compatibility fields, so initialize them here.
        if not hasattr(self, 'continuous_semantic_reliability'):
            self.continuous_semantic_reliability = bool(
                getattr(args, 'continuous_semantic_reliability', False)
            )
        if not hasattr(self, 'egia_model_payload'):
            self.egia_model_payload = None
            egia_path = str(getattr(args, 'egia_model', '') or '')
            if egia_path:
                with open(egia_path, 'rb') as stream:
                    self.egia_model_payload = pickle.load(stream)
        if self.pure_v5_policy and self.egia_model_payload is not None:
            raise ValueError('PURE_V5_LOADED_LEGACY_EGIA')
        if not self.pure_v5_policy and (
                not self.continuous_semantic_reliability or self.egia_model_payload is None):
            raise ValueError('FROZEN_H2_REQUIRED')
        self.arm = arm
        self.cost_factorial = arm in FACTORIAL_ARMS
        if self.cost_factorial:
            config = FACTORIAL_ARMS[arm]
            self.assignment_mask_enabled = config['assignment_mask_enabled']
            self.semantic_penalty_enabled = config['semantic_penalty_enabled']
            self.src_cost_enabled = self.assignment_mask_enabled or self.semantic_penalty_enabled
            self.src_update_enabled = True
            self.use_src = True
            self.use_egia = True
            self.src_shadow_only = False
            self.shuffle_pairs = False
            self.hard_update = False
        elif arm in PATH_ARMS:
            path_config = PATH_ARMS[arm]
            self.src_cost_enabled = path_config['src_cost_enabled']
            self.src_update_enabled = path_config['src_update_enabled']
            self.use_src = self.src_cost_enabled or self.src_update_enabled
            self.use_egia = path_config['use_egia']
            self.src_shadow_only = path_config['src_shadow_only']
            self.shuffle_pairs = False
            self.hard_update = False
        else:
            self.use_src, self.use_egia, self.shuffle_pairs, self.hard_update = ARMS[arm]
            self.src_cost_enabled = self.use_src
            self.src_update_enabled = self.use_src
        self.src_state_enabled = self.src_cost_enabled or self.src_update_enabled
        self.egia_ablation = bundle.get('egia_ablation', 'full') if bundle is not None else 'full'
        if self.egia_ablation not in ('full', 'disabled', 'no_foreground', 'no_coverage',
                                     'no_context', 'no_coherence', 'pair_shuffle',
                                     'no_geometry_cue', 'no_appearance_cue'):
            raise ValueError('unknown EGIA ablation')
        if self.egia_ablation != 'full':
            if arm not in ('fc_full', 'egia'):
                raise ValueError('detailed EGIA controls require full SRC or explicit SRC-off EGIA arm')
            if arm == 'fc_full' and (not self.src_state_enabled or not self.src_update_enabled):
                raise ValueError('fc_full detailed EGIA controls require full SRC')
            if arm == 'egia' and (self.src_state_enabled or self.src_update_enabled):
                raise ValueError('SRC-off EGIA control unexpectedly enables SRC')
            self.use_egia = self.egia_ablation != 'disabled'
            self.shuffle_pairs = self.egia_ablation == 'pair_shuffle'
        self.model = BundleAdapter(bundle) if arm != 'h2' else None
        self.association_foreground_model = (
            getattr(self.model, 'association_foreground_model', None)
            if self.model is not None else None
        )
        self.association_edge_model = (
            getattr(self.model, 'association_edge_model', None)
            if self.model is not None else None
        )
        self.association_edge_lambda = float(
            getattr(self.model, 'association_edge_lambda', 0.15)
            if self.model is not None else 0.15
        )
        # A source model is allowed to add a birth only under an explicit
        # fit-frozen certificate.  Native H2 admissions remain the hard
        # incumbent; this flag is opt-in so all historical bundles retain
        # their original replay semantics.
        self.safe_native_incumbent = bool(bundle.get('safe_native_incumbent', False)) if bundle is not None else False
        self.safe_new_probability = float(bundle.get('safe_new_probability', 0.8)) if bundle is not None else 0.8
        # Selective birth intervention is an abstaining policy: a native
        # admission can be rejected only when the learned non-new posterior
        # clears NEW_TRUE by the frozen margin; otherwise the native decision
        # is retained.  Native rejections can become births only under the
        # symmetric high-confidence NEW_TRUE condition.
        self.selective_birth_policy = bool(bundle.get('selective_birth_policy', False)) if bundle is not None else False
        self.selective_birth_margin = float(bundle.get('selective_birth_margin', 0.2)) if bundle is not None else 0.2
        # Cost-sensitive selective birth uses different calibrated margins for
        # vetoing a native birth and certifying a below-gate NEW birth.  The
        # two actions have different downstream error costs: suppressing a
        # true NEW causes a miss, while admitting a non-NEW candidate creates
        # a false positive/identity event.
        self.asymmetric_selective_birth_policy = bool(
            bundle.get('asymmetric_selective_birth_policy', False)
        ) if bundle is not None else False
        self.native_suppress_margin = float(
            bundle.get('native_suppress_margin', self.selective_birth_margin)
        ) if bundle is not None else self.selective_birth_margin
        self.new_certificate_margin = float(
            bundle.get('new_certificate_margin', self.selective_birth_margin)
        ) if bundle is not None else self.selective_birth_margin
        # Owner-selective birth intervention uses the latent owner posterior
        # only when a coherent source model can identify one existing owner.
        # This avoids treating a diffuse EXISTING posterior as a license to
        # suppress a native birth.
        self.owner_selective_birth_policy = bool(bundle.get('owner_selective_birth_policy', False)) if bundle is not None else False
        self.owner_posterior_threshold = float(bundle.get('owner_posterior_threshold', 0.7)) if bundle is not None else 0.7
        # Decision-specific birth policy: clutter can veto a native birth from
        # calibrated clutter evidence, while EXISTING can veto only when the
        # coherent source model also identifies a sufficiently certain owner.
        # This keeps the two failure modes (background and duplicate) separate
        # instead of using max(EXISTING, CLUTTER) as one undifferentiated veto.
        self.decision_specific_birth_policy = bool(
            bundle.get('decision_specific_birth_policy', False)
        ) if bundle is not None else False
        self.decision_specific_clutter_threshold = float(
            bundle.get('decision_specific_clutter_threshold', 0.70)
        ) if bundle is not None else 0.70
        self.decision_specific_owner_threshold = float(
            bundle.get('decision_specific_owner_threshold', 0.70)
        ) if bundle is not None else 0.70
        # Probability-level fusion keeps the frozen EGIA teacher's calibrated
        # three-way evidence instead of discarding it through argmax.  The
        # scalar weight is fit on calibration only and the downstream policy
        # still abstains in its uncertainty band.
        self.egia_fusion_policy = bool(bundle.get('egia_fusion_policy', False)) if bundle is not None else False
        self.egia_fusion_weight = float(bundle.get('egia_fusion_weight', 0.0)) if bundle is not None else 0.0
        # Primary paper deployment keeps U2MOT's association assignment as an
        # incumbent.  SRC-Belief still updates the latent track belief and can
        # feed EGIA-SOE, but it is not allowed to veto or replace an
        # association unless a separate association certificate is introduced.
        if arm not in PATH_ARMS and not self.cost_factorial:
            self.src_shadow_only = bool(bundle.get('src_shadow_only', False)) if bundle is not None else False
        # Optional association-side use of the same causal source posterior.
        # The source head first estimates foreground versus clutter; only the
        # clutter posterior is converted into an additive cost.  Since this
        # term can only increase a finite native cost, it cannot create a new
        # edge below the host threshold.  It is therefore an explicitly
        # auditable candidate arm rather than an implicit replacement of the
        # host matcher.
        self.clutter_match_policy = bool(bundle.get('clutter_match_policy', False)) if bundle is not None else False
        self.clutter_match_lambda = float(bundle.get('clutter_match_lambda', 0.25)) if bundle is not None else 0.25
        self.clutter_match_threshold = float(bundle.get('clutter_match_threshold', 0.70)) if bundle is not None else 0.70
        if not 0.0 < self.safe_new_probability < 1.0:
            raise ValueError('invalid safe-new probability')
        if self.selective_birth_margin < 0.0 or self.selective_birth_margin >= 1.0:
            raise ValueError('invalid selective birth margin')
        if not 0.0 <= self.native_suppress_margin < 1.0:
            raise ValueError('invalid native suppress margin')
        if not 0.0 <= self.new_certificate_margin < 1.0:
            raise ValueError('invalid NEW certificate margin')
        if self.owner_posterior_threshold < 0.0 or self.owner_posterior_threshold > 1.0:
            raise ValueError('invalid owner posterior threshold')
        if not 0.0 <= self.decision_specific_clutter_threshold <= 1.0:
            raise ValueError('invalid decision-specific clutter threshold')
        if not 0.0 <= self.decision_specific_owner_threshold <= 1.0:
            raise ValueError('invalid decision-specific owner threshold')
        if self.egia_fusion_weight < 0.0:
            raise ValueError('invalid EGIA fusion weight')
        if self.clutter_match_lambda < 0.0:
            raise ValueError('invalid clutter match lambda')
        if self.association_edge_lambda < 0.0:
            raise ValueError('invalid association edge lambda')
        if not 0.0 <= self.clutter_match_threshold < 1.0:
            raise ValueError('invalid clutter match threshold')
        if self.cost_factorial:
            # This diagnostic is a frozen two-factor comparison, not a new
            # tracker-policy surface.  Reject extra association consumers.
            forbidden = (
                'motion_guidance_geometry', 'motion_guidance', 'temporal_reid_cost',
                'temporal_reid_rescue', 'semantic_group_gate', 'semantic_soft_cost',
                'motion_core_expert', 'motion_ring_expert', 'category_motion_experts',
                'ambiguity_aware_fusion', 'occlusion_reid_rescue',
                'occlusion_reid_motion_gate', 'lost_track_appearance_only',
                'pre_occlusion_prototype', 'covariance_feasibility_gate',
                'conflict_ownership_cascade', 'ownership_global_semantic_gate',
                'human_vehicle_semantic_guard', 'track_perspective_assignment',
                'tracked_first_cascade', 'use_uncertainty', 'clutter_match_policy',
            )
            if any(bool(getattr(self, name, False)) for name in forbidden):
                raise ValueError('FACTORIAL_EXTRA_ASSOCIATION_POLICY')
            if (self.association_edge_model is not None or
                    bool(bundle.get('src_shadow_only', False))):
                raise ValueError('FACTORIAL_EXTRA_SRC_COST_POLICY')
            if (self.fuse_emb_and_iou != 'min' or not self.mask_emb_with_iou or
                    self.appearance_thresh != .25 or self.proximity_thresh != .5 or
                    not np.array_equal(self.super_cls,
                                       [0, 0, 2, 1, 1, 3, 4, 4, 3, 2])):
                raise ValueError('FACTORIAL_FROZEN_BASE_COST_CONTRACT')
            if ((arm != 'fc_full' and
                    (self.pure_v5_policy or not self.egia_fusion_policy)) or
                    (arm == 'fc_full' and not self.egia_fusion_policy and
                     not self.pure_v5_policy)):
                raise ValueError('FACTORIAL_REQUIRES_FROZEN_LEAF_EGIA_CONTEXT')
        self.observed = {}
        self.beliefs = {}
        self.det_ordinals = {}
        self.new_candidates = []
        self.pending = []
        self.counts = Counter()
        self.shuffle_intervention_counts = Counter()
        self.module_seconds = Counter()
        self.responsibility_sum = 0.
        self.belief_change_l1 = 0.
        self.counts['association_changed_vs_native_seams'] = 0
        self.counts['committed_changed_vs_native_seams'] = 0
        if self.cost_factorial:
            for name in ('factorial_assignment_verified_seams',
                         'factorial_native_assignment_calls',
                         'factorial_eligible_seams', 'factorial_empty_seams',
                         'factorial_penalty_off_seams', 'factorial_penalty_on_seams',
                         'factorial_mask_on_seams', 'factorial_mask_changed_edges',
                         'factorial_illegal_edge_recoveries',
                         'src_semantic_penalty_calls', 'src_cost_evaluation_calls',
                         'egia_source_probability_calls'):
                self.counts[name] = 0

    def _flush_committed(self):
        for track, observation, ratio, responsibility in self.pending:
            if track.frame_id != self.frame_id:
                raise AssertionError('BELIEF_UPDATE_BEFORE_HOST_COMMIT')
            tid = int(track.track_id)
            self.observed[tid] = observation
            if self.src_update_enabled:
                if ratio is None or responsibility is None:
                    raise AssertionError('SRC_UPDATE_MISSING_COMMITTED_STATE')
                started = time.perf_counter()
                before = self.beliefs.get(tid, np.asarray([0.5, 0.5], float))
                after = semantic_update(before, ratio, responsibility)
                if self.hard_update:
                    if responsibility != 1.0:
                        raise AssertionError('HARD_UPDATE_RESPONSIBILITY_NOT_ONE')
                    self.counts['hard_update_responsibility_one_updates'] += 1
                self.beliefs[tid] = after
                self.belief_change_l1 += float(np.abs(after-before).sum())
                self.responsibility_sum += float(responsibility)
                self.counts['belief_updates'] += 1
                self.module_seconds['src_update'] += time.perf_counter()-started
            self.counts['committed_observations'] += 1
        self.pending.clear()

    def _base_cost(self, trks, dets, fuse_score, iou_only):
        geometry = matching.iou_distance(trks, dets)
        geom_cost = matching.fuse_score(geometry.copy(), dets) if fuse_score else geometry.copy()
        appearance = matching.embedding_distance(trks, dets)
        base = geom_cost.copy()
        if not iou_only:
            embedding = appearance.copy()
            if len(self.super_cls) > 1:
                embedding = matching.gate_cost_matrix_by_cls(embedding, trks, dets, self.super_cls)
            embedding[embedding > self.appearance_thresh] = 1.
            if self.mask_emb_with_iou:
                embedding[geom_cost > self.proximity_thresh] = 1.
            base = self._fuse_cost_matrix(base, embedding)
        return base, appearance, geometry

    def _assignment_base(self, trks, dets, fuse_score, iou_only, *, mask_enabled,
                         reference=None):
        """Frozen assignment base with only soft coarse-class mask varied.

        ``_base_cost`` remains B_ref for learned responsibility features.  The
        raw appearance and geometry arrays can be reused so the two factors
        cannot alter the feature-extraction context or numeric input ordering.
        """
        if reference is None:
            reference = self._base_cost(trks, dets, fuse_score, iou_only)
        ref_base, appearance, geometry = reference
        if mask_enabled or iou_only:
            return ref_base.copy()
        geom_cost = matching.fuse_score(geometry.copy(), dets) if fuse_score else geometry.copy()
        embedding = appearance.copy()
        embedding[embedding > self.appearance_thresh] = 1.
        if self.mask_emb_with_iou:
            embedding[geom_cost > self.proximity_thresh] = 1.
        return self._fuse_cost_matrix(geom_cost, embedding)

    @staticmethod
    def _assignment_key(result):
        return (tuple(map(tuple, np.asarray(result[0], int).reshape(-1, 2))),
                tuple(np.asarray(result[1], int)), tuple(np.asarray(result[2], int)))

    def _factorial_assignment(self, trks, dets, thresh, fuse_score, iou_only,
                              reference, probabilities):
        """Apply mask/penalty factors, then observe the actual LAP result."""
        self.counts['factorial_eligible_seams'] += 1
        native_base = self._assignment_base(
            trks, dets, fuse_score, iou_only, mask_enabled=False, reference=reference)
        assignment_base = self._assignment_base(
            trks, dets, fuse_score, iou_only,
            mask_enabled=self.assignment_mask_enabled, reference=reference)
        finite = np.isfinite(assignment_base)
        self.counts['association_finite_edges'] += int(finite.sum())
        self.counts['factorial_mask_on_seams'] += int(self.assignment_mask_enabled)
        self.counts['factorial_mask_changed_edges'] += int(
            np.count_nonzero(assignment_base != native_base))
        penalty = np.zeros_like(assignment_base)
        cost = assignment_base.copy()
        if self.semantic_penalty_enabled:
            started = time.perf_counter()
            prior = np.asarray([
                self.beliefs.get(int(t.track_id), np.asarray([.5, .5], float))
                for t in trks])
            penalty = semantic_penalty(prior, probabilities/self.model.prior)
            self.counts['src_semantic_penalty_calls'] += 1
            self.counts['src_cost_evaluation_calls'] += 1
            cost[finite] += (1.-assignment_base[finite])*penalty[finite]
            self.counts['factorial_penalty_on_seams'] += 1
            self.module_seconds['src_cost'] += time.perf_counter()-started
        else:
            self.counts['factorial_penalty_off_seams'] += 1
        illegal = ((assignment_base > thresh) & (cost <= thresh))
        self.counts['factorial_illegal_edge_recoveries'] += int(illegal.sum())
        if (illegal.any() or np.any((native_base > thresh) & (cost <= thresh)) or
                np.any(cost[finite] < assignment_base[finite])):
            raise AssertionError('FACTORIAL_CREATED_ILLEGAL_BASE_EDGE')
        # The control must execute the frozen host's actual association seam.
        # The reconstructed unmasked matrix is independently checked against
        # that seam, so a future host change cannot silently redefine control.
        native_result = super().associate(trks, dets, thresh, fuse_score, iou_only)
        self.counts['factorial_native_assignment_calls'] += 1
        reconstructed_native = matching.linear_assignment(native_base, thresh=thresh)
        if self._assignment_key(native_result) != self._assignment_key(reconstructed_native):
            raise AssertionError('FACTORIAL_NATIVE_BASE_CONTRACT_MISMATCH')
        control = not self.assignment_mask_enabled and not self.semantic_penalty_enabled
        result = (native_result if control
                  else matching.linear_assignment(cost, thresh=thresh))
        # A direct observation of the actual solver inputs/outputs, including
        # threshold legality and the exact off/off host-assignment control.
        pairs = np.asarray(result[0], int).reshape(-1, 2)
        if any(not np.isfinite(cost[i, j]) or cost[i, j] > thresh for i, j in pairs):
            raise AssertionError('FACTORIAL_LAP_RETURNED_ILLEGAL_EDGE')
        if control:
            if self._assignment_key(result) != self._assignment_key(native_result):
                raise AssertionError('FACTORIAL_CONTROL_ASSIGNMENT_CHANGED')
            self.counts['native_assignment_control_seams'] += 1
        self.counts['factorial_assignment_verified_seams'] += 1
        native_pairs = set(map(tuple, np.asarray(native_result[0], int).reshape(-1, 2)))
        committed_pairs = set(map(tuple, pairs))
        changed = int(native_pairs != committed_pairs)
        self.counts['association_changed_vs_native_seams'] += changed
        self.counts['committed_changed_vs_native_seams'] += changed
        self.counts['association_added_vs_native_pairs'] += len(committed_pairs-native_pairs)
        self.counts['association_removed_vs_native_pairs'] += len(native_pairs-committed_pairs)
        self.counts['src_applied_edges'] += int(finite.sum()) if self.src_cost_enabled else 0
        self.counts['src_nonzero_penalty_edges'] += int(np.count_nonzero(penalty[finite]))
        self.counts['src_cost_changed_vs_local_h2_edges'] += int(np.count_nonzero(cost != native_base))
        return result

    def _association_clutter_probabilities(self, trks, dets):
        """Return causal clutter posteriors for the current association seam.

        The model was fit on candidate events, so this deliberately reuses
        the same candidate/source construction at association time.  No GT,
        future frame, or predicted track geometry is introduced.  Tracks are
        split by their host state only to preserve the active/lost source
        semantics; the resulting posterior is used as a soft cost term.
        """
        if not (self.clutter_match_policy and self.src_cost_enabled and len(dets)):
            return np.zeros(len(dets), dtype=float)
        active, lost = [], []
        for track in trks:
            # BaseTrack uses state=1 for tracked and state=2 for lost.  Keep
            # unknown states in the active pool so a canonical host extension
            # cannot silently discard an owner hypothesis.
            if int(getattr(track, 'state', 1)) == 2:
                lost.append(track)
            else:
                active.append(track)
        values = []
        if self.association_foreground_model is not None:
            classes = list(self.association_foreground_model.classes_)
            if 1 not in classes:
                raise ValueError('association foreground model has no clutter class')
            for candidate in dets:
                native = self._egia_features(candidate, active, lost)
                x = features_from_runtime(native)[None]
                p = self.association_foreground_model.predict_proba(x)[0]
                values.append(float(np.clip(p[classes.index(1)], 0., 1.)))
        elif not self.model.coherent_source:
            raise ValueError('clutter matching requires a coherent source head')
        else:
            # Fallback for the exploratory hierarchical action bundle.  This
            # branch is intentionally weaker than the association-specific
            # head and is retained for backward-compatible ablations.
            for candidate in dets:
                native = self._egia_features(candidate, active, lost)
                event = {'native_features': native}
                anchor_p = self.model.source.anchor.predict_proba(
                    legacy_vector(event)[None])[0]
                values.append(float(np.clip(anchor_p[2], 0., 1.)))
        result = np.asarray(values, dtype=float)
        if not np.isfinite(result).all():
            raise ValueError('invalid association clutter probabilities')
        self.counts['clutter_match_events'] += len(dets)
        self.counts['clutter_match_association_head_events'] += int(
            self.association_foreground_model is not None) * len(dets)
        self.counts['clutter_match_high_prob_detections'] += int(
            np.count_nonzero(result >= self.clutter_match_threshold))
        return result

    def _association_edge_penalty(self, trks, dets, base, appearance,
                                  geometry, thresh):
        """Return a bounded residual penalty for each native candidate edge.

        The fitted model predicts whether the current detection belongs to the
        already-held owner.  At runtime only current-frame detector metadata,
        the host's owner state, and native pair costs are used.  The penalty is
        nonnegative, so an edge rejected by the host can never be created.
        """
        if self.association_edge_model is None or not len(trks) or not len(dets):
            return np.zeros_like(base, dtype=float)
        rows = []
        positions = []
        for i, track in enumerate(trks):
            track_row = self.track_record(track)
            for j, det in enumerate(dets):
                if not np.isfinite(base[i, j]):
                    continue
                native = self.det_record(det)
                rows.append(pair_features_from_runtime(
                    track_row, native, base[i, j], geometry[i, j],
                    appearance[i, j], thresh))
                positions.append((i, j))
        if not rows:
            return np.zeros_like(base, dtype=float)
        classes = list(self.association_edge_model.classes_)
        if 1 not in classes:
            raise ValueError('association edge model has no valid-edge class')
        p = np.asarray(self.association_edge_model.predict_proba(
            np.asarray(rows, dtype=float))[:, classes.index(1)], dtype=float)
        if not np.isfinite(p).all() or ((p < 0.) | (p > 1.)).any():
            raise ValueError('invalid association edge probabilities')
        penalty = np.zeros_like(base, dtype=float)
        for (i, j), value in zip(positions, p):
            penalty[i, j] = 1. - float(value)
        self.counts['association_edge_model_events'] += len(rows)
        self.counts['association_edge_model_nonzero_penalties'] += int(
            np.count_nonzero(p < 0.95))
        return penalty

    @staticmethod
    def _egia_iou(a, b):
        ax1, ay1, ax2, ay2 = a.tlbr
        bx1, by1, bx2, by2 = b.tlbr
        left, top = max(ax1, bx1), max(ay1, by1)
        right, bottom = min(ax2, bx2), min(ay2, by2)
        inter = max(right - left, 0.) * max(bottom - top, 0.)
        area_a = max(ax2 - ax1, 0.) * max(ay2 - ay1, 0.)
        area_b = max(bx2 - bx1, 0.) * max(by2 - by1, 0.)
        return inter / max(area_a + area_b - inter, 1e-12)

    @staticmethod
    def _egia_similarity(a, b):
        if getattr(a, 'curr_feat', None) is None or getattr(b, 'smooth_feat', None) is None:
            return None
        return float(np.dot(a.curr_feat, b.smooth_feat))

    def _egia_features(self, candidate, active_tracks, lost_tracks):
        active_iou = [self._egia_iou(candidate, track) for track in active_tracks]
        lost_iou = [self._egia_iou(candidate, track) for track in lost_tracks]
        active_sim = [self._egia_similarity(candidate, track) for track in active_tracks]
        lost_sim = [self._egia_similarity(candidate, track) for track in lost_tracks]
        x, y, w, h = [float(value) for value in candidate.tlwh]
        frame_width = max(float(getattr(self, '_egia_frame_width', 1.)), 1.)
        frame_height = max(float(getattr(self, '_egia_frame_height', 1.)), 1.)
        return {
            'score': float(candidate.score),
            'class_confidence': float(getattr(candidate, 'semantic_score', candidate.score)),
            'max_active_iou': float(max(active_iou, default=0.)),
            'max_lost_iou': float(max(lost_iou, default=0.)),
            'max_active_reid_similarity': float(max((v for v in active_sim if v is not None), default=-1.)),
            'max_lost_reid_similarity': float(max((v for v in lost_sim if v is not None), default=-1.)),
            'active_track_count': len(active_tracks),
            'lost_track_count': len(lost_tracks),
            'bbox_area_ratio': float(w * h / (frame_width * frame_height)),
            'border_distance_ratio': float(min(
                x, y, max(frame_width - x - w, 0.), max(frame_height - y - h, 0.)
            ) / max(frame_width, frame_height)),
        }

    def _allow_new_track_activation(self, candidate):
        """Route canonical-host birth gates through EGIA-SOE when enabled."""
        self.counts['native_birth_gate_calls'] += 1
        low_class = getattr(candidate, 'semantic_score', candidate.score) < 0.7
        score_eligible = candidate.score >= self.new_track_thresh
        self.counts['birth_score_eligible_low_class_candidates'] += int(low_class and score_eligible)
        parent = super()
        native_gate = getattr(parent, '_allow_new_track_activation', None)
        native = bool(native_gate(candidate)) if native_gate is not None else bool(
            candidate.score >= self.new_track_thresh and
            getattr(candidate, 'semantic_score', candidate.score) >= self.new_track_min_class_confidence
        )
        self.counts['native_birth_gate_admitted'] += int(native)
        self.counts['native_birth_gate_low_class_admitted'] += int(native and low_class)
        if not native or not self.use_egia:
            # Identical birth observation and SRC initialization when EGIA is off.
            if native:
                self.new_candidates.append((candidate, self.det_record(candidate)))
            return native
        return bool(self._egia_allows_birth(
            candidate, list(getattr(self, 'tracked_stracks', [])),
            list(getattr(self, 'lost_stracks', []))
        ))

    def associate(self, trks, dets, thresh, fuse_score=False, iou_only=False,
                  diagnostic_stage=None):
        self._flush_committed()
        stage = self.stage
        self.stage += 1
        # The capture host exposes exactly two high/low association lists;
        # canonical U2MOT splits the first stage into tracked/lost cascades.
        # Resolve immutable detector-row identity. Equal boxes can have
        # different class/score metadata and can enter different cascade pools.
        if dets:
            used = set(self.det_ordinals.values())
            if len(used) != len(self.det_ordinals):
                raise AssertionError('DETECTION_ORDINAL_REUSED')
            rows = getattr(self, 'frame_observations', {})
            for det in dets:
                if id(det) in self.det_ordinals:
                    ordinal = self.det_ordinals[id(det)]
                    if (ordinal not in rows or
                            not self.detection_identity_matches(det, rows[ordinal])):
                        raise AssertionError('DETECTION_ORDINAL_METADATA_MISMATCH')
                    continue
                candidates = [int(i) for i, row in rows.items()
                              if int(i) not in used and
                              self.detection_identity_matches(det, row)]
                if not candidates:
                    raise AssertionError('DETECTION_ORDINAL_UNRESOLVED')
                # Fully identical metadata duplicates are interchangeable;
                # consume original ordinals deterministically and only once.
                ordinal = min(candidates)
                self.det_ordinals[id(det)] = ordinal
                used.add(ordinal)
        reference = None
        if (getattr(self, 'cost_factorial', False) and
                self.assignment_mask_enabled == self.semantic_penalty_enabled):
            from belief.reference_oracle import oracle_for_seam, verify_reference
            reference = oracle_for_seam(self, trks, dets, thresh, fuse_score, iou_only)
        elif not self.src_state_enabled:
            from belief.native_reference import native_oracle_for_seam, verify_native_reference
            reference = native_oracle_for_seam(self, trks, dets, thresh, fuse_score, iou_only)
        self.counts['association_seams'] += 1
        self.counts['association_edges'] += len(trks)*len(dets)
        probabilities = None
        base = appearance = geometry = None
        if (getattr(self, 'cost_factorial', False) and
                self.src_state_enabled and len(trks) and len(dets)):
            base, appearance, geometry = self._base_cost(trks, dets, fuse_score, iou_only)
            self.counts['src_observation_probability_calls'] += 1
            self.counts['src_observation_probability_rows'] += len(dets)
            probabilities = self.model.observation_probabilities([self.det_record(d) for d in dets])
            result = self._factorial_assignment(
                trks, dets, thresh, fuse_score, iou_only,
                (base, appearance, geometry), probabilities)
        elif self.src_state_enabled and len(trks) and len(dets):
            base, appearance, geometry = self._base_cost(trks, dets, fuse_score, iou_only)
            self.counts['src_observation_probability_calls'] += 1
            self.counts['src_observation_probability_rows'] += len(dets)
            probabilities = self.model.observation_probabilities(
                [self.det_record(d) for d in dets])
            finite = np.isfinite(base)
            self.counts['association_finite_edges'] += int(finite.sum())
            if self.src_cost_enabled:
                started = time.perf_counter()
                self.counts['src_cost_evaluation_calls'] += 1
                prior = np.asarray([
                    self.beliefs.get(int(t.track_id), np.asarray([0.5, 0.5], float))
                    for t in trks
                ])
                penalty = semantic_penalty(prior, probabilities/self.model.prior)
                self.counts['src_semantic_penalty_calls'] += 1
                cost = base.copy()
                cost[finite] += (1.-base[finite])*penalty[finite]
                if self.clutter_match_policy:
                    clutter = self._association_clutter_probabilities(trks, dets)
                    excess = np.clip(
                        (clutter - self.clutter_match_threshold) /
                        max(1. - self.clutter_match_threshold, 1e-12), 0., 1.)
                    clutter_penalty = self.clutter_match_lambda * excess
                    cost[finite] += ((1.-base) * clutter_penalty[None, :])[finite]
                    self.counts['clutter_match_changed_edges'] += int(
                        np.count_nonzero(clutter_penalty[None, :] > 0.)) * len(trks)
                    self.counts['clutter_match_nonzero_edges'] += int(
                        np.count_nonzero((clutter_penalty[None, :] > 0.) & finite))
                if self.association_edge_model is not None:
                    edge_penalty = self._association_edge_penalty(
                        trks, dets, base, appearance, geometry, thresh)
                    cost[finite] += ((1. - base) *
                                     self.association_edge_lambda *
                                     edge_penalty)[finite]
                    self.counts['association_edge_model_changed_edges'] += int(
                        np.count_nonzero((edge_penalty > 0.) & finite))
                if np.any((base > thresh) & (cost <= thresh)):
                    raise AssertionError('NEW_SRC_CREATED_ILLEGAL_BASE_EDGE')
                self.module_seconds['src_cost'] += time.perf_counter()-started
            else:
                cost = base
                penalty = np.zeros_like(base)
            if hasattr(matching, 'continuous_human_vehicle_semantic_cost'):
                native_cost, _ = matching.continuous_human_vehicle_semantic_cost(base, trks, dets)
            else:
                native_cost = base.copy()
            if self.src_cost_enabled:
                native_result = matching.linear_assignment(native_cost, thresh=thresh)
                cost_result = matching.linear_assignment(cost, thresh=thresh)
                result = native_result if self.src_shadow_only else cost_result
                native_pairs = set(map(tuple, np.asarray(native_result[0], int).reshape(-1, 2)))
                cost_pairs = set(map(tuple, np.asarray(cost_result[0], int).reshape(-1, 2)))
                committed_pairs = set(map(tuple, np.asarray(result[0], int).reshape(-1, 2)))
                self.counts['src_applied_edges'] += int(finite.sum())
                self.counts['src_nonzero_penalty_edges'] += int(np.count_nonzero(penalty[finite]))
                self.counts['src_cost_changed_vs_local_h2_edges'] += int(np.count_nonzero(cost != native_cost))
                self.counts['association_changed_vs_native_seams'] += int(native_pairs != cost_pairs)
                self.counts['committed_changed_vs_native_seams'] += int(native_pairs != committed_pairs)
                self.counts['association_added_vs_native_pairs'] += len(cost_pairs-native_pairs)
                self.counts['association_removed_vs_native_pairs'] += len(native_pairs-cost_pairs)
            elif self.src_update_enabled:
                # Keep the host association contract exact in update-only: the
                # SRC update path observes the matches chosen by the frozen
                # host, while its learned cost is never used to choose them.
                result = super().associate(trks, dets, thresh, fuse_score, iou_only)
                self.counts['native_assignment_control_seams'] += 1
                self.counts['committed_changed_vs_native_seams'] += 0
            else:
                result = matching.linear_assignment(native_cost, thresh=thresh)
            self.counts['src_shadow_only_seams'] += int(self.src_shadow_only and self.src_cost_enabled)
        else:
            result = super().associate(trks, dets, thresh, fuse_score, iou_only)
            if getattr(self, 'cost_factorial', False):
                # Empty matrices are real host seams too; no probability or
                # responsibility call is needed when there cannot be a pair.
                if len(trks) and len(dets):
                    raise AssertionError('FACTORIAL_STATE_PATH_DISABLED')
                self.counts['factorial_empty_seams'] += 1
                self.counts['factorial_native_assignment_calls'] += 1
                self.counts['factorial_assignment_verified_seams'] += 1
                if not self.assignment_mask_enabled and not self.semantic_penalty_enabled:
                    self.counts['native_assignment_control_seams'] += 1
        pairs = np.asarray(result[0], int).reshape(-1, 2)
        responsibilities = [None] * len(pairs)
        if self.src_update_enabled and len(pairs):
            started = time.perf_counter()
            # ``base`` is always the original masked B_ref. Assignment factors
            # only modify a separate matrix, never these frozen-head inputs.
            features = np.asarray([responsibility_features(base, appearance, geometry, i, j,
                                                           thresh, iou_only) for i, j in pairs])
            self.counts['src_responsibility_calls'] += 1
            self.counts['src_responsibility_rows'] += len(pairs)
            responsibilities = self.model.responsibilities(features)
            self.module_seconds['src_responsibility'] += time.perf_counter()-started
        for (i, j), responsibility in zip(pairs, responsibilities):
            observation = self.det_record(dets[j])
            ratio = (probabilities[j]/self.model.prior
                     if self.src_update_enabled and probabilities is not None else None)
            self.pending.append((trks[i], observation, ratio, responsibility))
        if reference is not None:
            if self.src_state_enabled:
                verify_reference(self, result, reference)
            else:
                verify_native_reference(self, result, reference)
        return result

    def _source_event(self, candidate, active_tracks, lost_tracks):
        nodes = {}
        for flag, pool in (('active', active_tracks), ('lost', lost_tracks)):
            for track in pool:
                tid = int(track.track_id)
                if tid not in nodes:
                    node = self.track_record(track)
                    if node['missing_observation']:
                        raise AssertionError('SOURCE_MISSING_CAUSAL_OBSERVATION')
                    similarity = self._egia_similarity(candidate, track)
                    node.update(active=False, lost=False,
                                paired_iou=float(self._egia_iou(candidate, track)),
                                reid_similarity=None if similarity is None else float(similarity))
                    nodes[tid] = node
                nodes[tid][flag] = True
                if self.src_state_enabled and tid in self.beliefs:
                    nodes[tid]['semantic_evidence'] = self.beliefs[tid].tolist()
        return {'sequence': self.sequence_name, 'frame': self.frame_id,
                'source_frame': self.source_frame,
                'detection_ordinal': self.det_ordinals[id(candidate)],
                'candidate': self.det_record(candidate),
                'native_features': self._egia_features(candidate, active_tracks, lost_tracks),
                'sources': list(nodes.values())}

    def _legacy_egia_probabilities(self, candidate, active_tracks, lost_tracks):
        """Return the frozen EGIA teacher posterior in NEW/EXISTING/CLUTTER order."""
        if self.egia_model_payload is None:
            return np.full(3, 1. / 3.)
        values = self._egia_features(candidate, active_tracks, lost_tracks)
        payload = self.egia_model_payload
        x = np.asarray([[values[name] for name in payload['features']]], dtype=float)
        raw = np.asarray(payload['model'].predict_proba(x)[0], dtype=float)
        result = np.zeros(3, dtype=float)
        for index, label in enumerate(payload['labels']):
            if label not in CLASS_NAMES:
                raise ValueError('unsupported frozen EGIA label')
            result[CLASS_NAMES.index(label)] = raw[index]
        if not np.isfinite(result).all() or result.sum() <= 0.:
            raise ValueError('invalid frozen EGIA posterior')
        return result / result.sum()

    def _egia_allows_birth(self, candidate, active_tracks, lost_tracks):
        self._flush_committed()
        self.counts['birth_seams'] += 1
        parent = super()
        if hasattr(parent, '_egia_allows_birth'):
            native = bool(parent._egia_allows_birth(candidate, active_tracks, lost_tracks))
        else:
            native = bool(
                candidate.score >= self.new_track_thresh and
                getattr(candidate, 'semantic_score', candidate.score) >= self.new_track_min_class_confidence
            )
        admitted = native
        self.counts['local_h2_admitted'] += int(native)
        if self.use_egia:
            started = time.perf_counter()
            event = self._source_event(candidate, active_tracks, lost_tracks)
            if self.shuffle_pairs:
                self.shuffle_intervention_counts['birth_events'] += 1
            self.counts['egia_source_probability_calls'] += 1
            probabilities, ids = self.model.source_probabilities(
                event, self.shuffle_pairs,
                summary_only=self.arm == 'summary_control',
                intervention_counts=(self.shuffle_intervention_counts
                                     if self.shuffle_pairs else None))
            if self.egia_fusion_policy:
                legacy = self._legacy_egia_probabilities(candidate, active_tracks, lost_tracks)
                fused_logits = (np.log(np.clip(probabilities, 1e-12, 1.)) +
                                self.egia_fusion_weight *
                                np.log(np.clip(legacy, 1e-12, 1.)))
                fused = np.exp(fused_logits - np.max(fused_logits))
                probabilities = fused / fused.sum()
                self.counts['egia_fusion_events'] += 1
                self.counts['egia_fusion_legacy_clutter_argmax'] += int(
                    int(np.argmax(legacy)) == CLASS_NAMES.index('CLUTTER_FALSE'))
            predicted_new = CLASS_NAMES[int(np.argmax(probabilities))] == 'NEW_TRUE'
            owner_posterior = 0.0
            need_owner = (self.owner_selective_birth_policy or
                          self.decision_specific_birth_policy)
            if need_owner and self.model.coherent_source:
                source_event = coverage_event(event, **self.model.coverage_render)
                _, owners, owner_ids = source_vectors(source_event)
                if len(owner_ids):
                    _, owner_posterior_values = self.model.source.log_evidence(owners)
                    owner_posterior = float(np.max(owner_posterior_values))
            if self.asymmetric_selective_birth_policy:
                new_probability = float(probabilities[0])
                nonnew_probability = float(np.max(probabilities[1:]))
                strong_nonnew = bool(
                    nonnew_probability >= new_probability + self.native_suppress_margin
                )
                strong_new = bool(
                    new_probability >= nonnew_probability + self.new_certificate_margin
                )
                if native:
                    admitted = bool(not strong_nonnew)
                else:
                    admitted = bool(strong_new)
                self.counts['asymmetric_native_kept'] += int(native and admitted)
                self.counts['asymmetric_native_rejected'] += int(native and not admitted)
                self.counts['asymmetric_new_certificate'] += int((not native) and strong_new)
                self.counts['asymmetric_rejected_without_certificate'] += int(
                    (not native) and not strong_new
                )
            elif self.decision_specific_birth_policy:
                new_probability = float(probabilities[0])
                existing_probability = float(probabilities[1])
                clutter_probability = float(probabilities[2])
                margin = self.selective_birth_margin
                strong_clutter = bool(
                    clutter_probability >= new_probability + margin and
                    clutter_probability >= self.decision_specific_clutter_threshold
                )
                strong_existing = bool(
                    existing_probability >= new_probability + margin and
                    owner_posterior >= self.decision_specific_owner_threshold
                )
                strong_new = bool(
                    new_probability >= existing_probability + margin and
                    new_probability >= clutter_probability + margin
                )
                if native:
                    # Keep native births in the uncertainty band.  Only a
                    # class-specific certificate may veto one.
                    admitted = bool(not (strong_clutter or strong_existing))
                else:
                    # A below-gate birth still needs a strict NEW certificate.
                    admitted = bool(strong_new)
                self.counts['decision_specific_native_kept'] += int(native and admitted)
                self.counts['decision_specific_native_rejected'] += int(native and not admitted)
                self.counts['decision_specific_clutter_certificate'] += int(strong_clutter)
                self.counts['decision_specific_existing_certificate'] += int(strong_existing)
                self.counts['decision_specific_owner_certified'] += int(
                    owner_posterior >= self.decision_specific_owner_threshold
                )
                self.counts['decision_specific_new_certificate'] += int((not native) and strong_new)
                self.counts['decision_specific_rejected_without_certificate'] += int(
                    (not native) and not strong_new
                )
            elif self.owner_selective_birth_policy:
                new_probability = float(probabilities[0])
                nonnew_probability = float(np.max(probabilities[1:]))
                margin = self.selective_birth_margin
                strong_new = new_probability >= nonnew_probability + margin
                strong_nonnew = (nonnew_probability >= new_probability + margin and
                                 owner_posterior >= self.owner_posterior_threshold)
                if native:
                    admitted = bool(not strong_nonnew)
                else:
                    admitted = bool(strong_new)
                self.counts['owner_selective_native_kept'] += int(native and admitted)
                self.counts['owner_selective_native_rejected'] += int(native and not admitted)
                self.counts['owner_selective_owner_certified'] += int(owner_posterior >= self.owner_posterior_threshold)
                self.counts['owner_selective_new_certificate'] += int((not native) and strong_new)
                self.counts['owner_selective_rejected_without_certificate'] += int((not native) and not strong_new)
            elif self.selective_birth_policy:
                new_probability = float(probabilities[0])
                nonnew_probability = float(np.max(probabilities[1:]))
                margin = self.selective_birth_margin
                strong_new = new_probability >= nonnew_probability + margin
                strong_nonnew = nonnew_probability >= new_probability + margin
                if native:
                    # Abstain in the uncertainty band and retain native H2.
                    admitted = bool(not strong_nonnew)
                else:
                    admitted = bool(strong_new)
                self.counts['selective_native_kept'] += int(native and admitted)
                self.counts['selective_native_rejected'] += int(native and not admitted)
                self.counts['selective_native_abstained'] += int(native and not strong_nonnew)
                self.counts['selective_new_certificate'] += int((not native) and strong_new)
                self.counts['selective_rejected_without_margin'] += int((not native) and not strong_new)
            elif self.safe_native_incumbent:
                # Do not convert a calibrated classification into a broad
                # suppression rule.  The only intervention is a high-
                # confidence NEW certificate on a native rejection.
                certificate = bool((not native) and
                                   probabilities[0] >= self.safe_new_probability and
                                   probabilities[0] >= probabilities[1] and
                                   probabilities[0] >= probabilities[2])
                admitted = bool(native or certificate)
                self.counts['safe_native_kept'] += int(native)
                self.counts['safe_new_certificate'] += int(certificate)
                self.counts['safe_new_rejected_without_certificate'] += int((not native) and not certificate)
            else:
                admitted = predicted_new
            self.module_seconds['egia_source'] += time.perf_counter()-started
            self.counts['source_events'] += 1
            self.counts['source_nodes'] += len(ids)
            self.counts['source_empty_events'] += int(not ids)
            self.counts['source_multiple_owner_events'] += int(len(ids) > 1)
            if self.owner_selective_birth_policy:
                self.counts['owner_posterior_certified_events'] += int(owner_posterior >= self.owner_posterior_threshold)
            if self.model.last_coherence is not None:
                self.counts['coherence_nonzero_events'] += int(abs(self.model.last_coherence) > 1e-9)
                anchor_admitted = int(np.argmax(self.model.source.anchor.predict_proba(legacy_vector(event)[None])[0])) == 0
                self.counts['birth_changed_vs_same_state_summary'] += int(admitted != anchor_admitted)
            self.counts['birth_changed_vs_local_h2'] += int(admitted != native)
            self.counts['birth_admitted_vs_local_h2_rejected'] += int(admitted and not native)
            self.counts['birth_rejected_vs_local_h2_admitted'] += int(native and not admitted)
        self.counts['admitted'] += int(admitted)
        if admitted:
            self.new_candidates.append((candidate, self.det_record(candidate)))
        return bool(admitted)

    def run_frame(self, record, args):
        if self.pending:
            raise AssertionError('UNFLUSHED_PREVIOUS_FRAME')
        self.source_frame = int(record['frame'])
        self._egia_frame_height = float(record['height'])
        self._egia_frame_width = float(record['width'])
        self.stage = 0
        self.new_candidates = []
        self.det_ordinals = {}
        if not record['called_update']:
            return ''
        if record['use_uncertainty']:
            raise AssertionError('UNCERTAINTY_ASSOCIATION_NOT_SUPPORTED')
        if self.frame_id != record['tracker_frame_before']:
            raise AssertionError('TRACKER_FRAME_MISMATCH')
        detections = record['detections'].copy()
        boxes = detections[:, :4].copy()
        scale = min(record['img_size'][0]/float(record['height']),
                    record['img_size'][1]/float(record['width']))
        boxes /= scale
        boxes[:, 2:] -= boxes[:, :2]
        self.high_ordinals = np.flatnonzero(detections[:, 4] > self.track_high_thresh)
        self.low_ordinals = np.flatnonzero((detections[:, 4] > self.track_low_thresh) &
                                          (detections[:, 4] < self.track_high_thresh))
        self.frame_observations = {
            i: {'sequence': self.sequence_name, 'frame': self.frame_id+1,
                'source_frame': self.source_frame, 'detection_ordinal': i,
                'bbox_tlwh': finite_list(boxes[i].astype(np.float64)),
                'score': float(d[4]), 'class_confidence': float(d[5]),
                'predicted_class': int(d[6]), 'frame_width': int(record['width']),
                'frame_height': int(record['height'])}
            for i, d in enumerate(detections)}
        tracks = self.update(detections, (record['height'], record['width']), record['img_size'],
                             embeddings=record['embeddings'].copy(), use_uncertainty=False, img=None)
        self._flush_committed()
        for track, observation in self.new_candidates:
            if track.start_frame != self.frame_id or track.track_id <= 0:
                raise AssertionError('BIRTH_BELIEF_BEFORE_HOST_ACTIVATION')
            tid = int(track.track_id)
            self.observed[tid] = observation
            if self.src_state_enabled:
                started = time.perf_counter()
                self.beliefs[tid] = self.model.observation_probabilities([observation])[0]
                self.counts['src_observation_probability_calls'] += 1
                self.counts['src_observation_probability_rows'] += 1
                self.module_seconds['src_birth'] += time.perf_counter()-started
                self.counts['belief_initializations'] += 1
        if self.stage != 3:
            raise AssertionError('UNEXPECTED_HOST_ASSOCIATION_SEAMS')
        self.counts['frames'] += 1
        self.counts['detections'] += len(detections)
        return render(tracks, record['frame'], args)

    def runtime_report(self):
        updates = self.counts['belief_updates']
        return {'arm': self.arm,
                'structural_audit': (self.model.source.structural_report()
                                     if hasattr(self.model.source, 'structural_report') else None),
                'egia_ablation': getattr(self, 'egia_ablation', 'full'),
                'src_cost_enabled': self.src_cost_enabled,
                'src_update_enabled': self.src_update_enabled,
                'src_state_enabled': self.src_state_enabled,
                'use_egia': getattr(self, 'use_egia', self.arm not in ('native', 'src')),
                'cost_factorial': ({
                    'assignment_mask_enabled': self.assignment_mask_enabled,
                    'semantic_penalty_enabled': self.semantic_penalty_enabled,
                    'responsibility_base': 'original masked B_ref in every cell',
                    'src_state_birth_update_enabled': True,
                    'context': 'frozen SRC memory with fixed current source heads and declared EGIA policy',
                } if getattr(self, 'cost_factorial', False) else None),
                'assignment_source': ('src_cost' if self.src_cost_enabled and not self.src_shadow_only
                                      else 'native'),
                'counts': dict(self.counts),
                'module_seconds': dict(self.module_seconds),
                'module_seconds_total': sum(self.module_seconds.values()),
                'mean_responsibility': self.responsibility_sum/updates if updates else None,
                'mean_belief_update_l1': self.belief_change_l1/updates if updates else None,
                'stored_beliefs': len(self.beliefs), 'observed_track_ids': len(self.observed),
                'birth_class_confidence_threshold': self.new_track_min_class_confidence,
                'birth_score_threshold': self.new_track_thresh,
                'action_change_reference': ('native host birth gate' if self.pure_v5_policy
                                            else 'same-state frozen H2 seam counterfactual'),
                'pure_v5_policy': self.pure_v5_policy,
                'legacy_egia_loaded': self.egia_model_payload is not None,
                'egia_fusion_policy': self.egia_fusion_policy,
                'selective_birth_policy': self.selective_birth_policy,
                'selective_birth_margin': self.selective_birth_margin,
                'operator_audit': ({'pair_shuffle': dict(self.shuffle_intervention_counts),
                                    'deterministic_key': 'sha256(sequence|frame|detection_ordinal)[:8]'}
                                   if self.shuffle_pairs else
                                   {'hard_update': {
                                       'committed_unit_responsibility_updates': self.counts.get(
                                           'hard_update_responsibility_one_updates', 0),
                                       'all_committed_updates_unit_weight': (
                                           self.counts.get('hard_update_responsibility_one_updates', 0)
                                           == updates)}} if self.hard_update else {}),
                'gt_read': False, 'host_capture_written': False}
