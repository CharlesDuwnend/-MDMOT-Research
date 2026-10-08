"""Only causal, actually captured inputs; identity IDs never enter a model."""
import hashlib
import math
import numpy as np
import copy

LEGACY_FEATURES = ['score', 'class_confidence', 'max_active_iou', 'max_lost_iou',
                   'max_active_reid_similarity', 'max_lost_reid_similarity',
                   'active_track_count', 'lost_track_count', 'bbox_area_ratio',
                   'border_distance_ratio']
CLASS_NAMES = ['NEW_TRUE', 'EXISTING_OR_DUPLICATE', 'CLUTTER_FALSE']


def semantic_features(observation):
    cls = int(observation['predicted_class'])
    if not 0 <= cls < 10:
        raise ValueError('unsupported selected class')
    conf = float(observation['class_confidence'])
    if not np.isfinite(conf) or not 0 <= conf <= 1:
        raise ValueError('invalid confidence')
    indicator = np.eye(10)[cls]
    logit = np.log(np.clip(conf, 1e-5, 1-1e-5) / np.clip(1-conf, 1e-5, 1))
    # Class-dependent confidence calibration, without repeating geometry priors.
    return np.r_[indicator, indicator * logit]


def legacy_vector(event):
    return np.array([event['native_features'][k] for k in LEGACY_FEATURES], float)


def unique_sources(event):
    result = {}
    for original in event['sources']:
        source = dict(original)
        tid = int(source['track_id'])
        if tid in result:
            for flag in ('active', 'lost'):
                result[tid][flag] = bool(result[tid].get(flag)) or bool(source.get(flag))
        else:
            result[tid] = source
    return list(result.values())


def coverage_event(event, min_box_area=1., aspect_ratio_thresh=100000.):
    """Restrict causal sources to renderable, updated current-frame output tracks."""
    result = copy.copy(event)
    result['sources'] = [s for s in unique_sources(event)
        if s.get('active') and int(s['last_update_frame']) == int(event['frame'])
        and float(s['bbox_tlwh'][2])*float(s['bbox_tlwh'][3]) > min_box_area
        and float(s['bbox_tlwh'][2])/float(s['bbox_tlwh'][3]) < aspect_ratio_thresh
        and float(s['bbox_tlwh'][3])/float(s['bbox_tlwh'][2]) < 4.]
    return result


def source_vectors(event, shuffle_pairs=False, intervention_counts=None):
    native = event['native_features']
    candidate = event['candidate']
    sources = unique_sources(event)
    cls = int(candidate['predicted_class'])
    candidate_x = np.r_[
        [native['score'], native['class_confidence'], native['bbox_area_ratio'],
         native['border_distance_ratio'], np.log1p(native['active_track_count']),
         np.log1p(native['lost_track_count'])], np.eye(10)[cls]]
    similarities = [(s.get('reid_similarity'), s.get('reid_similarity') is not None)
                    for s in sources]
    if shuffle_pairs and len(sources) > 1:
        key = '|'.join(str(event.get(k, '')) for k in
                       ('sequence', 'frame', 'detection_ordinal'))
        seed = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
        order = np.random.RandomState(seed).permutation(len(sources))
        if intervention_counts is not None:
            intervention_counts['eligible_multi_source_events'] += 1
            intervention_counts['nonidentity_permutation_events'] += int(
                not np.array_equal(order, np.arange(len(sources))))
            changed = sum(similarities[i] != similarities[j]
                          for i, j in enumerate(order))
            intervention_counts['changed_source_assignments'] += changed
            intervention_counts['changed_pairing_events'] += int(changed > 0)
        similarities = [similarities[i] for i in order]
    elif shuffle_pairs and intervention_counts is not None:
        intervention_counts['singleton_or_empty_source_events'] += 1
    rows = []
    for source, (similarity, observed) in zip(sources, similarities):
        iou = float(source['paired_iou'])
        reid = float(similarity) if observed else 0.
        evidence = np.asarray(source.get('semantic_evidence', [0., 0.]), float)
        total = float(evidence.sum())
        human = float(evidence[0] / total) if total > 0 else .5
        last = source.get('last_observation') or {}
        same_class = float(int(last.get('predicted_class', -1)) == cls)
        rows.append([iou, reid, float(observed), float(source.get('active', False)),
                     float(source.get('lost', False)), np.log1p(max(0, source['age'])),
                     np.log1p(max(0, source['gap'] or 0)), float(last.get('score', 0.)),
                     float(last.get('class_confidence', 0.)), same_class, human,
                     np.log1p(total), iou * reid, iou * same_class,
                     reid * same_class, float(bool(last))])
    owner_x = np.array(rows, float).reshape(-1, 16)
    if not np.isfinite(candidate_x).all() or not np.isfinite(owner_x).all():
        raise ValueError('nonfinite model feature')
    return candidate_x, owner_x, [int(s['track_id']) for s in sources]


def responsibility_features(base_cost, reid_distance, geometry_cost, ti, di,
                            threshold, stage_iou_only=False):
    """Pre-semantic features for an actually committed association."""
    cost = np.asarray(base_cost, float)
    value = cost[ti, di]
    row = np.delete(cost[ti], di)
    col = np.delete(cost[:, di], ti)
    row_min = min(float(row.min()), threshold) if row.size else threshold
    col_min = min(float(col.min()), threshold) if col.size else threshold
    reid = np.asarray(reid_distance, float)[ti, di]
    geom = np.asarray(geometry_cost, float)[ti, di]
    return np.array([value, geom, reid, row_min-value, col_min-value,
                     threshold-value, np.log1p(cost.shape[0]), np.log1p(cost.shape[1]),
                     float(stage_iou_only)], float)
