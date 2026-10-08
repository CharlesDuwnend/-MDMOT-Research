"""Causal features for an association-side foreground/clutter head."""
import numpy as np


FEATURES = [
    'score', 'class_confidence', 'max_active_iou', 'max_lost_iou',
    'active_track_count', 'lost_track_count', 'bbox_area_ratio',
    'border_distance_ratio',
]

# Pair-conditioned features for the association residual head.  These are
# deliberately defined on the current detection and the already-held owner
# state.  They never use a future frame or a GT-derived identity at runtime.
PAIR_FEATURES = [
    'base_cost', 'geometry_cost', 'appearance_distance',
    'detection_score', 'detection_class_confidence',
    'track_score', 'track_class_confidence', 'class_match',
    'track_active', 'log_gap', 'log_age', 'detection_area_ratio',
]


def iou_xywh(a, b):
    ax, ay, aw, ah = [float(v) for v in a]
    bx, by, bw, bh = [float(v) for v in b]
    left, top = max(ax, bx), max(ay, by)
    right, bottom = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(right - left, 0.) * max(bottom - top, 0.)
    area_a = max(aw, 0.) * max(ah, 0.)
    area_b = max(bw, 0.) * max(bh, 0.)
    return inter / max(area_a + area_b - inter, 1e-12)


def features_from_records(detection, tracks, frame_width, frame_height):
    """Use only current-frame detection and already-held track records."""
    active = [t for t in tracks if bool(t.get('is_activated', False))]
    lost = [t for t in tracks if not bool(t.get('is_activated', False))]
    box = detection['bbox_tlwh']
    x, y, w, h = [float(v) for v in box]
    fw, fh = max(float(frame_width), 1.), max(float(frame_height), 1.)
    area = max(w, 0.) * max(h, 0.)
    border = min(x, y, max(fw - x - w, 0.), max(fh - y - h, 0.))
    return np.asarray([
        float(detection['score']),
        float(detection['class_confidence']),
        max((iou_xywh(box, t['bbox_tlwh']) for t in active), default=0.),
        max((iou_xywh(box, t['bbox_tlwh']) for t in lost), default=0.),
        np.log1p(len(active)), np.log1p(len(lost)),
        area / (fw * fh), border / max(fw, fh),
    ], dtype=float)


def features_from_runtime(native_features):
    """Project BeliefTracker's causal runtime dictionary to this head."""
    return np.asarray([
        float(native_features[k]) for k in FEATURES
    ], dtype=float)


def pair_features_from_runtime(track, detection, base_cost, geometry_cost,
                               appearance_distance, threshold):
    """Build causal features for one existing-owner/detection edge."""
    box = detection['bbox_tlwh']
    x, y, w, h = [float(v) for v in box]
    fw = max(float(detection.get('frame_width', 1.)), 1.)
    fh = max(float(detection.get('frame_height', 1.)), 1.)
    gap = track.get('gap')
    gap = 0.0 if gap is None else max(float(gap), 0.0)
    age = max(float(track.get('age', 1.)), 1.0)
    # Keep the native threshold in the feature contract so a model cannot be
    # silently reused at a different host operating point.
    return np.asarray([
        float(base_cost) / max(float(threshold), 1e-6),
        float(geometry_cost),
        float(appearance_distance),
        float(detection['score']),
        float(detection['class_confidence']),
        float(track.get('score', 0.0)),
        float(track.get('class_confidence', track.get('score', 0.0))),
        float(int(track.get('predicted_class', -1)) == int(detection.get('predicted_class', -2))),
        float(bool(track.get('is_activated', False))),
        np.log1p(gap),
        np.log1p(age),
        max(w, 0.) * max(h, 0.) / (fw * fh),
    ], dtype=float)
