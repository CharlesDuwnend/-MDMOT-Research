"""Train-only pixel-derived observability probe; not a tracking method.

Local IDs only index observations. Features are sealed before optional labels.
No device calibration, physical synchronization, or planar-scene truth is assumed.
"""
import sys
sys.dont_write_bytecode = True
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
OLD = Path('/home/chenhc/mdmot_research_20261002')
sys.path.insert(0, str(OLD / 'p2/src'))
from data import PairData, split_assignments

PAIRS = ['23', '25', '29', '69', '78']
CONFIG = {'pairs': PAIRS, 'frames_per_pair': 8, 'lag': 8, 'max_image_side': 960,
          'sift_features': 2000, 'ratio': 0.75, 'ransac_pixels': 3.0,
          'min_inliers': 12, 'min_inlier_fraction': 0.25, 'min_hull_fraction': 0.02,
          'max_cycle_error_pixels': 3.0, 'bbox_mask_expansion_fraction': 0.10,
          'foreground_position': 'bbox bottom center', 'seed': 20261004,
          'no_tuning': True, 'no_GPU': True, 'no_dev_calibration_val_test': True}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def warp(matrix, points):
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.column_stack((points, np.ones(len(points)))) @ matrix.T
    denominator = homogeneous[:, 2:3]
    if np.any(np.abs(denominator) < 1e-8):
        raise ValueError('homography approaches infinity')
    return homogeneous[:, :2] / denominator


def jacobian(matrix, point):
    point = np.asarray(point, dtype=np.float64)
    denominator = matrix[2, :2] @ point + matrix[2, 2]
    mapped = warp(matrix, point[None])[0]
    return (matrix[:2, :2] - mapped[:, None] * matrix[2, :2]) / denominator


def feature_image(pair, view, frame, rows):
    path = Path('/raid/datasets/chc_data/MDMT/train') / view / (pair + '-' + view) / ('%08d.jpg' % frame)
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError('unreadable train image: ' + str(path))
    scale = min(1.0, CONFIG['max_image_side'] / max(gray.shape))
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    mask = np.full(gray.shape, 255, np.uint8)
    for row in rows:
        box = np.asarray(row['bbox'], dtype=np.float64) * scale
        width, height = box[2] - box[0], box[3] - box[1]
        if width <= 0 or height <= 0:
            continue
        expand = CONFIG['bbox_mask_expansion_fraction']
        low = np.floor(box[:2] - expand * np.array([width, height])).astype(int)
        high = np.ceil(box[2:] + expand * np.array([width, height])).astype(int)
        low = np.maximum(low, 0)
        high = np.minimum(high, np.array(gray.shape[::-1]) - 1)
        cv2.rectangle(mask, tuple(low), tuple(high), 0, -1)
    keypoints, descriptors = cv2.SIFT_create(nfeatures=CONFIG['sift_features']).detectAndCompute(gray, mask)
    points = np.array([k.pt for k in keypoints], dtype=np.float64).reshape(-1, 2)
    return {'points': points, 'descriptors': descriptors, 'shape': gray.shape,
            'scale': scale, 'path': str(path), 'sha256': sha(path),
            'unmasked_fraction': float(np.mean(mask > 0))}


def geometry(source, target):
    result = {'valid': False, 'matches': 0, 'inliers': 0, 'reason': 'too_few_descriptors'}
    if source['descriptors'] is None or target['descriptors'] is None:
        return None, result
    if min(len(source['points']), len(target['points'])) < 4:
        return None, result
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    def neighbors(left, right):
        pairs = matcher.knnMatch(left, right, k=2)
        return {m.queryIdx: m.trainIdx for pair in pairs if len(pair) == 2
                for m, n in [pair] if m.distance < CONFIG['ratio'] * n.distance}
    forward = neighbors(source['descriptors'], target['descriptors'])
    reverse = neighbors(target['descriptors'], source['descriptors'])
    matches = [(i, j) for i, j in forward.items() if reverse.get(j) == i]
    result['matches'] = len(matches)
    if len(matches) < CONFIG['min_inliers']:
        result['reason'] = 'too_few_mutual_background_matches'
        return None, result
    left = source['points'][[i for i, _ in matches]]
    right = target['points'][[j for _, j in matches]]
    matrix, mask = cv2.findHomography(left, right, cv2.RANSAC, CONFIG['ransac_pixels'], maxIters=5000, confidence=0.995)
    if matrix is None or mask is None or not np.isfinite(matrix).all():
        result['reason'] = 'homography_fit_failed'
        return None, result
    mask = mask[:, 0].astype(bool)
    result['inliers'] = int(mask.sum())
    result['inlier_fraction'] = float(mask.mean())
    if mask.sum() < CONFIG['min_inliers']:
        result['reason'] = 'too_few_inliers'
        return None, result
    hulls = [cv2.contourArea(cv2.convexHull(points[mask].astype(np.float32))) / np.prod(meta['shape'])
             for points, meta in [(left, source), (right, target)]]
    result['hull_fractions'] = [float(x) for x in hulls]
    residual = np.linalg.norm(warp(matrix, left) - right, axis=1)
    result['inlier_error_median'] = float(np.median(residual[mask]))
    result['inlier_error_p90'] = float(np.percentile(residual[mask], 90))
    result['valid'] = (result['inlier_fraction'] >= CONFIG['min_inlier_fraction']
                       and min(hulls) >= CONFIG['min_hull_fraction'])
    result['reason'] = 'supported_background_homography' if result['valid'] else 'weak_spatial_support'
    # These inlier residuals are fit diagnostics, not calibrated uncertainty.
    result['uncertainty_calibrated'] = False
    return (matrix if result['valid'] else None), result


def centers(rows, scale):
    return {r['local_track_id_index']: np.array([(r['bbox'][0] + r['bbox'][2]) / 2, r['bbox'][3]]) * scale
            for r in rows if not r['zero_area']}


def run_pair(pair):
    cv2.setNumThreads(1)
    cv2.setRNGSeed(CONFIG['seed'])
    data = PairData(pair, with_labels=False)
    assert data.role == 'fit'
    frames = np.unique(np.linspace(CONFIG['lag'] + 1, data.frames, CONFIG['frames_per_pair'], dtype=int))
    output = HERE / ('pair' + pair)
    output.mkdir(exist_ok=False)
    events = []
    for frame in frames:
        current, history = data.frame(int(frame)), data.frame(int(frame) - CONFIG['lag'])
        images, byview = {}, {}
        for camera in ('1', '2'):
            for age, packet in [('now', current), ('old', history)]:
                rows = [r for r in packet['rows'] if r['camera'] == camera]
                byview[(camera, age)] = rows
                time_index = int(frame) if age == 'now' else int(frame) - CONFIG['lag']
                images[(camera, age)] = feature_image(pair, camera, time_index, rows)
        transforms, diagnostics = {}, {}
        edges = {'time1': (('1', 'old'), ('1', 'now')),
                 'time2': (('2', 'old'), ('2', 'now')),
                 'cross_now': (('1', 'now'), ('2', 'now')),
                 'cross_old': (('1', 'old'), ('2', 'old'))}
        for name, (source, target) in edges.items():
            transforms[name], diagnostics[name] = geometry(images[source], images[target])
        event = {'pair': pair, 'frame': int(frame), 'history_frame': int(frame) - CONFIG['lag'],
                 'geometry': diagnostics, 'images': [{k: image[k] for k in ['path', 'sha256', 'unmasked_fraction']}
                                                   for image in images.values()],
                 'feature_rows': 0, 'status': 'GEOMETRY_UNAVAILABLE', 'PID_read': False}
        eligible = {camera: [] for camera in ('1', '2')}
        for camera in ('1', '2'):
            now_rows, old_rows = byview[(camera, 'now')], byview[(camera, 'old')]
            old_centers = centers(old_rows, images[(camera, 'old')]['scale'])
            now_centers = centers(now_rows, images[(camera, 'now')]['scale'])
            old_classes = {r['local_track_id_index']: r['class'] for r in old_rows}
            for row in now_rows:
                local = row['local_track_id_index']
                if local in old_centers and local in now_centers and old_classes[local] == row['class']:
                    eligible[camera].append((row, old_centers[local], now_centers[local]))
        event['history_eligible_per_view'] = {v: len(eligible[v]) for v in eligible}
        if all(matrix is not None for matrix in transforms.values()):
            left_points = np.array([item[2] for item in eligible['1']]).reshape(-1, 2)
            # Compare two background routes. This checks geometry only, never identity evidence.
            route_a = transforms['cross_now'] @ transforms['time1']
            route_b = transforms['time2'] @ transforms['cross_old']
            old_left = np.array([item[1] for item in eligible['1']]).reshape(-1, 2)
            cycle_errors = np.linalg.norm(warp(route_a, old_left) - warp(route_b, old_left), axis=1)
            event['geometry_cycle_error_median'] = float(np.median(cycle_errors)) if len(cycle_errors) else None
            features = []
            for li, (left, left_old, left_now) in enumerate(eligible['1']):
                if cycle_errors[li] > CONFIG['max_cycle_error_pixels']:
                    continue
                residual_left = left_now - warp(transforms['time1'], left_old[None])[0]
                transported = jacobian(transforms['cross_now'], left_now) @ residual_left
                for right, right_old, right_now in eligible['2']:
                    if left['class'] != right['class']:
                        continue
                    residual_right = right_now - warp(transforms['time2'], right_old[None])[0]
                    position_error = np.linalg.norm(warp(transforms['cross_now'], left_now[None])[0] - right_now)
                    noise = diagnostics['time1']['inlier_error_p90'] + diagnostics['time2']['inlier_error_p90']
                    features.append({'left_key': left['key'], 'right_key': right['key'], 'class': left['class'],
                                     'residual_left': residual_left.tolist(), 'residual_right': residual_right.tolist(),
                                     'transported_left': transported.tolist(),
                                     'motion_error_pixels': float(np.linalg.norm(transported - residual_right)),
                                     'static_position_error_pixels': float(position_error),
                                     'residual_magnitude_left': float(np.linalg.norm(residual_left)),
                                     'residual_magnitude_right': float(np.linalg.norm(residual_right)),
                                     'background_fit_error_proxy': noise,
                                     'geometry_cycle_error_pixels': float(cycle_errors[li])})
            event['feature_rows'] = len(features)
            event['status'] = 'PIXEL_RESIDUAL_FEATURES_EXTRACTED' if features else 'NO_SUPPORTED_TARGET_HISTORY'
            dump(output / ('features_%04d.json' % frame), features)
        events.append(event)
        dump(output / 'EVENTS.json', events)
        print(json.dumps({k: event[k] for k in ['pair', 'frame', 'status', 'feature_rows', 'history_eligible_per_view']}), flush=True)
    return events


def tests():
    matrix = np.array([[1.2, -0.2, 4.0], [0.3, 0.9, -3.0], [0.0002, -0.0001, 1.0]])
    point = np.array([100., 80.])
    numeric = np.column_stack([(warp(matrix, (point + axis * 1e-4)[None])[0] - warp(matrix, (point - axis * 1e-4)[None])[0]) / 2e-4 for axis in np.eye(2)])
    assert np.max(np.abs(numeric - jacobian(matrix, point))) < 1e-7
    affine = np.array([[1.1, -.1, 20.], [.1, 1.1, 10.], [0., 0., 1.]])
    current = warp(affine, point[None])[0]
    assert np.linalg.norm(current - warp(affine, point[None])[0]) == 0
    motion = np.array([3., -2.])
    assert np.allclose((current + motion) - warp(affine, point[None])[0], motion)
    return {'status': 'PASS', 'jacobian_finite_difference': True, 'static_target_zero_residual': True,
            'known_added_motion_recovered': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if not args.run:
        if (HERE / 'PROTOCOL.json').exists():
            raise ValueError('protocol already exists')
        assert set(PAIRS) <= set(split_assignments()['fit'])
        protocol = dict(CONFIG, created_utc=datetime.now(timezone.utc).isoformat(),
                        hypothesis='Background-separated target motion may add identity-specific cross-view evidence',
                        inference_identity_labels=False, common_coordinate_or_hardware_sync_assumed=False,
                        gate={'at_least_pairs_with_supported_geometry': 3, 'min_feature_edges': 200,
                              'remaining_gate': 'known-label true-vs-distractor motion evidence beyond static geometry, evaluated after feature freeze'},
                        failure_scope='SIFT/background-homography construction only; cannot rule out a learned geometry model',
                        novelty='HOLD: background compensation and Jacobian transport alone are established operators',
                        code_sha256=sha(Path(__file__)))
        dump(HERE / 'PROTOCOL.json', protocol)
        dump(HERE / 'TESTS.json', tests())
        print('PROTOCOL_AND_GEOMETRY_TESTS_WRITTEN')
        return
    protocol = json.loads((HERE / 'PROTOCOL.json').read_text())
    assert sha(Path(__file__)) == protocol['code_sha256']
    assert json.loads((HERE / 'TESTS.json').read_text())['status'] == 'PASS'
    started, results = time.monotonic(), {}
    with ProcessPoolExecutor(max_workers=2) as pool:
        jobs = {pool.submit(run_pair, pair): pair for pair in PAIRS}
        for job in as_completed(jobs):
            results[jobs[job]] = job.result()
    events = [event for pair in PAIRS for event in results[pair]]
    supported_pairs = [pair for pair in PAIRS if any(e['feature_rows'] for e in results[pair])]
    counts = {'image_reads': len(events) * 4, 'frames': len(events), 'feature_edges': sum(e['feature_rows'] for e in events),
              'supported_pairs': supported_pairs, 'supported_frames': sum(e['feature_rows'] > 0 for e in events)}
    receipt = {'status': 'FEATURES_FROZEN_BEFORE_LABELS', 'config': CONFIG, 'counts': counts,
               'geometry_gate_pass': len(supported_pairs) >= 3 and counts['feature_edges'] >= 200,
               'PID_read': False, 'raw_XML_read': False, 'cal_dev_val_test_read': False, 'gpu_used': False,
               'elapsed_seconds': time.monotonic() - started,
               'sources': [{'path': str(OLD / 'p2/src/data.py'), 'sha256': sha(OLD / 'p2/src/data.py')},
                           {'path': str(OLD / 'p2/cache/MANIFEST.json'), 'sha256': sha(OLD / 'p2/cache/MANIFEST.json')}],
               'files': [{'path': str(p), 'sha256': sha(p)} for p in sorted(HERE.rglob('*'))
                         if p.is_file() and p.suffix in ['.py', '.json']],
               'completed_utc': datetime.now(timezone.utc).isoformat()}
    dump(HERE / 'FEATURE_RECEIPT.json', receipt)
    (HERE / 'runner.exit').write_text('0\n')
    print(json.dumps({'status': receipt['status'], 'counts': counts, 'geometry_gate_pass': receipt['geometry_gate_pass']}), flush=True)


if __name__ == '__main__':
    main()
