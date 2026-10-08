"""Canonical U2MOT online entry point with dynamic SRC/EGIA wrapper.

The detector and ReID path are inherited from the canonical repository's
``tools/track.py``. Only the tracker object and its causal metadata adapter are
replaced. This script is intended for a frozen one-pass test-dev run.
"""
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

CANONICAL = Path(os.environ.get('UAVDT_ONLINE_HOST', '/home/chenhc/20250715/u2mot')).resolve()
ROOT = Path(__file__).resolve().parents[1]
# ``belief_data_common`` deliberately hides CUDA for cached CPU replay.  The
# online entry point must preserve the caller's explicit device binding before
# importing the shared runtime modules, otherwise the canonical model is moved
# to CUDA after that module has blanked the environment.
_ONLINE_CUDA_VISIBLE_DEVICES = os.environ.get('CUDA_VISIBLE_DEVICES')
os.environ.setdefault('BELIEF_HOST', str(CANONICAL))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(CANONICAL))

spec = importlib.util.spec_from_file_location('canonical_u2mot_track', CANONICAL / 'tools/track.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

from belief.online import OnlineBeliefTracker, tracker_args  # noqa: E402
from belief.runtime import load_bundle  # noqa: E402

if _ONLINE_CUDA_VISIBLE_DEVICES is not None:
    os.environ['CUDA_VISIBLE_DEVICES'] = _ONLINE_CUDA_VISIBLE_DEVICES


ORIGINAL_MAKE_PARSER = base.make_parser


def make_parser():
    parser = ORIGINAL_MAKE_PARSER()
    parser.add_argument('--belief-bundle', required=True, type=str)
    parser.add_argument('--belief-egia-model', default='', type=str)
    parser.add_argument('--belief-arm', choices=['native', 'src', 'egia', 'both',
                                                   'pair_shuffle', 'hard_update',
                                                   'path_no_src', 'path_update_only',
                                                   'path_cost_only', 'path_full',
                                                   'fc_control', 'fc_mask_only',
                                                   'fc_penalty_only', 'fc_full'], default='both')
    parser.add_argument('--belief-max-frames', type=int, default=None)
    return parser


base.make_parser = make_parser


def image_track(predictor, vis_folder, res_folder, video_path, args):
    files = base.get_image_list(video_path)
    files.sort()
    video_name = base.osp.basename(video_path)
    if args.cmc_method != 'none':
        args.cmc_seq_name = video_name
    result_path = base.osp.join(res_folder, video_name + '.txt')
    bundle = load_bundle(args.belief_bundle)
    pure_v5 = (bundle.get('source_architecture') == 'coverage_hierarchical_coherence_v5'
               and not bundle.get('egia_fusion_policy', False))
    if not pure_v5 and not args.belief_egia_model:
        raise ValueError('FROZEN_H2_MODEL_REQUIRED_FOR_LEGACY_POLICY')
    host_args = tracker_args(args, args.belief_egia_model, pure_v5=pure_v5)
    tracker = OnlineBeliefTracker(host_args, frame_rate=args.fps,
                                  arm=args.belief_arm, bundle=bundle)
    tracker.sequence_name = video_name
    timer = base.Timer()
    dataloader = base.create_data_reader(files, predictor.test_size)
    stride = min(predictor.model.head.strides)
    results = []
    previous_crop_requests = []
    emitted_frame_track_pairs = set()
    crop_enabled = bool(getattr(args, 'track_crop_rescue', False))
    crop_delayed = bool(getattr(args, 'track_crop_delayed_confirmation', False))
    started = time.monotonic()
    for frame_id, (img, img_info) in enumerate(dataloader, 1):
        if args.belief_max_frames is not None and frame_id > args.belief_max_frames:
            break
        if frame_id == 1 or frame_id % 100 == 0:
            print(json.dumps({'sequence': video_name, 'frame': frame_id, 'total_frames': len(files)}), flush=True)
        outputs, reid_feat, img_info = predictor.inference(img, img_info, timer)
        bound_crop_detections = []
        if crop_enabled and previous_crop_requests:
            bound_crop_detections = predictor.inference_track_crops(
                img_info['raw_img'], previous_crop_requests,
                predictor.last_standard_outputs[0])
        if outputs[0] is None and not bound_crop_detections:
            # Keep the canonical host's empty-frame skip, while including the
            # skipped detector/ReID input in the paired evidence receipt.
            empty_features = reid_feat[0].cpu().numpy()
            tracker._record_online_inputs(
                np.empty((0, 7), dtype=np.float32),
                np.empty((0, empty_features.shape[-1]), dtype=empty_features.dtype),
                (img_info['height'], img_info['width']), predictor.test_size, frame_id)
            timer.toc()
            previous_crop_requests = []
            continue
        if outputs[0] is None:
            outputs = np.empty((0, 7), dtype=np.float32)
        else:
            outputs = outputs[0].cpu().numpy()
            outputs = outputs[np.argsort(outputs[:, 0])]
        reid_feat = reid_feat[0].cpu().numpy()
        ny, nx = reid_feat.shape[:2]
        if len(outputs):
            x1, y1, x2, y2 = outputs[:, :4].T / stride
            xc = ((x1 + x2) / 2.).clip(0, nx - 1).astype('int32')
            yc = ((y1 + y2) / 2.).clip(0, ny - 1).astype('int32')
            embeddings = reid_feat[yc, xc, :]
        else:
            embeddings = np.empty((0, reid_feat.shape[-1]), dtype=reid_feat.dtype)
        if bound_crop_detections:
            centers = np.asarray([
                ((item['tlbr'][0] + item['tlbr'][2]) * 0.5,
                 (item['tlbr'][1] + item['tlbr'][3]) * 0.5)
                for item in bound_crop_detections], dtype=np.float64)
            scale = min(float(predictor.test_size[0]) / float(img_info['height']),
                        float(predictor.test_size[1]) / float(img_info['width']))
            crop_x = np.clip(centers[:, 0] * scale / stride, 0, nx - 1).astype('int32')
            crop_y = np.clip(centers[:, 1] * scale / stride, 0, ny - 1).astype('int32')
            crop_features = predictor.refine_embeddings(reid_feat[crop_y, crop_x, :])
            for item, feature in zip(bound_crop_detections, crop_features):
                item['feat'] = feature
        tracks = tracker.update_online(
            outputs, (img_info['height'], img_info['width']), predictor.test_size,
            embeddings, frame_id, use_uncertainty=args.use_uncertainty,
            img=img_info['raw_img'], bound_crop_detections=bound_crop_detections)
        next_crop_requests = []
        if crop_delayed:
            for pending in tracker.drain_confirmed_crop_outputs():
                tlwh = np.asarray(pending['tlwh'], dtype=np.float64)
                if (tlwh[2] * tlwh[3] > args.min_box_area and
                        tlwh[2] / tlwh[3] < args.aspect_ratio_thresh and
                        tlwh[3] / tlwh[2] < 4.):
                    key = (int(pending['frame_id']), int(pending['track_id']))
                    if key not in emitted_frame_track_pairs:
                        results.append(
                            f"{pending['frame_id']},{pending['track_id']},{tlwh[0]:.2f},{tlwh[1]:.2f},"
                            f"{tlwh[2]:.2f},{tlwh[3]:.2f},{pending['score']:.2f},"
                            f"{int(pending['class_id'])+1},-1,-1\n")
                        emitted_frame_track_pairs.add(key)
        for track in tracks:
            tlwh = track.tlwh
            if (tlwh[2] * tlwh[3] > args.min_box_area and
                    tlwh[2] / tlwh[3] < args.aspect_ratio_thresh and
                    tlwh[3] / tlwh[2] < 4.):
                results.append(
                    f"{frame_id},{track.track_id},{tlwh[0]:.2f},{tlwh[1]:.2f},"
                    f"{tlwh[2]:.2f},{tlwh[3]:.2f},{track.score:.2f},"
                    f"{int(track.cls)+1},-1,-1\n")
            if (crop_enabled and track.is_activated and
                    not (crop_delayed and
                         getattr(track, 'last_update_source', '') == 'crop')):
                next_crop_requests.append({
                    'track_id': int(track.track_id),
                    'tracklet_len': int(getattr(track, 'tracklet_len', 0)),
                    'class_id': int(track.cls),
                    'tlwh': np.asarray(tlwh, dtype=np.float64).copy(),
                })
        previous_crop_requests = next_crop_requests if crop_enabled else []
        timer.toc()
    Path(result_path).write_text(''.join(results))
    report = tracker.runtime_report()
    report.update({
        'sequence': video_name, 'arm': args.belief_arm,
        'frames_seen': min(len(files), args.belief_max_frames or len(files)),
        'output_rows': len(results), 'elapsed_seconds': time.monotonic() - started,
        'formal_metrics': False, 'official_eval': False,
        'canonical_host': str(CANONICAL),
    })
    Path(result_path + '.runtime.json').write_text(json.dumps(report, indent=2, sort_keys=True))
    gmc = getattr(tracker, 'gmc', None)
    if gmc is not None and getattr(gmc, 'gmcFile', None) is not None:
        gmc.gmcFile.close()


base.image_track = image_track


if __name__ == '__main__':
    parsed = base.make_parser().parse_args()
    experiment = base.get_exp(parsed.exp_file, parsed.name)
    base.main(experiment, parsed)
