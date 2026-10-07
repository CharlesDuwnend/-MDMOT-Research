"""Online adapter for running BeliefTracker inside canonical U2MOT track.py."""
import copy
import hashlib
import json
from types import SimpleNamespace

import numpy as np

from belief.runtime import BeliefTracker


class OnlineBeliefTracker(BeliefTracker):
    """Prepare causal observation metadata around the host's normal update.

    The canonical tracker owns frame advancement, cascades, Kalman state and
    output tracks. This adapter only supplies the same observation/ordinal
    ledger that cached replay uses, then commits the learned belief after the
    host update has finished.
    """

    @staticmethod
    def _array_input_digest(array):
        """Hash exact ordered input values without a dtype conversion."""
        value = np.asarray(array)
        header = json.dumps({'dtype': value.dtype.str, 'shape': list(value.shape)},
                            sort_keys=True, separators=(',', ':')).encode()
        digest = hashlib.sha256()
        digest.update(len(header).to_bytes(8, 'little'))
        digest.update(header)
        digest.update(value.tobytes(order='C'))
        return digest.hexdigest()

    def _record_online_inputs(self, detections, embeddings, img_info, img_size,
                              frame_number):
        """Record per-sequence detector/ReID evidence before the host update.

        This ledger never feeds a model, cost, memory update, or output sorter.
        The ledger stores hashes and array metadata only, avoiding full-stream
        dumps while retaining exact per-frame comparisons between arms.
        """
        sequence = str(self.sequence_name)
        if not hasattr(self, '_input_stream_hasher'):
            self._input_stream_sequence = sequence
            self._input_stream_hasher = hashlib.sha256(b'leaf-online-input-v1\0')
            self._input_stream_records = []
        if self._input_stream_sequence != sequence:
            raise AssertionError('ONLINE_INPUT_SEQUENCE_CHANGED')
        if (self._input_stream_records and
                int(frame_number) <= self._input_stream_records[-1]['frame']):
            raise AssertionError('ONLINE_INPUT_FRAME_ORDER')
        detector = np.asarray(detections)
        appearance = np.asarray(embeddings)
        record = {
            'sequence': sequence, 'frame': int(frame_number),
            'img_info': np.asarray(img_info).tolist(),
            'img_size': np.asarray(img_size).tolist(),
            'detector_dtype': detector.dtype.str,
            'detector_shape': list(detector.shape),
            'embedding_dtype': appearance.dtype.str,
            'embedding_shape': list(appearance.shape),
            'detector_sha256': self._array_input_digest(detector),
            'embedding_sha256': self._array_input_digest(appearance),
        }
        frame_payload = json.dumps(record, sort_keys=True, separators=(',', ':')).encode()
        record['frame_input_sha256'] = hashlib.sha256(frame_payload).hexdigest()
        self._input_stream_hasher.update(bytes.fromhex(record['frame_input_sha256']))
        record['rolling_input_sha256'] = self._input_stream_hasher.hexdigest()
        self._input_stream_records.append(record)

    def runtime_report(self):
        report = super().runtime_report()
        records = getattr(self, '_input_stream_records', [])
        report.update({
            'input_hash_schema': 'leaf-online-input-v1',
            'input_frames_hashed': len(records),
            'input_stream_sha256': (self._input_stream_hasher.hexdigest()
                                    if hasattr(self, '_input_stream_hasher') else None),
            'input_records': records,
        })
        return report

    def update_online(self, detections, img_info, img_size, embeddings, frame_number,
                      use_uncertainty=False, img=None, bound_crop_detections=None):
        if use_uncertainty:
            raise ValueError('ONLINE_BELIEF_UNCERTAINTY_UNSUPPORTED')
        self._record_online_inputs(detections, embeddings, img_info, img_size,
                                   frame_number)
        detections = np.asarray(detections).copy()
        self.source_frame = int(frame_number)
        self.stage = 0
        self.new_candidates = []
        self.det_ordinals = {}
        self._egia_frame_height = float(img_info[0])
        self._egia_frame_width = float(img_info[1])

        boxes = detections[:, :4].copy()
        scale = min(float(img_size[0]) / self._egia_frame_height,
                    float(img_size[1]) / self._egia_frame_width)
        boxes /= scale
        boxes[:, 2:] -= boxes[:, :2]
        self.frame_observations = {
            i: {
                'sequence': self.sequence_name,
                'frame': int(frame_number),
                'source_frame': int(frame_number),
                'detection_ordinal': i,
                'bbox_tlwh': np.asarray(boxes[i], dtype=np.float64).tolist(),
                'score': float(row[4]),
                'class_confidence': float(row[5]),
                'predicted_class': int(row[6]),
                'frame_width': int(img_info[1]),
                'frame_height': int(img_info[0]),
            }
            for i, row in enumerate(detections)
        }
        self.high_ordinals = np.flatnonzero(detections[:, 4] > self.track_high_thresh)
        self.low_ordinals = np.flatnonzero(
            (detections[:, 4] > self.track_low_thresh) &
            (detections[:, 4] < self.track_high_thresh)
        )
        update_kwargs = {
            'embeddings': np.asarray(embeddings).copy(),
            'use_uncertainty': False,
            'img': img,
        }
        # The frozen canonical host has no crop argument.  The crop-capable
        # legacy host exposes ``track_crop_rescue`` and accepts the extra
        # payload; keep the ordinary host call byte-compatible.
        if bound_crop_detections or hasattr(self, 'track_crop_rescue'):
            update_kwargs['bound_crop_detections'] = bound_crop_detections or []
        tracks = self.update(
            detections, (int(img_info[0]), int(img_info[1])), img_size,
            **update_kwargs
        )
        self._flush_committed()
        for track, observation in self.new_candidates:
            if track.start_frame != self.frame_id:
                raise AssertionError('ONLINE_BELIEF_BIRTH_FRAME_MISMATCH')
            tid = int(track.track_id)
            self.observed[tid] = observation
            if self.src_state_enabled:
                self.beliefs[tid] = self.model.observation_probabilities([observation])[0]
                self.counts['src_observation_probability_calls'] += 1
                self.counts['src_observation_probability_rows'] += 1
                self.counts['belief_initializations'] += 1
        return tracks


def tracker_args(args, egia_model, pure_v5=False):
    """Make a shallow host-args copy for the selected method policy."""
    out = copy.copy(args)
    out.continuous_semantic_reliability = not pure_v5
    out.egia_model = '' if pure_v5 else str(egia_model)
    out.egia_capture_dir = ''
    return out
