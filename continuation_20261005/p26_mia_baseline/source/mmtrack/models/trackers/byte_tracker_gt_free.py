"""GT-free restoration of the ByteTrack path shipped in the MDMT snapshot.

The repository's active ``ByteTracker`` was modified for the MIA demo: it
requires externally injected ``bil`` boxes/IDs and disables the normal memo
update.  The original upstream-style implementation is still present at the
top of ``byte_tracker.py`` as commented source.  This module restores that
code under an explicit name; it does not read annotations or allocate IDs from
anything other than detections.  Matching parameters come unchanged from the
released MDMT ByteTrack config.
"""
import lap
import numpy as np
import torch
from mmcv.runner import force_fp32
from mmdet.core import bbox_overlaps

from mmtrack.core.bbox import bbox_cxcyah_to_xyxy, bbox_xyxy_to_cxcyah
from mmtrack.models import TRACKERS
from .base_tracker import BaseTracker


@TRACKERS.register_module()
class ByteTrackerGTF(BaseTracker):
    """Official-config ByteTrack lifecycle with detection-only ID births."""

    def __init__(self,
                 obj_score_thrs=dict(high=0.6, low=0.1),
                 init_track_thr=0.7,
                 weight_iou_with_det_scores=True,
                 match_iou_thrs=dict(high=0.1, low=0.5, tentative=0.3),
                 num_tentatives=3,
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg=init_cfg, **kwargs)
        self.obj_score_thrs = obj_score_thrs
        self.init_track_thr = init_track_thr
        self.weight_iou_with_det_scores = weight_iou_with_det_scores
        self.match_iou_thrs = match_iou_thrs
        self.num_tentatives = num_tentatives

    @property
    def confirmed_ids(self):
        return [track_id for track_id, track in self.tracks.items()
                if not track.tentative]

    @property
    def unconfirmed_ids(self):
        return [track_id for track_id, track in self.tracks.items()
                if track.tentative]

    def init_track(self, track_id, obj):
        super().init_track(track_id, obj)
        self.tracks[track_id].tentative = (
            self.tracks[track_id].frame_ids[-1] != 0)
        bbox = bbox_xyxy_to_cxcyah(self.tracks[track_id].bboxes[-1])
        assert bbox.ndim == 2 and bbox.shape[0] == 1
        self.tracks[track_id].mean, self.tracks[track_id].covariance = (
            self.kf.initiate(bbox.squeeze(0).cpu().numpy()))

    def update_track(self, track_id, obj):
        super().update_track(track_id, obj)
        if self.tracks[track_id].tentative:
            if len(self.tracks[track_id].bboxes) >= self.num_tentatives:
                self.tracks[track_id].tentative = False
        bbox = bbox_xyxy_to_cxcyah(self.tracks[track_id].bboxes[-1])
        assert bbox.ndim == 2 and bbox.shape[0] == 1
        self.tracks[track_id].mean, self.tracks[track_id].covariance = (
            self.kf.update(self.tracks[track_id].mean,
                           self.tracks[track_id].covariance,
                           bbox.squeeze(0).cpu().numpy()))

    def pop_invalid_tracks(self, frame_id):
        invalid_ids = []
        for track_id, track in self.tracks.items():
            expired = frame_id - track.frame_ids[-1] >= self.num_frames_retain
            missed_tentative = (track.tentative and
                                track.frame_ids[-1] != frame_id)
            if expired or missed_tentative:
                invalid_ids.append(track_id)
        for track_id in invalid_ids:
            self.tracks.pop(track_id)

    def assign_ids(self,
                   ids,
                   det_bboxes,
                   weight_iou_with_det_scores=False,
                   match_iou_thr=0.5):
        track_bboxes = np.zeros((0, 4))
        for track_id in ids:
            track_bboxes = np.concatenate(
                (track_bboxes, self.tracks[track_id].mean[:4][None]), axis=0)
        track_bboxes = torch.from_numpy(track_bboxes).to(det_bboxes)
        track_bboxes = bbox_cxcyah_to_xyxy(track_bboxes)
        ious = bbox_overlaps(track_bboxes, det_bboxes[:, :4])
        if weight_iou_with_det_scores:
            ious *= det_bboxes[:, 4][None]
        dists = (1 - ious).cpu().numpy()
        if dists.size > 0:
            _, row, col = lap.lapjv(
                dists, extend_cost=True, cost_limit=1 - match_iou_thr)
        else:
            row = np.zeros(len(ids), dtype=np.int32) - 1
            col = np.zeros(len(det_bboxes), dtype=np.int32) - 1
        return row, col

    @force_fp32(apply_to=('img', 'bboxes'))
    def track(self,
              img,
              img_metas,
              model,
              bboxes,
              labels,
              frame_id,
              rescale=False,
              **kwargs):
        if not hasattr(self, 'kf'):
            self.kf = model.motion

        if self.empty or bboxes.size(0) == 0:
            valid_inds = bboxes[:, -1] > self.init_track_thr
            bboxes = bboxes[valid_inds]
            labels = labels[valid_inds]
            num_new_tracks = bboxes.size(0)
            ids = torch.arange(self.num_tracks,
                               self.num_tracks + num_new_tracks).to(labels)
            self.num_tracks += num_new_tracks
        else:
            ids = torch.full((bboxes.size(0),), -1,
                             dtype=labels.dtype, device=labels.device)
            first_det_inds = bboxes[:, -1] > self.obj_score_thrs['high']
            first_det_bboxes = bboxes[first_det_inds]
            first_det_labels = labels[first_det_inds]
            first_det_ids = ids[first_det_inds]
            second_det_inds = (~first_det_inds) & (
                bboxes[:, -1] > self.obj_score_thrs['low'])
            second_det_bboxes = bboxes[second_det_inds]
            second_det_labels = labels[second_det_inds]
            second_det_ids = ids[second_det_inds]

            for track_id in self.confirmed_ids:
                if self.tracks[track_id].frame_ids[-1] != frame_id - 1:
                    self.tracks[track_id].mean[7] = 0
                self.tracks[track_id].mean, self.tracks[track_id].covariance = (
                    self.kf.predict(self.tracks[track_id].mean,
                                    self.tracks[track_id].covariance))

            first_match_track_inds, first_match_det_inds = self.assign_ids(
                self.confirmed_ids, first_det_bboxes,
                self.weight_iou_with_det_scores,
                self.match_iou_thrs['high'])
            valid = first_match_det_inds > -1
            first_det_ids[valid] = torch.tensor(
                self.confirmed_ids)[first_match_det_inds[valid]].to(labels)
            first_match_det_bboxes = first_det_bboxes[valid]
            first_match_det_labels = first_det_labels[valid]
            first_match_det_ids = first_det_ids[valid]
            first_unmatch_det_bboxes = first_det_bboxes[~valid]
            first_unmatch_det_labels = first_det_labels[~valid]
            first_unmatch_det_ids = first_det_ids[~valid]

            _, tentative_match_det_inds = self.assign_ids(
                self.unconfirmed_ids, first_unmatch_det_bboxes,
                self.weight_iou_with_det_scores,
                self.match_iou_thrs['tentative'])
            valid = tentative_match_det_inds > -1
            first_unmatch_det_ids[valid] = torch.tensor(self.unconfirmed_ids)[
                tentative_match_det_inds[valid]].to(labels)

            first_unmatch_track_ids = []
            for i, track_id in enumerate(self.confirmed_ids):
                if (first_match_track_inds[i] == -1 and
                        self.tracks[track_id].frame_ids[-1] == frame_id - 1):
                    first_unmatch_track_ids.append(track_id)
            _, second_match_det_inds = self.assign_ids(
                first_unmatch_track_ids, second_det_bboxes, False,
                self.match_iou_thrs['low'])
            valid = second_match_det_inds > -1
            second_det_ids[valid] = torch.tensor(first_unmatch_track_ids)[
                second_match_det_inds[valid]].to(ids)

            valid = second_det_ids > -1
            bboxes = torch.cat((first_match_det_bboxes,
                                first_unmatch_det_bboxes,
                                second_det_bboxes[valid]), dim=0)
            labels = torch.cat((first_match_det_labels,
                                first_unmatch_det_labels,
                                second_det_labels[valid]), dim=0)
            ids = torch.cat((first_match_det_ids,
                             first_unmatch_det_ids,
                             second_det_ids[valid]), dim=0)
            new_track_inds = ids == -1
            ids[new_track_inds] = torch.arange(
                self.num_tracks,
                self.num_tracks + int(new_track_inds.sum())).to(labels)
            self.num_tracks += int(new_track_inds.sum())

        self.update(ids=ids, bboxes=bboxes, labels=labels, frame_ids=frame_id)
        return bboxes, labels, ids
