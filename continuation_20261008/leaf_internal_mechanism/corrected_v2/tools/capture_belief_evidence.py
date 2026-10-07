"""GT-free shadow capture at frozen H2 association and birth seams."""
from belief_data_common import *
import argparse
import copy
import sys
import time
from collections import Counter
from types import SimpleNamespace
import numpy as np
sys.path.insert(0, str(HOST))
from yolox.tracker import u2mot_tracker as host_module
from yolox.tracker import matching
from yolox.tracker.gmc import GMC
from yolox.tracker.basetrack import BaseTrack
from yolox.tracker.u2mot_tracker import U2MOTTracker, STrack


class AbsoluteGMC(GMC):
    def __init__(self, method='file', downscale=2, verbose=None):
        assert method == 'file'
        super().__init__(method='none', downscale=downscale, verbose=verbose)
        self.method = method
        self.gmcFile = (HOST/'yolox/tracker/GMC_files'/verbose[1]/('GMC-'+verbose[0]+'.txt')).open('r')


def make_tracker(payload, cls=U2MOTTracker):
    original = host_module.GMC
    host_module.GMC = AbsoluteGMC
    try: return cls(SimpleNamespace(**payload['args']), frame_rate=payload['args']['fps'])
    finally: host_module.GMC = original


def render(tracks, frame, args):
    result = []
    for track in tracks:
        b = track.tlwh
        if b[2]*b[3] > args['min_box_area'] and b[2]/b[3] < args['aspect_ratio_thresh'] and b[3]/b[2] < 4.:
            result.append(f"{frame},{track.track_id},{b[0]:.2f},{b[1]:.2f},{b[2]:.2f},{b[3]:.2f},{track.score:.2f},{int(track.cls)+1},-1,-1\n")
    return ''.join(result)


def finite_list(a):
    x = np.asarray(a)
    assert np.isfinite(x).all()
    return x.tolist()


class CaptureTracker(U2MOTTracker):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.observed = {}
        self.det_ordinals = {}
        self.streams = {}
        self.counts = Counter()

    @staticmethod
    def detection_identity_matches(det, row):
        """Match a new detection to its immutable detector-row metadata."""
        try:
            box = np.asarray(getattr(det, '_tlwh', det.tlwh), dtype=float)
            original_box = np.asarray(row['bbox_tlwh'], dtype=float)
            values = np.asarray([float(det.score), float(det.semantic_score),
                                 float(row['score']), float(row['class_confidence'])])
            return bool(box.shape == (4,) and original_box.shape == (4,) and
                        np.isfinite(box).all() and np.isfinite(original_box).all() and
                        np.isfinite(values).all() and
                        np.array_equal(box, original_box) and
                        int(det.cls) == int(row['predicted_class']) and
                        values[0] == values[2] and values[1] == values[3])
        except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
            return False

    def det_record(self, det):
        ordinal = self.det_ordinals[id(det)]
        row = dict(self.frame_observations[ordinal])
        assert np.array_equal(np.asarray(row['bbox_tlwh']), det._tlwh), 'DETECTION_ORDINAL_GEOMETRY_MISMATCH'
        if not self.detection_identity_matches(det, row):
            raise AssertionError('DETECTION_ORDINAL_METADATA_MISMATCH')
        row['semantic_evidence'] = finite_list(
            getattr(det, 'semantic_group_evidence', [0.0, 1.0]))
        return row

    def track_record(self, track):
        previous = self.observed.get(int(track.track_id))
        return {'track_id': int(track.track_id), 'bbox_tlwh': finite_list(track.tlwh),
                'state': int(track.state), 'is_activated': bool(track.is_activated),
                'start_frame': int(track.start_frame), 'last_update_frame': int(track.frame_id),
                'age': int(self.frame_id-track.start_frame+1),
                'gap': None if previous is None else int(self.frame_id-previous['frame']),
                'score': float(track.score), 'predicted_class': int(track.cls),
                'class_confidence': float(getattr(track, 'semantic_score', track.score)),
                'semantic_evidence': finite_list(
                    getattr(track, 'semantic_group_evidence', [0.0, 1.0])),
                'last_observation': copy.deepcopy(previous),
                'missing_observation': previous is None,
                'missing_appearance': track.smooth_feat is None}

    def associate(self, trks, dets, thresh, fuse_score=False, iou_only=False,
                  diagnostic_stage=None):
        stage = self.stage; self.stage += 1
        # Canonical U2MOT can split the first cascade into tracked/lost and
        # later residual stages.  Detection order therefore cannot be inferred
        # from a two-list high/low partition.  Resolve each object against the
        # frame's original-image box and reserve each ordinal once.
        used = set(self.det_ordinals.values())
        rows = getattr(self, 'frame_observations', {})
        for det in dets:
            if id(det) in self.det_ordinals:
                continue
            box = np.asarray(getattr(det, 'tlwh', ()), dtype=float)
            candidates = [int(i) for i, row_value in rows.items()
                          if int(i) not in used and np.allclose(
                              box, np.asarray(row_value['bbox_tlwh'], dtype=float),
                              rtol=0., atol=1e-4)]
            if not candidates:
                candidates = [int(i) for i in rows if int(i) not in used]
            if not candidates:
                raise AssertionError('DETECTION_ORDINAL_UNRESOLVED')
            ordinal = candidates[0]
            self.det_ordinals[id(det)] = ordinal
            used.add(ordinal)
        assert all(id(d) in self.det_ordinals for d in dets)
        geometry = matching.iou_distance(trks, dets)
        geom_cost = matching.fuse_score(geometry.copy(), dets) if fuse_score else geometry.copy()
        appearance = matching.embedding_distance(trks, dets)
        base = geom_cost.copy()
        if not iou_only:
            emb = appearance.copy()
            if len(self.super_cls) > 1: emb = matching.gate_cost_matrix_by_cls(emb, trks, dets, self.super_cls)
            emb[emb > self.appearance_thresh] = 1.
            if self.mask_emb_with_iou: emb[geom_cost > self.proximity_thresh] = 1.
            base = self._fuse_cost_matrix(base, emb)
        row = {'sequence': self.sequence_name, 'frame': self.frame_id,
               'source_frame': self.source_frame, 'stage': stage, 'threshold': float(thresh),
               'iou_only': bool(iou_only), 'fuse_score': bool(fuse_score),
               'tracks': [self.track_record(t) for t in trks],
               'detections': [self.det_record(d) for d in dets],
               'geometry_cost': finite_list(geometry), 'appearance_distance': finite_list(appearance),
               'base_cost': finite_list(base)}
        original = matching.linear_assignment
        def logged(cost, thresh):
            row['native_cost'] = finite_list(cost)
            return original(cost, thresh=thresh)
        matching.linear_assignment = logged
        try:
            try:
                result = super().associate(
                    trks, dets, thresh, fuse_score, iou_only,
                    diagnostic_stage=diagnostic_stage)
            except TypeError as error:
                if 'diagnostic_stage' not in str(error):
                    raise
                result = super().associate(trks, dets, thresh, fuse_score, iou_only)
        finally: matching.linear_assignment = original
        row['matches'] = np.asarray(result[0], dtype=int).reshape(-1,2).tolist()
        row['unmatched_tracks'] = np.asarray(result[1],dtype=int).tolist()
        row['unmatched_detections'] = np.asarray(result[2],dtype=int).tolist()
        emit(self.streams['association'], row); self.counts['association'] += 1
        # These matches will be committed immediately when associate returns.
        # No other seam executes between this ledger update and host update.
        for i,j in row['matches']:
            self.observed[int(trks[i].track_id)] = self.det_record(dets[j])
        return result

    def _egia_allows_birth(self, candidate, active_tracks, lost_tracks):
        values = self._egia_features(candidate, active_tracks, lost_tracks)
        ids = {}; active_ids=[]; lost_ids=[]
        for label,pool in [('active',active_tracks),('lost',lost_tracks)]:
            for track in pool:
                tid = int(track.track_id)
                (active_ids if label == 'active' else lost_ids).append(tid)
                if tid not in ids:
                    node = self.track_record(track)
                    similarity = self._egia_similarity(candidate, track)
                    node.update({'active': False, 'lost': False,
                                 'paired_iou': float(self._egia_iou(candidate,track)),
                                 'reid_similarity': None if similarity is None else float(similarity),
                                 'missing_reid': similarity is None})
                    ids[tid] = node
                ids[tid][label] = True
        prob = self.egia_model_payload['model'].predict_proba(np.asarray([[values[n] for n in self.egia_model_payload['features']]]))[0]
        parent = super()
        native_method = getattr(parent, '_egia_allows_birth', None)
        native = bool(native_method(candidate, active_tracks, lost_tracks)) if native_method else bool(
            candidate.score >= self.new_track_thresh and
            candidate.semantic_score >= self.new_track_min_class_confidence
        )
        row = {'sequence': self.sequence_name, 'frame': self.frame_id,
               'source_frame': self.source_frame, 'birth_ordinal': self.birth_ordinal,
               'detection_ordinal': self.det_ordinals[id(candidate)], 'candidate': self.det_record(candidate),
               'native_features': values, 'native_probabilities': finite_list(prob),
               'native_labels': list(self.egia_model_payload['labels']), 'native_admitted': native,
               'active_track_ids': active_ids, 'lost_track_ids': lost_ids,
               'sources': list(ids.values())}
        emit(self.streams['birth'], row); self.counts['birth'] += 1; self.birth_ordinal += 1
        self.counts['birth_sources'] += len(ids)
        self.counts['native_admitted'] += int(native)
        if native: self.new_candidates.append((candidate,self.det_record(candidate)))
        return native

    def run_frame(self, record, args):
        self.source_frame = int(record['frame']); self.stage = 0; self.birth_ordinal = 0
        self.new_candidates = []; self.det_ordinals = {}
        if not record['called_update']: return ''
        assert not record['use_uncertainty'], 'UNCERTAINTY_ASSOCIATION_NOT_CAPTURED'
        assert self.frame_id == record['tracker_frame_before']
        det = record['detections'].copy()
        boxes = det[:,:4].copy()
        scale = min(record['img_size'][0]/float(record['height']),record['img_size'][1]/float(record['width']))
        boxes /= scale
        boxes[:,2:] -= boxes[:,:2]
        self.high_ordinals = np.flatnonzero(det[:,4] > self.track_high_thresh)
        self.low_ordinals = np.flatnonzero((det[:,4] > self.track_low_thresh) & (det[:,4] < self.track_high_thresh))
        self.frame_observations = {}
        for i,d in enumerate(det):
            row = {'sequence': self.sequence_name, 'frame': self.frame_id+1,
                   'source_frame': self.source_frame, 'detection_ordinal': i,
                   'bbox_tlwh': finite_list(boxes[i].astype(np.float64)),
                   'score': float(d[4]), 'class_confidence': float(d[5]), 'predicted_class': int(d[6]),
                   'frame_width': int(record['width']), 'frame_height': int(record['height'])}
            self.frame_observations[i] = row
            emit(self.streams['observations'],row); self.counts['observations'] += 1
        tracks = self.update(det,(record['height'],record['width']),record['img_size'],
                             embeddings=record['embeddings'].copy(), use_uncertainty=False,img=None)
        for track, observation in self.new_candidates:
            assert track.start_frame == self.frame_id
            self.observed[int(track.track_id)] = observation
        assert self.stage == 3
        self.counts['frames'] += 1
        return render(tracks,record['frame'],args)


def canonical(value):
    from collections import deque
    if isinstance(value,np.ndarray): return {'array':value.tolist(),'dtype':str(value.dtype),'shape':list(value.shape)}
    if isinstance(value,np.generic): return value.item()
    if isinstance(value,dict): return {str(k):canonical(v) for k,v in sorted(value.items(),key=lambda q:str(q[0]))}
    if isinstance(value,(tuple,list)): return [canonical(v) for v in value]
    if isinstance(value,deque): return {'deque':[canonical(v) for v in value],'maxlen':value.maxlen}
    if value is None or isinstance(value,(str,float,int,bool)): return value
    if hasattr(value,'__dict__'): return {'class':type(value).__name__,'fields':canonical(vars(value))}
    raise TypeError(type(value))

def core_digest(tracker):
    gmc={k:v for k,v in vars(tracker.gmc).items() if k!='gmcFile'}
    value={'frame':tracker.frame_id,'count':BaseTrack._count,
           'pools':{k:getattr(tracker,k) for k in ('tracked_stracks','lost_stracks','removed_stracks')},
           'kalman':tracker.kalman_filter,'shared_kalman':STrack.shared_kalman,
           'gmc':gmc,'gmc_pos':tracker.gmc.gmcFile.tell()}
    return hashlib.sha256(json.dumps(canonical(value),sort_keys=True,allow_nan=True).encode()).hexdigest()

def run(sequence, out, verify_state=False):
    p=cache(sequence); directory=out/sequence; directory.mkdir(); t=make_tracker(p,CaptureTracker)
    assert t.continuous_semantic_reliability and t.egia_model_payload is not None
    streams={key:gzip.open(directory/(key+'.jsonl.gz'),'wt',compresslevel=3) for key in ('association','birth','observations')}
    t.streams=streams; output=[]; states=[]; started=time.monotonic()
    try:
        for record in p['frames']:
            output.append(t.run_frame(record,p['args']))
            if verify_state: states.append(core_digest(t))
    finally:
        for stream in streams.values(): stream.close()
        t.gmc.gmcFile.close()
    raw=''.join(output).encode(); expected=(CAP/'track_res'/(sequence+'.txt')).read_bytes()
    assert raw==expected,'H2_BYTE_PARITY_FAILED'
    if verify_state:
        native=make_tracker(p)
        for i,record in enumerate(p['frames']):
            if record['called_update']:
                native.update(record['detections'].copy(),(record['height'],record['width']),record['img_size'],embeddings=record['embeddings'].copy(),use_uncertainty=record['use_uncertainty'],img=None)
            assert core_digest(native)==states[i], 'H2_STATE_PARITY_FAILED'
        native.gmc.gmcFile.close()
    result={'sequence':sequence,'split':split_map()[sequence], 'counts':dict(t.counts),
            'source_frames':len(p['frames']),'native_byte_parity':True,'native_state_parity':True if verify_state else None,
            'output_sha256':hashlib.sha256(raw).hexdigest(), 'seconds':time.monotonic()-started,
            'input_sha256':{'cache':sha(CAP/(sequence+'.pkl.gz')),'native_output':sha(CAP/'track_res'/(sequence+'.txt')),'gmc':sha(p['gmc_source'])},
            'files_sha256':{f.name:sha(f) for f in directory.glob('*.gz')}}
    write_json(directory/'receipt.json',result); return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--sequences',nargs='*');ap.add_argument('--verify-state',action='store_true');a=ap.parse_args()
    original_cwd=os.getcwd()
    a.out=a.out.resolve(); seqs=selected_sequences(a.sequences);a.out.mkdir(parents=True,exist_ok=False)
    code=[Path(__file__),Path(__file__).with_name('belief_data_common.py')]
    hashes=snapshot_scripts(a.out,code)
    source={str(p):sha(p) for p in [HOST/'yolox/tracker/u2mot_tracker.py',HOST/'yolox/tracker/matching.py',HOST/'yolox/tracker/gmc.py',SPLIT,CAP/'receipt.json']}
    model=Path(cache(seqs[0])['args']['egia_model']);source[str(model)]=sha(model)
    assert source[str(model)]=='aa48c972a3ddbe4c5e4b82c5f2a64f9032978d7c15a863a5a3caefe67ac04ce8'
    schema_tracker=make_tracker(cache(seqs[0]));feature_order=list(schema_tracker.egia_model_payload['features']);schema_tracker.gmc.gmcFile.close()
    write_json(a.out/'SCHEMA.json',{'schema_version':1,'association_stream':'association.jsonl.gz','birth_stream':'birth.jsonl.gz','observation_stream':'observations.jsonl.gz','label_join':'sequence,source_frame,stage or detection_ordinal; labels preserve stream order','geometry':'actual original-image host seam tlwh','source_deduplication':'track_id union with active/lost membership','owner_observation':'last actually accepted detection at/before seam; no GT','native_feature_order':feature_order})
    write_json(a.out/'preflight.json',{'sequences':seqs,'source_sha256':source,'script_sha256':hashes,'gt_read':False,'cwd':original_cwd})
    results=[];start=time.monotonic()
    for seq in seqs:
        result=run(seq,a.out,a.verify_state);results.append(result)
        write_json(a.out/'progress.json',{'completed':len(results),'total':len(seqs),'results':results});print(json.dumps({'sequence':seq,'counts':result['counts'],'seconds':result['seconds']}),flush=True)
    assert all(sha(p)==h for p,h in source.items()),'SOURCE_CHANGED_DURING_CAPTURE'
    assert all(sha(p)==h for p,h in hashes.items()),'SCRIPT_CHANGED_DURING_CAPTURE'
    assert os.getcwd()==original_cwd,'CWD_CHANGED'
    write_json(a.out/'receipt.json',{'status':'PASS_H2_BELIEF_SHADOW_CAPTURE','gt_read':False,'heldout_read':False,'formal_metrics':False,'sequence_count':len(results),'results':results,'source_sha256':source,'script_sha256':hashes,'elapsed_seconds':time.monotonic()-start,'python':sys.executable,'cpu_threads':1,'cuda_visible_devices':'','cwd_unchanged':True})

if __name__=='__main__':main()
