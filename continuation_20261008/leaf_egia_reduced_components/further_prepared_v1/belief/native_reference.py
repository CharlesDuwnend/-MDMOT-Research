"""Same-state whole native association method comparison for SRC-off cells."""
import copy
import numpy as np
from belief.reference_oracle import assignment_key, snapshot
from capture_belief_evidence import U2MOTTracker

def native_oracle_for_seam(tracker,trks,dets,thresh,fuse_score,iou_only):
    assert not tracker.src_state_enabled and not tracker.pending
    before=snapshot(tracker)
    clone=object.__new__(U2MOTTracker)
    clone.__dict__=tracker.__dict__.copy()
    for k,v in tracker.__dict__.items():
        if isinstance(v,(dict,list,set)):
            clone.__dict__[k]=copy.copy(v)
    # The ordinary association method does not use GMC or the SRC/EGIA bundle.
    clone_trks,clone_dets=copy.deepcopy((trks,dets))
    result=U2MOTTracker.associate(clone,clone_trks,clone_dets,thresh,fuse_score,iou_only)
    assert snapshot(tracker)==before,'NATIVE_REFERENCE_MUTATED_ACTUAL_STATE'
    return dict(assignment=result,stage=tracker.stage,
        observations=[tracker.det_record(dets[int(j)]) for i,j in np.asarray(result[0],int).reshape(-1,2)],
        track_ids=[id(trks[int(i)]) for i,j in np.asarray(result[0],int).reshape(-1,2)])

def verify_native_reference(tracker,result,reference):
    assert assignment_key(result)==assignment_key(reference['assignment']),'NATIVE_ASSOCIATION_CHANGED'
    assert tracker.stage==reference['stage']
    assert len(tracker.pending)==len(reference['observations'])
    for actual,observation,track_id in zip(tracker.pending,reference['observations'],reference['track_ids']):
        track,obs,ratio,responsibility=actual
        assert id(track)==track_id and obs==observation
        assert ratio is None and responsibility is None
    tracker.counts['native_reference_seams_verified']+=1
