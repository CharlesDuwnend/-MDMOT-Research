#!/usr/bin/env python3
"""Independent saved-row verification using grouped trajectory segments."""
import hashlib
import json
import time
from collections import Counter,defaultdict
from pathlib import Path

import numpy as np

HERE=Path(__file__).resolve().parent
FIT=['23','25','28','29','39','44','45','51','53','63','66','69','70','74','78']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(1048576),b''):h.update(data)
    return h.hexdigest()


def save(path,obj):path.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')


def main():
    start=time.monotonic();summary=json.loads((HERE/'summary.json').read_text())
    assert summary['pairs']==FIT and summary['sequence_count']==30
    assert summary['horizons']==[1,4,8] and summary['detector_score_min']==.1
    assert summary['calibration_dev_official_val_test_data_used_in_computation'] is False
    assert summary['gpu_used'] is False and summary['images_read'] is False
    assert summary['training_or_inference_executed'] is False
    inputs=json.loads((HERE/'INPUTS.json').read_text())
    for path,record in inputs.items():
        assert sha(path)==record['sha256'],path
        if record['kind'].startswith('fit_'):
            assert Path(path).name.split('-')[0] in FIT,path
    assert sorted(p.stem for p in (HERE/'rows').glob('*.npz'))==sorted(f'{p}-{v}' for p in FIT for v in ('1','2'))
    total=Counter();verified=[]
    for record in summary['sequences']:
        seq=record['sequence']
        with np.load(HERE/'rows'/f'{seq}.npz',allow_pickle=False) as z:a={k:z[k] for k in z.files}
        n=len(a['frame']);assert all(len(v)==n for v in a.values())
        assert np.all(a['frame']>=1) and np.all(a['frame']<=record['frame_end'])
        assert np.all(np.diff(a['frame'])>=0)
        assert n==record['counts']['gt_observations']
        det=a['current_detected'];miss=~det
        assert len(set(zip(a['frame'].tolist(),a['raw_id'].tolist(),a['class'].tolist())))==n
        assert len(set(zip(a['frame'][det].tolist(),a['matched_detector_index'][det].tolist())))==int(det.sum())
        assert np.all(a['matched_detector_index'][miss]==-1) and np.all(a['matched_label_row'][miss]==-1)
        assert np.all(a['matched_iou'][det]>=.5) and np.all(a['matched_iou'][miss]==0)
        assert np.all(a['prior_detector_frame']<a['frame'])
        assert np.all(a['seen_prior_1']<=a['seen_prior_4']) and np.all(a['seen_prior_4']<=a['seen_prior_8'])
        assert np.all(a['prior_detector_frame'][a['segment_start'].astype(bool)]==-1)
        width=a['bbox_x2']-a['bbox_x1'];height=a['bbox_y2']-a['bbox_y1']
        assert np.allclose(np.minimum(width,height),a['minimum_side_px'],atol=.0005)
        assert np.array_equal(np.where(a['minimum_side_px']<16,0,np.where(a['minimum_side_px']<32,1,2)),a['minside_bin'])
        groups=defaultdict(list)
        for i,(rid,cls) in enumerate(zip(a['raw_id'],a['class'])):groups[(int(rid),int(cls))].append(i)
        computed={k:np.zeros(n,np.int32) for k in ('first_annotation','stream_start_left_censored','segment_start','reappearance',
            'segment_start_frame','annotation_age','prior_detector_frame','prior_detector_age','seen_prior_1','seen_prior_4','seen_prior_8',
            'first_detector_in_segment','first_detector_local_lifetime','prefix_miss_run_age','miss_category')}
        expected_runs=[]
        for ids in groups.values():
            ids=np.array(ids);frames=a['frame'][ids];curr=det[ids]
            assert np.all(np.diff(frames)>0)
            computed['first_annotation'][ids[0]]=1
            computed['stream_start_left_censored'][ids[0]]=int(frames[0]==1)
            local_detected=np.flatnonzero(curr)
            if len(local_detected):computed['first_detector_local_lifetime'][ids[local_detected[0]]]=1
            bounds=np.r_[0,np.flatnonzero(np.diff(frames)!=1)+1,len(ids)]
            for begin,end in zip(bounds[:-1],bounds[1:]):
                seg=ids[begin:end];fs=a['frame'][seg];ds=det[seg];length=len(seg)
                computed['segment_start'][seg[0]]=1
                computed['reappearance'][seg[0]]=int(begin>0)
                computed['segment_start_frame'][seg]=fs[0]
                computed['annotation_age'][seg]=np.arange(1,length+1)
                detected_positions=np.flatnonzero(ds)
                if len(detected_positions):computed['first_detector_in_segment'][seg[detected_positions[0]]]=1
                # Search in sorted detected frame numbers, with side='left' excluding CURRENT.
                seen_frames=fs[detected_positions]
                where=np.searchsorted(seen_frames,fs,side='left')-1
                prior=np.full(length,-1,np.int32)
                valid=where>=0
                prior[valid]=seen_frames[where[valid]]
                age=np.where(valid,fs-prior,-1)
                computed['prior_detector_frame'][seg]=prior
                computed['prior_detector_age'][seg]=age
                for h in (1,4,8):computed[f'seen_prior_{h}'][seg]=valid&(age<=h)
                padded=np.r_[False,~ds,False].astype(np.int8)
                starts=np.flatnonzero(np.diff(padded)==1)
                ends=np.flatnonzero(np.diff(padded)==-1)
                for left,right in zip(starts,ends):
                    computed['prefix_miss_run_age'][seg[left:right]]=np.arange(1,right-left+1)
                    expected_runs.append((int(a['raw_id'][seg[left]]),int(a['class'][seg[left]]),int(fs[left]),int(fs[right-1]),right-left))
                category=np.where(ds,0,np.where(~valid,3,np.where(age<=8,5,4)))
                if not ds[0]:category[0]=1 if begin==0 else 2
                computed['miss_category'][seg]=category
        for k,value in computed.items():
            if not np.array_equal(value,a[k]):
                bad=np.flatnonzero(value!=a[k]);raise AssertionError(f'{seq} {k} mismatches={len(bad)} first={bad[0]}')
        assert int(det.sum())==record['counts']['current_detected']
        assert int(miss.sum())==record['counts']['current_missing']
        for h in (1,4,8):assert int(np.count_nonzero(miss&a[f'seen_prior_{h}'].astype(bool)))==record['counts'][f'missing_prior_{h}']
        for field in ('class','occluded','minside_bin'):
            for val,stats in record['counts'][field].items():
                mask=a[field]==int(val)
                assert int(mask.sum())==stats['gt_observations']
                assert int(np.count_nonzero(mask&miss))==stats['current_missing']
                for h in (1,4,8):assert int(np.count_nonzero(mask&miss&a[f'seen_prior_{h}'].astype(bool)))==stats[f'missing_prior_{h}']
        with np.load(HERE/'miss_runs'/f'{seq}.npz',allow_pickle=False) as z:runs={k:z[k] for k in z.files}
        actual_runs=list(zip(*(runs[k].tolist() for k in ('raw_id','class','start_frame','end_frame','length'))))
        assert sorted(actual_runs)==sorted(expected_runs)
        assert int(runs['length'].sum())==int(miss.sum())
        assert np.all(runs['end_frame']-runs['start_frame']+1==runs['length'])
        total.update({'gt_observations':n,'current_detected':int(det.sum()),'current_missing':int(miss.sum()),
                      **{f'missing_prior_{h}':int(np.count_nonzero(miss&a[f'seen_prior_{h}'].astype(bool))) for h in (1,4,8)}})
        verified.append({'sequence':seq,'rows':n,'trajectories':len(groups),'miss_runs':len(expected_runs),'all_temporal_columns_equal':True})
    for key,value in total.items():assert value==summary['totals'][key]
    cross=Counter()
    for pair in FIT:
        with np.load(HERE/'cross_view'/f'{pair}.npz',allow_pickle=False) as z:c={k:z[k] for k in z.files}
        views=[]
        for v in ('1','2'):
            with np.load(HERE/'rows'/f'{pair}-{v}.npz',allow_pickle=False) as z:views.append({k:z[k] for k in z.files})
        for j,view in enumerate(views,1):
            take=c[f'view{j}_row']
            for k in ('frame','raw_id','class'):assert np.array_equal(c[k],view[k][take])
            assert np.array_equal(c[f'current_v{j}'],view['current_detected'][take])
        assert np.array_equal(c['both_missing'],~(c['current_v1'].astype(bool)|c['current_v2'].astype(bool)))
        for h in (1,4,8):
            one=views[0][f'seen_prior_{h}'][c['view1_row']].astype(bool)
            two=views[1][f'seen_prior_{h}'][c['view2_row']].astype(bool)
            assert np.array_equal(c[f'prior_{h}_either'],one|two)
            assert np.array_equal(c[f'prior_{h}_both'],one&two)
        both=c['both_missing'].astype(bool)
        cross.update({'coannotated_same_label_equal_id_observations':len(both),'both_current_missing':int(both.sum()),
                      **{f'both_missing_prior_{h}_{how}':int(np.count_nonzero(both&c[f'prior_{h}_{how}'].astype(bool))) for h in (1,4,8) for how in ('either','both')}})
    assert dict(cross)==summary['cross_view_totals_ANNOTATION_CONVENTION_ONLY']
    tests=json.loads((HERE/'INVARIANT_TESTS.json').read_text());assert tests['status']=='PASS'
    verify={'status':'PASS','synthetic':tests,'independent_algorithm':'group by local identity/class, split annotation gaps, strict-left searchsorted on detected frame numbers',
            'input_hashes_rechecked':len(inputs),'sequences':verified,'total_rows_verified':total['gt_observations'],
            'all_temporal_columns_verified':True,'one_to_one_verified':True,'cross_view_saved_row_reference_verified':True,
            'full_miss_run_mass_and_intervals_verified':True,'elapsed_seconds':round(time.monotonic()-start,3)}
    save(HERE/'VERIFY.json',verify)
    outputs={}
    for path in sorted(HERE.rglob('*')):
        if path.is_file() and path.name not in ('RECEIPT.json','RECEIPT.sha256','verify.log'):
            outputs[str(path.relative_to(HERE))]={'sha256':sha(path),'bytes':path.stat().st_size}
    receipt={'schema':'p27_visibility_receipt_v1','status':'PASS_FIT_ORACLE_OBSERVATION_DIAGNOSTIC',
             'python_executable':'/home/chenhc/.conda/envs/remdet_paper/bin/python','fit_pairs':FIT,
             'inputs':inputs,'outputs':outputs,'verification_status':'PASS',
             'compute_scope':'FIT labels and raw XML only, no images/GPU/model inference/training',
             'threshold_restriction':'old stage1 score >= 0.1; exact P26 parity not claimed',
             'gt_history_use':'offline Oracle observation diagnostics only; no deployable recovery or method metric claim'}
    save(HERE/'RECEIPT.json',receipt)
    (HERE/'RECEIPT.sha256').write_text(sha(HERE/'RECEIPT.json')+'  RECEIPT.json\n')
    print(json.dumps({'status':'PASS','rows':total['gt_observations'],'input_files':len(inputs),'output_files':len(outputs),'receipt_sha256':sha(HERE/'RECEIPT.json')}))


if __name__=='__main__':main()
