#!/usr/bin/env python3
"""Offline FIT-only flow geometry audit, without images or model inference."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
import csv,hashlib,importlib.util,json,sys,time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
P28=HERE.parent
LABELS=Path('/home/chenhc/mdmot_research_20261002/p0/data')
RAFT_SOURCE=Path('/home/chenhc/.conda/envs/remdet_paper/lib/python3.11/site-packages/torchvision/models/optical_flow/raft.py')
FIT=['23','25','28','29','39','44','45','51','53','63','66','69','70','74','78']
INPUTS={}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def bind(path,expected=None):
    path=Path(path).resolve();digest=sha(path)
    if expected is not None:assert digest==expected,str(path)
    INPUTS[str(path)]=digest
    return path


def read(path,expected=None):return json.loads(bind(path,expected).read_text())
def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def sample_flow(flow,centers,original_hw):
    """Same halfpixel and border convention as P28's flow sampling."""
    assert flow.ndim==3 and flow.shape[0]==2
    centers=np.asarray(centers,np.float64).reshape(-1,2)
    oh,ow=original_hw;fh,fw=flow.shape[-2:]
    xy=centers*np.array([fw/ow,fh/oh])-0.5
    x=np.clip(xy[:,0],0,fw-1);y=np.clip(xy[:,1],0,fh-1)
    x0=np.floor(x).astype(int);y0=np.floor(y).astype(int)
    x1=np.minimum(x0+1,fw-1);y1=np.minimum(y0+1,fh-1)
    wx=x-x0;wy=y-y0
    sampled=(flow[:,y0,x0]*(1-wx)*(1-wy)+flow[:,y0,x1]*wx*(1-wy)+
             flow[:,y1,x0]*(1-wx)*wy+flow[:,y1,x1]*wx*wy).T
    displacement=sampled*np.array([ow/fw,oh/fh])
    return centers+displacement,displacement,sampled,xy


def torch_sample(flow,centers,original_hw,dtype):
    import torch
    from torch.nn import functional as F
    centers=np.asarray(centers,np.float64).reshape(-1,2)
    oh,ow=original_hw
    grid=torch.as_tensor(centers/np.array([ow,oh])*2-1,dtype=dtype)[None,:,None,:]
    value=F.grid_sample(torch.as_tensor(flow,dtype=dtype)[None],grid,mode='bilinear',padding_mode='border',align_corners=False)
    return value[0,:,:,0].T.numpy()*np.array([ow/flow.shape[2],oh/flow.shape[1]])


def eligible(frames,t,lag,rid,cls):
    past=frames.get(t-lag,{}).get((rid,cls))
    if past is None:return None,'past_endpoint_absent'
    if any((rid,cls) not in frames.get(f,{}) for f in range(t-lag+1,t)):
        return None,'interior_annotation_gap'
    return past,'eligible'


def tests():
    import torch
    torch.set_num_threads(2)
    found=[]
    f=np.zeros((2,36,64),np.float64);f[0]=2;f[1]=-3
    points=np.array([[100.,200.],[.1,.1],[1919.9,1079.9]])
    projected,delta,_,_=sample_flow(f,points,(1080,1920))
    assert np.allclose(delta,np.array([60.,-90.]),rtol=0,atol=1e-12)
    assert np.allclose(projected,points+[60,-90],rtol=0,atol=1e-12)
    assert np.linalg.norm((points-delta)-(points+[60,-90]),axis=1).min()>0
    found+=['known_current_to_past_translation_plus_sign_and_scale','constant_flow_border_extension']
    z=sample_flow(np.zeros_like(f),points,(1080,1920))[0]
    assert np.array_equal(z,points);found+=['zero_flow_identity']
    yy,xx=np.mgrid[:36,:64];ramp=np.stack([.02*xx+.1,.03*yy-.2])
    center=np.array([[200.,300.],[700.,500.]])
    p,d,s,xy=sample_flow(ramp,center,(1080,1920))
    assert np.allclose(s,np.stack([.02*xy[:,0]+.1,.03*xy[:,1]-.2],axis=1),rtol=0,atol=1e-14)
    found+=['spatial_ramp_exact_halfpixel_mapping']
    rng=np.random.default_rng(19);random_flow=rng.normal(size=(2,36,64))
    points=np.r_[rng.uniform([0,0],[1920,1080],size=(80,2)),[[0,0],[1920,1080]]]
    delta=sample_flow(random_flow,points,(1080,1920))[1]
    expected=torch_sample(random_flow,points,(1080,1920),torch.float64)
    error=float(np.max(np.abs(delta-expected)))
    assert error<1e-10;found+=['independent_cpu_grid_sample_halfpixel_parity']
    frames={1:{(3,0):{}},2:{(3,0):{}},4:{(3,0):{}},5:{(3,0):{}}}
    assert eligible(frames,2,1,3,0)[1]=='eligible'
    assert eligible(frames,5,4,3,0)[1]=='interior_annotation_gap'
    assert eligible(frames,5,1,3,1)[1]=='past_endpoint_absent'
    found+=['continuous_annotation_gap_and_class_identity_guard']
    return {'status':'PASS','tests':found,'grid_sample_max_component_error_original_px':error,'GPU_used':False}


def stats(a,mask):
    n=int(mask.sum())
    if not n:return {'n':0}
    answer={'n':n,'raft_better_than_zero_fraction':float(np.mean(a['raft_error_px'][mask]<a['zero_error_px'][mask])),
            'current_center_outside_count':int(np.count_nonzero(~a['current_center_inside'][mask])),
            'past_center_outside_count':int(np.count_nonzero(~a['past_center_inside'][mask])),
            'raft_endpoint_outside_count':int(np.count_nonzero(~a['raft_endpoint_inside'][mask]))}
    for key in ('raft_error_px','zero_error_px','raft_error_over_current_sqrt_area','zero_error_over_current_sqrt_area',
                'raft_error_over_past_sqrt_area','zero_error_over_past_sqrt_area'):
        v=a[key][mask];answer[key]={'mean':float(v.mean()),'median':float(np.median(v)),'p90':float(np.quantile(v,.9))}
    answer['mean_paired_error_reduction_px']=float(np.mean(a['zero_error_px'][mask]-a['raft_error_px'][mask]))
    return answer


def main():
    import torch
    start=time.monotonic();test=tests();dump(HERE/'TESTS.json',test)
    bind(__file__);bind(HERE/'PROTOCOL.md')
    training=read(P28/'training/TRAIN_RECEIPT.json');train_inputs=training['inputs']
    for path in (LABELS/'build_detection_labels.py',LABELS/'sequences_train.json'):
        bind(path,train_inputs[str(path)])
    manifest={(x['pair'],str(x['view'])):x for x in read(LABELS/'sequences_train.json')}
    spec=importlib.util.spec_from_file_location('p28_fit_motion_XML_reader',LABELS/'build_detection_labels.py')
    parser=importlib.util.module_from_spec(spec);spec.loader.exec_module(parser)
    config=read(P28/'CONFIG.json');assert config['fit_pairs']==FIT
    groups=[g for g in read(P28/'GROUPS.json') if g['role']=='fit'];assert len(groups)==240
    receipt=read(P28/'fit/FLOW_RECEIPT.json')
    index=read(P28/'fit/FLOW_INDEX.json',receipt['flow_index_sha256'])
    protocol=read(P28/'fit/FLOW_PROTOCOL.json',receipt['protocol_sha256'])
    assert index['role']==receipt['role']==protocol['role']=='fit'
    assert protocol['lags']==[1,4,8] and protocol['groups']==receipt['groups']==len(groups)
    bind(P28/'export_flow.py',receipt['code_sha256']);bind(P28/'train_adapter.py');bind(P28/'temporal_module.py')
    bind(P28/'GROUPS.json',protocol['groups_sha256']);bind(P28/'FRAMES.json',protocol['frames_sha256'])
    source=bind(RAFT_SOURCE).read_text();util=bind(RAFT_SOURCE.parent/'_utils.py').read_text()
    assert 'flow=(coords1 - coords0)' in source and 'self.corr_block.build_pyramid(fmap1, fmap2)' in source
    assert 'coords[::-1]' in util
    flow_export=(P28/'export_flow.py').read_text()
    assert 'model(tensors[0],ref,num_flow_updates=12)' in flow_export
    direction=dict(status='PASS_SOURCE_DIRECTION_CONTRACT',export='RAFT(current,past)',
        raft_output='coords1-coords0; channels dx,dy',projection='current_center + sampled_current_to_past_displacement',
        original_center_to_flow_index='center_xy * (flow_width/original_width,flow_height/original_height) - 0.5',
        border_mode='border',align_corners=False,source_sha256=sha(RAFT_SOURCE),runtime_current_torch=torch.__version__)
    dump(HERE/'DIRECTION_AUDIT.json',direction)
    assert set(index['by_key'])=={g['key'] for g in groups}
    rows=[];group_info=[];exclude=Counter();max_numeric=0.;field_shapes=set()
    for pair in FIT:
        for view in ('1','2'):
            record=manifest[(pair,view)];xp=bind(record['xml_path'],record['xml_sha256'])
            assert train_inputs[str(xp)]==record['xml_sha256']
            original_frames,tracks,_=parser.read_xml(pair,view,record['xml_sha256'])
            frames={f:{(g['raw_id'],g['class']):g for g in boxes} for f,boxes in original_frames.items()}
            for group in [g for g in groups if g['pair']==pair and str(g['view'])==view]:
                t=int(group['frame']);assert group['frames']==[t,t-1,t-4,t-8]
                ent=index['entries'][index['by_key'][group['key']]]
                assert ent['image_keys']==group['image_keys'] and ent['frame']==t
                fp=bind(ent['path'],ent['sha256']);assert receipt['dependencies'][str(fp)]==ent['sha256']
                with np.load(fp,allow_pickle=False) as z:
                    flow=z['flow'];hw=z['original_hw'].tolist();fhw=z['flow_img_hw'].tolist()
                    assert z['lags'].tolist()==[1,4,8]
                assert hw==ent['original_hw'] and fhw==ent['flow_img_hw']
                assert flow.shape==(3,2,*fhw) and flow.dtype==np.float32 and np.isfinite(flow).all()
                field_shapes.add(tuple(hw+fhw));oh,ow=hw
                counts=Counter()
                current=[g for g in original_frames.get(t,[]) if g['class'] in (0,1,2)]
                counts['current_supported_annotations']=len(current)
                for li,lag in enumerate((1,4,8)):
                    pairs=[]
                    for cur in current:
                        past,reason=eligible(frames,t,lag,cur['raw_id'],cur['class'])
                        counts[f'lag{lag}_{reason}']+=1
                        if reason!='eligible':exclude[f'lag{lag}_{reason}']+=1;continue
                        pairs.append((cur,past))
                    if not pairs:continue
                    centers=np.array([[(a['bbox'][0]+a['bbox'][2])/2,(a['bbox'][1]+a['bbox'][3])/2] for a,b in pairs])
                    past_centers=np.array([[(b['bbox'][0]+b['bbox'][2])/2,(b['bbox'][1]+b['bbox'][3])/2] for a,b in pairs])
                    projected,disp,sampled,xy=sample_flow(flow[li],centers,hw)
                    numeric=torch_sample(flow[li],centers,hw,torch.float32)
                    numeric_error=float(np.max(np.abs(numeric-disp)));max_numeric=max(max_numeric,numeric_error)
                    assert numeric_error<=.01,'Sampler numerical mismatch'
                    for j,(cur,past) in enumerate(pairs):
                        cb=np.array(cur['bbox']);pb=np.array(past['bbox']);cw,ch=cb[2:]-cb[:2];pw,ph=pb[2:]-pb[:2]
                        cs,ps=np.sqrt(cw*ch),np.sqrt(pw*ph)
                        er=float(np.linalg.norm(projected[j]-past_centers[j]));ez=float(np.linalg.norm(centers[j]-past_centers[j]))
                        row=dict(pair=int(pair),view=int(view),frame=t,past_frame=t-lag,lag=lag,raw_id=cur['raw_id'],class_id=cur['class'],
                            current_occluded=int(bool(cur['occluded'])),past_occluded=int(bool(past['occluded'])),
                            path_any_occluded=int(any(frames[f][(cur['raw_id'],cur['class'])]['occluded'] for f in range(t-lag,t+1))),
                            current_min_side=min(cw,ch),past_min_side=min(pw,ph),current_sqrt_area=cs,past_sqrt_area=ps,
                            current_min_side_lt16=min(cw,ch)<16,current_sqrt_area_lt16=cs<16,
                            current_center_inside=bool(np.all(centers[j]>=0) and np.all(centers[j]<[ow,oh])),
                            past_center_inside=bool(np.all(past_centers[j]>=0) and np.all(past_centers[j]<[ow,oh])),
                            raft_endpoint_inside=bool(np.all(projected[j]>=0) and np.all(projected[j]<[ow,oh])),
                            current_cx=centers[j,0],current_cy=centers[j,1],past_cx=past_centers[j,0],past_cy=past_centers[j,1],
                            flow_index_x=xy[j,0],flow_index_y=xy[j,1],flow_dx=sampled[j,0],flow_dy=sampled[j,1],
                            original_dx=disp[j,0],original_dy=disp[j,1],projected_past_cx=projected[j,0],projected_past_cy=projected[j,1],
                            raft_error_px=er,zero_error_px=ez,raft_error_over_current_sqrt_area=er/cs,zero_error_over_current_sqrt_area=ez/cs,
                            raft_error_over_past_sqrt_area=er/ps,zero_error_over_past_sqrt_area=ez/ps)
                        rows.append(row)
                group_info.append(dict(key=group['key'],pair=pair,view=view,frame=t,counts=dict(counts)))
            print(json.dumps({'sequence':f'{pair}-{view}','groups_done':len(group_info),'comparisons_so_far':len(rows)}),flush=True)
    a={k:np.asarray([r[k] for r in rows]) for k in rows[0]}
    assert len(set(zip(a['pair'],a['view'],a['frame'],a['lag'],a['raw_id'],a['class_id'])))==len(rows)
    assert np.all(a['past_frame']==a['frame']-a['lag']) and len(group_info)==240
    np.savez_compressed(HERE/'observations.npz',**a)
    strata={'overall':stats(a,np.ones(len(rows),bool)),'by_lag':{},'by_pair':{},'lag_size_occlusion':{}}
    for lag in (1,4,8):
        lm=a['lag']==lag;out={'all':stats(a,lm)}
        for field,vals in [('current_min_side_lt16',[False,True]),('current_sqrt_area_lt16',[False,True]),
                           ('current_occluded',[0,1]),('past_occluded',[0,1]),('path_any_occluded',[0,1]),('class_id',[0,1,2])]:
            out[field]={str(v):stats(a,lm&(a[field]==v)) for v in vals}
        strata['by_lag'][str(lag)]=out
        for sizefield in ('current_min_side_lt16','current_sqrt_area_lt16'):
            for small in (False,True):
                for occ in (0,1):strata['lag_size_occlusion'][f'{lag}:{sizefield}={small}:occluded={occ}']=stats(a,lm&(a[sizefield]==small)&(a['current_occluded']==occ))
    for pair in FIT:strata['by_pair'][pair]={str(lag):stats(a,(a['pair']==int(pair))&(a['lag']==lag)) for lag in (1,4,8)}
    summary=dict(status='PASS_FIT_OFFLINE_FLOW_CENTER_DIAGNOSTIC',groups=240,fit_pairs=FIT,comparisons=len(rows),
        original_and_flow_hw=[list(x) for x in sorted(field_shapes)],exclusions=dict(exclude),group_counts=group_info,
        max_numpy64_vs_grid_sample32_displacement_component_difference_original_px=max_numeric,
        no_GPU_images_model_inference_calibration_GT=True,source_direction_verified=True,synthetic_contract_tests='PASS',
        strata=strata,elapsed_seconds=round(time.monotonic()-start,3))
    dump(HERE/'SUMMARY.json',summary);dump(HERE/'INPUTS.json',INPUTS)
    for path,digest in INPUTS.items():assert sha(path)==digest,'Input changed: '+path
    print(json.dumps({'status':summary['status'],'comparisons':len(rows),'exclusions':dict(exclude),'by_lag':{k:v['all'] for k,v in strata['by_lag'].items()},'seconds':summary['elapsed_seconds']}),flush=True)


if __name__=='__main__':main()
