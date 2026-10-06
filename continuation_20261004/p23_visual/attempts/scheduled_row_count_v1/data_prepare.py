"""Rebuild scheduled fit crops, preserving the exact P14 input row contract."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import deque
import json
import shutil
import time
import numpy as np

HERE=Path(__file__).resolve().parent
OLD=Path('/home/chenhc/mdmot_research_20261002')
OUT=HERE/'data'
CACHE=Path('/raid/datasets/chc_data/mdmot_research_cache_20261004/p14_dino_schedule')
sys.path[:0]=[str(OLD/'p2/src'),str(OLD/'p4_diagnosis'),str(OLD/'p14_fusion')]
from data import PairData,split_assignments,sha256
from extract_schedule_dino import frame_image_map
from extract_dino_crop32 import decode_image


def dump(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    if OUT.exists():raise FileExistsError(OUT)
    OUT.mkdir(parents=True)
    start=time.monotonic()
    schedule=json.loads((OLD/'p7r/SCHEDULE.json').read_text())
    assert len(schedule)==1200
    assert {s['pair'] for s in schedule}==set(split_assignments()['fit'])
    records={r['sequence']:r for r in json.loads((OLD/'p2/cache/MANIFEST.json').read_text())['sequences']}
    metadata=[];pids=[];dinos=[];fpns=[];tasks=[];lookup={};dependencies={};expected_groups={}
    for pair in sorted(split_assignments()['fit']):
        data=PairData(pair,with_labels=True).require_fit()
        table=CACHE/(pair+'.npz');receipt=json.loads((CACHE/(pair+'_RECEIPT.json')).read_text())
        assert sha256(table)==receipt['npz']['sha256']
        dependencies[str(table)]=sha256(table)
        archive=np.load(table,allow_pickle=False)
        for view in [0,1]:
            arrays=data._views[view];indices=archive[f'view_{view}_index']
            keys=archive[f'view_{view}_key'];vectors=archive[f'view_{view}_unit_dino']
            record=records[f'{pair}-{view+1}']
            for info in [record['input_cache'],record['label_cache'],record['sources']['features']]:
                assert sha256(info['path'])==info['sha256'];dependencies[info['path']]=info['sha256']
            raw=np.load(record['sources']['features']['path'],allow_pickle=False)['visual_feature']
            mapping=frame_image_map(pair,str(view+1),set(map(int,arrays['frame'][indices])))
            by_frame={}
            for index,key,vec in zip(indices,keys,vectors):
                index=int(index);frame=int(arrays['frame'][index]);cls=int(arrays['cls'][index])
                expected=f'{pair}-{view+1}:{frame}:{int(arrays["detector_index"][index])}'
                assert str(key)==expected and arrays['feature_present'][index]
                assert str(key) not in lookup
                global_index=len(metadata);lookup[str(key)]=global_index
                metadata.append(dict(key=str(key),pair=pair,view=view,frame=frame,cls=cls,source_row_index=index,
                                     bbox=arrays['bbox'][index].tolist(),feature_present=True))
                pids.append(int(arrays['pid'][index]));dinos.append(vec);fpns.append(raw[arrays['feature_row'][index]])
                by_frame.setdefault(frame,[]).append(global_index)
            for frame,idx in sorted(by_frame.items()):
                image_record=dict(mapping[frame],pair=pair,view=view)
                tasks.append((image_record,idx,[dict(bbox=metadata[i]['bbox']) for i in idx]))
        for s in schedule:
            if s['pair']!=pair:continue
            sides=[]
            for view in [0,1]:
                arrays=data._views[view]
                idx=np.flatnonzero((arrays['frame']==s['frame'])&(arrays['cls']==s['class'])&arrays['feature_present'])
                keys=[f'{pair}-{view+1}:{s["frame"]}:{int(arrays["detector_index"][i])}' for i in idx]
                sides.append([lookup[k] for k in keys])
            expected_groups[s['step']]=sides
        print(json.dumps({'stage':'index_fit','pair':pair,'total_rows':len(metadata)}),flush=True)
    n=len(metadata);assert n==56647
    assert shutil.disk_usage(HERE).free > n*3*224*224+3*1024**3
    groups=[dict(**s,sides=expected_groups[s['step']]) for s in schedule]
    labels=np.array(pids,np.int64)
    for g in groups:
        la,lb=[labels[x] for x in g['sides']]
        assert len(np.unique(la[la>=0]))==int((la>=0).sum())
        assert len(np.unique(lb[lb>=0]))==int((lb>=0).sum())
        assert len(np.intersect1d(la[la>=0],lb[lb>=0]))>0
    dump(OUT/'GROUPS.json',groups)
    np.save(OUT/'offline_pid.npy',labels);np.save(OUT/'offline_known.npy',labels>=0)
    np.save(OUT/'frozen_dino.npy',np.asarray(dinos,np.float32));np.save(OUT/'raw_fpn.npy',np.asarray(fpns,np.float32))
    del dinos,fpns
    sources=[Path(__file__),OLD/'p7r/SCHEDULE.json',OLD/'p0/data/ASSOCIATION_DIAGNOSTIC_SPLIT.json',
             OLD/'p2/src/data.py',OLD/'p14_fusion/extract_schedule_dino.py',OLD/'p4_diagnosis/extract_dino_crop32.py']
    dependencies.update({str(p):sha256(p) for p in sources})
    dump(OUT/'PREDECODE.json',dict(rows=n,images=len(tasks),fit_pairs=list(split_assignments()['fit']),
          source_rows=metadata,dependencies=dependencies,groups_sha256=sha256(OUT/'GROUPS.json'),
          unknown_rows=int((labels<0).sum()),no_new_labels_or_GT_in_inputs=True))
    crops=np.lib.format.open_memmap(OUT/'crops.npy',mode='w+',dtype=np.uint8,shape=(n,3,224,224))
    valid=np.zeros(n,bool);bounds=np.zeros((n,4),np.int32);image_index=np.full(n,-1,np.int32)
    image_refs=[]
    with ThreadPoolExecutor(max_workers=8) as pool:
        pending=deque();iterator=iter(tasks)
        def submit():
            try:pending.append(pool.submit(decode_image,next(iterator)))
            except StopIteration:pass
        for _ in range(8):submit()
        while pending:
            idx,values,boxes,oks,ref=pending.popleft().result();submit()
            im_index=len(image_refs);image_refs.append(ref)
            for i,value,box,ok in zip(idx,values,boxes,oks):
                valid[i]=ok;bounds[i]=box;image_index[i]=im_index
                if ok:crops[i]=value
                else:crops[i]=0
            if len(image_refs)%200==0:print(json.dumps({'stage':'crops','images_done':len(image_refs),'total':len(tasks)}),flush=True)
    crops.flush();assert valid.all() and (image_index>=0).all()
    # Re-decode a fixed first sample and exact compare to the persisted bytes.
    idx,values,_,_,_=decode_image(tasks[0]);assert np.array_equal(crops[idx[0]],values[0])
    np.savez_compressed(OUT/'inputs.npz',keys=np.array([r['key'] for r in metadata]),
        pair=np.array([r['pair'] for r in metadata]),view=np.array([r['view'] for r in metadata],np.int8),
        frame=np.array([r['frame'] for r in metadata],np.int32),cls=np.array([r['cls'] for r in metadata],np.int8),
        bbox=np.array([r['bbox'] for r in metadata],np.float32),source_row_index=np.array([r['source_row_index'] for r in metadata],np.int32),
        crop_valid=valid,crop_bounds=bounds,image_index=image_index)
    dump(OUT/'IMAGES.json',image_refs)
    files={str(p):sha256(p) for p in sorted(OUT.iterdir()) if p.is_file()}
    result=dict(status='FIT_CROPS_COMPLETE',rows=n,images=len(tasks),groups=len(groups),
                unknown_rows=int((labels<0).sum()),files=files,dependencies=dependencies,
                shape=[n,3,224,224],dtype='uint8',all_crops_valid=bool(valid.all()),
                label_mode='same_label_equal_id',physical_global_ids_verified=False,
                crop_recipe='exact existing decode_image; integer enclosing clipped predicted bbox, bicubic224 RGB',
                cal_dev_val_test_read=False,GPU_used=False,elapsed_seconds=time.monotonic()-start)
    dump(OUT/'RECEIPT.json',result);(OUT/'runner.exit').write_text('0\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['files','dependencies']}),flush=True)


if __name__=='__main__':main()
