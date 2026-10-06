#!/usr/bin/env python3
"""Export selected native P26 float32 FPN tensors and fresh head detections.

GT-free image inference only. The explicit role limits image access; the plan
contains other roles but only selected-role paths are opened. No old detections.
"""
import argparse
import copy
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

sys.dont_write_bytecode = True
HERE=Path(__file__).resolve().parent
P26=HERE.parent/'p26_mia_baseline'
UUID='GPU-aaa79a66-b5c4-e0a0-6e2f-28c1ac0e3e6a'
CKPT=Path('/raid/datasets/chc_data/MDMT/checkpoints/autoassign_r50_fpn_8x2_1x_full_mdmt/epoch_60.pth')
CKPT_SHA='8894ea5ffe8309017d78e2dac359d405b38aa66725e1b465a8dce30beb12c901'
DETECTOR_CFG=P26/'source/configs/mot/bytetrack/bytetrack_autoassign_full_mdmt-private-half.py'
RAID=Path('/raid/datasets/chc_data/claude_try_MDMOT_p28_causal_fgfa_20261005')
sys.path[:0]=[str(P26/'adapters'),str(P26/'source'),str(P26/'source/demo'),str(P26/'source/demo/utils')]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(2**20),b''):h.update(chunk)
    return h.hexdigest()


def plain(value):
    import numpy as np
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,Path):return str(value)
    if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [plain(v) for v in value]
    if value is None or isinstance(value,(str,int,float,bool)):return value
    raise TypeError('Non-JSON metadata: '+type(value).__name__)


def dump(path,value):
    path=Path(path)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(plain(value),indent=2,sort_keys=True,allow_nan=False)+'\n')
    temp.replace(path)


def source_bindings():
    imported=json.loads((P26/'SOURCE_IMPORT.json').read_text())
    candidates=[v for v in imported.values() if isinstance(v,dict) and any(str(k).startswith('source/') for k in v)]
    assert len(candidates)==1,'Unrecognized P26 source import manifest'
    hashes={}
    for relative,expected in candidates[0].items():
        if relative.endswith('.py'):
            path=P26/relative
            assert sha(path)==expected,'P26 source changed: '+str(path)
            hashes[str(path)]=expected
    required={P26/'adapters/mdmt_compat_runtime.py':'8a62f13c2102b04dbb75a6b902775c02fdd4859d17ec907da497e29e2aa9ef1b',
              DETECTOR_CFG:'97a718d988553bd073f98d3b1f8f888c87c457d05e1e023107afd63f39b748b1',CKPT:CKPT_SHA}
    for path,expected in required.items():
        assert sha(path)==expected,'Pinned runtime/config/checkpoint changed: '+str(path)
        hashes[str(path)]=expected
    for path in (P26/'SOURCE_IMPORT.json',HERE/'CONFIG.json',HERE/'GROUPS.json',HERE/'FRAMES.json',HERE/'PLAN_INPUTS.json',Path(__file__),HERE/'run_fpn.sh'):
        hashes[str(path)]=sha(path)
    return hashes


def select_plan(role):
    config=json.loads((HERE/'CONFIG.json').read_text())
    assert Path(config['cache_root'])==RAID
    assert config['lags']==[0,1,4,8]
    frames=[x for x in json.loads((HERE/'FRAMES.json').read_text()) if x['role']==role]
    groups=[x for x in json.loads((HERE/'GROUPS.json').read_text()) if x['role']==role]
    allowed=config['fit_pairs' if role=='fit' else 'calibration_pairs']
    count=config['fit_anchors_per_sequence' if role=='fit' else 'calibration_anchors_per_sequence']
    assert len(groups)==len(allowed)*2*count
    assert len({r['key'] for r in frames})==len(frames)
    assert len({g['key'] for g in groups})==len(groups)
    fmap={x['key']:x for x in frames};used=set()
    for g in groups:
        assert g['pair'] in allowed and g['view'] in (1,2)
        assert g['frames']==[g['frame']-lag for lag in (0,1,4,8)]
        assert g['image_keys'][0]==g['key'] and len(g['image_keys'])==4
        for key,f in zip(g['image_keys'],g['frames']):
            item=fmap[key]
            assert (item['role'],item['pair'],item['view'],item['frame'])==(role,g['pair'],g['view'],f)
            used.add(key)
    assert used==set(fmap)
    for item in frames:
        assert item['pair'] in allowed and item['view'] in (1,2) and item['frame']>=1
        path=Path(item['image_path'])
        expected=Path('/raid/datasets/chc_data/MDMT/train')/str(item['view'])/f"{item['pair']}-{item['view']}"
        assert path.parent==expected and path.stem==item['image_stem']
        assert item['key']==f"{item['pair']}-{item['view']}-t{item['frame']:06d}"
        assert path.suffix.lower() in ('.jpg','.jpeg','.png')
    return config,frames,groups


def environment(torch,np,mmcv,mmdet,mmtrack):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==UUID
    assert os.environ.get('CUDA_DEVICE_ORDER')=='PCI_BUS_ID'
    assert torch.cuda.device_count()==1
    query=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used','--format=csv,noheader,nounits'],text=True)
    line=next(x for x in query.splitlines() if UUID in x)
    fields=[x.strip() for x in line.split(',')]
    assert fields[0]=='3' and fields[1]==UUID and 'A100' in fields[2] and int(fields[3])==40960
    prop=torch.cuda.get_device_properties(0)
    assert 'A100' in prop.name and 39*2**30<prop.total_memory<41*2**30
    assert Path(mmtrack.__file__).resolve().is_relative_to(P26/'source') if hasattr(Path(), 'is_relative_to') else str(Path(mmtrack.__file__).resolve()).startswith(str(P26/'source')+'/')
    return dict(python=sys.version,python_executable=sys.executable,torch=torch.__version__,numpy=np.__version__,
                mmcv=mmcv.__version__,mmdet=mmdet.__version__,mmtrack_path=mmtrack.__file__,
                physical_gpu=dict(index=3,uuid=UUID,name=fields[2],memory_total_mib=int(fields[3]),memory_used_mib_at_start=int(fields[4])),
                logical_gpu=dict(index=0,name=prop.name,total_memory_bytes=prop.total_memory),
                torch_threads=torch.get_num_threads(),cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
                cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,
                cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32)


def run(args):
    import numpy as np
    import torch
    import mmcv
    import mmdet
    import mmtrack
    from mdmt_compat_runtime import init_model_mdmt
    from mmcv.parallel import collate,scatter
    from mmdet.core import bbox2result
    from mmdet.datasets.pipelines import Compose
    torch.set_num_threads(2)
    config,frames,groups=select_plan(args.role)
    assert args.role=='fit' or args.calibration_authorized,'Calibration export requires explicit later invocation flag'
    runtime=environment(torch,np,mmcv,mmdet,mmtrack)
    inputs=source_bindings()
    out=HERE/args.role
    # Launcher owns only these status paths. Existing features/indices are never reused.
    out.mkdir(exist_ok=True)
    for name in ('features','FEATURE_INDEX.json','FEATURE_RECEIPT.json','SMOKE.json','PRE_EXPORT.json'):
        assert not (out/name).exists() and not (out/name).is_symlink(),'Existing export target: '+str(out/name)
    target=RAID/args.role/'features'
    assert not target.exists(),'Existing RAID feature target; inspect before a new run: '+str(target)
    target.mkdir(parents=True,exist_ok=False)
    (out/'features').symlink_to(target,target_is_directory=True)
    before_free=shutil.disk_usage(RAID).free
    assert before_free>=40*2**30,'Less than 40 GiB free on RAID'
    dump(out/'PRE_EXPORT.json',dict(role=args.role,frames=len(frames),groups=len(groups),gt_or_labels_read=False,
        image_access_role=args.role,inputs=inputs,environment=runtime,disk_free_bytes=before_free,
        array_schema={'features':'f0..f4, float32 [1,256,H,W], FPN strides8/16/32/64/128',
                      'detections':'det0..det2, float32 [N,5], native post-NMS xyxy score rescaled to original image',
                      'compression':'np.savez stored/uncompressed; float32 preserved'},
        baseline='Fresh bbox_head.simple_test on the exact exported features; no historical streams or GT'))
    model=init_model_mdmt(str(DETECTOR_CFG),str(CKPT),device='cuda:0')
    model.eval();model.requires_grad_(False)
    detector=model.detector
    assert all(not p.requires_grad for p in model.parameters()) and not detector.training
    assert detector.bbox_head.num_classes==3
    assert detector.bbox_head.test_cfg.score_thr==.05
    assert detector.bbox_head.test_cfg.nms.iou_threshold==.6 and detector.bbox_head.test_cfg.max_per_img==100
    pipeline=Compose(model.cfg.data.test.pipeline)
    dump(out/'RESOLVED_DETECTOR_CONFIG.json',plain(model.cfg.model.detector))
    dump(out/'RESOLVED_TEST_PIPELINE.json',plain(model.cfg.data.test.pipeline))
    t0=time.monotonic();entries=[];smoke=None;total_dets=0
    for ordinal,record in enumerate(frames,1):
        image_path=Path(record['image_path'])
        assert sha(image_path)==record['image_sha256'],'Image changed before decode: '+str(image_path)
        data=pipeline(dict(img_info=dict(filename=str(image_path)),img_prefix=None))
        batch=scatter(collate([data],samples_per_gpu=1),[torch.device('cuda:0')])[0]
        assert set(batch)=={'img','img_metas'} and len(batch['img'])==len(batch['img_metas'])==1,'Unexpected augmentation or extra inputs'
        img=batch['img'][0];metas=batch['img_metas'][0]
        assert img.dtype==torch.float32 and img.ndim==4 and img.shape[0]==1 and img.shape[1]==3
        assert len(metas)==1 and not metas[0]['flip']
        metas[0]['batch_input_shape']=tuple(img.shape[-2:])
        for field in ('img_shape','pad_shape','ori_shape','scale_factor','img_norm_cfg'):assert field in metas[0]
        with torch.no_grad():
            feats=detector.extract_feat(img)
            assert len(feats)==5
            expected_shapes=[]
            for level,(feature,stride) in enumerate(zip(feats,(8,16,32,64,128))):
                expected=(1,256,math.ceil(img.shape[2]/stride),math.ceil(img.shape[3]/stride))
                assert tuple(feature.shape)==expected,(level,tuple(feature.shape),expected)
                assert feature.dtype==torch.float32 and not feature.requires_grad
                assert bool(torch.isfinite(feature).all())
                expected_shapes.append(list(expected))
            pairs=detector.bbox_head.simple_test(feats,metas,rescale=True)
            assert len(pairs)==1
            native=bbox2result(pairs[0][0],pairs[0][1],3)
            assert len(native)==3
            if ordinal==1:
                direct=detector.simple_test(img,copy.deepcopy(metas),rescale=True)
                assert len(direct)==1 and len(direct[0])==3
                details=[]
                for cls,(left,right) in enumerate(zip(native,direct[0])):
                    equal=left.shape==right.shape and np.array_equal(left,right)
                    allclose=left.shape==right.shape and bool(np.allclose(left,right,rtol=0,atol=0))
                    details.append(dict(class_id=cls,head_shape=list(left.shape),direct_shape=list(right.shape),
                                        exact_array_equal=equal,allclose_rtol0_atol0=allclose,
                                        max_abs_error=float(np.abs(left-right).max()) if left.shape==right.shape and left.size else (0 if left.shape==right.shape else None)))
                smoke=dict(status='PASS' if all(x['exact_array_equal'] for x in details) else 'FAIL',
                    key=record['key'],image_sha256=record['image_sha256'],fpn_shapes=expected_shapes,
                    direct_api='model.detector.simple_test(img,metadata,rescale=True)',
                    cached_api='model.detector.extract_feat -> bbox_head.simple_test -> bbox2result',
                    class_comparisons=details,input_dtype=str(img.dtype),feature_dtype='float32',
                    metadata=plain(metas[0]),source_paths=dict(detector=inspect.getfile(type(detector)),head=inspect.getfile(type(detector.bbox_head))))
                dump(out/'SMOKE.json',smoke)
                assert smoke['status']=='PASS','First-image direct/head exact parity failed; preserve evidence and stop'
                print(json.dumps({'event':'SMOKE_PASS','key':record['key'],'fpn_shapes':expected_shapes,'det_counts':[len(x) for x in native]}),flush=True)
        arrays={f'f{i}':feature.detach().cpu().numpy() for i,feature in enumerate(feats)}
        for cls,boxes in enumerate(native):
            assert boxes.dtype==np.float32 and boxes.ndim==2 and boxes.shape[1]==5 and np.isfinite(boxes).all()
            arrays[f'det{cls}']=boxes
        archive=target/f"{record['key']}.npz"
        assert not archive.exists()
        temporary=archive.with_suffix('.npz.part')
        with temporary.open('xb') as handle:np.savez(handle,**arrays)
        temporary.replace(archive)
        if ordinal==1:
            with np.load(archive,allow_pickle=False) as z:
                assert set(z.files)==set(arrays)
                for key,value in arrays.items():assert np.array_equal(z[key],value) and z[key].dtype==np.float32
        counts=[len(x) for x in native];total_dets+=sum(counts)
        item=dict(record,archive_path=str(archive),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,
                  fpn_shapes=expected_shapes,dtype='float32',det_counts=counts,metadata=plain(metas[0]))
        entries.append(item)
        with (out/'FEATURE_EVENTS.jsonl').open('a') as handle:handle.write(json.dumps(item,sort_keys=True,allow_nan=False)+'\n')
        if ordinal%10==0 or ordinal==len(frames):
            elapsed=time.monotonic()-t0
            progress=dict(event='EXPORT_PROGRESS',role=args.role,completed_frames=ordinal,total_frames=len(frames),
                          completed_detections=total_dets,last_key=record['key'],elapsed_seconds=round(elapsed,3),
                          disk_free_bytes=shutil.disk_usage(RAID).free,peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
            dump(out/'PROGRESS.json',progress);print(json.dumps(progress),flush=True)
        del feats,arrays,pairs,native,img,batch,data
    # Neither the plans, script nor frozen source/checkpoint may drift during export.
    for path,expected in inputs.items():assert sha(path)==expected,'Source/plan changed during export: '+path
    module_inputs={}
    for name,module in list(sys.modules.items()):
        if name.split('.')[0] not in ('mmdet','mmcv','mmtrack','mdmt_compat_runtime'):continue
        file=getattr(module,'__file__',None)
        if file and Path(file).is_file() and Path(file).suffix in ('.py','.so'):
            module_inputs[str(Path(file).resolve())]=sha(file)
    index=dict(schema='p28_native_fpn_index_v1',role=args.role,feature_dtype='float32',leading_batch_dimension=True,
        feature_names=[f'f{i}' for i in range(5)],detection_names=[f'det{i}' for i in range(3)],
        metadata_coordinate_system='img_shape resized-unpadded; pad_shape tensor domain; det arrays rescaled original pixels',
        entries=entries,by_key={x['key']:i for i,x in enumerate(entries)},
        groups=[dict(g,current_metadata_key=g['image_keys'][0]) for g in groups])
    dump(out/'FEATURE_INDEX.json',index)
    expected_names={x['key']+'.npz' for x in frames}
    assert {x.name for x in target.iterdir()}==expected_names,'Missing, extra or partial archive'
    outputs={name:sha(out/name) for name in ('PRE_EXPORT.json','SMOKE.json','RESOLVED_DETECTOR_CONFIG.json','RESOLVED_TEST_PIPELINE.json','FEATURE_EVENTS.jsonl','PROGRESS.json','FEATURE_INDEX.json')}
    receipt=dict(schema='p28_native_fpn_receipt_v1',status='PASS_FROZEN_NATIVE_FPN_EXPORT',role=args.role,
        frame_count=len(entries),group_count=len(groups),total_native_detections=total_dets,
        archive_count=len(expected_names),archive_total_bytes=sum(x['archive_bytes'] for x in entries),
        image_hashes={x['image_path']:x['image_sha256'] for x in entries},
        archive_hashes={x['archive_path']:x['archive_sha256'] for x in entries},
        source_config_plan_hashes=inputs,imported_runtime_module_hashes=module_inputs,outputs=outputs,
        environment=runtime,model_parameter_requires_grad_count=sum(p.requires_grad for p in model.parameters()),
        model_training=False,gt_or_labels_read=False,prior_stream_detections_used=False,
        untouched_roles=[role for role in ('fit','calibration') if role!=args.role],
        fresh_baseline='native head predictions from the exact exported float32 FPN; score .05, NMS .6, max_per_img100 unchanged',
        smoke_status=smoke['status'],elapsed_seconds=round(time.monotonic()-t0,3),
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),disk_free_bytes_end=shutil.disk_usage(RAID).free)
    dump(out/'FEATURE_RECEIPT.json',receipt)
    (out/'FEATURE_RECEIPT.sha256').write_text(sha(out/'FEATURE_RECEIPT.json')+'  FEATURE_RECEIPT.json\n')
    print(json.dumps({'event':'EXPORT_COMPLETE','status':receipt['status'],'role':args.role,'frames':len(entries),
                      'groups':len(groups),'receipt_sha256':sha(out/'FEATURE_RECEIPT.json'),'elapsed_seconds':receipt['elapsed_seconds']}),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--role',choices=('fit','calibration'),required=True)
    parser.add_argument('--calibration-authorized',action='store_true')
    args=parser.parse_args()
    try:run(args)
    except Exception as exc:
        out=HERE/args.role
        if out.exists():
            failure=out/('FAILURE_'+time.strftime('%Y%m%dT%H%M%S')+'.json')
            dump(failure,dict(status='FAIL',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),
                              role=args.role,source_code_sha256=sha(__file__),preserve_partial_outputs=True))
        raise


if __name__=='__main__':main()
