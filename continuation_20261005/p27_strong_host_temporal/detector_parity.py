"""Fresh native detector replay on 90 fit images; compare >=.1 stream subset.

No labels, raw GT, tracker state, threshold search or change to P26.
"""
import os
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import gzip
import hashlib
import json
import subprocess
import time
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'p26_mia_baseline'
OLD=Path('/home/chenhc/mdmot_research_20261002')
CKPT=Path('/raid/datasets/chc_data/MDMT/checkpoints/autoassign_r50_fpn_8x2_1x_full_mdmt/epoch_60.pth')
UUID='GPU-aaa79a66-b5c4-e0a0-6e2f-28c1ac0e3e6a'
sys.path[:0]=[str(BASE/'adapters'),str(BASE/'source')]
from mdmt_compat_runtime import init_model_mdmt
from mmcv.parallel import collate, scatter
from mmdet.datasets.pipelines import Compose

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def dump(path,obj):path.write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n')

def main():
    out=HERE/'detector_parity';out.mkdir(exist_ok=False);t=time.monotonic()
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==UUID
    torch.set_num_threads(2)
    physical=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total','--format=csv,noheader,nounits'],text=True)
    gpu=next(x for x in physical.splitlines() if UUID in x);assert gpu.startswith('3,') and 'A100' in gpu
    assert sha(CKPT)=='8894ea5ffe8309017d78e2dac359d405b38aa66725e1b465a8dce30beb12c901'
    fit=json.loads((OLD/'p0/data/ASSOCIATION_DIAGNOSTIC_SPLIT.json').read_text())['assignments']['fit']
    audit=json.loads((OLD/'p0/artifacts/FROZEN_INPUT_AUDIT.json').read_text());records={r['sequence']:r for r in audit['sequences']}
    protocol=dict(pairs=fit,views=[1,2],frames='ordinal 1, floor((T+1)/2), T',
        scope='native P26 detector post-NMS outputs vs historical .1-retained subset; not tracker parity',
        score_subset=.1,native_config_unchanged=True,tolerance_bbox_or_score_absolute=.001,
        GT_read=False,labels_read=False,cal_dev_official_val_test_read=False,gpu=gpu,
        checkpoint_sha256=sha(CKPT),source_import_sha256=sha(BASE/'SOURCE_IMPORT.json'),code_sha256=sha(__file__))
    dump(out/'PROTOCOL.json',protocol)
    model=init_model_mdmt(str(BASE/'source/configs/mot/bytetrack/bytetrack_autoassign_full_mdmt-private-half.py'),str(CKPT),device='cuda:0')
    pipe=Compose(model.cfg.data.test.pipeline)
    results=[];files={};all_pass=True
    for pair in fit:
        for view in ['1','2']:
            rec=records[f'{pair}-{view}'];source=rec['sources']['stream'];assert sha(source['path'])==source['sha256']
            files[source['path']]=source['sha256'];frames=rec['counts']['frames'];selected={1,(frames+1)//2,frames}
            with gzip.open(source['path'],'rt') as stream:
                saved=[json.loads(line) for ordinal,line in enumerate(stream,1) if ordinal in selected]
            images=Path('/raid/datasets/chc_data/MDMT/train')/view/f'{pair}-{view}'
            index={p.stem:p for p in images.iterdir() if p.suffix.lower() in ['.jpg','.jpeg','.png']}
            for row in saved:
                assert row['gt_read'] is False and row['gt_initialized'] is False
                path=index[str(row['image_stem'])];files[str(path)]=sha(path)
                batch=scatter(collate([pipe(dict(img_info=dict(filename=str(path)),img_prefix=None))],samples_per_gpu=1),[torch.device('cuda:0')])[0]
                with torch.no_grad():native=model.detector(return_loss=False,rescale=True,**batch)[0]
                fresh=[]
                for cls,boxes in enumerate(native):
                    for b in boxes:
                        if b[4]>=.1:fresh.append(list(map(float,b))+[cls])
                old=[r['bbox']+[r['score'],r['class']] for r in row['detector_detections']]
                fresh=np.asarray(fresh).reshape(-1,6);old=np.asarray(old).reshape(-1,6)
                error=float(np.max(np.abs(fresh[:,:5]-old[:,:5]))) if fresh.shape==old.shape and len(old) else (0. if fresh.shape==old.shape else None)
                cls_equal=bool(fresh.shape==old.shape and np.array_equal(fresh[:,-1],old[:,-1]))
                passed=bool(cls_equal and error is not None and error<=.001);all_pass&=passed
                npz=out/f'{pair}-{view}_t{row["frame"]:04d}.npz'
                np.savez_compressed(npz,**{f'native_class_{c}':b for c,b in enumerate(native)},historical=old,subset=fresh)
                r=dict(pair=pair,view=view,frame=row['frame'],image_stem=row['image_stem'],image_path=str(path),
                    old_count=len(old),new_subset_count=len(fresh),native_count=sum(len(a) for a in native),
                    classes_equal=cls_equal,max_abs_error=error,passed=passed,npz_sha256=sha(npz))
                results.append(r);print(json.dumps(r),flush=True)
    dump(out/'RECEIPT.json',dict(status='PASS_SAMPLED_NATIVE_DETECTOR_PARITY' if all_pass else 'FAIL_SAMPLED_PARITY',
         results=results,images=len(results),seconds=time.monotonic()-t,gpu=gpu,
         peak_allocated_bytes=torch.cuda.max_memory_allocated(),dependencies=files,
         scope='sampled >=.1 detector subset parity; visibility from historical full streams is only this subset',
         no_GT_or_labels_or_tracker=True,protocol_sha256=sha(out/'PROTOCOL.json')))
    assert all_pass,'Inspect mismatched frames, preserve exact arrays; do not silently substitute tracker.'

if __name__=='__main__':main()
