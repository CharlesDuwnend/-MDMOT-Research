"""Frozen official RAFT current-to-past image flow, with no annotation access."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import sys
sys.dont_write_bytecode=True
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import time
import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.optical_flow import raft_large

HERE=Path(__file__).resolve().parent
WEIGHT=HERE.parents[1]/'continuation_20261004/p22_dense_motion/assets/raft_large_C_T_SKHT_V2-ff5fadd5.pth'
UUID=os.environ.get('MDMT_GPU_UUID','GPU-aaa79a66-b5c4-e0a0-6e2f-28c1ac0e3e6a')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')

def main(role):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==UUID
    torch.set_num_threads(2);cv2.setNumThreads(1)
    config=json.loads((HERE/'CONFIG.json').read_text());groups=[g for g in json.loads((HERE/'GROUPS.json').read_text()) if g['role']==role]
    images={x['key']:x for x in json.loads((HERE/'FRAMES.json').read_text()) if x['role']==role}
    out=HERE/role;out.mkdir(exist_ok=True)
    if (out/'FLOW_RECEIPT.json').exists() or (out/'FLOW_PROTOCOL.json').exists():raise FileExistsError('preserve previous extraction')
    cache=Path(config['cache_root'])/role/'flows';cache.mkdir(parents=True,exist_ok=False)
    (out/'flows').symlink_to(cache,target_is_directory=True)
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total','--format=csv,noheader,nounits'],text=True)
    gpu=next(x for x in gpu.splitlines() if UUID in x);assert UUID in gpu and 'A100' in gpu
    old_weight=json.loads((WEIGHT.parent/'WEIGHT_RECEIPT.json').read_text())
    digest=sha(WEIGHT)
    assert digest in json.dumps(old_weight)
    protocol=dict(role=role,groups=len(groups),direction='current image -> past image; past feature backward sampling',
        lags=[1,4,8],flow_image='full RGB image resized per-axis to rounded multiple8, maximum side about640; no padding or letterbox',
        resize_interpolation='cv2.INTER_AREA',normalization='RGB float32 /127.5 -1',raft_updates=12,weights_sha256=digest,
        torch=torch.__version__,torchvision=torchvision.__version__,gpu=gpu,GT_or_labels_read=False,
        config_sha256=sha(HERE/'CONFIG.json'),groups_sha256=sha(HERE/'GROUPS.json'),frames_sha256=sha(HERE/'FRAMES.json'),code_sha256=sha(__file__))
    dump(out/'FLOW_PROTOCOL.json',protocol)
    start=time.monotonic();model=raft_large(weights=None,progress=False).eval().requires_grad_(False)
    model.load_state_dict(torch.load(WEIGHT,map_location='cpu',weights_only=True),strict=True);model.cuda()
    records=[]
    for i,g in enumerate(groups):
        assert g['frames']==[g['frame']-v for v in [0,1,4,8]] and min(g['frames'])>=1
        decoded=[];shapes=[]
        for k in g['image_keys']:
            r=images[k];assert r['pair']==g['pair'] and r['view']==g['view'] and sha(r['image_path'])==r['image_sha256']
            im=cv2.cvtColor(cv2.imread(r['image_path']),cv2.COLOR_BGR2RGB);shapes.append(im.shape[:2]);decoded.append(im)
        assert len(set(shapes))==1
        h,w=shapes[0];scale=config['flow_max_side']/max(h,w);fw=max(8,int(round(w*scale/8))*8);fh=max(8,int(round(h*scale/8))*8)
        tensors=[torch.from_numpy(cv2.resize(im,(fw,fh),interpolation=cv2.INTER_AREA)).permute(2,0,1)[None].float().cuda()/127.5-1 for im in decoded]
        flow=[]
        with torch.inference_mode():
            for ref in tensors[1:]:
                u=model(tensors[0],ref,num_flow_updates=12)[-1][0]
                assert torch.isfinite(u).all() and tuple(u.shape)==(2,fh,fw)
                flow.append(u.cpu().numpy())
        path=cache/(g['key']+'.npz')
        np.savez_compressed(path,flow=np.stack(flow).astype(np.float32),original_hw=np.array([h,w]),flow_img_hw=np.array([fh,fw]),lags=np.array([1,4,8]))
        records.append(dict(key=g['key'],path=str(path),sha256=sha(path),original_hw=[h,w],flow_img_hw=[fh,fw],image_keys=g['image_keys'],frame=g['frame'],pair=g['pair'],view=g['view']))
        if (i+1)%10==0 or i==0:print(json.dumps(dict(role=role,groups=i+1,total=len(groups),flow_calls=3*(i+1),seconds=time.monotonic()-start)),flush=True)
    dump(out/'FLOW_INDEX.json',dict(role=role,entries=records,by_key={r['key']:i for i,r in enumerate(records)}))
    dump(out/'FLOW_RECEIPT.json',dict(status='COMPLETE_FROZEN_CAUSAL_FLOW',role=role,groups=len(groups),flow_calls=len(groups)*3,
       seconds=time.monotonic()-start,gpu=gpu,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
       flow_index_sha256=sha(out/'FLOW_INDEX.json'),protocol_sha256=sha(out/'FLOW_PROTOCOL.json'),code_sha256=sha(__file__),GT_or_labels_read=False,
       dependencies={r['path']:r['sha256'] for r in records}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--role',choices=['fit','calibration'],default='fit');main(p.parse_args().role)
