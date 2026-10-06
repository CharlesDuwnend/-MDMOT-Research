"""Matched continuation: P14 head-only versus actual DINO visual fine-tuning.

This is a visual-adaptation baseline, not a proposed novel method.
"""
import os
GPU_UUID='GPU-1b297aba-ae7e-e326-5903-476f1bb683d7'
os.environ['CUDA_VISIBLE_DEVICES']=GPU_UUID
os.environ['XFORMERS_DISABLED']='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ['OPENBLAS_NUM_THREADS']='1'
import sys
sys.dont_write_bytecode=True
import argparse
from pathlib import Path
import json
import math
import hashlib
import subprocess
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE=Path(__file__).resolve().parent
OLD=Path('/home/chenhc/mdmot_research_20261002')
ASSETS=OLD/'p4_diagnosis/assets'
WEIGHT=ASSETS/'dinov2_vits14_pretrain.pth'
INIT=OLD/'p14_fusion/runs/frozen_adapter/fixed_step_001200.pt'
INIT_SHA='ae114496f076c056d871f55b776f75e99de0821280157caa40ff19e6decd56cb'
WEIGHT_SHA='b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9'
COMMIT='7764ea0f912e53c92e82eb78a2a1631e92725fc8'
sys.path.insert(0,str(OLD/'p1/src'))
from train_association_probe import masked_matching_loss
SETTINGS={'seed':42,'updates':1200,'microbatch':24,'head_lr':1e-4,'backbone_lr':1e-5,
          'head_weight_decay':1e-4,'backbone_weight_decay':.01,'warmup_steps':60,
          'lr_schedule':'linear warmup then cosine to 0.1','loss':'existing bidirectional masked CE',
          'temperature':.1,'augmentation':'none in either arm','precision':'float32; TF32 disabled',
          'gradient_clip':1.,'stochastic_layers':'eval mode, gradients enabled',
          'gradient_cache':'full untruncated group embedding loss, exact chunked vector-Jacobian replay'}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def dump(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def state_sha(model):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection=nn.Sequential(nn.Linear(256,128),nn.LayerNorm(128),nn.GELU())
        self.ffn=nn.Sequential(nn.Linear(128,256),nn.GELU(),nn.Linear(256,128))
        self.norm=nn.LayerNorm(128)
        self.adapter=nn.Sequential(nn.LayerNorm(384),nn.Linear(384,128))

    def forward(self,fpn,dino):
        h=self.projection(fpn)+self.adapter(dino)
        return F.normalize(self.norm(h+self.ffn(h)),dim=-1)


class VisualModel(nn.Module):
    def __init__(self,arm,with_backbone=True):
        super().__init__();self.arm=arm
        self.head=Head()
        assert sha(INIT)==INIT_SHA
        ckpt=torch.load(INIT,map_location='cpu',weights_only=False)
        assert ckpt['arm']=='frozen_adapter' and ckpt['fixed_step']==1200
        self.head.load_state_dict(ckpt['state_dict'],strict=True)
        self.head.requires_grad_(False);self.head.adapter.requires_grad_(True)
        self.backbone=None
        if with_backbone:
            assert sha(WEIGHT)==WEIGHT_SHA
            assert subprocess.check_output(['git','-C',str(ASSETS/'dinov2'),'rev-parse','HEAD'],text=True).strip()==COMMIT
            self.backbone=torch.hub.load(str(ASSETS/'dinov2'),'dinov2_vits14',source='local',pretrained=False)
            self.backbone.load_state_dict(torch.load(WEIGHT,map_location='cpu',weights_only=True),strict=True)
            self.backbone.requires_grad_(arm=='visual_ft')
            self.backbone.mask_token.requires_grad_(False)
        self.register_buffer('mean',torch.tensor([.485,.456,.406])[None,:,None,None],persistent=False)
        self.register_buffer('std',torch.tensor([.229,.224,.225])[None,:,None,None],persistent=False)
        self.eval()

    def encode_rgb(self,rgb):
        x=(rgb.float()/255.-self.mean)/self.std
        return F.normalize(self.backbone.forward_features(x)['x_norm_clstoken'],dim=-1)

    def forward(self,fpn,dino=None,rgb=None):
        if rgb is not None:dino=self.encode_rgb(rgb)
        return self.head(fpn,dino)


def setup():
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.manual_seed(42);np.random.seed(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    assert torch.cuda.device_count()==1
    assert torch.cuda.get_device_properties(0).total_memory>39*1024**3
    snap=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used','--format=csv,noheader'],text=True)
    assert any(l.startswith('1, '+GPU_UUID) and 'A100' in l for l in snap.splitlines())
    return snap


def load_data(verify=True):
    path=HERE/'data/RECEIPT.json';receipt=json.loads(path.read_text())
    assert receipt['status']=='FIT_CROPS_COMPLETE' and not receipt['cal_dev_val_test_read']
    if verify:
        for p,digest in receipt['files'].items():assert sha(p)==digest,p
    arrays={name:np.load(HERE/'data'/(name+'.npy'),mmap_mode='r',allow_pickle=False)
            for name in ['crops','frozen_dino','raw_fpn','offline_pid','offline_known']}
    groups=json.loads((HERE/'data/GROUPS.json').read_text())
    assert len(groups)==1200
    return arrays,groups,receipt


def batch(arrays,g):
    idx=np.array(g['sides'][0]+g['sides'][1],np.int64);cut=len(g['sides'][0])
    fpn=torch.from_numpy(arrays['raw_fpn'][idx].copy()).cuda()
    dino=torch.from_numpy(arrays['frozen_dino'][idx].copy()).cuda()
    labels=torch.from_numpy(arrays['offline_pid'][idx].copy()).cuda()
    valid=torch.from_numpy(arrays['offline_known'][idx].copy()).cuda()
    return idx,cut,fpn,dino,labels,valid


def loss(emb,cut,labels,valid):
    value,count=masked_matching_loss(emb[:cut],emb[cut:],labels[:cut],labels[cut:],valid[:cut],valid[cut:],.1)
    assert value is not None and count>0 and torch.isfinite(value)
    return value,count


def cache_backward(model,arrays,idx,fpn,cut,labels,valid,micro):
    chunks=[]
    with torch.no_grad():
        for start in range(0,len(idx),micro):
            rgb=torch.from_numpy(arrays['crops'][idx[start:start+micro]].copy()).cuda()
            chunks.append(model(fpn[start:start+micro],rgb=rgb))
    emb=torch.cat(chunks).requires_grad_(True)
    value,count=loss(emb,cut,labels,valid)
    grad=torch.autograd.grad(value,emb)[0]
    for start in range(0,len(idx),micro):
        rgb=torch.from_numpy(arrays['crops'][idx[start:start+micro]].copy()).cuda()
        output=model(fpn[start:start+micro],rgb=rgb)
        output.backward(grad[start:start+micro])
    return value.detach(),count


def smoke():
    snapshot=setup();arrays,groups,_=load_data()
    model=VisualModel('visual_ft').cuda()
    # Bounded real subset supports a full-graph versus chunked-gradient comparison.
    g=groups[0];idx,cut,fpn,dino,labels,valid=batch(arrays,g)
    rgb=torch.from_numpy(arrays['crops'][idx[:8]].copy()).cuda()
    with torch.no_grad():
        fresh=model.encode_rgb(rgb)
        cosine=F.cosine_similarity(fresh,dino[:len(rgb)]).min().item()
        fresh_emb=model(fpn[:len(rgb)],dino=fresh)
        cached_emb=model(fpn[:len(rgb)],dino=dino[:len(rgb)])
        error=(fresh_emb-cached_emb).abs().max().item()
    # Tolerance is measurement-based, not preference. Over all 54,488 rows the
    # re-encoded crop cosine to its own cached DINO row is min 0.9998621, mean
    # 0.99999897, with exactly 1 row below 0.9999
    # (probe_population_floor.json). Batch size is not the cause
    # (probe_batch_effect.json). Row<->vector integrity is instead established by
    # a stronger check: over 512 rows every re-encoded crop is its own cached
    # row's argmax with a 0.0196 minimum margin
    # (probe_cosine_source.json). The functional contract below is unchanged.
    assert cosine>0.9998 and error<2e-4,(cosine,error)
    # Use arbitrary nonzero output cotangents: checks exact VJP decomposition
    # independently of label mix or current ranking on this small sample.
    target=torch.linspace(-.01,.01,len(rgb)*128,device='cuda').reshape(len(rgb),128)
    model.zero_grad(set_to_none=True)
    model(fpn[:len(rgb)],rgb=rgb).backward(target)
    names=['backbone.patch_embed.proj.weight','backbone.blocks.11.attn.qkv.weight','head.adapter.1.weight']
    expected={k:p.grad.detach().clone() for k,p in model.named_parameters() if k in names}
    model.zero_grad(set_to_none=True)
    for start in range(0,len(rgb),3):
        end=min(start+3,len(rgb))
        model(fpn[start:end],rgb=rgb[start:end]).backward(target[start:end])
    grad_errors={}
    for k,p in model.named_parameters():
        if k in expected:
            relative=(p.grad-expected[k]).norm()/expected[k].norm().clamp_min(1e-12)
            grad_errors[k]=float(relative);assert relative<1e-4,(k,relative)
    model.zero_grad(set_to_none=True)
    out=model(fpn,dino=dino);base,count=loss(out,cut,labels,valid)
    altered=labels.clone();altered[~valid]=999999
    invariant,_=loss(out,cut,altered,valid);assert torch.equal(base,invariant)
    initial=state_sha(model.backbone)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-5)
    value,count=cache_backward(model,arrays,idx,fpn,cut,labels,valid,SETTINGS['microbatch'])
    backbone_norm=float(torch.nn.utils.clip_grad_norm_(model.backbone.parameters(),1.))
    assert np.isfinite(backbone_norm) and backbone_norm>0
    optimizer.step();assert state_sha(model.backbone)!=initial
    result={'status':'PASS_REAL_VISUAL_GRADIENT_CONTRACT','fresh_vs_cached_dino_cosine_min':cosine,
            'initial_fused_output_max_abs_error':error,'microbatch_vjp_relative_errors':grad_errors,
            'unknown_label_mask_invariance':True,'backbone_gradient_norm':backbone_norm,
            'actual_backbone_updated_in_discarded_smoke':True,'group':g,'loss':float(value),'queries':count,
            'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'gpu_snapshot':snapshot,
            'source_sha256':sha(__file__),'data_receipt_sha256':sha(HERE/'data/RECEIPT.json')}
    dump(HERE/'SMOKE.json',result);print(json.dumps(result),flush=True)


def freeze():
    smoke=json.loads((HERE/'SMOKE.json').read_text());assert smoke['status']=='PASS_REAL_VISUAL_GRADIENT_CONTRACT'
    assert smoke['source_sha256']==sha(__file__)
    assert not (HERE/'FROZEN.json').exists()
    paths=[Path(__file__),INIT,WEIGHT,HERE/'data/RECEIPT.json',HERE/'data/GROUPS.json',HERE/'SMOKE.json',
           OLD/'p1/src/train_association_probe.py']
    sources={str(p):sha(p) for p in paths}
    for p in sorted((ASSETS/'dinov2/dinov2').rglob('*.py')):sources[str(p)]=sha(p)
    result={'status':'FROZEN_BEFORE_TRAINING','arms':['head_only','visual_ft'],'settings':SETTINGS,
            'initial_head':'P14 frozen_adapter final1200; identical in both arms; fresh optimizer',
            'initial_backbone':'official DINOv2 ViT-S/14; same frozen visual features at step0',
            'frozen_P1_base':True,'training':'15fit pairs only; full groups incl unknown, no row truncation',
            'eval':'after both final checkpoints; same P4 fit15/cal5 candidate support; no dev/official-val/test',
            'progression':'visual_ft cal pair-macro >= head_only+.01 and >=3/5 pair wins -> keep as stronger visual baseline; otherwise hold and implementation look-back',
            'novel_method_claim':False,'whole_system_unseen_scene_claim':False,'sources':sources,
            'gpu_uuid':GPU_UUID,'no_calibration_used_to_fit_weights':True}
    dump(HERE/'FROZEN.json',result);print('FROZEN')


def train(arm):
    snapshot=setup();frozen=json.loads((HERE/'FROZEN.json').read_text())
    assert frozen['settings']==SETTINGS
    for p,digest in frozen['sources'].items():assert sha(p)==digest,p
    arrays,groups,_=load_data();out=HERE/'runs'/arm
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);dump(out/'GPU_START.json',{'snapshot':snapshot,'uuid':GPU_UUID})
    model=VisualModel(arm,with_backbone=arm=='visual_ft').cuda()
    initial_head=state_sha(model.head);initial_backbone=state_sha(model.backbone) if model.backbone else None
    groups_opt=[{'params':model.head.adapter.parameters(),'lr':SETTINGS['head_lr'],'weight_decay':SETTINGS['head_weight_decay'],'base_lr':SETTINGS['head_lr']}]
    if model.backbone:groups_opt.append({'params':[p for p in model.backbone.parameters() if p.requires_grad],
        'lr':SETTINGS['backbone_lr'],'weight_decay':SETTINGS['backbone_weight_decay'],'base_lr':SETTINGS['backbone_lr']})
    optimizer=torch.optim.AdamW(groups_opt)
    started=time.monotonic();history=[];seen=0;supervised=0
    for step,g in enumerate(groups,1):
        idx,cut,fpn,dino,labels,valid=batch(arrays,g)
        progress=max(0,(step-SETTINGS['warmup_steps'])/(SETTINGS['updates']-SETTINGS['warmup_steps']))
        factor=step/SETTINGS['warmup_steps'] if step<=SETTINGS['warmup_steps'] else .1+.9*.5*(1+math.cos(math.pi*progress))
        for group in optimizer.param_groups:group['lr']=group['base_lr']*factor
        optimizer.zero_grad(set_to_none=True)
        if arm=='head_only':
            value,count=loss(model(fpn,dino=dino),cut,labels,valid);value.backward()
        else:value,count=cache_backward(model,arrays,idx,fpn,cut,labels,valid,SETTINGS['microbatch'])
        gradnorm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],SETTINGS['gradient_clip'])
        assert torch.isfinite(gradnorm),(step,gradnorm)
        optimizer.step();history.append(float(value.detach()));seen+=len(idx);supervised+=count
        if step==1 or step%50==0:
            row={'step':step,'loss':history[-1],'last50_mean':float(np.mean(history[-50:])),
                 'elapsed_seconds':time.monotonic()-started,'rows_seen':seen,'gradient_norm':float(gradnorm)}
            with (out/'progress.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
    final_head=state_sha(model.head);final_backbone=state_sha(model.backbone) if model.backbone else None
    assert final_head!=initial_head
    if model.backbone:assert final_backbone!=initial_backbone
    path=out/'fixed_step_001200.pt'
    torch.save({'arm':arm,'head_state':model.head.state_dict(),'backbone_state':model.backbone.state_dict() if model.backbone else None,
                'step':1200,'protocol_sha256':sha(HERE/'FROZEN.json'),'settings':SETTINGS},path)
    result={'status':'FIXED_TRAINING_COMPLETE','arm':arm,'steps':1200,'rows_seen':seen,'supervised_queries':supervised,
            'first50_mean_loss':float(np.mean(history[:50])),'last50_mean_loss':float(np.mean(history[-50:])),
            'initial_head_sha256':initial_head,'final_head_sha256':final_head,
            'initial_backbone_sha256':initial_backbone,'final_backbone_sha256':final_backbone,
            'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'checkpoint':str(path),'checkpoint_sha256':sha(path),'protocol_sha256':sha(HERE/'FROZEN.json'),
            'elapsed_seconds':time.monotonic()-started,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
            'cal_dev_val_test_read':False,'GPU2_used':False}
    dump(out/'RECEIPT.json',result);(out/'runner.exit').write_text('0\n');print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');parser.add_argument('--freeze',action='store_true')
    parser.add_argument('--arm',choices=['head_only','visual_ft']);args=parser.parse_args()
    if args.smoke:smoke()
    elif args.freeze:freeze()
    elif args.arm:train(args.arm)
    else:parser.error('select smoke, freeze, or arm')
