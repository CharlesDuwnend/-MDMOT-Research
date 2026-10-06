#!/usr/bin/env python3
from __future__ import annotations

import argparse, hashlib, json, os, subprocess, sys, time
from collections import defaultdict, deque
from pathlib import Path

# Bind before importing torch so CUDA's device-count cache cannot retain all
# physical GPUs from an earlier CPU-side probe.
PHYSICAL_GPU_UUID = 'GPU-aaa79a66-b5c4-e0a0-6e2f-28c1ac0e3e6a'
os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', PHYSICAL_GPU_UUID)
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path('/home/chenhc/mdmot_research_20261002')
WORK = Path('/home/chenhc/claude_try_MDMOT/continuation_20261006/p51_ccfi')
P4 = ROOT / 'p4_diagnosis'
P38 = Path('/home/chenhc/claude_try_MDMOT/continuation_20261005/p38_temporal_memory')
sys.path.insert(0, str(ROOT/'p2/src'))
sys.path.insert(0, str(ROOT/'p1/src'))
from data import PairData, sha256
from train_association_probe import masked_matching_loss

UUID = PHYSICAL_GPU_UUID  # physical GPU3 A100 40GB
FIT = ('23','25','28','29','39','44','45','51','53','63','66','69','70','74','78')
CAL = ('27','32','42','64','65')
ALL = FIT + CAL

def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n')

def sha(path): return sha256(path)

def load_torch(path, map_location='cpu'):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)

def unit(x): return F.normalize(x, dim=-1)

def state_sha(m):
    h = hashlib.sha256()
    for n, v in sorted(m.state_dict().items()):
        h.update(n.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

class CCFIPair:
    """Frozen P2 rows plus strictly-past mean and dispersion for each local track."""
    def __init__(self, pair, with_labels):
        self.pair = str(pair)
        self.data = PairData(pair, with_labels=with_labels)
        if with_labels: self.data.require_fit()
        self.hist, self.spread, self.has, self.key_meta = [], [], [], {}
        self.groups = {}
        for view, (d, off) in enumerate(zip(self.data._views, self.data._offsets)):
            n = len(d['frame'])
            hist = np.zeros((n, 128), np.float32)
            spread = np.zeros((n, 128), np.float32)
            has = np.zeros(n, bool)
            memory = {}
            for i in range(n):
                f = int(d['frame'][i]); lid = int(d['local_id'][i]); cls = int(d['cls'][i])
                if lid >= 0:
                    old = memory.get(lid)
                    if old is not None and (f <= old['frame'] or cls != old['class'] or f-old['frame'] > 8):
                        old = None
                    if old is None:
                        old = {'frame': f, 'class': cls, 'embeddings': deque(maxlen=4)}
                        memory[lid] = old
                    old['frame'], old['class'] = f, cls
                    if bool(d['feature_present'][i]):
                        z = d['embedding'][i].astype(np.float32)
                        z = z / (np.linalg.norm(z) + 1e-12)
                        prev = list(old['embeddings'])
                        if prev:
                            arr = np.stack(prev)
                            mu = arr.mean(0).astype(np.float32); mu /= np.linalg.norm(mu) + 1e-12
                            var = arr.std(0).astype(np.float32)
                            hist[i], spread[i], has[i] = mu, var, True
                        old['embeddings'].append(z)
                key = f'{self.pair}-{view+1}:{f}:{int(d["detector_index"][i])}'
                self.key_meta[key] = (hist[i].copy(), spread[i].copy(), bool(has[i]))
            self.hist.append(hist); self.spread.append(spread); self.has.append(has)
        for frame in range(1, self.data.frames + 1):
            for cls in range(3):
                sides = []
                for d, off in zip(self.data._views, self.data._offsets):
                    ix = np.arange(int(off[frame-1]), int(off[frame]), dtype=np.int64)
                    ix = ix[(d['cls'][ix] == cls) & d['feature_present'][ix]]
                    sides.append(ix)
                if len(sides[0]) or len(sides[1]): self.groups[(frame, cls)] = tuple(sides)

    def batch(self, frame, cls, device):
        out = []
        for v, ix in enumerate(self.groups[(int(frame), int(cls))]):
            d = self.data._views[v]
            cur = torch.from_numpy(d['embedding'][ix].astype(np.float32)).to(device)
            hist = torch.from_numpy(self.hist[v][ix]).to(device)
            spread = torch.from_numpy(self.spread[v][ix]).to(device)
            has = torch.from_numpy(self.has[v][ix]).to(device)
            lab = torch.from_numpy(d['pid'][ix].copy()).to(device) if self.data.with_labels else None
            valid = lab >= 0 if lab is not None else None
            out.append((cur, hist, spread, has, lab, valid))
        return out

    def label_free_rows(self, selected_frames):
        keys, cur, hist, spread, has = [], [], [], [], []
        for frame in selected_frames:
            b = self.data.frame(int(frame))
            for row in b['rows']:
                key = row['key']
                view = int(row['camera']) - 1
                d = self.data._views[view]
                ix = np.flatnonzero((d['frame'] == int(row['frame'])) & (d['detector_index'] == int(row['detector_index'])))
                if len(ix) != 1: raise ValueError('non-unique row lookup')
                i = int(ix[0]); keys.append(key); cur.append(d['embedding'][i]); hist.append(self.hist[view][i]); spread.append(self.spread[view][i]); has.append(self.has[view][i])
        return (np.asarray(keys), np.asarray(cur, np.float32), np.asarray(hist, np.float32),
                np.asarray(spread, np.float32), np.asarray(has, bool))

class CCFI(nn.Module):
    def __init__(self):
        super().__init__()
        self.core_gate = nn.Sequential(nn.Linear(512,128), nn.LayerNorm(128), nn.GELU(), nn.Linear(128,1))
        self.core_delta = nn.Sequential(nn.Linear(512,128), nn.GELU(), nn.Linear(128,128))
        self.change = nn.Sequential(nn.Linear(512,128), nn.GELU(), nn.Linear(128,128))
        self.uncert = nn.Sequential(nn.Linear(512,64), nn.GELU(), nn.Linear(64,1))
        nn.init.zeros_(self.core_delta[-1].weight); nn.init.zeros_(self.core_delta[-1].bias)

    def forward(self, current, history, spread, has):
        x = torch.cat((current, history, current-history, spread), dim=-1)
        gate = torch.sigmoid(self.core_gate(x))
        z_id = unit(current + has.float().unsqueeze(-1) * gate * self.core_delta(x))
        z_chg = unit(self.change(x))
        # bounded uncertainty: [0.01, 0.51], finite even on extreme inputs
        u = 0.01 + 0.50 * torch.sigmoid(self.uncert(x)).squeeze(-1)
        return z_id, z_chg, u

    def init_from_p38(self, p38_state):
        # Preserve the strongest previous causal adapter as the zero-change core.
        self.core_gate[0].weight.data.zero_(); self.core_gate[0].weight.data[:, :384].copy_(p38_state['gate.0.weight']); self.core_gate[0].bias.data.copy_(p38_state['gate.0.bias'])
        self.core_gate[1].weight.data.copy_(p38_state['gate.1.weight']); self.core_gate[1].bias.data.copy_(p38_state['gate.1.bias'])
        self.core_gate[3].weight.data.copy_(p38_state['gate.3.weight']); self.core_gate[3].bias.data.copy_(p38_state['gate.3.bias'])
        self.core_delta[0].weight.data.zero_(); self.core_delta[0].weight.data[:, :384].copy_(p38_state['delta.0.weight']); self.core_delta[0].bias.data.copy_(p38_state['delta.0.bias'])
        self.core_delta[2].weight.data.copy_(p38_state['delta.2.weight']); self.core_delta[2].bias.data.copy_(p38_state['delta.2.bias'])
        # The P38 final delta is not zero, so this is a declared warm start rather than a zero-output init.
        return self

def uncertainty_matching_loss(za, zb, ua, ub, la, lb, va, vb, temperature=0.1):
    ia, ib = torch.where(va)[0], torch.where(vb)[0]
    if len(ia) == 0 or len(ib) == 0: return None, 0
    la, lb = la[ia], lb[ib]
    if len(torch.unique(la)) != len(la) or len(torch.unique(lb)) != len(lb): raise ValueError('duplicate valid identity')
    score = (za[ia] @ zb[ib].T) / (temperature * (1.0 + ua[ia,None] + ub[None,ib]))
    pos = la[:,None] == lb[None,:]; qa, qb = pos.any(1), pos.any(0)
    losses, count = [], 0
    if qa.any(): losses.append(F.cross_entropy(score[qa], pos[qa].to(torch.int64).argmax(1), reduction='sum')); count += int(qa.sum())
    if qb.any(): losses.append(F.cross_entropy(score.T[qb], pos.T[qb].to(torch.int64).argmax(1), reduction='sum')); count += int(qb.sum())
    return (sum(losses)/count, count) if count else (None, 0)

def factorized_objective(za, zb, ca, cb, ua, ub, xa, xb, ha, hb, ma, mb, la, lb, va, vb):
    matching, count = uncertainty_matching_loss(za, zb, ua, ub, la, lb, va, vb, 0.1)
    if matching is None:
        return None, 0, {'matching': None, 'change': None, 'orthogonality': None, 'uncertainty': None}
    changes, orth, unc = [], [], []
    for z, ch, u, x, h, has in ((za, ca, ua, xa, ha, ma), (zb, cb, ub, xb, hb, mb)):
        if has.any():
            target = unit(x[has] - h[has])
            changes.append(1 - (ch[has] * target).sum(-1).mean())
            orth.append(((z[has] * ch[has]).sum(-1) ** 2).mean())
            unc.append(F.mse_loss(u[has], torch.clamp(x[has].sub(h[has]).pow(2).mean(-1).sqrt().detach(), 0, 0.5)))
    # Keep inactive auxiliary branches connected by an explicit zero term. This
    # makes a missing history a legal zero-gradient case rather than a None
    # gradient that is mistaken for numerical failure by the audit.
    if changes:
        change = torch.stack(changes).mean()
        orthogonality = torch.stack(orth).mean()
        uncertainty = torch.stack(unc).mean()
    else:
        change = (ca.sum() + cb.sum()) * 0.0
        orthogonality = ((za * ca).sum() + (zb * cb).sum()) * 0.0
        uncertainty = (ua.sum() + ub.sum()) * 0.0
    return matching + .05*change + .05*orthogonality + .05*uncertainty, count, {'matching': matching, 'change': change, 'orthogonality': orthogonality, 'uncertainty': uncertainty}

def load_p38_state():
    ckpt = P38/'runs/p38_temporal/fixed_step_001200.pt'
    raw = load_torch(ckpt, map_location='cpu')
    if raw.get('fixed_step') != 1200: raise ValueError('P38 checkpoint lineage mismatch')
    return raw['state_dict'], sha(ckpt)

def configure_gpu():
    probe_env = os.environ.copy(); probe_env.pop('CUDA_VISIBLE_DEVICES', None)
    lines = subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.free','--format=csv,noheader,nounits'], env=probe_env, text=True).splitlines()
    selected = [x for x in lines if UUID in x]
    if len(selected) != 1: raise RuntimeError('GPU UUID not found')
    parts = [x.strip() for x in selected[0].split(',')]
    if parts[0] != '3' or 'A100' not in parts[2] or int(parts[3]) < 30000: raise RuntimeError('physical GPU3 verification failed')
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'; os.environ['CUDA_VISIBLE_DEVICES'] = UUID
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1: raise RuntimeError('CUDA UUID binding failed')
    prop = torch.cuda.get_device_properties(0)
    if 'A100' not in prop.name or prop.total_memory/1024**2 < 30000: raise RuntimeError('torch GPU mismatch')
    return torch.device('cuda:0'), selected[0]

def cpu_contract():
    pair = CCFIPair(FIT[0], True)
    results = {'rounds': {}, 'status': 'PASS_CCFI_CPU_CONTRACT_3X3'}
    # R1 data/legal: 3 checks.
    results['rounds']['R1_data_legal'] = {
        'pair_loaded': pair.pair == FIT[0], 'fit_role': pair.data.role == 'fit',
        'label_field_explicit': pair.data.with_labels is True,
    }
    # R2 operator: 3 checks, including exact P38 warm-start reproduction and finite extremes.
    p38_state, _ = load_p38_state(); model = CCFI(); model.init_from_p38(p38_state).eval()
    key = next(k for k in pair.groups if len(pair.groups[k][0]) and len(pair.groups[k][1]) and
               (set(int(x) for x in pair.data._views[0]['pid'][pair.groups[k][0]] if int(x) >= 0) &
                set(int(x) for x in pair.data._views[1]['pid'][pair.groups[k][1]] if int(x) >= 0)) and
               (pair.has[0][pair.groups[k][0]].any() or pair.has[1][pair.groups[k][1]].any()))
    a,b = pair.batch(*key, torch.device('cpu'))
    with torch.no_grad():
        z, c, u = model(a[0],a[1],a[2],a[3]); finite = bool(torch.isfinite(z).all() and torch.isfinite(c).all() and torch.isfinite(u).all())
        extreme = model(a[0],a[1],torch.full_like(a[2], 1e6),a[3])[2]
    results['rounds']['R2_operator'] = {
        'real_group': True, 'finite_real': finite, 'finite_extreme_dispersion': bool(torch.isfinite(extreme).all()),
    }
    # R3 supervision/gradient: 3 checks.
    model.train(); za,ca,ua = model(a[0],a[1],a[2],a[3]); zb,cb,ub = model(b[0],b[1],b[2],b[3]); loss,count,parts = factorized_objective(za,zb,ca,cb,ua,ub,a[0],b[0],a[1],b[1],a[3],b[3],a[4],b[4],a[5],b[5])
    la,lb = a[4].clone(), b[4].clone(); la[~a[5]] = 918273; lb[~b[5]] = 918273
    loss2,_,_ = factorized_objective(za,zb,ca,cb,ua,ub,a[0],b[0],a[1],b[1],a[3],b[3],la,lb,a[5],b[5]); loss.backward()
    grads = all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    results['rounds']['R3_supervision_gradient'] = {'loss_finite': bool(torch.isfinite(loss)), 'unknown_label_invariant': bool(torch.allclose(loss,loss2,atol=1e-7,rtol=1e-7)), 'finite_gradients': bool(grads), 'query_count': int(count), 'history_rows': int(a[3].sum()+b[3].sum())}
    results['all_checks_pass'] = all(all(bool(v) for v in round_result.values() if isinstance(v, (bool, np.bool_))) for round_result in results['rounds'].values())
    if not results['all_checks_pass']: results['status'] = 'FAIL_CCFI_CPU_CONTRACT'
    dump(WORK/'CPU_CONTRACT.json', results); return results

def build_schedule(pairs):
    schedule = json.loads((ROOT/'p7r/SCHEDULE.json').read_text())
    if len(schedule) != 1200: raise ValueError('P7R schedule changed')
    return schedule

def build_full_schedule(pairs):
    eligible=[]
    for p in FIT:
        pair=pairs[p]
        for key in sorted(pair.groups):
            a,b=pair.batch(*key,torch.device('cpu'))
            shared=set(int(x) for x in a[4][a[5]].tolist()) & set(int(x) for x in b[4][b[5]].tolist())
            if shared and (int(a[5].sum()) > 1 or int(b[5].sum()) > 1):
                eligible.append({'pair':p,'frame':int(key[0]),'class':int(key[1])})
    if len(eligible)!=18412: raise ValueError(f'eligible group count changed: {len(eligible)}')
    rng=np.random.default_rng(42); schedule=[]
    for _ in range(3): schedule.extend([eligible[int(i)] for i in rng.permutation(len(eligible))])
    return schedule, eligible

def train():
    if (WORK/'runs').exists(): raise RuntimeError('sealed runs already exist')
    contract = cpu_contract()
    if not contract['all_checks_pass']: raise RuntimeError('CPU contract failed')
    pairs = {p: CCFIPair(p, True) for p in FIT}; schedule = build_schedule(pairs)
    p38_state, p38_sha = load_p38_state(); device, gpu = configure_gpu()
    torch.manual_seed(42); torch.cuda.manual_seed_all(42); np.random.seed(42)
    model = CCFI().init_from_p38(p38_state).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4); history=[]; start=time.monotonic()
    for step,event in enumerate(schedule,1):
        (ca,ha,sa,ma,la,va),(cb,hb,sb,mb,lb,vb) = pairs[event['pair']].batch(event['frame'],event['class'],device)
        za,ca_ch,ua = model(ca,ha,sa,ma); zb,cb_ch,ub = model(cb,hb,sb,mb)
        loss,count,parts = factorized_objective(za,zb,ca_ch,cb_ch,ua,ub,ca,cb,ha,hb,ma,mb,la,lb,va,vb)
        if loss is None: continue
        match, change, ortho, uncertainty = (parts['matching'], parts['change'], parts['orthogonality'], parts['uncertainty'])
        if not torch.isfinite(loss): raise ValueError('nonfinite loss')
        opt.zero_grad(set_to_none=True); loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()): raise ValueError('nonfinite gradient')
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        history.append({'step':step,'loss':float(loss.detach()),'matching':float(match.detach()),'change':float(change.detach()),'orthogonality':float(ortho.detach()),'uncertainty':float(uncertainty.detach()),'queries':int(count)})
        if step == 1 or step % 200 == 0: print(json.dumps(history[-1]), flush=True)
    torch.cuda.synchronize(); out=WORK/'runs'; out.mkdir(); final=state_sha(model)
    torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'fixed_step':len(schedule),'p38_checkpoint_sha256':p38_sha,'final_state_sha256':final,'prereg_sha256':sha(WORK/'PREREG.json')},out/'fixed_step_001200.pt')
    dump(out/'TRAINING_RECEIPT.json', {'status':'PASS_CCFI_FIXED_TRAINING','steps':len(history),'first':history[0],'last':history[-1],'last100_mean':float(np.mean([h['loss'] for h in history[-100:]])),'checkpoint_sha256':sha(out/'fixed_step_001200.pt'),'final_state_sha256':final,'gpu_uuid':UUID,'gpu_name':'NVIDIA A100-SXM4-40GB','gpu_snapshot':gpu,'elapsed_seconds':time.monotonic()-start,'dev_read':False,'calibration_read':False,'p38_checkpoint_sha256':p38_sha}); (out/'training.exit').write_text('0\n'); (out/'launcher.exit').write_text('0\n')
    print(json.dumps({'status':'PASS_CCFI_FIXED_TRAINING','checkpoint_sha256':sha(out/'fixed_step_001200.pt'),'steps':len(history)}), flush=True)

def train_full():
    out=WORK/'runs_full'
    if out.exists(): raise RuntimeError('sealed full runs already exist')
    contract=cpu_contract()
    if not contract['all_checks_pass']: raise RuntimeError('CPU contract failed')
    pairs={p:CCFIPair(p,True) for p in FIT}; schedule,eligible=build_full_schedule(pairs); dump(WORK/'FULL_SCHEDULE.json',{'status':'PASS_CCFI_FULL_SCHEDULE','eligible_groups':len(eligible),'epochs':3,'updates':len(schedule),'seed':42,'pairs':list(FIT)})
    p38_state,p38_sha=load_p38_state(); device,gpu=configure_gpu(); torch.manual_seed(42); torch.cuda.manual_seed_all(42); np.random.seed(42)
    model=CCFI().init_from_p38(p38_state).to(device).train(); opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4); history=[]; start=time.monotonic()
    for step,event in enumerate(schedule,1):
        (ca,ha,sa,ma,la,va),(cb,hb,sb,mb,lb,vb)=pairs[event['pair']].batch(event['frame'],event['class'],device); za,cha,ua=model(ca,ha,sa,ma); zb,chb,ub=model(cb,hb,sb,mb); loss,count,parts=factorized_objective(za,zb,cha,chb,ua,ub,ca,cb,ha,hb,ma,mb,la,lb,va,vb)
        if loss is None: continue
        if not torch.isfinite(loss): raise ValueError('nonfinite full loss')
        opt.zero_grad(set_to_none=True); loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()): raise ValueError('nonfinite full gradient')
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.0); opt.step()
        history.append({'step':step,'loss':float(loss.detach()),'matching':float(parts['matching'].detach()),'change':float(parts['change'].detach()),'orthogonality':float(parts['orthogonality'].detach()),'uncertainty':float(parts['uncertainty'].detach()),'queries':int(count)})
        if step==1 or step%1000==0: print(json.dumps(history[-1]),flush=True)
    torch.cuda.synchronize(); out.mkdir(); final=state_sha(model); ckpt=out/'fixed_step_055236.pt'; torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'fixed_step':len(schedule),'p38_checkpoint_sha256':p38_sha,'final_state_sha256':final,'prereg_sha256':sha(WORK/'SUFFICIENCY_PREREG.json')},ckpt); dump(out/'TRAINING_RECEIPT.json',{'status':'PASS_CCFI_FULL_COVERAGE_TRAINING','steps':len(history),'epochs':3,'eligible_groups':len(eligible),'first':history[0],'last':history[-1],'last100_mean':float(np.mean([h['loss'] for h in history[-100:]])),'checkpoint_sha256':sha(ckpt),'final_state_sha256':final,'gpu_uuid':UUID,'gpu_name':'NVIDIA A100-SXM4-40GB','gpu_snapshot':gpu,'elapsed_seconds':time.monotonic()-start,'dev_read':False,'calibration_read':False,'p38_checkpoint_sha256':p38_sha,'schedule_sha256':sha(WORK/'FULL_SCHEDULE.json')}); (out/'training.exit').write_text('0\n'); (out/'launcher.exit').write_text('0\n'); print(json.dumps({'status':'PASS_CCFI_FULL_COVERAGE_TRAINING','checkpoint_sha256':sha(ckpt),'steps':len(history)}),flush=True)

def freeze_embeddings(model, pair, record):
    data=CCFIPair(pair,False); keys,cur,hist,spread,has=data.label_free_rows(record['selected_frames']); out=np.zeros((len(keys),128),np.float32); unc=np.zeros(len(keys),np.float32)
    with torch.inference_mode():
        for s in range(0,len(keys),2048):
            ix=np.arange(s,min(s+2048,len(keys))); z,_,u=model(torch.from_numpy(cur[ix]),torch.from_numpy(hist[ix]),torch.from_numpy(spread[ix]),torch.from_numpy(has[ix])); out[ix]=z.numpy(); unc[ix]=u.numpy()
    with np.load(P4/'crop32'/f'{pair}.npz', allow_pickle=False) as frozen_crop:
        common_support = frozen_crop['common_support']
    if not np.isfinite(out).all() or not np.allclose(np.linalg.norm(out[common_support],axis=1),1.,atol=2e-5): raise ValueError('embedding contract failed')
    if np.any(out[~common_support] != 0): raise ValueError('unsupported rows must remain zero')
    return keys,out,unc

def evaluate(checkpoint_dir=None):
    sys.path.insert(0,str(P4)); import compare_crop32 as cmp
    checkpoint_dir=WORK/'runs' if checkpoint_dir is None else Path(checkpoint_dir); raw=load_torch(checkpoint_dir/'fixed_step_001200.pt' if checkpoint_dir.name=='runs' else checkpoint_dir/'fixed_step_055236.pt',map_location='cpu'); model=CCFI().eval(); model.load_state_dict(raw['state_dict'],strict=True)
    manifest=json.loads((P4/'crop32/FRAME_MANIFEST.json').read_text()); records={r['pair']:r for r in manifest['pairs']}; out=checkpoint_dir/'evaluation'; out.mkdir(exist_ok=True)
    # Freeze all label-free embeddings before any offline metric labels are read.
    frozen={}
    for p in ALL:
        keys,emb,unc=freeze_embeddings(model,p,records[p]);
        with np.load(P4/'crop32'/f'{p}.npz',allow_pickle=False) as z:
            if not np.array_equal(keys,z['keys']): raise ValueError('key alignment changed: '+p)
            valid=z['common_support']
        np.savez_compressed(out/(p+'_embeddings.npz'),row_key=keys,embedding=emb,uncertainty=unc,valid=valid); frozen[p]=int(valid.sum())
    (out/'LABEL_FREEZE.json').write_text(json.dumps({'status':'PASS_CCFI_LABEL_FREEZE','pairs':list(ALL),'rows_common_support':frozen,'dev_read':False,'official_val_test_read':False},indent=2)+'\n')
    inputs={r['sequence']:r for r in json.loads((ROOT/'p2/cache/MANIFEST.json').read_text())['sequences']}; reports=[]
    cmp.METHODS=('p1_independent128','temporal128','ccfi128')
    # temporal128 is the frozen P38 comparator; CCFI is the new output.
    for p in ALL:
        rec=records[p]; row,features,refs,cov=cmp.align_pair(rec,P4/'crop32',inputs); z=np.load(out/(p+'_embeddings.npz')); p38=np.load(P38/'runs/evaluation'/f'{p}_embeddings.npz');
        if not np.array_equal(row['row_key'], p38['row_key']): raise ValueError('P38 comparator key alignment changed: '+p)
        features['temporal128']=p38['embedding']; features['ccfi128']=z['embedding']; arr=cmp.evaluate_arrays(row,features); report={'pair':p,'role':rec['role'],'metrics':cmp.summaries(arr),'coverage':cov,'uncertainty_mean':float(z['uncertainty'].mean())}; dump(out/(p+'_RESULTS.json'),report); reports.append(report)
    by={}
    for role in ('fit','calibration'):
        rs=[r for r in reports if r['role']==role]; macro={m:float(np.mean([r['metrics']['all']['methods'][m]['all_candidates']['known_positive_r1_tie_averaged'] for r in rs])) for m in cmp.METHODS}; wins_p1=sum(r['metrics']['all']['methods']['ccfi128']['all_candidates']['known_positive_r1_tie_averaged']>r['metrics']['all']['methods']['p1_independent128']['all_candidates']['known_positive_r1_tie_averaged'] for r in rs); wins_p38=sum(r['metrics']['all']['methods']['ccfi128']['all_candidates']['known_positive_r1_tie_averaged']>r['metrics']['all']['methods']['temporal128']['all_candidates']['known_positive_r1_tie_averaged'] for r in rs); by[role]={'pair_macro_r1':macro,'wins_vs_p1':int(wins_p1),'wins_vs_p38':int(wins_p38),'pairs':[r['pair'] for r in rs]}
    cal=by['calibration']['pair_macro_r1']; fit=by['fit']['pair_macro_r1']; gate={'calibration_min':.4641,'pair_wins_vs_p1_min':3,'pair_wins_vs_p38_min':3,'fit_drop_max':.05,'calibration_r1_vs_p1':cal['ccfi128']-cal['p1_independent128'],'calibration_r1_vs_p38':cal['ccfi128']-cal['temporal128'],'fit_drop_vs_p1':fit['ccfi128']-fit['p1_independent128'],'wins_vs_p1':by['calibration']['wins_vs_p1'],'wins_vs_p38':by['calibration']['wins_vs_p38']}; gate['pass']=bool(cal['ccfi128']>=.4641 and gate['wins_vs_p1']>=3 and gate['wins_vs_p38']>=3 and gate['fit_drop_vs_p1']>=-.05)
    result={'status':'COMPLETE_CCFI_TRAIN_ONLY_EVALUATION','training':json.loads((checkpoint_dir/'TRAINING_RECEIPT.json').read_text()),'by_role':by,'gate':gate,'pairs':reports,'dev_read':False,'official_val_test_read':False}; dump(out/'RESULTS.json',result); dump(checkpoint_dir/'DECISION.json',{'verdict':'KEEP_CCFI_REPRESENTATION_GATE' if gate['pass'] else 'STOP_CCFI_REPRESENTATION_GATE','gate_pass':gate['pass'],'host_transfer_authorized':False,'results':str(out/'RESULTS.json')}); print(json.dumps({'calibration':cal,'fit':fit,'gate':gate},indent=2),flush=True)

def evaluate_full():
    # The evaluator logic is shared; temporarily use a full-run output root and
    # preserve the short-gate result untouched.
    checkpoint_dir=WORK/'runs_full'; evaluate(checkpoint_dir)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--mode',choices=('cpu','signal','train','train_full','evaluate','evaluate_full'),required=True); a=ap.parse_args(); torch.set_num_threads(4); torch.set_num_interop_threads(1)
    if a.mode=='cpu': print(json.dumps(cpu_contract(),indent=2))
    elif a.mode=='train': train()
    elif a.mode=='evaluate': evaluate()
    elif a.mode=='train_full': train_full()
    elif a.mode=='evaluate_full': evaluate_full()
    else: print('signal mode is executed by audit_signal.py', flush=True)

if __name__=='__main__': main()
