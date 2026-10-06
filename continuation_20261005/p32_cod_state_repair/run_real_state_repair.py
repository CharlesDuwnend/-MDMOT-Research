#!/usr/bin/env python3
"""Train-only causal owner-state revision audit for MDMT-COD.

The runtime state machine consumes only frozen candidate/replay artifacts.  GT is
loaded in a separate, post-replay diagnostic pass to label final edges; it is
never read by a decision function.  This is a mechanism gate, not official MOT
evaluation.
"""
from __future__ import annotations
import argparse, collections, datetime as dt, gzip, hashlib, json, math, os, statistics, sys
from dataclasses import dataclass, asdict
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path('/home/chenhc/claude_try_MDMOT/continuation_20261005/p32_cod_state_repair')
CACHE = Path('/home/chenhc/mdmot_global_host_stage15_20260905/cache/train')
REPLAY = Path('/home/chenhc/mdmt_cod_revision_audit_20260926/artifacts_v2/replay')
STREAM = Path('/home/chenhc/mdmot_strong_host_stage1_20260905/streams/train')
XML = Path('/raid/datasets/chc_data/MDMT/new_xml')
PAIRS = ['23','25','27','28','29','30','32','39','42','44','45','50','51','53','54','58','63','64','65','66','69','70','74','76','78']


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def stable(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True, separators=(',',':'), default=list).encode()).hexdigest()

def load_json_gz(path):
    with gzip.open(path,'rt',encoding='utf-8') as f: return json.load(f)

def node(x): return (str(x[0]), int(x[1]))

def edge(a,b):
    a,b=node(a),node(b)
    return (a,b) if a[0]=='1' else (b,a)

def edge_key(e): return [list(e[0]),list(e[1])]

@dataclass(frozen=True)
class Proposal:
    pair: str
    ready: int
    a: tuple
    b: tuple
    cost: float
    baseline_accept: bool
    owner_reason: str
    batch_index: int
    source: str = 'frozen_hungarian_real_assignment'
    @property
    def edge(self): return (self.a,self.b)

@dataclass
class Active:
    proposal: Proposal
    published_at: int
    state: str = 'provisional'


def extract_proposals(pair, payload, replay):
    """Extract each selected real assignment once; no labels are consulted."""
    out=[]
    for bi,batch in enumerate(replay):
        ready=int(batch['ready_frame'])
        for assignment in batch.get('assignments',[]):
            if assignment.get('kind')!='real': continue
            a,b=edge(assignment['camera1_node'],assignment['camera2_node'])
            out.append(Proposal(pair,ready,a,b,float(assignment['cost']),
                                bool(assignment.get('owner_accepted',False)),
                                str(assignment.get('owner_reason','unknown')),bi))
    # One real assignment per batch edge; preserve deterministic order.
    out.sort(key=lambda p:(p.ready,p.cost,p.a,p.b,p.batch_index))
    return out


def conflicts(active, p):
    """Endpoint owner conflict: one local track cannot have two mates."""
    vals=[]
    for e,r in active.items():
        if e[0]==p.a[0] and e[0] == p.a: pass
        if (e[0]==p.a and e[1]!=p.b) or (e[1]==p.b and e[0]!=p.a): vals.append((e,r))
        # Cross orientation is fixed, so the two additional endpoint tests are
        # explicit and readable rather than relying on set equality.
        if e[0]==p.a and e[1]!=p.b: pass
        if e[1]==p.b and e[0]!=p.a: pass
        if e[0]==p.a or e[1]==p.b:
            if e != p.edge and (e[0]==p.a or e[1]==p.b):
                if (e,r) not in vals: vals.append((e,r))
    return vals

# Replace the deliberately verbose conflict helper with an exact orientation-safe version.
def endpoint_conflicts(active, p):
    return [(e,r) for e,r in active.items()
            if e != p.edge and (e[0] == p.a or e[1] == p.b)]


def simulate_delay(proposals, W):
    """Equal-information/equal-delay baseline: no provisional revision."""
    by_ready=collections.defaultdict(list)
    for p in proposals: by_ready[p.ready].append(p)
    active={}; accepted=[]; log=[]; pending=[]; timeline=[]
    ready_values=sorted(by_ready)
    for t in ready_values:
        pending.extend(by_ready[t])
        due=[p for p in pending if p.ready+W <= t]
        pending=[p for p in pending if p.ready+W > t]
        for p in sorted(due,key=lambda q:(q.ready,q.cost,q.a,q.b,q.batch_index)):
            c=endpoint_conflicts(active,p)
            if c:
                log.append({'op':'delay_reject_conflict','edge':edge_key(p.edge),'ready':p.ready,
                            'commit_frame':t,'conflicts':[edge_key(x[0]) for x in c]})
                continue
            active[p.edge]=Active(p,t,'final'); accepted.append(p)
            log.append({'op':'delay_finalize','edge':edge_key(p.edge),'ready':p.ready,'commit_frame':t})
        timeline.append({'frame':t,'pending':len(pending),'active':len(active)})
    end=(max(ready_values)+W) if ready_values else W
    for p in sorted(pending,key=lambda q:(q.ready,q.cost,q.a,q.b,q.batch_index)):
        c=endpoint_conflicts(active,p)
        if c: log.append({'op':'delay_reject_conflict','edge':edge_key(p.edge),'ready':p.ready,'commit_frame':end,'conflicts':[edge_key(x[0]) for x in c]})
        else:
            active[p.edge]=Active(p,end,'final'); accepted.append(p)
            log.append({'op':'delay_finalize','edge':edge_key(p.edge),'ready':p.ready,'commit_frame':end})
    return {'mode':'same_information_same_delay','W':W,'active':active,'accepted':accepted,'log':log,'timeline':timeline}


def simulate_repair(proposals,W):
    """Causal provisional publication with bounded lower-cost replacement."""
    by_ready=collections.defaultdict(list)
    for p in proposals: by_ready[p.ready].append(p)
    active={}; log=[]; accepted=[]; ready_values=sorted(by_ready)
    for t in ready_values:
        # Freeze the frontier before consuming evidence at t.
        for e,r in list(active.items()):
            if t-r.proposal.ready > W:
                r.state='final'
        for p in sorted(by_ready[t],key=lambda q:(q.cost,q.a,q.b,q.batch_index)):
            c=endpoint_conflicts(active,p)
            if not c:
                active[p.edge]=Active(p,t,'provisional'); accepted.append(p)
                log.append({'op':'provisional_publish','edge':edge_key(p.edge),'ready':p.ready,'frame':t,'cost':p.cost})
                continue
            open_conf=[(e,r) for e,r in c if r.state=='provisional' and t-r.proposal.ready <= W]
            # Replacement requires a strict cost improvement over every edge it revokes.
            if len(open_conf)==len(c) and open_conf and all(p.cost < r.proposal.cost-1e-12 for _,r in open_conf):
                for e,r in open_conf:
                    del active[e]
                    log.append({'op':'revoke','edge':edge_key(e),'ready':r.proposal.ready,'frame':t,
                                'reason':'lower_cost_conflict','replacement':edge_key(p.edge),
                                'old_cost':r.proposal.cost,'new_cost':p.cost})
                active[p.edge]=Active(p,t,'provisional'); accepted.append(p)
                log.append({'op':'revision_publish','edge':edge_key(p.edge),'ready':p.ready,'frame':t,'cost':p.cost})
            else:
                log.append({'op':'repair_reject','edge':edge_key(p.edge),'ready':p.ready,'frame':t,
                            'cost':p.cost,'conflicts':[edge_key(e) for e,_ in c],
                            'open_conflicts':len(open_conf)})
    end=(max(ready_values)+W) if ready_values else W
    for r in active.values(): r.state='final'
    return {'mode':'provisional_repair','W':W,'active':active,'accepted':accepted,'log':log,'timeline':[],'final_frame':end}


def load_stream(pair,uav):
    by_frame=collections.defaultdict(dict)
    with gzip.open(STREAM/f'{pair}-{uav}.jsonl.gz','rt') as f:
        for line in f:
            row=json.loads(line)
            fr=int(row['frame'])
            for x in row.get('track_outputs',[]):
                by_frame[fr][int(x['local_track_id'])]=x
    return by_frame

def iou(a,b):
    x1,y1=max(a[0],b[0]),max(a[1],b[1]); x2,y2=min(a[2],b[2]),min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]); bb=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    return inter/max(aa+bb-inter,1e-12)

def load_gt(pair,uav):
    by_frame=collections.defaultdict(list)
    root=ET.parse(XML/f'{uav}/{pair}-{uav}.xml').getroot()
    for tr in root.findall('track'):
        gid=int(tr.attrib['id'])
        for b in tr.findall('box'):
            if int(b.attrib.get('outside','0')): continue
            by_frame[int(b.attrib['frame'])+1].append((gid,[float(b.attrib[k]) for k in ('xtl','ytl','xbr','ybr')]))
    return by_frame

def local_gid_maps(pair, streams):
    maps={}
    for u in ('1','2'):
        gt=load_gt(pair,u); votes=collections.defaultdict(collections.Counter)
        for fr,rows in streams[u].items():
            pairs=[]
            for lid,row in rows.items():
                for gid,box in gt.get(fr,[]):
                    ov=iou(row['bbox'],box)
                    if ov>=0.5: pairs.append((-ov,lid,gid))
            used_l=set(); used_g=set()
            for neg,lid,gid in sorted(pairs):
                if lid in used_l or gid in used_g: continue
                used_l.add(lid);used_g.add(gid);votes[lid][gid]+=1
        maps[u]={lid:(cnt.most_common(1)[0][0] if cnt else None) for lid,cnt in votes.items()}
    return maps

def diagnostics(edges, streams, maps, frame):
    correct=false=unknown=wrong_future=coobs=0
    rows=[]
    for e,r in edges.items():
        a,b=e; ga,gb=maps[a[0]].get(a[1]),maps[b[0]].get(b[1])
        if ga is not None and gb is not None:
            if ga==gb: correct+=1; label='correct'
            else: false+=1; label='false'
        else: unknown+=1; label='unknown'
        wrong=0; overlap=0
        cutoff=max(int(frame), int(r.proposal.ready))
        for fr in sorted(set(streams['1']) & set(streams['2'])):
            if fr<=cutoff: continue
            if a[1] in streams[a[0]].get(fr,{}) and b[1] in streams[b[0]].get(fr,{}):
                overlap+=1
                # If either GT map is unavailable, do not convert it to a wrong label.
                if ga is not None and gb is not None and ga!=gb: wrong+=1
        coobs+=overlap; wrong_future+=wrong
        rows.append({'edge':edge_key(e),'ready':r.proposal.ready,'effective':r.published_at,'cutoff_frame':cutoff,
                     'cost':r.proposal.cost,'label':label,'left_gid':ga,'right_gid':gb,
                     'future_coobserved_frames':overlap,'future_wrong_coobserved_frames':wrong})
    return {'correct_edges':correct,'false_edges':false,'unknown_edges':unknown,
            'future_coobserved_frames':coobs,'future_wrong_coobserved_frames':wrong_future,'rows':rows}

def b0_edges(proposals):
    return {p.edge:Active(p,p.ready,'final') for p in proposals if p.baseline_accept}

def summarize(pair, proposals, delay, repair, b0, streams, maps, W):
    end=max([int(x.ready) for x in proposals],default=0)+W
    ds=diagnostics(delay['active'],streams,maps,0)
    rs=diagnostics(repair['active'],streams,maps,0)
    bs=diagnostics(b0,streams,maps,0)
    op=collections.Counter(x['op'] for x in repair['log'])
    changed_delay=(set(delay['active']) ^ set(repair['active']))
    changed_b0=(set(b0) ^ set(repair['active']))
    return {'pair':pair,'W':W,'proposal_count':len(proposals),
            'baseline_selected_owner_accept':sum(p.baseline_accept for p in proposals),
            'delay_final_edges':len(delay['active']),'repair_final_edges':len(repair['active']),'b0_edges':len(b0),
            'repair_ops':dict(op),'repair_revokes':op['revoke'],'repair_revisions':op['revision_publish'],
            'repair_rejections':op['repair_reject'],'repair_changes_vs_delay':len(changed_delay),
            'repair_changes_vs_b0':len(changed_b0),'delay_diagnostics':{k:v for k,v in ds.items() if k!='rows'},
            'repair_diagnostics':{k:v for k,v in rs.items() if k!='rows'},
            'b0_diagnostics':{k:v for k,v in bs.items() if k!='rows'},
            'repair_edge_rows':rs['rows']}

def run_pair(pair,W):
    payload=load_json_gz(CACHE/f'{pair}.json.gz')
    replay=[json.loads(x) for x in gzip.open(REPLAY/f'{pair}.solver.jsonl.gz','rt')]
    proposals=extract_proposals(pair,payload,replay)
    delay=simulate_delay(proposals,W); repair=simulate_repair(proposals,W); b0=b0_edges(proposals)
    streams={u:load_stream(pair,u) for u in ('1','2')}; maps=local_gid_maps(pair,streams)
    summary=summarize(pair,proposals,delay,repair,b0,streams,maps,W)
    return summary, {'proposals':[asdict(p) for p in proposals],
                     'delay_log':delay['log'],'repair_log':repair['log'],
                     'delay_edges':{stable(edge_key(e)):edge_key(e) for e in delay['active']},
                     'repair_edges':{stable(edge_key(e)):edge_key(e) for e in repair['active']},
                     'b0_edges':{stable(edge_key(e)):edge_key(e) for e in b0}}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--W',type=int,default=4); ap.add_argument('--pairs',default=','.join(PAIRS)); ap.add_argument('--out',default=str(ROOT/'runs'))
    args=ap.parse_args(); assert args.W>=0
    out=Path(args.out)/f'W{args.W}'; out.mkdir(parents=True,exist_ok=True)
    pairs=[x for x in args.pairs.split(',') if x]
    summaries=[]; input_hashes={}
    for pair in pairs:
        input_hashes[f'cache/{pair}.json.gz']=sha(CACHE/f'{pair}.json.gz')
        input_hashes[f'replay/{pair}.solver.jsonl.gz']=sha(REPLAY/f'{pair}.solver.jsonl.gz')
        s,detail=run_pair(pair,args.W); summaries.append(s)
        (out/f'{pair}.json').write_text(json.dumps({'summary':s,'detail':detail},indent=2,default=list)+'\n')
        print(json.dumps({k:s[k] for k in ('pair','proposal_count','delay_final_edges','repair_final_edges','repair_revokes','repair_revisions','repair_changes_vs_delay','repair_diagnostics')}),flush=True)
    def macro(mode,key):
        vals=[x[f'{mode}_diagnostics'][key] for x in summaries]
        return {'mean':statistics.mean(vals) if vals else None,'median':statistics.median(vals) if vals else None,'sum':sum(vals)}
    aggregate={'status':'TRAIN_ONLY_STATE_REPAIR_AUDIT_COMPLETE','W':args.W,'pairs':pairs,'pair_count':len(pairs),
               'proposal_count':sum(x['proposal_count'] for x in summaries),
               'repair_revoke_count':sum(x['repair_revokes'] for x in summaries),
               'repair_revision_count':sum(x['repair_revisions'] for x in summaries),
               'pair_macro':{mode:{k:macro(mode,k) for k in ('correct_edges','false_edges','unknown_edges','future_wrong_coobserved_frames','future_coobserved_frames')} for mode in ('b0','delay','repair')},
               'repair_changed_pairs':sum(x['repair_changes_vs_delay']>0 for x in summaries),
               'summaries':summaries,'input_sha256':input_hashes,
               'runtime_decision_uses_gt':False,'gt_usage':'post-replay train-only diagnostic labels; no GT in state transitions',
               'official_val_or_test':False,'formal_mot_metrics':False,
               'code_sha256':sha(Path(__file__))}
    (out/'AGGREGATE.json').write_text(json.dumps(aggregate,indent=2)+'\n')
    print(json.dumps({'aggregate':str(out/'AGGREGATE.json'),'status':aggregate['status'],'code_sha256':aggregate['code_sha256']}))

if __name__=='__main__': main()
