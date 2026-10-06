#!/usr/bin/env python3
"""Pre-training, fixed-form train-free signal gate for CCFI.

The alpha and uncertainty rule are declared constants; there is no calibration
search. Features are frozen before offline labels are opened.
"""
import json, sys
from pathlib import Path
import numpy as np

WORK = Path('/home/chenhc/claude_try_MDMOT/continuation_20261006/p51_ccfi')
ROOT = Path('/home/chenhc/mdmot_research_20261002')
P4 = ROOT/'p4_diagnosis'
sys.path.insert(0, str(WORK/'src'))
from run_ccfi import CCFIPair, CAL, FIT, unit

def main():
    manifest=json.loads((P4/'crop32/FRAME_MANIFEST.json').read_text()); records={r['pair']:r for r in manifest['pairs']}; out=WORK/'signal'; out.mkdir(exist_ok=True)
    # Declared fixed rule: history is a 0.5 convex contribution; higher dispersion
    # attenuates only the history contribution, never the current observation.
    alpha=0.5; dispersion_scale=2.0; frozen={}
    for p in CAL:
        d=CCFIPair(p,False); keys,cur,hist,spread,has=d.label_free_rows(records[p]['selected_frames']); disp=np.sqrt((spread**2).mean(-1)); weight=np.where(has,alpha*np.exp(-dispersion_scale*disp),0.0); z=cur.copy(); z[has]=(1-weight[has,None])*cur[has]+weight[has,None]*hist[has]; z=z/(np.linalg.norm(z,axis=1,keepdims=True)+1e-12); path=out/(p+'_embeddings.npz'); np.savez_compressed(path,row_key=keys,embedding=z,history_used=has,history_weight=weight); frozen[p]={'rows':int(len(keys)),'history_rows':int(has.sum()),'sha256':__import__('hashlib').sha256(path.read_bytes()).hexdigest()}
    (out/'LABEL_FREEZE.json').write_text(json.dumps({'status':'PASS_CCFI_TRAIN_FREE_LABEL_FREEZE','pairs':list(CAL),'alpha':alpha,'dispersion_scale':dispersion_scale,'rule':'fixed_no_search','frozen':frozen},indent=2)+'\n')
    sys.path.insert(0,str(P4)); import compare_crop32 as cmp
    cmp.METHODS=('p1_independent128','ccfi_static')
    inputs={r['sequence']:r for r in json.loads((ROOT/'p2/cache/MANIFEST.json').read_text())['sequences']}; reports=[]
    for p in CAL:
        row,features,refs,cov=cmp.align_pair(records[p],P4/'crop32',inputs); z=np.load(out/(p+'_embeddings.npz')); features['ccfi_static']=z['embedding']; arr=cmp.evaluate_arrays(row,features); r={'pair':p,'metrics':cmp.summaries(arr),'coverage':cov}; (out/(p+'_RESULTS.json')).write_text(json.dumps(r,indent=2)+'\n'); reports.append(r)
    base=[r['metrics']['all']['methods']['p1_independent128']['all_candidates']['known_positive_r1_tie_averaged'] for r in reports]; cand=[r['metrics']['all']['methods']['ccfi_static']['all_candidates']['known_positive_r1_tie_averaged'] for r in reports]; wins=sum(c>b for c,b in zip(cand,base)); result={'status':'PASS_CCFI_TRAIN_FREE_SIGNAL_GATE' if wins>=3 and float(np.mean(cand))>float(np.mean(base)) else 'STOP_CCFI_TRAIN_FREE_SIGNAL','rule':{'alpha':alpha,'dispersion_scale':dispersion_scale},'pair_r1':{p:{'p1':b,'ccfi_static':c,'delta':c-b} for p,b,c in zip(CAL,base,cand)},'macro':{'p1':float(np.mean(base)),'ccfi_static':float(np.mean(cand)),'delta':float(np.mean(cand)-np.mean(base))},'wins':int(wins),'required_wins':3,'training_authorized':bool(wins>=3 and np.mean(cand)>np.mean(base))}; (WORK/'TRAIN_FREE_SIGNAL.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
