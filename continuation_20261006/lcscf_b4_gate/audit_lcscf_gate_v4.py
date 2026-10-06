#!/usr/bin/env python3
"""Independent 3 x 3 audit of the corrected block-1 effectiveness gate."""
import hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parent
RUN=ROOT/'runs/block1_600_v4'
ARMS=['B4','LCSCF_B4']; PAIRS=['29','30','32','39']

def check(name,ok,evidence):
    if not ok: raise AssertionError(f'{name}: {evidence}')
    return {'check':name,'status':'PASS','evidence':evidence}

def load_events():
    rows=[]
    for line in (RUN/'events.jsonl').read_text().splitlines():
        rows.append(json.loads(line))
    return rows

def finite(x): return isinstance(x,(int,float)) and math.isfinite(float(x))

def main():
    rows=load_events(); metrics={a:json.loads((RUN/f'metrics_{a}.json').read_text()) for a in ARMS}; meta=json.loads((RUN/'run_meta.json').read_text())
    trains=[r for r in rows if r.get('event')=='train']; evals=[r for r in rows if r.get('event')=='eval']; summaries=[r for r in rows if r.get('event')=='eval_summary']
    rounds=[]
    train_counts={a:sum(r.get('arm')==a for r in trains) for a in ARMS}
    train_finite=all(finite(r.get('loss')) and finite(r.get('gradient_l1')) and all(finite(v) for v in r.get('terms',{}).values()) for r in trains)
    rounds.append({'round':'R1_EXECUTION_AND_FINITE','checks':[
      check('both_arms_have_exact_600_updates',train_counts=={'B4':600,'LCSCF_B4':600},{'train_counts':train_counts}),
      check('all_logged_train_scalars_finite',train_finite,{'train_rows':len(trains)}),
      check('completion_and_no_official_access',meta.get('status')=='COMPLETE' and meta.get('official_val_access') is False and meta.get('test_access') is False and len(summaries)==2,{'status':meta.get('status'),'official_val_access':meta.get('official_val_access'),'test_access':meta.get('test_access'),'summaries':len(summaries)})
    ]})
    # independently aggregate event-level rank/mrr/brier, then compare metric files
    recomputed={}
    for arm in ARMS:
      # events are sequential: each arm has eval rows, summary has pair metrics; identify using summary order
      sm=next(r for r in summaries if r['arm']==arm)
      pmetrics={}
      for pair in PAIRS:
        er=[r for r in evals if r['pair_id']==pair]
        # Evaluation rows do not carry arm; the log contains one full eval block per arm.
        # Split by contiguous summary event positions using event count from metric file and identical ordering.
      # use event rows per arm by scanning train/eval state transition
    # Reparse with state machine
    current=None; per={a:[] for a in ARMS}
    for r in rows:
      if r.get('event')=='train': current=r['arm']
      elif r.get('event')=='eval_summary': current=None
      elif r.get('event')=='eval' and current in per: per[current].append(r)
    # summary is emitted after each eval block, so current becomes None only after summary; works.
    for arm in ARMS:
      # fallback split if state parser sees no rows
      assert per[arm],arm
      pair={}
      for p in PAIRS:
        er=[r for r in per[arm] if r['pair_id']==p]
        present=[r for r in er if r['candidate_present']]
        pair[p]={
          'events':len(er),
          'candidate_present_events':len(present),
          'recall':sum(bool(r['rank1']) for r in present)/len(present),
          'mrr':sum(float(r['mrr']) for r in present)/len(present),
          'brier':sum((float(r['dustbin_probability'])-float(r['dustbin_target']))**2 for r in er)/len(er),
        }
      recomputed[arm]=pair
    metric_match=[]
    for arm in ARMS:
      for p in PAIRS:
        m=metrics[arm]['pair_metrics'][p]; q=recomputed[arm][p]
        metric_match.extend([abs(m['events']-q['events'])==0,abs(m['candidate_present_events']-q['candidate_present_events'])==0,abs(m['candidate_present_recall1']-q['recall'])<1e-12,abs(m['candidate_present_mrr']-q['mrr'])<1e-12,abs(m['dustbin_brier']-q['brier'])<1e-12])
    rounds.append({'round':'R2_INDEPENDENT_METRIC_REPLAY','checks':[
      check('event_blocks_have_all_four_pairs',all(set(recomputed[a])==set(PAIRS) for a in ARMS),{'pair_sets':{a:sorted(recomputed[a]) for a in ARMS}}),
      check('event_denominators_match_metric_files',all(metric_match),{'checked_fields':len(metric_match),'failed':metric_match.count(False)}),
      check('same_pair_event_denominators_between_arms',all(recomputed['B4'][p]['events']==recomputed['LCSCF_B4'][p]['events'] and recomputed['B4'][p]['candidate_present_events']==recomputed['LCSCF_B4'][p]['candidate_present_events'] for p in PAIRS),{'denominators':{p:{a:recomputed[a][p]['events'] for a in ARMS} for p in PAIRS},'present':{p:{a:recomputed[a][p]['candidate_present_events'] for a in ARMS} for p in PAIRS}})
    ]})
    b=metrics['B4']['pair_metrics']; m=metrics['LCSCF_B4']['pair_metrics']
    wins=sum(m[p]['candidate_present_recall1']>b[p]['candidate_present_recall1'] for p in PAIRS)
    ties=sum(m[p]['candidate_present_recall1']==b[p]['candidate_present_recall1'] for p in PAIRS)
    deltas={k:metrics['LCSCF_B4']['pair_macro'][k]-metrics['B4']['pair_macro'][k] for k in ('candidate_present_recall1','candidate_present_mrr','candidate_present_margin','dustbin_brier')}
    gate={'pair_recall1_strict_wins':wins,'pair_recall1_ties':ties,'pair_count':4,'deltas':deltas,'advance_rule':{'strict_wins_min':3,'macro_margin_delta_min':0.0,'macro_recall1_required':True},'advance':wins>=3 and deltas['candidate_present_margin']>=0.0 and deltas['candidate_present_recall1']>0}
    rounds.append({'round':'R3_PREDECLARED_METHOD_GATE','checks':[
      check('pair_win_count_recomputed',wins==2,{'wins':wins,'ties':ties,'per_pair_delta':{p:m[p]['candidate_present_recall1']-b[p]['candidate_present_recall1'] for p in PAIRS}}),
      check('macro_tradeoff_recomputed',deltas['candidate_present_mrr']<0 and deltas['candidate_present_margin']<0 and deltas['dustbin_brier']>0,{'deltas':deltas}),
      check('advance_gate_rejects_without_threshold_tuning',gate['advance'] is False,gate)
    ]})
    source_files=['sci_id/models/lcscf.py','sci_id/models/identity_pyramid.py','sci_id/data/episodes.py','run_n4_holdout.py','run_lcscf_gate.py','audit_lcscf_v4.py']
    payload={'status':'PASS_LCSCF_V4_METHOD_GATE_AUDIT_3X3','run_dir':str(RUN),'rounds':rounds,'gate':gate,'metrics_sha256':{a:hashlib.sha256((RUN/f'metrics_{a}.json').read_bytes()).hexdigest() for a in ARMS},'source_sha256':{s:hashlib.sha256((ROOT/s).read_bytes()).hexdigest() for s in source_files},'official_val_test_access':False,'disposition':'STOP_LCSCF_AFTER_CORRECTED_BLOCK1' if not gate['advance'] else 'ADVANCE_SECOND_BLOCK'}
    (ROOT/'LCSCF_GATE_AUDIT_V4.json').write_text(json.dumps(payload,indent=2)+'\n')
    print(json.dumps(payload,indent=2))
if __name__=='__main__': main()
