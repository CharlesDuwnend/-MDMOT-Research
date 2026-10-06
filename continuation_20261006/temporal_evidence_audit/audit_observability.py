#!/usr/bin/env python3
"""Audit whether a temporal evidence representation is observable and legal."""
import gzip,json,hashlib,collections,importlib.util
from pathlib import Path
STAGE=Path('/home/chenhc/mdmot_global_host_stage15_20260905')
STAGE1=Path('/home/chenhc/mdmot_strong_host_stage1_20260905')
EP=Path('/home/chenhc/mdmot_sci_id_20260906/data/train_episodes_v1')
FEAT=Path('/home/chenhc/cross_uav_query_mamba_20260830/features/train')
PAIRS=['23','25','27','28','29','30','32','39','42','44','45','50','51','53','54','58','63','64','65','66','69','70','74','76','78']
spec=importlib.util.spec_from_file_location('frozen_stage15_alignment',STAGE/'scripts/stage15_global_host.py')
FROZEN=importlib.util.module_from_spec(spec)
spec.loader.exec_module(FROZEN)

def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def idx_map(pair,uav):
 p=FEAT/f'{pair}-{uav}.jsonl'; out={}; rows=0
 if not p.is_file(): return out,{'path':str(p),'exists':False,'rows':0}
 features=collections.defaultdict(dict)
 for line in p.open():
  x=json.loads(line); rows+=1
  flat=int(x['feature_key'].rsplit(':',1)[1])
  # The frozen alignment function only tests feature presence and carries
  # its value; an integer row address avoids decoding unrelated embeddings.
  features[int(x['frame_id'])+1][flat]={'flattened_index':flat,'bbox':x['bbox'],'class':int(x['class_id']),'score':float(x['score']),'feature':int(x['feature_row'])}
 stream=STAGE1/f'streams/train/{pair}-{uav}.jsonl.gz'
 if not stream.is_file(): return out,{'path':str(p),'exists':True,'rows':rows,'sha256':sha(p),'stream_exists':False}
 stream_rows={}
 for line in gzip.open(stream,'rt'):
  row=json.loads(line); stream_rows[int(row['frame'])]=row
 by_frame,_,_,_,audit=FROZEN.derive_track_rows(stream_rows,features)
 for frame,tracked in by_frame.items():
  for row in tracked.values():
   if row['feature'] is not None:
    out[(frame,int(row['detector_index']))]=int(row['feature'])
 return out,{'path':str(p),'exists':True,'rows':rows,'sha256':sha(p),'stream':str(stream),'stream_rows':len(stream_rows),'stream_detections_matched':len(out),'alignment_audit':audit,'alignment_source_sha256':sha(STAGE/'scripts/stage15_global_host.py'),'npz_exists':(FEAT/f'{pair}-{uav}.npz').is_file()}

def main():
 counts=collections.Counter(); support_hist=collections.Counter(); pair_counts={}; missing=[]; gt_fields=collections.Counter(); index_meta={}
 all_support=0; all_evidence=0
 for pair in PAIRS:
  cache=STAGE/f'cache/train/{pair}.json.gz'; ep=EP/f'{pair}.jsonl.gz'
  if not cache.is_file() or not ep.is_file(): raise FileNotFoundError(pair)
  maps={};
  for uav in (1,2): maps[uav],index_meta[f'{pair}-{uav}']=idx_map(pair,uav)
  header=None; events=[]; 
  with gzip.open(cache,'rt') as f:
   header=json.loads(next(f))
  # Stage15 cache is one JSON object whose event list carries evidence;
  # train_episodes_v1 carries the separate train-only label contract.
  events=header.get('events', [])
  episode_rows=[json.loads(line) for line in gzip.open(ep,'rt')]
  pair_counts[pair]={'events':len(events),'episode_rows':len(episode_rows),'support_events':0,'support_entries':0,'support_rows_found':0,'support_rows_missing':0,'complete_candidates':0,'partial_candidates':0,'positive_events':0,'unknown_events':0}
  if header.get('gt_read') is not False or header.get('test_access') is not False: raise AssertionError(f'header leakage {pair}')
  for e in episode_rows:
   if 'target_train_gt_id' in e: gt_fields['episode_target_train_gt_id']+=1
   # labels are allowed for train-only supervision, but never model input.
   if e.get('unmapped_candidate_indices'): pair_counts[pair]['unknown_events']+=1
   if e.get('same_id_candidate_indices'): pair_counts[pair]['positive_events']+=1
   for k in e.keys():
    if 'gt_id' in k and k!='target_train_gt_id': gt_fields[k]+=1
  # Only the frozen Stage15 candidate stream contributes support evidence.
  for e in events:
   for c in e.get('candidates',[]):
    sf=int(c.get('support_frames',0)); support_hist[sf]+=1; all_support+=1
    evs=c.get('evidence',[])
    if sf != len(evs): counts['support_count_mismatch']+=1
    if sf>0: pair_counts[pair]['support_events']+=1
    pair_counts[pair]['support_entries']+=len(evs); all_evidence+=len(evs)
    candidate_missing=False
    for ev in evs:
     frame=int(ev['frame']);
     # Stage15 is 1-based frame numbering; feature index is zero-based.
     for key in ('target_detector_index','source_detector_index'):
      uav=int(e['target_uav']) if key.startswith('target') else int(e['source_uav'])
      row=maps[uav].get((frame,int(ev[key])))
      if row is None:
       candidate_missing=True
       pair_counts[pair]['support_rows_missing']+=1
       if len(missing)<20: missing.append({'pair':pair,'uav':uav,'frame':frame,'detector_index':int(ev[key]),'candidate_source_local_track_id':c.get('source_local_track_id')})
      else: pair_counts[pair]['support_rows_found']+=1
    if candidate_missing: pair_counts[pair]['partial_candidates']+=1
    else: pair_counts[pair]['complete_candidates']+=1
  # strict candidate feature alignment in all support rows
  if pair_counts[pair]['support_rows_missing']: counts['pair_missing_support_rows']+=1
  counts['pairs']+=1
 total_events=sum(v['events'] for v in pair_counts.values())
 total_candidates=all_support
 found=sum(v['support_rows_found'] for v in pair_counts.values()); missing_n=sum(v['support_rows_missing'] for v in pair_counts.values())
 forbidden_input=['target_train_gt_id','source_train_gt_id','target_local_track_id','source_local_track_id']
 primary=['23','25','27','28','29','30','32','39']
 payload={'status':'KEEP_TEMPORAL_EVIDENCE_FOR_NOVELTY_GATE_WITH_EXPLICIT_MISSING_MASK' if all(pair_counts[p]['support_rows_missing']==0 for p in primary) and counts['support_count_mismatch']==0 else 'HOLD_TEMPORAL_EVIDENCE_PRIMARY_OBSERVABILITY','candidate':'Cross-View Temporal Differential Co-Movement Field','split':'train_only','pairs':PAIRS,'counts':{'events':total_events,'episode_rows':sum(v['episode_rows'] for v in pair_counts.values()),'candidate_entries':total_candidates,'complete_candidates':sum(v['complete_candidates'] for v in pair_counts.values()),'partial_candidates':sum(v['partial_candidates'] for v in pair_counts.values()),'support_evidence_entries':all_evidence,'support_frames_histogram':dict(sorted(support_hist.items())),'support_rows_found':found,'support_rows_missing':missing_n,'support_row_coverage':found/(found+missing_n) if found+missing_n else 0.0,'support_count_mismatch':counts['support_count_mismatch'],'pairs_with_missing_rows':counts['pair_missing_support_rows'],'primary_block1_all_rows_present':all(pair_counts[p]['support_rows_missing']==0 for p in primary),'unknown_candidate_events':sum(v['unknown_events'] for v in pair_counts.values()),'positive_events':sum(v['positive_events'] for v in pair_counts.values())},'pair_counts':pair_counts,'feature_index':index_meta,'input_contract':{'legal': ['current target/source ROI features','support-frame ROI feature rows addressed by frozen detector/frame indices','support-frame masks and elapsed frame offsets','train-only same-ID/known-negative labels for loss only'],'forbidden':forbidden_input,'gt_in_inference_features':False,'tracker_ids_as_model_input':False,'future_frame_access':False,'missing_support_policy':'per-support validity mask; no zero fill, no row dropping, no future support'},'source_sha256':{'stage15_manifest':sha(STAGE/'manifest.json'),'train_episode_manifest':sha(EP/'manifest.json')},'missing_examples':missing,'novelty_boundary':'candidate uses cross-view temporal first-difference co-movement field; aggregate-distance reranking, tracklet mean, ordinary temporal memory and candidate-set thresholding are excluded and require collision audit.','official_val_test_access':False}
 (Path(__file__).parent/'TEMPORAL_EVIDENCE_OBSERVABILITY_AUDIT.json').write_text(json.dumps(payload,indent=2)+'\n')
 print(json.dumps(payload,indent=2))
if __name__=='__main__':main()
