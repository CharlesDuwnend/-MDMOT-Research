#!/usr/bin/env python3
"""FIT-only, XML-oracle temporal observation audit. No model or image imports."""
import csv
import gzip
import hashlib
import importlib.util
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
P0 = Path('/home/chenhc/mdmot_research_20261002/p0')
STAGE1 = Path('/home/chenhc/mdmot_strong_host_stage1_20260905')
FIT = ['23','25','28','29','39','44','45','51','53','63','66','69','70','74','78']
HORIZONS = (1, 4, 8)
TEMPORAL = ('first_annotation','stream_start_left_censored','segment_start','reappearance',
            'segment_start_frame','annotation_age','prior_detector_frame','prior_detector_age',
            'seen_prior_1','seen_prior_4','seen_prior_8','first_detector_in_segment',
            'first_detector_local_lifetime','prefix_miss_run_age','miss_category')
CAT = {0:'current_detected',1:'annotation_birth_no_prior_annotation',
       2:'reappearance_after_annotation_gap',3:'continuous_segment_never_detected_before',
       4:'continuous_prior_detection_older_than_8',5:'continuous_prior_detection_within_8'}
INPUTS = {}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda:handle.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def bound(path, expected=None, kind='input'):
    path = Path(path).resolve()
    value = sha(path)
    if expected is not None and value != expected:
        raise AssertionError(f'Hash mismatch: {path}')
    INPUTS[str(path)] = {'sha256':value,'bytes':path.stat().st_size,'kind':kind}
    return path


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False)+'\n')


def read_json(path, expected=None):
    return json.loads(bound(path,expected).read_text())


def unique(keys, what):
    seen = set()
    for key in keys:
        if key in seen:
            raise AssertionError('Duplicate '+what+': '+str(key))
        seen.add(key)
    return seen


def temporal_features(a):
    """Online prefix pass. State changes after emitting history at the current row."""
    n = len(a['frame'])
    out = {k:np.zeros(n,np.int32) for k in TEMPORAL}
    states = {}
    for i in range(n):
        f, rid, cls = (int(a[k][i]) for k in ('frame','raw_id','class'))
        key = (rid,cls)
        current = bool(a['current_detected'][i])
        prior = states.get(key)
        first = prior is None
        if prior is not None:
            assert f > prior['frame']
        newseg = first or f != prior['frame']+1
        start = f if newseg else prior['start']
        last = -1 if newseg else prior['last_seen']
        age = -1 if last < 0 else f-last
        vals = dict(first_annotation=first,stream_start_left_censored=first and f==1,
                    segment_start=newseg,reappearance=newseg and not first,
                    segment_start_frame=start,annotation_age=f-start+1,
                    prior_detector_frame=last,prior_detector_age=age,
                    first_detector_in_segment=current and last<0,
                    first_detector_local_lifetime=current and (first or not prior['ever_seen']),
                    prefix_miss_run_age=0 if current else (1 if newseg else prior['miss_age']+1))
        vals.update({f'seen_prior_{h}':0<age<=h for h in HORIZONS})
        vals['miss_category'] = (0 if current else 1 if first else 2 if newseg else
                                3 if last<0 else 5 if age<=8 else 4)
        for k,v in vals.items():
            out[k][i] = v
        states[key] = dict(frame=f,start=start,last_seen=f if current else last,
                           ever_seen=current or (False if first else prior['ever_seen']),
                           miss_age=vals['prefix_miss_run_age'])
    return out


def synthetic_tests():
    tests = []
    def data(frames,det,rids=None,classes=None):
        return dict(frame=np.array(frames),raw_id=np.array(rids or [1]*len(frames)),
                    **{'class':np.array(classes or [0]*len(frames))},current_detected=np.array(det))
    a=data(list(range(1,12)),[1]+[0]*8+[1,0]); b=temporal_features(a)
    assert not b['seen_prior_8'][0] and b['seen_prior_1'][1]
    assert b['seen_prior_4'][4] and not b['seen_prior_4'][5]
    assert b['seen_prior_8'][8] and not b['seen_prior_8'][9]
    assert b['seen_prior_1'][10] and b['prior_detector_frame'][9]==1
    tests.append('strictly_past_current_excluded_fixed_1_4_8_boundaries')
    for end in range(1,12):
        p=temporal_features({k:v[:end] for k,v in a.items()})
        for k in TEMPORAL:assert np.array_equal(p[k],b[k][:end]),k
    changed={k:v.copy() for k,v in a.items()};changed['current_detected'][7:]=1
    c=temporal_features(changed)
    for k in TEMPORAL:assert np.array_equal(c[k][:7],b[k][:7]),k
    tests.append('future_truncation_and_suffix_perturbation_invariance')
    g=temporal_features(data([1,2,5,6,7],[1,0,0,0,1]))
    assert list(g['first_annotation'])==[1,0,0,0,0]
    assert list(g['reappearance'])==[0,0,1,0,0]
    assert g['seen_prior_8'][2]==0 and g['prior_detector_frame'][2]==-1
    assert g['prefix_miss_run_age'][2]==1 and g['first_detector_in_segment'][4]==1
    assert g['first_detector_local_lifetime'][4]==0
    tests.append('annotation_gap_reset_birth_reappearance_first_detection_distinct')
    c=temporal_features(data([1,1,2,2],[1,0,0,0],[1,2,1,2]))
    assert c['seen_prior_8'][2]==1 and c['seen_prior_8'][3]==0
    c=temporal_features(data([1,2],[1,0],[1,1],[0,1]))
    assert c['seen_prior_8'][1]==0
    tests.append('identity_and_class_state_isolation')
    for label in ('detector row','matched GT'):
        try:unique([(1,2,0),(1,2,0)],label)
        except AssertionError:pass
        else:raise AssertionError('Duplicate rejection failed')
    tests.append('one_to_one_duplicate_detector_and_matched_gt_rejection')
    return {'status':'PASS','tests':tests}


def summaries(a):
    det=a['current_detected'].astype(bool);miss=~det
    result={'gt_observations':len(det),'current_detected':int(det.sum()),'current_missing':int(miss.sum())}
    result['current_recall']=float(det.mean()) if len(det) else None
    for h in HORIZONS:
        result[f'missing_prior_{h}']=int(np.count_nonzero(miss & a[f'seen_prior_{h}'].astype(bool)))
    result['annotation_births']=int(a['first_annotation'].sum())
    result['stream_start_left_censored_births']=int(a['stream_start_left_censored'].sum())
    result['annotation_reappearances']=int(a['reappearance'].sum())
    result['first_detector_in_segment']=int(a['first_detector_in_segment'].sum())
    result['miss_categories']={name:int(np.count_nonzero(a['miss_category']==k)) for k,name in CAT.items() if k}
    assert sum(result['miss_categories'].values())==result['current_missing']
    for field,values in [('occluded',[0,1]),('minside_bin',[0,1,2]),('class',[0,1,2])]:
        result[field]={}
        for v in values:
            mask=a[field]==v
            result[field][str(v)]={'gt_observations':int(mask.sum()),'current_missing':int(np.count_nonzero(mask&miss)),
                **{f'missing_prior_{h}':int(np.count_nonzero(mask&miss&a[f'seen_prior_{h}'].astype(bool))) for h in HORIZONS}}
    result['occlusion_x_minside']={}
    for oc in (0,1):
        for sz in (0,1,2):
            mask=(a['occluded']==oc)&(a['minside_bin']==sz)
            result['occlusion_x_minside'][f'{oc}:{sz}']={'gt_observations':int(mask.sum()),
                'current_missing':int(np.count_nonzero(mask&miss)),
                **{f'missing_prior_{h}':int(np.count_nonzero(mask&miss&a[f'seen_prior_{h}'].astype(bool))) for h in HORIZONS}}
    return result


def merge_counts(items):
    merged={}
    for item in items:
        for key,value in item.items():
            if key=='current_recall':continue
            if isinstance(value,dict):merged[key]=merge_counts([merged.get(key,{}),value])
            else:merged[key]=merged.get(key,0)+value
    if 'current_detected' in merged:
        merged['current_recall']=merged['current_detected']/merged['gt_observations'] if merged['gt_observations'] else None
    return merged


def miss_runs(a, nframes):
    groups=defaultdict(list)
    for i,(rid,cls) in enumerate(zip(a['raw_id'],a['class'])):groups[(int(rid),int(cls))].append(i)
    columns=('raw_id','class','start_frame','end_frame','length','start_row','end_row',
             'preceded_by_current_detection','end_reason','prior_8_at_start')
    result={k:[] for k in columns}
    def add(indices,previous,nextrow):
        first,last=indices[0],indices[-1]
        end=int(a['frame'][last])
        # 1 recovered immediately, 2 annotation gap/reappearance, 3 last annotation/stream end.
        reason=3 if nextrow is None else (1 if int(a['frame'][nextrow])==end+1 else 2)
        vals=(int(a['raw_id'][first]),int(a['class'][first]),int(a['frame'][first]),end,len(indices),first,last,
              int(previous is not None and int(a['frame'][previous])==int(a['frame'][first])-1 and a['current_detected'][previous]),
              reason,int(a['seen_prior_8'][first]))
        for k,v in zip(columns,vals):result[k].append(v)
    for ids in groups.values():
        run=[];prior=None;before=None
        for i in ids:
            if run and (a['current_detected'][i] or int(a['frame'][i])!=int(a['frame'][run[-1]])+1):
                add(run,before,i);run=[]
            if not a['current_detected'][i]:
                if not run:before=prior
                run.append(i)
            prior=i
        if run:add(run,before,None)
    out={k:np.array(v,np.int32) for k,v in result.items()}
    assert int(out['length'].sum())==int((~a['current_detected']).sum())
    lengths=out['length']
    stats={'runs':len(lengths),'missing_observation_mass':int(lengths.sum()),
           'max_length':int(lengths.max()) if len(lengths) else 0,
           'exact_length_histogram':{str(k):int(v) for k,v in sorted(Counter(lengths.tolist()).items())},
           'end_reason_counts':{str(k):int(v) for k,v in sorted(Counter(out['end_reason'].tolist()).items())}}
    return out,stats


def sequence(pair,view,label_record,seq_record,helper):
    seq=f'{pair}-{view}'
    assert pair in FIT and label_record['diagnostic_role']=='fit'
    lp=bound(label_record['output_path'],label_record['output_sha256'],'fit_detector_labels')
    with np.load(lp,allow_pickle=False) as z:labels={k:z[k] for k in z.files}
    ap=bound(P0/'artifacts/aligned_indices'/f'{seq}.npz',label_record['aligned_sha256'],'fit_aligned_rows')
    with np.load(ap,allow_pickle=False) as z:
        for k in ('frame','detector_index','detector_class'):
            assert np.array_equal(z[k],labels[k]),(seq,k)
    keys=unique(zip(labels['frame'].tolist(),labels['detector_index'].tolist()),'detector row')
    matched=labels['matched_gt']
    assert np.all(labels['matched_iou'][matched]>=.5)
    assert np.all(labels['gt_class'][matched]==labels['detector_class'][matched])
    assert np.all(labels['raw_local_identity'][matched]>=0)
    indices=np.flatnonzero(matched)
    mkeys=[tuple(int(labels[k][i]) for k in ('frame','raw_local_identity','detector_class')) for i in indices]
    unique(mkeys,'matched GT')
    lookup=dict(zip(mkeys,indices.tolist()))
    sp=bound(STAGE1/'streams/train'/f'{seq}.jsonl.gz',label_record['stream_sha256'],'fit_frozen_stream')
    offset=0;frames=[];minimum_score=1.;detector_sources=set()
    with gzip.open(sp,'rt') as handle:
        for line in handle:
            x=json.loads(line);f=int(x['frame']);frames.append(f)
            assert x['split']=='train' and str(x['pair_id'])==pair and str(x['uav_id'])==view
            assert x['gt_read'] is False and x['gt_initialized'] is False
            assert x['thresholds']['detector_score_min']==.1
            detector_sources.add(x['detector_source'])
            ds=sorted(x['detector_detections'],key=lambda d:int(d['detector_index']))
            end=offset+len(ds)
            assert np.array_equal(labels['frame'][offset:end],np.full(len(ds),f))
            assert np.array_equal(labels['detector_index'][offset:end],[d['detector_index'] for d in ds])
            assert np.array_equal(labels['detector_class'][offset:end],[d['class'] for d in ds])
            if ds:
                minimum_score=min(minimum_score,min(d['score'] for d in ds))
                assert minimum_score>=.1
            offset=end
    assert offset==len(labels['frame'])==len(keys)
    assert frames==list(range(1,len(frames)+1))
    assert len(frames)==label_record['counts']['frames']==seq_record['image_names_count']
    xp=bound(seq_record['xml_path'],seq_record['xml_sha256'],'fit_raw_XML_offline_only')
    assert str(xp)==label_record['xml_path'] and seq_record['xml_sha256']==label_record['xml_sha256']
    gframes,tracks,helperpath=helper.read_xml(pair,view,seq_record['xml_sha256'])
    assert helperpath.resolve()==xp
    assert all(1<=f<=len(frames) for f in gframes),'GT outside full stream range'
    cols=('frame','raw_id','class','occluded','bbox_x1','bbox_y1','bbox_x2','bbox_y2',
          'minimum_side_px','minside_bin','current_detected','matched_detector_index','matched_label_row','matched_iou')
    rows={k:[] for k in cols};consumed=set();unsupported=0
    for f in frames:
        for gt in gframes.get(f,[]):
            if gt['class'] not in (0,1,2):unsupported+=1;continue
            key=(f,gt['raw_id'],gt['class']);row=lookup.get(key)
            current=row is not None
            if current:consumed.add(key)
            box=gt['bbox'];size=min(box[2]-box[0],box[3]-box[1]);sizebin=0 if size<16 else 1 if size<32 else 2
            vals=(f,gt['raw_id'],gt['class'],int(bool(gt['occluded'])),*box,size,sizebin,current,
                  int(labels['detector_index'][row]) if current else -1,row if current else -1,
                  float(labels['matched_iou'][row]) if current else 0.)
            for k,v in zip(cols,vals):rows[k].append(v)
    assert consumed==set(lookup),'Matched detector not consumed by GT observation'
    a={k:np.array(v,np.float32 if k.startswith('bbox_') or k in ('minimum_side_px','matched_iou') else
                  np.bool_ if k=='current_detected' else np.int32) for k,v in rows.items()}
    a.update(temporal_features(a))
    output=HERE/'rows'/f'{seq}.npz';np.savez_compressed(output,**a)
    runs,rstats=miss_runs(a,len(frames));np.savez_compressed(HERE/'miss_runs'/f'{seq}.npz',**runs)
    counts=summaries(a)
    expected_boxes=sum(seq_record['valid_visible_boxes_by_label'].get(k,0) for k in ('pedestrian','person','bicycle','car'))
    assert counts['gt_observations']==expected_boxes
    assert counts['current_detected']==label_record['counts']['matched_gt_rows']
    info={'sequence':seq,'pair':pair,'view':view,'frame_start':1,'frame_end':len(frames),
          'stream_frames':len(frames),'detector_rows':len(labels['frame']),'minimum_stream_detection_score':minimum_score,
          'detector_sources':sorted(detector_sources),'unsupported_gt_observations_excluded':unsupported,
          'counts':counts,'miss_run_summary_RETROSPECTIVE_ONLY':rstats}
    print(json.dumps({'sequence':seq,'frames':len(frames),'counts':{k:counts[k] for k in ('gt_observations','current_missing','missing_prior_8')}}),flush=True)
    return a,tracks,info


def cross_view(pair,a,b,ta,tb):
    shared={rid for rid in ta.keys()&tb.keys() if ta[rid]['class']>=0 and ta[rid]['label']==tb[rid]['label']}
    other={(int(f),int(rid),int(c)):i for i,(f,rid,c) in enumerate(zip(b['frame'],b['raw_id'],b['class']))}
    columns=('frame','raw_id','class','view1_row','view2_row','current_v1','current_v2','both_missing',
             'occluded_v1','occluded_v2','minside_bin_v1','minside_bin_v2',
             'prior_1_either','prior_1_both','prior_4_either','prior_4_both','prior_8_either','prior_8_both')
    rows={k:[] for k in columns}
    for i in range(len(a['frame'])):
        rid=int(a['raw_id'][i]);key=(int(a['frame'][i]),rid,int(a['class'][i]))
        j=other.get(key)
        if rid not in shared or j is None:continue
        da,db=bool(a['current_detected'][i]),bool(b['current_detected'][j])
        vals=(*key,i,j,da,db,not (da or db),int(a['occluded'][i]),int(b['occluded'][j]),
              int(a['minside_bin'][i]),int(b['minside_bin'][j]))
        past=[]
        for h in HORIZONS:
            x,y=bool(a[f'seen_prior_{h}'][i]),bool(b[f'seen_prior_{h}'][j]);past.extend([x or y,x and y])
        for k,v in zip(columns,(*vals,*past)):rows[k].append(v)
    c={k:np.array(v,np.int32) for k,v in rows.items()};np.savez_compressed(HERE/'cross_view'/f'{pair}.npz',**c)
    both=c['both_missing'].astype(bool)
    return {'coannotated_same_label_equal_id_observations':len(both),'both_current_missing':int(both.sum()),
            **{f'both_missing_prior_{h}_{how}':int(np.count_nonzero(both & c[f'prior_{h}_{how}'].astype(bool))) for h in HORIZONS for how in ('either','both')}}


def write_csv(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def report(summary):
    t=summary['totals'];miss=t['current_missing'];gt=t['gt_observations']
    lines=['# P27 FIT temporal observation evidence','',
        '**PASS diagnostic execution; old-stream Oracle evidence only.**',
        f"15 FIT pairs, 30 sequences, {summary['stream_frames']:,} full stream frames; {gt:,} supported annotated observations.",
        f"CURRENT detected {t['current_detected']:,} ({100*t['current_recall']:.3f}%); missing {miss:,} ({100*miss/gt:.3f}%).",'',
        'The detector streams apply score >= 0.1. This is not exact P26 detector evidence (native post-NMS default 0.05 reported separately). No inference, training, images, GPU, or non-FIT data used in the computation.', '',
        '| Strictly prior horizon | Missing with same-GT past evidence | % of missing | % of all GT |',
        '|---|---:|---:|---:|']
    for h in HORIZONS:
        n=t[f'missing_prior_{h}'];lines.append(f'| {h} | {n:,} | {100*n/miss:.3f}% | {100*n/gt:.3f}% |')
    lines.extend(['','These counts describe available historical observations with Oracle local GT correspondence. They do not establish recoverable detections, feature alignment quality, or expected FGFA/MDA gains. Annotation gaps reset every history.','',
                  '| Miss category | Observations | % of missing |','|---|---:|---:|'])
    for key,value in t['miss_categories'].items():lines.append(f'| {key} | {value:,} | {100*value/miss:.3f}% |')
    lines.extend(['','| XML occlusion | GT observations | Missing | Missing with prior 8 |','|---|---:|---:|---:|'])
    for key,v in t['occluded'].items():lines.append(f"| {key} | {v['gt_observations']:,} | {v['current_missing']:,} | {v['missing_prior_8']:,} |")
    lines.extend(['','| Minimum GT side | GT observations | Missing | Missing with prior 8 |','|---|---:|---:|---:|'])
    for key,v in t['minside_bin'].items():lines.append(f"| {['<16','16 to <32','>=32'][int(key)]} | {v['gt_observations']:,} | {v['current_missing']:,} | {v['missing_prior_8']:,} |")
    lines.extend(['','| Pair | GT observations | Missing | Missing prior 1 | Missing prior 4 | Missing prior 8 |','|---|---:|---:|---:|---:|---:|'])
    for pair,v in summary['per_pair'].items():
        lines.append(f"| {pair} | {v['gt_observations']:,} | {v['current_missing']:,} | {v['missing_prior_1']:,} | {v['missing_prior_4']:,} | {v['missing_prior_8']:,} |")
    x=summary['cross_view_totals_ANNOTATION_CONVENTION_ONLY']
    lines.extend(['',f"Cross-view appendix: {x['coannotated_same_label_equal_id_observations']:,} coannotated conventional matches; both-current-missing {x['both_current_missing']:,}.",
                  'Equal raw ID and normalized class label define the cohort; physical identity and hardware synchronization are unverified.', '',
                  '| Prior horizon | Either-view evidence among both missing | Both-view evidence among both missing |','|---|---:|---:|'])
    for h in HORIZONS:lines.append(f"| {h} | {x[f'both_missing_prior_{h}_either']:,} | {x[f'both_missing_prior_{h}_both']:,} |")
    lengths=Counter()
    for s in summary['sequences']:lengths.update({int(k):v for k,v in s['miss_run_summary_RETROSPECTIVE_ONLY']['exact_length_histogram'].items()})
    lines.extend(['',f"Retrospective miss runs: {sum(lengths.values()):,}; longest {max(lengths,default=0)} frames. Full lengths/endpoints are summary-only and never enter causal history flags.",
                  '','| Full miss-run length | Runs | Missing observation mass |','|---|---:|---:|'])
    for lo,hi,label in [(1,1,'1'),(2,4,'2 to 4'),(5,8,'5 to 8'),(9,16,'9 to 16'),(17,1000000,'>=17')]:
        lines.append(f'| {label} | {sum(v for k,v in lengths.items() if lo<=k<=hi):,} | {sum(k*v for k,v in lengths.items() if lo<=k<=hi):,} |')
    lines.extend(['','`rows/*.npz` retains every supported GT observation. `cases.csv` deterministically retains first misses per sequence/category/occlusion/size cell, not selected success imagery. `miss_runs/*.npz` and `cross_view/*.npz` are separate explicitly marked diagnostics.',
                  '`summary.json` retains class, occlusion × size, pair, sequence, and exact run-length counts. `VERIFY.json` records synthetic and independent saved-row verification; `RECEIPT.json` binds all inputs, code, and outputs.',''])
    (HERE/'REPORT.md').write_text('\n'.join(lines))


def main():
    start=time.monotonic()
    test=synthetic_tests();write_json(HERE/'INVARIANT_TESTS.json',test)
    bound(HERE/'PROTOCOL.md',kind='protocol');bound(__file__,kind='code')
    bound(P0/'data/LABEL_INTERFACE.md',kind='label_contract')
    split=read_json(P0/'data/ASSOCIATION_DIAGNOSTIC_SPLIT.json')
    assert split['assignments']['fit']==FIT
    manifest=read_json(P0/'data/DETECTION_LABELS_MANIFEST.json')
    assert manifest['status']=='PASS_OFFLINE_PSEUDO_SUPERVISION_ADAPTER'
    assert sha(P0/'data/ASSOCIATION_DIAGNOSTIC_SPLIT.json')==manifest['diagnostic_split_sha256']
    seqs=read_json(P0/'data/sequences_train.json')
    records={r['sequence']:r for r in manifest['sequences'] if r['pair'] in FIT}
    seqrecords={(r['pair'],r['view']):r for r in seqs if r['pair'] in FIT}
    helperpath=bound(P0/'data/build_detection_labels.py',manifest['script_sha256'],'read_only_XML_helper')
    bound(STAGE1/'scripts/export_gt_free_bytetrack.py',kind='detector_threshold_source')
    spec=importlib.util.spec_from_file_location('p0_detection_labels_readonly',helperpath)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    for directory in ('rows','miss_runs','cross_view'):(HERE/directory).mkdir(exist_ok=True)
    results=[];pair_counts={};cross_counts={};case_rows=[]
    for pair in FIT:
        aa,ta,ra=sequence(pair,'1',records[pair+'-1'],seqrecords[(pair,'1')],helper)
        bb,tb,rb=sequence(pair,'2',records[pair+'-2'],seqrecords[(pair,'2')],helper)
        results.extend([ra,rb]);pair_counts[pair]=merge_counts([ra['counts'],rb['counts']])
        cross_counts[pair]=cross_view(pair,aa,bb,ta,tb)
        for a,seq in [(aa,pair+'-1'),(bb,pair+'-2')]:
            seen=Counter()
            for i in np.flatnonzero(~a['current_detected']):
                key=(int(a['miss_category'][i]),int(a['occluded'][i]),int(a['minside_bin'][i]))
                if seen[key]>=2:continue
                seen[key]+=1
                case_rows.append({'sequence':seq,'frame':int(a['frame'][i]),'raw_id':int(a['raw_id'][i]),
                    'class':int(a['class'][i]),'category':CAT[key[0]],'occluded':key[1],'minside_bin':key[2],
                    'minimum_side_px':float(a['minimum_side_px'][i]),'segment_start_frame':int(a['segment_start_frame'][i]),
                    'prior_detector_frame':int(a['prior_detector_frame'][i]),'prior_detector_age':int(a['prior_detector_age'][i]),
                    **{f'seen_prior_{h}':int(a[f'seen_prior_{h}'][i]) for h in HORIZONS}})
    totals=merge_counts([r['counts'] for r in results])
    assert totals['current_detected']==manifest['by_diagnostic_role']['fit']['matched_gt_rows']
    summary={'schema':'p27_fit_temporal_observation_v1','status':'PASS_FIT_ORACLE_OBSERVATION_DIAGNOSTIC',
        'pairs':FIT,'sequence_count':len(results),'stream_frames':sum(r['stream_frames'] for r in results),
        'detector_rows':sum(r['detector_rows'] for r in results),'totals':totals,'per_pair':pair_counts,'sequences':results,
        'cross_view_per_pair_ANNOTATION_CONVENTION_ONLY':cross_counts,
        'cross_view_totals_ANNOTATION_CONVENTION_ONLY':merge_counts(list(cross_counts.values())),
        'raw_data_scope':'FIT15 only','calibration_dev_official_val_test_data_used_in_computation':False,
        'gpu_used':False,'images_read':False,'training_or_inference_executed':False,
        'exact_P26_detector_parity_claimed':False,'detector_score_min':.1,'horizons':list(HORIZONS),
        'elapsed_seconds':round(time.monotonic()-start,3)}
    assert summary['stream_frames']==manifest['by_diagnostic_role']['fit']['frames']
    write_json(HERE/'summary.json',summary)
    write_json(HERE/'INPUTS.json',INPUTS)
    flatkeys=('gt_observations','current_detected','current_missing','current_recall','missing_prior_1','missing_prior_4','missing_prior_8','annotation_births','stream_start_left_censored_births','annotation_reappearances')
    write_csv(HERE/'per_pair.csv',[{'pair':p,**{k:v[k] for k in flatkeys}} for p,v in pair_counts.items()])
    write_csv(HERE/'per_sequence.csv',[{'sequence':r['sequence'],'stream_frames':r['stream_frames'],'detector_rows':r['detector_rows'],**{k:r['counts'][k] for k in flatkeys}} for r in results])
    write_csv(HERE/'cases.csv',case_rows)
    report(summary)
    print(json.dumps({'status':summary['status'],'totals':totals,'elapsed_seconds':summary['elapsed_seconds']}),flush=True)


if __name__=='__main__':main()
