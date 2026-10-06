#!/usr/bin/env python3
import hashlib, json, math, re, shutil, subprocess
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
P24=ROOT/'p24_background_bootstrap'; P25=ROOT/'p25_pixel_localization'; P26=ROOT/'p26_mia_baseline'; P27=ROOT/'p27_strong_host_temporal'; P28=ROOT/'p28_causal_fgfa'; P29=ROOT/'p29_learned_alignment'; P30RAID=Path('/raid/datasets/chc_data/claude_try_MDMOT_p30_full_host_20261005')
OUT=Path(__file__).resolve().parent

def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def load(p): return json.loads(Path(p).read_text())
def check(name, ok, evidence, severity='PASS'):
 return {'name':name,'status':severity if ok else ('FAIL_'+severity if severity=='PASS' else severity),'ok':bool(ok),'evidence':evidence}
checks=[]
# Early-round scope and outcomes.
p24=load(P24/'DECISION.json'); checks.append(check('P24_scope', p24['status']=='KEEP_IMAGE_GEOMETRY_SUBSTRATE_NOT_NOVEL_METHOD', p24['status']))
p25=load(P25/'STATUS.json'); checks.append(check('P25_not_run', p25['training_started'] is False and p25['arrays_created'] is False, p25['status']))
p27=load(P27/'DECISION.json'); checks.append(check('P27_parity_isolated', p27['strict_detector_parity']['status']=='FAIL_SAMPLED_PARITY' and p27['strict_detector_parity']['strict_gate_relaxed'] is False and p27['mot_effect_established'] is False, {'status':p27['strict_detector_parity']['status'],'failed_images':p27['strict_detector_parity']['failed_images'],'mot_effect_established':p27['mot_effect_established']}))
# P28/P29 receipt artifact hashes.
for label, receipt in [('P28_CANONICAL',P28/'PREDICTION_RECEIPT_CANONICAL.json'),('P29_PREDICTION',P29/'PREDICTION_RECEIPT.json')]:
 d=load(receipt); missing=[]; mism=[]
 for p,h in d.get('artifacts',{}).items():
  q=Path(p)
  if not q.exists(): missing.append(p)
  elif sha(q)!=h: mism.append({'path':p,'expected':h,'actual':sha(q)})
 checks.append(check(label+'_ARTIFACT_HASHES', not missing and not mism, {'artifact_count':len(d.get('artifacts',{})),'missing':missing[:3],'mismatch':mism[:3]}))
# P29 exact numerical readout and documentation typo.
summ=load(P29/'detection_readout/SUMMARY.json'); arms=summ['arms']
vals={k:float(arms[k]['ap50']['pair_macro_ap50']['mean']) for k in ['native','current_residual','past_fixed_offsets','past_learned_offsets']}
# Accommodate SUMMARY schema if pair macro key nested differently.
if not vals or any(math.isnan(x) for x in vals.values()): raise RuntimeError('unexpected P29 SUMMARY schema')
dec=load(P29/'DECISION.json')
exact=(vals['past_learned_offsets']-vals['current_residual'])*100
reported=float(dec['comparisons']['learned_minus_current_pp'])
report_text=(P29/'REPORT.md').read_text()
checks.append(check('P29_metric_arithmetic', abs(exact-reported)<2e-7, {'computed_pp':exact,'decision_pp':reported,'absolute_difference':abs(exact-reported)}, severity='ROUNDING_ONLY'))
typo='.088467' in report_text
checks.append(check('P29_report_doc_only_typo', typo and abs(exact-.088467)>1e-6, {'computed_pp':exact,'report_token':'.088467'}, severity='DOC_ONLY'))
# P29 source static + synthetic defect/repair evidence.
old=(P29/'residual_deformable.py').read_text(); repaired=(ROOT/'p29_repair_20261005/residual_deformable.py').read_text()
old_bug='x = torch.arange(width, device=device, dtype=dtype)\n    flow_x ='
repair='x = torch.arange(width, device=device, dtype=dtype) * float(stride)\n    flow_x ='
checks.append(check('P29_stride_coordinate_defect_identified', old_bug in old and repair in repaired, {'original_sha256':sha(P29/'residual_deformable.py'),'repair_sha256':sha(ROOT/'p29_repair_20261005/residual_deformable.py')}))
# Run/record contract test without data or labels.
proc=subprocess.run(['/home/chenhc/mdmt_phase0_reproduction_20260829_202909/mdmt_env/bin/python',str(ROOT/'p29_repair_20261005/test_repair_contract.py')],capture_output=True,text=True)
checks.append(check('P29_repair_cpu_contract',proc.returncode==0 and 'PASS_P29_REPAIR_FLOW_REFERENCE_CONTRACT' in proc.stdout, {'returncode':proc.returncode,'stdout':proc.stdout.strip(),'stderr':proc.stderr[-500:]}))
# Direct original P26 host vs P30 native exact replay on pair 27.
direct=Path('/tmp/p30_impl_audit_direct/direct_native'); p30native=P30RAID/'host/native/json/p30_native'
replay=[]
for n in ['27-1.json','27-2.json']:
 a=direct/n; b=p30native/n
 replay.append({'file':n,'exists':a.exists() and b.exists(),'sha_direct':sha(a) if a.exists() else None,'sha_p30':sha(b) if b.exists() else None,'byte_equal':a.exists() and b.exists() and a.read_bytes()==b.read_bytes()})
checks.append(check('P30_native_host_direct_replay', all(x['byte_equal'] for x in replay), replay))
# P30 receipts and override structure.
for mode in ['native','past_fixed_offsets']:
 r=load(P30RAID/f'calibration/overrides/{mode}/RECEIPT.json')
 checks.append(check('P30_'+mode+'_receipt', r['status']=='COMPLETE_P30_DETECTOR_OVERRIDES' and r['frame_count']==3900 and r['gt_or_labels_read'] is False and r['native_capacity']=={'max_per_img':300,'nms_iou':.6,'nms_pre':2000,'score_thr':.05}, {'frame_count':r['frame_count'],'total_rows':r['total_rows'],'capacity':r['native_capacity']}))
 # all 10 seq dirs and contiguous ordinal arrays; sample all arrays finite Nx5
 root=P30RAID/f'calibration/overrides/{mode}'; seqs=sorted(p for p in root.iterdir() if p.is_dir()); bad=[]; count=0; rows=0
 for d in seqs:
  fs=sorted(d.glob('*.npy')); expected=[f'{i:06d}.npy' for i in range(len(fs))]
  if [x.name for x in fs]!=expected: bad.append({'seq':d.name,'reason':'noncontiguous'})
  for f in fs:
   a=np.load(f,allow_pickle=False); count+=1; rows+=len(a)
   if a.ndim!=2 or a.shape[1]!=5 or not np.isfinite(a).all() or len(a)>300: bad.append({'path':str(f),'shape':list(a.shape),'finite':bool(np.isfinite(a).all()),'max_rows':len(a)})
 checks.append(check('P30_'+mode+'_arrays', len(seqs)==10 and count==3900 and not bad and rows==r['total_rows'], {'seq_count':len(seqs),'array_count':count,'rows':rows,'receipt_rows':r['total_rows'],'bad':bad[:3]}))
# P30 host JSON structural and manifest hashes.
for mode, sub in [('native','p30_native'),('fixed','p30_fixed')]:
 jroot=P30RAID/f'host/{mode if mode=="native" else "fixed"}/json/{sub}'
 files=sorted(jroot.glob('*.json')); bad=[]; total_frames=0; total_rows=0
 for f in files:
  d=load(f); keys=list(d)
  if not keys or keys!=[f'frame={i}' for i in range(len(keys))]: bad.append({'path':str(f),'reason':'frame_keys'})
  for k,rows in d.items():
   ids=[]
   for row in rows:
    if not isinstance(row,list) or len(row)<5: bad.append({'path':str(f),'frame':k,'reason':'row_schema'}); continue
    if not all(isinstance(v,(int,float)) and math.isfinite(float(v)) for v in row[:5]): bad.append({'path':str(f),'frame':k,'reason':'nonfinite'})
    ids.append(row[0])
   if len(ids)!=len(set(ids)): bad.append({'path':str(f),'frame':k,'reason':'duplicate_id'})
   total_rows+=len(rows)
  total_frames+=len(d)
 checks.append(check('P30_'+mode+'_host_json', len(files)==10 and total_frames==3900, {'json_count':len(files),'frames':total_frames,'rows':total_rows,'duplicate_id_frames':bad[:3]}, severity='INHERITED_HOST_BEHAVIOR'))
manifest=load(P30RAID/'host/HOST_MANIFEST.json'); mm=[]
for group in ['valid_native_outputs','valid_fixed_outputs']:
 for p,meta in manifest[group].items():
  q=Path(p)
  if not q.exists() or sha(q)!=meta['sha256']: mm.append({'path':p,'exists':q.exists()})
checks.append(check('P30_host_manifest_hashes',not mm,{'mismatches':mm[:3]}))
# P30 runtime static guards and original semantics.
runtime=(ROOT/'p30_full_host/p30_override_runtime.py').read_text(); byte=(P26/'source/mmtrack/models/mot/byte_track.py').read_text()
checks.append(check('P30_override_guard', 'DET_OVERRIDE_DIR' in runtime and 'AutoAssign' in runtime and 'torch.zeros_like' in runtime, {'guard_tokens':all(x in runtime for x in ['DET_OVERRIDE_DIR','AutoAssign','torch.zeros_like'])}))
checks.append(check('P26_zero_label_semantics_mirrored', 'np.concatenate((det_results[0][0], det_results[0][1], det_results[0][2])' in byte and 'det_labels = torch.zeros_like(det_labels)' in byte and 'det_labels = torch.zeros_like(det_labels)' in runtime, {'source':'P26 ByteTrack and P30 override'}))
# decision: material implementation defect means prior P29/P30 negative is not accepted as corrected candidate verdict.
result={'schema':'implementation_audit_20261005_v1','status':'P29_IMPLEMENTATION_DEFECT_FOUND_P29_P30_NEGATIVE_NOT_FINAL','scope':['P24','P25','P26','P27','P28','P29','P30'], 'checks':checks,'summary':{'total':len(checks),'passed':sum(x['ok'] for x in checks),'failed':sum(not x['ok'] for x in checks)},'next_gate':'repair P29 stride-domain reference, rerun real GPU smoke, then retrain/re-predict/re-score before deciding P29; P30 host comparison must be repeated only after corrected predictions','official_val_or_test_read':False,'novelty_claim':False}
(OUT/'AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,sort_keys=True)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
