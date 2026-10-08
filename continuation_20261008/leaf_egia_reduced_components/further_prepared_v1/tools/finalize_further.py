"""Independent completion audit and combined table, no partial-result claims."""
import csv,datetime,importlib.util,json,pickle,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from run_mechanism import sha,check_frozen,audit_runtime,write
PARENT=Path('/home/chenhc/leaf_egia_reduced_pure_20261008_v1')
OLD_NATIVE=Path('/home/chenhc/leaf_egia_src_off_gate_20261008_v1/YOLOX_outputs/S0_G0_disabled')

def read(p):return json.loads(Path(p).read_text())
def model_counts(b):
    s=b['source_model'];a=s.anchor;n=params=0
    for m in [a.foreground_model,a.coverage_model,s.geometry_model,s.appearance_model]:
        if m is None:continue
        lr=m.steps[-1][1] if hasattr(m,'steps') else m
        n+=1;params+=int(lr.coef_.size+lr.intercept_.size)
    return n,params

def main():
    m=read(ROOT/'manifest.json');check_frozen(m)
    for name in ['smoke.exit','full.exit','pipeline.exit']:
        assert (ROOT/'artifacts'/name).read_text().strip()=='0'
    smoke=read(ROOT/'artifacts/smoke_grid_receipt.json')
    grid=read(ROOT/'artifacts/grid_receipt.json')
    assert smoke['status']=='PASS_PURE_REFERENCE_AND_THREE_NEW_SMOKES'
    assert grid['status']=='PASS_ALL_THREE_FURTHER_PURE_ARMS_SCORED'
    assert grid['manifest_sha256']==smoke['manifest_sha256']==sha(ROOT/'manifest.json')
    assert read(ROOT/'artifacts/full_progress.json')['completed']==3
    reference=read(m['full_reference_metrics'])
    inputs=read(m['input_reference_receipt'])['input_sha256']
    expected=m['sequence_frames'];rows=[];runtime_hashes={};totals={}
    labels={'R00_full':'完整 pure EGIA','R01_no_foreground_head':'去前景头（保留来源修正）',
        'R02_no_coverage_head':'去覆盖头（E 修正同时失去作用）','R03_no_source_cues':'前景头 + 覆盖头 + selective',
        'A04_coverage_only_selective':'仅覆盖头 + selective',
        'A05_two_heads_argmax':'前景头 + 覆盖头，直接 argmax',
        'A06_full_argmax':'完整判别及来源修正，直接 argmax',
        'native_completed_reference':'Native（SRC、EGIA、0.7 gate 均关）'}
    old_spec=importlib.util.spec_from_file_location('old_grid_audit',PARENT/'tools/grid_audit.py')
    old_audit=importlib.util.module_from_spec(old_spec);old_spec.loader.exec_module(old_audit)
    parent_arms=read(PARENT/'manifest.json')['arms']
    all_specs=[('native_completed_reference',OLD_NATIVE,None,False)]
    all_specs += [(a['name'],PARENT/'YOLOX_outputs'/a['name'],a,False) for a in parent_arms]
    all_specs += [(a['name'],ROOT/'YOLOX_outputs'/a['name'],a,True) for a in m['arms']]
    for name,folder,arm,new in all_specs:
        receipt=read(folder/'receipt.json');metric=read(folder/'tracking_metrics.json')
        assert receipt['status']=='PASS_SCORED_ARM_COMPLETE'
        assert receipt['sequence_count']==17 and receipt['frames']==6635
        assert receipt['input_sha256']==inputs
        assert receipt['metrics_sha256']==sha(folder/'tracking_metrics.json')
        assert metric['groundtruth_files_sha256']==reference['groundtruth_files_sha256']
        assert metric['metrics']['OVERALL']['num_objects']==229506
        files={p.stem:p for p in (folder/'track_res').glob('*.txt')}
        assert set(files)==set(expected)
        reports={s:read(str(p)+'.runtime.json') for s,p in files.items()}
        if arm:
            (audit_runtime if new else old_audit.audit_grid_runtime)(reports,arm,False)
            bundle_root=ROOT if new else PARENT
            assert receipt['bundle_sha256']==sha(bundle_root/arm['bundle'])==arm['bundle_sha256']
            b=pickle.loads((bundle_root/arm['bundle']).read_bytes())
            n,params=model_counts(b)
            selective=b['selective_birth_policy'];coherence=b['source_model'].weight!=0.
            assert receipt['same_process_gpu_verification']['cuda_uuid']==arm['gpu_uuid']
            assert arm['gpu_physical_index'] in [0,1,3]
        else:
            n=params=0;selective=False;coherence=False
        for s,p in files.items():
            r=reports[s];c=r['counts']
            assert r['frames_seen']==expected[s] and r['input_stream_sha256']==inputs[s]
            assert not r['src_state_enabled'] and r['birth_class_confidence_threshold']==0.
            assert c['native_reference_seams_verified']==c['association_seams']
            assert sha(p)==receipt['result_sha256'][s]==metric['result_files_sha256'][s]
            if new:
                assert r['selective_birth_policy']==arm['selective']
                assert r['pure_v5_policy'] and not r['legacy_egia_loaded']
                runtime_hashes[name+'/'+s]=sha(str(p)+'.runtime.json')
            if arm is None:assert not r['use_egia']
        if new:
            assert receipt['gt_guard']['status']=='PASS_NO_GT_READ' and receipt['gt_guard']['blocked_gt_attempts']==0
            assert not any(c.get('egia_fusion_events',0) for c in [r['counts'] for r in reports.values()])
        v=metric['metrics']['OVERALL']
        assert abs(v['mota']-(1.-(v['num_false_positives']+v['num_misses']+v['num_switches'])/v['num_objects']))<1e-12
        rows.append(dict(name=name,scheme=labels[name],new_full_inference=new,
            classifiers=n,classifier_coefficients_and_intercepts=params,selective=selective,
            coherence_branch_present=coherence,MOTA=100*v['mota'],IDF1=100*v['idf1'],
            FP=v['num_false_positives'],FN=v['num_misses'],IDs=v['num_switches'],FM=v['num_fragmentations'],
            births=receipt['runtime_totals']['final_birth_activations'],metrics_sha256=receipt['metrics_sha256']))
        if new:totals[name]=receipt['runtime_totals']
    by={r['name']:r for r in rows}
    contrasts={}
    pairs=[('foreground_without_coherence','R03_no_source_cues','A04_coverage_only_selective'),
        ('coherence_with_selective','R00_full','R03_no_source_cues'),
        ('coherence_without_selective','A06_full_argmax','A05_two_heads_argmax'),
        ('selective_with_coherence','R00_full','A06_full_argmax'),
        ('selective_without_coherence','R03_no_source_cues','A05_two_heads_argmax')]
    for label,a,b in pairs:
        contrasts[label]={k:by[a][k]-by[b][k] for k in ['MOTA','IDF1','FP','FN','IDs']}
    interaction={k:contrasts['coherence_with_selective'][k]-contrasts['coherence_without_selective'][k]
        for k in ['MOTA','IDF1','FP','FN','IDs']}
    out=ROOT/'review/results';out.mkdir(exist_ok=False)
    with (out/'tracking_results.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    lines=['# Native + pure EGIA 内部消融（续）','',
        '所有方案均以 Native 为底座，SRC 与额外类别置信度 gate=0.7 关闭。三个新 EGIA 方案均不加载 H2 分类器、关闭 fusion。检测分数出生阈值保持 0.6，其余权重与输入不变。','',
        '|方案|分类器数|MOTA ↑|IDF1 ↑|FP ↓|FN ↓|IDs ↓|FM ↓|结果来源|',
        '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in rows:
        lines.append('|{scheme}|{classifiers}|{MOTA:.4f}|{IDF1:.4f}|{FP}|{FN}|{IDs}|{FM}|{origin}|'.format(
            **r,origin='本轮真实完整推理' if r['new_full_inference'] else '复用已完成并重新核验的结果'))
    lines+=['','|来源一致性|selective|MOTA ↑|IDF1 ↑|IDs ↓|','|---|---|---:|---:|---:|']
    for name,c,s in [('R00_full','开','开'),('R03_no_source_cues','关','开'),
                     ('A06_full_argmax','开','关'),('A05_two_heads_argmax','关','关')]:
        lines.append('|{}|{}|{:.4f}|{:.4f}|{}|'.format(c,s,by[name]['MOTA'],by[name]['IDF1'],by[name]['IDs']))
    lines+=['','两个主分类头在这个 2×2 中均保留。selective 关闭使用当前实现原有直接 argmax 分支，不另设阈值；selective 开启沿用 margin=0.2。',
        '仅覆盖头方案物理删除 foreground、geometry、appearance 三个模型，f=1，C 类概率严格为零；保留的 coverage 模型参数不变。',
        '去任一来源 cue 时一致性项数学上为零，属于去整个来源一致性分支的等价控制；没有将解析等价当成新增 GPU 推理分数。',
        '本轮三个新增方案各完成 17 序列/6635 帧；已有五行显式复用，逐一验证预测哈希、原始 GT、输入哈希及 native association seams。无重训、无 test-dev 选配置、无显著性声明，未改论文或作图。']
    (out/'RESULTS_ZH.md').write_text('\n'.join(lines)+'\n')
    write(out/'CONTRASTS.json',dict(contrasts=contrasts,coherence_x_selective_interaction=interaction,
        units='percentage points for MOTA/IDF1; counts otherwise',formal_significance_claim=False))
    write(out/'RUNTIME_AUDIT.json',dict(status='PASS_ALL_THREE_ACTUAL_REDUCTIONS_AND_POLICIES',
        runtime_files_sha256=runtime_hashes,runtime_totals=totals))
    write(out/'FINAL_AUDIT.json',dict(status='COMPLETE_THREE_NEW_NATIVE_PLUS_PURE_EGIA_INTERNAL_ABLATIONS',
        completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        manifest_sha256=sha(ROOT/'manifest.json'),new_full_runs=3,new_full_frames=19905,new_full_prediction_files=51,
        reused_completed_rows=5,SRC=False,birth_class_gate=0.,fusion=False,GT_objects=229506,
        new_legacy_classifier_loaded=False,ordered_detector_ReID_parity=True,original_GT_parity=True,
        all_results=rows,contrasts=contrasts,interaction=interaction,testdev_selection=False,
        current_paper_modified=False,actual_native_association_seams=sum(t['native_reference_seams_verified'] for t in totals.values())))
    write(out/'RESULT_FILES_SHA256.json',{p.name:sha(p) for p in out.iterdir() if p.is_file()})
    check_frozen(m)
    print(json.dumps(dict(status='COMPLETE',new_full_runs=3,table=str(out/'RESULTS_ZH.md'),results=rows),ensure_ascii=False))

if __name__=='__main__':main()
