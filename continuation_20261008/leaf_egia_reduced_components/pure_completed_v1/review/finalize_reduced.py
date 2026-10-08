"""Verify complete actual runs and report every declared component reduction."""
import csv,datetime,hashlib,json,pickle,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from run_mechanism import sha,check_frozen,audit_runtime,write
PARENT=Path('/home/chenhc/leaf_egia_src_off_gate_20261008_v1')
LABELS={'R00_full':'完整 EGIA','R01_no_foreground_head':'去前景分类头',
        'R02_no_coverage_head':'去覆盖分类头','R03_no_source_cues':'去来源一致性分支'}
def read(p):return json.loads(Path(p).read_text())
def parameter_counts(bundle,summary):
    source=bundle['source_model'];a=source.anchor
    models=[a.foreground_model,a.coverage_model,source.geometry_model,source.appearance_model]
    counts=[]
    for m in models:
        if m is None:continue
        lr=m.steps[-1][1] if hasattr(m,'steps') else m
        counts.append(int(lr.coef_.size+lr.intercept_.size))
    return len(counts),sum(counts)
def main():
    manifest=read(ROOT/'manifest.json');check_frozen(manifest)
    for n in ['smoke.exit','full.exit','pipeline.exit']:
        assert (ROOT/'artifacts'/n).read_text().strip()=='0'
    all_runs=read(ROOT/'artifacts/grid_receipt.json')
    smoke=read(ROOT/'artifacts/smoke_grid_receipt.json')
    assert all_runs['status']=='PASS_ALL_4_SCORED_REDUCED_CELLS_COMPLETE'
    assert all_runs['manifest_sha256']==smoke['manifest_sha256']==sha(ROOT/'manifest.json')
    assert read(ROOT/'artifacts/full_progress.json')['completed']==4
    expected=manifest['sequence_frames'];inputs=read(manifest['input_reference_receipt'])['input_sha256']
    reference=read(manifest['full_reference_metrics']);records=[];source_audits={};runtime_hashes={}
    summary=pickle.loads((ROOT/'models/summary_frozen.pkl').read_bytes())
    for arm in manifest['arms']:
        folder=ROOT/'YOLOX_outputs'/arm['name'];r=read(folder/'receipt.json');metrics=read(folder/'tracking_metrics.json')
        assert r['status']=='PASS_SCORED_ARM_COMPLETE'
        assert r['sequence_count']==17 and r['frames']==6635
        assert r['metrics_sha256']==sha(folder/'tracking_metrics.json')
        assert r['bundle_sha256']==sha(ROOT/arm['bundle'])==arm['bundle_sha256']
        assert r['input_sha256']==inputs
        assert r['gt_guard']['status']=='PASS_NO_GT_READ' and r['gt_guard']['blocked_gt_attempts']==0
        assert metrics['groundtruth_files_sha256']==reference['groundtruth_files_sha256']
        assert metrics['metrics']['OVERALL']['num_objects']==229506
        assert r['same_process_gpu_verification']['cuda_uuid']==arm['gpu_uuid']
        assert arm['gpu_physical_index'] in [0,1,3]
        files={p.stem:p for p in (folder/'track_res').glob('*.txt')}
        assert set(files)==set(expected)
        reports={s:read(str(p)+'.runtime.json') for s,p in files.items()}
        audit_runtime(reports,arm,False)
        for seq,p in files.items():
            assert sha(p)==r['result_sha256'][seq]==metrics['result_files_sha256'][seq]
            assert reports[seq]['frames_seen']==expected[seq]
            assert reports[seq]['input_stream_sha256']==inputs[seq]
            runtime_hashes[arm['name']+'/'+seq]=sha(str(p)+'.runtime.json')
        source_audits[arm['name']]={s:v['structural_audit'] for s,v in reports.items()}
        if arm['name']=='R00_full':
            assert r['result_sha256']==reference['result_files_sha256']
            assert r['historical_prediction_comparison']['matched']==17
            assert r['metrics']==reference['metrics']['OVERALL']
        m=metrics['metrics']['OVERALL']
        assert abs(m['mota']-(1.-(m['num_false_positives']+m['num_misses']+m['num_switches'])/m['num_objects']))<1e-12
        b=pickle.loads((ROOT/arm['bundle']).read_bytes());n,params=parameter_counts(b,summary)
        records.append(dict(name=arm['name'],scheme=LABELS[arm['name']],MOTA=100*m['mota'],IDF1=100*m['idf1'],
            FP=m['num_false_positives'],FN=m['num_misses'],IDs=m['num_switches'],FM=m['num_fragmentations'],
            classifiers=n,classifier_coefficients_and_intercepts=params,
            births=r['runtime_totals']['final_birth_activations'],metrics_sha256=r['metrics_sha256'],
            gpu_physical_index=arm['gpu_physical_index'],gpu_uuid=arm['gpu_uuid']))
    # Native is an explicitly reused completed score under exactly this protocol.
    native_folder=PARENT/'YOLOX_outputs/S0_G0_disabled'
    native_r=read(native_folder/'receipt.json');native_m=read(native_folder/'tracking_metrics.json')
    assert native_r['status']=='PASS_SCORED_ARM_COMPLETE' and native_r['input_sha256']==inputs
    assert native_m['groundtruth_files_sha256']==reference['groundtruth_files_sha256']
    assert native_r['metrics_sha256']==sha(native_folder/'tracking_metrics.json')
    native_reports={p.stem:read(str(p)+'.runtime.json') for p in (native_folder/'track_res').glob('*.txt')}
    assert set(native_reports)==set(expected)
    for s,r in native_reports.items():
        assert not r['use_egia'] and not r['src_state_enabled'] and r['birth_class_confidence_threshold']==0.
        assert r['frames_seen']==expected[s] and r['input_stream_sha256']==inputs[s]
        assert r['counts']['native_reference_seams_verified']==r['counts']['association_seams']
        assert sha(native_folder/'track_res'/(s+'.txt'))==native_m['result_files_sha256'][s]
    m=native_m['metrics']['OVERALL']
    native=dict(name='native_completed_reference',scheme='基线（SRC、EGIA、gate 均关）',MOTA=100*m['mota'],IDF1=100*m['idf1'],
        FP=m['num_false_positives'],FN=m['num_misses'],IDs=m['num_switches'],FM=m['num_fragmentations'],
        classifiers=0,classifier_coefficients_and_intercepts=0,births=native_r['runtime_totals']['final_birth_activations'],
        metrics_sha256=native_r['metrics_sha256'],gpu_physical_index=None,gpu_uuid=None)
    baseline=records[0]
    for r in records:
        r.update(delta_MOTA_vs_full=r['MOTA']-baseline['MOTA'],delta_IDF1_vs_full=r['IDF1']-baseline['IDF1'],
            delta_FP_vs_full=r['FP']-baseline['FP'],delta_FN_vs_full=r['FN']-baseline['FN'],delta_IDs_vs_full=r['IDs']-baseline['IDs'])
    native.update(delta_MOTA_vs_full=native['MOTA']-baseline['MOTA'],delta_IDF1_vs_full=native['IDF1']-baseline['IDF1'],
        delta_FP_vs_full=native['FP']-baseline['FP'],delta_FN_vs_full=native['FN']-baseline['FN'],delta_IDs_vs_full=native['IDs']-baseline['IDs'])
    out=ROOT/'review/results';out.mkdir(exist_ok=False)
    rows=[native]+records
    with (out/'tracking_results.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(rows)
    lines=['# EGIA 减部件结果','',
        '本轮四个 EGIA 方案统一关闭 SRC、类别置信度出生 gate 和 H2 fusion；不加载旧 H2 分类器。检测分数出生条件仍为 0.6，保留 selective margin=0.2，其余当前 source 权重与输入固定。','',
        '|方案|EGIA 分类器数|MOTA ↑|IDF1 ↑|FP ↓|FN ↓|IDs ↓|FM ↓|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:lines.append('|{scheme}|{classifiers}|{MOTA:.4f}|{IDF1:.4f}|{FP}|{FN}|{IDs}|{FM}|'.format(**r))
    lines+=['','|减少部件方案|ΔMOTA 相对完整|ΔIDF1 相对完整|ΔFP|ΔFN|ΔIDs|',
        '|---|---:|---:|---:|---:|---:|']
    for r in records[1:]:lines.append('|{scheme}|{delta_MOTA_vs_full:+.4f}|{delta_IDF1_vs_full:+.4f}|{delta_FP_vs_full:+d}|{delta_FN_vs_full:+d}|{delta_IDs_vs_full:+d}|'.format(**r))
    lines+=['','分类器数包含 foreground、coverage、geometry、appearance；未把未使用的 SRC 头和未加载的 H2 summary 头计入。分类器系数与截距数分别为完整 26、去 FG 15、去 coverage 15、去来源分支 22；不包含检测/ReID、归一化统计、先验和单独 gamma。',
        '去 FG 是删除目标/杂波判别，f=1；去 coverage 是删除覆盖判别，c=0，source 的 E 类也取消，因此 E-only 来源一致性不再具备决策作用。这是功能删除，不是同一三分类任务下的架构公平性或参数效率结论。selective 保留，fusion 关闭。',
        '四个 EGIA 格均为本轮真实完成的 17 序列/6635 帧；Native 行复用此前同一协议已完成的基线，预测哈希、全部 detector/ReID 输入、GT 和门控设置已重新核对。本轮完整 EGIA 的 17 份预测与历史结果逐字节一致。',
        '不重训、不以 test-dev 选阈值，不隐藏退化、零效应或改善结果。MOTA/IDF1 差值单位为百分点；未做统计显著性声明。']
    (out/'RESULTS_ZH.md').write_text('\n'.join(lines)+'\n')
    write(out/'STRUCTURAL_RUNTIME_AUDIT.json',dict(status='PASS_ACTUAL_COMPONENT_DELETION_AND_RETAINED_POLICY',
        per_sequence=source_audits,runtime_files_sha256=runtime_hashes))
    write(out/'FINAL_AUDIT.json',dict(status='COMPLETE_ALL_FOUR_PREREGISTERED_ACTUAL_COMPONENT_REDUCTIONS',
        completed_utc=datetime.datetime.utcnow().isoformat()+'Z',manifest_sha256=sha(ROOT/'manifest.json'),
        new_full_runs=4,new_full_prediction_files=68,new_full_frames=26540,reference_prediction_byte_parity=17,
        SRC=False,birth_class_gate=0.,fusion=False,selective=True,new_pure_arm_legacy_classifier_loaded=False,GT_objects=229506,
        original_GT_and_ordered_detector_ReID_parity=True,testdev_selection=False,
        native_score_reused=True,native_metrics_sha256=sha(native_folder/'tracking_metrics.json'),
        current_paper_modified=False,all_results=rows,
        actual_native_association_seams=sum(read(ROOT/'YOLOX_outputs'/a['name']/'receipt.json')['runtime_totals']['native_reference_seams_verified'] for a in manifest['arms'])))
    write(out/'RESULT_FILES_SHA256.json',{str(p.relative_to(out)):sha(p) for p in out.iterdir() if p.is_file()})
    check_frozen(manifest)
    print(json.dumps(dict(status='COMPLETE',table=str(out/'RESULTS_ZH.md'),results=rows),ensure_ascii=False))
if __name__=='__main__':main()
