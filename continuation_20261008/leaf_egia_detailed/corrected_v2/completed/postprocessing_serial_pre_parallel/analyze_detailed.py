"""Read-only verification and tables after every declared full run completes.

Postprocessing is separate from the frozen inference dependency closure. All
zero/negative controls remain in the report; no fitted or operating point is
selected here.
"""
import csv
import datetime
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES']=''
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from run_mechanism import audit_runtime, check_frozen, sha, write
from verify_detailed import verify_input_ledger

LABELS={
 'D00_full_reference':'完整 SRC + EGIA',
 'D01_EGIA_bypass':'关闭 EGIA，保留完整 SRC',
 'D02_no_foreground_evidence':'前景判断替换为训练先验',
 'D03_no_coverage_evidence':'覆盖判断替换为训练先验',
 'D04_no_learned_context':'两路上下文判断均替换为训练先验',
 'D05_no_source_coherence':'关闭源一致性项',
 'D06_broken_source_pairing':'打乱几何—外观源配对',
 'D07_no_geometry_cue':'几何线索置为中性',
 'D08_no_appearance_cue':'外观线索置为中性'}
FIELDS=['mota','idf1','num_false_positives','num_misses','num_switches','num_fragmentations']


def read(p):
    return json.loads(Path(p).read_text())


def verify_all():
    manifest=read(ROOT/'manifest.json')
    assert len(manifest['arms'])==9
    assert {a['name'] for a in manifest['arms']}==set(LABELS)
    assert len(manifest['sequence_frames'])==17 and sum(manifest['sequence_frames'].values())==6635
    assert not manifest['testdev_parameter_selection']
    check_frozen(manifest)
    for name in ['pipeline.exit','launcher.exit','smoke_launcher.exit']:
        assert (ROOT/'artifacts'/name).read_text().strip()=='0', 'INCOMPLETE_TERMINAL_STATE'
    assert not (ROOT/'artifacts/live.lock').exists() and not (ROOT/'artifacts/stop.json').exists()
    gate=read(ROOT/'artifacts/INPUT_IDENTITY_AUDIT_GATE.json')
    assert gate['sealed'] and gate['status']=='RESOLVED_DETECTION_METADATA_IDENTITY'
    assert gate['manifest_sha256']==sha(ROOT/'manifest.json')
    for p,h in gate['evidence_sha256'].items():
        assert sha(p)==h
    global_receipt=read(ROOT/'artifacts/receipt.json')
    assert global_receipt['status']=='PASS_PREDECLARED_ONLINE_MECHANISM_ADDENDUM_COMPLETE'
    assert global_receipt['manifest_sha256']==sha(ROOT/'manifest.json')
    assert len(global_receipt['arms'])==9 and not global_receipt['testdev_selection']
    reference=read(manifest['full_reference_metrics'])
    expected=manifest['sequence_frames']
    assert reference['metrics']['OVERALL']['num_objects']==229506
    for sequence,h in reference['groundtruth_files_sha256'].items():
        assert sha(Path(manifest['gt_root'])/(sequence+'.txt'))==h
    all_records=[]
    baseline_ledgers=None
    actual_pred_files=0
    total_reference_seams=0
    for arm,sealed_record in zip(manifest['arms'],global_receipt['arms']):
        assert arm['runtime_arm']=='fc_full' and arm['mask'] and arm['penalty']
        assert arm['fusion'] and arm['selective']
        output=ROOT/'YOLOX_outputs'/arm['name']
        r=read(output/'receipt.json')
        assert r==sealed_record and r['status']=='PASS_SCORED_ARM_COMPLETE'
        assert r['frames']==6635 and r['sequence_count']==17
        assert r['bundle_sha256']==arm['bundle_sha256']==sha(ROOT/arm['bundle'])
        device=r['same_process_gpu_verification']
        assert device['status']=='VERIFIED_PHYSICAL_GPU'
        assert device['cuda_uuid']==manifest['gpu_uuid']==device['selected']['uuid']
        assert device['selected']['index']==1 and device['selected']['memory_mib']==40960
        assert device['cuda_total_memory_bytes']==42298834944
        assert 'A100' in device['cuda_name'] and device['cuda_visible_devices']==manifest['gpu_uuid']
        assert r['gt_guard']==read(ROOT/'artifacts'/(arm['name']+'_gt_guard.json'))
        assert r['gt_guard']['blocked_gt_attempts']==0 and r['gt_guard']['checked_file_open_calls']>0
        assert r['gt_guard']['status']=='PASS_NO_GT_READ'
        assert r['gt_guard']['guarded_annotation_root']==manifest['gt_root']
        metric_path=output/'tracking_metrics.json'
        metric=read(metric_path)
        assert sha(metric_path)==r['metrics_sha256']
        assert metric['metrics']['OVERALL']==r['metrics']
        assert metric['groundtruth_files_sha256']==reference['groundtruth_files_sha256']
        assert metric['eval_split']=='test-dev' and metric['expected_sequences']==17
        assert metric['sequence_names']==reference['sequence_names']
        assert metric['results_folder']==str(output/'track_res')
        assert metric['groundtruth_folder']==manifest['gt_root']
        assert set(metric['result_files_sha256'])==set(expected)
        files={p.stem:p for p in (output/'track_res').glob('*.txt')}
        assert set(files)==set(expected)
        reports={}
        ledgers={}
        output_rows=0
        shuffle={}
        for sequence,frames in expected.items():
            p=files[sequence]
            assert sha(p)==metric['result_files_sha256'][sequence]==r['result_sha256'][sequence]
            actual_pred_files+=1
            rows=p.read_text().splitlines()
            assert all(1<=int(line.split(',')[0])<=frames for line in rows)
            output_rows+=len(rows)
            reports[sequence]=read(str(p)+'.runtime.json')
            ledgers[sequence]=verify_input_ledger(reports[sequence],sequence,frames)
            assert reports[sequence]['input_stream_sha256']==r['input_sha256'][sequence]
            assert metric['metrics'][sequence]['num_objects']==reference['metrics'][sequence]['num_objects']
            for key,value in reports[sequence].get('operator_audit',{}).get('pair_shuffle',{}).items():
                shuffle[key]=shuffle.get(key,0)+value
        audit_runtime(reports,arm)
        if baseline_ledgers is None:
            baseline_ledgers=ledgers
            assert metric['result_files_sha256']==reference['result_files_sha256']
            assert r['input_sha256']==read(Path(manifest['full_reference_metrics']).parent/'receipt.json')['input_sha256']
        else:
            assert ledgers==baseline_ledgers,'DETECTOR_REID_LEDGER_CHANGED'
        totals={k:sum(report['counts'].get(k,0) for report in reports.values())
                for k in set(k for report in reports.values() for k in report['counts'])}
        assert totals==r['runtime_totals']
        assert totals['association_seams']==totals['corrected_reference_seams_verified']==19905
        total_reference_seams+=totals['corrected_reference_seams_verified']
        if arm['mode']=='pair_shuffle':
            assert shuffle['eligible_multi_source_events']>0 and shuffle['changed_pairing_events']>0
            assert shuffle['changed_source_assignments']>0
        evaluation_log=(ROOT/'logs'/(arm['name']+'_evaluation.log')).read_text()
        assert all(sequence in evaluation_log for sequence in expected) and 'OVERALL' in evaluation_log
        m=r['metrics']
        assert abs(m['mota']-(1.-(m['num_false_positives']+m['num_misses']+m['num_switches'])/229506.))<1e-12
        all_records.append(dict(name=arm['name'],label=LABELS[arm['name']],mode=arm['mode'],
            metrics=m,metrics_sha256=sha(metric_path),result_sha256=r['result_sha256'],
            bundle_sha256=r['bundle_sha256'],raw_output_rows=output_rows,
            native_candidate_gate_calls=totals['native_birth_gate_calls'],
            native_gate_admitted=totals['native_birth_gate_admitted'],
            final_births=totals['belief_initializations'],
            source_probability_calls=totals.get('egia_source_probability_calls',0),
            EGIA_native_vetoes=totals.get('birth_rejected_vs_local_h2_admitted',0),
            EGIA_NEW_certificates=totals.get('birth_admitted_vs_local_h2_rejected',0),
            src_updates=totals['belief_updates'],src_seams=totals['association_seams'],
            pairing_intervention=shuffle,counts=totals))
    records={r['name']:r for r in all_records}
    degeneration=['D05_no_source_coherence','D07_no_geometry_cue','D08_no_appearance_cue']
    assert all(records[n]['result_sha256']==records[degeneration[0]]['result_sha256'] for n in degeneration)
    assert all(records[n]['metrics']==records[degeneration[0]]['metrics'] for n in degeneration)
    audit=dict(status='PASS_ALL_NINE_FULL_SRC_DETAILED_EGIA_RUNS_AND_SOURCE_INPUT_SCORE_AUDIT',
        created_utc=datetime.datetime.utcnow().isoformat()+'Z',manifest_sha256=sha(ROOT/'manifest.json'),
        global_receipt_sha256=sha(ROOT/'artifacts/receipt.json'),actual_predictions_rehashed=actual_pred_files,
        frames_per_arm=6635,sequences_per_arm=17,verified_original_reference_seams=total_reference_seams,
        full_corrected_C11_prediction_parity=True,all_ordered_detector_ReID_ledgers_equal=True,
        GT_denominator=229506,all_SRC_parts_active_all_cells=True,fit_only_neutral_priors=True,
        no_refit_or_testdev_selection=True,neutral_cue_zero_gamma_predictions_identical=True,
        postprocessing_source_sha256=sha(__file__),formal_accuracy_scope='VisDrone2019 original test-dev only')
    return manifest,all_records,audit


def tables(manifest,records,audit):
    destination=ROOT/'artifacts/completed_detailed'
    destination.mkdir(exist_ok=False)
    base=records[0]['metrics']
    deltas=[]
    for r in records[1:]:
        delta={k:base[k]-r['metrics'][k] for k in FIELDS}
        delta['mota']*=100.;delta['idf1']*=100.
        deltas.append(dict(full_minus_control=r['name'],delta=delta))
    write(destination/'FINAL_AUDIT.json',audit)
    write(destination/'RESULTS.json',dict(records=records,full_minus_control=deltas,
        degeneration_controls=['D05_no_source_coherence','D07_no_geometry_cue','D08_no_appearance_cue'],
        interpretation='Same frozen head evidence controls; neutral-cue cells are not independent gains.'))
    with (destination/'tracking_results.csv').open('x',newline='') as stream:
        writer=csv.writer(stream)
        writer.writerow(['cell','label','MOTA_percent','IDF1_percent','FP','FN','IDs','FM','recall_percent'])
        for r in records:
            m=r['metrics']
            writer.writerow([r['name'],r['label'],100*m['mota'],100*m['idf1'],
                m['num_false_positives'],m['num_misses'],m['num_switches'],m['num_fragmentations'],100*m['recall']])
    with (destination/'admission_statistics.csv').open('x',newline='') as stream:
        writer=csv.writer(stream)
        cols=['name','native_candidate_gate_calls','native_gate_admitted','final_births',
              'EGIA_native_vetoes','source_probability_calls','raw_output_rows','src_updates']
        writer.writerow(cols)
        writer.writerows([[r[k] for k in cols] for r in records])
    lines=['# 完整 SRC 下的 EGIA 细致消融','',
        '固定 SRC 全部组件、F00 头权重、fusion=0.40、selective margin=0.20 和所有运行设置。17 个完整序列，每组 6,635 帧；未重训或挑选测试集配置。', '',
        '| 对照 | MOTA ↑ | IDF1 ↑ | FP ↓ | FN ↓ | IDs ↓ | FM ↓ |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in records:
        m=r['metrics']
        lines.append('| %s | %.4f | %.4f | %d | %d | %d | %d |' %
            (r['label'],100*m['mota'],100*m['idf1'],m['num_false_positives'],m['num_misses'],m['num_switches'],m['num_fragmentations']))
    lines+=['','前景/覆盖中性化采用原始 fit39 二分类加权先验，只替换指定一路输出；其他头和 SRC 参数保持固定。',
            '几何或外观线索中性化都会使源一致性项退化为零。这两行与关闭一致性项逐预测文件相同，作为算子退化验证，论文中不应当作三项独立增益。',
            '打乱源配对保留两路线索的边缘分布和上下文输入；实际置换计数见准入统计/JSON。不同闭环对照的候选池可以改变，不能直接比较其候选事件比例来宣称因果分类精度。',
            'IDs 应与 IDF1、FN 和准入数量一起解释。原始输出行数包括评测过滤的目标，不是 GT 匹配后的精确率分母。', '',
            '| 对照 | 原生允许候选 | 最终出生目标 | EGIA 否决 | 输出行数 | Recall (%) |',
            '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in records:
        lines.append('| %s | %d | %d | %d | %d | %.4f |' %
            (r['label'],r['native_gate_admitted'],r['final_births'],r['EGIA_native_vetoes'],r['raw_output_rows'],100*r['metrics']['recall']))
    lines+=['','新完整配置与前次修正元数据后的 C11：17 个预测文件及有序 detector/ReID 输入全部一致。9 组均完成 GT-free 推理与原始 GT 评分，所有真实关联 seam 同时通过不可变整段参考方法核对。',
            '所有负向或零差异对照保留。当前结果不包含跨数据集推广、显著性或重训架构优越性的结论。']
    (destination/'RESULTS_FOR_AUTHOR_ZH.md').write_text('\n'.join(lines)+'\n')
    # Retain the prior four policy cells only after the fresh full prediction
    # gate. This supplements the new frozen-evidence controls, not new runs.
    parent=Path(manifest['parent_root'])
    policy_names=['E00_fusion_off_no_selective','E10_fusion_no_selective',
                  'E01_fusion_off_selective','C11_full_reference']
    policy=[]
    for name in policy_names:
        p=parent/'YOLOX_outputs'/name
        receipt=read(p/'receipt.json');metric=read(p/'tracking_metrics.json')
        assert receipt['status']=='PASS_SCORED_ARM_COMPLETE'
        assert sha(p/'tracking_metrics.json')==receipt['metrics_sha256']
        assert receipt['input_sha256']==read(parent/'YOLOX_outputs/C11_full_reference/receipt.json')['input_sha256']
        assert metric['groundtruth_files_sha256']==read(manifest['full_reference_metrics'])['groundtruth_files_sha256']
        for sequence,h in receipt['result_sha256'].items():
            assert sha(p/'track_res'/(sequence+'.txt'))==h
        policy.append(dict(name=name,metrics=receipt['metrics'],
            metrics_sha256=receipt['metrics_sha256'],role='previous completed corrected policy factorial; source/input parity verified'))
    write(destination/'RETAINED_POLICY_FACTORIAL.json',dict(rows=policy,fresh_full_parity_gate=True,new_runs=False))
    print(json.dumps(dict(status=audit['status'],results_folder=str(destination),rows=len(records)),ensure_ascii=False))


if __name__=='__main__':
    manifest,records,audit=verify_all()
    tables(manifest,records,audit)
