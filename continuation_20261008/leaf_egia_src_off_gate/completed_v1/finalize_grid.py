"""Verify every declared scored cell, report both gate states and archive."""
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import traceback

ROOT=Path(__file__).resolve().parents[1]
REPO=Path('/home/chenhc/claude_try_MDMOT')
PREFIX='continuation_20261008/leaf_egia_src_off_gate/completed_v1/'
STAGE='leaf_egia_src_off_gate_23_cells_complete_20261008_v1'
sys.path.insert(0,str(ROOT/'tools'))
from run_mechanism import check_frozen,sha,write,audit_runtime
from verify_detailed import verify_input_ledger

LABELS={'disabled':'关闭 EGIA','full':'完整 EGIA',
 'no_foreground':'前景概率固定为训练先验','no_coverage':'覆盖概率固定为训练先验',
 'no_context':'两路概率均固定为训练先验','no_coherence':'关闭来源一致性项',
 'pair_shuffle':'打乱几何—外观来源配对','fusion_off':'关闭 fusion',
 'selective_off':'关闭 selective','fusion_selective_off':'fusion/selective 均关闭'}
METRICS=['mota','idf1','num_false_positives','num_misses','num_switches','num_fragmentations']

def read(p):return json.loads(Path(p).read_text())
def git(*args,env=None):return subprocess.check_output(['git',*args],cwd=REPO,env=env)

def audit():
    m=read(ROOT/'manifest.json');check_frozen(m)
    assert len(m['arms'])==23 and len(m['sequence_frames'])==17 and sum(m['sequence_frames'].values())==6635
    assert not m['testdev_parameter_selection']
    for n in ['smoke.exit','full.exit','pipeline.exit']:assert (ROOT/'artifacts'/n).read_text().strip()=='0'
    assert not (ROOT/'artifacts/grid_stop.json').exists()
    for n in ['full.live.lock','smoke.live.lock']:assert not (ROOT/'artifacts'/n).exists()
    receipt=read(ROOT/'artifacts/grid_receipt.json')
    assert receipt['status']=='PASS_ALL_23_SCORED_GRID_CELLS_COMPLETE'
    assert receipt['manifest_sha256']==sha(ROOT/'manifest.json') and not receipt['testdev_selection']
    assert len(receipt['arms'])==23
    reference=read(m['full_reference_metrics'])
    parent_receipt=read(m['input_reference_receipt'])
    for s,h in reference['groundtruth_files_sha256'].items():assert sha(Path(m['gt_root'])/(s+'.txt'))==h
    records=[];first_ledger=None;pred_count=0;native_seams=0;SRC_seams=0
    for a,sealed in zip(m['arms'],receipt['arms']):
        output=ROOT/'YOLOX_outputs'/a['name'];r=read(output/'receipt.json');score=read(output/'tracking_metrics.json')
        assert sealed==r and r['name']==a['name'] and r['status']=='PASS_SCORED_ARM_COMPLETE'
        assert r['frames']==6635 and r['sequence_count']==17
        assert sha(ROOT/a['bundle'])==a['bundle_sha256']==r['bundle_sha256']
        assert sha(output/'tracking_metrics.json')==r['metrics_sha256']
        assert score['metrics']['OVERALL']==r['metrics'] and score['groundtruth_files_sha256']==reference['groundtruth_files_sha256']
        assert score['eval_split']=='test-dev' and score['expected_sequences']==17
        assert score['results_folder']==str(output/'track_res') and score['groundtruth_folder']==m['gt_root']
        assert r['input_sha256']==parent_receipt['input_sha256']
        hardware=r['same_process_gpu_verification']
        assert hardware['status']=='VERIFIED_PHYSICAL_GPU'
        assert hardware['cuda_uuid']==a['gpu_uuid']==hardware['selected']['uuid']
        assert hardware['selected']['index']==a['gpu_physical_index'] and a['gpu_physical_index'] in [0,1,3]
        assert hardware['selected']['memory_mib']==40960 and hardware['cuda_total_memory_bytes']==42298834944
        assert hardware['cuda_visible_devices']==a['gpu_uuid'] and 'A100' in hardware['cuda_name']
        guard=read(ROOT/'artifacts'/(a['name']+'_gt_guard.json'))
        assert guard==r['gt_guard'] and guard['status']=='PASS_NO_GT_READ'
        assert guard['blocked_gt_attempts']==0 and guard['checked_file_open_calls']>0
        assert guard['guarded_annotation_root']==m['gt_root']
        reports={};ledgers={};rows=0
        assert {p.stem for p in (output/'track_res').glob('*.txt')}==set(m['sequence_frames'])
        for sequence,frames in m['sequence_frames'].items():
            p=output/'track_res'/(sequence+'.txt')
            assert sha(p)==r['result_sha256'][sequence]==score['result_files_sha256'][sequence]
            pred_count+=1;rows+=len(p.read_text().splitlines())
            reports[sequence]=read(str(p)+'.runtime.json')
            ledgers[sequence]=verify_input_ledger(reports[sequence],sequence,frames)
        audit_runtime(reports,a)
        if first_ledger is None:first_ledger=ledgers
        else:assert ledgers==first_ledger,'ORDERED_INPUTS_CHANGED'
        totals={k:sum(v['counts'].get(k,0) for v in reports.values()) for k in set(k for v in reports.values() for k in v['counts'])}
        assert totals==r['runtime_totals'] and totals['association_seams']==19905
        if a['src']:SRC_seams+=totals['corrected_reference_seams_verified']
        else:native_seams+=totals['native_reference_seams_verified']
        if not a['gate']:
            assert totals['native_birth_gate_low_class_admitted']>0,'GATE_OFF_NOT_ACTUALLY_EXERCISED'
        if a['control']=='pair_shuffle':
            shuffled={k:sum(v.get('operator_audit',{}).get('pair_shuffle',{}).get(k,0) for v in reports.values())
                for k in ['changed_pairing_events','eligible_multi_source_events','changed_source_assignments']}
            assert shuffled['changed_pairing_events']>0 and shuffled['changed_source_assignments']>0
        else:shuffled={}
        metrics=r['metrics']
        assert metrics['num_objects']==229506
        assert abs(metrics['mota']-(1-sum(metrics[k] for k in ['num_false_positives','num_misses','num_switches'])/229506.))<1e-12
        logs=(ROOT/'logs'/(a['name']+'_evaluation.log')).read_text()
        assert 'OVERALL' in logs and all(s in logs for s in m['sequence_frames'])
        records.append(dict(arm=a,metrics=metrics,counts=totals,raw_output_rows=rows,
            metrics_sha256=r['metrics_sha256'],result_sha256=r['result_sha256'],pair_shuffle=shuffled))
    assert records[0]['result_sha256']==reference['result_files_sha256']
    assert records[0]['metrics']==reference['metrics']['OVERALL']
    assert pred_count==391 and native_seams==398100 and SRC_seams==59715
    return m,records,dict(status='PASS_ALL_23_ACTUAL_GRID_CELLS_AND_SOURCE_DEVICE_INPUT_SCORE_AUDIT',
        created_utc=datetime.datetime.utcnow().isoformat()+'Z',manifest_sha256=sha(ROOT/'manifest.json'),
        cells=23,sequences_per_cell=17,frames_per_cell=6635,prediction_files_rehashed=pred_count,
        same_ordered_detector_ReID_ledgers_all_cells=True,native_whole_method_seams_verified=native_seams,
        SRC_original_whole_method_seams_verified=SRC_seams,fresh_full_SRC_gate_on_parity=True,
        both_gate_states_actually_exercised=True,all_negative_and_null_cells_retained=True,
        original_GT_denominator=229506,no_refit_no_testdev_selection=True,
        postprocessing_source_sha256=sha(Path(__file__)))

def delta(treatment,control):
    d={k:treatment[k]-control[k] for k in METRICS}
    d['mota']*=100;d['idf1']*=100
    return d

def tables(m,records,seal):
    out=ROOT/'artifacts/completed_grid';out.mkdir(exist_ok=False)
    primary=[r for r in records if not r['arm']['src']]
    by={(r['arm']['control'],r['arm']['gate']):r for r in primary}
    gains={str(int(g)):delta(by['full',g]['metrics'],by['disabled',g]['metrics']) for g in [True,False]}
    interaction={k:gains['0'][k]-gains['1'][k] for k in METRICS}
    write(out/'FINAL_AUDIT.json',seal)
    write(out/'RESULTS.json',dict(records=records,full_EGIA_minus_disabled_within_gate=gains,
        difference_of_EGIA_effect_gate_off_minus_gate_on=interaction,
        unit='MOTA/IDF1 percentage points; counts in original units',
        scientific_scope='Frozen original heads with native association; not independently refitted Table III EGIA-only models'))
    with (out/'tracking_results.csv').open('x',newline='') as f:
        w=csv.writer(f);w.writerow(['cell','SRC','gate_on','EGIA_control','MOTA','IDF1','FP','FN','IDs','FM','Recall'])
        for r in records:
            a=r['arm'];s=r['metrics'];w.writerow([a['name'],a['src'],a['gate'],a['control'],100*s['mota'],100*s['idf1'],
                s['num_false_positives'],s['num_misses'],s['num_switches'],s['num_fragmentations'],100*s['recall']])
    with (out/'birth_statistics.csv').open('x',newline='') as f:
        w=csv.writer(f);w.writerow(['cell','gate_calls','native_admitted','low_class_native_admitted','final_births','EGIA_vetoes','output_rows'])
        for r in records:
            c=r['counts'];w.writerow([r['arm']['name'],c['native_birth_gate_calls'],c['native_birth_gate_admitted'],
                c['native_birth_gate_low_class_admitted'],c['final_birth_activations'],c.get('birth_rejected_vs_local_h2_admitted',0),r['raw_output_rows']])
    lines=['# SRC 关闭后的 EGIA × 出生门控交叉消融','',
        '固定 detector/ReID、GMC、所有既有头权重与其它参数。类别置信度出生门控为 0.7 开启 / 0.0 关闭；检测分数出生门槛均为 0.6。每格 17 序列、6,635 帧，原始 test-dev GT。','',
        '## 主表：SRC 全部关闭','',
        '| EGIA 配置 | 门控 | MOTA ↑ | IDF1 ↑ | FP ↓ | FN ↓ | IDs ↓ | FM ↓ |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in primary:
        a=r['arm'];s=r['metrics']
        lines.append('| %s | %s | %.4f | %.4f | %d | %d | %d | %d |'%(LABELS[a['control']],
            '开启' if a['gate'] else '关闭',100*s['mota'],100*s['idf1'],s['num_false_positives'],s['num_misses'],s['num_switches'],s['num_fragmentations']))
    lines+=['','## 各门控下完整 EGIA 相比关闭 EGIA','',
        '| 门控 | ΔMOTA (pp) | ΔIDF1 (pp) | ΔFP | ΔFN | ΔIDs |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for g in [1,0]:
        d=gains[str(g)];lines.append('| %s | %+.4f | %+.4f | %+d | %+d | %+d |'%(
            '开启' if g else '关闭',d['mota'],d['idf1'],d['num_false_positives'],d['num_misses'],d['num_switches']))
    lines+=['','门控交互：上述差值在 gate-off 与 gate-on 之间作差。原始计数变化是闭环结果，不能把不同行候选的比例当作同一批目标的因果分类准确率。','',
        '## 完整 SRC 参照','',
        '| 配置 | 门控 | MOTA ↑ | IDF1 ↑ | FP ↓ | FN ↓ | IDs ↓ |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for r in records:
        if not r['arm']['src']:continue
        a=r['arm'];s=r['metrics'];lines.append('| 完整 SRC + %s | %s | %.4f | %.4f | %d | %d | %d |'%(
            LABELS[a['control']],'开启' if a['gate'] else '关闭',100*s['mota'],100*s['idf1'],s['num_false_positives'],s['num_misses'],s['num_switches']))
    old=read(Path(m['parent_root'])/'YOLOX_outputs/D01_EGIA_bypass/receipt.json')
    s=old['metrics'];lines.append('| 完整 SRC + 关闭 EGIA（前次完成记录） | 开启 | %.4f | %.4f | %d | %d | %d |'%(
        100*s['mota'],100*s['idf1'],s['num_false_positives'],s['num_misses'],s['num_switches']))
    lines+=['','前景/覆盖固定先验仅干预指定的概率头输出，保留其余模型。几何/外观的中性化在当前算子下与关闭一致性项代数等价，故未作为额外独立增益重复统计。',
        '采用原有 full-SRC 上下文拟合权重，本轮是 SRC-off 运行的冻结头消融；不能把它当作 Table III 中另行拟合的原生分支 EGIA-only。',
        '所有二十格结果同时报告；不据 test-dev 总分重新选模型或隐藏门控。门控可在实现细节、表注或补充材料中描述，但应保证可复现。',
        '当前实验不包含跨数据集验证、重训架构优越性或统计显著性结论。']
    (out/'RESULTS_FOR_AUTHOR_ZH.md').write_text('\n'.join(lines)+'\n')
    return out

def archive(out):
    dest=REPO/PREFIX;dest.mkdir(exist_ok=False)
    for p in out.iterdir():shutil.copy2(p,dest/p.name)
    for n in ['manifest.json','PREREGISTRATION.json','EXPERIMENT_PLAN.md','AGENTS.md']:
        shutil.copy2(ROOT/n,dest/n)
    shutil.copy2(ROOT/'review/finalize_grid.py',dest/'finalize_grid.py')
    for n in ['grid_receipt.json','smoke_grid_receipt.json','smoke.exit','full.exit','pipeline.exit','PREFLIGHT_RESOLUTION_AUDIT.json']:
        shutil.copy2(ROOT/'artifacts'/n,dest/n)
    target=dest/'scores';target.mkdir()
    for a in read(ROOT/'manifest.json')['arms']:
        folder=target/a['name'];folder.mkdir()
        for n in ['receipt.json','tracking_metrics.json']:shutil.copy2(ROOT/'YOLOX_outputs'/a['name']/n,folder/n)
    inventory={str(p.relative_to(dest)):sha(p) for p in dest.rglob('*') if p.is_file()}
    write(dest/'ARCHIVE_RECEIPT.json',dict(status='ALL_23_SCORED_CELLS_AND_TABLES_COMPLETE',files_sha256=inventory,
        raw_predictions_or_weights_in_git=False,runtime_root=str(ROOT),remote_backup=False))
    eligible={p.decode() for p in git('ls-files','--cached','--others','--exclude-standard','-z','--',PREFIX).split(b'\0') if p}
    expected={str(p.relative_to(REPO)) for p in dest.rglob('*') if p.is_file()}
    assert eligible==expected and len(eligible)>50
    before=git('diff','--cached','--binary');old=git('rev-parse','HEAD').decode().strip()
    fd,index=tempfile.mkstemp(prefix='leaf_gate_grid_final_index_',dir='/tmp');os.close(fd);os.unlink(index)
    env=os.environ.copy();env['GIT_INDEX_FILE']=index
    try:
        git('read-tree','HEAD',env=env)
        cmd=['python3','scripts/stage_snapshot.py','--stage',STAGE,'--message',
             'LEAF EGIA: complete SRC-off internal controls crossed with birth gate on/off and SRC reference anchors','--local-only']
        for p in sorted(eligible):cmd+=['--include',p]
        subprocess.run(cmd,cwd=REPO,env=env,check=True)
    finally:
        head=git('rev-parse','HEAD').decode().strip()
        if head!=old:
            changes=git('diff','--name-only',old,head).decode().splitlines()
            assert all(p.startswith(PREFIX) or p.startswith('versioning/stages/'+STAGE+'/') for p in changes)
            git('restore','--staged','--source=HEAD','--',*changes)
        assert git('diff','--cached','--binary')==before
        if Path(index).exists():Path(index).unlink()
    return dict(local_head=head,archive_root=str(dest),remote_backup=False,
        preserved_staged_diff_sha256=hashlib.sha256(before).hexdigest())

def main():
    manifest,records,seal=audit();out=tables(manifest,records,seal)
    archival=archive(out)
    write(ROOT/'review/FINALIZATION_RECEIPT.json',dict(status='COMPLETE_VERIFIED_23_CELLS_AND_BOTH_GATE_TABLES',
        completed_utc=datetime.datetime.utcnow().isoformat()+'Z',author_table=str(out/'RESULTS_FOR_AUTHOR_ZH.md'),
        final_audit_sha256=sha(out/'FINAL_AUDIT.json'),**archival))
    (ROOT/'review/finalization.exit').write_text('0\n')

if __name__=='__main__':
    try:main()
    except BaseException as error:
        write(ROOT/'review/FINALIZATION_RECEIPT.json',dict(status='FINALIZATION_FAILED_NO_COMPLETE_CLAIM',error=str(error),traceback=traceback.format_exc()))
        (ROOT/'review/finalization.exit').write_text('1\n')
        raise
