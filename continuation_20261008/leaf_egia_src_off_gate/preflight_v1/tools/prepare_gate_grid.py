"""Declare SRC-off EGIA controls crossed with two prescribed gate states."""
import copy
import datetime
import hashlib
import json
import pickle
import os
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
os.environ['BELIEF_HOST']=str(ROOT/'host')
os.environ['UAVDT_ONLINE_HOST']=str(ROOT/'host')
os.environ['CUDA_VISIBLE_DEVICES']=''
PARENT=Path('/home/chenhc/leaf_egia_detailed_20261008_v2')
PREVIOUS=Path('/home/chenhc/leaf_internal_mechanism_20261008_v2_metadata')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):
    with p.open('x') as f:json.dump(d,f,indent=2,sort_keys=True);f.write('\n')

old=json.loads((PARENT/'manifest.json').read_text())
for p,h in old['frozen_files_sha256'].items():assert sha(Path(p))==h
models={a['mode']:a['bundle'] for a in old['arms']}
source=pickle.loads((ROOT/'models/F00_full.pkl').read_bytes())
for control,fusion,selective in [('fusion_off',False,True),('selective_off',True,False),
        ('fusion_selective_off',False,False)]:
    obj=copy.deepcopy(source);obj['egia_fusion_policy']=fusion;obj['selective_birth_policy']=selective
    target=ROOT/'models'/('GRID_'+control+'.pkl')
    with target.open('xb') as f:pickle.dump(obj,f,protocol=4)
    models[control]=str(target.relative_to(ROOT))
controls=['disabled','full','no_foreground','no_coverage','no_context','no_coherence',
          'pair_shuffle','fusion_off','selective_off','fusion_selective_off']
gpu={0:'GPU-2a55197e-2c19-45ff-cf33-13031951f9d0',
     1:'GPU-1b297aba-ae7e-e326-5903-476f1bb683d7',
     3:'GPU-aaa79a66-b5c4-e0a0-6e2f-28c1ac0e3e6a'}
arms=[]
def add(name,control,gate,src,index):
    bundle=models[control]
    obj=pickle.loads((ROOT/bundle).read_bytes())
    arm=dict(name=name,control=control,mode=obj.get('egia_ablation','full'),
        runtime_arm='fc_full' if src else 'egia',src=src,mask=src,penalty=src,
        gate=gate,birth_class_gate=.7 if gate else 0.,
        use_egia=control!='disabled',fusion=obj['egia_fusion_policy'],
        selective=obj['selective_birth_policy'],bundle=bundle,bundle_sha256=sha(ROOT/bundle),
        corrected_reference_oracle=src,gpu_uuid=gpu[index],gpu_physical_index=index,
        historical_reference_metrics=None)
    if name=='R_SRC_on_G1_full':arm['historical_reference_metrics']=str(PARENT/'YOLOX_outputs/D00_full_reference/tracking_metrics.json')
    arms.append(arm)
# Full reference precedes parallel queues; every output is exclusive.
add('R_SRC_on_G1_full','full',True,True,0)
for i,control in enumerate(controls):
    # Each gate pair shares the same physical GPU; both states are interleaved.
    index=[0,1,3][i%3]
    for gate in [True,False]:add('S0_G%d_%s'%(gate,control),control,gate,False,index)
add('R_SRC_on_G0_full','full',False,True,1)
add('R_SRC_on_G0_disabled','disabled',False,True,3)
registration=dict(status='DECLARED_BEFORE_ANY_NEW_INFERENCE_OR_NEW_SCORE',
    created_utc=datetime.datetime.utcnow().isoformat()+'Z',
    user_question='SRC disabled EGIA detailed ablation, birth class confidence .7 on and off together',
    parent_root=str(PARENT),parent_manifest_sha256=sha(PARENT/'manifest.json'),
    new_complete_scores_seen=0,new_full_inference_cells=23,primary_cells=20,bridge_cells=3,
    all_prior_results_retained=True,new_figures=False,
    interventions='EGIA control x gate state on SRC-off, with exact fresh SRC-on reference and two gate-off bridges',
    fixed_birth_detector_score=.6,class_confidence_gate_on=.7,class_confidence_gate_off=0.,
    no_refit_no_threshold_sweep=True,no_testdev_configuration_selection=True,
    fitting_lineage='Original full-SRC-fitted F00 heads reused for every cell; native association runtime does not make these independently host-refitted EGIA-only models',
    degeneration_aliases_omitted='neutral geometry/appearance controls already byte-equal to gamma=0 in previous full-SRC experiment; not counted as independent effects',
    fit_prior_diagnostic='/home/chenhc/leaf_egia_prior_diagnostic_20261008_v1/RESULTS.json',
    reported_outcomes=['MOTA','IDF1','FP','FN','IDs','FM','native gate candidates','low-class admissions','final births'],
    scoring='Each full sequence original test-dev GT, denominator 229506; all detector/ReID ordered ledgers fixed',
    inference_GT_reads_forbidden=True,run_order='Bridge reference first; then declared per-GPU queues, gate pairs consecutive',
    paired_units='17 sequences in 14 flight groups; no frame independence or significance claims',
    substantive_gate_report_required=True,arms=arms)
write(ROOT/'PREREGISTRATION.json',registration)
plan='''# SRC 关闭后的 EGIA 内部消融与出生门控交叉对照

主问题：关闭全部 SRC 后，EGIA 在类别置信度出生门控开启/关闭时各有多少收益？增益是否依赖这道固定门控？

主表 20 格：10 种 EGIA 控制 × G1/G0。控制为关闭 EGIA、完整 EGIA、前景概率固定训练先验、覆盖概率固定训练先验、两路固定先验、关闭一致性项、打乱来源配对、关闭 fusion、关闭 selective、fusion/selective 同时关闭。

另跑 3 个桥接格：完整 SRC+完整 EGIA+G1（全部预测哈希对齐历史完成结果），完整 SRC+完整 EGIA+G0，完整 SRC+关闭 EGIA+G0。结合先前已完成的完整 SRC+关闭 EGIA+G1，形成 SRC×gate×EGIA 的锚点对照。

SRC 关闭必须禁用代价修正、出生语义 belief 初始化和递归更新，真实关联方法逐 seam 对照原生 U2MOT whole method。记录观测元数据只用于 EGIA 的现有输入，不能执行 SRC 模型。

G1/G0 只切换类别置信度出生资格 0.7/0.0；检测分数出生条件 0.6、匹配与 detector/ReID/GMC 不变。外层 gate 和内部 native fallback 使用同一个配置字段，记录低于 .7 的真实准入次数。

模型、fusion=.4、selective margin=.2 不重训或重选。每列的关闭 EGIA 是该门控的对应基准，不能混用 G1 baseline 与 G0 treatment。主对比是各门控下 Full−Disabled，以及二者之差；EGIA 子项作用也在各门控内比较。

所有格完成 17 序列、6635 帧真实推理和原始 GT 评分。共同输入逐帧核验、GPU 同进程 UUID核验、不读 GT 守卫、原生关联对照及完整终态先于统计汇总。只生成表格。结果较小、零效应和负效应全部报告；关闭 SRC 不能保证效应放大。

固定先验是指定概率头输出置常数的证据干预，保留其它学习部件，不能标作重训架构删除。既有校准集上学习覆盖头 AUC≈.997，常数 .5；最终 MOT 的微小反向结果仍需完整记录。

0.7 可在正文主方法之外的实现细节、表注或补充材料中交代，但不能从可复现实验记录中省略。冻结完整 F00 是 SRC 上下文拟合的权重；本轮 EGIA-only 为相同头的 SRC-off 运行，不与 Table III 的另行原生分支拟合偷换。
'''
(ROOT/'EXPERIMENT_PLAN.md').write_text(plan)
write(ROOT/'design.json',dict(**{k:old[k] for k in ['data_root','gt_root','detector','fixed','sequence_frames','smoke_sequence','smoke_frames','corrected_reference_sources']},
    status='GRID_DESIGN_NOT_YET_FROZEN',arms=arms,parent_root=str(PARENT),
    full_reference_metrics=str(PARENT/'YOLOX_outputs/D00_full_reference/tracking_metrics.json'),
    input_reference_receipt=str(PARENT/'YOLOX_outputs/D00_full_reference/receipt.json'),
    testdev_parameter_selection=False,gpu_uuid=gpu[0],gpu_physical_index=0))
print(json.dumps({'declared_cells':len(arms),'primary_cells':20,'reference_cells':3}))
