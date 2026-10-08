"""Present both gate states side by side without choosing or dropping cells."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
results=ROOT/'artifacts/completed_grid'
records=json.loads((results/'RESULTS.json').read_text())['records']
labels={'disabled':'关闭 EGIA','full':'完整 EGIA','no_foreground':'前景概率固定先验',
    'no_coverage':'覆盖概率固定先验','no_context':'两路概率固定先验','no_coherence':'关闭来源一致性',
    'pair_shuffle':'打乱来源配对','fusion_off':'关闭 fusion','selective_off':'关闭 selective',
    'fusion_selective_off':'fusion/selective 均关闭'}
by={(r['arm']['control'],r['arm']['gate']):r for r in records if not r['arm']['src']}
assert len(by)==20 and {k[0] for k in by}==set(labels)
lines=['# SRC 关闭后的 EGIA 消融：两种门控并排','',
    '类别置信度出生门控为 0.7（开启）与 0.0（关闭）；其余检测分数、模型权重和参数固定。每格均完成 17 序列、6,635 帧。','',
    '| EGIA 配置 | Gate 开 MOTA | Gate 开 IDF1 | Gate 开 IDs | Gate 关 MOTA | Gate 关 IDF1 | Gate 关 IDs |',
    '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
rows=[]
for control in labels:
    on=by[control,True]['metrics'];off=by[control,False]['metrics']
    lines.append('| %s | %.4f | %.4f | %d | %.4f | %.4f | %d |'%(
        labels[control],100*on['mota'],100*on['idf1'],on['num_switches'],
        100*off['mota'],100*off['idf1'],off['num_switches']))
    row=[control,labels[control]]
    for values in [on,off]:row.extend([100*values['mota'],100*values['idf1'],
        values['num_false_positives'],values['num_misses'],values['num_switches'],values['num_fragmentations']])
    rows.append(row)
lines+=['','FP、FN、FM 和出生准入统计完整保留在同目录 tracking_results.csv 与 birth_statistics.csv。',
    '每列以自身的“关闭 EGIA”行为基准；不能用 gate 开启的基准去比较 gate 关闭的完整方法。所有常数先验、零差异和负差异结果均保留。']
out=results/'PAIRED_GATE_TABLE_ZH.md'
with out.open('x') as f:f.write('\n'.join(lines)+'\n')
with (results/'paired_gate_results.csv').open('x',newline='') as f:
    w=csv.writer(f);w.writerow(['control','label']+[g+'_'+k for g in ['gate_on','gate_off'] for k in ['MOTA','IDF1','FP','FN','IDs','FM']]);w.writerows(rows)
print(json.dumps({'status':'ALL_TWENTY_PRIMARY_CELLS_PAIRED_WITHOUT_SELECTION','paired_rows':10,
    'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'source_results_sha256':hashlib.sha256((results/'RESULTS.json').read_bytes()).hexdigest()}))
