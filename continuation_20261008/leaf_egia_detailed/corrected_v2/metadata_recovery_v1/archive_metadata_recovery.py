"""Archive compact repair lineage with an isolated, explicitly scoped index."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[1]
REPO=Path('/home/chenhc/claude_try_MDMOT')
PREFIX='continuation_20261008/leaf_egia_detailed/corrected_v2/metadata_recovery_v1/'
STAGE='leaf_egia_parallel_metadata_recovery_verified_20261008_v1'

def git(*args,env=None):return subprocess.check_output(['git',*args],cwd=REPO,env=env)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

dest=REPO/PREFIX
dest.mkdir(exist_ok=False)
names=['recover_parallel_metadata.py','launch_metadata_recovery.sh','archive_metadata_recovery.py',
 'PARALLEL_METADATA_RECOVERY_PREREGISTRATION.json','METADATA_RECOVERY_ROUND1.json',
 'METADATA_RECOVERY_ROUND2.json','METADATA_RECOVERY_ROUND3.json','METADATA_RECOVERY_RECEIPT.json',
 'METADATA_FAILURE_RESOLUTION_AUDIT.json','PARALLEL_STOP.json','original_parallel_launcher.exit',
 'original_parallel_pipeline.exit','ADDITIONAL_STORAGE_RELOCATION_RECEIPT.json',
 'relocate_additional_closed_artifacts.py','FINALIZATION_RECEIPT.json']
for n in names:shutil.copy2(ROOT/'review'/n,dest/n)
rows=json.loads((ROOT/'artifacts/completed_detailed/RESULTS.json').read_text())['records']
by={r['name']:r['metrics'] for r in rows}
full=by['D00_full_reference'];off=by['D01_EGIA_bypass'];pair=by['D06_broken_source_pairing']
text='''# 本次消融的论文解释

全部固定完整 SRC；同一组已拟合权重，只对指定的 EGIA 输出或一致性计算作干预。9 组均完成 VisDrone2019 原始 test-dev 的 17 个序列、6,635 帧推理和评分。

完整 EGIA 相比关闭 EGIA，MOTA 增加 %.4f、IDF1 增加 %.4f 个百分点，FP 减少 %d、IDs 减少 %d；同时 FN 增加 %d。这支持整体 EGIA 在当前协议中的作用，并反映精度与召回的权衡。

打乱来源配对保留两路线索的边缘分布和其他上下文；相对完整配置，IDF1 下降 %.4f 个百分点、IDs 增加 %d、FP 增加 %d。该干预支持正确来源对应关系的作用。

完整配置的 FP、IDs、FM 在这 9 组中最低。不过，覆盖判断替换为训练先验，以及两路上下文判断均替换为训练先验，MOTA、IDF1 略高于完整配置。应完整报告这些结果，将它们解释为精度、召回和身份稳定性之间的权衡。当前证据不支持“每个子项都独立提高 MOTA 和 IDF1”的表述。

两路判断使用原始 fit39 加权训练先验中性化，是冻结模型的证据依赖干预，不能宣称为重新训练的架构消融。

几何/外观线索中性化仅作用于来源一致性计算中的线索；其他上下文头仍使用其原有几何、外观等输入。两组各自实际推理后，全部 17 个预测文件均与关闭一致性项逐字节一致。这是算子退化验证，不应写成三项独立收益。

先前的 fusion/selective 四格消融已再次核验，与新完整配置保持逐预测文件和输入记录一致。该表与本次内部证据消融可并列解释。

完整九行结果保留在 ../completed/RESULTS_FOR_AUTHOR_ZH.md；原始数值与准入统计分别为 ../completed/tracking_results.csv、../completed/admission_statistics.csv。
''' % (100*(full['mota']-off['mota']),100*(full['idf1']-off['idf1']),
 off['num_false_positives']-full['num_false_positives'],off['num_switches']-full['num_switches'],
 full['num_misses']-off['num_misses'],100*(full['idf1']-pair['idf1']),
 pair['num_switches']-full['num_switches'],pair['num_false_positives']-full['num_false_positives'])
assert all(full[k]==min(r['metrics'][k] for r in rows) for k in ['num_false_positives','num_switches','num_fragmentations'])
(dest/'INTERPRETATION_FOR_AUTHOR_ZH.md').write_text(text)
(dest/'README.md').write_text('# Verified parallel completion metadata recovery\n\nPython 3.8 lacks ast.unparse. The error occurred after D06 inference, scoring and score receipt, while the two remaining arms continued independently. No inference or scoring was rerun. The compatible metadata recovery preserved the failure receipt and the controlled serial scheduler handoff, verified all nine scores and prediction files, then emitted genuine successful completion terminals. Three rounds of three checks and the unchanged scoped completion archive are recorded here.\n\nAdditional closed SRC artifacts were moved intact to RAID with byte and dependency-hash verification; stable original logical paths remain accessible. No datasets or artifacts were discarded.\n\nAll nine results and all zero or negative contrasts remain reported. Remote write is still denied to authenticated account luwnend; this phase is local-only.\n')
inventory={str(p.relative_to(dest)):sha(p) for p in dest.iterdir() if p.is_file()}
(dest/'ARCHIVE_RECEIPT.json').write_text(json.dumps(dict(status='VERIFIED_METADATA_RECOVERY_AND_ALL_SIGNED_INTERPRETATION',files_sha256=inventory),indent=2)+'\n')
eligible={s.decode() for s in git('ls-files','--cached','--others','--exclude-standard','-z','--',PREFIX).split(b'\0') if s}
expected={str(p.relative_to(REPO)) for p in dest.iterdir() if p.is_file()}
assert eligible==expected and len(eligible)>=18
before=git('diff','--cached','--binary');oldhead=git('rev-parse','HEAD').decode().strip()
fd,index=tempfile.mkstemp(prefix='leaf_metadata_recovery_index_',dir='/tmp');os.close(fd);os.unlink(index)
env=os.environ.copy();env['GIT_INDEX_FILE']=index
try:
    git('read-tree','HEAD',env=env)
    cmd=['python3','scripts/stage_snapshot.py','--stage',STAGE,'--message',
         'LEAF EGIA: verify Python 3.8 metadata recovery and preserve every ablation outcome','--local-only']
    for p in sorted(eligible):cmd+=['--include',p]
    subprocess.run(cmd,cwd=REPO,env=env,check=True)
finally:
    head=git('rev-parse','HEAD').decode().strip()
    if head!=oldhead:
        changed=git('diff','--name-only',oldhead,head).decode().splitlines()
        assert all(p.startswith(PREFIX) or p.startswith('versioning/stages/'+STAGE+'/') for p in changed)
        git('restore','--staged','--source=HEAD','--',*changed)
    assert git('diff','--cached','--binary')==before
    if Path(index).exists():Path(index).unlink()
receipt=dict(status='COMPLETE_TABLES_AND_REPAIR_LINEAGE_ARCHIVED',local_head=head,
    archive_root=str(dest),remote_backup=False,preserved_staged_diff_sha256=hashlib.sha256(before).hexdigest())
(ROOT/'review/METADATA_ARCHIVE_RECEIPT.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
