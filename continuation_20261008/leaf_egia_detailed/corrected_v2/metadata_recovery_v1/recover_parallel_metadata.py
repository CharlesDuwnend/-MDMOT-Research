"""Recover completion bookkeeping only; never launch inference or rescore."""
import ast
import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'review'),str(ROOT/'tools')]
import run_mechanism as frozen
import parallel_coordinator as scheduling

def read(path):return json.loads(Path(path).read_text())

def main():
    manifest=read(ROOT/'manifest.json')
    amendment=read(ROOT/'review/GPU_SCHEDULING_AMENDMENT.json')
    frozen.check_frozen(manifest)
    before=dict(status='SEALED_METADATA_ONLY_RECOVERY_BEFORE_COMPLETION',
        declared_utc=datetime.datetime.utcnow().isoformat()+'Z',
        issue='Python 3.8 lacks ast.unparse; failure occurs after full D06 scoring and receipt',
        unchanged_scheduler_source_sha256=frozen.sha(ROOT/'review/parallel_coordinator.py'),
        original_amendment_sha256=frozen.sha(ROOT/'review/GPU_SCHEDULING_AMENDMENT.json'),
        recovery_source_sha256=frozen.sha(__file__),
        launcher_source_sha256=frozen.sha(ROOT/'review/launch_metadata_recovery.sh'),
        no_inference_or_score_reexecution=True,all_cells_and_values_retained=True)
    frozen.write(ROOT/'review/PARALLEL_METADATA_RECOVERY_PREREGISTRATION.json',before)
    while not (ROOT/'artifacts/pipeline.exit').exists():time.sleep(5)
    assert (ROOT/'artifacts/pipeline.exit').read_text().strip()=='1'
    error=read(ROOT/'review/PARALLEL_STOP.json')
    assert error['error']=="module 'ast' has no attribute 'unparse'",error
    records=[]
    for arm in manifest['arms']:
        output=ROOT/'YOLOX_outputs'/arm['name']
        record=read(output/'receipt.json')
        assert record['status']=='PASS_SCORED_ARM_COMPLETE'
        assert record['frames']==6635 and record['sequence_count']==17
        assert record['metrics_sha256']==frozen.sha(output/'tracking_metrics.json')
        for seq,h in record['result_sha256'].items():assert frozen.sha(output/'track_res'/(seq+'.txt'))==h
        device=record['same_process_gpu_verification']
        assignment=amendment['actual_allocation'][arm['name']]
        assert device['selected']['index']==assignment['index'] and assignment['index'] in [0,1,3]
        assert device['cuda_uuid']==assignment['uuid'] and device['selected']['memory_mib']==40960
        assert record['input_sha256']==read(ROOT/'YOLOX_outputs/D00_full_reference/receipt.json')['input_sha256']
        records.append(record)
    for source,expected in amendment['source_sha256'].items():assert frozen.sha(ROOT/source)==expected
    frozen.check_frozen(manifest)
    tree=ast.parse((ROOT/'tools/run_mechanism.py').read_text())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_arm')
    start=next(i for i,n in enumerate(function.body) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='hardware' for t in n.targets))
    function.name='complete_previously_launched_arm'
    function.args=ast.parse('def f(manifest,arm,env,started,name,output,log,smoke=False): pass').body[0].args
    function.body=function.body[start:]
    suffix=ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[]))
    compile(suffix,'<validated_unchanged_completion_suffix>','exec')
    assert hasattr(ast,'dump') and not hasattr(ast,'unparse')
    frozen.write(ROOT/'review/EXISTING_INFERENCE_COMPLETION_RECEIPT.json',dict(
        status='ORIGINAL_INFERENCE_COMPLETED_WITH_UNCHANGED_VALIDATION_AND_SCORING',
        inference_pid=scheduling.INFERENCE_PID,serial_pid=scheduling.SERIAL_PID,
        suffix_ast_sha256=hashlib.sha256(ast.dump(suffix,include_attributes=False).encode()).hexdigest(),
        ast_serialization='Python 3.8 compatible ast.dump; metadata serialization only',
        source_sha256=frozen.sha(ROOT/'tools/run_mechanism.py'),
        elapsed_time_origin='hardware receipt modification time; scheduler metadata only',
        metrics_sha256=records[6]['metrics_sha256'],
        original_inference_exit_verified_in='METADATA_RECOVERY_ROUND1.json'))
    frozen.write(ROOT/'review/METADATA_RECOVERY_ROUND2.json',dict(
        status='PASS_COMPLETED_INFERENCE_AND_METADATA_FIX_CONTRACTS',
        checks=[{'check':'all 9 actual scores and 153 prediction files rehashed','pass':True},
          {'check':'explicit permitted physical devices and unchanged detector/ReID ledgers','pass':True},
          {'check':'Python 3.8 AST suffix compilation and unchanged frozen source closure','pass':True}]))
    for name in ['launcher.exit','pipeline.exit']:
        path=ROOT/'artifacts'/name
        assert path.read_text().strip()=='1'
        path.rename(ROOT/'review'/('original_parallel_'+name))
    scheduling.retire_serial()
    fields=['mota','idf1','num_false_positives','num_misses','num_switches','num_fragmentations']
    lines=['arm,'+','.join(fields)]+[r['name']+','+','.join(str(r['metrics'][k]) for k in fields) for r in records]
    (ROOT/'artifacts/results.csv').write_text('\n'.join(lines)+'\n')
    frozen.write(ROOT/'artifacts/receipt.json',dict(status='PASS_PREDECLARED_ONLINE_MECHANISM_ADDENDUM_COMPLETE',
        manifest_sha256=frozen.sha(ROOT/'manifest.json'),arms=records,testdev_selection=False,
        gpu_scheduling_amendment_sha256=frozen.sha(ROOT/'review/GPU_SCHEDULING_AMENDMENT.json'),
        metadata_recovery_preregistration_sha256=frozen.sha(ROOT/'review/PARALLEL_METADATA_RECOVERY_PREREGISTRATION.json')))
    frozen.write(ROOT/'artifacts/progress.json',dict(status='COMPLETE',active_arm=None,completed_arms=records))
    frozen.write(ROOT/'review/parallel_progress.json',dict(status='COMPLETE',completed=[r['name'] for r in records[6:]],active=[]))
    lock=ROOT/'review/parallel.live.lock'
    pid=int(lock.read_text().strip().split('=')[1])
    assert not Path(f'/proc/{pid}').exists(),'FAILED_METADATA_COORDINATOR_STILL_ACTIVE'
    lock.unlink()
    frozen.write(ROOT/'review/METADATA_RECOVERY_RECEIPT.json',dict(
        status='RESOLVED_COMPLETION_METADATA_ONLY_NO_INFERENCE_RERUN',
        completed_utc=datetime.datetime.utcnow().isoformat()+'Z',
        failed_original_parallel_exit=1,controlled_original_serial_exit=143,
        scores_unchanged=True,all_9_actual_scores_complete=True,
        recovery_preregistration_sha256=frozen.sha(ROOT/'review/PARALLEL_METADATA_RECOVERY_PREREGISTRATION.json')))

if __name__=='__main__':
    try:main()
    except BaseException as error:
        frozen.write(ROOT/'review/METADATA_RECOVERY_STOP.json',dict(status='STOP_NO_COMPLETE_CLAIM',
            error=str(error),traceback=traceback.format_exc()))
        raise
