"""Run the imported mature MIA baseline on an entire fit sequence, unmodified.

This is a pipeline/reproducibility replay, not a new test14 score. The official
initial GT box/ID/state protocol is retained explicitly. No parameter tuning.
"""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import time
import sys

HERE=Path(__file__).resolve().parent
PY=Path('/home/chenhc/mdmt_phase0_reproduction_20260829_202909/mdmt_env/bin/python')
CKPT=Path('/raid/datasets/chc_data/MDMT/checkpoints/autoassign_r50_fpn_8x2_1x_full_mdmt/epoch_60.pth')
UUID='GPU-aaa79a66-b5c4-e0a0-6e2f-28c1ac0e3e6a'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')

def main():
    run=HERE/'runs/fit23';run.mkdir(exist_ok=False)
    start=time.monotonic()
    source=json.loads((HERE/'SOURCE_IMPORT.json').read_text())
    for rel,digest in source['files'].items():assert sha(HERE/rel)==digest,rel
    assert sha(CKPT)=='8894ea5ffe8309017d78e2dac359d405b38aa66725e1b465a8dce30beb12c901'
    wrapper=HERE/'wrappers/supplement_mia.py'
    assert sha(wrapper)=='96c91509b3cf517dcbcbad5a0a73f2d40e46eae57b0950300027c65d55a422d8'
    physical=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,memory.total','--format=csv,noheader,nounits'],text=True)
    line=next(x for x in physical.splitlines() if UUID in x)
    assert line.startswith('3,') and 'A100' in line and float(line.split(',')[-1])>=40000
    cfg=HERE/'source/configs/mot/bytetrack/bytetrack_autoassign_full_mdmt-private-half.py'
    env=dict(os.environ,PYTHONPATH=':'.join(map(str,[HERE/'adapters',HERE/'source',HERE/'source/demo',HERE/'source/demo/utils'])),
             PYTHONDONTWRITEBYTECODE='1',CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=UUID,SUPPL_NO_VIS='1',
             OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
    for name in ['SUPPL_DUMP','DET_OVERRIDE_DIR','CKPT_OVERRIDE','ALLOC_NO_VIS']:env.pop(name,None)
    images={};input_hashes={}
    for view in [1,2]:
        ps=sorted((HERE/'inputs/fit23'/str(view)/f'23-{view}').iterdir(),key=lambda p:int(p.stem))
        ps=[p for p in ps if p.suffix.lower() in ['.jpg','.jpeg','.png']]
        images[str(view)]=[p.name for p in ps]
        input_hashes.update({str(p):sha(p) for p in ps})
        input_hashes[str(HERE/'xml'/f'23-{view}.xml')]=sha(HERE/'xml'/f'23-{view}.xml')
    assert images['1']==images['2'] and len(images['1'])>1
    command=[str(PY),str(wrapper),'--config',str(cfg),'--checkpoint',str(CKPT),
        '--input',str(HERE/'inputs/fit23/1')+'/', '--xml_dir',str(HERE/'xml')+'/',
        '--result_dir',str(run/'json'),'--method','mia_baseline','--output',str(run/'vis1'),
        '--output2',str(run/'vis2'),'--device','cuda:0']
    dump(run/'PRE_RUN.json',dict(command=command,gpu=line,input_frames=len(images['1']),input_hashes=input_hashes,
        checkpoint_sha256=sha(CKPT),wrapper_sha256=sha(wrapper),source_import_sha256=sha(HERE/'SOURCE_IMPORT.json'),
        protocol='official first-frame XML boxes/IDs + confirmed tracks/Kalman + geometric initialization; retained',
        split='fit pair23 only; no new test inference',hyperparameters='unchanged official defaults',environment={k:env[k] for k in ['PYTHONPATH','PYTHONDONTWRITEBYTECODE','CUDA_DEVICE_ORDER','CUDA_VISIBLE_DEVICES','SUPPL_NO_VIS']}))
    probe="import json,torch,mmcv,mmdet,mmtrack;from mdmt_compat_runtime import init_model_mdmt;print(json.dumps(dict(torch=torch.__version__,mmcv=mmcv.__version__,mmdet=mmdet.__version__,mmtrack=mmtrack.__file__,device=torch.cuda.get_device_name(0))))"
    packages=subprocess.check_output([str(PY),'-c',probe],env=env,cwd=run,text=True,stderr=subprocess.STDOUT)
    (run/'ENVIRONMENT.txt').write_text(packages)
    assert str(HERE/'source/mmtrack') in packages
    with (run/'inference.log').open('w') as log:
        rc=subprocess.call(command,env=env,cwd=run,stdout=log,stderr=subprocess.STDOUT)
    (run/'launcher.exit').write_text(str(rc)+'\n')
    if rc:raise RuntimeError('MIA inference failed; preserve log and partial outputs')
    outputs={};counts={}
    for view in [1,2]:
        p=run/'json/mia_baseline'/f'23-{view}.json';data=json.loads(p.read_text())
        assert list(data)==[f'frame={i}' for i in range(len(images[str(view)]))]
        outputs[str(p)]=sha(p);counts[str(view)]=dict(frames=len(data),rows=sum(map(len,data.values())))
    for rel,digest in source['files'].items():assert sha(HERE/rel)==digest,rel
    result=dict(status='COMPLETE_FULL_FIT23_MIA_REPLAY',seconds=time.monotonic()-start,gpu=line,counts=counts,outputs=outputs,
        pre_run_sha256=sha(run/'PRE_RUN.json'),environment_sha256=sha(run/'ENVIRONMENT.txt'),
        launcher_exit=0,no_new_test_inference=True,no_parameter_tuning=True,
        boundary='full training-sequence replay validates execution; independently re-scored archived test14 establishes paper-level parity')
    dump(run/'RECEIPT.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
