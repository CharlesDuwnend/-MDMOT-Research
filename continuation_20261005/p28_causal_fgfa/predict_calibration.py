#!/usr/bin/env python3
"""Freeze all planned detector outputs before any calibration labels open."""
import copy
import json
from pathlib import Path
import time
import torch
from train_adapter import HERE, CachedGroups, model_init, make_adapter, state_sha
from export_fpn import sha, dump, CKPT, CKPT_SHA, source_bindings


def main():
    torch.set_num_threads(2)
    protocol=json.loads((HERE/'TRAIN_PROTOCOL.json').read_text())
    receipt_path=HERE/'training/TRAIN_RECEIPT.json'
    training=json.loads(receipt_path.read_text())
    assert training['status']=='COMPLETE_THREE_ARM_FIT'
    assert training['protocol_sha256']==sha(HERE/'TRAIN_PROTOCOL.json')
    assert set(training['checkpoints'])==set(protocol['arms'])
    for path,expected in training['inputs'].items():
        # This checks file bytes, without parsing any XML annotations.
        assert sha(path)==expected,path
    inputs=source_bindings()
    for path in [receipt_path,HERE/'TRAIN_PROTOCOL.json',HERE/'temporal_module.py',HERE/'train_adapter.py',Path(__file__)]:inputs[str(path)]=sha(path)
    for name in ['FEATURE_INDEX.json','FEATURE_RECEIPT.json','FEATURE_VERIFY.json','FLOW_INDEX.json','FLOW_RECEIPT.json','FLOW_PROTOCOL.json']:
        path=HERE/'calibration'/name;inputs[str(path)]=sha(path)
    dataset=CachedGroups('calibration',verify_archives=True)
    assert len(dataset.groups)==40
    model,runtime=model_init();head=model.detector.bbox_head
    assert state_sha(model)==training['frozen_model_state_sha256']
    out=HERE/'predictions';out.mkdir(exist_ok=False)
    dump(out/'PRE_PREDICT.json',dict(inputs=inputs,labels_read=False,group_count=40,runtime=runtime))
    models={};checkpoints={}
    for arm,mode in protocol['arms'].items():
        c=training['checkpoints'][arm]
        assert sha(c['path'])==c['sha256']
        checkpoint=torch.load(c['path'],map_location='cpu')
        assert checkpoint['arm']==arm and checkpoint['mode']==mode and checkpoint['steps']==protocol['steps']
        assert checkpoint['module_sha256']==sha(HERE/'temporal_module.py')
        assert checkpoint['protocol_sha256']==sha(HERE/'TRAIN_PROTOCOL.json')
        adapter=make_adapter(mode,protocol['seed']);adapter.load_state_dict(checkpoint['state_dict'],strict=True);adapter.eval().requires_grad_(False)
        assert state_sha(adapter)==c['final_state_sha256']
        models[arm]=adapter
        checkpoints[arm]={'path':c['path'],'sha256':c['sha256']}
    predictions={arm:{} for arm in ['native']+list(models)}
    from mmdet.core import bbox2result
    start=time.monotonic();mechanism=[]
    with torch.no_grad():
        for i,g in enumerate(dataset.groups,1):
            current,past,flow,meta,geometry,native=dataset.load(g['key'])
            predictions['native'][g['key']]={str(c):a.tolist() for c,a in enumerate(native)}
            for arm,adapter in models.items():
                aux=adapter(current,past,flow,**geometry,return_aux=True)
                features=aux['features']
                mechanism.append(dict(key=g['key'],arm=arm,levels=[dict(level=l,
                    mean_current_weight=float(w[0].mean()),
                    min_current_weight=float(w[0].min()),
                    max_current_weight=float(w[0].max()),
                    mean_valid_past=float(v.float().mean()) if v.numel() else None,
                    relative_feature_l2=float((f-c).norm()/c.norm().clamp_min(1e-12)))
                    for l,(w,v,f,c) in enumerate(zip(aux['weights'],aux['valid'],features,current))]))
                det=head.simple_test(features,[copy.deepcopy(meta)],rescale=True)[0]
                arrays=bbox2result(det[0],det[1],3)
                predictions[arm][g['key']]={str(c):a.tolist() for c,a in enumerate(arrays)}
            if i%10==0:print(json.dumps(dict(event='CAL_PREDICTION_PROGRESS',groups=i,total=40,seconds=time.monotonic()-start)),flush=True)
    artifacts={};predictions_by_arm={}
    mechanism_path=out/'MECHANISM.json';dump(mechanism_path,mechanism);artifacts[str(mechanism_path)]=sha(mechanism_path)
    for arm,pred in predictions.items():
        path=out/(arm+'.json');dump(path,pred);artifacts[str(path)]=sha(path)
        predictions_by_arm[arm]={'path':str(path),'sha256':sha(path)}
    for c in checkpoints.values():artifacts[c['path']]=c['sha256']
    for name in ['GROUPS.json','CONFIG.json','TRAIN_PROTOCOL.json']:
        path=HERE/name;artifacts[str(path)]=sha(path)
    for path,expected in inputs.items():assert sha(path)==expected,path
    assert state_sha(model)==training['frozen_model_state_sha256']
    dump(HERE/'PREDICTION_RECEIPT.json',dict(status='COMPLETE_CALIBRATION_PREDICTIONS_FROZEN',group_count=40,arms=list(predictions),artifacts=artifacts,predictions_by_arm=predictions_by_arm,checkpoints_by_arm=checkpoints,native_checkpoint={'path':str(CKPT),'sha256':CKPT_SHA},inputs=inputs,labels_read=False,seconds=time.monotonic()-start,runtime=runtime,original_output_semantics='native score_thr=.05 before score factors; NMS.6 max100; no added postfilter',training_receipt_sha256=sha(receipt_path),protocol_sha256=sha(HERE/'TRAIN_PROTOCOL.json')))


if __name__=='__main__':main()
