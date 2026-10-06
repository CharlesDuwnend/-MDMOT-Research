#!/usr/bin/env python3
"""P50 initialization audit: FPN-initialized/frozen identity pyramid."""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE))
import map_gate as m  # noqa: E402


def init_fpn(model):
    payload=torch.load(m.CKPT,map_location="cpu"); state=payload.get("state_dict",payload)
    # The protected AutoAssign FPN starts at C3 (512 channels), while the
    # identity pyramid additionally exposes a C2/P2 branch (256 channels).
    # Therefore only the shape-compatible C3/C4 lateral branches are copied;
    # P2 remains trainable and is audited separately rather than silently
    # slicing a checkpoint tensor.
    pairs=[("p3_id","neck.lateral_convs.0.conv"),("p4_id","neck.lateral_convs.1.conv")]
    with torch.no_grad():
        for name,key in pairs:
            block=getattr(model.pyramid,name).block
            w=state[key+".weight"]; assert tuple(w.shape)==tuple(block[0].weight.shape),(name,w.shape,block[0].weight.shape)
            block[0].weight.copy_(w)
            block[1].weight.fill_(1.0); block[1].bias.zero_()
        for name,key in (("refine_p2","neck.fpn_convs.0.conv"),("refine_p3","neck.fpn_convs.0.conv"),("refine_p4","neck.fpn_convs.1.conv")):
            conv=getattr(model.pyramid,name); w=state[key+".weight"]; b=state[key+".bias"]
            assert tuple(w.shape)==tuple(conv.weight.shape),(name,w.shape,conv.weight.shape)
            conv.weight.copy_(w); conv.bias.copy_(b)


def main():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hist=m.build_teacher_lookup(m.FIT); train=m.eligible_dataset(m.FIT,hist)
    eval_hist=m.build_teacher_lookup(m.EVAL); eval_rows={sid:m.eligible_dataset((sid,),eval_hist) for sid in m.EVAL}
    model=m.SCIIDBackboneModel(channels=256,prototypes=2,output_size=7,backbone_init=m.CKPT); init_fpn(model); model.head=m.MapResidualHead(256)
    for p in model.backbone.parameters(): p.requires_grad=False
    # P2 has no shape-compatible frozen FPN source and remains trainable; the
    # copied C3/C4 branches are frozen to isolate the initialization audit.
    for name,p in model.pyramid.named_parameters():
        p.requires_grad = name.startswith("p2_id") or name.startswith("refine_p2")
    model.to(device).train(); opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=2e-4,weight_decay=1e-4)
    history=[]
    for step in range(1200):
        selected=[train[(step*2+j)%len(train)] for j in range(2)]
        b,t=m.batch(selected,device); opt.zero_grad(set_to_none=True); out,loss=m.loss_and_out(model,b,t); loss.backward(); opt.step()
        if step in (0,399,799,1199): history.append({"step":step+1,"loss":float(loss.detach()),"grad_l1":float(sum(p.grad.abs().sum() for p in model.parameters() if p.grad is not None))})
    lookup=m.build_feature_lookup(m.EVAL); ck=HERE.parent/"MAP_FPN_GATE_MODEL.pt"; torch.save(model.state_dict(),ck)
    results={sid:m.evaluate(model,rows,device,lookup,limit=500) for sid,rows in eval_rows.items()}
    result={"status":"PASS_P50_FPN_INIT_MAP_GATE_COMPLETE","device":str(device),"fit_rows":len(train),"steps":1200,"trainable_parameters":sum(p.numel() for p in model.parameters() if p.requires_grad),"frozen_backbone_pyramid":True,"history":history,"per_pair":results,"teacher_in_inference":False,"official_val_test_access":False,"baseline":"frozen_stage3_visual_feature_cosine_same_definition_as_feature_gate","checkpoint":str(ck),"checkpoint_sha256":hashlib.sha256(ck.read_bytes()).hexdigest(),"metrics_are_train_only_diagnostics":True}
    out=HERE.parent/"MAP_FPN_GATE.json"; out.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result,indent=2))


if __name__=="__main__": main()
