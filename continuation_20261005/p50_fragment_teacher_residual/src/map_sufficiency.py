#!/usr/bin/env python3
"""Training-sufficiency audit for the P50 ROI-map gate.

Continues the saved 1,200-step map checkpoint to 3.0 effective epochs over
the 2,124 eligible train episodes. It remains train-only and never imports a
MOT evaluator or validation/test labels.
"""
from __future__ import annotations
import json, hashlib, sys
from pathlib import Path
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import map_gate as m  # noqa: E402


def main():
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hist=m.build_teacher_lookup(m.FIT); train=m.eligible_dataset(m.FIT,hist)
    eval_hist=m.build_teacher_lookup(m.EVAL); eval_rows={sid:m.eligible_dataset((sid,),eval_hist) for sid in m.EVAL}
    model=m.SCIIDBackboneModel(channels=32,prototypes=2,output_size=7,backbone_init=m.CKPT); model.head=m.MapResidualHead(32); model.to(device).train()
    ck=Path(__file__).resolve().parents[1]/"MAP_GATE_MODEL.pt"; model.load_state_dict(torch.load(ck,map_location=device))
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-4)
    logs=[]; start=1200; total=3200
    for step in range(start,total):
        selected=[train[(step*2+j)%len(train)] for j in range(2)]
        b,t=m.batch(selected,device); opt.zero_grad(set_to_none=True); out,loss=m.loss_and_out(model,b,t); loss.backward(); opt.step()
        if (step+1)%400==0: logs.append({"step":step+1,"loss":float(loss.detach()),"grad_l1":float(sum(p.grad.abs().sum() for p in model.parameters() if p.grad is not None))})
    lookup=m.build_feature_lookup(m.EVAL); model_path=Path(__file__).resolve().parents[1]/"MAP_SUFFICIENCY_MODEL.pt"; torch.save(model.state_dict(),model_path)
    results={sid:m.evaluate(model,rows,device,lookup,limit=500) for sid,rows in eval_rows.items()}
    result={"status":"PASS_P50_MAP_TRAINING_SUFFICIENCY_AUDIT_COMPLETE","device":str(device),"fit_rows":len(train),"start_step":start,"total_steps":total,"effective_epochs":total*2/len(train),"history":logs,"per_pair":results,"teacher_in_inference":False,"official_val_test_access":False,"checkpoint":str(model_path),"checkpoint_sha256":hashlib.sha256(model_path.read_bytes()).hexdigest(),"interpretation":"3-epoch train-only sufficiency audit; still no formal MOT result"}
    out=Path(__file__).resolve().parents[1]/"MAP_SUFFICIENCY.json"; out.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result,indent=2))


if __name__=="__main__": main()
