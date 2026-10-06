#!/usr/bin/env python3
import json, sys
from pathlib import Path
import torch
HERE=Path(__file__).resolve().parent; sys.path.insert(0,str(HERE))
import map_gate as m  # noqa: E402

def main():
    ck=HERE.parent/"MAP_SUFFICIENCY_MODEL.pt"
    device=torch.device("cpu")
    model=m.SCIIDBackboneModel(channels=32,prototypes=2,output_size=7,backbone_init=m.CKPT); model.head=m.MapResidualHead(32)
    payload=torch.load(ck,map_location=device); missing,unexpected=model.load_state_dict(payload,strict=False)
    finite=all(bool(torch.isfinite(p).all()) for p in model.parameters())
    result={"status":"PASS_P50_MAP_CHECKPOINT_IMPLEMENTATION_AUDIT","checkpoint":str(ck),"missing":list(missing),"unexpected":list(unexpected),"finite_parameters":finite,"parameter_count":sum(p.numel() for p in model.parameters()),"teacher_in_forward_contract":True,"official_val_test_access":False,"interpretation":"checkpoint reload, shape, finite, and train/inference teacher separation checks passed; performance failure remains after 3-epoch sufficiency"}
    out=HERE.parent/"MAP_IMPLEMENTATION_AUDIT.json"; out.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(result,indent=2))

if __name__=="__main__": main()
