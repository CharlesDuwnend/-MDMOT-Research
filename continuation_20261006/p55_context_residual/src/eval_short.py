#!/usr/bin/env python3
"""Independent group-level replay of P55 short checkpoints."""
import json
from pathlib import Path
import numpy as np
import torch
from train_short import CVCR, P23, ROOT, HOLDOUT_PAIRS


def evaluate(model, enabled, target, context, fpn, labels, groups):
    per = {p: {"ranks": [], "mrr": [], "no_match_correct": 0, "no_match_total": 0} for p in HOLDOUT_PAIRS}
    model.eval()
    with torch.inference_mode():
        for g in groups:
            p = str(g["pair"])
            if p not in HOLDOUT_PAIRS:
                continue
            ia, ib = np.asarray(g["sides"][0]), np.asarray(g["sides"][1])
            def direction(qi, ci):
                tq = torch.from_numpy(np.asarray(target[qi], dtype=np.float32)).cuda()
                cq = torch.from_numpy(np.asarray(context[qi], dtype=np.float32)).cuda()
                fq = torch.from_numpy(np.asarray(fpn[qi], dtype=np.float32)).cuda()
                tc = torch.from_numpy(np.asarray(target[ci], dtype=np.float32)).cuda()
                cc = torch.from_numpy(np.asarray(context[ci], dtype=np.float32)).cuda()
                fc = torch.from_numpy(np.asarray(fpn[ci], dtype=np.float32)).cuda()
                yq, yc = torch.from_numpy(np.asarray(labels[qi], dtype=np.int64)), torch.from_numpy(np.asarray(labels[ci], dtype=np.int64))
                zq, dq = model(tq, cq, fq, context_enabled=enabled)
                zc, _ = model(tc, cc, fc, context_enabled=enabled)
                for i in range(len(qi)):
                    if int(yq[i]) < 0: continue
                    valid = yc >= 0
                    logits = (zq[i:i+1] @ zc[valid].T).flatten()/.07 if bool(valid.any()) else torch.empty(0, device='cuda')
                    logits = torch.cat((logits, dq[i:i+1]))
                    pos = torch.where(yc[valid] == yq[i])[0] if bool(valid.any()) else torch.empty(0, dtype=torch.long)
                    expected_dust = len(pos) == 0
                    prediction = int(torch.argmax(logits).item())
                    if expected_dust:
                        per[p]["no_match_total"] += 1
                        per[p]["no_match_correct"] += int(prediction == len(logits)-1)
                    else:
                        rank = 1 + int(torch.sum(logits[:-1] > logits[int(pos[0])]).item())
                        per[p]["ranks"].append(rank); per[p]["mrr"].append(1/rank)
            direction(ia, ib); direction(ib, ia)
    summary = {}
    for p, v in per.items():
        summary[p] = {"queries": len(v["ranks"]), "r1": float(np.mean(np.asarray(v["ranks"]) == 1)) if v["ranks"] else None,
                       "mrr": float(np.mean(v["mrr"])) if v["mrr"] else None,
                       "no_match_accuracy": float(v["no_match_correct"] / v["no_match_total"]) if v["no_match_total"] else None,
                       "no_match_queries": v["no_match_total"]}
    valid = [x["r1"] for x in summary.values() if x["r1"] is not None]
    return summary, float(np.mean(valid)) if valid else None


def main():
    target, context, fpn, labels, groups = __import__('train_short').load()
    out = {"status": "P55_SHORT_HOLDOUT_REPLAY", "holdout_pairs": sorted(HOLDOUT_PAIRS), "arms": {}}
    for name, enabled in (("target_only", False), ("context_residual", True)):
        ck = torch.load(ROOT / "continuation_20261006/p55_context_residual/runs" / name / "fixed_step_001200.pt", map_location="cuda", weights_only=False)
        model = CVCR().cuda(); model.load_state_dict(ck["state_dict"], strict=True)
        per, macro = evaluate(model, enabled, target, context, fpn, labels, groups)
        out["arms"][name] = {"per_pair": per, "macro_r1": macro}
    out["delta_macro_r1"] = out["arms"]["context_residual"]["macro_r1"] - out["arms"]["target_only"]["macro_r1"]
    out["pair_wins"] = sum(out["arms"]["context_residual"]["per_pair"][p]["r1"] > out["arms"]["target_only"]["per_pair"][p]["r1"] for p in HOLDOUT_PAIRS)
    out["official_val_test_read"] = False
    (ROOT / "continuation_20261006/p55_context_residual/SHORT_GATE.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": out["status"], "macro": {k: v["macro_r1"] for k,v in out["arms"].items()}, "delta": out["delta_macro_r1"], "pair_wins": out["pair_wins"]}))


if __name__ == "__main__": main()
