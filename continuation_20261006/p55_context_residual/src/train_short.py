#!/usr/bin/env python3
"""Matched target-only/context-residual train-only screening run."""
import hashlib
import json
import os
import random
import subprocess
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
P23 = ROOT / "continuation_20261004/p23_visual/data"
OUT = ROOT / "continuation_20261006/p55_context_residual/runs"
FIT_PAIRS = {"23", "25", "28", "29", "39", "44", "45", "51", "53", "63", "66", "69", "70", "74", "78"}
HOLDOUT_PAIRS = {"51", "53", "78"}


class CVCR(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(1408, 384), nn.GELU(), nn.Linear(384, 128))
        self.dust = nn.Sequential(nn.Linear(1408, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, target, context, fpn, context_enabled=True):
        if not context_enabled:
            context = torch.zeros_like(context)
        x = torch.cat((target, context, target-context, fpn), dim=-1)
        return F.normalize(self.proj(x), dim=-1), self.dust(x).squeeze(-1)


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def group_loss(z_a, d_a, y_a, z_b, d_b, y_b, temp=.07):
    """Full candidate-plus-dustbin CE; unknown labels are excluded explicitly."""
    def one(q, qd, qy, c, cy):
        rows, dust_targets = [], []
        for i in range(len(q)):
            if int(qy[i]) < 0:
                continue
            valid = cy >= 0
            if bool(valid.any()):
                logits = torch.cat(((q[i:i+1] @ c[valid].T).flatten() / temp, qd[i:i+1]))
                pos = torch.where(cy[valid] == qy[i])[0]
                target = int(pos[0]) if len(pos) else len(logits)-1
            else:
                logits = qd[i:i+1]
                target = 0
            rows.append(F.cross_entropy(logits[None], torch.tensor([target], device=logits.device)))
            dust_targets.append(int(target == len(logits)-1))
        if not rows:
            return q.sum() * 0., 0, 0
        return torch.stack(rows).mean(), len(rows), int(sum(dust_targets))
    la, na, da = one(z_a, d_a, y_a, z_b, y_b)
    lb, nb, db = one(z_b, d_b, y_b, z_a, y_a)
    return (la + lb) / 2, na + nb, da + db


def load():
    x = np.load(P23 / "inputs.npz", allow_pickle=False)
    target = np.load(P23 / "frozen_dino.npy", mmap_mode="r")
    fpn = np.load(P23 / "raw_fpn.npy", mmap_mode="r")
    context = np.load(ROOT / "continuation_20261006/p55_context_residual/context_dino.npy", mmap_mode="r")
    labels = np.load(P23 / "offline_pid.npy", mmap_mode="r")
    groups = json.loads((P23 / "GROUPS.json").read_text())
    if not (len(target) == len(fpn) == len(context) == len(labels) == len(x["keys"])):
        raise ValueError("feature row count mismatch")
    return target, context, fpn, labels, groups


def run_arm(name, target, context, fpn, labels, groups, seed=42):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    model = CVCR().cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    hist = []
    train = [g for g in groups if str(g["pair"]) in FIT_PAIRS - HOLDOUT_PAIRS]
    if len(train) < 500:
        raise ValueError("unexpected grouped train count")
    start = time.monotonic()
    for step in range(1200):
        g = train[step % len(train)]
        ia, ib = np.asarray(g["sides"][0], dtype=np.int64), np.asarray(g["sides"][1], dtype=np.int64)
        ta = torch.from_numpy(np.asarray(target[ia], dtype=np.float32)).cuda()
        tb = torch.from_numpy(np.asarray(target[ib], dtype=np.float32)).cuda()
        ca = torch.from_numpy(np.asarray(context[ia], dtype=np.float32)).cuda()
        cb = torch.from_numpy(np.asarray(context[ib], dtype=np.float32)).cuda()
        fa = torch.from_numpy(np.asarray(fpn[ia], dtype=np.float32)).cuda()
        fb = torch.from_numpy(np.asarray(fpn[ib], dtype=np.float32)).cuda()
        ya = torch.from_numpy(np.asarray(labels[ia], dtype=np.int64)).cuda()
        yb = torch.from_numpy(np.asarray(labels[ib], dtype=np.int64)).cuda()
        za, da = model(ta, ca, fa, context_enabled=(name == "context_residual"))
        zb, db = model(tb, cb, fb, context_enabled=(name == "context_residual"))
        loss, count, dust = group_loss(za, da, ya, zb, db, yb)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step == 0 or (step + 1) % 100 == 0:
            hist.append({"step": step + 1, "loss": float(loss.detach().cpu()), "queries": count, "dust_targets": dust})
    out = OUT / name; out.mkdir(parents=True, exist_ok=False)
    torch.save({"name": name, "step": 1200, "state_dict": model.state_dict()}, out / "fixed_step_001200.pt")
    receipt = {"status": "COMPLETE_P55_SHORT_TRAIN", "arm": name, "updates": 1200,
               "fit_pairs": sorted(FIT_PAIRS - HOLDOUT_PAIRS), "heldout_pairs": sorted(HOLDOUT_PAIRS),
               "history": hist, "checkpoint_sha256": sha(out / "fixed_step_001200.pt"),
               "context_receipt_sha256": sha(ROOT / "continuation_20261006/p55_context_residual/CONTEXT_DINO_RECEIPT.json"),
               "elapsed_seconds": time.monotonic() - start, "official_val_test_read": False}
    (out / "TRAINING_RECEIPT.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    (out / "training.exit").write_text("0\n")
    return receipt


def main():
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("run through scripts/verified_gpu.py")
    if torch.cuda.get_device_properties(0).total_memory < 39 * 1024**3:
        raise RuntimeError("not a permitted large GPU")
    target, context, fpn, labels, groups = load()
    receipts = [run_arm(n, target, context, fpn, labels, groups) for n in ("target_only", "context_residual")]
    out = {"status": "P55_SHORT_TRAIN_COMPLETE", "receipts": receipts,
           "gpu_name": torch.cuda.get_device_name(0), "gpu_total_memory": torch.cuda.get_device_properties(0).total_memory,
           "official_val_test_read": False}
    (OUT / "TRAINING_RECEIPT.json").write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": out["status"], "gpu": out["gpu_name"], "arms": [r["arm"] for r in receipts]}))


if __name__ == "__main__": main()
