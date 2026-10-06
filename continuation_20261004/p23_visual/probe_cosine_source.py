#!/usr/bin/env python3
"""Look-back on the p23_visual smoke failure.

smoke() asserted min cosine(re-encoded DINO from crops, frozen DINO row) > 0.999999
and got 0.9999980926513672. This probe separates the two possible causes:
  (a) the crops.npy byte content differs from the recipe used for the frozen rows;
  (b) the crops are identical and the gap is float32/batch-kernel noise.
It also measures whether the deviation changes retrieval ranking, which is the
property the training actually depends on.
"""
import json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_visual as TV  # sets CUDA_VISIBLE_DEVICES to GPU1 at import

snapshot = TV.setup()
arrays, groups, _ = TV.load_data()
model = TV.VisualModel('visual_ft').cuda()
g = groups[0]
idx, cut, fpn, dino, labels, valid = TV.batch(arrays, g)
n = min(8, len(idx))
rgb8 = torch.from_numpy(arrays['crops'][idx[:n]].copy()).cuda()

out = {}
with torch.no_grad():
    a = model.encode_rgb(rgb8)
    out['cosine_batch8'] = float(F.cosine_similarity(a, dino[:n]).min())

    # same rows, but encoded inside the batch size the frozen extraction used (128)
    big = arrays['crops'][idx[:128]].copy() if len(idx) >= 128 else None
    if big is not None:
        bigt = torch.from_numpy(big).cuda()
        bb = model.encode_rgb(bigt)[:n]
        out['cosine_batch128_prefix'] = float(F.cosine_similarity(bb, dino[:n]).min())
        out['batch8_vs_batch128_prefix_max_abs'] = float((a - bb).abs().max())
    else:
        out['cosine_batch128_prefix'] = None

    # full-batch reproducibility inside one call, and per-sample vs batched
    per = torch.cat([model.encode_rgb(rgb8[i:i + 1]) for i in range(n)])
    out['cosine_persample'] = float(F.cosine_similarity(per, dino[:n]).min())
    out['persample_vs_batch8_max_abs'] = float((per - a).abs().max())

    # retrieval consequence: does the tiny deviation change top-1 ranking?
    for name, emb in (('batch8', a), ('persample', per)):
        sim = emb @ dino[:n].T
        out[f'top1_agreement_{name}'] = float((sim.argmax(1) == torch.arange(n, device=sim.device)).float().mean())

out['gpu_snapshot'] = snapshot
out['frozen_dino_l2_norm_range'] = [float(dino[:n].norm(dim=-1).min()), float(dino[:n].norm(dim=-1).max())]
print(json.dumps(out, indent=2))
(HERE / 'probe_cosine_source.json').write_text(json.dumps(out, indent=2) + '\n')
