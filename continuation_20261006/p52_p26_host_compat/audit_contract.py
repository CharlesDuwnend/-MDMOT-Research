#!/usr/bin/env python3
"""Offline contract checks before starting the P26 detector replay."""
import json
from pathlib import Path
import numpy as np
import torch

from p52_p26_host_replay import FeatureStore, HookState, P52_CKPT, P26, SEQ


def main():
    store = FeatureStore()
    state = HookState(store)
    assert P52_CKPT.exists()
    assert state.model.training is False
    # Use two real P26 rows from frame 10 and verify the resolver is bbox-based.
    pred = json.loads((P26 / "runs/fit23/json/mia_baseline/23-1.json").read_text())
    pred2 = json.loads((P26 / "runs/fit23/json/mia_baseline/23-2.json").read_text())
    a = np.asarray(pred["frame=10"][0], dtype=np.float32)
    b = np.asarray(pred2["frame=10"], dtype=np.float32)
    fa, ia = store.resolve(1, 10, a)
    assert fa is not None and fa.shape == (256,) and ia >= 0.5
    fb = [store.resolve(2, 10, row)[0] for row in b]
    fb = [x for x in fb if x is not None]
    assert fb
    c = torch.from_numpy(np.stack(fb)[None])
    aa = torch.from_numpy(fa[None])
    mask = torch.ones((1, len(fb)), dtype=torch.bool)
    with torch.no_grad():
        logits, dust, z = state.model(aa, c, mask)
    assert logits.shape == (1, len(fb)) and dust.shape == (1,) and z.shape == c.shape
    assert torch.isfinite(logits).all() and torch.isfinite(dust).all() and torch.isfinite(z).all()
    # Empty candidate sets must remain finite and must not manufacture a row.
    empty = torch.zeros((1, 0, 256), dtype=torch.float32)
    emask = torch.zeros((1, 0), dtype=torch.bool)
    with torch.no_grad():
        el, ed, ez = state.model(aa, empty, emask)
    assert el.shape == (1, 0) and ed.shape == (1,) and ez.shape == (1, 0, 256)
    assert torch.isfinite(ed).all()
    result = {
        "status": "PASS_P52_P26_INTERFACE_CONTRACT",
        "sequence": SEQ,
        "feature_store": store.receipt,
        "frame": 10,
        "target_iou": float(ia),
        "candidate_rows": len(fb),
        "score_shape": list(logits.shape),
        "empty_shape": list(el.shape),
        "checkpoint_sha256": __import__("hashlib").sha256(P52_CKPT.read_bytes()).hexdigest(),
        "official_val_test_access": False,
    }
    out = Path(__file__).resolve().parent / "INTERFACE_CONTRACT.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
