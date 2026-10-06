#!/usr/bin/env python3
"""Extract target-masked local-context DINO features without a crop dump."""
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
P23 = ROOT / "continuation_20261004/p23_visual/data"
OUT = ROOT / "continuation_20261006/p55_context_residual"
ASSET = Path("/home/chenhc/mdmot_research_20261002/p4_diagnosis/assets")
WEIGHT = ASSET / "dinov2_vits14_pretrain.pth"
WEIGHT_SHA = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def masked_context(arr, box):
    h, w = arr.shape[:2]
    x1, y1, x2, y2 = [float(x) for x in box]
    cx, cy = (x1+x2)/2, (y1+y2)/2
    bw, bh = max(x2-x1, 2.), max(y2-y1, 2.)
    ex = [max(0, int(np.floor(cx-1.5*bw))), max(0, int(np.floor(cy-1.5*bh))),
          min(w, int(np.ceil(cx+1.5*bw))), min(h, int(np.ceil(cy+1.5*bh)))]
    patch = arr[ex[1]:ex[3], ex[0]:ex[2]].copy()
    tx1, ty1 = max(0, int(np.floor(x1))-ex[0]), max(0, int(np.floor(y1))-ex[1])
    tx2, ty2 = min(patch.shape[1], int(np.ceil(x2))-ex[0]), min(patch.shape[0], int(np.ceil(y2))-ex[1])
    keep = np.ones(patch.shape[:2], dtype=bool); keep[ty1:ty2, tx1:tx2] = False
    border = patch[keep]
    fill = np.median(border, axis=0) if len(border) else np.median(patch.reshape(-1, 3), axis=0)
    patch[ty1:ty2, tx1:tx2] = np.asarray(fill, dtype=np.uint8)
    return np.asarray(Image.fromarray(patch).resize((224, 224), Image.Resampling.BICUBIC), dtype=np.uint8)


def main():
    start = time.monotonic()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("the verified launcher must expose exactly one GPU")
    prop = torch.cuda.get_device_properties(0)
    if prop.total_memory < 39 * 1024**3 or "A100" not in prop.name:
        raise RuntimeError("unexpected physical GPU: %s %.0f MiB" % (prop.name, prop.total_memory/2**20))
    if sha(WEIGHT) != WEIGHT_SHA:
        raise RuntimeError("DINO weight hash changed")
    inputs = np.load(P23 / "inputs.npz", allow_pickle=False)
    n = len(inputs["keys"])
    images = json.loads((P23 / "IMAGES.json").read_text())
    output = np.lib.format.open_memmap(OUT / "context_dino.npy", mode="w+", dtype=np.float32, shape=(n, 384))
    model = torch.hub.load(str(ASSET / "dinov2"), "dinov2_vits14", source="local", pretrained=False)
    model.load_state_dict(torch.load(WEIGHT, map_location="cpu", weights_only=True), strict=True)
    model.eval().cuda()
    mean = torch.tensor([.485, .456, .406], device="cuda")[None,:,None,None]
    std = torch.tensor([.229, .224, .225], device="cuda")[None,:,None,None]
    current_idx, current_image = None, None
    source_hashes = {}
    batch_rows, batch_arrays = [], []

    def flush():
        nonlocal batch_rows, batch_arrays
        if not batch_rows:
            return
        x = torch.from_numpy(np.stack(batch_arrays).transpose(0, 3, 1, 2)).cuda().float() / 255.
        with torch.inference_mode():
            z = F.normalize(model.forward_features((x-mean)/std)["x_norm_clstoken"], dim=-1).float().cpu().numpy()
        output[np.asarray(batch_rows)] = z
        batch_rows, batch_arrays = [], []

    for i in range(n):
        idx = int(inputs["image_index"][i])
        if idx != current_idx:
            record = images[idx]; path = Path(record["image_path"])
            if sha(path) != record["sha256"]:
                raise RuntimeError("source image hash changed: " + str(path))
            with Image.open(path) as handle:
                current_image = np.asarray(handle.convert("RGB"))
            source_hashes[str(path)] = record["sha256"]
            current_idx = idx
        batch_rows.append(i); batch_arrays.append(masked_context(current_image, inputs["bbox"][i]))
        if len(batch_rows) >= 64:
            flush()
        if (i + 1) % 5000 == 0:
            print(json.dumps({"stage": "context_dino", "rows": i + 1, "total": n}), flush=True)
    flush(); output.flush()
    assert np.isfinite(output).all()
    receipt = {
        "status": "P55_CONTEXT_DINO_COMPLETE", "rows": n, "images": len(source_hashes),
        "shape": [n, 384], "target_mask_recipe": "3x predicted box expansion, median-filled interior, bicubic224 RGB",
        "p23_inputs_sha256": sha(P23 / "inputs.npz"), "p23_images_manifest_sha256": sha(P23 / "IMAGES.json"),
        "dino_weight_sha256": WEIGHT_SHA, "source_hashes": source_hashes,
        "labels_or_xml_read": False, "cal_dev_val_test_read": False,
        "physical_gpu": {"name": prop.name, "total_memory": prop.total_memory},
        "elapsed_seconds": time.monotonic() - start
    }
    (OUT / "CONTEXT_DINO_RECEIPT.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    (OUT / "context_dino.exit").write_text("0\n")
    print(json.dumps({k: receipt[k] for k in ("status", "rows", "images", "elapsed_seconds")}))


if __name__ == "__main__":
    main()
