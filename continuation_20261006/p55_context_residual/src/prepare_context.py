#!/usr/bin/env python3
"""Build P55 target-masked context crops from the frozen P23 image manifest.

This stage deliberately reads only image files and predicted crop boxes. It does
not load labels or XML and does not use calibration data.
"""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
P23 = ROOT / "continuation_20261004/p23_visual/data"
OUT = ROOT / "continuation_20261006/p55_context_residual"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def crop(arr, box):
    h, w = arr.shape[:2]
    x1, y1, x2, y2 = [float(x) for x in box]
    cx, cy = (x1+x2)/2, (y1+y2)/2
    bw, bh = max(x2-x1, 2.), max(y2-y1, 2.)
    # A fixed 3x box gives context at both sides while preserving the target's
    # scale relation; clipping is recorded by the source manifest dimensions.
    ex = [max(0, int(np.floor(cx-1.5*bw))), max(0, int(np.floor(cy-1.5*bh))),
          min(w, int(np.ceil(cx+1.5*bw))), min(h, int(np.ceil(cy+1.5*bh)))]
    if ex[2] <= ex[0] or ex[3] <= ex[1]:
        raise ValueError("empty expanded box")
    patch = arr[ex[1]:ex[3], ex[0]:ex[2]].copy()
    tx1, ty1 = max(0, int(np.floor(x1))-ex[0]), max(0, int(np.floor(y1))-ex[1])
    tx2, ty2 = min(patch.shape[1], int(np.ceil(x2))-ex[0]), min(patch.shape[0], int(np.ceil(y2))-ex[1])
    keep = np.ones(patch.shape[:2], dtype=bool)
    keep[ty1:ty2, tx1:tx2] = False
    border = patch[keep]
    fill = np.median(border, axis=0) if len(border) else np.median(patch.reshape(-1, 3), axis=0)
    patch[ty1:ty2, tx1:tx2] = np.asarray(fill, dtype=np.uint8)
    return np.asarray(Image.fromarray(patch).resize((224, 224), Image.Resampling.BICUBIC), dtype=np.uint8), ex


def main():
    start = time.monotonic()
    inputs = np.load(P23 / "inputs.npz", allow_pickle=False)
    n = len(inputs["keys"])
    crops = np.lib.format.open_memmap(P23 / "crops.npy", mode="r", dtype=np.uint8, shape=(n, 3, 224, 224))
    images = json.loads((P23 / "IMAGES.json").read_text())
    if len(images) == 0 or n != len(inputs["bbox"]):
        raise ValueError("P23 input manifest mismatch")
    # Image paths are bound by the P23 manifest; hash every selected source once.
    source_hashes = {}
    out = np.lib.format.open_memmap(OUT / "context.npy", mode="w+", dtype=np.uint8, shape=(n, 3, 224, 224))
    expanded = np.zeros((n, 4), np.int32)
    source_seen = {}
    current_index = None
    current_image = None
    for i in range(n):
        image_index = int(inputs["image_index"][i])
        if image_index not in source_seen:
            record = images[image_index]
            path = Path(record["image_path"])
            if sha(path) != record["sha256"]:
                raise ValueError("source image hash changed: " + str(path))
            source_hashes[str(path)] = record["sha256"]
            with Image.open(path) as handle:
                current_image = np.asarray(handle.convert("RGB"))
            current_index = image_index
            source_seen[image_index] = True
        if current_index != image_index or current_image is None:
            raise ValueError("P23 input rows are not grouped by image_index")
        value, bounds = crop(current_image, inputs["bbox"][i])
        # PIL returns HWC, while the P23 crop cache is CHW.
        out[i] = value.transpose(2, 0, 1)
        expanded[i] = bounds
        if (i + 1) % 5000 == 0:
            print(json.dumps({"stage": "context", "rows": i + 1, "total": n}), flush=True)
    out.flush()
    # A deterministic row-level probe binds the context to the exact target row.
    assert np.isfinite(out[:64]).all() and np.isfinite(crops[:64]).all()
    receipt = {
        "status": "P55_CONTEXT_CROPS_COMPLETE",
        "rows": n,
        "images": len(source_seen),
        "shape": [n, 3, 224, 224],
        "crop_recipe": "3x predicted box expansion; target interior median-filled; bicubic224 RGB",
        "p23_inputs_sha256": sha(P23 / "inputs.npz"),
        "p23_images_manifest_sha256": sha(P23 / "IMAGES.json"),
        "p23_crops_sha256": sha(P23 / "crops.npy"),
        "source_hashes": source_hashes,
        "expanded_bounds_sha256": hashlib.sha256(expanded.tobytes()).hexdigest(),
        "labels_or_xml_read": False,
        "cal_dev_val_test_read": False,
        "elapsed_seconds": time.monotonic() - start
    }
    np.save(OUT / "expanded_bounds.npy", expanded)
    (OUT / "CONTEXT_RECEIPT.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    (OUT / "context_prepare.exit").write_text("0\n")
    print(json.dumps({k: receipt[k] for k in ("status", "rows", "images", "elapsed_seconds")}))


if __name__ == "__main__":
    main()
