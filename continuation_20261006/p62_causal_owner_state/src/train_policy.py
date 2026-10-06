#!/usr/bin/env python3
"""Train and replay the P62 owner-state conditioned association policy.

The only empirical targets here are legal train-pair prefix labels.  Replay is
kept separate from official MOT evaluation: it measures prefix selection and
owner-state contamination on the held-out calibration pairs.
"""
import argparse
import csv
import ctypes
import gzip
import hashlib
import json
import random
import subprocess
import uuid
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from prefix_replay_contract import BipartitePartition, CAL, FIT, HOST, AUD, OUT, PAIRS


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_cuda(requested):
    """Verify the physical UUID after numeric masking for the legacy CUDA stack."""
    rows = [dict(zip(("index", "uuid", "name", "memory_mib", "bus"), r))
            for r in csv.reader(subprocess.check_output([
                "nvidia-smi", "--query-gpu=index,uuid,name,memory.total,pci.bus_id",
                "--format=csv,noheader,nounits"], text=True).splitlines())]
    selected = next((r for r in rows if r["uuid"].strip() == requested), None)
    if selected is None or int(selected["index"]) == 2 or int(selected["memory_mib"]) <= 4096:
        raise RuntimeError("requested GPU is absent, physical GPU2, or <=4GB")
    lib = ctypes.CDLL("libcuda.so.1")
    dev = ctypes.c_int(); value = (ctypes.c_ubyte * 16)()
    if lib.cuInit(0) != 0 or lib.cuDeviceGet(ctypes.byref(dev), 0) != 0 or lib.cuDeviceGetUuid(ctypes.byref(value), dev) != 0:
        raise RuntimeError("CUDA UUID query failed")
    actual = "GPU-" + str(uuid.UUID(bytes=bytes(value)))
    prop = torch.cuda.get_device_properties(0)
    if actual != requested or prop.name != selected["name"].strip() or abs(prop.total_memory / 2**20 - int(selected["memory_mib"])) > 1024:
        raise RuntimeError(f"physical GPU mismatch: requested={requested} actual={actual} name={prop.name}")
    return {"requested_uuid": requested, "actual_uuid": actual, "nvidia_index": int(selected["index"]),
            "name": prop.name, "total_memory_bytes": int(prop.total_memory), "visible_devices": __import__("os").environ.get("CUDA_VISIBLE_DEVICES")}


def base_features(candidate, event):
    values = np.asarray([float(x["distance"]) for x in candidate["evidence"]], dtype=np.float64)
    slope = float(np.polyfit(np.arange(len(values)), values, 1)[0]) if len(values) > 1 else 0.0
    return np.asarray([
        float(candidate["event_frame_distance"]),
        float(candidate["aggregate_distance"]),
        float(candidate["support_frames"]) / max(1, int(event["ready_frame"]) - int(event["confirmation_frame"]) + 1),
        float(values.mean()), float(values.std()), slope,
        float(values[-1] - values[0]),
    ], dtype=np.float32)


def load_pair(pair):
    cache_path = HOST / f"cache/train/{pair}.json.gz"
    labels_path = AUD / f"artifacts_v2/events/{pair}.jsonl.gz"
    commits_path = AUD / f"artifacts_v2/commits/{pair}.jsonl.gz"
    with gzip.open(cache_path, "rt") as f:
        cache = json.load(f)
    with gzip.open(labels_path, "rt") as f:
        labels = [json.loads(line) for line in f]
    with gzip.open(commits_path, "rt") as f:
        commits = [json.loads(line) for line in f]
    assert cache["split"] == "train" and cache["gt_read"] is False and cache["test_access"] is False
    assert len(cache["events"]) == len(labels)
    assert all(c.get("future_outcome_is_training_input") is False for c in commits)
    return cache, labels, commits, {str(cache_path): sha(cache_path), str(labels_path): sha(labels_path), str(commits_path): sha(commits_path)}


def make_events(pairs):
    """Return chronologically ordered events with strict-prefix baseline state."""
    all_events, sources = [], {}
    for pair in pairs:
        cache, labels, commits, hashes = load_pair(pair)
        sources.update(hashes)
        accepted = sorted((c for c in commits if c.get("owner_accepted") is True),
                          key=lambda c: (int(c["ready_frame"]), int(c["left_lid"]), int(c["right_lid"])))
        state = BipartitePartition()
        ci = 0
        paired = list(enumerate(zip(cache["events"], labels)))
        paired.sort(key=lambda x: (int(x[1][1]["ready_frame"]), x[0]))
        for original_idx, (source_event, label_event) in paired:
            ready = int(label_event["ready_frame"])
            while ci < len(accepted) and int(accepted[ci]["ready_frame"]) < ready:
                c = accepted[ci]
                state.add(("1", int(c["left_lid"])), ("2", int(c["right_lid"])))
                ci += 1
            positive = set(int(x) for x in label_event["correct_source_lids_at_confirmation"])
            candidates = []
            for candidate in source_event["candidates"]:
                source_lid = int(candidate["source_local_track_id"])
                candidates.append({
                    "source_lid": source_lid,
                    "base": base_features(candidate, label_event),
                    "state": np.asarray(state.feature_tuple("1", int(label_event["target_lid"]), "2", source_lid), dtype=np.float32),
                    "label": int(label_event["label_state"] == "eligible" and source_lid in positive),
                })
            all_events.append({
                "pair": pair, "original_index": original_idx, "ready_frame": ready,
                "target_lid": int(label_event["target_lid"]), "target_uav": str(label_event["target_uav"]),
                "source_uav": str(label_event["source_uav"]), "label_state": label_event["label_state"],
                "availability": label_event["availability"], "positive_ids": sorted(positive),
                "candidates": candidates,
            })
    all_events.sort(key=lambda e: (e["pair"], e["ready_frame"], e["original_index"]))
    return all_events, sources


class COSTPolicy(nn.Module):
    """State-gated candidate scorer plus a per-event no-link dustbin."""

    def __init__(self, in_dim=14, state_dim=7):
        super().__init__()
        self.edge = nn.Sequential(nn.Linear(in_dim, 64), nn.LayerNorm(64), nn.GELU(), nn.Linear(64, 32), nn.GELU())
        self.state = nn.Sequential(nn.Linear(state_dim, 32), nn.LayerNorm(32), nn.GELU())
        self.gate = nn.Sequential(nn.Linear(state_dim, 32), nn.Sigmoid())
        self.candidate = nn.Linear(32, 1)
        self.dustbin = nn.Sequential(nn.Linear(32, 16), nn.GELU(), nn.Linear(16, 1))

    def score(self, x, state):
        edge = self.edge(x)
        state_embed = self.state(state)
        gated = self.gate(state) * edge + (1.0 - self.gate(state)) * state_embed
        return self.candidate(gated).squeeze(-1), self.dustbin(gated.mean(dim=0, keepdim=True)).squeeze()


def event_loss(model, event, device, state_mode="baseline"):
    if not event["candidates"]:
        # No candidate is already an explicit dustbin decision; it carries no
        # gradient signal and is excluded from the sampled optimization batch.
        return torch.zeros((), dtype=torch.float32, device=device, requires_grad=True)
    base = torch.as_tensor(np.stack([c["base"] for c in event["candidates"]]), dtype=torch.float32, device=device)
    state_np = np.stack([c["state"] for c in event["candidates"]])
    if state_mode == "descriptor_only":
        state_np = np.zeros_like(state_np)
    state = torch.as_tensor(state_np, dtype=torch.float32, device=device)
    cand, dust = model.score(torch.cat([base, state], dim=1), state)
    labels = torch.as_tensor([c["label"] for c in event["candidates"]], dtype=torch.float32, device=device)
    pos = torch.where(labels > 0.5)[0]
    target = int(pos[0].item()) if len(pos) == 1 else len(labels)
    logits = torch.cat([cand, dust.reshape(1)])
    ce = F.cross_entropy(logits.reshape(1, -1), torch.tensor([target], device=device))
    bce = F.binary_cross_entropy_with_logits(cand, labels)
    # The dustbin is a genuine action.  CE alone can satisfy the highly
    # imbalanced event distribution by rejecting every candidate; these
    # legal-prefix margin terms make a positive link outrank dustbin when a
    # positive prefix label exists and make rejection explicit otherwise.
    if len(pos) == 1:
        link_margin = F.relu(0.5 - (cand[pos[0]] - dust))
        return ce + 0.2 * bce + 0.5 * link_margin
    reject_margin = F.relu(0.5 - (dust - cand.max()))
    return ce + 0.2 * bce + 0.5 * reject_margin


def train(model, events, steps, device, seed):
    rng = random.Random(seed)
    fit_events = [e for e in events if e["pair"] in FIT and e["candidates"]]
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    model.train()
    losses = []
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        batch = [fit_events[rng.randrange(len(fit_events))] for _ in range(32)]
        loss = torch.stack([event_loss(model, e, device) for e in batch]).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        loss.backward()
        if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()):
            raise FloatingPointError(f"non-finite gradient at step {step}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return {"steps": steps, "loss_first10": float(np.mean(losses[:10])), "loss_last10": float(np.mean(losses[-10:])), "loss_min": float(np.min(losses))}


def score_event(model, event, state, device, state_mode):
    if not event["candidates"]:
        return None
    base = np.stack([c["base"] for c in event["candidates"]])
    state_np = np.stack([state.feature_tuple(event["target_uav"], event["target_lid"], event["source_uav"], c["source_lid"]) for c in event["candidates"]]).astype(np.float32)
    if state_mode == "descriptor_only":
        state_np[:] = 0.0
    x = torch.as_tensor(np.concatenate([base, state_np], axis=1), dtype=torch.float32, device=device)
    s = torch.as_tensor(state_np, dtype=torch.float32, device=device)
    with torch.no_grad():
        cand, dust = model.score(x, s)
    values = (cand - dust).detach().cpu().numpy()
    index = int(np.argmax(values))
    return index, float(values[index]), float(cand[index].detach().cpu()), float(dust.detach().cpu())


def replay(model, events, pairs, device, mode):
    """Replay with frame-snapshot decisions and an explicit owner transition."""
    model.eval()
    by_pair = defaultdict(list)
    for e in events:
        if e["pair"] in pairs:
            by_pair[e["pair"]].append(e)
    pair_rows = []
    for pair in pairs:
        _, _, commits, _ = load_pair(pair)
        accepted = sorted((c for c in commits if c.get("owner_accepted") is True), key=lambda c: (int(c["ready_frame"]), int(c["left_lid"]), int(c["right_lid"])))
        state = BipartitePartition()
        ci = 0
        accepted_count = correct = false = 0
        selected_count = 0
        for ready, group_it in __import__("itertools").groupby(by_pair[pair], key=lambda e: e["ready_frame"]):
            group = list(group_it)
            cutoff = int(ready) if mode in ("dynamic", "log_only") else max(0, int(ready) - 4)
            if mode in ("log_only", "delayed4"):
                while ci < len(accepted) and int(accepted[ci]["ready_frame"]) < cutoff:
                    c = accepted[ci]
                    state.add(("1", int(c["left_lid"])), ("2", int(c["right_lid"])))
                    ci += 1
            proposals = []
            for event in group:
                scored = score_event(model, event, state, device, mode if mode == "descriptor_only" else "baseline")
                if scored is None:
                    continue
                idx, margin, cand_logit, dust = scored
                # A positive margin is the learned no-link decision boundary.
                if margin > 0.0:
                    c = event["candidates"][idx]
                    proposals.append((margin, event, c))
            used = set()
            for margin, event, candidate in sorted(proposals, key=lambda x: (-x[0], x[1]["target_lid"], x[2]["source_lid"])):
                left = (event["target_uav"], event["target_lid"])
                right = (event["source_uav"], candidate["source_lid"])
                if left in used or right in used or state.degree[left] > 0 or state.degree[right] > 0:
                    continue
                used.update((left, right))
                selected_count += 1
                accepted_count += 1
                is_correct = candidate["source_lid"] in event["positive_ids"] and event["label_state"] == "eligible"
                correct += int(is_correct)
                false += int(not is_correct)
                if mode == "dynamic":
                    state.add(left, right)
            # log_only/delayed4 intentionally never mutate from policy actions.
        pair_rows.append({"pair": pair, "selected": selected_count, "correct": correct, "false": false,
                          "precision": float(correct / selected_count) if selected_count else 0.0,
                          "mode": mode})
    return {"mode": mode, "pairs": pair_rows, "selected": sum(r["selected"] for r in pair_rows),
            "correct": sum(r["correct"] for r in pair_rows), "false": sum(r["false"] for r in pair_rows),
            "precision": float(sum(r["correct"] for r in pair_rows) / max(1, sum(r["selected"] for r in pair_rows)))}


def baseline(events, pairs):
    rows = []
    for pair in pairs:
        selected = correct = false = 0
        for event in events:
            if event["pair"] != pair or event["label_state"] != "eligible" or event["availability"] != "candidate_present":
                continue
            c = min(event["candidates"], key=lambda x: float(x["base"][1]))
            selected += 1
            is_correct = c["source_lid"] in event["positive_ids"]
            correct += int(is_correct)
            false += int(not is_correct)
        rows.append({"pair": pair, "selected": selected, "correct": correct, "false": false,
                     "precision": float(correct / selected) if selected else 0.0})
    selected = sum(r["selected"] for r in rows)
    return {"mode": "aggregate_distance_top1", "pairs": rows, "selected": selected,
            "correct": sum(r["correct"] for r in rows), "false": sum(r["false"] for r in rows),
            "precision": float(sum(r["correct"] for r in rows) / max(1, selected))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=62062)
    ap.add_argument("--checkpoint", default=str(OUT / "runs/p62_cost_policy.pt"))
    ap.add_argument("--gpu_uuid", default="GPU-aaa79a66-b5c4-e0a0-6e2f-28c1ac0e3e6a")
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device)
    gpu_receipt = verify_cuda(args.gpu_uuid) if device.type == "cuda" else None
    events, sources = make_events(PAIRS)
    model = COSTPolicy().to(device)
    # Explicit CPU/GPU smoke check before a long run.
    smoke = event_loss(model, events[0], device)
    smoke.backward()
    assert torch.isfinite(smoke) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    model.zero_grad(set_to_none=True)
    train_report = train(model, events, args.steps, device, args.seed)
    Path(args.checkpoint).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "seed": args.seed, "steps": args.steps}, args.checkpoint)
    result = {
        "status": "PASS_P62_PREFIX_POLICY_TRAIN_REPLAY" if args.steps > 0 else "PASS_P62_PREFIX_POLICY_CPU_SMOKE",
        "steps": args.steps, "seed": args.seed, "device": str(device),
        "gpu_receipt": gpu_receipt,
        "pairs_fit": list(FIT), "pairs_calibration": list(CAL),
        "events": len(events), "model_parameters": sum(p.numel() for p in model.parameters()),
        "train": train_report,
        "baseline_calibration": baseline(events, CAL),
        "dynamic_calibration": replay(model, events, CAL, device, "dynamic"),
        "log_only_calibration": replay(model, events, CAL, device, "log_only"),
        "delayed4_calibration": replay(model, events, CAL, device, "delayed4"),
        "descriptor_only_calibration": replay(model, events, CAL, device, "descriptor_only"),
        "future_outcome_features_used": False, "official_val_test_read": False,
        "source_sha256": sources, "script_sha256": sha(Path(__file__)),
    }
    out_path = OUT / ("CPU_SMOKE_RESULT.json" if args.steps <= 5 and str(device) == "cpu" else "TRAIN_REPLAY_RESULT.json")
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (OUT / "TRAIN_REPLAY_RECEIPT.json").write_text(json.dumps({"status": result["status"], "result_sha256": sha(out_path), "checkpoint_path": str(Path(args.checkpoint).resolve()), "checkpoint_sha256": sha(args.checkpoint), "official_val_test_read": False}, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "steps", "device", "train", "baseline_calibration", "dynamic_calibration", "log_only_calibration", "delayed4_calibration", "descriptor_only_calibration")}, indent=2))


if __name__ == "__main__":
    main()
