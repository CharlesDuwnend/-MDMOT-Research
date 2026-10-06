#!/usr/bin/env python3
"""Check hook direction, collision behavior, and checkpoint reload parity."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from p52_p26_host_replay import P52_CKPT, HookState, FeatureStore, make_refresh
import native_set_residual_gate as p52


def fake_state(index):
    state = SimpleNamespace(stats=defaultdict(int), frame=0)
    state.score = lambda *args, **kwargs: {
        "index": index, "scores": [0.1, 0.2], "dustbin": -1.0,
        "candidate_indices": [0, 1],
    }
    return state


def main():
    # A->B: the new A owner chooses the second B candidate.
    a = np.asarray([[5, 0, 0, 10, 10, .9], [8, 20, 20, 30, 30, .9]], dtype=np.float32)
    b = np.asarray([[6, 0, 0, 10, 10, .9], [7, 20, 20, 30, 30, .9]], dtype=np.float32)
    hook = make_refresh(fake_state(1), "A_TO_B")
    out_a, out_b, matched, _, co = hook(
        [5], [np.asarray([5, 5], dtype=np.float32)], [], np.eye(3),
        np.asarray([[5, 5], [25, 25]], dtype=np.float32), a.copy(), b.copy(), [],
        np.empty((0, 6), dtype=np.float32), np.empty((0, 6), dtype=np.float32),
        None, [], thres=80)
    assert int(out_b[1, 0]) == 5 and matched == [5] and co == [5]

    # Non-contiguous geometry indices must still map a local scorer index to
    # the correct source row exactly once.
    a_gap = np.asarray([[5, 0, 0, 10, 10, .9]], dtype=np.float32)
    b_gap = np.asarray([[9, 1000, 1000, 1010, 1010, .9],
                        [6, 20, 20, 30, 30, .9],
                        [8, 1000, 1000, 1010, 1010, .9],
                        [7, 40, 40, 50, 50, .9]], dtype=np.float32)
    hook = make_refresh(fake_state(1), "A_TO_B")
    _, out_gap, matched, _, _ = hook(
        [5], [np.asarray([25, 25], dtype=np.float32)], [], np.eye(3),
        np.asarray([[25, 25]], dtype=np.float32), a_gap.copy(), b_gap.copy(), [],
        np.empty((0, 6), dtype=np.float32), np.empty((0, 6), dtype=np.float32),
        None, [], thres=80)
    assert int(out_gap[3, 0]) == 5 and matched == [5]

    # B->A: the new B owner chooses the second A candidate; direction must be
    # mirrored, otherwise this would mutate the wrong row.
    hook = make_refresh(fake_state(1), "B_TO_A")
    out_a, out_b, matched, _, co = hook(
        [7], [np.asarray([25, 25], dtype=np.float32)], [], np.eye(3),
        np.asarray([[5, 5], [25, 25]], dtype=np.float32), a.copy(), b.copy(), [],
        np.empty((0, 6), dtype=np.float32), np.empty((0, 6), dtype=np.float32),
        None, [], thres=80)
    assert int(out_a[1, 0]) == 7 and matched == [7] and co == [7]

    # Existing owner continuity must prevent a duplicate assignment.
    collision_state = fake_state(0)
    hook = make_refresh(collision_state, "A_TO_B")
    before = b.copy()
    _, out_b, matched, _, _ = hook(
        [5], [np.asarray([5, 5], dtype=np.float32)], [], np.eye(3),
        np.asarray([[5, 5]], dtype=np.float32), a.copy(), b.copy(), [6],
        np.empty((0, 6), dtype=np.float32), np.empty((0, 6), dtype=np.float32),
        None, [], thres=80)
    assert np.array_equal(out_b, before) and matched == [6]
    assert collision_state.stats["collision_skip"] == 1

    # A fresh process reload must reproduce the scorer output exactly.
    store = FeatureStore()
    first = HookState(store)
    second = p52.NativeSetResidual(relational=True).eval()
    second.load_state_dict(first.model.state_dict())
    a_t = torch.randn(1, 256)
    c_t = torch.randn(1, 3, 256)
    mask = torch.ones(1, 3, dtype=torch.bool)
    with torch.no_grad():
        x1 = first.model(a_t, c_t, mask)
        x2 = second(a_t, c_t, mask)
    assert all(torch.equal(x, y) for x, y in zip(x1, x2))
    result = {
        "status": "PASS_P52_P26_HOOK_DIRECTION_COLLISION_RELOAD",
        "direction_a_to_b": "PASS",
        "direction_b_to_a": "PASS",
        "collision_guard": "PASS",
        "reload_exact": "PASS",
        "checkpoint_sha256": hashlib.sha256(P52_CKPT.read_bytes()).hexdigest(),
        "official_val_test_access": False,
    }
    out = Path(__file__).resolve().parent / "HOOK_CONTRACT.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
