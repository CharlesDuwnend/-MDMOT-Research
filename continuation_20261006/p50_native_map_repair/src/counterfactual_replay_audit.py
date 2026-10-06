#!/usr/bin/env python3
"""Independent replay of v2 checkpoint readouts from fresh event construction."""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import torch

P = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
import counterfactual_gate as c
import native_set_residual_gate as n

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = {}
for sid in list(n.FIT + n.EVAL):
    data[sid], _ = n.build_events(sid, include_teacher=False)
evals = {sid: [e for e in data[sid] if e.get("strict_cf_eligible")] for sid in n.EVAL}
replay = {}
for arm in ("cf_off", "cf_on"):
    replay[arm] = []
    for seed in (7, 17):
        ck = P / f"counterfactual_{arm}_seed{seed}.pt"
        payload = torch.load(str(ck), map_location=DEVICE)
        model = n.NativeSetResidual(relational=True).to(DEVICE)
        model.load_state_dict(payload["state_dict"])
        got = {sid: {**n.evaluate(model, events, DEVICE), **c.evaluate_cf(model, events, DEVICE)}
               for sid, events in evals.items()}
        replay[arm].append({"seed": seed, "per_pair": got})

stored = json.loads((P / "COUNTERFACTUAL_GATE.json").read_text())
errors = []
for arm in replay:
    for index, item in enumerate(replay[arm]):
        old = stored["arms"][arm][index]["per_pair"]
        for sid in evals:
            for key in ("student_recall1", "positive_removal_dustbin_rate", "negative_removal_retain_r1"):
                if abs(float(item["per_pair"][sid][key]) - float(old[sid][key])) > 1e-7:
                    errors.append((arm, item["seed"], sid, key,
                                   item["per_pair"][sid][key], old[sid][key]))

result = {
    "status": "PASS_COUNTERFACTUAL_REPLAY_AUDIT" if not errors else "FAIL_COUNTERFACTUAL_REPLAY_AUDIT",
    "event_counts": {sid: len(evals[sid]) for sid in evals},
    "errors": errors[:20],
    "error_count": len(errors),
    "result_sha256": hashlib.sha256((P / "COUNTERFACTUAL_GATE.json").read_bytes()).hexdigest(),
    "official_val_test_access": False,
}
(P / "COUNTERFACTUAL_REPLAY_AUDIT.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
raise SystemExit(0 if not errors else 1)
