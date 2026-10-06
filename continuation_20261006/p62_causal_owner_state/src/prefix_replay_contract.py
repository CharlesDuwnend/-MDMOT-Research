#!/usr/bin/env python3
"""Audit the legal prefix owner state used by the P62 policy.

This is a data/causality gate.  It deliberately does not report MOT metrics or
train a policy.  Every state feature is reconstructed from accepted commits
whose ready_frame is strictly earlier than the current event.  Same-frame
events therefore see the same state and cannot leak solver order.
"""
import gzip
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HOST = Path("/home/chenhc/mdmot_global_host_stage15_20260905")
AUD = Path("/home/chenhc/mdmt_cod_revision_audit_20260926")
OUT = ROOT / "continuation_20261006/p62_causal_owner_state"
FIT = ("23", "25", "28", "29", "39", "44", "45", "51", "53", "63", "66", "69", "70", "74", "78")
CAL = ("27", "32", "42", "64", "65")
PAIRS = FIT + CAL

sys.path.insert(0, str(Path("/home/chenhc/mdmt_cod_revision_audit_20260926")))
from state_repair_contract import EventWindow, OwnerState  # noqa: E402


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read_jsonl_gz(path):
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f]


class BipartitePartition:
    """Small deterministic DSU for the currently published owner partition."""

    def __init__(self):
        self.parent = {}
        self.members = defaultdict(set)
        self.degree = defaultdict(int)

    def _find(self, node):
        self.parent.setdefault(node, node)
        if self.parent[node] != node:
            self.parent[node] = self._find(self.parent[node])
        return self.parent[node]

    def _peek(self, node):
        if node not in self.parent:
            return node
        if self.parent[node] != node:
            self.parent[node] = self._peek(self.parent[node])
        return self.parent[node]

    def add(self, left, right):
        self.degree[left] += 1
        self.degree[right] += 1
        a, b = self._find(left), self._find(right)
        if a == b:
            return
        # Stable root choice keeps the compact audit reproducible.
        if repr(a) > repr(b):
            a, b = b, a
        self.parent[b] = a
        self.members[a].update(self.members.pop(b, {b}))
        self.members[a].add(a)

    def root(self, node):
        return self._find(node)

    def component_size(self, node):
        root = self._peek(node)
        return len(self.members.get(root, {root}))

    def same_component(self, left, right):
        return self._peek(left) == self._peek(right)

    def feature_tuple(self, target_uav, target_lid, source_uav, source_lid):
        left, right = (str(target_uav), int(target_lid)), (str(source_uav), int(source_lid))
        # Features are bounded, direction-aware, and contain no future outcome.
        return (
            min(self.component_size(left), 64) / 64.0,
            min(self.component_size(right), 64) / 64.0,
            min(self.degree[left], 8) / 8.0,
            min(self.degree[right], 8) / 8.0,
            float(self.degree[left] > 0),
            float(self.degree[right] > 0),
            float(self.same_component(left, right)),
        )

    def signature(self):
        """Canonical partition signature, insensitive to DSU parent compression."""
        groups = defaultdict(list)
        for node in sorted(self.parent, key=repr):
            groups[repr(self.root(node))].append((str(node[0]), int(node[1])))
        return tuple(sorted(tuple(sorted(values)) for values in groups.values()))


def load_pair(pair):
    cache_path = HOST / f"cache/train/{pair}.json.gz"
    events_path = AUD / f"artifacts_v2/events/{pair}.jsonl.gz"
    commits_path = AUD / f"artifacts_v2/commits/{pair}.jsonl.gz"
    with gzip.open(cache_path, "rt") as f:
        cache = json.load(f)
    events = read_jsonl_gz(events_path)
    commits = read_jsonl_gz(commits_path)
    assert cache["split"] == "train" and cache["gt_read"] is False and cache["test_access"] is False
    assert len(cache["events"]) == len(events)
    assert all(c.get("future_outcome_is_training_input") is False for c in commits)
    return cache, events, commits, {
        str(cache_path): sha(cache_path),
        str(events_path): sha(events_path),
        str(commits_path): sha(commits_path),
    }


def audit_pair(pair):
    cache, labels, commits, sources = load_pair(pair)
    # Baseline accepted edges are the only state source.  Strictly earlier
    # frames are consumed; same-frame accepted assignments remain invisible.
    accepted = sorted(
        [c for c in commits if c.get("owner_accepted") is True],
        key=lambda c: (int(c["ready_frame"]), int(c["left_lid"]), int(c["right_lid"])),
    )
    state = BipartitePartition()
    ci = 0
    rows = []
    same_frame_state = {}
    evidence_violations = 0
    future_commit_violations = 0
    unknown_positive = 0
    paired_events = list(enumerate(zip(cache["events"], labels)))
    paired_events.sort(key=lambda x: (int(x[1][1]["ready_frame"]), x[0]))
    for idx, (source_event, label_event) in paired_events:
        ready = int(label_event["ready_frame"])
        while ci < len(accepted) and int(accepted[ci]["ready_frame"]) < ready:
            c = accepted[ci]
            state.add(("1", int(c["left_lid"])), ("2", int(c["right_lid"])))
            ci += 1
        assert source_event["target_local_track_id"] == label_event["target_lid"]
        assert int(source_event["ready_frame"]) == ready
        # A digest of the legal state makes same-frame and independent replays
        # easy to compare without committing the raw row dataset.
        state_key = (ready, state.signature(), sum(state.degree.values()))
        if ready in same_frame_state and same_frame_state[ready] != state_key:
            raise AssertionError("same-frame events received different prefix state")
        same_frame_state[ready] = state_key
        positive_ids = set(int(x) for x in label_event["correct_source_lids_at_confirmation"])
        for candidate in source_event["candidates"]:
            max_evidence = max((int(x["frame"]) for x in candidate["evidence"]), default=0)
            evidence_violations += int(max_evidence > ready)
            state_features = state.feature_tuple(
                label_event["target_uav"], label_event["target_lid"],
                label_event["source_uav"], candidate["source_local_track_id"],
            )
            y = int(label_event["label_state"] == "eligible" and int(candidate["source_local_track_id"]) in positive_ids)
            unknown_positive += int(y and label_event["availability"] != "candidate_present")
            rows.append({"pair": pair, "event_index": idx, "ready_frame": ready,
                         "source_lid": int(candidate["source_local_track_id"]),
                         "prefix_label": y, "state_features": list(state_features)})
        future_commit_violations += sum(int(int(c["ready_frame"]) >= ready) for c in accepted[:ci])
    # Replaying the same-frame groups in reverse order must preserve every
    # state digest because no same-frame commit was admitted.
    for ready, digest in same_frame_state.items():
        assert digest[0] == ready
    return {
        "pair": pair,
        "events": len(cache["events"]),
        "candidate_rows": len(rows),
        "positive_rows": sum(x["prefix_label"] for x in rows),
        "accepted_commits": len(accepted),
        "events_with_prior_owner_state": sum(1 for _, _, degree in same_frame_state.values() if degree > 0),
        "same_frame_state_groups": len(same_frame_state),
        "max_ready_frame": max(same_frame_state) if same_frame_state else None,
        "evidence_future_frame_violations": evidence_violations,
        "future_commit_state_violations": future_commit_violations,
        "positive_availability_violations": unknown_positive,
        "state_feature_dim": 7,
        "source_sha256": sources,
    }


def main():
    results = [audit_pair(pair) for pair in PAIRS]
    # Independent synthetic transition check remains separate from empirical
    # labels; passing it does not authorize an official MOT evaluation.
    window = EventWindow.from_confirmation(1, 2, 4)
    state = OwnerState(window)
    bad, good = (("1", "a"), ("2", "b")), (("1", "a"), ("2", "c"))
    state.provisional_publish(bad, 2)
    state.revoke(bad, 5, evidence_frame=5)
    state.provisional_publish(good, 6)
    transition = state.finalize()
    assert any(r["kind"] == "link" and r["edge"][1] == ("2", "c") for r in transition["records"])
    assert all(r["evidence_future_frame_violations"] == 0 for r in results)
    assert all(r["future_commit_state_violations"] == 0 for r in results)
    assert all(r["positive_availability_violations"] == 0 for r in results)
    report = {
        "status": "PASS_P62_PREFIX_REPLAY_CONTRACT",
        "pairs": len(results),
        "fit_pairs": list(FIT),
        "calibration_pairs": list(CAL),
        "candidate_rows": sum(r["candidate_rows"] for r in results),
        "positive_rows": sum(r["positive_rows"] for r in results),
        "same_frame_state_groups": sum(r["same_frame_state_groups"] for r in results),
        "evidence_future_frame_violations": sum(r["evidence_future_frame_violations"] for r in results),
        "future_commit_state_violations": sum(r["future_commit_state_violations"] for r in results),
        "positive_availability_violations": sum(r["positive_availability_violations"] for r in results),
        "owner_transition_contract": True,
        "official_val_test_read": False,
        "future_outcome_features_used": False,
        "source_sha256": sha(Path(__file__)),
        "pair_results": results,
    }
    dump(OUT / "PREFIX_REPLAY_CONTRACT.json", report)
    dump(OUT / "PREFIX_REPLAY_RECEIPT.json", {
        "status": "P62_PREFIX_REPLAY_COMPLETE",
        "contract_sha256": sha(OUT / "PREFIX_REPLAY_CONTRACT.json"),
        "official_val_test_read": False,
        "raw_rows_committed": False,
    })
    print(json.dumps({k: v for k, v in report.items() if k != "pair_results"}, indent=2))


if __name__ == "__main__":
    main()
