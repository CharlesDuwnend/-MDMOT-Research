#!/usr/bin/env python3
"""Sealed-calibration detector readout; standard-library/NumPy/SciPy only.

The default CLI refuses label access before all four prediction sets, their
checkpoints, and GROUPS/CONFIG/TRAIN_PROTOCOL pass the prediction receipt.
--self-test runs only synthetic numerical contracts and reads no artifacts.

AP50 uses score-ranked, class-aware greedy one-to-one IoU>=.5 matching and
101-point interpolated precision. Default-output counts instead maximize
matching cardinality, then total IoU; the two are intentionally different.
All emitted Nx5 rows are evaluated without an additional confidence filter.
Boxes are continuous xyxy in original image coordinates (no +1 area term).
"""

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment


HERE = Path(__file__).resolve().parent
ARMS = ("native", "current_residual", "past_fixed_offsets", "past_learned_offsets")
CLASSES = {0: "pedestrian", 1: "bicycle", 2: "car"}
PARSER_PATH = Path("/home/chenhc/mdmot_research_20261002/p0/data/build_detection_labels.py")
MANIFEST_PATH = PARSER_PATH.parent / "sequences_train.json"
PARSER_SHA256 = "914d466cd4e6f48e0c9f23ff6f0a498923ec2269262288ec686b01e521b1b7df"
MANIFEST_SHA256 = "fc881493e1b8f6ac1fe569cc02653d06fe4a584f16153e19272802abe81b9a23"
NATIVE_CHECKPOINT_SHA256 = "8894ea5ffe8309017d78e2dac359d405b38aa66725e1b465a8dce30beb12c901"
IOU_THRESHOLD = .5


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def absolute_path(value):
    path = Path(value)
    require(path.is_absolute(), "receipt artifact paths must be absolute")
    return path.resolve()


def normalize_bindings(mapping):
    require(isinstance(mapping, dict), "hash bindings must be an object")
    result = {}
    for name, digest in mapping.items():
        path = absolute_path(name)
        require(isinstance(digest, str) and len(digest) == 64 and
                all(c in "0123456789abcdef" for c in digest), "invalid SHA256")
        require(path not in result or result[path] == digest, "conflicting alias hashes")
        result[path] = digest
    return result


def verify_bindings(bindings):
    for path, digest in bindings.items():
        require(path.is_file(), "missing sealed artifact: " + str(path))
        require(sha(path) == digest, "sealed artifact changed: " + str(path))


def read_bound_json(path, bindings):
    path = Path(path).resolve()
    require(path in bindings, "unsealed JSON artifact: " + str(path))
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == bindings[path], "JSON changed while reading")
    return json.loads(raw)


def entry_binding(entry, bindings, description):
    require(isinstance(entry, dict) and set(entry) >= {"path", "sha256"},
            "missing path/hash for " + description)
    path = absolute_path(entry["path"])
    require(bindings.get(path) == entry["sha256"], "unbound " + description)
    return path


def validate_prediction_payload(payload, keys):
    require(isinstance(payload, dict) and set(payload) == set(keys),
            "prediction group keys differ from all forty calibration groups")
    parsed = {}
    for key in sorted(keys):
        require(isinstance(payload[key], dict) and set(payload[key]) == {"0", "1", "2"},
                "each prediction group must contain exactly classes 0/1/2")
        parsed[key] = {}
        for cls in CLASSES:
            raw = payload[key][str(cls)]
            array = np.asarray(raw, dtype=np.float64)
            if array.size == 0:
                require(isinstance(raw, list) and len(raw) == 0, "malformed empty detections")
                array = array.reshape(0, 5)
            require(array.ndim == 2 and array.shape[1] == 5, "detections must be Nx5")
            require(np.isfinite(array).all(), "nonfinite detection")
            # Clipped detector outputs may have zero extent. Keep them as
            # emitted false positives (IoU zero), rather than adding a filter.
            require(np.all(array[:, 2:4] >= array[:, :2]), "inverted detection extent")
            require(np.all((array[:, 4] >= 0) & (array[:, 4] <= 1)), "invalid confidence")
            parsed[key][cls] = array
    return parsed


def gate_frozen_predictions(receipt_path):
    """Verify bytes and parse predictions before any calibration XML access."""
    receipt_path = Path(receipt_path).resolve()
    receipt_raw = receipt_path.read_bytes()
    receipt = json.loads(receipt_raw)
    require(receipt.get("status") == "COMPLETE_CALIBRATION_PREDICTIONS_FROZEN",
            "calibration prediction receipt is not complete/frozen")
    require(receipt.get("labels_read") is False, "producer must attest labels_read:false")
    require(receipt.get("group_count") == 40, "expected forty calibration groups")
    require(len(receipt.get("arms", [])) == 4 and set(receipt["arms"]) == set(ARMS),
            "receipt does not contain exactly the four declared arms")
    artifacts = normalize_bindings(receipt.get("artifacts"))
    inputs = normalize_bindings(receipt.get("inputs", {}))
    bindings = dict(inputs)
    for path, digest in artifacts.items():
        require(path not in bindings or bindings[path] == digest, "input/artifact hash conflict")
        bindings[path] = digest
    # Prediction inputs must not cause the gate itself to open raw XML labels.
    require(not any(p.suffix.lower() == ".xml" and "new_xml" in p.parts for p in bindings),
            "raw annotation XML cannot be a pre-label prediction artifact")
    for name in ("GROUPS.json", "CONFIG.json", "TRAIN_PROTOCOL.json"):
        require((HERE / name).resolve() in artifacts, "receipt must seal " + name)
    verify_bindings(bindings)
    config = read_bound_json(HERE / "CONFIG.json", bindings)
    protocol = read_bound_json(HERE / "TRAIN_PROTOCOL.json", bindings)
    groups_all = read_bound_json(HERE / "GROUPS.json", bindings)
    require(set(protocol["arms"]) == set(ARMS) - {"native"}, "training arm protocol mismatch")
    require(receipt.get("protocol_sha256") == bindings[(HERE / "TRAIN_PROTOCOL.json").resolve()],
            "prediction/training protocol binding mismatch")
    groups = [g for g in groups_all if g.get("role") == "calibration"]
    require(len(groups) == 40 and len({g["key"] for g in groups}) == 40,
            "calibration groups must have forty unique keys")
    allowed = {str(p) for p in config["calibration_pairs"]}
    require(len(allowed) == 5 and {str(g["pair"]) for g in groups} == allowed,
            "calibration pair allowlist mismatch")
    require(not (allowed & {str(p) for p in config["fit_pairs"]}), "fit/calibration pair overlap")
    counts = Counter((str(g["pair"]), str(g["view"])) for g in groups)
    require(set(counts) == {(p, v) for p in allowed for v in ("1", "2")} and
            set(counts.values()) == {4}, "require four anchors per pair/view")
    require(len({(str(g["pair"]), str(g["view"]), int(g["frame"])) for g in groups}) == 40,
            "duplicate calibration image/frame")
    for group in groups:
        require(group["frames"] == [group["frame"] - lag for lag in (0, 1, 4, 8)],
                "group temporal order differs from frozen lag contract")
        require(group["image_keys"][0] == group["key"], "current key mismatch")
    entries = receipt.get("predictions_by_arm", {})
    require(set(entries) == set(ARMS), "missing per-arm prediction bindings")
    checkpoints = receipt.get("checkpoints_by_arm", {})
    require(set(checkpoints) in (set(ARMS), set(ARMS) - {"native"}),
            "missing adapter checkpoint bindings")
    prediction_paths, checkpoint_paths, predictions = {}, {}, {}
    for arm in ARMS:
        path = entry_binding(entries[arm], artifacts, arm + " predictions")
        prediction_paths[arm] = str(path)
        predictions[arm] = validate_prediction_payload(read_bound_json(path, bindings),
                                                        [g["key"] for g in groups])
        entry = checkpoints.get(arm, receipt.get("native_checkpoint") if arm == "native" else None)
        checkpoint = entry_binding(entry, bindings, arm + " checkpoint")
        if arm == "native":
            require(bindings[checkpoint] == NATIVE_CHECKPOINT_SHA256,
                    "native checkpoint is not the frozen P26 AutoAssign baseline")
        checkpoint_paths[arm] = str(checkpoint)
    require(len(set(prediction_paths.values())) == 4, "four arms need distinct prediction files")
    # Pin audited parser and XML-hash manifest bytes; this reads only code and
    # metadata, never the annotation XML. Parser import/read_xml happens later.
    require(sha(PARSER_PATH) == PARSER_SHA256, "trusted XML parser source changed")
    require(sha(MANIFEST_PATH) == MANIFEST_SHA256, "trusted XML hash manifest changed")
    return dict(receipt=receipt, receipt_path=receipt_path,
                receipt_sha256=hashlib.sha256(receipt_raw).hexdigest(),
                bindings=bindings, groups=groups, predictions=predictions,
                prediction_paths=prediction_paths, checkpoint_paths=checkpoint_paths,
                gated_at=utc())


def load_calibration_targets(gate):
    """Only the successful receipt-gated main calls this function."""
    # Recheck the receipt and source immediately before importing the parser.
    require(sha(gate["receipt_path"]) == gate["receipt_sha256"], "receipt changed before labels")
    require(sha(PARSER_PATH) == PARSER_SHA256 and sha(MANIFEST_PATH) == MANIFEST_SHA256,
            "label parser/manifest changed after gate")
    spec = importlib.util.spec_from_file_location("p28_audited_xml_parser", str(PARSER_PATH))
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)
    require(parser.CLASS == {"person": 0, "pedestrian": 0, "bicycle": 1, "car": 2},
            "trusted class mapping changed")
    manifest = {(str(x["pair"]), str(x["view"])): x
                for x in json.loads(MANIFEST_PATH.read_text())}
    sequence_keys = sorted({(str(g["pair"]), str(g["view"])) for g in gate["groups"]})
    frames, bindings = {}, {PARSER_PATH.resolve(): PARSER_SHA256,
                           MANIFEST_PATH.resolve(): MANIFEST_SHA256}
    for pair, view in sequence_keys:
        require((pair, view) in manifest, "sequence absent from XML audit manifest")
        record = manifest[(pair, view)]
        # read_xml verifies the declared bytes, converts frame=XML frame+1,
        # excludes outside=1, validates boxes, and orders rows by raw ID.
        value, _, path = parser.read_xml(pair, view, record["xml_sha256"])
        frames[(pair, view)] = value
        bindings[Path(path).resolve()] = record["xml_sha256"]
    targets, excluded = {}, {}
    for group in gate["groups"]:
        rows = frames[(str(group["pair"]), str(group["view"]))].get(group["frame"], [])
        targets[group["key"]] = [dict(bbox=list(row["bbox"]),
                                      **{"class": int(row["class"]),
                                         "occluded": int(row["occluded"])})
                                  for row in rows if row["class"] in CLASSES]
        excluded[group["key"]] = sum(row["class"] not in CLASSES for row in rows)
    return targets, bindings, excluded


def iou_matrix(boxes_a, boxes_b):
    a = np.asarray(boxes_a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=np.float64).reshape(-1, 4)
    top = np.maximum(a[:, None, :2], b[None, :, :2])
    bottom = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.maximum(bottom - top, 0)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.maximum(a[:, 2:] - a[:, :2], 0).prod(axis=1)
    area_b = np.maximum(b[:, 2:] - b[:, :2], 0).prod(axis=1)
    union = area_a[:, None] + area_b[None, :] - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def maximum_cardinality_matches(iou, threshold=IOU_THRESHOLD):
    """Lexicographic max cardinality then max total IoU, with row dummies."""
    iou = np.asarray(iou, dtype=np.float64)
    require(iou.ndim == 2 and np.isfinite(iou).all(), "invalid IoU matrix")
    require(np.all((iou >= 0) & (iou <= 1)), "IoU must lie in [0,1]")
    n, m = iou.shape
    if n == 0 or m == 0:
        return []
    bonus = float(min(n, m) + 1)
    cost = np.zeros((n, m + n), dtype=np.float64)
    cost[:, :m] = np.where(iou >= threshold, -bonus - iou, bonus)
    rows, cols = linear_sum_assignment(cost)
    return [(int(r), int(c), float(iou[r, c])) for r, c in zip(rows, cols)
            if c < m and iou[r, c] >= threshold]


def ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def counts_report(tp, fp, fn):
    tp, fp, fn = int(tp), int(fp), int(fn)
    return dict(tp=tp, fp=fp, fn=fn, gt=tp + fn, predictions=tp + fp,
                recall=ratio(tp, tp + fn), precision=ratio(tp, tp + fp))


def ap50_for_class(predictions, targets, image_keys, cls):
    gt_by_image = {key: [row["bbox"] for row in targets[key] if row["class"] == cls]
                   for key in image_keys}
    total_gt = sum(len(rows) for rows in gt_by_image.values())
    ranked = sorted(((-float(row[4]), key, index) for key in image_keys
                     for index, row in enumerate(predictions[key][cls])))
    matched = {key: np.zeros(len(rows), dtype=bool) for key, rows in gt_by_image.items()}
    matrices = {key: iou_matrix(predictions[key][cls][:, :4], gt_by_image[key])
                for key in image_keys}
    true_positive = []
    for _, key, pred_index in ranked:
        candidates = np.flatnonzero((matrices[key][pred_index] >= IOU_THRESHOLD) & ~matched[key])
        if len(candidates):
            # np.argmax chooses first GT row on exact ties; parser row order is fixed.
            best = int(candidates[np.argmax(matrices[key][pred_index, candidates])])
            matched[key][best] = True
            true_positive.append(1)
        else:
            true_positive.append(0)
    tp = np.cumsum(np.asarray(true_positive, dtype=np.float64))
    fp = np.cumsum(1 - np.asarray(true_positive, dtype=np.float64))
    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    if total_gt:
        recall = tp / total_gt
        envelope = np.maximum.accumulate(precision[::-1])[::-1]
        sampled = [float(envelope[recall >= r].max()) if np.any(recall >= r) else 0.
                   for r in np.linspace(0, 1, 101)]
        ap = float(np.mean(sampled))
    else:
        recall, sampled, ap = np.zeros_like(tp), None, None
    return dict(ap50=ap, gt=total_gt, predictions=len(ranked),
                score_ranked_tp=int(tp[-1]) if len(tp) else 0,
                score_ranked_fp=int(fp[-1]) if len(fp) else 0,
                max_recall=float(recall[-1]) if len(recall) and total_gt else (0. if total_gt else None),
                precision_at_101_recalls=sampled)


def mean_present(values):
    kept = [float(value) for value in values if value is not None]
    return dict(mean=float(np.mean(kept)) if kept else None, eligible_count=len(kept))


def ap_report(predictions, targets, groups):
    keys = sorted(g["key"] for g in groups)
    classes = {str(cls): ap50_for_class(predictions, targets, keys, cls) for cls in CLASSES}
    pairs = {}
    for pair in sorted({str(g["pair"]) for g in groups}):
        pair_keys = sorted(g["key"] for g in groups if str(g["pair"]) == pair)
        values = {str(cls): ap50_for_class(predictions, targets, pair_keys, cls) for cls in CLASSES}
        pairs[pair] = dict(classes=values, ap50=mean_present(x["ap50"] for x in values.values()))
    return dict(classes=classes, pooled_class_macro_ap50=mean_present(x["ap50"] for x in classes.values()),
                pairs=pairs, pair_macro_ap50=mean_present(x["ap50"]["mean"] for x in pairs.values()))


def gt_strata(row):
    x1, y1, x2, y2 = row["bbox"]
    size = "sqrt_area_lt16" if math.sqrt((x2 - x1) * (y2 - y1)) < 16 else "sqrt_area_ge16"
    occlusion = "occluded_0" if row["occluded"] == 0 else (
        "occluded_1" if row["occluded"] == 1 else "occlusion_other")
    return size, occlusion


def gt_stratum_report(targets, matched, keys=None):
    keys = set(targets) if keys is None else set(keys)
    result = {"sqrt_area_lt16": Counter(), "sqrt_area_ge16": Counter(),
              "occluded_0": Counter(), "occluded_1": Counter(), "occlusion_other": Counter()}
    for key in keys:
        for index, row in enumerate(targets[key]):
            for stratum in gt_strata(row):
                result[stratum]["gt"] += 1
                result[stratum]["tp"] += int((key, index) in matched)
    return {name: dict(gt=value["gt"], tp=value["tp"], fn=value["gt"] - value["tp"],
                       recall=ratio(value["tp"], value["gt"])) for name, value in result.items()}


def default_output_report(predictions, targets, groups):
    matched_gt, matches_by_image = set(), {}
    totals, by_class, by_pair = Counter(), defaultdict(Counter), defaultdict(Counter)
    image_counts, fp_sizes = {}, Counter()
    for group in sorted(groups, key=lambda g: g["key"]):
        key, pair = group["key"], str(group["pair"])
        image = Counter()
        matches_by_image[key] = []
        for cls in CLASSES:
            gt_indices = [i for i, row in enumerate(targets[key]) if row["class"] == cls]
            pred = predictions[key][cls]
            iou = iou_matrix(pred[:, :4], [targets[key][i]["bbox"] for i in gt_indices])
            matches = maximum_cardinality_matches(iou)
            matched_pred = {r for r, _, _ in matches}
            counts = dict(tp=len(matches), fp=len(pred) - len(matches), fn=len(gt_indices) - len(matches))
            for accumulator in (totals, image, by_class[str(cls)], by_pair[pair]):
                accumulator.update(counts)
            for r, c, value in matches:
                global_gt_index = gt_indices[c]
                matched_gt.add((key, global_gt_index))
                matches_by_image[key].append(dict(**{"class": cls}, prediction_index=r,
                                                   gt_index=global_gt_index, iou=value))
            for r, row in enumerate(pred):
                if r not in matched_pred:
                    area = (row[2] - row[0]) * (row[3] - row[1])
                    fp_sizes["predicted_sqrt_area_lt16" if math.sqrt(area) < 16 else
                             "predicted_sqrt_area_ge16"] += 1
        image_counts[key] = counts_report(image["tp"], image["fp"], image["fn"])
    classes = {cls: counts_report(x["tp"], x["fp"], x["fn"]) for cls, x in sorted(by_class.items())}
    pairs = {pair: counts_report(x["tp"], x["fp"], x["fn"]) for pair, x in sorted(by_pair.items())}
    require(len(matched_gt) == totals["tp"], "one-to-one matching inconsistency")
    report = dict(overall=counts_report(totals["tp"], totals["fp"], totals["fn"]),
                  classes=classes, pairs=pairs, images=image_counts,
                  pair_macro_recall=mean_present(x["recall"] for x in pairs.values()),
                  pair_macro_precision=mean_present(x["precision"] for x in pairs.values()),
                  gt_strata=gt_stratum_report(targets, matched_gt),
                  unmatched_prediction_size_counts=dict(fp_sizes))
    return report, matched_gt, matches_by_image


def recovery_counts(native, candidate, universe):
    native, candidate = native & universe, candidate & universe
    missed = universe - native
    recovered, lost = candidate - native, native - candidate
    require(len(candidate) - len(native) == len(recovered) - len(lost), "recovery conservation")
    return dict(gt=len(universe), native_tp=len(native), native_missed=len(missed),
                candidate_tp=len(candidate), kept_tp=len(native & candidate),
                recovered_native_missed=len(recovered), lost_native_tp=len(lost),
                remaining_native_missed=len(missed - candidate),
                recovery_rate=ratio(len(recovered), len(missed)),
                loss_rate=ratio(len(lost), len(native)), net_tp_change=len(candidate) - len(native))


def recovery_report(native, candidate, targets, groups):
    universe = {(key, i) for key, rows in targets.items() for i in range(len(rows))}
    by_pair, strata = {}, defaultdict(set)
    for pair in sorted({str(g["pair"]) for g in groups}):
        keys = {g["key"] for g in groups if str(g["pair"]) == pair}
        by_pair[pair] = recovery_counts(native, candidate, {x for x in universe if x[0] in keys})
    for node in universe:
        for name in gt_strata(targets[node[0]][node[1]]):
            strata[name].add(node)
    return dict(overall=recovery_counts(native, candidate, universe), pairs=by_pair,
                gt_strata={name: recovery_counts(native, candidate, nodes)
                           for name, nodes in sorted(strata.items())})


def write_json(path, value):
    raw = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(raw)
    temporary.replace(path)


def run_self_tests():
    """Synthetic only: no filesystem, predictions, manifest or XML reads."""
    checked = 0
    # Exhaustive eligible graphs up to 3x3, checked against a brute-force
    # assignment enumerator independent of the Hungarian implementation.
    for n, m in ((0, 3), (3, 0), (1, 1), (2, 3), (3, 3)):
        for bits in itertools.product((0, 1), repeat=n * m):
            matrix = np.asarray(bits, dtype=np.float64).reshape(n, m) * .7
            matches = maximum_cardinality_matches(matrix)
            best = (0, 0.)
            for assignment in itertools.product(range(-1, m), repeat=n):
                used = [c for c in assignment if c >= 0]
                if len(set(used)) != len(used):
                    continue
                if any(c >= 0 and matrix[r, c] < .5 for r, c in enumerate(assignment)):
                    continue
                score = (len(used), sum(matrix[r, c] for r, c in enumerate(assignment) if c >= 0))
                best = max(best, score)
            require(len(matches) == best[0], "synthetic maximum cardinality failed")
            require(abs(sum(x[2] for x in matches) - best[1]) < 1e-10, "synthetic IoU tie objective")
            checked += 1
    rng = np.random.RandomState(7)
    for _ in range(32):
        matrix = rng.rand(3, 3)
        matches = maximum_cardinality_matches(matrix)
        feasible = []
        for assignment in itertools.product(range(-1, 3), repeat=3):
            used = [c for c in assignment if c >= 0]
            if len(set(used)) == len(used) and all(c < 0 or matrix[r, c] >= .5 for r, c in enumerate(assignment)):
                feasible.append((len(used), sum(matrix[r, c] for r, c in enumerate(assignment) if c >= 0)))
        expected = max(feasible)
        require(len(matches) == expected[0] and abs(sum(x[2] for x in matches) - expected[1]) < 1e-10,
                "weighted graph objective failed")
        checked += 1
    empty = np.empty((0, 5), dtype=np.float64)
    targets = {"a": [{"bbox": [0, 0, 10, 10], "class": 0, "occluded": 0}], "b": []}
    predictions = {"a": {0: np.array([[0, 0, 10, 10, .8]]), 1: empty, 2: empty},
                   "b": {0: np.array([[20, 20, 30, 30, .9]]), 1: empty, 2: empty}}
    require(abs(ap50_for_class(predictions, targets, ["a", "b"], 0)["ap50"] - .5) < 1e-12,
            "AP must count high-score FP on empty-GT image")
    require(ap50_for_class(predictions, targets, ["a", "b"], 1)["ap50"] is None,
            "class without GT must not silently become AP zero")
    targets["a"].append({"bbox": [5, 0, 15, 10], "class": 0, "occluded": 1})
    predictions["a"][0] = np.array([[2, 0, 12, 10, .9], [0, 0, 10, 10, .8]])
    # Score-greedy AP gets one TP, while maximum-cardinality gets two.
    require(ap50_for_class(predictions, targets, ["a"], 0)["score_ranked_tp"] == 1,
            "AP must preserve score-ranked matching")
    groups = [{"key": "a", "pair": "synthetic"}]
    default, matched, _ = default_output_report({"a": predictions["a"]}, {"a": targets["a"]}, groups)
    require(default["overall"]["tp"] == 2 and len(matched) == 2, "default cardinality differs from AP")
    recovery = recovery_counts({("a", 0)}, {("a", 1)}, {("a", 0), ("a", 1)})
    require(recovery["recovered_native_missed"] == recovery["lost_native_tp"] == 1,
            "recovered/lost matching-set accounting")
    degenerate = validate_prediction_payload(
        {"a": {"0": [[0, 0, 0, 10, .8]], "1": [], "2": []}}, ["a"])
    zero, _, _ = default_output_report(degenerate, {"a": targets["a"]}, groups)
    require(zero["overall"]["fp"] == 1 and zero["overall"]["tp"] == 0,
            "zero-extent emitted prediction must remain a false positive")
    print(json.dumps(dict(status="PASS_SYNTHETIC_DETECTION_SCORE_CONTRACTS", graph_cases=checked,
                          raw_gt_read=False, predictions_read=False, artifacts_read=False,
                          gpu_used=False), sort_keys=True))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--receipt", type=Path, default=HERE / "PREDICTION_RECEIPT.json")
    cli.add_argument("--output", type=Path, default=HERE / "detection_readout")
    cli.add_argument("--self-test", action="store_true")
    args = cli.parse_args()
    if args.self_test:
        run_self_tests()
        return
    sys.dont_write_bytecode = True
    scorer_sha256 = sha(__file__)
    gate = gate_frozen_predictions(args.receipt)
    require(not args.output.exists() or not any(args.output.iterdir()),
            "readout output must be new or empty; existing evidence is not overwritten")
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "PRE_LABEL_GATE.json",
               dict(status="PASS_FOUR_ARM_PREDICTION_FREEZE_GATE", gated_at=gate["gated_at"],
                    receipt_path=str(gate["receipt_path"]), receipt_sha256=gate["receipt_sha256"],
                    prediction_paths=gate["prediction_paths"], checkpoint_paths=gate["checkpoint_paths"],
                    verified_bindings={str(p): v for p, v in gate["bindings"].items()},
                    labels_read=False, scorer_sha256=scorer_sha256))
    write_json(args.output / "LABEL_ACCESS_STARTED.json",
               dict(at=utc(), prediction_receipt_sha256=gate["receipt_sha256"],
                    scope="audited calibration-pair XML parsing; scoring only forty current anchors"))
    targets, label_bindings, excluded = load_calibration_targets(gate)
    results, matched, match_details = {}, {}, {}
    for arm in ARMS:
        default, matched[arm], match_details[arm] = default_output_report(
            gate["predictions"][arm], targets, gate["groups"])
        results[arm] = dict(ap50=ap_report(gate["predictions"][arm], targets, gate["groups"]),
                            default_output=default)
    for arm in ARMS:
        results[arm]["relative_to_native"] = recovery_report(matched["native"], matched[arm],
                                                               targets, gate["groups"])
    verify_bindings(gate["bindings"])
    verify_bindings(label_bindings)
    require(sha(gate["receipt_path"]) == gate["receipt_sha256"], "prediction receipt changed during scoring")
    require(sha(__file__) == scorer_sha256, "scorer source changed during scoring")
    summary = dict(status="COMPLETE_FROZEN_CALIBRATION_DETECTION_READOUT", group_count=40,
        pairs=sorted({str(g["pair"]) for g in gate["groups"]}), arms=results,
        class_names={str(k): v for k, v in CLASSES.items()}, unsupported_gt_excluded_by_image=excluded,
        protocol=dict(coordinates="native continuous xyxy; no +1 area", iou_threshold=.5,
            confidence_filter="none added; evaluate every frozen output row",
            ap="score-ranked per-class one-to-one greedy IoU>=.5; 101-point precision interpolation",
            ap_ties="descending score, lexicographic group key, original row index; GT row order fixed",
            ap_empty_gt="class/pair AP null if no GT; exclude only null class/pair means; FP in empty images retained",
            pair_macro="mean within-pair eligible-class AP50, then mean over eligible pairs",
            default_counts="maximum cardinality, then maximum total IoU; per image/class, no score reordering",
            degenerate_predictions="zero extent retained with IoU zero; nonfinite/inverted rows fail validation",
            size="GT original-coordinate sqrt(area)<16 vs >=16, not min-side width",
            occlusion="original audited XML occluded field, outside=1 excluded by parser",
            strata="GT strata report TP/FN/recall only; unmatched FP have no GT occlusion label",
            recovery="matched GT-node set difference vs native; maximum-cardinality tie-dependent diagnostic, not identity metric",
            scale="AP/recall/precision in [0,1]", scope="calibration detector readout only; no MDA/IDF1/MOTA"))
    write_json(args.output / "SUMMARY.json", summary)
    write_json(args.output / "DEFAULT_MATCHES.json", match_details)
    artifacts = {str(p.resolve()): sha(p) for p in args.output.iterdir() if p.is_file()}
    write_json(args.output / "RECEIPT.json",
               dict(status=summary["status"], completed_at=utc(), group_count=40,
                    prediction_receipt_path=str(gate["receipt_path"]),
                    prediction_receipt_sha256=gate["receipt_sha256"],
                    labels_read_by_prediction_producer=False, labels_read_by_evaluator=True,
                    label_bindings={str(p): v for p, v in label_bindings.items()},
                    scorer_path=str(Path(__file__).resolve()), scorer_sha256=scorer_sha256,
                    outputs=artifacts, gpu_used=False, official_val_or_test_read=False,
                    model_or_threshold_selection_performed=False))
    print(json.dumps({arm: dict(pair_macro_ap50=results[arm]["ap50"]["pair_macro_ap50"],
                               **results[arm]["default_output"]["overall"]) for arm in ARMS},
                     sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
