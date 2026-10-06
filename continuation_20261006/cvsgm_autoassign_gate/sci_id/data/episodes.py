"""Deterministic MDMT train episode loading with frame-level image packing."""
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


LEVELS = ("p2_id", "p3_id", "p4_id")


def _load_jsonl_gz(path: Path) -> List[dict]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_source_line"] = line_number
            rows.append(row)
    return rows


def _resize_to_tensor(
    path: Path,
    image_size: Tuple[int, int],
    mean: Tensor,
    std: Tensor,
    value_scale: float = 255.0,
    channel_order: str = "rgb",
) -> Tuple[Tensor, Tuple[int, int]]:
    with Image.open(path) as image:
        image = image.convert("RGB")
        original_width, original_height = image.size
        image = image.resize((image_size[1], image_size[0]), resample=Image.BILINEAR)
        array = np.array(image, dtype=np.uint8, copy=True)
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous().float()
    if channel_order == "bgr":
        tensor = tensor[[2, 1, 0]]
    elif channel_order != "rgb":
        raise ValueError("channel_order must be 'rgb' or 'bgr'")
    if value_scale <= 0:
        raise ValueError("value_scale must be positive")
    tensor = tensor.div_(value_scale)
    tensor = (tensor - mean[:, None, None]) / std[:, None, None]
    return tensor, (original_width, original_height)


def _normalise_box(box: Sequence[float], original_size: Tuple[int, int]) -> List[float]:
    width, height = original_size
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    x1, y1, x2, y2 = [float(value) for value in box]
    values = [x1 / width, y1 / height, x2 / width, y2 / height]
    values = [min(1.0, max(0.0, value)) for value in values]
    if values[2] < values[0] or values[3] < values[1]:
        raise ValueError(f"invalid xyxy box {box!r}")
    return values


class PackedFrameEpisodeDataset(Dataset):
    """Index frozen train episodes without touching val/test.

    The dataset only reads ``data/train_episodes_v1`` shards. Image decoding is
    intentionally deferred to ``collate_packed_frame_episodes`` so multiple
    episodes referring to one frame are decoded once per batch.
    """

    def __init__(self, root: Union[str, Path], pairs: Optional[Sequence[str]] = None) -> None:
        self.root = Path(root).resolve()
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"episode manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("split") != "train" or manifest.get("test_access"):
            raise ValueError("only train-only episode manifests are accepted")
        selected = set(str(pair) for pair in pairs) if pairs is not None else None
        self.rows: List[dict] = []
        self.shards: List[str] = []
        for shard in manifest.get("shards", []):
            pair = str(shard["pair_id"])
            if selected is not None and pair not in selected:
                continue
            path = Path(shard["output"]).resolve()
            if self.root not in path.parents:
                raise ValueError("episode shard escapes the selected train root")
            self.shards.append(str(path))
            self.rows.extend(_load_jsonl_gz(path))
        if selected is not None and set(self.shard_pairs()) != selected:
            missing = sorted(selected - set(self.shard_pairs()))
            if missing:
                raise ValueError(f"requested pairs are absent from manifest: {missing}")
        if not self.rows:
            raise ValueError("episode dataset is empty")

    def shard_pairs(self) -> List[str]:
        return [Path(path).stem.split(".")[0] for path in self.shards]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = dict(self.rows[index])
        row.pop("_source_line", None)
        return row


def _validate_episode(episode: Mapping[str, object]) -> None:
    if episode.get("split") != "train":
        raise ValueError("collator accepts train episodes only")
    if bool(episode.get("supervision_valid")) != (episode.get("target_train_gt_id") is not None):
        raise ValueError("supervision_valid disagrees with target_train_gt_id")
    candidates = episode.get("candidates")
    if not isinstance(candidates, list) or len(candidates) > 64:
        raise ValueError("candidate list must have length in [0,64]")
    if int(episode.get("candidate_count", -1)) != len(candidates):
        raise ValueError("candidate_count disagrees with candidates")
    candidate_count = len(candidates)
    for key in (
        "same_id_candidate_indices",
        "known_negative_indices",
        "unmapped_candidate_indices",
        "remove_all_positive_indices",
        "same_cardinality_remove_negative_indices",
    ):
        indices = episode.get(key, [])
        if any(not isinstance(index, int) or index < 0 or index >= candidate_count for index in indices):
            raise ValueError(f"{key} has an out-of-range index")
    positives = set(episode.get("same_id_candidate_indices", []))
    remove_all = set(episode.get("remove_all_positive_indices", []))
    if positives != remove_all:
        raise ValueError("remove_all_positive_indices must equal same-ID candidates")
    if len(episode.get("same_cardinality_remove_negative_indices", [])) not in (0, len(positives)):
        raise ValueError("remove-negative control must match positive cardinality")


def collate_packed_frame_episodes(
    episodes: Sequence[Mapping[str, object]],
    image_size: Tuple[int, int] = (256, 448),
    mean: Sequence[float] = (0.485, 0.456, 0.406),
    std: Sequence[float] = (0.229, 0.224, 0.225),
    value_scale: float = 255.0,
    channel_order: str = "rgb",
) -> Dict[str, object]:
    """Decode each unique frame once and pack variable-K episodes.

    ``frame_images`` is ``[F,3,H,W]`` and indices map each episode's target
    and source view into that tensor. Candidate boxes are normalized against
    each source image's original dimensions, matching the RoIAlign contract.
    ``factual_target`` is a bool multi-positive target with a final dustbin
    column; rows with an unmapped target are marked ``supervision_valid=False``
    and contain no target rather than being mislabeled as no-match.
    """
    if not episodes:
        raise ValueError("cannot collate an empty episode batch")
    height, width = [int(value) for value in image_size]
    if height < 32 or width < 32:
        raise ValueError("image_size is too small for the ResNet stride")
    mean_tensor = torch.tensor(list(mean), dtype=torch.float32)
    std_tensor = torch.tensor(list(std), dtype=torch.float32)
    if mean_tensor.shape != (3,) or std_tensor.shape != (3,) or bool(torch.any(std_tensor <= 0)):
        raise ValueError("mean/std must be positive RGB triples")

    unique_paths: List[str] = []
    path_to_index: Dict[str, int] = {}
    for episode in episodes:
        _validate_episode(episode)
        for key in ("target_image", "source_image"):
            path = str(Path(str(episode[key])).resolve())
            if path not in path_to_index:
                path_to_index[path] = len(unique_paths)
                unique_paths.append(path)

    frame_images = []
    original_sizes = []
    for path in unique_paths:
        image, original_size = _resize_to_tensor(
            Path(path), (height, width), mean_tensor, std_tensor,
            value_scale=value_scale, channel_order=channel_order
        )
        frame_images.append(image)
        original_sizes.append(original_size)
    frame_images_tensor = torch.stack(frame_images, dim=0)

    batch_size = len(episodes)
    max_k = max(len(episode["candidates"]) for episode in episodes)
    tensor_k = max(1, max_k)
    target_boxes = torch.zeros(batch_size, 4, dtype=torch.float32)
    candidate_boxes = torch.zeros(batch_size, tensor_k, 4, dtype=torch.float32)
    mask = torch.zeros(batch_size, tensor_k, dtype=torch.bool)
    positive_mask = torch.zeros(batch_size, tensor_k, dtype=torch.bool)
    known_negative_mask = torch.zeros(batch_size, tensor_k, dtype=torch.bool)
    remove_control_mask = torch.zeros(batch_size, tensor_k, dtype=torch.bool)
    factual_target = torch.zeros(batch_size, tensor_k + 1, dtype=torch.bool)
    supervision_valid = torch.zeros(batch_size, dtype=torch.bool)
    target_frame_index = torch.empty(batch_size, dtype=torch.long)
    source_frame_index = torch.empty(batch_size, dtype=torch.long)
    episode_ids, pair_ids = [], []
    for batch_index, episode in enumerate(episodes):
        target_path = str(Path(str(episode["target_image"])).resolve())
        source_path = str(Path(str(episode["source_image"])).resolve())
        target_frame_index[batch_index] = path_to_index[target_path]
        source_frame_index[batch_index] = path_to_index[source_path]
        target_size = original_sizes[target_frame_index[batch_index].item()]
        source_size = original_sizes[source_frame_index[batch_index].item()]
        target_boxes[batch_index] = torch.tensor(_normalise_box(episode["target_bbox_xyxy"], target_size))
        candidates = episode["candidates"]
        for candidate_index, candidate in enumerate(candidates):
            candidate_boxes[batch_index, candidate_index] = torch.tensor(
                _normalise_box(candidate["bbox_xyxy"], source_size)
            )
        count = len(candidates)
        if count:
            mask[batch_index, :count] = True
        positives = list(episode.get("same_id_candidate_indices", []))
        negatives = list(episode.get("known_negative_indices", []))
        controls = list(episode.get("same_cardinality_remove_negative_indices", []))
        if positives:
            positive_mask[batch_index, positives] = True
        if negatives:
            known_negative_mask[batch_index, negatives] = True
        if controls:
            remove_control_mask[batch_index, controls] = True
        valid = bool(episode["supervision_valid"])
        supervision_valid[batch_index] = valid
        if valid:
            if positives:
                factual_target[batch_index, positives] = True
            else:
                factual_target[batch_index, tensor_k] = True
        episode_ids.append(str(episode["episode_id"]))
        pair_ids.append(str(episode["pair_id"]))

    return {
        "frame_images": frame_images_tensor,
        "frame_paths": unique_paths,
        "target_frame_index": target_frame_index,
        "source_frame_index": source_frame_index,
        "target_boxes": target_boxes,
        "candidate_boxes": candidate_boxes,
        "mask": mask,
        "positive_mask": positive_mask,
        "known_negative_mask": known_negative_mask,
        "remove_control_mask": remove_control_mask,
        "factual_target": factual_target,
        "supervision_valid": supervision_valid,
        "episode_ids": episode_ids,
        "pair_ids": pair_ids,
        "image_size": (height, width),
        "normalization": {"mean": list(mean), "std": list(std)},
        "value_scale": float(value_scale),
        "channel_order": channel_order,
    }
