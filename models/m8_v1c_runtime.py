"""Repo-local M8_v1c runtime integration for size-aware FPN scale weighting.

This module keeps the runtime hook local to the repo by subclassing the
Ultralytics detection model and trainer rather than editing site-packages.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import RANK

from models.scale_weighting import SCALE_ORDER, SizeThresholds, classify_yolo_box_size_group

SUPPORTED_SIZE_GROUPS = ("small", "medium", "large", "unknown")


@dataclass(frozen=True)
class RuntimePolicyGroup:
    """One size-group policy row for M8_v1c runtime weighting."""

    mode: str
    preferred_scale: str | None
    weight_delta: dict[str, float]


@dataclass(frozen=True)
class M8V1CRuntimePolicy:
    """Validated runtime policy derived from the mined M8_v1c policy candidate."""

    method_name: str
    supported_scales: tuple[str, ...] = SCALE_ORDER
    size_thresholds: SizeThresholds = field(default_factory=SizeThresholds)
    groups: dict[str, RuntimePolicyGroup] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "supported_scales", tuple(self.supported_scales))
        if self.supported_scales != SCALE_ORDER:
            raise ValueError(f"supported_scales must be exactly {SCALE_ORDER}, got {self.supported_scales!r}.")

        actual_groups = set(self.groups)
        expected_groups = set(SUPPORTED_SIZE_GROUPS)
        if actual_groups != expected_groups:
            raise ValueError(f"groups must define {sorted(expected_groups)}, got {sorted(actual_groups)}.")

        for group_name, group in self.groups.items():
            if group.mode not in {"preferred_scale", "identity"}:
                raise ValueError(f"{group_name} mode must be 'preferred_scale' or 'identity', got {group.mode!r}.")
            if set(group.weight_delta) != set(self.supported_scales):
                raise ValueError(
                    f"{group_name} weight_delta must define {list(self.supported_scales)}, got {sorted(group.weight_delta)}."
                )
            for scale_name, weight in group.weight_delta.items():
                if not math.isfinite(weight) or weight <= 0.0:
                    raise ValueError(f"Invalid weight for {group_name}/{scale_name}: {weight!r}.")
            if group.mode == "identity":
                for scale_name, weight in group.weight_delta.items():
                    if abs(weight - 1.0) > 1e-9:
                        raise ValueError(
                            f"{group_name} uses identity mode, but {scale_name} has non-identity weight {weight!r}."
                        )
            if group.mode == "preferred_scale":
                if group.preferred_scale not in self.supported_scales:
                    raise ValueError(
                        f"{group_name} preferred_scale must be one of {self.supported_scales}, "
                        f"got {group.preferred_scale!r}."
                    )
                resolved_max = max(group.weight_delta, key=group.weight_delta.get)
                if resolved_max != group.preferred_scale:
                    raise ValueError(
                        f"{group_name} preferred_scale={group.preferred_scale!r} but highest weight resolves to "
                        f"{resolved_max!r}."
                    )

    def weights_for_group(self, size_group: str | None) -> dict[str, float]:
        resolved_group = size_group if size_group in self.groups else "unknown"
        return dict(self.groups[resolved_group].weight_delta)


def _identity_weights() -> dict[str, float]:
    return {scale_name: 1.0 for scale_name in SCALE_ORDER}


def load_yaml_mapping(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping at {path}")
    return payload


def runtime_policy_from_config_payload(payload: Mapping[str, Any]) -> M8V1CRuntimePolicy:
    method_name = str(payload.get("method_name", "m8_v1c_runtime_policy"))
    supported_scales = payload.get("supported_scales")
    size_thresholds = payload.get("size_thresholds")
    size_aware_policy = payload.get("size_aware_policy")

    if not isinstance(supported_scales, Sequence) or isinstance(supported_scales, (str, bytes)):
        raise ValueError("supported_scales must be a sequence.")
    if not isinstance(size_thresholds, Mapping):
        raise ValueError("size_thresholds must be a mapping.")
    if not isinstance(size_aware_policy, Mapping):
        raise ValueError("size_aware_policy must be a mapping.")

    small_max = size_thresholds.get("small_max_pixel_area")
    medium_max = size_thresholds.get("medium_max_pixel_area")
    if small_max is None or medium_max is None:
        raise ValueError("size_thresholds must include small_max_pixel_area and medium_max_pixel_area.")

    groups: dict[str, RuntimePolicyGroup] = {}
    for group_name in SUPPORTED_SIZE_GROUPS:
        raw_group = size_aware_policy.get(group_name)
        if not isinstance(raw_group, Mapping):
            raise ValueError(f"size_aware_policy[{group_name!r}] must be a mapping.")
        mode = str(raw_group.get("mode", "identity"))
        preferred_scale = raw_group.get("preferred_scale")
        if preferred_scale is not None:
            preferred_scale = str(preferred_scale)

        raw_weight_delta = raw_group.get("weight_delta", _identity_weights())
        if not isinstance(raw_weight_delta, Mapping):
            raise ValueError(f"size_aware_policy[{group_name!r}].weight_delta must be a mapping.")
        weight_delta = {str(scale): float(weight) for scale, weight in raw_weight_delta.items()}
        groups[group_name] = RuntimePolicyGroup(
            mode=mode,
            preferred_scale=preferred_scale,
            weight_delta=weight_delta,
        )

    return M8V1CRuntimePolicy(
        method_name=method_name,
        supported_scales=tuple(str(scale_name) for scale_name in supported_scales),
        size_thresholds=SizeThresholds(
            small_max_area=float(small_max),
            medium_max_area=float(medium_max),
        ),
        groups=groups,
    )


def runtime_policy_from_config_path(path: Path) -> M8V1CRuntimePolicy:
    return runtime_policy_from_config_payload(load_yaml_mapping(path))


def infer_image_size_groups(
    batch: Mapping[str, torch.Tensor],
    thresholds: SizeThresholds,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Resolve one conservative size group per image from YOLO-format GT boxes."""

    images = batch.get("img")
    batch_idx = batch.get("batch_idx")
    bboxes = batch.get("bboxes")
    if not isinstance(images, torch.Tensor) or not isinstance(batch_idx, torch.Tensor) or not isinstance(bboxes, torch.Tensor):
        return ["unknown"], [{"reason": "missing batch tensors"}]

    batch_size = int(images.shape[0])
    image_height = int(images.shape[2])
    image_width = int(images.shape[3])
    counters = [Counter() for _ in range(batch_size)]

    flat_batch_idx = batch_idx.view(-1).detach().cpu()
    flat_bboxes = bboxes.detach().cpu()
    for sample_index, bbox in zip(flat_batch_idx.tolist(), flat_bboxes.tolist(), strict=False):
        if not isinstance(sample_index, int) or sample_index < 0 or sample_index >= batch_size:
            continue
        if len(bbox) != 4:
            continue
        width_norm = float(bbox[2])
        height_norm = float(bbox[3])
        group_name = classify_yolo_box_size_group(width_norm, height_norm, image_width, image_height, thresholds)
        counters[sample_index][group_name] += 1

    resolved_groups: list[str] = []
    debug_rows: list[dict[str, Any]] = []
    for sample_index, counter in enumerate(counters):
        if not counter:
            resolved_groups.append("unknown")
            debug_rows.append({"sample_index": sample_index, "reason": "no_boxes", "counts": {}})
            continue

        ranked = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
        best_group, best_count = ranked[0]
        second_count = ranked[1][1] if len(ranked) > 1 else -1
        if best_group == "unknown" or best_count == second_count:
            resolved_group = "unknown"
            reason = "unknown_present_or_tied_counts"
        else:
            resolved_group = best_group
            reason = "dominant_group"
        resolved_groups.append(resolved_group if resolved_group in SUPPORTED_SIZE_GROUPS else "unknown")
        debug_rows.append(
            {
                "sample_index": sample_index,
                "reason": reason,
                "counts": dict(counter),
                "resolved_group": resolved_groups[-1],
            }
        )
    return resolved_groups, debug_rows


def _expand_scale_weights(
    tensor: torch.Tensor,
    size_groups: Sequence[str],
    scale_name: str,
    policy: M8V1CRuntimePolicy,
) -> torch.Tensor:
    weights = [policy.weights_for_group(size_group).get(scale_name, 1.0) for size_group in size_groups]
    weight_tensor = torch.tensor(weights, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1, 1)
    if weight_tensor.shape[0] != tensor.shape[0]:
        raise ValueError(
            f"Per-sample scale-weight tensor batch size mismatch for {scale_name}: "
            f"{weight_tensor.shape[0]} vs feature batch {tensor.shape[0]}."
        )
    return weight_tensor


def apply_runtime_policy_to_features(
    features: Sequence[torch.Tensor],
    size_groups: Sequence[str],
    policy: M8V1CRuntimePolicy,
) -> list[torch.Tensor]:
    if len(features) != len(policy.supported_scales):
        raise ValueError(
            f"Expected {len(policy.supported_scales)} feature tensors ordered as {policy.supported_scales}, "
            f"got {len(features)} tensors."
        )
    weighted: list[torch.Tensor] = []
    for scale_name, tensor in zip(policy.supported_scales, features, strict=False):
        if tensor.ndim != 4:
            raise ValueError(f"Feature tensor for {scale_name} must be 4D, got {tuple(tensor.shape)!r}.")
        weight_tensor = _expand_scale_weights(tensor, size_groups, scale_name, policy)
        weighted.append(tensor * weight_tensor)
    return weighted


class M8V1CRuntimeDetectionModel(DetectionModel):
    """DetectionModel with a repo-local hook before Detect consumes P2/P3/P4/P5."""

    def __init__(
        self,
        cfg: str | dict[str, Any] = "yolo11n.yaml",
        ch: int = 3,
        nc: int | None = None,
        verbose: bool = True,
        *,
        runtime_policy: M8V1CRuntimePolicy,
    ) -> None:
        self.runtime_policy = runtime_policy
        self.runtime_method_integration_verified = False
        self.current_batch_size_groups: list[str] = []
        self.current_batch_debug: list[dict[str, Any]] = []
        self.last_runtime_application: dict[str, Any] = {
            "applied": False,
            "reason": "bootstrap_init",
        }
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)
        self.runtime_method_integration_verified = True
        self.last_runtime_application = {"applied": False, "reason": "not_run_yet"}

    def set_runtime_batch_metadata(self, batch: Mapping[str, torch.Tensor]) -> None:
        size_groups, debug_rows = infer_image_size_groups(batch, self.runtime_policy.size_thresholds)
        self.current_batch_size_groups = size_groups
        self.current_batch_debug = debug_rows

    def clear_runtime_batch_metadata(self) -> None:
        self.current_batch_size_groups = []
        self.current_batch_debug = []

    def _apply_policy_if_needed(self, features: Any, detect_layer: torch.nn.Module) -> Any:
        if not isinstance(features, list):
            self.last_runtime_application = {
                "applied": False,
                "reason": "detect_input_not_list",
                "detect_layer_index": getattr(detect_layer, "i", None),
            }
            return features
        if len(features) != len(self.runtime_policy.supported_scales):
            self.last_runtime_application = {
                "applied": False,
                "reason": "unexpected_feature_count",
                "detect_layer_index": getattr(detect_layer, "i", None),
                "feature_count": len(features),
            }
            return features

        batch_size = int(features[0].shape[0])
        size_groups = self.current_batch_size_groups or ["unknown"] * batch_size
        if len(size_groups) != batch_size:
            size_groups = ["unknown"] * batch_size

        weighted = apply_runtime_policy_to_features(features, size_groups, self.runtime_policy)
        self.last_runtime_application = {
            "applied": True,
            "reason": "runtime_scale_weighting_applied",
            "detect_layer_index": getattr(detect_layer, "i", None),
            "detect_input_from": getattr(detect_layer, "f", None),
            "size_groups": list(size_groups),
            "batch_debug": list(self.current_batch_debug),
            "supported_scales": list(self.runtime_policy.supported_scales),
        }
        return weighted

    def _predict_once(self, x, profile=False, visualize=False, embed=None):
        y, dt, embeddings = [], [], []
        embed = frozenset(embed) if embed is not None else {-1}
        max_idx = max(embed)
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            if profile:
                self._profile_one_layer(m, x, dt)
            if type(m).__name__ == "Detect":
                x = self._apply_policy_if_needed(x, m)
            x = m(x)
            y.append(x if m.i in self.save else None)
            if visualize:
                from ultralytics.utils.plotting import feature_visualization

                feature_visualization(x, m.type, m.i, save_dir=visualize)
            if m.i in embed:
                embeddings.append(torch.nn.functional.adaptive_avg_pool2d(x, (1, 1)).squeeze(-1).squeeze(-1))
                if m.i == max_idx:
                    return torch.unbind(torch.cat(embeddings, 1), dim=0)
        return x


class M8V1CPolicyMiningTrainer(DetectionTrainer):
    """DetectionTrainer subclass that wires the M8_v1c runtime policy into the model."""

    def __init__(self, cfg=None, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None):
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)
        method_config = getattr(self.args, "method_config", None)
        if not method_config:
            raise ValueError("M8V1CPolicyMiningTrainer requires args.method_config.")
        self.method_config_path = Path(str(method_config)).resolve()
        self.runtime_policy = runtime_policy_from_config_path(self.method_config_path)

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        model = M8V1CRuntimeDetectionModel(
            cfg or self.args.model,
            nc=self.data["nc"],
            ch=self.data["channels"],
            verbose=verbose and RANK == -1,
            runtime_policy=self.runtime_policy,
        )
        if weights:
            model.load(weights)
        return model

    def preprocess_batch(self, batch: dict) -> dict:
        batch = super().preprocess_batch(batch)
        model = self.model.module if hasattr(self.model, "module") else self.model
        if hasattr(model, "set_runtime_batch_metadata"):
            model.set_runtime_batch_metadata(batch)
        return batch
