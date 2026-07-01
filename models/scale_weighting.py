"""Conservative size-aware scale weighting for FPN-like feature maps.

This module is repo-local and designed for inspection, ablation planning, and
synthetic-only verification. It does not load checkpoints, datasets, or any
Ultralytics internals.
"""

from dataclasses import dataclass, field
import math
from typing import Mapping, Sequence

import torch

SCALE_ORDER = ("P2", "P3", "P4", "P5")
SIZE_GROUP_ORDER = ("small", "medium", "large", "unknown")


@dataclass(frozen=True)
class SizeThresholds:
    """Pixel-area thresholds for conservative object-size grouping."""

    small_max_area: float = 32.0 * 32.0
    medium_max_area: float = 96.0 * 96.0

    def classify_area(self, pixel_area: float) -> str:
        if not math.isfinite(pixel_area) or pixel_area <= 0.0:
            return "unknown"
        if pixel_area < self.small_max_area:
            return "small"
        if pixel_area < self.medium_max_area:
            return "medium"
        return "large"


@dataclass(frozen=True)
class ScaleWeightingPolicy:
    """Validated conservative weights for each size group and FPN scale."""

    supported_scales: tuple[str, ...] = SCALE_ORDER
    size_thresholds: SizeThresholds = field(default_factory=SizeThresholds)
    scale_weights: dict[str, dict[str, float]] = field(
        default_factory=lambda: {
            "small": {"P2": 1.05, "P3": 1.15, "P4": 0.95, "P5": 0.90},
            "medium": {"P2": 0.90, "P3": 1.00, "P4": 1.10, "P5": 1.15},
            "large": {"P2": 0.90, "P3": 0.95, "P4": 1.10, "P5": 1.15},
            "unknown": {"P2": 1.00, "P3": 1.00, "P4": 1.00, "P5": 1.00},
        }
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "supported_scales", tuple(self.supported_scales))
        if self.supported_scales != SCALE_ORDER:
            raise ValueError(f"supported_scales must be exactly {SCALE_ORDER}, got {self.supported_scales!r}.")

        required_groups = set(SIZE_GROUP_ORDER)
        actual_groups = set(self.scale_weights)
        if actual_groups != required_groups:
            raise ValueError(f"scale_weights must define {sorted(required_groups)}, got {sorted(actual_groups)}.")

        for group_name, group_weights in self.scale_weights.items():
            if set(group_weights) != set(self.supported_scales):
                raise ValueError(
                    f"{group_name} weights must define {list(self.supported_scales)}, got {sorted(group_weights)}."
                )
            for scale_name, weight in group_weights.items():
                if not math.isfinite(weight) or weight <= 0.0:
                    raise ValueError(f"Invalid weight for {group_name}/{scale_name}: {weight!r}.")

    def weights_for_group(self, size_group: str | None) -> dict[str, float]:
        resolved_group = size_group if size_group in self.scale_weights else "unknown"
        return dict(self.scale_weights[resolved_group])


@dataclass(frozen=True)
class RuntimeIntegrationAdapter:
    """Declarative adapter boundary for future trainer-time wiring.

    This object intentionally does not patch Ultralytics or execute any runtime
    hook. It only records the expected integration contract so callers can fail
    closed until the real trainer/FPN insertion point exists.
    """

    method_name: str
    supported_scales: tuple[str, ...]
    size_groups: tuple[str, ...]
    required_hook_point: str = "after multi-scale feature assembly and before Detect consumes P2/P3/P4/P5"
    requires_size_metadata: bool = True
    identity_fallback_group: str = "unknown"
    runtime_integration_verified: bool = False


def policy_from_config_payload(payload: Mapping[str, object]) -> ScaleWeightingPolicy:
    """Build and validate a policy from a parsed method-config payload."""

    supported_scales = payload.get("supported_scales")
    size_groups = payload.get("size_groups")
    size_thresholds = payload.get("size_thresholds")
    scale_weights = payload.get("scale_weights")

    if not isinstance(supported_scales, Sequence) or isinstance(supported_scales, (str, bytes)):
        raise ValueError("supported_scales must be a sequence of scale names.")
    if not isinstance(size_groups, Sequence) or isinstance(size_groups, (str, bytes)):
        raise ValueError("size_groups must be a sequence of group names.")
    if not isinstance(size_thresholds, Mapping):
        raise ValueError("size_thresholds must be a mapping.")
    if not isinstance(scale_weights, Mapping):
        raise ValueError("scale_weights must be a mapping.")

    small_max = size_thresholds.get("small_max_pixel_area")
    medium_max = size_thresholds.get("medium_max_pixel_area")
    if small_max is None or medium_max is None:
        raise ValueError("size_thresholds must include small_max_pixel_area and medium_max_pixel_area.")

    normalized_weights: dict[str, dict[str, float]] = {}
    for group_name, group_weights in scale_weights.items():
        if not isinstance(group_name, str):
            raise ValueError("scale_weights keys must be strings.")
        if not isinstance(group_weights, Mapping):
            raise ValueError(f"scale_weights[{group_name!r}] must be a mapping.")
        normalized_weights[group_name] = {}
        for scale_name, weight in group_weights.items():
            if not isinstance(scale_name, str):
                raise ValueError(f"scale name for group {group_name!r} must be a string.")
            normalized_weights[group_name][scale_name] = float(weight)

    return ScaleWeightingPolicy(
        supported_scales=tuple(str(scale_name) for scale_name in supported_scales),
        size_thresholds=SizeThresholds(
            small_max_area=float(small_max),
            medium_max_area=float(medium_max),
        ),
        scale_weights=normalized_weights,
    )


def build_runtime_integration_adapter(
    method_name: str,
    policy: ScaleWeightingPolicy | None = None,
) -> RuntimeIntegrationAdapter:
    """Describe the expected future trainer-time integration boundary."""

    resolved_policy = policy or ScaleWeightingPolicy()
    return RuntimeIntegrationAdapter(
        method_name=method_name,
        supported_scales=resolved_policy.supported_scales,
        size_groups=tuple(SIZE_GROUP_ORDER),
    )


def yolo_box_to_pixel_area(
    width_norm: float,
    height_norm: float,
    image_width: int | float,
    image_height: int | float,
) -> float | None:
    """Convert normalized YOLO box width/height into pixel area."""

    values = (width_norm, height_norm, image_width, image_height)
    if any(not math.isfinite(float(value)) for value in values):
        return None
    if width_norm <= 0.0 or height_norm <= 0.0 or image_width <= 0.0 or image_height <= 0.0:
        return None
    return float(width_norm) * float(height_norm) * float(image_width) * float(image_height)


def classify_yolo_box_size_group(
    width_norm: float,
    height_norm: float,
    image_width: int | float,
    image_height: int | float,
    thresholds: SizeThresholds | None = None,
) -> str:
    """Assign a YOLO-format box to small/medium/large/unknown."""

    resolved_thresholds = thresholds or SizeThresholds()
    pixel_area = yolo_box_to_pixel_area(width_norm, height_norm, image_width, image_height)
    if pixel_area is None:
        return "unknown"
    return resolved_thresholds.classify_area(pixel_area)


def resolve_group_scale_weights(
    size_group: str | None,
    policy: ScaleWeightingPolicy | None = None,
) -> dict[str, float]:
    """Return validated per-scale weights for a size group."""

    resolved_policy = policy or ScaleWeightingPolicy()
    return resolved_policy.weights_for_group(size_group)


class SizeAwareScaleWeighter:
    """Apply scalar per-scale weights without changing tensor shapes."""

    def __init__(self, policy: ScaleWeightingPolicy | None = None) -> None:
        self.policy = policy or ScaleWeightingPolicy()

    def apply(
        self,
        features: Mapping[str, torch.Tensor] | Sequence[torch.Tensor],
        *,
        size_group: str | None = None,
    ) -> Mapping[str, torch.Tensor] | list[torch.Tensor]:
        scale_weights = resolve_group_scale_weights(size_group, self.policy)
        if isinstance(features, Mapping):
            self._validate_feature_mapping(features)
            weighted = {}
            for scale_name, tensor in features.items():
                if scale_name in scale_weights:
                    weighted[scale_name] = self._weight_tensor(tensor, scale_weights[scale_name], scale_name)
                else:
                    weighted[scale_name] = tensor
            return weighted

        if len(features) != len(self.policy.supported_scales):
            raise ValueError(
                "Feature sequence must contain exactly "
                f"{len(self.policy.supported_scales)} tensors ordered as {self.policy.supported_scales}."
            )
        weighted_list: list[torch.Tensor] = []
        for scale_name, tensor in zip(self.policy.supported_scales, features):
            weighted_list.append(self._weight_tensor(tensor, scale_weights[scale_name], scale_name))
        return weighted_list

    def _validate_feature_mapping(self, features: Mapping[str, torch.Tensor]) -> None:
        missing = [scale for scale in self.policy.supported_scales if scale not in features]
        if missing:
            raise ValueError(f"Feature mapping missing required scales: {missing}.")

    def _weight_tensor(self, tensor: torch.Tensor, weight: float, scale_name: str) -> torch.Tensor:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Feature for {scale_name} must be a torch.Tensor.")
        if tensor.ndim < 2:
            raise ValueError(f"Feature for {scale_name} must have at least 2 dimensions, got {tensor.shape!r}.")
        weighted = tensor * weight
        if weighted.shape != tensor.shape:
            raise RuntimeError(f"Weighted tensor shape changed for {scale_name}: {tensor.shape!r} -> {weighted.shape!r}.")
        return weighted


def identity_or_group_weights(
    size_group: str | None,
    has_valid_size_information: bool,
    policy: ScaleWeightingPolicy | None = None,
) -> dict[str, float]:
    """Return identity weights when size evidence is unavailable."""

    resolved_policy = policy or ScaleWeightingPolicy()
    if not has_valid_size_information:
        return resolved_policy.weights_for_group("unknown")
    return resolved_policy.weights_for_group(size_group)
