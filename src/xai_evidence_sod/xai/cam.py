"""Lightweight post-hoc CAM extraction utilities for baseline YOLO analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


@dataclass(frozen=True)
class PreprocessedImage:
    """One preprocessed image plus geometry needed for CAM remapping."""

    image_path: str
    orig_width: int
    orig_height: int
    resized_width: int
    resized_height: int
    pad_left: int
    pad_top: int
    scale: float
    tensor: torch.Tensor
    original_rgb: np.ndarray
    input_size: int


@dataclass(frozen=True)
class CamResult:
    """One CAM output mapped back to original image space."""

    method: str
    target_layer: int
    cam_original: np.ndarray
    cam_model_input: np.ndarray


class BaseCamExtractor:
    """Small abstraction so later Grad-CAM variants can plug into the same pipeline."""

    method_name = "base"

    def __init__(self, model: Any, target_layer: int) -> None:
        self.model = model
        self.target_layer = target_layer
        self._activation: torch.Tensor | None = None

    def extract(self, prepared: PreprocessedImage) -> CamResult:
        raise NotImplementedError

    def _register_hook(self):
        layer = self.model.model[self.target_layer]

        def hook(_module, _inputs, output):
            activation = output[0] if isinstance(output, (tuple, list)) else output
            self._activation = activation.detach()

        return layer.register_forward_hook(hook)


class EigenCamExtractor(BaseCamExtractor):
    """Gradient-free EigenCAM over a chosen feature layer."""

    method_name = "eigencam"

    def extract(self, prepared: PreprocessedImage) -> CamResult:
        hook_handle = self._register_hook()
        self.model.eval()
        with torch.no_grad():
            _ = self.model(prepared.tensor)
        hook_handle.remove()
        if self._activation is None:
            raise RuntimeError("No activation was captured for EigenCAM extraction.")

        activation = self._activation[0].float().cpu()
        cam_input = _compute_eigencam(activation)
        cam_original = remap_cam_to_original(
            cam_input,
            orig_width=prepared.orig_width,
            orig_height=prepared.orig_height,
            resized_width=prepared.resized_width,
            resized_height=prepared.resized_height,
            pad_left=prepared.pad_left,
            pad_top=prepared.pad_top,
        )
        return CamResult(
            method=self.method_name,
            target_layer=self.target_layer,
            cam_original=cam_original,
            cam_model_input=cam_input,
        )


class PlaceholderGradCamExtractor(BaseCamExtractor):
    """Interface stub for future gradient-based CAM methods."""

    def extract(self, prepared: PreprocessedImage) -> CamResult:
        raise NotImplementedError(f"{self.method_name} is not implemented in Phase 5. Use eigencam for now.")


class GradCamExtractor(PlaceholderGradCamExtractor):
    method_name = "gradcam"


class GradCamPlusPlusExtractor(PlaceholderGradCamExtractor):
    method_name = "gradcam++"


def build_cam_extractor(method: str, model: Any, target_layer: int) -> BaseCamExtractor:
    """Factory for supported CAM methods."""

    normalized = method.strip().lower()
    if normalized == "eigencam":
        return EigenCamExtractor(model=model, target_layer=target_layer)
    if normalized == "gradcam":
        return GradCamExtractor(model=model, target_layer=target_layer)
    if normalized == "gradcam++":
        return GradCamPlusPlusExtractor(model=model, target_layer=target_layer)
    raise ValueError(f"Unsupported CAM method: {method}")


def preprocess_image_for_yolo(image_path: str | Path, imgsz: int, device: str | torch.device) -> PreprocessedImage:
    """Load one image, letterbox it to a square canvas, and create a model tensor."""

    image = Image.open(image_path).convert("RGB")
    original_rgb = np.asarray(image)
    orig_width, orig_height = image.size

    scale = min(imgsz / orig_width, imgsz / orig_height)
    resized_width = max(1, int(round(orig_width * scale)))
    resized_height = max(1, int(round(orig_height * scale)))
    resized = image.resize((resized_width, resized_height), Image.BILINEAR)

    pad_left = (imgsz - resized_width) // 2
    pad_top = (imgsz - resized_height) // 2
    canvas = Image.new("RGB", (imgsz, imgsz), color=(114, 114, 114))
    canvas.paste(resized, (pad_left, pad_top))

    tensor = torch.from_numpy(np.array(canvas, copy=True).transpose(2, 0, 1)).float() / 255.0
    tensor = tensor.unsqueeze(0).to(device)

    return PreprocessedImage(
        image_path=str(Path(image_path).expanduser().resolve()),
        orig_width=orig_width,
        orig_height=orig_height,
        resized_width=resized_width,
        resized_height=resized_height,
        pad_left=pad_left,
        pad_top=pad_top,
        scale=scale,
        tensor=tensor,
        original_rgb=original_rgb,
        input_size=imgsz,
    )


def remap_cam_to_original(
    cam_input: np.ndarray,
    orig_width: int,
    orig_height: int,
    resized_width: int,
    resized_height: int,
    pad_left: int,
    pad_top: int,
) -> np.ndarray:
    """Upsample CAM to model input, remove padding, and resize back to original image space."""

    cam_tensor = torch.from_numpy(cam_input).float().unsqueeze(0).unsqueeze(0)
    upsampled = F.interpolate(
        cam_tensor,
        size=(max(pad_top * 2 + resized_height, cam_input.shape[0]), max(pad_left * 2 + resized_width, cam_input.shape[1])),
        mode="bilinear",
        align_corners=False,
    )[0, 0].numpy()
    cropped = upsampled[pad_top : pad_top + resized_height, pad_left : pad_left + resized_width]
    cropped_tensor = torch.from_numpy(cropped).float().unsqueeze(0).unsqueeze(0)
    restored = F.interpolate(cropped_tensor, size=(orig_height, orig_width), mode="bilinear", align_corners=False)[0, 0].numpy()
    return _normalize_map(restored)


def _compute_eigencam(activation: torch.Tensor) -> np.ndarray:
    channels, height, width = activation.shape
    matrix = activation.reshape(channels, height * width).transpose(0, 1).numpy()
    matrix = matrix - matrix.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(matrix, full_matrices=False)
    principal = matrix @ vh[0]
    cam = principal.reshape(height, width)
    if cam.mean() < 0:
        cam = -cam
    cam = np.maximum(cam, 0.0)
    return _normalize_map(cam)


def _normalize_map(cam: np.ndarray) -> np.ndarray:
    cam = cam.astype(np.float32)
    cam_min = float(cam.min())
    cam_max = float(cam.max())
    if cam_max <= cam_min:
        return np.zeros_like(cam, dtype=np.float32)
    return (cam - cam_min) / (cam_max - cam_min)
