"""Switchable frozen RGB-D feature backbones for the reactive controller.

The async worker caches frozen-backbone tokens. Channel adaptation deliberately
stays in ``RGBDFusion`` (LayerNorm + Linear(C, 128)), where BC/PPO can train it.
"""
from __future__ import annotations

import rexnavdp  # noqa: F401  (sys.path bootstrap)

from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping
import hashlib
import importlib
import sys

import cv2
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


DAV2_PREPROCESS_VERSION = "xnavdp_eval_rgbd_224_v1"
YOLO_RGB_PREPROCESS_VERSION = "ultralytics_rgb_letterbox_224_div255_v1"
YOLO_DEPTH_PREPROCESS_VERSION = "metric_depth_0p1_5m_div5_letterbox_v1"
PREPROCESS_VERSION = DAV2_PREPROCESS_VERSION  # legacy public constant
DAV2_TYPE = "depth_anything_v2"
YOLO26_TYPE = "yolo26_depth"


@dataclass(frozen=True)
class VisualEncoderMetadata:
    visual_encoder_type: str
    visual_encoder_model: str
    visual_feature_layer: str
    visual_token_dim: int
    visual_grid_size: int
    visual_encoder_frozen: bool
    preprocess_version: str
    depth_preprocess_version: str
    fingerprint: str
    input_size: int | tuple[int, int] = 224
    raw_feature_shape: tuple[int, int, int] | None = None
    source_revision: str | None = None

    def to_dict(self):
        return asdict(self)


def _validate_rgbd(rgb, depth):
    rgb, depth = np.asarray(rgb), np.asarray(depth)
    if rgb.ndim != 4 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
        raise ValueError("RGB must be B x H x W x 3 uint8 in RGB channel order")
    if depth.shape != (*rgb.shape[:3], 1):
        raise ValueError("Depth must be aligned B x H x W x 1 in metres")
    if min(rgb.shape[:3]) <= 0:
        raise ValueError("Empty RGB-D batch")
    return rgb, depth


def preprocess_rgbd(rgb, depth):
    """Legacy DA-V2 preprocessing; kept compatible with the existing agent."""
    rgb, depth = _validate_rgbd(rgb, depth)
    images, depths, validity = [], [], []
    scale = 224 / max(rgb.shape[1:3])
    for image, distance in zip(rgb, depth):
        distance = np.array(distance, dtype=np.float32, copy=True)
        distance[~np.isfinite(distance)] = 0
        im = cv2.resize(image, (0, 0), fx=scale, fy=scale)
        dep = cv2.resize(distance, (0, 0), fx=scale, fy=scale)
        ph, pw = (224-im.shape[0])//2, (224-im.shape[1])//2
        im = np.pad(im, ((ph, ph), (pw, pw), (0, 0)))
        dep = np.pad(dep, ((ph, ph), (pw, pw)))
        im = cv2.resize(im, (224, 224)).astype(np.float32)/255.
        dep = cv2.resize(dep, (224, 224))
        valid = (dep >= .1) & (dep <= 5.)
        dep[~valid] = 0
        images.append(im); depths.append(dep[..., None]); validity.append(float(valid.mean()))
    return np.stack(images), np.stack(depths), np.asarray(validity, dtype=np.float32)


def _letterbox_tensor(x, size, padding):
    """GPU-native centered Ultralytics-style LetterBox.

    ``size`` is either a square ``int`` (legacy DA-V2 convention) or a
    ``(H, W)`` pair so the YOLO backend can keep the native (e.g. 360x640)
    camera aspect instead of down-scaling and square-padding to 224.
    """
    if x.ndim != 4:
        raise ValueError("LetterBox input must be BCHW")
    if isinstance(size, (tuple, list)):
        target_h, target_w = int(size[0]), int(size[1])
    else:
        target_h = target_w = int(size)
    height, width = x.shape[-2:]
    scale = min(target_h/height, target_w/width)
    new_h, new_w = round(height*scale), round(width*scale)
    if (new_h, new_w) != (height, width):
        x = F.interpolate(x, (new_h, new_w), mode="bilinear", align_corners=False)
    dh, dw = target_h-new_h, target_w-new_w
    top, bottom = round(dh/2-.1), round(dh/2+.1)
    left, right = round(dw/2-.1), round(dw/2+.1)
    return F.pad(x, (left, right, top, bottom), value=padding)


def preprocess_yolo_rgb(rgb, device, image_size=224):
    rgb = np.asarray(rgb)
    if rgb.ndim != 4 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8 or min(rgb.shape[:3]) <= 0:
        raise ValueError("RGB must be B x H x W x 3 uint8 in RGB channel order")
    image = torch.as_tensor(rgb, device=device).permute(0, 3, 1, 2).float().div_(255.)
    return _letterbox_tensor(image, image_size, 114./255.).contiguous()


@dataclass(frozen=True)
class DepthAugmentationConfig:
    enabled: bool = False
    pixel_dropout: float = 0.
    gaussian_std_m: float = 0.
    proportional_std: float = 0.
    quantization_m: float = 0.
    patch_dropout: float = 0.
    patch_size: int = 16

    def __post_init__(self):
        if any(x < 0 or x > 1 for x in (self.pixel_dropout, self.patch_dropout)):
            raise ValueError("Depth dropout probabilities must be in [0, 1]")
        if min(self.gaussian_std_m, self.proportional_std, self.quantization_m) < 0:
            raise ValueError("Depth noise magnitudes must be non-negative")
        if self.patch_size < 1:
            raise ValueError("Depth patch_size must be positive")


class MetricDepthPreprocessor:
    """Independent, versioned metric-depth preprocessing and augmentation."""
    def __init__(self, min_depth=.1, max_depth=5., augmentation=None, training=False):
        if not 0 < min_depth < max_depth:
            raise ValueError("Expected 0 < min_depth < max_depth")
        self.min_depth, self.max_depth = float(min_depth), float(max_depth)
        self.augmentation = DepthAugmentationConfig(**(augmentation or {}))
        self.training = bool(training)

    def __call__(self, depth, device, image_size=224):
        array = np.asarray(depth)
        if array.ndim != 4 or array.shape[-1] != 1 or min(array.shape[:3]) <= 0:
            raise ValueError("Depth must be B x H x W x 1 in metres")
        distance = torch.as_tensor(array, device=device, dtype=torch.float32).permute(0, 3, 1, 2)
        valid = torch.isfinite(distance) & (distance >= self.min_depth) & (distance <= self.max_depth)
        distance = torch.where(valid, distance, torch.zeros_like(distance))
        cfg = self.augmentation
        if self.training and cfg.enabled:
            if cfg.gaussian_std_m:
                distance += torch.randn_like(distance)*cfg.gaussian_std_m
            if cfg.proportional_std:
                distance += torch.randn_like(distance)*distance*cfg.proportional_std
            if cfg.quantization_m:
                distance = torch.round(distance/cfg.quantization_m)*cfg.quantization_m
            if cfg.pixel_dropout:
                valid &= torch.rand_like(distance) >= cfg.pixel_dropout
            if cfg.patch_dropout:
                h, w = distance.shape[-2:]
                mask = torch.rand(distance.shape[0], 1, max(1, (h+cfg.patch_size-1)//cfg.patch_size),
                    max(1, (w+cfg.patch_size-1)//cfg.patch_size), device=distance.device) >= cfg.patch_dropout
                valid &= F.interpolate(mask.float(), (h, w), mode="nearest").bool()
            valid &= torch.isfinite(distance) & (distance >= self.min_depth) & (distance <= self.max_depth)
            distance = torch.where(valid, distance, torch.zeros_like(distance))
        fraction = valid.float().mean((1, 2, 3))
        distance = distance.clamp(0, self.max_depth).div_(self.max_depth)
        distance = _letterbox_tensor(distance, image_size, 0.)
        return distance.repeat(1, 3, 1, 1).contiguous(), fraction


class RGBDFeatureEncoder(nn.Module):
    metadata: dict[str, Any]

    @property
    def token_dim(self):
        return int(self.metadata["visual_token_dim"])

    @property
    def grid_size(self):
        return int(self.metadata["visual_grid_size"])

    @property
    def preprocess_version(self):
        return str(self.metadata["preprocess_version"])


class DAV2RGBDEncoder(RGBDFeatureEncoder):
    """Legacy dual Depth Anything V2 ViT-S feature encoder."""
    def __init__(self, rgb_model, depth_model, fingerprint="unverified"):
        super().__init__()
        self.rgb_model, self.depth_model = rgb_model, depth_model
        self.fingerprint = fingerprint
        self.register_buffer("mean", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))
        self.metadata = VisualEncoderMetadata(DAV2_TYPE, "depth-anything-v2-vits",
            "dinov2.get_intermediate_layers[0]", 384, 8, True,
            DAV2_PREPROCESS_VERSION, DAV2_PREPROCESS_VERSION, fingerprint,
            raw_feature_shape=(384, 16, 16)).to_dict()
        self.requires_grad_(False); self.eval()

    @classmethod
    def from_checkpoint(cls, checkpoint, device="cpu"):
        from third_party.depth_anything.depth_anything_v2.dpt import DepthAnythingV2
        state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        models, digest = [], hashlib.sha256()
        for branch in ("rgb_model", "depth_model"):
            prefix = f"rgbd_encoder.{branch}."
            weights = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
            if not weights:
                raise ValueError(f"Missing {prefix} weights in {checkpoint}")
            model = DepthAnythingV2(encoder="vits", features=64,
                out_channels=[48, 96, 192, 384]).pretrained.float()
            model.load_state_dict(weights, strict=True)
            for name, tensor in sorted(weights.items()):
                digest.update((branch+name).encode()); digest.update(tensor.contiguous().numpy().tobytes())
            models.append(model)
        result = cls(*models, fingerprint=digest.hexdigest()).to(device)
        result.checkpoint = str(Path(checkpoint).resolve())
        return result

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def encode_preprocessed(self, rgb, depth, pool=True):
        device = self.mean.device
        image = torch.as_tensor(rgb, device=device, dtype=torch.float32).permute(0, 3, 1, 2)
        distance = torch.as_tensor(depth, device=device, dtype=torch.float32).permute(0, 3, 1, 2)
        if image.shape[1:] != (3, 224, 224) or distance.shape != (image.shape[0], 1, 224, 224):
            raise ValueError("Expected preprocessed NHWC 224 RGB-D")
        rgb_token = self.rgb_model.get_intermediate_layers((image-self.mean)/self.std)[0]
        depth_token = self.depth_model.get_intermediate_layers(distance.repeat(1, 3, 1, 1))[0]
        for token in (rgb_token, depth_token):
            if token.shape[1:] != (256, 384):
                raise ValueError(f"Unexpected Depth Anything token shape: {token.shape}")
        if pool:
            def reduce(token):
                grid = token.transpose(1, 2).reshape(-1, 384, 16, 16)
                return F.avg_pool2d(grid, 2).flatten(2).transpose(1, 2).contiguous()
            rgb_token, depth_token = reduce(rgb_token), reduce(depth_token)
        return {"rgb_tokens": rgb_token, "depth_tokens": depth_token}

    def forward(self, rgb, depth):
        image, distance, valid = preprocess_rgbd(rgb, depth)
        features = self.encode_preprocessed(image, distance)
        features["depth_valid_fraction"] = torch.as_tensor(valid, device=self.mean.device)
        return features


FrozenRGBDEncoder = DAV2RGBDEncoder


class YOLOP3Extractor(nn.Module):
    """Early-exit Ultralytics graph runner; the final depth head never runs."""
    def __init__(self, core, feature_layer="depth_head_p3"):
        super().__init__()
        self.core = core
        self.feature_index = self._resolve_feature_index(feature_layer)

    def _resolve_feature_index(self, requested):
        modules, head = self.core.model, self.core.model[-1]
        sources = getattr(head, "f", None)
        if head.__class__.__name__ != "Depth" or not isinstance(sources, (tuple, list)) or not sources:
            raise ValueError("Expected an Ultralytics Depth head with P3/P4/P5 source indices")
        semantic = int(sources[0])
        if requested in ("auto", "p3", "p3_stride8", "depth_head_p3"):
            index = semantic
        else:
            try:
                index = int(requested)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Unknown YOLO feature layer {requested!r}") from exc
            if index != semantic:
                raise ValueError(f"Layer {index} is not the Depth-head P3 input ({semantic})")
        if index < 0 or index >= len(modules)-1:
            raise ValueError(f"Invalid YOLO feature layer index {index}")
        return index

    def forward(self, x):
        saved = []
        for module in self.core.model:
            if module.f != -1:
                x = saved[module.f] if isinstance(module.f, int) else [x if j == -1 else saved[j] for j in module.f]
            x = module(x)
            saved.append(x if module.i in self.core.save else None)
            if module.i == self.feature_index:
                if not isinstance(x, torch.Tensor) or x.ndim != 4:
                    raise ValueError("YOLO P3 layer did not return BCHW features")
                return x
        raise RuntimeError(f"YOLO feature layer {self.feature_index} was not executed")


def _import_vendored_yolo():
    third_party = rexnavdp.BASE / "third_party"
    if not (third_party/"ultralytics"/"__init__.py").is_file():
        raise ImportError(f"Vendored Ultralytics source is missing under {third_party}")
    old_path = list(sys.path)
    try:
        sys.path.insert(0, str(third_party))
        module = importlib.import_module("ultralytics")
    finally:
        sys.path[:] = old_path
    if third_party.resolve() not in Path(module.__file__).resolve().parents:
        raise ImportError(f"Expected vendored Ultralytics, imported {module.__file__}")
    return module.YOLO


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


class YOLO26DepthRGBDEncoder(RGBDFeatureEncoder):
    """Dual frozen YOLO26-Depth P3 feature extractors.

    Returns pooled raw P3 tokens ``[B, 64, C]``. The policy projection is the
    trainable ``C -> 128`` adapter, so it remains learnable from cached data.
    """
    def __init__(self, rgb_model, depth_model, *, model_name="yolo26n-depth",
                 feature_layer="depth_head_p3", fingerprint="unverified",
                 image_size=224, grid_size=8, depth_preprocessor=None,
                 compile_mode=None,
                 source_revision="82737b9e3aa61aaf104d61a055db4c773a4e7e8d"):
        super().__init__()
        self.rgb_model = YOLOP3Extractor(rgb_model, feature_layer)
        self.depth_model = YOLOP3Extractor(depth_model, feature_layer)
        if compile_mode:
            # Eager-mode small CNNs are CUDA-kernel-launch-bound at batch 1;
            # torch.compile fuses the graph (reduce-overhead adds CUDA graphs),
            # cutting the P3 extraction from ~5.7 ms to ~0.75 ms per branch.
            self.rgb_model = torch.compile(self.rgb_model, mode=compile_mode)
            self.depth_model = torch.compile(self.depth_model, mode=compile_mode)
        if isinstance(image_size, (tuple, list)):
            size_h, size_w = int(image_size[0]), int(image_size[1])
        else:
            size_h = size_w = int(image_size)
        if size_h % 32 or size_w % 32:
            raise ValueError(
                f"YOLO image_size must be divisible by 32 (max stride); got {(size_h, size_w)}. "
                "Letterbox the camera to the nearest 32-multiple (e.g. [384, 640] for a 640x360 stream).")
        self.image_size, self.output_grid_size = image_size, int(grid_size)
        self._size_hw = (size_h, size_w)
        self.depth_preprocessor = depth_preprocessor or MetricDepthPreprocessor()
        self.fingerprint, self.forward_count = fingerprint, 0
        self.requires_grad_(False); self.eval()
        device = next(self.rgb_model.parameters()).device
        with torch.inference_mode():
            feature = self.rgb_model(torch.zeros(1, 3, size_h, size_w, device=device))
        expected_h, expected_w = size_h//8, size_w//8
        if feature.shape[-2:] != (expected_h, expected_w):
            raise ValueError(f"Resolved YOLO P3 is not stride 8: {tuple(feature.shape)}")
        channels, height, width = map(int, feature.shape[1:])
        target = self.rgb_model.feature_index
        self.metadata = VisualEncoderMetadata(YOLO26_TYPE, model_name,
            f"model.{target}:Depth.P3/stride8", channels, self.output_grid_size, True,
            YOLO_RGB_PREPROCESS_VERSION, YOLO_DEPTH_PREPROCESS_VERSION, fingerprint,
            input_size=self.image_size, raw_feature_shape=(channels, height, width),
            source_revision=source_revision).to_dict()

    @classmethod
    def from_weights(cls, weights, device="cpu", *, model_name=None,
                     feature_layer="depth_head_p3", image_size=224, grid_size=8,
                     depth_preprocess=None, depth_augmentation=None, training=False,
                     compile_mode=None):
        path = Path(weights).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"YOLO26-Depth weights not found: {path}. Set visual_encoder.weights to a local yolo26*-depth.pt file")
        YOLO = _import_vendored_yolo()
        core = YOLO(str(path), task="depth").model.float().to(device).eval()
        preprocessor = MetricDepthPreprocessor(**(depth_preprocess or {}),
            augmentation=depth_augmentation, training=training)
        return cls(core, deepcopy(core), model_name=model_name or path.stem,
            feature_layer=feature_layer, fingerprint=_sha256_file(path),
            image_size=image_size, grid_size=grid_size, depth_preprocessor=preprocessor,
            compile_mode=compile_mode)

    def train(self, mode=True):
        super().train(False)
        return self

    @torch.no_grad()
    def encode_preprocessed(self, rgb, depth):
        # Clone each extractor output immediately: under torch.compile(reduce-overhead)
        # the two extractors replay CUDA graphs whose static output buffers are reused,
        # so the second replay would overwrite the first branch's map before it is pooled.
        rgb_map, depth_map = self.rgb_model(rgb).clone(), self.depth_model(depth).clone()
        if rgb_map.shape != depth_map.shape:
            raise ValueError(f"RGB/depth YOLO shapes differ: {rgb_map.shape} vs {depth_map.shape}")
        def pool(feature):
            return F.adaptive_avg_pool2d(feature, (self.output_grid_size, self.output_grid_size)).flatten(2).transpose(1, 2).contiguous()
        return {"rgb_tokens": pool(rgb_map), "depth_tokens": pool(depth_map)}

    def forward(self, rgb, depth):
        rgb, depth = _validate_rgbd(rgb, depth)
        device = next(self.rgb_model.parameters()).device
        image = preprocess_yolo_rgb(rgb, device, self.image_size)
        distance, valid = self.depth_preprocessor(depth, device, self.image_size)
        features = self.encode_preprocessed(image, distance)
        features["depth_valid_fraction"] = valid
        self.forward_count += 1
        return features


def build_rgbd_encoder(config: Mapping[str, Any] | None, x_navdp_checkpoint=None,
                       device="cpu", *, training=False):
    """Backend factory. Missing config intentionally means legacy DA-V2."""
    options = dict(config or {})
    encoder_type = str(options.pop("type", DAV2_TYPE)).lower().replace("-", "_")
    if encoder_type in {"dav2", "da_v2", "depth_anything", DAV2_TYPE}:
        unknown = set(options)-{"model", "checkpoint", "frozen"}
        if unknown:
            raise ValueError(f"Unknown DA-V2 visual_encoder options: {sorted(unknown)}")
        checkpoint = options.get("checkpoint") or x_navdp_checkpoint
        if checkpoint is None:
            raise ValueError("DA-V2 backend requires an X-NavDP checkpoint")
        return DAV2RGBDEncoder.from_checkpoint(checkpoint, device)
    if encoder_type in {"yolo26", "yolo_depth", YOLO26_TYPE}:
        weights = options.pop("weights", None) or options.pop("checkpoint", None)
        if weights is None:
            raise ValueError("YOLO26 backend requires visual_encoder.weights")
        if not options.pop("frozen", True):
            raise ValueError("YOLO26 backbones are frozen in this implementation")
        unknown = set(options)-{"model_name", "feature_layer", "image_size",
                               "grid_size", "depth_preprocess", "depth_augmentation",
                               "compile_mode"}
        if unknown:
            raise ValueError(f"Unknown YOLO26 visual_encoder options: {sorted(unknown)}")
        return YOLO26DepthRGBDEncoder.from_weights(weights, device, training=training, **options)
    raise ValueError(f"Unknown visual_encoder.type {encoder_type!r}")


def resolve_policy_visual_config(policy: Mapping[str, Any], encoder: RGBDFeatureEncoder):
    """Derive cached-token shape from the selected backend."""
    result = dict(policy)
    result["token_dim"], result["grid_size"] = encoder.token_dim, encoder.grid_size
    return result
