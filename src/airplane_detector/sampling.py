"""Training-patch sampling.

The network looks at one ``window x window`` patch and answers two questions:
*is there an airplane filling the central anchor box?* and *where exactly are
its edges?*  This module turns annotated images into such patches.

Geometry
--------
A patch is described by its centre ``(cx, cy)`` in original-image pixels and a
pyramid scale ``s`` (the image is conceptually resized by ``s`` before the
``window``-sized crop is taken). The anchor is the centred square of side
``anchor / s`` original pixels.

* **positive**  – IoU(anchor, some ground-truth box) >= ``pos_iou``
* **negative**  – IoU(anchor, every ground-truth box) <  ``neg_iou``
* anything in between is never sampled (it is ambiguous by construction).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .boxes import as_boxes, encode, iou_matrix
from .config import Config
from .data import Sample

BORDER = cv2.BORDER_REFLECT_101


# --------------------------------------------------------------------------
# pyramid
# --------------------------------------------------------------------------
def pyramid_scales(cfg: Config, height: int | None = None, width: int | None = None) -> np.ndarray:
    """Scales of the image pyramid, largest (finest) first.

    Scale ``s`` makes an object of side ``anchor / s`` pixels fill the anchor.
    When the image size is given, scales that could only match objects larger
    than the image itself are dropped.
    """
    n = int(np.ceil(np.log2(cfg.max_object / cfg.min_object) * cfg.scales_per_octave)) + 1
    sizes = cfg.min_object * 2.0 ** (np.arange(n) / cfg.scales_per_octave)
    scales = cfg.anchor / sizes
    if height is not None and width is not None:
        keep = sizes <= max(height, width) * 1.01
        keep[0] = True  # always keep at least the finest level
        scales = scales[keep]
    return scales.astype(np.float64)


def scale_range(cfg: Config, height: int, width: int) -> tuple[float, float]:
    scales = pyramid_scales(cfg, height, width)
    return float(scales.min()), float(scales.max())


# --------------------------------------------------------------------------
# patch extraction
# --------------------------------------------------------------------------
def window_box(cx: float, cy: float, scale: float, window: int) -> np.ndarray:
    """Integer-aligned window ``[x1, y1, x2, y2]`` in original pixels."""
    side = max(2, int(round(window / scale)))
    x1 = int(round(cx - side / 2.0))
    y1 = int(round(cy - side / 2.0))
    return np.array([x1, y1, x1 + side, y1 + side], dtype=np.float32)


def anchor_of(window: np.ndarray, cfg: Config) -> np.ndarray:
    """The anchor box (centred square) of a window, in original pixels."""
    x1, y1, x2, y2 = window
    side = (x2 - x1) * cfg.anchor / cfg.window
    cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    return np.array([cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2], dtype=np.float32)


def extract_window(img: np.ndarray, window: np.ndarray, out_size: int) -> np.ndarray:
    """Crop ``window`` from ``img`` (reflecting at the borders) and resize."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in window)
    left, top = max(0, -x1), max(0, -y1)
    right, bottom = max(0, x2 - w), max(0, y2 - h)
    if left or top or right or bottom:
        img = cv2.copyMakeBorder(img, top, bottom, left, right, BORDER)
        x1, x2, y1, y2 = x1 + left, x2 + left, y1 + top, y2 + top
    crop = img[y1:y2, x1:x2]
    if crop.shape[0] == out_size and crop.shape[1] == out_size:
        return crop.copy()
    interp = cv2.INTER_AREA if crop.shape[0] > out_size else cv2.INTER_LINEAR
    return cv2.resize(crop, (out_size, out_size), interpolation=interp)


# --------------------------------------------------------------------------
# augmentation
# --------------------------------------------------------------------------
def dihedral(patch: np.ndarray, boxes: np.ndarray, k: int, flip: bool) -> tuple[np.ndarray, np.ndarray]:
    """Rotate a square patch by ``k`` x 90 degrees and optionally mirror it.

    Aerial imagery has no preferred orientation, so all eight symmetries of
    the square are valid augmentations. ``boxes`` are in patch pixels and are
    transformed consistently.
    """
    size = patch.shape[0]
    boxes = as_boxes(boxes).copy()
    patch = np.rot90(patch, k % 4)
    for _ in range(k % 4):
        # counter-clockwise quarter turn: (x, y) -> (y, size - x)
        x1, y1, x2, y2 = boxes.T.copy()
        boxes = np.stack([y1, size - x2, y2, size - x1], axis=1)
    if flip:
        patch = patch[:, ::-1]
        x1, x2 = boxes[:, 0].copy(), boxes[:, 2].copy()
        boxes[:, 0], boxes[:, 2] = size - x2, size - x1
    return np.ascontiguousarray(patch), boxes.astype(np.float32)


def photometric(patch: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Random contrast, brightness and per-channel gain (uint8 in, uint8 out)."""
    contrast = rng.uniform(0.75, 1.3)
    brightness = rng.uniform(-25, 25)
    gain = rng.uniform(0.93, 1.07, size=3)
    out = (patch.astype(np.float32) - 127.5) * contrast * gain + 127.5 + brightness
    return np.clip(out, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# sampler
# --------------------------------------------------------------------------
@dataclass
class HardNegative:
    """A false positive found on a training image: where it was and how big."""

    image_index: int
    cx: float
    cy: float
    scale: float


class PatchSampler:
    """Draws labelled training patches from a list of annotated images."""

    def __init__(self, samples: list[Sample], cfg: Config, augment: bool = True):
        self.samples = samples
        self.cfg = cfg
        self.augment = augment
        self.images = [s.load() for s in samples]
        self.ranges = [scale_range(cfg, *img.shape[:2]) for img in self.images]
        self.box_index = [(i, j) for i, s in enumerate(samples) for j in range(len(s.boxes))]

    # ------------------------------------------------------------ helpers
    def _centre_limits(self, index: int, scale: float) -> tuple[float, float, float, float]:
        """Range of window centres reachable at inference time (see detect.py)."""
        cfg = self.cfg
        h, w = self.images[index].shape[:2]
        lo = (cfg.window / 2 - cfg.pad) / scale
        hi_x = max(lo, (w * scale + cfg.pad - cfg.window / 2) / scale)
        hi_y = max(lo, (h * scale + cfg.pad - cfg.window / 2) / scale)
        return lo, hi_x, lo, hi_y

    def _clamp(self, index: int, cx: float, cy: float, scale: float) -> tuple[float, float]:
        lo_x, hi_x, lo_y, hi_y = self._centre_limits(index, scale)
        return float(np.clip(cx, lo_x, hi_x)), float(np.clip(cy, lo_y, hi_y))

    def _max_iou(self, index: int, window: np.ndarray) -> tuple[float, int]:
        boxes = self.samples[index].boxes
        if len(boxes) == 0:
            return 0.0, -1
        ious = iou_matrix(anchor_of(window, self.cfg)[None], boxes)[0]
        j = int(ious.argmax())
        return float(ious[j]), j

    def _finish(
        self, index: int, window: np.ndarray, gt: np.ndarray | None, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Cut the patch, augment it and build the 5-vector target."""
        cfg = self.cfg
        patch = extract_window(self.images[index], window, cfg.window)
        to_patch = cfg.window / (window[2] - window[0])
        box = np.zeros((1, 4), np.float32)
        if gt is not None:
            box = (as_boxes(gt) - np.tile(window[:2], 2)) * to_patch
        if self.augment:
            patch, box = dihedral(patch, box, int(rng.integers(4)), bool(rng.integers(2)))
            patch = photometric(patch, rng)
        target = np.zeros(5, np.float32)
        if gt is not None:
            half = (cfg.window - cfg.anchor) / 2
            anchor = np.array([[half, half, half + cfg.anchor, half + cfg.anchor]], np.float32)
            target[0] = 1.0
            target[1:] = encode(anchor, box)[0]
        return patch, target

    # ----------------------------------------------------------- positives
    def sample_positive(self, index: int, box_id: int, rng: np.random.Generator):
        """A jittered window whose anchor overlaps the given box, or ``None``.

        ``None`` is returned for objects the pyramid cannot represent (smaller
        than ``min_object`` or far larger than ``max_object``).
        """
        cfg = self.cfg
        gt = self.samples[index].boxes[box_id]
        gw, gh = gt[2] - gt[0], gt[3] - gt[1]
        gx, gy = gt[0] + gw / 2, gt[1] + gh / 2
        size = float(np.sqrt(gw * gh))
        s_min, s_max = self.ranges[index]
        best = None
        for attempt in range(12):
            jitter = self.augment and attempt < 11
            side = size * (2.0 ** rng.uniform(-0.3, 0.3) if jitter else 1.0)
            scale = float(np.clip(cfg.anchor / side, s_min, s_max))
            side = cfg.anchor / scale
            dx, dy = (rng.uniform(-0.2, 0.2, size=2) * side) if jitter else (0.0, 0.0)
            cx, cy = self._clamp(index, gx + dx, gy + dy, scale)
            window = window_box(cx, cy, scale, cfg.window)
            iou = float(iou_matrix(anchor_of(window, cfg)[None], gt[None])[0, 0])
            if iou >= cfg.pos_iou:
                return self._finish(index, window, gt, rng)
            if best is None or iou > best[0]:
                best = (iou, window)
        # Elongated boxes can never reach pos_iou with a square anchor. Like
        # the "best anchor for each object" rule of Faster R-CNN, accept the
        # best window found as long as it is not outright a negative.
        if best is not None and best[0] >= cfg.neg_iou:
            return self._finish(index, best[1], gt, rng)
        return None

    # ----------------------------------------------------------- negatives
    def sample_random_negative(self, rng: np.random.Generator):
        cfg = self.cfg
        for _ in range(50):
            index = int(rng.integers(len(self.samples)))
            s_min, s_max = self.ranges[index]
            scale = float(rng.uniform(s_min, s_max))
            lo_x, hi_x, lo_y, hi_y = self._centre_limits(index, scale)
            window = window_box(rng.uniform(lo_x, hi_x), rng.uniform(lo_y, hi_y), scale, cfg.window)
            if self._max_iou(index, window)[0] < cfg.neg_iou:
                return self._finish(index, window, None, rng)
        return None

    def sample_near_miss(self, rng: np.random.Generator):
        """A window next to an object: wrong scale, or shifted off-centre.

        These are the negatives that teach localisation - without them the
        classifier fires on anything that contains part of a plane.
        """
        cfg = self.cfg
        if not self.box_index:
            return None
        for _ in range(50):
            index, box_id = self.box_index[int(rng.integers(len(self.box_index)))]
            gt = self.samples[index].boxes[box_id]
            gw, gh = gt[2] - gt[0], gt[3] - gt[1]
            gx, gy = gt[0] + gw / 2, gt[1] + gh / 2
            size = float(np.sqrt(gw * gh))
            s_min, s_max = self.ranges[index]
            if rng.random() < 0.5:  # wrong scale, roughly centred
                side = size * 2.0 ** (rng.choice([-1, 1]) * rng.uniform(0.8, 1.8))
                offset = rng.uniform(-0.25, 0.25, size=2) * max(side, size)
            else:  # right scale, shifted
                side = size * 2.0 ** rng.uniform(-0.3, 0.3)
                angle = rng.uniform(0, 2 * np.pi)
                offset = rng.uniform(0.45, 1.1) * side * np.array([np.cos(angle), np.sin(angle)])
            scale = float(np.clip(cfg.anchor / side, s_min, s_max))
            cx, cy = self._clamp(index, gx + offset[0], gy + offset[1], scale)
            window = window_box(cx, cy, scale, cfg.window)
            if self._max_iou(index, window)[0] < cfg.neg_iou:
                return self._finish(index, window, None, rng)
        return None

    def sample_hard_negative(self, hard: HardNegative, rng: np.random.Generator):
        cfg = self.cfg
        s_min, s_max = self.ranges[hard.image_index]
        scale = float(np.clip(hard.scale * 2.0 ** rng.uniform(-0.08, 0.08), s_min, s_max))
        shift = rng.uniform(-2.0, 2.0, size=2) / scale
        cx, cy = self._clamp(hard.image_index, hard.cx + shift[0], hard.cy + shift[1], scale)
        window = window_box(cx, cy, scale, cfg.window)
        if self._max_iou(hard.image_index, window)[0] < cfg.neg_iou:
            return self._finish(hard.image_index, window, None, rng)
        return None

    # --------------------------------------------------------------- epoch
    def sample_epoch(
        self,
        rng: np.random.Generator,
        hard_pool: list[HardNegative] | None = None,
        pos_per_box: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """One epoch worth of patches: ``X`` uint8 (N, W, W, 3) BGR and
        ``Y`` float32 (N, 1, 1, 5) = ``[label, tx, ty, tw, th]``."""
        cfg = self.cfg
        pos_per_box = cfg.pos_per_box if pos_per_box is None else pos_per_box
        patches: list[np.ndarray] = []
        targets: list[np.ndarray] = []

        def add(item) -> bool:
            if item is None:
                return False
            patches.append(item[0])
            targets.append(item[1])
            return True

        for index, box_id in self.box_index:
            for _ in range(pos_per_box):
                add(self.sample_positive(index, box_id, rng))
        n_pos = len(patches)
        n_neg = int(round(max(n_pos, 1) * cfg.neg_per_pos))
        n_hard = int(round(n_neg * cfg.hard_fraction)) if hard_pool else 0
        n_near = int(round((n_neg - n_hard) * cfg.near_miss_fraction))
        n_rand = n_neg - n_hard - n_near

        if n_hard:
            picks = rng.integers(len(hard_pool), size=n_hard)
            n_rand += sum(not add(self.sample_hard_negative(hard_pool[int(i)], rng)) for i in picks)
        n_rand += sum(not add(self.sample_near_miss(rng)) for _ in range(n_near))
        for _ in range(n_rand):
            add(self.sample_random_negative(rng))

        x = np.stack(patches).astype(np.uint8)
        y = np.stack(targets).astype(np.float32).reshape(-1, 1, 1, 5)
        order = rng.permutation(len(x))
        return x[order], y[order]
