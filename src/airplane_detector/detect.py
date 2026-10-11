"""Multi-scale fully-convolutional inference.

For every level of an image pyramid the network is run once over the whole
(padded) level. Each output cell corresponds to one 40x40 window of the level;
its objectness score and box offsets are decoded back to original-image
coordinates, and the candidates of all levels are merged with NMS.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .boxes import clip_boxes, decode, nms, vote_boxes
from .config import Config
from .sampling import BORDER, pyramid_scales


@dataclass
class Detections:
    boxes: np.ndarray  # (N, 4) x1 y1 x2 y2 in original pixels
    scores: np.ndarray  # (N,) objectness probability
    anchors: np.ndarray  # (N, 4) the anchor each box was regressed from
    scales: np.ndarray  # (N,) pyramid scale the detection came from

    def __len__(self) -> int:
        return len(self.scores)

    def select(self, index) -> Detections:
        return Detections(self.boxes[index], self.scores[index], self.anchors[index], self.scales[index])

    def above(self, threshold: float) -> Detections:
        return self.select(self.scores >= threshold)

    @classmethod
    def empty(cls) -> Detections:
        z = np.zeros((0, 4), np.float32)
        return cls(z, np.zeros(0, np.float32), z.copy(), np.zeros(0, np.float32))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def prepare_level(img: np.ndarray, scale: float, cfg: Config) -> tuple[np.ndarray, float, float]:
    """Resize ``img`` by ``scale`` and pad it for the network.

    Returns the padded level and the exact x / y scale factors actually used
    (they differ slightly from ``scale`` because sizes are rounded to pixels).
    The padding is ``cfg.pad`` on the top/left and at least ``cfg.pad`` on the
    bottom/right, extended so the stride-8 grid ends exactly at the border.
    """
    h, w = img.shape[:2]
    lw, lh = max(2, int(round(w * scale))), max(2, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    level = img if (lw, lh) == (w, h) else cv2.resize(img, (lw, lh), interpolation=interp)

    def tail(size: int) -> int:
        total = max(size + 2 * cfg.pad, cfg.window)
        total += (-(total - cfg.window)) % cfg.stride
        return total - size - cfg.pad

    level = cv2.copyMakeBorder(level, cfg.pad, tail(lh), cfg.pad, tail(lw), BORDER)
    return level, lw / w, lh / h


class Detector:
    """Wraps a trained network and turns images into airplane boxes."""

    def __init__(self, model, cfg: Config):
        import tensorflow as tf

        self.model = model
        self.cfg = cfg
        self._forward = tf.function(
            lambda x: model(x, training=False),
            input_signature=[tf.TensorSpec([None, None, None, 3], tf.float32)],
            autograph=False,
        )

    # ------------------------------------------------------------ loading
    @classmethod
    def load(cls, model_path: str | Path, config_path: str | Path | None = None) -> Detector:
        """Load ``<name>.keras`` together with its sibling ``<name>.yaml``."""
        from .model import load_model

        model_path = Path(model_path)
        config_path = Path(config_path) if config_path else model_path.with_suffix(".yaml")
        return cls(load_model(str(model_path)), Config.from_yaml(config_path))

    # ------------------------------------------------------------ network
    def forward(self, level: np.ndarray) -> np.ndarray:
        """Dense network output (rows, cols, 5) for one padded pyramid level.

        Large levels are processed in tiles that overlap by exactly the
        receptive field minus one stride, which makes the stitched result
        identical to a single pass while bounding memory.
        """
        cfg = self.cfg
        h, w = level.shape[:2]
        rows = (h - cfg.window) // cfg.stride + 1
        cols = (w - cfg.window) // cfg.stride + 1
        chunk = max(1, int((np.sqrt(cfg.max_level_pixels) - cfg.window) // cfg.stride) + 1)
        if rows <= chunk and cols <= chunk:
            return self._forward(level[None].astype(np.float32)).numpy()[0]
        out = np.zeros((rows, cols, 5), np.float32)
        for r0 in range(0, rows, chunk):
            r1 = min(rows, r0 + chunk)
            for c0 in range(0, cols, chunk):
                c1 = min(cols, c0 + chunk)
                tile = level[
                    r0 * cfg.stride : (r1 - 1) * cfg.stride + cfg.window,
                    c0 * cfg.stride : (c1 - 1) * cfg.stride + cfg.window,
                ]
                out[r0:r1, c0:c1] = self._forward(tile[None].astype(np.float32)).numpy()[0]
        return out

    # ---------------------------------------------------------- detection
    def candidates(
        self,
        img: np.ndarray,
        score_floor: float | None = None,
        use_regression: bool = True,
        scales: np.ndarray | None = None,
    ) -> Detections:
        """All windows scoring above ``score_floor``, before NMS."""
        cfg = self.cfg
        floor = cfg.score_floor if score_floor is None else score_floor
        h, w = img.shape[:2]
        if scales is None:
            scales = pyramid_scales(cfg, h, w)
        boxes, scores, anchors, level_scales = [], [], [], []
        for scale in scales:
            level, sx, sy = prepare_level(img, float(scale), cfg)
            out = self.forward(level)
            prob = _sigmoid(out[..., 0])
            rows, cols = np.nonzero(prob >= floor)
            if rows.size == 0:
                continue
            # Anchor centres in (unpadded) level coordinates.
            ax = cols * cfg.stride - cfg.pad + cfg.window / 2.0
            ay = rows * cfg.stride - cfg.pad + cfg.window / 2.0
            inside = (ax >= 0) & (ax <= w * sx) & (ay >= 0) & (ay <= h * sy)
            rows, cols, ax, ay = rows[inside], cols[inside], ax[inside], ay[inside]
            if rows.size == 0:
                continue
            half = cfg.anchor / 2.0
            anchor = np.stack([ax - half, ay - half, ax + half, ay + half], axis=1).astype(np.float32)
            box = decode(anchor, out[rows, cols, 1:]) if use_regression else anchor.copy()
            to_image = np.array([sx, sy, sx, sy], np.float32)
            boxes.append(box / to_image)
            anchors.append(anchor / to_image)
            scores.append(prob[rows, cols].astype(np.float32))
            level_scales.append(np.full(rows.size, scale, np.float32))
        if not boxes:
            return Detections.empty()
        det = Detections(
            clip_boxes(np.concatenate(boxes), h, w),
            np.concatenate(scores),
            np.concatenate(anchors),
            np.concatenate(level_scales),
        )
        sides = np.minimum(det.boxes[:, 2] - det.boxes[:, 0], det.boxes[:, 3] - det.boxes[:, 1])
        return det.select(sides >= 2.0)

    def merge(
        self, det: Detections, nms_iou: float | None = None, vote_iou: float | None = None
    ) -> Detections:
        """NMS followed by box voting; turns raw candidates into final boxes."""
        cfg = self.cfg
        if len(det) == 0:
            return det
        nms_iou = cfg.nms_iou if nms_iou is None else nms_iou
        vote_iou = cfg.vote_iou if vote_iou is None else vote_iou
        out = det.select(nms(det.boxes, det.scores, nms_iou)[: cfg.max_detections])
        if vote_iou > 0:
            out.boxes = vote_boxes(out.boxes, det.boxes, det.scores, vote_iou)
        return out

    def detect(
        self,
        img: np.ndarray,
        score_threshold: float | None = None,
        nms_iou: float | None = None,
        use_regression: bool = True,
        scales: np.ndarray | None = None,
        vote_iou: float | None = None,
    ) -> Detections:
        """Detect airplanes in a BGR image.

        With ``score_threshold=None`` every detection above ``cfg.score_floor``
        is returned (what evaluation needs to trace the full PR curve); pass
        ``cfg.score_threshold`` to get the calibrated operating point.
        """
        det = self.candidates(img, None, use_regression, scales)
        det = self.merge(det, nms_iou, vote_iou)
        if score_threshold is not None:
            det = det.above(score_threshold)
        return det
