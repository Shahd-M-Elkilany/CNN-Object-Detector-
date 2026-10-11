"""Box geometry: IoU, non-maximum suppression and box-regression encoding.

All boxes are ``[x1, y1, x2, y2]`` in pixels with ``x2 > x1`` and ``y2 > y1``.
Everything is plain NumPy so it can be unit-tested without TensorFlow.
"""

from __future__ import annotations

import numpy as np


def as_boxes(boxes) -> np.ndarray:
    """Coerce to a float32 array of shape (N, 4), accepting empty input."""
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def box_area(boxes: np.ndarray) -> np.ndarray:
    boxes = as_boxes(boxes)
    return np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two sets of boxes -> array of shape (len(a), len(b))."""
    a, b = as_boxes(a), as_boxes(b)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    union = box_area(a)[:, None] + box_area(b)[None, :] - inter
    return (inter / np.maximum(union, 1e-9)).astype(np.float32)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    """Greedy non-maximum suppression.

    Returns the indices of the kept boxes, highest score first.
    """
    boxes = as_boxes(boxes)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if len(boxes) == 0:
        return np.zeros(0, dtype=np.int64)
    areas = box_area(boxes)
    order = np.argsort(-scores, kind="stable")
    keep: list[int] = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        rest = order[1:]
        if rest.size == 0:
            break
        x1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        y1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        x2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        y2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        iou = inter / np.maximum(areas[i] + areas[rest] - inter, 1e-9)
        order = rest[iou <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def vote_boxes(
    kept: np.ndarray, candidates: np.ndarray, scores: np.ndarray, iou_threshold: float
) -> np.ndarray:
    """Refine each kept box with the candidates NMS suppressed in its favour.

    Neighbouring windows and pyramid levels all regress the same object; their
    score-weighted average is a better estimate of its edges than the single
    top-scoring box ("box voting", Gidaris & Komodakis 2015).
    """
    kept, candidates = as_boxes(kept), as_boxes(candidates)
    if len(kept) == 0 or len(candidates) == 0:
        return kept
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    weights = (iou_matrix(kept, candidates) >= iou_threshold) * scores[None, :].astype(np.float64)
    total = weights.sum(axis=1, keepdims=True)
    voted = weights @ candidates.astype(np.float64) / np.maximum(total, 1e-12)
    # A mean cannot leave the range of its inputs; clipping removes the last
    # bit of floating-point overshoot so boxes stay inside the image.
    voted = np.clip(voted, candidates.min(axis=0), candidates.max(axis=0))
    return np.where(total > 0, voted, kept).astype(np.float32)


def encode(anchors: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """R-CNN box parameterisation of ``gt`` relative to ``anchors``.

    ``t = (dx / wa, dy / ha, log(w / wa), log(h / ha))`` where ``dx, dy`` are
    centre offsets. Inverse of :func:`decode`.
    """
    anchors, gt = as_boxes(anchors), as_boxes(gt)
    wa = anchors[:, 2] - anchors[:, 0]
    ha = anchors[:, 3] - anchors[:, 1]
    xa = anchors[:, 0] + 0.5 * wa
    ya = anchors[:, 1] + 0.5 * ha
    w = np.maximum(gt[:, 2] - gt[:, 0], 1e-3)
    h = np.maximum(gt[:, 3] - gt[:, 1], 1e-3)
    x = gt[:, 0] + 0.5 * (gt[:, 2] - gt[:, 0])
    y = gt[:, 1] + 0.5 * (gt[:, 3] - gt[:, 1])
    return np.stack([(x - xa) / wa, (y - ya) / ha, np.log(w / wa), np.log(h / ha)], axis=1).astype(np.float32)


#: Regressed log-sizes are clamped to this range (a factor of ~4.5 either way)
#: so an untrained or confused network cannot emit absurd boxes.
MAX_LOG_SIZE = 1.5


def decode(anchors: np.ndarray, deltas: np.ndarray) -> np.ndarray:
    """Apply regression ``deltas`` to ``anchors``. Inverse of :func:`encode`."""
    anchors = as_boxes(anchors)
    deltas = np.asarray(deltas, dtype=np.float32).reshape(-1, 4)
    wa = anchors[:, 2] - anchors[:, 0]
    ha = anchors[:, 3] - anchors[:, 1]
    xa = anchors[:, 0] + 0.5 * wa
    ya = anchors[:, 1] + 0.5 * ha
    x = xa + deltas[:, 0] * wa
    y = ya + deltas[:, 1] * ha
    w = wa * np.exp(np.clip(deltas[:, 2], -MAX_LOG_SIZE, MAX_LOG_SIZE))
    h = ha * np.exp(np.clip(deltas[:, 3], -MAX_LOG_SIZE, MAX_LOG_SIZE))
    return np.stack([x - 0.5 * w, y - 0.5 * h, x + 0.5 * w, y + 0.5 * h], axis=1).astype(np.float32)


def clip_boxes(boxes: np.ndarray, height: int, width: int) -> np.ndarray:
    boxes = as_boxes(boxes).copy()
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)
    return boxes
