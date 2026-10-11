"""Drawing helpers."""

from __future__ import annotations

import cv2
import numpy as np

from .boxes import as_boxes, iou_matrix

# BGR colours
GREEN = (80, 200, 60)
RED = (60, 60, 235)
YELLOW = (40, 210, 250)
WHITE = (255, 255, 255)


def upscale(img: np.ndarray, min_side: int = 512) -> tuple[np.ndarray, float]:
    """Enlarge small tiles so 1 px outlines and labels stay readable."""
    factor = max(1.0, min_side / min(img.shape[:2]))
    if factor == 1.0:
        return img.copy(), 1.0
    size = (int(round(img.shape[1] * factor)), int(round(img.shape[0] * factor)))
    return cv2.resize(img, size, interpolation=cv2.INTER_CUBIC), factor


def draw_boxes(
    img: np.ndarray,
    boxes,
    color=GREEN,
    scores=None,
    thickness: int = 2,
    factor: float = 1.0,
) -> np.ndarray:
    """Draw ``boxes`` (original-image pixels) on ``img`` (already scaled by ``factor``)."""
    boxes = as_boxes(boxes) * factor
    for i, (x1, y1, x2, y2) in enumerate(boxes.round().astype(int)):
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        if scores is not None:
            label = f"{scores[i]:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            y = y1 - 3 if y1 - th - 4 >= 0 else y2 + th + 3
            cv2.rectangle(img, (x1, y - th - 2), (x1 + tw + 2, y + 2), color, -1)
            cv2.putText(img, label, (x1 + 1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1, cv2.LINE_AA)
    return img


def draw_detections(img: np.ndarray, boxes, scores=None, min_side: int = 512) -> np.ndarray:
    """Image with predicted boxes and scores, for the CLI and the demo app."""
    canvas, factor = upscale(img, min_side)
    return draw_boxes(canvas, boxes, GREEN, scores, factor=factor)


def draw_evaluation(img: np.ndarray, gt, boxes, iou: float = 0.5, min_side: int = 512) -> np.ndarray:
    """Colour-code a result: green = correct, red = false alarm, yellow = missed."""
    gt, boxes = as_boxes(gt), as_boxes(boxes)
    canvas, factor = upscale(img, min_side)
    matched_gt = np.zeros(len(gt), bool)
    is_tp = np.zeros(len(boxes), bool)
    if len(gt) and len(boxes):
        ious = iou_matrix(boxes, gt)
        for d in range(len(boxes)):  # boxes are already sorted by score
            candidates = np.where(~matched_gt, ious[d], 0.0)
            g = int(candidates.argmax())
            if candidates[g] >= iou:
                matched_gt[g] = True
                is_tp[d] = True
    draw_boxes(canvas, gt[~matched_gt], YELLOW, factor=factor)
    draw_boxes(canvas, boxes[~is_tp], RED, factor=factor)
    draw_boxes(canvas, boxes[is_tp], GREEN, factor=factor)
    return canvas
