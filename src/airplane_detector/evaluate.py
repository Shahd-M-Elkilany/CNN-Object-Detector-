"""Detection metrics: average precision, precision/recall/F1, size breakdown.

The protocol follows COCO:

* detections are matched greedily in descending score order, each ground-truth
  box can be claimed once, a match needs IoU >= the threshold;
* AP is the mean interpolated precision at 101 equally spaced recall levels;
* ``mAP`` averages AP over IoU thresholds 0.50:0.05:0.95;
* for the per-size numbers, ground truth outside the size range is *ignored*
  (neither rewarded nor punished), as are unmatched detections outside it.

The implementation is cross-checked against ``pycocotools`` in the test suite.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .boxes import as_boxes, box_area, iou_matrix

COCO_IOUS = tuple(np.round(np.arange(0.5, 0.96, 0.05), 2))
RECALL_GRID = np.linspace(0.0, 1.0, 101)

#: Object-size buckets by sqrt(box area) in pixels. The dataset's objects are
#: far smaller than COCO's, so COCO's 32/96 px limits would be meaningless.
SIZE_BUCKETS = {"small": (0.0, 20.0), "medium": (20.0, 45.0), "large": (45.0, 1e9)}


@dataclass
class ImageResult:
    gt: np.ndarray  # (G, 4)
    boxes: np.ndarray  # (D, 4)
    scores: np.ndarray  # (D,)


def match_image(
    result: ImageResult, iou_threshold: float, size_range: tuple[float, float] | None = None
) -> tuple[np.ndarray, np.ndarray, int]:
    """Match one image's detections to its ground truth.

    Returns ``(is_tp, is_ignored, n_gt)`` where the two flag arrays are aligned
    with ``result.scores`` and ``n_gt`` is the number of non-ignored objects.
    """
    gt, boxes = as_boxes(result.gt), as_boxes(result.boxes)
    scores = np.asarray(result.scores, np.float32).reshape(-1)
    n_det = len(boxes)
    is_tp = np.zeros(n_det, bool)
    is_ignored = np.zeros(n_det, bool)
    if size_range is None:
        gt_ignore = np.zeros(len(gt), bool)
        det_out = np.zeros(n_det, bool)
    else:
        lo, hi = size_range
        gt_size = np.sqrt(box_area(gt))
        gt_ignore = (gt_size < lo) | (gt_size >= hi)
        det_size = np.sqrt(box_area(boxes))
        det_out = (det_size < lo) | (det_size >= hi)
    n_gt = int((~gt_ignore).sum())
    if n_det == 0:
        return is_tp, is_ignored, n_gt
    if len(gt) == 0:
        return is_tp, det_out, n_gt
    ious = iou_matrix(boxes, gt)
    taken = np.zeros(len(gt), bool)
    # Visit regular objects before ignored ones, so a detection only falls
    # back to an ignored object when no regular one is available.
    gt_order = np.argsort(gt_ignore, kind="stable")
    for d in np.argsort(-scores, kind="stable"):
        best, best_iou = -1, iou_threshold
        for g in gt_order:
            if taken[g]:
                continue
            if best >= 0 and not gt_ignore[best] and gt_ignore[g]:
                break
            if ious[d, g] < best_iou:
                continue
            best, best_iou = int(g), ious[d, g]
        if best >= 0:
            taken[best] = True
            if gt_ignore[best]:
                is_ignored[d] = True
            else:
                is_tp[d] = True
        elif det_out[d]:
            is_ignored[d] = True
    return is_tp, is_ignored, n_gt


def precision_recall(
    scores: np.ndarray, is_tp: np.ndarray, n_gt: int, weights: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precision and recall after each detection, in descending score order."""
    order = np.argsort(-scores, kind="stable")
    w = np.ones(len(scores)) if weights is None else weights[order]
    tp = np.cumsum(is_tp[order] * w)
    fp = np.cumsum((~is_tp[order]) * w)
    recall = tp / max(n_gt, 1e-9)
    precision = tp / np.maximum(tp + fp, 1e-9)
    return precision, recall, scores[order]


def average_precision(precision: np.ndarray, recall: np.ndarray) -> float:
    """COCO-style AP: mean of the precision envelope at 101 recall levels."""
    if len(precision) == 0:
        return 0.0
    envelope = np.maximum.accumulate(precision[::-1])[::-1]
    idx = np.searchsorted(recall, RECALL_GRID, side="left")
    sampled = np.where(idx < len(envelope), envelope[np.minimum(idx, len(envelope) - 1)], 0.0)
    return float(sampled.mean())


def _pool(results: list[ImageResult], iou: float, size_range=None):
    scores, tps, image_ids, n_gt_per_image = [], [], [], []
    for i, r in enumerate(results):
        is_tp, is_ignored, n_gt = match_image(r, iou, size_range)
        keep = ~is_ignored
        scores.append(np.asarray(r.scores, np.float32).reshape(-1)[keep])
        tps.append(is_tp[keep])
        image_ids.append(np.full(int(keep.sum()), i))
        n_gt_per_image.append(n_gt)
    return (
        np.concatenate(scores) if scores else np.zeros(0, np.float32),
        np.concatenate(tps) if tps else np.zeros(0, bool),
        np.concatenate(image_ids) if image_ids else np.zeros(0, int),
        np.asarray(n_gt_per_image),
    )


def ap_at(results: list[ImageResult], iou: float, size_range=None) -> float:
    scores, tps, _, n_gt = _pool(results, iou, size_range)
    precision, recall, _ = precision_recall(scores, tps, int(n_gt.sum()))
    return average_precision(precision, recall)


def operating_point(results: list[ImageResult], threshold: float, iou: float = 0.5) -> dict:
    """Precision / recall / F1 when only detections >= ``threshold`` are kept."""
    scores, tps, _, n_gt = _pool(results, iou)
    keep = scores >= threshold
    tp = int(tps[keep].sum())
    fp = int(keep.sum()) - tp
    fn = int(n_gt.sum()) - tp
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def best_f1_threshold(results: list[ImageResult], iou: float = 0.5) -> float:
    """Score threshold maximising F1 - to be chosen on the *validation* split."""
    scores, tps, _, n_gt = _pool(results, iou)
    if len(scores) == 0:
        return 0.5
    precision, recall, sorted_scores = precision_recall(scores, tps, int(n_gt.sum()))
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
    return float(sorted_scores[int(f1.argmax())])


def bootstrap_ap(
    results: list[ImageResult], iou: float = 0.5, n_boot: int = 1000, seed: int = 0
) -> tuple[float, float]:
    """95 % confidence interval of AP from resampling *images* with replacement."""
    scores, tps, image_ids, n_gt = _pool(results, iou)
    rng = np.random.default_rng(seed)
    order = np.argsort(-scores, kind="stable")
    tps_sorted, ids_sorted = tps[order], image_ids[order]
    values = np.empty(n_boot)
    for b in range(n_boot):
        counts = np.bincount(rng.integers(len(results), size=len(results)), minlength=len(results))
        w = counts[ids_sorted].astype(np.float64)
        tp = np.cumsum(tps_sorted * w)
        fp = np.cumsum((~tps_sorted) * w)
        present = w > 0
        total = float((counts * n_gt).sum())
        recall = tp[present] / max(total, 1e-9)
        precision = tp[present] / np.maximum(tp[present] + fp[present], 1e-9)
        values[b] = average_precision(precision, recall)
    lo, hi = np.percentile(values, [2.5, 97.5])
    return float(lo), float(hi)


def evaluate(
    results: list[ImageResult],
    threshold: float | None = None,
    n_boot: int = 1000,
    with_curve: bool = True,
) -> dict:
    """Full report for a list of per-image results."""
    report: dict = {
        "images": len(results),
        "objects": int(sum(len(r.gt) for r in results)),
        "detections": int(sum(len(r.scores) for r in results)),
    }
    aps = {iou: ap_at(results, iou) for iou in COCO_IOUS}
    report["AP50"] = aps[0.5]
    report["AP75"] = aps[0.75]
    report["mAP"] = float(np.mean(list(aps.values())))
    for name, size_range in SIZE_BUCKETS.items():
        n = int(
            sum(
                ((np.sqrt(box_area(r.gt)) >= size_range[0]) & (np.sqrt(box_area(r.gt)) < size_range[1])).sum()
                for r in results
            )
        )
        report[f"AP50_{name}"] = ap_at(results, 0.5, size_range) if n else None
        report[f"objects_{name}"] = n
    if n_boot:
        report["AP50_ci95"] = list(bootstrap_ap(results, 0.5, n_boot))

    scores, tps, _, n_gt = _pool(results, 0.5)
    precision, recall, sorted_scores = precision_recall(scores, tps, int(n_gt.sum()))
    report["max_recall"] = float(recall[-1]) if len(recall) else 0.0
    report["best_f1_threshold"] = best_f1_threshold(results)
    used = report["best_f1_threshold"] if threshold is None else threshold
    report["operating_point"] = operating_point(results, used)
    if with_curve:
        step = max(1, len(recall) // 400)
        report["pr_curve"] = {
            "recall": [round(float(v), 4) for v in recall[::step]],
            "precision": [round(float(v), 4) for v in precision[::step]],
            "score": [round(float(v), 4) for v in sorted_scores[::step]],
        }
    return report
