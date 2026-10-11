import numpy as np
import pytest

from airplane_detector import evaluate as ev

GT = np.array([[0, 0, 10, 10], [20, 20, 30, 30]], np.float32)


def result(boxes, scores, gt=GT):
    return ev.ImageResult(
        np.asarray(gt, np.float32).reshape(-1, 4),
        np.asarray(boxes, np.float32).reshape(-1, 4),
        np.asarray(scores, np.float32),
    )


def test_perfect_detections_score_one():
    report = ev.evaluate([result(GT, [0.9, 0.8])], n_boot=0)
    assert report["AP50"] == report["AP75"] == report["mAP"] == pytest.approx(1.0)
    assert report["operating_point"]["f1"] == pytest.approx(1.0)


def test_no_detections_score_zero():
    report = ev.evaluate([result([], [])], n_boot=0)
    assert report["AP50"] == 0.0 and report["operating_point"]["recall"] == 0.0


def test_ap_hand_computed():
    # ranked: hit (0.9), false alarm (0.8), hit (0.7)  ->  precision 1, 1/2, 2/3 at recall .5, .5, 1
    boxes = [[0, 0, 10, 10], [50, 50, 60, 60], [20, 20, 30, 30]]
    ap = ev.ap_at([result(boxes, [0.9, 0.8, 0.7])], 0.5)
    # envelope is 1.0 up to recall 0.5 (51 grid points) and 2/3 beyond (50 points)
    assert ap == pytest.approx((51 * 1.0 + 50 * 2 / 3) / 101)


def test_duplicate_detection_is_a_false_positive():
    is_tp, ignored, n_gt = ev.match_image(result([[0, 0, 10, 10], [0, 0, 10, 10]], [0.9, 0.8]), 0.5)
    assert is_tp.tolist() == [True, False] and not ignored.any() and n_gt == 2


def test_higher_score_claims_the_object_first():
    boxes = [[0, 0, 10, 8], [0, 0, 10, 10]]  # lower-scored box fits better but comes second
    is_tp, _, _ = ev.match_image(result(boxes, [0.5, 0.9]), 0.5)
    assert is_tp.tolist() == [False, True]


def test_iou_threshold_is_respected():
    r = result([[0, 0, 10, 6]], [0.9])  # IoU 0.6 with the first object
    assert ev.match_image(r, 0.5)[0].tolist() == [True]
    assert ev.match_image(r, 0.75)[0].tolist() == [False]


def test_size_range_ignores_instead_of_punishing():
    gt = [[0, 0, 10, 10], [100, 100, 160, 160]]
    r = result([[0, 0, 10, 10], [100, 100, 160, 160], [200, 200, 205, 205]], [0.9, 0.8, 0.7], gt)
    is_tp, ignored, n_gt = ev.match_image(r, 0.5, size_range=(45.0, 1e9))
    assert n_gt == 1
    assert is_tp.tolist() == [False, True, False]
    assert ignored.tolist() == [True, False, True]  # small object's match, and a small stray box
    assert ev.ap_at([r], 0.5, size_range=(45.0, 1e9)) == pytest.approx(1.0)


def test_operating_point_counts():
    boxes = [[0, 0, 10, 10], [50, 50, 60, 60], [20, 20, 30, 30]]
    op = ev.operating_point([result(boxes, [0.9, 0.8, 0.3])], threshold=0.5)
    assert (op["tp"], op["fp"], op["fn"]) == (1, 1, 1)
    assert op["precision"] == op["recall"] == op["f1"] == pytest.approx(0.5)


def test_best_f1_threshold_picks_the_knee():
    boxes = [[0, 0, 10, 10], [20, 20, 30, 30], [50, 50, 60, 60], [70, 70, 80, 80]]
    threshold = ev.best_f1_threshold([result(boxes, [0.9, 0.8, 0.2, 0.1])])
    assert threshold == pytest.approx(0.8)


def test_images_are_pooled_not_averaged():
    easy = result(GT, [0.9, 0.8])
    hard = result([[50, 50, 60, 60]], [0.95], gt=[[0, 0, 10, 10]])
    # ranked over both images: false alarm, hit, hit; 3 objects in total
    ap = ev.ap_at([easy, hard], 0.5)
    recall_levels = np.linspace(0, 1, 101)
    expected = np.where(recall_levels <= 2 / 3, 2 / 3, 0.0).mean()
    assert ap == pytest.approx(expected)


def test_bootstrap_interval_brackets_the_estimate():
    rng = np.random.default_rng(0)
    results = _random_results(rng, 40)
    lo, hi = ev.bootstrap_ap(results, n_boot=200)
    assert lo <= ev.ap_at(results, 0.5) <= hi and hi - lo < 0.5


def _random_results(rng, n_images):
    results = []
    for _ in range(n_images):
        n_gt = int(rng.integers(0, 8))
        xy = rng.uniform(0, 200, size=(n_gt, 2))
        gt = np.concatenate([xy, xy + rng.uniform(8, 60, size=(n_gt, 2))], axis=1)
        # detections: jittered copies of some objects plus random false alarms
        keep = rng.random(n_gt) < 0.8
        jittered = gt[keep] + rng.normal(0, 4, size=(int(keep.sum()), 4))
        jittered[:, 2:] = np.maximum(jittered[:, 2:], jittered[:, :2] + 2)  # keep boxes valid
        n_fp = int(rng.integers(0, 5))
        xy = rng.uniform(0, 200, size=(n_fp, 2))
        stray = np.concatenate([xy, xy + rng.uniform(8, 60, size=(n_fp, 2))], axis=1)
        boxes = np.concatenate([jittered, stray]).reshape(-1, 4)
        results.append(result(boxes, rng.random(len(boxes)), gt))
    return results


def test_matches_pycocotools():
    """Independent check of AP50 / AP75 / mAP against the reference implementation."""
    pytest.importorskip("pycocotools")
    import contextlib
    import io

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    results = _random_results(np.random.default_rng(7), 60)
    images, annotations, detections = [], [], []
    for image_id, r in enumerate(results, start=1):
        images.append({"id": image_id, "width": 300, "height": 300})
        for x1, y1, x2, y2 in r.gt:
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "area": float((x2 - x1) * (y2 - y1)),
                    "iscrowd": 0,
                }
            )
        for (x1, y1, x2, y2), score in zip(r.boxes, r.scores, strict=True):
            detections.append(
                {
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": float(score),
                }
            )
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO()
        coco.dataset = {
            "images": images,
            "annotations": annotations,
            "categories": [{"id": 1, "name": "airplane"}],
        }
        coco.createIndex()
        coco_eval = COCOeval(coco, coco.loadRes(detections), "bbox")
        coco_eval.params.maxDets = [1, 10, 1000]
        coco_eval.evaluate()
        coco_eval.accumulate()
    precision = coco_eval.eval["precision"][:, :, 0, 0, 2]  # [iou, recall], all areas, maxDets=1000
    report = ev.evaluate(results, n_boot=0, with_curve=False)
    assert report["AP50"] == pytest.approx(precision[0].mean(), abs=1e-6)
    assert report["AP75"] == pytest.approx(precision[5].mean(), abs=1e-6)
    assert report["mAP"] == pytest.approx(precision.mean(), abs=1e-6)
