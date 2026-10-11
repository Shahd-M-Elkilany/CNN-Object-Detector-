import numpy as np
import pytest

from airplane_detector.boxes import clip_boxes, decode, encode, iou_matrix, nms


def test_iou_known_values():
    a = [[0, 0, 10, 10]]
    b = [[0, 0, 10, 10], [5, 0, 15, 10], [10, 10, 20, 20], [2, 2, 8, 8]]
    np.testing.assert_allclose(iou_matrix(a, b)[0], [1.0, 1 / 3, 0.0, 0.36], atol=1e-6)


def test_iou_empty_inputs():
    assert iou_matrix(np.zeros((0, 4)), [[0, 0, 1, 1]]).shape == (0, 1)
    assert iou_matrix([[0, 0, 1, 1]], []).shape == (1, 0)


def test_iou_is_symmetric():
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 50, size=(20, 2))
    boxes = np.concatenate([xy, xy + rng.uniform(1, 30, size=(20, 2))], axis=1)
    m = iou_matrix(boxes, boxes)
    np.testing.assert_allclose(m, m.T, atol=1e-6)
    np.testing.assert_allclose(np.diag(m), 1.0, atol=1e-6)


def test_nms_suppresses_overlaps_and_keeps_best():
    boxes = [[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60], [0, 0, 10, 10]]
    scores = [0.6, 0.9, 0.3, 0.5]
    assert nms(boxes, scores, 0.5).tolist() == [1, 2]


def test_nms_threshold_one_keeps_everything():
    boxes = [[0, 0, 10, 10], [1, 1, 11, 11]]
    assert sorted(nms(boxes, [0.1, 0.2], 1.0).tolist()) == [0, 1]


def test_nms_empty():
    assert nms(np.zeros((0, 4)), np.zeros(0), 0.5).size == 0


def test_encode_decode_roundtrip():
    rng = np.random.default_rng(1)
    xy = rng.uniform(0, 100, size=(50, 2))
    anchors = np.concatenate([xy, xy + rng.uniform(10, 40, size=(50, 2))], axis=1)
    xy = rng.uniform(0, 100, size=(50, 2))
    gt = np.concatenate([xy, xy + rng.uniform(10, 40, size=(50, 2))], axis=1)
    np.testing.assert_allclose(decode(anchors, encode(anchors, gt)), gt, atol=1e-3)


def test_zero_deltas_return_the_anchor():
    anchors = np.array([[8, 8, 32, 32]], np.float32)
    np.testing.assert_allclose(decode(anchors, np.zeros((1, 4))), anchors)
    np.testing.assert_allclose(encode(anchors, anchors), 0, atol=1e-6)


def test_decode_clamps_extreme_sizes():
    out = decode([[0, 0, 10, 10]], [[0, 0, 50.0, -50.0]])[0]
    assert out[2] - out[0] == pytest.approx(10 * np.exp(1.5), rel=1e-4)
    assert out[3] - out[1] == pytest.approx(10 * np.exp(-1.5), rel=1e-4)


def test_clip_boxes():
    out = clip_boxes([[-5, -5, 300, 20]], height=100, width=200)
    assert out.tolist() == [[0, 0, 200, 20]]


def test_vote_boxes_is_a_score_weighted_mean():
    from airplane_detector.boxes import vote_boxes

    candidates = np.array([[0, 0, 10, 10], [2, 0, 12, 10], [50, 50, 60, 60]], np.float32)
    scores = np.array([0.75, 0.25, 0.9], np.float32)
    voted = vote_boxes(candidates[[0, 2]], candidates, scores, iou_threshold=0.5)
    np.testing.assert_allclose(voted[0], [0.5, 0, 10.5, 10], atol=1e-5)  # pulled a quarter of the way
    np.testing.assert_allclose(voted[1], [50, 50, 60, 60], atol=1e-5)  # nothing overlaps it but itself


def test_vote_boxes_empty():
    from airplane_detector.boxes import vote_boxes

    assert vote_boxes(np.zeros((0, 4)), np.zeros((0, 4)), np.zeros(0), 0.5).shape == (0, 4)
