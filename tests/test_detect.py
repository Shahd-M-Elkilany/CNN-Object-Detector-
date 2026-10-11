import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from airplane_detector.config import Config  # noqa: E402
from airplane_detector.detect import Detections, Detector, prepare_level  # noqa: E402
from airplane_detector.model import build_model  # noqa: E402


def one_hot_model(row: int, col: int, deltas=(0.0, 0.0, 0.0, 0.0)):
    """A fake network that fires on exactly one output cell."""

    def model(x, training=False):
        shape = tf.shape(x)
        rows, cols = (shape[1] - 40) // 8 + 1, (shape[2] - 40) // 8 + 1
        logit = tf.fill([1, rows, cols, 1], -20.0) + tf.scatter_nd(
            [[0, row, col, 0]], [40.0], [1, rows, cols, 1]
        )
        box = tf.tile(tf.reshape(tf.constant(deltas, tf.float32), [1, 1, 1, 4]), [1, rows, cols, 1])
        return tf.concat([logit, box], axis=-1)

    return model


@pytest.mark.parametrize(
    "size,scale", [((256, 256), 1.0), ((196, 262), 0.75), ((30, 50), 0.1), ((427, 620), 3.0)]
)
def test_prepare_level_fits_the_stride_grid(cfg, size, scale):
    img = np.zeros((*size, 3), np.uint8)
    level, sx, sy = prepare_level(img, scale, cfg)
    for dim in level.shape[:2]:
        assert dim >= cfg.window and (dim - cfg.window) % cfg.stride == 0
    assert sx == pytest.approx(scale, rel=0.2) and sy == pytest.approx(scale, rel=0.2)


def test_cell_maps_back_to_the_right_pixels(cfg):
    img = np.zeros((256, 256, 3), np.uint8)
    det = Detector(one_hot_model(row=5, col=9), cfg).detect(img, scales=np.array([1.0]))
    assert len(det) == 1 and det.scores[0] > 0.99
    # window top-left = 8 * (col, row) - pad; anchor = its central 24 px square
    np.testing.assert_allclose(
        det.boxes[0], [9 * 8 - 16 + 8, 5 * 8 - 16 + 8, 9 * 8 - 16 + 32, 5 * 8 - 16 + 32]
    )
    np.testing.assert_allclose(det.anchors[0], det.boxes[0])


def test_pyramid_scale_is_undone(cfg):
    img = np.zeros((256, 256, 3), np.uint8)
    full = Detector(one_hot_model(5, 9), cfg).detect(img, scales=np.array([1.0])).boxes[0]
    double = Detector(one_hot_model(5, 9), cfg).detect(img, scales=np.array([2.0])).boxes[0]
    np.testing.assert_allclose(double, full / 2, atol=1e-4)


def test_regression_offsets_are_applied_and_can_be_disabled(cfg):
    img = np.zeros((256, 256, 3), np.uint8)
    detector = Detector(one_hot_model(5, 9, deltas=(0.5, 0.0, np.log(2.0), 0.0)), cfg)
    moved = detector.detect(img, scales=np.array([1.0]))
    plain = detector.detect(img, scales=np.array([1.0]), use_regression=False)
    anchor = plain.boxes[0]
    np.testing.assert_allclose(moved.anchors[0], anchor)
    # shifted right by half an anchor (12 px) and twice as wide (48 px)
    np.testing.assert_allclose(
        moved.boxes[0], [anchor[0] + 12 - 12, anchor[1], anchor[2] + 12 + 12, anchor[3]], atol=1e-4
    )


def test_boxes_are_clipped_to_the_image(cfg):
    img = np.zeros((64, 64, 3), np.uint8)
    det = Detector(one_hot_model(0, 0, deltas=(-1.0, -1.0, 0.0, 0.0)), cfg).detect(
        img, scales=np.array([1.0])
    )
    assert len(det) == 0 or (det.boxes.min() >= 0 and det.boxes.max() <= 64)


def test_score_threshold_filters(cfg):
    img = np.zeros((128, 128, 3), np.uint8)
    detector = Detector(one_hot_model(2, 2), cfg)
    assert len(detector.detect(img, scales=np.array([1.0]), score_threshold=0.5)) == 1
    assert len(detector.detect(img, scales=np.array([1.0]), score_threshold=1.1)) == 0


def test_tiled_forward_equals_single_pass():
    tf.keras.utils.set_random_seed(0)
    small = Config(widths=(4, 8, 8, 16), dropout=0.0)
    model = build_model(small)
    level = np.random.default_rng(0).integers(0, 255, (40 + 8 * 30, 40 + 8 * 41, 3)).astype(np.uint8)
    whole = Detector(model, small).forward(level)
    tiled_cfg = Config(widths=(4, 8, 8, 16), dropout=0.0, max_level_pixels=120 * 120)
    tiled = Detector(model, tiled_cfg).forward(level)
    assert whole.shape == (31, 42, 5)
    np.testing.assert_allclose(tiled, whole, atol=1e-4)


def test_real_network_runs_on_odd_sized_images(cfg):
    tf.keras.utils.set_random_seed(0)
    detector = Detector(build_model(cfg), cfg)
    img = np.random.default_rng(0).integers(0, 255, (97, 143, 3)).astype(np.uint8)
    det = detector.detect(img)
    assert isinstance(det, Detections)
    assert det.boxes.shape == (len(det), 4) and len(det) <= cfg.max_detections
    if len(det):
        assert det.boxes[:, [0, 2]].max() <= 143 and det.boxes[:, [1, 3]].max() <= 97
        assert np.all(np.diff(det.scores) <= 1e-6), "detections are sorted by score"


def test_load_reads_model_and_sibling_config(cfg, tmp_path):
    model = build_model(cfg)
    model.save(tmp_path / "net.keras")
    cfg.score_threshold = 0.77
    cfg.to_yaml(tmp_path / "net.yaml")
    detector = Detector.load(tmp_path / "net.keras")
    assert detector.cfg.score_threshold == 0.77 and detector.cfg.widths == cfg.widths


def test_merge_votes_and_can_be_disabled(cfg):
    boxes = np.array([[0, 0, 10, 10], [2, 0, 12, 10]], np.float32)
    det = Detections(boxes, np.array([0.75, 0.25], np.float32), boxes.copy(), np.ones(2, np.float32))
    detector = Detector(one_hot_model(0, 0), cfg)
    plain = detector.merge(det, nms_iou=0.3, vote_iou=0.0)
    voted = detector.merge(det, nms_iou=0.3, vote_iou=0.5)
    assert len(plain) == len(voted) == 1
    np.testing.assert_allclose(plain.boxes[0], [0, 0, 10, 10])
    np.testing.assert_allclose(voted.boxes[0], [0.5, 0, 10.5, 10], atol=1e-5)
