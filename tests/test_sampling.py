import numpy as np
import pytest

from airplane_detector import sampling
from airplane_detector.boxes import decode, iou_matrix
from airplane_detector.data import load_dataset


def test_pyramid_covers_the_configured_object_range(cfg):
    scales = sampling.pyramid_scales(cfg)
    sizes = cfg.anchor / scales
    assert sizes.min() == pytest.approx(cfg.min_object)
    assert sizes.max() >= cfg.max_object
    ratios = scales[:-1] / scales[1:]
    np.testing.assert_allclose(ratios, 2 ** (1 / cfg.scales_per_octave), rtol=1e-6)


def test_pyramid_drops_levels_for_objects_larger_than_the_image(cfg):
    scales = sampling.pyramid_scales(cfg, 64, 64)
    assert (cfg.anchor / scales).max() <= 64 * 1.01
    assert len(scales) < len(sampling.pyramid_scales(cfg))


def test_window_and_anchor_geometry(cfg):
    window = sampling.window_box(100, 60, 0.5, cfg.window)
    assert (window[2] - window[0], window[3] - window[1]) == (80, 80)
    anchor = sampling.anchor_of(window, cfg)
    np.testing.assert_allclose(anchor, [76, 36, 124, 84])


def test_extract_window_inside_and_across_the_border():
    img = np.arange(100 * 100 * 3, dtype=np.uint32).reshape(100, 100, 3).astype(np.uint8)
    inside = sampling.extract_window(img, np.array([10, 20, 50, 60]), 40)
    np.testing.assert_array_equal(inside, img[20:60, 10:50])
    outside = sampling.extract_window(img, np.array([-10, -10, 30, 30]), 40)
    assert outside.shape == (40, 40, 3)
    np.testing.assert_array_equal(outside[10:, 10:], img[:30, :30])
    resized = sampling.extract_window(img, np.array([0, 0, 80, 80]), 40)
    assert resized.shape == (40, 40, 3)


@pytest.mark.parametrize("k", [0, 1, 2, 3])
@pytest.mark.parametrize("flip", [False, True])
def test_dihedral_moves_boxes_with_the_pixels(k, flip):
    patch = np.zeros((40, 40, 3), np.uint8)
    box = np.array([[4, 10, 18, 22]], np.float32)  # deliberately off-centre and non-square
    patch[10:22, 4:18] = 255
    out, new_box = sampling.dihedral(patch, box, k, flip)
    ys, xs = np.nonzero(out[:, :, 0])
    np.testing.assert_allclose(new_box[0], [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])


def test_photometric_keeps_dtype_and_range():
    rng = np.random.default_rng(0)
    patch = rng.integers(0, 256, size=(40, 40, 3), dtype=np.uint8)
    out = sampling.photometric(patch, rng)
    assert out.dtype == np.uint8 and out.shape == patch.shape


@pytest.fixture
def sampler(toy_dataset, cfg):
    return sampling.PatchSampler(load_dataset(toy_dataset), cfg, augment=True)


def test_epoch_shapes_and_class_ratio(sampler, cfg):
    x, y = sampler.sample_epoch(np.random.default_rng(0))
    assert x.dtype == np.uint8 and x.shape[1:] == (cfg.window, cfg.window, 3)
    assert y.shape == (len(x), 1, 1, 5)
    labels = y[:, 0, 0, 0]
    n_pos = int(labels.sum())
    assert n_pos == 15 * cfg.pos_per_box  # 0+1+2+3+4+5 objects
    assert len(labels) - n_pos == pytest.approx(n_pos * cfg.neg_per_pos, rel=0.05)
    assert np.all(y[labels == 0, 0, 0, 1:] == 0), "negatives carry no box target"


def test_positive_targets_point_at_the_object(sampler, cfg):
    """Decoding a positive's target must land on a bright square in its patch."""
    x, y = sampler.sample_epoch(np.random.default_rng(1))
    half = (cfg.window - cfg.anchor) / 2
    anchor = np.array([[half, half, half + cfg.anchor, half + cfg.anchor]], np.float32)
    checked = 0
    for patch, target in zip(x, y[:, 0, 0], strict=True):
        if target[0] != 1:
            continue
        box = decode(anchor, target[None, 1:])[0]
        assert iou_matrix(anchor, box[None])[0, 0] >= cfg.neg_iou
        x1, y1, x2, y2 = np.clip(box.round().astype(int) + [2, 2, -2, -2], 0, cfg.window)
        if x2 > x1 and y2 > y1:
            assert patch[y1:y2, x1:x2].mean() > 150
            checked += 1
    assert checked > 50


def test_negatives_do_not_overlap_objects(sampler, cfg):
    rng = np.random.default_rng(2)
    for draw in (sampler.sample_random_negative, sampler.sample_near_miss):
        for _ in range(100):
            item = draw(rng)
            assert item is not None and item[1][0] == 0


def test_hard_negative_overlapping_an_object_is_rejected(sampler, cfg):
    rng = np.random.default_rng(3)
    box = sampler.samples[1].boxes[0]
    on_object = sampling.HardNegative(
        1, (box[0] + box[2]) / 2, (box[1] + box[3]) / 2, cfg.anchor / (box[2] - box[0])
    )
    assert sampler.sample_hard_negative(on_object, rng) is None
    background = sampling.HardNegative(0, 64.0, 64.0, 1.0)  # image 0 has no objects
    patch, target = sampler.sample_hard_negative(background, rng)
    assert patch.shape == (cfg.window, cfg.window, 3) and target[0] == 0


def test_sampling_is_reproducible(sampler):
    a = sampler.sample_epoch(np.random.default_rng(5))
    b = sampler.sample_epoch(np.random.default_rng(5))
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])
