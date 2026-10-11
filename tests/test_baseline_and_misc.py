import json

import numpy as np
import pytest

from airplane_detector import baseline as bl
from airplane_detector import viz
from airplane_detector.cli import build_parser
from airplane_detector.config import Config
from airplane_detector.data import Sample, load_dataset


def test_grid_windows_match_the_notebook_loop():
    windows = bl.grid_windows(256, 256)
    assert len(windows) == 49  # 7 x 7 positions
    assert windows[0].tolist() == [0, 0, 64, 64] and windows[-1].tolist() == [192, 192, 256, 256]
    assert bl.grid_windows(50, 300).shape == (0, 4)


def test_recall_ceiling(tmp_path):
    import cv2

    path = str(tmp_path / "blank.jpg")
    cv2.imwrite(path, np.zeros((256, 256, 3), np.uint8))
    # One object that coincides with a grid window, one that is far too small for any 64 px window.
    sample = Sample("blank", path, np.array([[64, 64, 128, 128], [10, 10, 34, 34]], np.float32))
    assert bl.window_recall_ceiling([sample]) == pytest.approx(0.5)


def test_make_crops_labels_and_ratio(tmp_path, toy_dataset):
    import cv2

    img = np.zeros((256, 256, 3), np.uint8)
    img[64:128, 64:128] = 255
    path = str(tmp_path / "one.jpg")
    cv2.imwrite(path, img)
    sample = Sample("one", path, np.array([[64, 64, 128, 128]], np.float32))
    x, y = bl.make_crops([sample], np.random.default_rng(0))
    assert x.shape == (16, 64, 64, 3) and x.dtype == np.float32 and 0 <= x.min() and x.max() <= 1
    assert y.sum() == 1 and x[y == 1].mean() > 0.9, "the positive crop is the bright square"
    # The toy objects are 16-40 px: no 64 px window can reach IoU 0.5 with any of them.
    with pytest.raises(ValueError):
        bl.make_crops(load_dataset(toy_dataset), np.random.default_rng(0))


def test_config_yaml_roundtrip(tmp_path):
    cfg = Config(widths=(8, 16, 32, 64), score_threshold=0.42, extra={"note": "x"})
    cfg.to_yaml(tmp_path / "c.yaml")
    loaded = Config.from_yaml(tmp_path / "c.yaml")
    assert loaded == cfg and isinstance(loaded.widths, tuple)
    json.dumps(cfg.to_dict())  # must be JSON-serialisable


def test_config_rejects_unknown_keys_and_bad_values():
    with pytest.raises(ValueError):
        Config.from_dict({"learning_rate": 1.0})
    with pytest.raises(ValueError):
        Config(pad=12)
    with pytest.raises(ValueError):
        Config(neg_iou=0.7, pos_iou=0.5)


def test_shipped_config_matches_code_defaults():
    assert Config.from_yaml("configs/default.yaml") == Config()


def test_cli_parses_every_command():
    parser = build_parser()
    assert parser.parse_args(["download"]).data_dir == "data/raw"
    assert parser.parse_args(["train", "--seed", "3"]).seed == 3
    args = parser.parse_args(["evaluate", "--split", "val", "--no-regression"])
    assert args.split == "val" and args.no_regression and not args.single_scale
    assert parser.parse_args(["predict", "x.jpg", "--threshold", "0.3"]).threshold == 0.3
    assert parser.parse_args(["baseline"]).out == "results/baseline_v1.json"


def test_draw_evaluation_colours():
    img = np.zeros((100, 100, 3), np.uint8)
    gt = [[10, 10, 40, 40], [60, 60, 90, 90]]
    out = viz.draw_evaluation(img, gt, [[10, 10, 40, 40], [60, 10, 90, 40]], min_side=100)
    assert tuple(out[10, 25]) == viz.GREEN  # matched prediction
    assert tuple(out[10, 75]) == viz.RED  # false alarm
    assert tuple(out[60, 75]) == viz.YELLOW  # missed object


def test_draw_detections_upscales_small_tiles():
    out = viz.draw_detections(np.zeros((64, 128, 3), np.uint8), [[5, 5, 30, 30]], [0.9], min_side=256)
    assert out.shape == (256, 512, 3) and out.any()
