import numpy as np
import pytest

from airplane_detector import data


def test_parse_annotation_file_ignores_count_line(tmp_path):
    path = tmp_path / "a.csv"
    path.write_text("2\n1 2 30 40\n5 6 70 80\n")
    np.testing.assert_array_equal(data.parse_annotation_file(path), [[1, 2, 30, 40], [5, 6, 70, 80]])


def test_parse_annotation_file_tolerates_junk(tmp_path):
    path = tmp_path / "a.csv"
    path.write_text("three\n1 2 30 40\nnot a box\n\n1,2,3,4\n")
    assert data.parse_annotation_file(path).shape == (2, 4)


def test_parse_empty_annotation(tmp_path):
    path = tmp_path / "a.csv"
    path.write_text("0\n")
    assert data.parse_annotation_file(path).shape == (0, 4)


def test_clean_boxes_clips_orders_and_drops_degenerate():
    boxes = [[250, 10, 257, 60], [40, 40, 10, 10], [5, 5, 6, 50], [-4, -4, 20, 20]]
    out = data.clean_boxes(boxes, height=256, width=256)
    assert out.tolist() == [[250, 10, 256, 60], [10, 10, 40, 40], [0, 0, 20, 20]]


def test_load_dataset(toy_dataset):
    samples = data.load_dataset(toy_dataset)
    assert [s.name for s in samples] == [f"img_{i}" for i in range(6)]
    assert [len(s.boxes) for s in samples] == list(range(6))
    assert samples[3].load().shape == (128, 128, 3)


def test_load_dataset_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_dataset(tmp_path)


def test_excluded_images_are_skipped(toy_dataset, monkeypatch):
    monkeypatch.setitem(data.EXCLUDED, "img_2", "test")
    assert "img_2" not in [s.name for s in data.load_dataset(toy_dataset)]


def _fake_samples(n=200):
    rng = np.random.default_rng(3)
    return [
        data.Sample(f"im{i:03d}", "", np.zeros((int(rng.integers(0, 40)), 4), np.float32)) for i in range(n)
    ]


def test_splits_are_disjoint_complete_and_deterministic():
    samples = _fake_samples()
    a = data.make_splits(samples, seed=7)
    b = data.make_splits(list(reversed(samples)), seed=7)
    assert a == b, "split must not depend on input order"
    names = a["train"] + a["val"] + a["test"]
    assert len(names) == len(set(names)) == len(samples)
    assert len(a["val"]) == pytest.approx(30, abs=3)
    assert len(a["test"]) == pytest.approx(30, abs=3)
    assert data.make_splits(samples, seed=8) != a


def test_splits_are_stratified_by_object_count():
    samples = _fake_samples(400)
    splits = data.make_splits(samples, seed=0)
    by_name = {s.name: len(s.boxes) for s in samples}
    dense = {k: np.mean([by_name[n] >= 30 for n in v]) for k, v in splits.items()}
    assert max(dense.values()) - min(dense.values()) < 0.05


def test_split_roundtrip(tmp_path):
    samples = _fake_samples(40)
    splits = data.make_splits(samples)
    data.save_splits(splits, tmp_path / "s.json")
    assert data.load_splits(tmp_path / "s.json") == splits
    assert [s.name for s in data.get_split(samples, splits, "val")] == splits["val"]


def test_get_split_reports_missing_images():
    with pytest.raises(KeyError):
        data.get_split(_fake_samples(3), {"train": ["nope"]}, "train")
