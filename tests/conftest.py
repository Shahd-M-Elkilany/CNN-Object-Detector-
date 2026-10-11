"""Shared fixtures. The whole suite runs on synthetic data - no download needed."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from airplane_detector.config import Config


@pytest.fixture
def cfg() -> Config:
    # Narrow network: the tests check behaviour, not accuracy.
    return Config(widths=(4, 8, 8, 16), dropout=0.0)


@pytest.fixture
def toy_dataset(tmp_path):
    """Six 128x128 images with bright square 'objects' and matching CSV files."""
    rng = np.random.default_rng(0)
    (tmp_path / "Images").mkdir()
    (tmp_path / "Airplanes_Annotations").mkdir()
    for i in range(6):
        img = rng.integers(0, 60, size=(128, 128, 3), dtype=np.uint8)
        boxes = []
        for _ in range(i):  # image 0 is empty, image i has i objects
            side = int(rng.integers(16, 40))
            x, y = (int(v) for v in rng.integers(0, 128 - side, size=2))
            img[y : y + side, x : x + side] = 255
            boxes.append((x, y, x + side, y + side))
        cv2.imwrite(str(tmp_path / "Images" / f"img_{i}.jpg"), img)
        lines = [str(len(boxes))] + [" ".join(map(str, b)) for b in boxes]
        (tmp_path / "Airplanes_Annotations" / f"img_{i}.csv").write_text("\n".join(lines) + "\n")
    return tmp_path
