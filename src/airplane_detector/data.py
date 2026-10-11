"""Dataset access: download, annotation parsing, cleaning and image-level splits.

The data is the public *Airplanes* aerial-imagery set distributed with
https://github.com/1297rohit/RCNN (Apache-2.0): 733 JPEG tiles and one CSV per
image holding axis-aligned airplane boxes. It is downloaded on demand and is
not redistributed in this repository.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .boxes import as_boxes

# Pinned to an exact commit so the data can never change underneath the code.
_SOURCE_COMMIT = "6c71fdce8dffb32c0ee719c194eff6319bd30a3c"
_SOURCE_URL = "https://raw.githubusercontent.com/1297rohit/RCNN/{commit}/{name}"
ARCHIVES = {
    "Images.zip": "701580c3eef3606c282f59a3c9194562b7a31e553249e248ae0e8165fd509475",
    "Airplanes_Annotations.zip": "0c57f0887abb998e72b92ba749b5d368b17c611fe231ad427c1fb2020556466c",
}

#: Images removed during cleaning, with the reason. See docs/DATA.md.
EXCLUDED = {
    "Lhasa": "2637x4800 full scene with an empty annotation file (planes visible but unlabelled)",
    "hotanairport5-rgb": "byte-identical duplicate of hotanairport5",
}

#: Boxes with a side shorter than this (after clipping) are dropped as noise.
MIN_BOX_SIDE = 3


@dataclass(frozen=True)
class Sample:
    name: str
    image_path: str
    boxes: np.ndarray  # (N, 4) float32, x1 y1 x2 y2, clipped to the image

    def load(self) -> np.ndarray:
        """Read the image as BGR uint8 (OpenCV convention)."""
        img = cv2.imread(self.image_path, cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(self.image_path)
        return img


# --------------------------------------------------------------------------
# download
# --------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_dataset(data_dir: str | Path, force: bool = False) -> Path:
    """Download and unpack the dataset into ``data_dir``; verify checksums."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    if not force and (data_dir / "Images").is_dir() and (data_dir / "Airplanes_Annotations").is_dir():
        return data_dir
    for name, expected in ARCHIVES.items():
        archive = data_dir / name
        if force or not archive.exists():
            url = _SOURCE_URL.format(commit=_SOURCE_COMMIT, name=name)
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, archive)  # noqa: S310 - fixed https URL
        actual = _sha256(archive)
        if expected and actual != expected:
            raise RuntimeError(f"checksum mismatch for {name}: got {actual}, expected {expected}")
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(data_dir)
    return data_dir


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------
def parse_annotation_file(path: str | Path) -> np.ndarray:
    """Parse one annotation CSV into an (N, 4) array of ``x1 y1 x2 y2``.

    The format is a count on the first line followed by one whitespace
    separated box per line. Lines that are not four integers are ignored, so a
    wrong or missing count line does not break parsing.
    """
    boxes: list[list[float]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            parts = line.replace(",", " ").split()
            if len(parts) != 4:
                continue
            try:
                boxes.append([float(p) for p in parts])
            except ValueError:
                continue
    return as_boxes(boxes)


def clean_boxes(boxes: np.ndarray, height: int, width: int) -> np.ndarray:
    """Order corners, clip to the image and drop degenerate boxes."""
    boxes = as_boxes(boxes)
    if len(boxes) == 0:
        return boxes
    x1 = np.minimum(boxes[:, 0], boxes[:, 2])
    x2 = np.maximum(boxes[:, 0], boxes[:, 2])
    y1 = np.minimum(boxes[:, 1], boxes[:, 3])
    y2 = np.maximum(boxes[:, 1], boxes[:, 3])
    out = np.stack(
        [np.clip(x1, 0, width), np.clip(y1, 0, height), np.clip(x2, 0, width), np.clip(y2, 0, height)],
        axis=1,
    )
    keep = ((out[:, 2] - out[:, 0]) >= MIN_BOX_SIDE) & ((out[:, 3] - out[:, 1]) >= MIN_BOX_SIDE)
    return out[keep]


def load_dataset(data_dir: str | Path) -> list[Sample]:
    """Load every annotated image, applying the cleaning rules above."""
    data_dir = Path(data_dir)
    image_dir, ann_dir = data_dir / "Images", data_dir / "Airplanes_Annotations"
    if not image_dir.is_dir() or not ann_dir.is_dir():
        raise FileNotFoundError(f"Dataset not found under {data_dir}. Run `airdet download` first.")
    samples: list[Sample] = []
    for image_path in sorted(image_dir.glob("*.jpg")):
        name = image_path.stem
        ann_path = ann_dir / f"{name}.csv"
        if name in EXCLUDED or not ann_path.exists():
            continue
        # Reading only the header would be faster but OpenCV has no such API;
        # 733 small JPEGs decode in well under a second.
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        height, width = img.shape[:2]
        boxes = clean_boxes(parse_annotation_file(ann_path), height, width)
        samples.append(Sample(name=name, image_path=str(image_path), boxes=boxes))
    return samples


# --------------------------------------------------------------------------
# splits
# --------------------------------------------------------------------------
def _count_bucket(n: int) -> int:
    for bucket, upper in enumerate((0, 1, 3, 9, 29)):
        if n <= upper:
            return bucket
    return 5


def make_splits(
    samples: list[Sample], seed: int = 42, val_fraction: float = 0.15, test_fraction: float = 0.15
) -> dict[str, list[str]]:
    """Split by *image*, stratified by the number of objects per image.

    Splitting by image (rather than by crop, as the original notebook did)
    guarantees that no pixel of a validation or test image is ever seen during
    training. Stratifying keeps the handful of very dense scenes from all
    landing in one split.
    """
    rng = np.random.default_rng(seed)
    buckets: dict[int, list[str]] = {}
    for s in samples:
        buckets.setdefault(_count_bucket(len(s.boxes)), []).append(s.name)
    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for bucket in sorted(buckets):
        names = sorted(buckets[bucket])
        rng.shuffle(names)
        n_val = int(round(len(names) * val_fraction))
        n_test = int(round(len(names) * test_fraction))
        splits["val"] += names[:n_val]
        splits["test"] += names[n_val : n_val + n_test]
        splits["train"] += names[n_val + n_test :]
    return {k: sorted(v) for k, v in splits.items()}


def save_splits(splits: dict[str, list[str]], path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(splits, fh, indent=1)
        fh.write("\n")


def load_splits(path: str | Path) -> dict[str, list[str]]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def get_split(samples: list[Sample], splits: dict[str, list[str]], which: str) -> list[Sample]:
    by_name = {s.name: s for s in samples}
    missing = [n for n in splits[which] if n not in by_name]
    if missing:
        raise KeyError(f"{len(missing)} images of split '{which}' are missing, e.g. {missing[:3]}")
    return [by_name[n] for n in splits[which]]
