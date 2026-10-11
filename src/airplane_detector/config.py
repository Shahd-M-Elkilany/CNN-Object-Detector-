"""Central configuration.

Every number that changes the behaviour of the detector lives here, so an
experiment is fully described by one small YAML file (see ``configs/``).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    # ------------------------------------------------------------------ data
    data_dir: str = "data/raw"
    splits_file: str = "data/splits.json"
    split_seed: int = 42
    val_fraction: float = 0.15
    test_fraction: float = 0.15

    # ----------------------------------------------------------------- model
    #: Side of the square receptive field of the network, in pixels.
    window: int = 40
    #: Side of the reference box (the single "anchor") centred in the window.
    #: Objects are rescaled by the image pyramid so that they roughly fill it.
    anchor: int = 24
    #: Total stride of the network (fixed by the three 2x2 max-pools).
    stride: int = 8
    #: Channel widths of the four convolutional stages.
    widths: tuple[int, int, int, int] = (24, 48, 96, 192)
    dropout: float = 0.3

    # ------------------------------------------------------------- sampling
    #: A window is a positive when IoU(anchor, ground truth) >= pos_iou ...
    pos_iou: float = 0.5
    #: ... and a negative when its IoU with every ground-truth box is < neg_iou.
    neg_iou: float = 0.3
    #: Jittered positive windows drawn per ground-truth box and per epoch.
    pos_per_box: int = 8
    #: Negatives drawn per positive.
    neg_per_pos: float = 3.0
    #: Share of negatives drawn next to an object (wrong scale / off-centre).
    near_miss_fraction: float = 0.3
    #: Share of negatives taken from the mined hard-negative pool (stage 2).
    hard_fraction: float = 0.5

    # ------------------------------------------------------------- training
    seed: int = 0
    batch_size: int = 128
    epochs_stage1: int = 14
    epochs_stage2: int = 8
    #: Number of hard-negative mining rounds; 0 disables stage 2 entirely.
    mining_rounds: int = 2
    lr: float = 2e-3
    lr_stage2: float = 5e-4
    weight_decay: float = 1e-4
    box_loss_weight: float = 2.0
    #: Detections scoring above this on *training* images that do not overlap
    #: any object are kept as hard negatives.
    mining_score: float = 0.2
    mining_max_per_image: int = 60

    # ------------------------------------------------------------ inference
    #: Smallest / largest object side (px, original image) the pyramid covers.
    min_object: float = 8.0
    max_object: float = 280.0
    #: Pyramid levels per octave (3 -> consecutive scales differ by 2**(1/3)).
    scales_per_octave: int = 3
    #: Border (in network-input pixels) added around every pyramid level so
    #: that objects touching the image edge can still be centred in a window.
    pad: int = 16
    #: Detections below this score are dropped before NMS. Kept low so the
    #: precision-recall curve is traced all the way out.
    score_floor: float = 0.05
    nms_iou: float = 0.3
    #: Box voting: every kept box is replaced by the score-weighted mean of
    #: the candidates overlapping it by at least this IoU (0 disables it).
    vote_iou: float = 0.5
    max_detections: int = 300
    #: Pyramid levels larger than this many pixels are processed in tiles.
    max_level_pixels: int = 1_500_000
    #: Score threshold used when drawing / counting detections. Overwritten by
    #: the value calibrated on the validation split after training.
    score_threshold: float = 0.5

    # ---------------------------------------------------------------- misc
    output_dir: str = "runs/default"
    extra: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ io
    @classmethod
    def from_yaml(cls, path: str | Path) -> Config:
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> Config:
        known = {f.name for f in fields(cls)}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        raw = dict(raw)
        if "widths" in raw:
            raw["widths"] = tuple(raw["widths"])
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["widths"] = list(self.widths)
        return out

    def to_yaml(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False)

    def __post_init__(self) -> None:
        self.widths = tuple(self.widths)
        if self.window % 2:
            raise ValueError("window must be even")
        if self.pad % self.stride:
            raise ValueError("pad must be a multiple of the network stride")
        if not 0 < self.anchor <= self.window:
            raise ValueError("anchor must fit inside the window")
        if not self.neg_iou <= self.pos_iou:
            raise ValueError("neg_iou must not exceed pos_iou")
