"""Training: patch pre-training followed by hard-negative mining rounds.

Stage 1 trains on object-centred positives, random background and "near miss"
windows. The resulting detector is then run over the *training* images; every
confident window that does not overlap an object is a mistake the network
actually makes, and is added to a pool of hard negatives for the next round.
Model selection uses detection AP@0.5 on the validation images - never the
test split.
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

from . import evaluate as ev
from .boxes import iou_matrix
from .config import Config
from .data import Sample, get_split, load_dataset, load_splits, make_splits, save_splits
from .detect import Detections, Detector
from .sampling import HardNegative, PatchSampler


def set_seed(seed: int) -> None:
    import tensorflow as tf

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def load_split_samples(cfg: Config) -> dict[str, list[Sample]]:
    """Load the dataset and return ``{"train": [...], "val": [...], "test": [...]}``."""
    samples = load_dataset(cfg.data_dir)
    if Path(cfg.splits_file).exists():
        splits = load_splits(cfg.splits_file)
    else:
        splits = make_splits(samples, cfg.split_seed, cfg.val_fraction, cfg.test_fraction)
        save_splits(splits, cfg.splits_file)
    return {name: get_split(samples, splits, name) for name in ("train", "val", "test")}


def run_detector(
    detector: Detector, samples: list[Sample], desc: str = "detect", progress: bool = True, **kw
) -> list[Detections]:
    iterator = tqdm(samples, desc=desc, leave=False) if progress else samples
    return [detector.detect(s.load(), **kw) for s in iterator]


def to_results(samples: list[Sample], detections: list[Detections]) -> list[ev.ImageResult]:
    return [ev.ImageResult(s.boxes, d.boxes, d.scores) for s, d in zip(samples, detections, strict=True)]


def mine_hard_negatives(
    detector: Detector, sampler: PatchSampler, cfg: Config, progress: bool = True
) -> list[HardNegative]:
    """Confident windows on training images whose anchor overlaps no object."""
    pool: list[HardNegative] = []
    indices = range(len(sampler.samples))
    for index in tqdm(indices, desc="mining", leave=False) if progress else indices:
        det = detector.detect(sampler.images[index], score_threshold=cfg.mining_score)
        if len(det) == 0:
            continue
        gt = sampler.samples[index].boxes
        if len(gt):
            det = det.select(iou_matrix(det.anchors, gt).max(axis=1) < cfg.neg_iou)
        det = det.select(np.argsort(-det.scores)[: cfg.mining_max_per_image])
        for anchor, scale in zip(det.anchors, det.scales, strict=True):
            pool.append(
                HardNegative(
                    index, float(anchor[0] + anchor[2]) / 2, float(anchor[1] + anchor[3]) / 2, float(scale)
                )
            )
    return pool


def calibrate(detector: Detector, samples: list[Sample], cfg: Config, progress: bool = True) -> dict:
    """Pick NMS / box-voting IoU (by AP@0.5) and score threshold (by F1) on validation."""
    iterator = tqdm(samples, desc="calibrate", leave=False) if progress else samples
    candidates = [detector.candidates(s.load()) for s in iterator]
    best: dict = {}
    for nms_iou in (0.3, 0.4, 0.5):
        for vote_iou in (0.0, 0.5, 0.7):
            results = to_results(samples, [detector.merge(c, nms_iou, vote_iou) for c in candidates])
            ap = ev.ap_at(results, 0.5)
            if not best or ap > best["val_AP50"]:
                best = {
                    "nms_iou": nms_iou,
                    "vote_iou": vote_iou,
                    "val_AP50": ap,
                    "score_threshold": ev.best_f1_threshold(results),
                }
    return best


def train(cfg: Config, progress: bool = True) -> dict:
    """Train a detector as described by ``cfg``; returns the training summary."""
    import tensorflow as tf

    from .model import DetectionLoss, build_model, objectness_accuracy

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    splits = load_split_samples(cfg)
    sampler = PatchSampler(splits["train"], cfg, augment=True)
    val_sampler = PatchSampler(splits["val"], cfg, augment=False)
    x_val, y_val = val_sampler.sample_epoch(np.random.default_rng(1234), pos_per_box=4)

    model = build_model(cfg)
    optimizer = tf.keras.optimizers.AdamW(learning_rate=cfg.lr, weight_decay=cfg.weight_decay)
    optimizer.exclude_from_weight_decay(var_names=["bn", "bias"])
    model.compile(optimizer, loss=DetectionLoss(cfg.box_loss_weight), metrics=[objectness_accuracy])
    detector = Detector(model, cfg)
    eval_every = int(cfg.extra.get("eval_every", 2))

    history: list[dict] = []
    best = {"val_AP50": -1.0, "weights": None, "epoch": 0}
    hard_pool: list[HardNegative] = []
    stages = [("stage1", cfg.epochs_stage1, cfg.lr)]
    stages += [(f"mining{r + 1}", cfg.epochs_stage2, cfg.lr_stage2) for r in range(cfg.mining_rounds)]
    epoch = 0
    start = time.time()
    for stage, n_epochs, base_lr in stages:
        if stage != "stage1":
            # Mine with the best weights so far, then continue from them.
            if best["weights"] is not None:
                model.set_weights(best["weights"])
            mined = mine_hard_negatives(detector, sampler, cfg, progress)
            hard_pool += mined
            print(f"[{stage}] mined {len(mined)} hard negatives (pool: {len(hard_pool)})", flush=True)
        for e in range(n_epochs):
            epoch += 1
            warmup = 0.3 if (stage == "stage1" and e == 0) else 1.0
            lr = base_lr * warmup * (0.02 + 0.98 * 0.5 * (1 + np.cos(np.pi * e / n_epochs)))
            optimizer.learning_rate.assign(lr)
            x, y = sampler.sample_epoch(rng, hard_pool or None)
            fit = model.fit(x, y, batch_size=cfg.batch_size, epochs=1, shuffle=True, verbose=0)
            val_loss, val_acc = model.evaluate(x_val, y_val, batch_size=512, verbose=0)
            row = {
                "epoch": epoch,
                "stage": stage,
                "lr": float(lr),
                "patches": int(len(x)),
                "loss": float(fit.history["loss"][0]),
                "patch_acc": float(fit.history["objectness_accuracy"][0]),
                "val_loss": float(val_loss),
                "val_patch_acc": float(val_acc),
                "minutes": round((time.time() - start) / 60, 2),
            }
            if e == n_epochs - 1 or (e + 1) % eval_every == 0:
                dets = run_detector(detector, splits["val"], "val", progress)
                row["val_AP50"] = ev.ap_at(to_results(splits["val"], dets), 0.5)
                if row["val_AP50"] > best["val_AP50"]:
                    best = {"val_AP50": row["val_AP50"], "weights": model.get_weights(), "epoch": epoch}
            history.append(row)
            print(json.dumps(row), flush=True)
            with open(out_dir / "history.json", "w", encoding="utf-8") as fh:
                json.dump(history, fh, indent=1)

    model.set_weights(best["weights"])
    calibration = calibrate(detector, splits["val"], cfg, progress)
    cfg.nms_iou = float(calibration["nms_iou"])
    cfg.vote_iou = float(calibration["vote_iou"])
    cfg.score_threshold = float(calibration["score_threshold"])
    model.save(out_dir / "model.keras")
    cfg.to_yaml(out_dir / "model.yaml")
    summary = {
        "best_epoch": best["epoch"],
        "epochs": epoch,
        "hard_negatives": len(hard_pool),
        "parameters": int(model.count_params()),
        "train_minutes": round((time.time() - start) / 60, 1),
        **calibration,
    }
    with open(out_dir / "train_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps(summary), flush=True)
    return summary
