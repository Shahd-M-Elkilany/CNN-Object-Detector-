"""Regenerate every figure in ``assets/`` from the files in ``results/``.

Usage:  python scripts/make_figures.py            (needs the dataset and model)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from airplane_detector import baseline as bl  # noqa: E402
from airplane_detector import evaluate as ev  # noqa: E402
from airplane_detector.boxes import box_area  # noqa: E402
from airplane_detector.detect import Detector  # noqa: E402
from airplane_detector.train import load_split_samples  # noqa: E402
from airplane_detector.viz import draw_evaluation  # noqa: E402

BLUE, ORANGE, GRAY, INK, MUTED = "#2a78d6", "#eb6834", "#8a8984", "#0b0b0b", "#52514e"
SURFACE, GRID = "#fcfcfb", "#e6e5e0"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "font.size": 10,
        "legend.frameon": False,
    }
)


def load(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def save(fig, out: Path, name: str) -> None:
    fig.savefig(out / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out / name)


def fig_object_sizes(splits, cfg, ceiling: float, out: Path) -> None:
    """Why the first version could not work: object sizes vs. what a 64 px window can match."""
    samples = splits["train"] + splits["val"] + splits["test"]
    sizes = np.concatenate([np.sqrt(box_area(s.boxes)) for s in samples if len(s.boxes)])
    fig, ax = plt.subplots(figsize=(8, 3.6))
    bins = np.geomspace(4, 320, 40)
    ax.hist(sizes, bins=bins, color=GRAY, alpha=0.55, edgecolor=SURFACE, linewidth=0.8)
    # A square 64 px window reaches IoU 0.5 only with squares of side 45-90 px.
    lo, hi = bl.CROP * np.sqrt(0.5), bl.CROP / np.sqrt(0.5)
    ax.axvspan(lo, hi, color=ORANGE, alpha=0.18, lw=0)
    ax.axvspan(cfg.min_object * np.sqrt(0.5), cfg.max_object, ymax=0.06, color=BLUE, alpha=0.9, lw=0)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.28)
    top = ax.get_ylim()[1]
    ax.text(
        np.sqrt(lo * hi),
        top * 0.97,
        f"v1: a 64 px grid window reaches\nIoU ≥ 0.5 with {ceiling:.0%} of objects",
        color="#a8431c",
        ha="center",
        va="top",
        fontsize=9,
    )
    ax.text(
        6.2,
        top * 0.085,
        "v2: the image pyramid covers this whole range",
        color="#1c5cab",
        fontsize=9,
        va="bottom",
    )
    ax.set_xscale("log")
    ax.set_xticks([5, 10, 20, 40, 80, 160, 320], labels=["5", "10", "20", "40", "80", "160", "320"])
    ax.minorticks_off()
    ax.set_xlabel("object size  (√ box area, pixels)")
    ax.set_ylabel("objects")
    ax.set_title(f"Airplane sizes in the dataset  (n = {len(sizes):,}, median {np.median(sizes):.0f} px)")
    save(fig, out, "object_sizes.png")


def fig_training(history: list[dict], out: Path) -> None:
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    axes[0].plot(epochs, [h["loss"] for h in history], color=GRAY, lw=2, label="train patches")
    axes[0].plot(epochs, [h["val_loss"] for h in history], color=BLUE, lw=2, label="validation patches")
    axes[0].set_title("Patch loss")
    axes[0].legend()
    ap = [(h["epoch"], h["val_AP50"]) for h in history if "val_AP50" in h]
    axes[1].plot(*zip(*ap, strict=True), color=BLUE, lw=2, marker="o", ms=5)
    axes[1].set_title("Detection AP@0.5 on validation images")
    axes[1].set_ylim(0, 1)
    stages = [h["stage"] for h in history]
    for ax in axes:
        ax.set_xlabel("epoch")
        for i in range(1, len(stages)):
            if stages[i] != stages[i - 1]:
                ax.axvline(epochs[i] - 0.5, color=ORANGE, lw=1.2, ls=(0, (4, 3)))
    first_mining = next((h["epoch"] for h in history if h["stage"] != "stage1"), None)
    if first_mining:
        axes[1].text(first_mining - 0.2, 0.06, "+ hard negatives", color="#a8431c", fontsize=9, ha="left")
    save(fig, out, "training_curves.png")


def fig_pr(new: dict, old: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    for report, color, label in ((old, ORANGE, "v1 baseline"), (new, BLUE, "v2 (this repo)")):
        curve = report["pr_curve"]
        ax.plot(
            curve["recall"],
            curve["precision"],
            color=color,
            lw=2,
            label=f"{label}  ·  AP50 {report['AP50']:.3f}",
        )
    op = new["operating_point"]
    ax.plot([op["recall"]], [op["precision"]], "o", ms=9, color=BLUE, mec=SURFACE, mew=2)
    ax.annotate(
        f"operating point\nP {op['precision']:.2f} · R {op['recall']:.2f}",
        (op["recall"], op["precision"]),
        xytext=(-12, -34),
        textcoords="offset points",
        ha="right",
        fontsize=9,
        color=MUTED,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_title("Precision-recall on the held-out test images (IoU ≥ 0.5)")
    ax.legend(loc="center left", bbox_to_anchor=(0.2, 0.45))
    save(fig, out, "pr_curve.png")


def fig_ladder(rows: list[tuple[str, float]], out: Path) -> None:
    labels, values = zip(*rows, strict=True)
    fig, ax = plt.subplots(figsize=(8, 0.55 * len(rows) + 1.2))
    y = np.arange(len(rows))[::-1]
    colors = [ORANGE if label.startswith("v1") else BLUE for label in labels]
    ax.barh(y, values, height=0.6, color=colors)
    for yi, v in zip(y, values, strict=True):
        ax.text(v + 0.012, yi, f"{v:.3f}", va="center", color=INK, fontsize=10)
    ax.set_yticks(y, labels=labels)
    ax.tick_params(axis="y", colors=INK, length=0)
    ax.set_xlim(0, 1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("AP@0.5 on the test split")
    ax.set_title("What each component is worth")
    save(fig, out, "ablation.png")


def _tile(img: np.ndarray, side: int = 384) -> np.ndarray:
    h, w = img.shape[:2]
    scale = side / max(h, w)
    img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    canvas = np.full((side, side, 3), 252, np.uint8)
    y, x = (side - img.shape[0]) // 2, (side - img.shape[1]) // 2
    canvas[y : y + img.shape[0], x : x + img.shape[1]] = img
    return canvas


def _grid(tiles: list[np.ndarray], cols: int, gap: int = 6) -> np.ndarray:
    side = tiles[0].shape[0]
    rows = int(np.ceil(len(tiles) / cols))
    canvas = np.full((rows * side + (rows - 1) * gap, cols * side + (cols - 1) * gap, 3), 252, np.uint8)
    for i, tile in enumerate(tiles):
        r, c = divmod(i, cols)
        canvas[r * (side + gap) : r * (side + gap) + side, c * (side + gap) : c * (side + gap) + side] = tile
    return canvas


def fig_qualitative(detector: Detector, baseline_model, samples, out: Path, n: int = 8) -> None:
    """Test images drawn with green = correct, red = false alarm, yellow = missed."""
    threshold = detector.cfg.score_threshold
    rng = np.random.default_rng(3)
    with_objects = [s for s in samples if 1 <= len(s.boxes) <= 40]
    picks = [with_objects[i] for i in rng.choice(len(with_objects), size=n, replace=False)]
    new_tiles, old_tiles = [], []
    for s in picks:
        img = s.load()
        det = detector.detect(img, score_threshold=threshold)
        new_tiles.append(_tile(draw_evaluation(img, s.boxes, det.boxes, min_side=768)))
        if baseline_model is not None:
            old = bl.detect_baseline(baseline_model, img, score_floor=bl.NOTEBOOK_THRESHOLD)
            old_tiles.append(_tile(draw_evaluation(img, s.boxes, old.boxes, min_side=768)))
    cv2.imwrite(str(out / "qualitative.jpg"), _grid(new_tiles, 4), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("wrote", out / "qualitative.jpg")
    if old_tiles:
        pairs = [t for pair in zip(old_tiles[:4], new_tiles[:4], strict=True) for t in pair]
        order = pairs[0::2] + pairs[1::2]  # top row v1, bottom row v2
        cv2.imwrite(str(out / "v1_vs_v2.jpg"), _grid(order, 4), [cv2.IMWRITE_JPEG_QUALITY, 90])
        print("wrote", out / "v1_vs_v2.jpg")


def fig_failures(detector: Detector, samples, out: Path, n: int = 4) -> None:
    """The test images with the most mistakes at the operating point."""
    threshold = detector.cfg.score_threshold
    scored = []
    for s in samples:
        if len(s.boxes) > 40:  # very dense scenes are unreadable at thumbnail size
            continue
        det = detector.detect(s.load(), score_threshold=threshold)
        op = ev.operating_point([ev.ImageResult(s.boxes, det.boxes, det.scores)], threshold)
        scored.append((op["fp"] + op["fn"], s, det))
    scored.sort(key=lambda t: -t[0])
    tiles = [_tile(draw_evaluation(s.load(), s.boxes, det.boxes, min_side=768)) for _, s, det in scored[:n]]
    cv2.imwrite(str(out / "failure_cases.jpg"), _grid(tiles, n), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print("wrote", out / "failure_cases.jpg")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="models/airplane_fcn.keras")
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="assets")
    parser.add_argument("--baseline-model", default="runs/baseline_v1.keras")
    args = parser.parse_args()
    out, results = Path(args.out), Path(args.results)
    out.mkdir(parents=True, exist_ok=True)

    detector = Detector.load(args.model)
    splits = load_split_samples(detector.cfg)
    new, old = load(results / "test_metrics.json"), load(results / "baseline_v1.json")
    ablations = load(results / "ablations.json")["variants"]
    rows = [
        ("v1 baseline: fixed 64 px windows", old["AP50"]),
        ("v2 without the image pyramid", ablations["single_scale"]["AP50"]),
        ("v2 without box regression", ablations["no_regression"]["AP50"]),
    ]
    no_mining = results / "no_mining_test_metrics.json"
    if no_mining.exists():
        rows.append(("v2 without hard-negative mining", load(no_mining)["AP50"]))
    rows += [("v2 without box voting", ablations["no_voting"]["AP50"]), ("v2 full", new["AP50"])]

    fig_object_sizes(splits, detector.cfg, old["recall_ceiling_iou50"], out)
    fig_training(load(results / "training_history.json"), out)
    fig_pr(new, old, out)
    fig_ladder(rows, out)
    baseline_model = None
    if Path(args.baseline_model).exists():
        import tensorflow as tf

        baseline_model = tf.keras.models.load_model(args.baseline_model, compile=False)
    fig_qualitative(detector, baseline_model, splits["test"], out)
    fig_failures(detector, splits["test"], out)


if __name__ == "__main__":
    main()
