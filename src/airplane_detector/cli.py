"""Command-line interface: ``airdet <command>``.

airdet download                      fetch the dataset
airdet train --config configs/default.yaml
airdet evaluate --model models/airplane_fcn.keras --split test
airdet predict  --model models/airplane_fcn.keras path/to/image_or_folder
airdet baseline                      train + evaluate the original v1 pipeline
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .config import Config

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _write_json(obj: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1)
        fh.write("\n")


def _summary(report: dict) -> str:
    op = report["operating_point"]
    return (
        f"AP50 {report['AP50']:.3f}  AP75 {report['AP75']:.3f}  mAP {report['mAP']:.3f}  |  "
        f"P {op['precision']:.3f}  R {op['recall']:.3f}  F1 {op['f1']:.3f}  @ score >= {op['threshold']:.2f}"
    )


# ----------------------------------------------------------------- commands
def cmd_download(args) -> None:
    from .data import download_dataset, load_dataset

    path = download_dataset(args.data_dir, force=args.force)
    samples = load_dataset(path)
    print(f"{len(samples)} images, {sum(len(s.boxes) for s in samples)} boxes in {path}")


def cmd_train(args) -> None:
    from .train import train

    cfg = Config.from_yaml(args.config) if args.config else Config()
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.seed is not None:
        cfg.seed = args.seed
    train(cfg, progress=not args.quiet)


def cmd_evaluate(args) -> None:
    from . import evaluate as ev
    from .detect import Detector
    from .sampling import pyramid_scales
    from .train import load_split_samples, run_detector, to_results

    detector = Detector.load(args.model)
    cfg = detector.cfg
    if args.data_dir:
        cfg.data_dir = args.data_dir
    samples = load_split_samples(cfg)[args.split]
    kwargs = {"use_regression": not args.no_regression}
    if args.single_scale:
        kwargs["scales"] = np.array([1.0])
    start = time.time()
    detections = run_detector(detector, samples, args.split, progress=not args.quiet, **kwargs)
    seconds = time.time() - start
    report = ev.evaluate(to_results(samples, detections), threshold=cfg.score_threshold)
    report.update(
        split=args.split,
        model=str(args.model),
        use_regression=not args.no_regression,
        pyramid_levels=1 if args.single_scale else int(len(pyramid_scales(cfg))),
        nms_iou=cfg.nms_iou,
        vote_iou=cfg.vote_iou,
        seconds_per_image=round(seconds / max(len(samples), 1), 4),
    )
    print(_summary(report))
    if args.out:
        _write_json(report, args.out)


def cmd_calibrate(args) -> None:
    from .detect import Detector
    from .train import calibrate, load_split_samples

    detector = Detector.load(args.model)
    cfg = detector.cfg
    result = calibrate(detector, load_split_samples(cfg)["val"], cfg, progress=not args.quiet)
    cfg.nms_iou, cfg.vote_iou = float(result["nms_iou"]), float(result["vote_iou"])
    cfg.score_threshold = float(result["score_threshold"])
    cfg.to_yaml(Path(args.model).with_suffix(".yaml"))
    print(json.dumps(result))


def cmd_predict(args) -> None:
    import cv2

    from .detect import Detector
    from .viz import draw_detections

    detector = Detector.load(args.model)
    threshold = detector.cfg.score_threshold if args.threshold is None else args.threshold
    source = Path(args.source)
    paths = (
        sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if source.is_dir()
        else [source]
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for path in paths:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"skip (not an image): {path}")
            continue
        det = detector.detect(img, score_threshold=threshold)
        cv2.imwrite(str(out_dir / f"{path.stem}_detections.jpg"), draw_detections(img, det.boxes, det.scores))
        records.append(
            {
                "image": path.name,
                "detections": [
                    {"box": [round(float(v), 1) for v in b], "score": round(float(s), 4)}
                    for b, s in zip(det.boxes, det.scores, strict=True)
                ],
            }
        )
        print(f"{path.name}: {len(det)} airplane(s)")
    _write_json({"score_threshold": threshold, "results": records}, out_dir / "detections.json")
    print(f"wrote {len(records)} annotated image(s) and detections.json to {out_dir}")


def cmd_baseline(args) -> None:
    from . import baseline as bl
    from . import evaluate as ev
    from .train import load_split_samples, to_results

    cfg = Config(data_dir=args.data_dir) if args.data_dir else Config()
    splits = load_split_samples(cfg)
    everything = splits["train"] + splits["val"] + splits["test"]
    ceiling = bl.window_recall_ceiling(everything)
    print(f"recall ceiling of fixed 64x64 / stride 32 windows at IoU 0.5: {ceiling:.3f}")
    model, info = bl.train_baseline(splits["train"], splits["val"], seed=args.seed)
    if args.save_model:
        Path(args.save_model).parent.mkdir(parents=True, exist_ok=True)
        model.save(args.save_model)
    start = time.time()
    detections = [bl.detect_baseline(model, s.load()) for s in splits["test"]]
    seconds = time.time() - start
    report = ev.evaluate(to_results(splits["test"], detections), threshold=bl.NOTEBOOK_THRESHOLD)
    report.update(
        split="test",
        model="v1 baseline (fixed 64x64 windows, stride 32)",
        recall_ceiling_iou50=ceiling,
        seconds_per_image=round(seconds / len(splits["test"]), 4),
        **info,
    )
    print(_summary(report))
    _write_json(report, args.out)


# --------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="airdet", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="download and unpack the dataset")
    p.add_argument("--data-dir", default="data/raw")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_download)

    p = sub.add_parser("train", help="train a detector")
    p.add_argument("--config", help="YAML config (defaults are used when omitted)")
    p.add_argument("--output-dir")
    p.add_argument("--seed", type=int)
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="evaluate a trained detector on a split")
    p.add_argument("--model", default="models/airplane_fcn.keras")
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--data-dir")
    p.add_argument("--out", help="write the full report to this JSON file")
    p.add_argument("--no-regression", action="store_true", help="ablation: use the anchor as the box")
    p.add_argument("--single-scale", action="store_true", help="ablation: no image pyramid")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("calibrate", help="re-select NMS, box voting and score threshold on validation")
    p.add_argument("--model", default="models/airplane_fcn.keras")
    p.add_argument("--quiet", action="store_true")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("predict", help="detect airplanes in an image or a folder of images")
    p.add_argument("source")
    p.add_argument("--model", default="models/airplane_fcn.keras")
    p.add_argument("--out", default="predictions")
    p.add_argument("--threshold", type=float, help="score threshold (default: calibrated value)")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("baseline", help="train and evaluate the original v1 pipeline")
    p.add_argument("--data-dir")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="results/baseline_v1.json")
    p.add_argument("--save-model", help="also save the trained v1 classifier here")
    p.set_defaults(func=cmd_baseline)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
