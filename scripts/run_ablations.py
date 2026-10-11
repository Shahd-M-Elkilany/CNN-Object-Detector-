"""Decode-time ablations of a trained detector on the test split.

The same weights are evaluated with one component switched off at a time, so
any difference is caused by that component alone:

* ``full``           - image pyramid + box regression + box voting
* ``no_voting``      - plain NMS, no score-weighted box averaging
* ``no_regression``  - the anchor itself is reported as the box
* ``single_scale``   - no pyramid: the network only sees the original resolution

Usage:  python scripts/run_ablations.py --model models/airplane_fcn.keras
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from airplane_detector import evaluate as ev
from airplane_detector.detect import Detector
from airplane_detector.train import load_split_samples, run_detector, to_results

VARIANTS = {
    "full": {},
    "no_voting": {"vote_iou": 0.0},
    "no_regression": {"use_regression": False},
    "single_scale": {"scales": np.array([1.0])},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", default="models/airplane_fcn.keras")
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", default="results/ablations.json")
    args = parser.parse_args()

    detector = Detector.load(args.model)
    samples = load_split_samples(detector.cfg)[args.split]
    rows = {}
    for name, kwargs in VARIANTS.items():
        detections = run_detector(detector, samples, name, **kwargs)
        report = ev.evaluate(to_results(samples, detections), n_boot=0, with_curve=False)
        rows[name] = {k: report[k] for k in ("AP50", "AP75", "mAP", "max_recall")}
        rows[name]["best_f1"] = report["operating_point"]["f1"]
        print(f"{name:14s} AP50 {report['AP50']:.3f}  AP75 {report['AP75']:.3f}  mAP {report['mAP']:.3f}")
    # Robustness check: the same detections without the one very dense scene.
    detections = run_detector(detector, samples, "full")
    kept = [(s, d) for s, d in zip(samples, detections, strict=True) if len(s.boxes) < 100]
    results = to_results([s for s, _ in kept], [d for _, d in kept])
    sparse = {
        "note": "split without scenes of 100+ airplanes",
        "images": len(results),
        "objects": int(sum(len(r.gt) for r in results)),
        "AP50": ev.ap_at(results, 0.5),
        "AP50_ci95": list(ev.bootstrap_ap(results)),
        "operating_point": ev.operating_point(results, detector.cfg.score_threshold),
    }
    print(f"without dense scenes: AP50 {sparse['AP50']:.3f}  F1 {sparse['operating_point']['f1']:.3f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(Path(args.out).with_name("test_metrics_without_dense_scene.json"), "w", encoding="utf-8") as fh:
        json.dump(sparse, fh, indent=1)
        fh.write("\n")
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"split": args.split, "model": args.model, "variants": rows}, fh, indent=1)
        fh.write("\n")


if __name__ == "__main__":
    main()
