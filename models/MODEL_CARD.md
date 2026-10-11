# Model card — `airplane_fcn.keras`

| | |
|---|---|
| Task | Detect airplanes (one class, axis-aligned boxes) in overhead RGB imagery |
| Architecture | Fully-convolutional CNN, 7 conv layers, 40 px receptive field, stride 8, 217,229 parameters |
| Output | Per grid cell: objectness logit + 4 box offsets relative to a 24 px anchor |
| Input | BGR image, any size, pixel values 0–255 (normalisation is inside the model) |
| Training data | 511 images / 1,877 airplanes from the Airplanes set of 1297rohit/RCNN (Apache-2.0) |
| Initialisation | Random. No pretrained weights. |
| Training | 30 epochs, AdamW, two rounds of hard-negative mining, seed 0, 67 min on 2 CPU cores |
| Files | `airplane_fcn.keras` (weights), `airplane_fcn.yaml` (config incl. calibrated thresholds) |
| Licence | MIT |

## Performance

Held-out test split, 110 images, 560 airplanes. Full report: `results/test_metrics.json`.

| Metric | Value |
|---|---:|
| AP@0.5 | 0.940 (95 % CI 0.917–0.975) |
| AP@0.75 | 0.429 |
| mAP@[.5:.95] | 0.486 |
| Precision / recall / F1 at score ≥ 0.80 | 0.859 / 0.921 / 0.889 |
| AP@0.5 small (< 20 px) / medium / large (≥ 45 px) | 0.787 / 0.956 / 0.965 |

The score threshold (0.80), NMS IoU (0.3) and box-voting IoU (0.5) were selected on the validation split.

## Intended use

A compact, readable reference for how a CNN classifier becomes an object detector, and a baseline for the
same dataset. Suitable for teaching, experimentation and as a starting point for further work.

## Out of scope

- Operational, safety-critical or surveillance use. The model was trained on 511 images from a few airports
  and has not been validated beyond its test split.
- Imagery that differs from the training data: other ground resolutions, off-nadir views, night, infrared or
  SAR, or scenes where airplanes are smaller than about 8 px.

## Known failure modes

- Boxes are loose: many detections that count at IoU 0.5 fail at IoU 0.75.
- Small (< 20 px) and low-contrast aircraft are missed more often.
- Tightly parked aircraft may be merged into one box or split into two.
- Scores are not calibrated probabilities; most confident detections score close to 1.0.

## Reproduce

```bash
airdet download
airdet train --config configs/default.yaml --output-dir runs/main
airdet evaluate --model runs/main/model.keras --split test
```

CPU training is multi-threaded and not bit-for-bit deterministic, so a re-run will land near, not exactly on,
the numbers above.
