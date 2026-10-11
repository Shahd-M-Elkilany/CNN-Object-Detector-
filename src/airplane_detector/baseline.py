"""Version 1 of this project, ported faithfully from the original notebook.

The first version (``notebooks/legacy/Object_Detector_CNN_Based.ipynb``) cut
every image into fixed 64x64 crops on a stride-32 grid, labelled a crop as
"airplane" when it had IoU >= 0.5 with a ground-truth box, trained a small CNN
classifier on the crops and used the crop itself as the predicted box.

It is kept here - same architecture, same hyper-parameters - so that it can be
evaluated under exactly the same protocol as the new detector. The only change
is that crops are split *by image* instead of at random, which removes the
train/test leakage of the original evaluation.
"""

from __future__ import annotations

import numpy as np

from .boxes import iou_matrix, nms
from .data import Sample
from .detect import Detections

CROP = 64
STRIDE = 32
POS_IOU = 0.5
NEG_POS_RATIO = 15
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-4
NOTEBOOK_THRESHOLD = 0.4
NMS_IOU = 0.3


def grid_windows(height: int, width: int, crop: int = CROP, stride: int = STRIDE) -> np.ndarray:
    """All crop boxes of the sliding-window grid, shape (N, 4)."""
    ys = range(0, height - crop + 1, stride)
    xs = range(0, width - crop + 1, stride)
    return np.array([[x, y, x + crop, y + crop] for y in ys for x in xs], np.float32).reshape(-1, 4)


def window_recall_ceiling(
    samples: list[Sample], crop: int = CROP, stride: int = STRIDE, iou: float = 0.5
) -> float:
    """Share of objects that *any* grid window overlaps with IoU >= ``iou``.

    A detector that outputs grid windows as boxes cannot have a recall above
    this number, no matter how good its classifier is.
    """
    hit = total = 0
    for s in samples:
        if len(s.boxes) == 0:
            continue
        h, w = s.load().shape[:2]
        windows = grid_windows(h, w, crop, stride)
        total += len(s.boxes)
        if len(windows):
            hit += int((iou_matrix(windows, s.boxes).max(axis=0) >= iou).sum())
    return hit / max(total, 1)


def make_crops(samples: list[Sample], rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Labelled crops (RGB, float32 in [0, 1]) with negatives sub-sampled 15:1."""
    positives, negatives = [], []
    for s in samples:
        img = s.load()[:, :, ::-1]
        windows = grid_windows(*img.shape[:2])
        if len(windows) == 0:
            continue
        best = iou_matrix(windows, s.boxes).max(axis=1) if len(s.boxes) else np.zeros(len(windows))
        for (x1, y1, x2, y2), overlap in zip(windows.astype(int), best, strict=True):
            (positives if overlap >= POS_IOU else negatives).append(img[y1:y2, x1:x2])
    if not positives:
        raise ValueError("no 64x64 grid window overlaps any object with IoU >= 0.5")
    n_neg = min(len(negatives), len(positives) * NEG_POS_RATIO)
    picks = rng.choice(len(negatives), size=n_neg, replace=False)
    crops = positives + [negatives[i] for i in picks]
    x = np.stack(crops).astype(np.float32) / 255.0
    y = np.concatenate([np.ones(len(positives)), np.zeros(n_neg)]).astype(np.float32)
    order = rng.permutation(len(x))
    return x[order], y[order]


def build_baseline_model():
    """The original three-convolution classifier (109,889 parameters)."""
    import tensorflow as tf
    from tensorflow.keras import layers

    inp = tf.keras.Input(shape=(CROP, CROP, 3))
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid", name="prob")(x)
    return tf.keras.Model(inp, out, name="cnn_detector_v1")


def train_baseline(train: list[Sample], val: list[Sample], seed: int = 42, verbose: int = 2):
    """Train the v1 classifier exactly as the notebook did."""
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    rng = np.random.default_rng(seed)
    x_train, y_train = make_crops(train, rng)
    x_val, y_val = make_crops(val, rng)
    n_pos = float(y_train.sum())
    n = float(len(y_train))
    class_weight = {0: n / (2 * (n - n_pos)), 1: n / (2 * n_pos)}  # sklearn "balanced"
    model = build_baseline_model()
    model.compile(tf.keras.optimizers.Adam(LEARNING_RATE), loss="binary_crossentropy", metrics=["accuracy"])
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=[tf.keras.callbacks.EarlyStopping("val_loss", patience=5, restore_best_weights=True)],
        verbose=verbose,
    )
    info = {
        "train_crops": int(n),
        "train_positives": int(n_pos),
        "val_crops": int(len(y_val)),
        "val_positives": int(y_val.sum()),
        "epochs_run": len(history.history["loss"]),
    }
    return model, info


def detect_baseline(model, img_bgr: np.ndarray, score_floor: float = 0.05) -> Detections:
    """Classify every grid window and merge with NMS (windows are the boxes)."""
    windows = grid_windows(*img_bgr.shape[:2])
    if len(windows) == 0:
        return Detections.empty()
    rgb = img_bgr[:, :, ::-1]
    crops = np.stack([rgb[y1:y2, x1:x2] for x1, y1, x2, y2 in windows.astype(int)])
    scores = model.predict(crops.astype(np.float32) / 255.0, batch_size=256, verbose=0).reshape(-1)
    keep = scores >= score_floor
    windows, scores = windows[keep], scores[keep]
    order = nms(windows, scores, NMS_IOU)
    boxes = windows[order]
    return Detections(boxes, scores[order], boxes.copy(), np.ones(len(order), np.float32))
