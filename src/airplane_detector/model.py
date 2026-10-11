"""The network: a small fully-convolutional classifier + box regressor.

Design constraints, all of which are checked by the unit tests:

* Every convolution uses ``valid`` padding and the receptive field is exactly
  ``window`` pixels with a total stride of 8.  Applying the network to a
  ``window x window`` patch therefore yields a single 1x1 prediction, and
  applying the *same weights* to a large image yields the dense map of what a
  sliding window with stride 8 would have produced - computed in one forward
  pass instead of thousands (the OverFeat trick).
* The output has five channels per location: an objectness logit and the four
  R-CNN box offsets ``(tx, ty, tw, th)`` relative to the anchor.
* Pixel normalisation is part of the model, so callers pass raw 0-255 images.
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers

from .config import Config


def _conv_bn_relu(x, filters: int, kernel: int, name: str):
    x = layers.Conv2D(filters, kernel, padding="valid", use_bias=False, name=f"{name}_conv")(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    return layers.ReLU(name=f"{name}_relu")(x)


def build_model(cfg: Config) -> tf.keras.Model:
    """Build the detector network for ``cfg.window`` (must be 40 for this layout)."""
    if cfg.window != 40 or cfg.stride != 8:
        raise ValueError("this architecture has a 40 px receptive field and stride 8")
    c1, c2, c3, c4 = cfg.widths
    inputs = tf.keras.Input(shape=(None, None, 3), name="image_bgr")
    x = layers.Rescaling(1.0 / 127.5, offset=-1.0, name="normalise")(inputs)
    #                                                    spatial size for a 40 px patch
    x = _conv_bn_relu(x, c1, 3, "stage1a")  # 38
    x = _conv_bn_relu(x, c1, 3, "stage1b")  # 36
    x = layers.MaxPooling2D(2, name="pool1")(x)  # 18
    x = _conv_bn_relu(x, c2, 3, "stage2")  # 16
    x = layers.MaxPooling2D(2, name="pool2")(x)  # 8
    x = _conv_bn_relu(x, c3, 3, "stage3a")  # 6
    x = _conv_bn_relu(x, c3, 3, "stage3b")  # 4
    x = layers.MaxPooling2D(2, name="pool3")(x)  # 2
    x = _conv_bn_relu(x, c4, 2, "head")  # 1
    if cfg.dropout > 0:
        x = layers.Dropout(cfg.dropout, name="dropout")(x)
    outputs = layers.Conv2D(5, 1, name="objectness_and_box")(x)
    return tf.keras.Model(inputs, outputs, name="airplane_fcn")


@tf.keras.utils.register_keras_serializable(package="airplane_detector")
class DetectionLoss(tf.keras.losses.Loss):
    """Binary cross-entropy on objectness + smooth-L1 on the box offsets.

    ``y_true`` and ``y_pred`` are (..., 5): ``[label, tx, ty, tw, th]`` and
    ``[logit, tx, ty, tw, th]``. The box term only counts on positives and is
    normalised by their number, as in Fast R-CNN.
    """

    def __init__(self, box_weight: float = 2.0, beta: float = 0.11, name: str = "detection_loss", **kw):
        super().__init__(name=name, **kw)
        self.box_weight = box_weight
        self.beta = beta

    def call(self, y_true, y_pred):
        y_true = tf.reshape(tf.cast(y_true, tf.float32), (-1, 5))
        y_pred = tf.reshape(tf.cast(y_pred, tf.float32), (-1, 5))
        label = y_true[:, 0]
        cls = tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=y_pred[:, 0])
        diff = tf.abs(y_true[:, 1:] - y_pred[:, 1:])
        smooth_l1 = tf.where(diff < self.beta, 0.5 * diff * diff / self.beta, diff - 0.5 * self.beta)
        box = tf.reduce_sum(smooth_l1, axis=1) * label
        n = tf.cast(tf.size(label), tf.float32)
        n_pos = tf.maximum(tf.reduce_sum(label), 1.0)
        # Return per-sample values whose *mean* is cls_mean + w * box_sum / n_pos.
        return cls + self.box_weight * box * n / n_pos

    def get_config(self):
        return {**super().get_config(), "box_weight": self.box_weight, "beta": self.beta}


def objectness_accuracy(y_true, y_pred):
    """Patch-level accuracy of the objectness output (threshold 0.5)."""
    y_true = tf.reshape(y_true, (-1, 5))
    y_pred = tf.reshape(y_pred, (-1, 5))
    hit = tf.equal(y_true[:, 0] > 0.5, y_pred[:, 0] > 0.0)
    return tf.reduce_mean(tf.cast(hit, tf.float32))


def load_model(path: str, cfg: Config | None = None) -> tf.keras.Model:
    """Load a detector for inference.

    ``*.weights.h5`` (the released format) holds only the weights: the network
    is rebuilt from ``cfg`` and the weights are loaded into it. This keeps the
    file readable across Keras versions, which whole-model ``.keras`` archives
    are not. A ``.keras`` archive is still accepted for convenience.
    """
    if str(path).endswith(".keras"):
        return tf.keras.models.load_model(path, compile=False)
    if cfg is None:
        raise ValueError("a Config is required to rebuild the network for a weights file")
    model = build_model(cfg)
    model.load_weights(path)
    return model
