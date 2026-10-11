import numpy as np
import pytest

tf = pytest.importorskip("tensorflow")

from airplane_detector.config import Config  # noqa: E402
from airplane_detector.model import DetectionLoss, build_model, load_model, objectness_accuracy  # noqa: E402


@pytest.fixture(scope="module")
def model():
    tf.keras.utils.set_random_seed(0)
    net = build_model(Config(widths=(4, 8, 8, 16), dropout=0.0))
    # Give batch-norm non-trivial statistics so the tests exercise it.
    net(np.random.default_rng(0).uniform(0, 255, (8, 40, 40, 3)).astype(np.float32), training=True)
    return net


def test_one_window_gives_one_prediction(model):
    assert model(np.zeros((2, 40, 40, 3), np.float32)).shape == (2, 1, 1, 5)


def test_stride_is_eight(model):
    assert model(np.zeros((1, 40 + 8 * 5, 40 + 8 * 3, 3), np.float32)).shape == (1, 6, 4, 5)


def test_dense_output_equals_sliding_window(model):
    """The claim the whole detector rests on: one pass == every window."""
    rng = np.random.default_rng(1)
    image = rng.uniform(0, 255, (1, 104, 120, 3)).astype(np.float32)
    dense = model(image, training=False).numpy()[0]
    for row, col in [(0, 0), (3, 7), (8, 10), (5, 2)]:
        patch = image[:, row * 8 : row * 8 + 40, col * 8 : col * 8 + 40]
        np.testing.assert_allclose(model(patch, training=False).numpy()[0, 0, 0], dense[row, col], atol=1e-4)


def test_default_model_size():
    assert build_model(Config()).count_params() == 217_229


def test_wrong_window_is_rejected():
    with pytest.raises(ValueError):
        build_model(Config(window=48))


def test_loss_matches_hand_computation():
    loss = DetectionLoss(box_weight=2.0, beta=0.5)
    y_true = np.array([[1, 0.0, 0.0, 0.0, 0.0], [0, 9, 9, 9, 9]], np.float32).reshape(2, 1, 1, 5)
    y_pred = np.array([[0.0, 0.2, 0.0, 0.0, 1.0], [0.0, 5, 5, 5, 5]], np.float32).reshape(2, 1, 1, 5)
    cls = np.log(2.0)  # both logits are 0
    box = 0.5 * 0.2**2 / 0.5 + (1.0 - 0.25)  # one small (quadratic) and one large (linear) error
    assert float(loss(y_true, y_pred)) == pytest.approx(cls + 2.0 * box / 1, rel=1e-5)


def test_loss_ignores_box_error_on_negatives():
    loss = DetectionLoss()
    y_true = np.zeros((4, 1, 1, 5), np.float32)
    a = np.zeros((4, 1, 1, 5), np.float32)
    b = a.copy()
    b[..., 1:] = 100.0
    assert float(loss(y_true, a)) == pytest.approx(float(loss(y_true, b)))


def test_objectness_accuracy():
    y_true = np.array([1, 0, 1, 0], np.float32).reshape(4, 1, 1, 1) * np.ones((1, 1, 1, 5), np.float32)
    y_pred = np.array([3, -3, -3, 3], np.float32).reshape(4, 1, 1, 1) * np.ones((1, 1, 1, 5), np.float32)
    assert float(objectness_accuracy(y_true, y_pred)) == pytest.approx(0.5)


def test_save_and_load_roundtrip(model, tmp_path):
    path = tmp_path / "m.keras"
    model.save(path)
    x = np.random.default_rng(2).uniform(0, 255, (1, 56, 56, 3)).astype(np.float32)
    np.testing.assert_allclose(load_model(str(path))(x).numpy(), model(x).numpy(), atol=1e-5)


def test_one_training_step_reduces_loss():
    tf.keras.utils.set_random_seed(0)
    net = build_model(Config(widths=(4, 8, 8, 16), dropout=0.0))
    net.compile(tf.keras.optimizers.Adam(1e-2), loss=DetectionLoss(), metrics=[objectness_accuracy])
    rng = np.random.default_rng(0)
    x = rng.integers(0, 255, (64, 40, 40, 3)).astype(np.uint8)
    y = np.zeros((64, 1, 1, 5), np.float32)
    y[:32, 0, 0, 0] = 1
    x[:32] //= 4  # make the two classes separable by brightness
    history = net.fit(x, y, batch_size=16, epochs=8, verbose=0).history["loss"]
    assert history[-1] < history[0] * 0.5
