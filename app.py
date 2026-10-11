"""Interactive demo:  python app.py   (needs `pip install -e ".[demo]"`)."""

from __future__ import annotations

import cv2
import gradio as gr

from airplane_detector.detect import Detector
from airplane_detector.viz import draw_detections

MODEL_PATH = "models/airplane_fcn.keras"
detector = Detector.load(MODEL_PATH)


def predict(image_rgb, threshold: float):
    if image_rgb is None:
        return None, "Upload an image first."
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    det = detector.detect(image_bgr, score_threshold=threshold)
    annotated = draw_detections(image_bgr, det.boxes, det.scores)
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), f"{len(det)} airplane(s) at score >= {threshold:.2f}"


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="numpy", label="Aerial image"),
        gr.Slider(0.05, 0.99, value=detector.cfg.score_threshold, step=0.01, label="Score threshold"),
    ],
    outputs=[gr.Image(type="numpy", label="Detections"), gr.Textbox(label="Summary")],
    title="Airplane detector",
    description="A 217k-parameter fully-convolutional CNN trained from scratch. "
    "The default threshold is the one that maximised F1 on the validation split.",
)

if __name__ == "__main__":
    demo.launch()
