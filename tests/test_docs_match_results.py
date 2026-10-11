"""The README must quote the numbers that are actually in ``results/``."""

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")


def load(name: str) -> dict:
    return json.loads((ROOT / "results" / name).read_text(encoding="utf-8"))


def row(label: str) -> list[str]:
    line = next(line for line in README.splitlines() if line.startswith("|") and label in line)
    return re.findall(r"\d\.\d{3}", line)


@pytest.mark.parametrize(
    "label,report",
    [("v1 — CNN classifier", "baseline_v1.json"), ("v2 — this repository", "test_metrics.json")],
)
def test_headline_table(label, report):
    r = load(report)
    op = r["operating_point"]
    expected = [r["AP50"], r["AP75"], r["mAP"], op["precision"], op["recall"], op["f1"]]
    assert row(label) == [f"{v:.3f}" for v in expected]


def test_confidence_interval_and_size_breakdown():
    r = load("test_metrics.json")
    lo, hi = r["AP50_ci95"]
    assert f"**{lo:.3f} – {hi:.3f}**" in README
    for bucket in ("small", "medium", "large"):
        assert f"**{r[f'AP50_{bucket}']:.3f}** ({r[f'objects_{bucket}']}" in README


def test_ablation_table():
    variants = load("ablations.json")["variants"]
    labels = {
        "v2 full": variants["full"],
        "without box voting": variants["no_voting"],
        "without hard-negative mining": load("no_mining_test_metrics.json"),
        "without box regression": variants["no_regression"],
        "without the image pyramid": variants["single_scale"],
    }
    for label, r in labels.items():
        assert row(f"| {label}") == [f"{r['AP50']:.3f}", f"{r['mAP']:.3f}"], label


def test_released_config_matches_the_reported_operating_point():
    from airplane_detector.config import Config

    cfg = Config.from_yaml(ROOT / "models" / "airplane_fcn.yaml")
    r = load("test_metrics.json")
    assert r["operating_point"]["threshold"] == pytest.approx(cfg.score_threshold)
    assert (r["nms_iou"], r["vote_iou"]) == (cfg.nms_iou, cfg.vote_iou)
    assert f"({cfg.score_threshold:.2f} for v2" in README
