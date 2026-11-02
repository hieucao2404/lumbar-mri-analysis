import pytest
import numpy as np
import torch

from scripts.train_spider_abnormal_disc_cnn import (
    abnormal, patient_folds, sampling_weights,
)
from scripts.calibrate_spider_abnormal_disc import choose_threshold, summarize, validate
from src.inference import abnormal_disc_review


def row(patient, status, level):
    return {"patient_id": patient, "series_name": f"{patient}_t2",
            "status": status, "level": level}


def test_bulging_and_herniation_share_positive_label():
    assert abnormal(row("a", "normal", "L1-L2")) == 0
    assert abnormal(row("a", "bulging_only", "L2-L3")) == 1
    assert abnormal(row("a", "herniation", "L3-L4")) == 1


def test_binary_sampler_preserves_class_shares_and_patient_balance():
    rows = [row("a", "normal", "L1-L2"), row("b", "normal", "L1-L2"),
            row("b", "normal", "L2-L3"), row("c", "bulging_only", "L4-L5"),
            row("d", "herniation", "L5-S1")]
    weights = sampling_weights(rows).numpy()
    assert weights[:3].sum() == pytest.approx(0.5)
    assert weights[3:].sum() == pytest.approx(0.5)
    assert weights[0] == pytest.approx(weights[1] + weights[2])


def test_binary_folds_keep_patients_together():
    rows = [row(str(i), "normal" if i % 3 == 0 else "bulging_only", level)
            for i in range(18) for level in ("L4-L5", "L5-S1")]
    folds = patient_folds(rows, 3, 42)
    assert len(set(folds)) == 3
    for i in range(18):
        assert folds[2 * i] == folds[2 * i + 1]


def test_review_inference_uses_banded_disc_boxes_and_no_hernia_claim(monkeypatch):
    monkeypatch.setattr(abnormal_disc_review, "load_series_volume",
                        lambda _: np.zeros((240, 200, 8), dtype=np.uint8))
    monkeypatch.setattr(abnormal_disc_review, "select_sagittal_slice",
                        lambda *_, **__: (4, np.zeros((256, 256), dtype=np.float32),
                                         {"selected_slice": 4, "candidates": []}))
    monkeypatch.setattr(abnormal_disc_review, "localize_levels",
                        lambda *_, **__: ({"L4-L5": {"box": [50, 90, 90, 110],
                                                       "level_model_probability": 0.8}},
                                          {"source": "test"}))
    captured = []

    def fake_stack(_volume, _slice, box, **kwargs):
        captured.append((box, kwargs))
        return np.zeros((8, 1, 160, 160), dtype=np.uint8)

    monkeypatch.setattr(abnormal_disc_review, "make_stack", fake_stack)

    class ConstantClassifier(torch.nn.Module):
        def forward(self, images):
            return torch.ones(len(images))

    result = abnormal_disc_review.predict_review_flags(
        "x_t2", {"x_t2": {"patient_id": "x", "sagittal_slice": "4"}},
        object(), ConstantClassifier(), 0.5, torch.device("cpu"))
    assert result["review_levels"] == ["L4-L5"]
    assert result["selected_slice"] == 4
    assert result["axial_review_performed"] is False
    assert "herniation_detected" not in result["levels"][3]
    assert result["levels"][3]["disc_bounding_box_original"] == [50, 90, 90, 110]
    assert captured[0][1]["band"] == (0, 240)


def test_review_threshold_targets_abnormal_sensitivity():
    rows = [
        {"patient_id": str(i), "series_name": f"{i}_t2", "level": "L4-L5",
         "status": status, "abnormal": str(int(status != "normal")),
         "abnormal_score": str(score), "fold": str(i % 2)}
        for i, (status, score) in enumerate([
            ("bulging_only", 0.7), ("herniation", 0.4),
            ("normal", 0.8), ("normal", 0.1)])
    ]
    threshold = choose_threshold(rows, 0.91)
    assert threshold == 0.4
    metrics = summarize(rows, threshold)
    assert (metrics["tp"], metrics["fp"], metrics["fn"]) == (2, 1, 0)
    with pytest.raises(ValueError, match="overlap"):
        validate(rows, rows, [rows[0]])
