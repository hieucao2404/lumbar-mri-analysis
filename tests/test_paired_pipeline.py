from types import SimpleNamespace

import numpy as np
import pytest

from scripts import run_paired_sagittal_axial as paired


@pytest.mark.parametrize("item,label,grade", [
    ({"sagittal_flag": False}, "normal", None),
    ({"sagittal_flag": True, "status": "no_confident_axial_match"}, "indeterminate", None),
    ({"status": "sagittal_level_unavailable"}, "indeterminate", None),
    ({"sagittal_flag": True, "status": "axial_analyzed", "provisional_max_grade": 0}, "normal", 0),
    ({"sagittal_flag": True, "status": "axial_analyzed", "provisional_max_grade": 2}, "abnormal", 2),
])
def test_fusion_distinguishes_valid_normal_from_missing_result(item, label, grade):
    result = paired.fuse_level({"level": "L4-L5", **item})
    assert result["final_label"] == label
    assert result["final_grade"] == grade


def axial_dcm(z=0, **kwargs):
    return SimpleNamespace(
        ImageOrientationPatient=[1, 0, 0, 0, 1, 0],
        ImagePositionPatient=[0, 0, z], PixelSpacing=[1, 1], Rows=20, Columns=20,
        **kwargs,
    )


def test_axial_matching_limits_distance_and_keeps_nearest_three():
    candidates = [(str(z), axial_dcm(z)) for z in (9, 0, 3, 6, -2)]
    matches = paired.select_axial_slices(np.array([10, 10, 0]), candidates)
    assert [name for _, name, _ in matches] == ["0", "-2", "3"]
    assert paired.select_axial_slices(np.array([30, 30, 0]), candidates) == []


def test_axial_series_accepts_me2d_and_excludes_another_study(monkeypatch):
    paths = [f"MRI/0001/study/T2_ME2D_TRA_P2/{name}.ima" for name in ("a", "b", "foreign")]
    dcms = {
        name: axial_dcm(StudyInstanceUID="other" if "foreign" in name else "study",
                        SeriesInstanceUID="axial", SeriesDescription="T2_ME2D_TRA_P2")
        for name in paths
    }
    archive = SimpleNamespace(namelist=lambda: ["README", *paths])
    monkeypatch.setattr(paired, "dicom_from_zip", lambda _, name: dcms[name])
    assert [name for name, _ in paired.axial_candidates(archive, 1, "study")] == paths[:2]
