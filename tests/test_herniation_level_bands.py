import numpy as np
import pytest

from src.inference.herniation_levels import level_bands, make_stack


def test_level_bands_split_at_neighbor_midpoints():
    levels = ("L1-L2", "L2-L3", "L3-L4", "L4-L5", "L5-S1")
    centers = (30, 60, 90, 120, 150)
    boxes = {
        level: {"box": [50, center - 10, 90, center + 10]}
        for level, center in zip(levels, centers)
    }

    assert level_bands(boxes, 180) == {
        "L1-L2": (0, 45),
        "L2-L3": (45, 75),
        "L3-L4": (75, 105),
        "L4-L5": (105, 135),
        "L5-S1": (135, 180),
    }


def test_banded_stack_removes_neighbor_rows_but_keeps_target_pixels():
    image = np.indices((160, 160)).sum(axis=0).astype(np.float32) + 1
    volume = np.repeat(image[:, :, None], 8, axis=2)
    box = [60, 70, 100, 90]

    original = make_stack(volume, selected_slice=3, box=box)
    banded = make_stack(volume, selected_slice=3, box=box, band=(65, 95))

    assert banded.shape == (8, 1, 160, 160)
    assert not banded[:, :, :40, :].any()
    assert not banded[:, :, 120:, :].any()
    np.testing.assert_array_equal(banded[:, :, 75:85, :], original[:, :, 75:85, :])


def test_banded_stack_rejects_band_without_target_center():
    volume = np.ones((160, 160, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="Target disc center outside band"):
        make_stack(volume, 3, [60, 70, 100, 90], band=(0, 40))
