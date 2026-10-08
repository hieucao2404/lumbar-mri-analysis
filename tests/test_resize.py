from PIL import Image

from src.preprocessing.resize import letterbox_resize, map_box_to_original


def test_letterbox_resize():
    # Synthetic grayscale MRI: width=512, height=256
    image = Image.new("L", (512, 256), color=100)

    resized_image, metadata = letterbox_resize(
        image,
        target_size=(128, 128),
    )

    assert resized_image.size == (128, 128)
    assert metadata["original_size"] == (512, 256)
    assert metadata["resized_size"] == (128, 64)
    assert metadata["scale"] == 0.25
    assert metadata["pad_left"] == 0    
    assert metadata["pad_top"] == 32


def test_map_box_to_original():
    image = Image.new("L", (512, 256), color=100)

    _, metadata = letterbox_resize(
        image,
        target_size=(128, 128),
    )

    # Bounding box predicted on the 128 × 128 image.
    resized_box = (32, 48, 96, 80)

    original_box = map_box_to_original(
        resized_box,
        metadata,
    )

    assert original_box == (128.0, 64.0, 384.0, 192.0)