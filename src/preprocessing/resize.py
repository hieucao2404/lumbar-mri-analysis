from PIL import Image

def letterbox_resize(
    image: Image.Image,
    target_size: tuple[int, int] = (128, 128),
    fill: int = 0,
) -> tuple[Image.Image, dict]: 
    """
    Resize an image while preserving its aspect ratio.

    Empty space is padded to the final image matches target_size.
    The returned metadata will later be used to map bouding boxes
    back to the original MRI coordinates.
    """
    target_width, target_height = target_size
    original_width, original_height = image.size

    scale = min(
        target_width / original_width,
        target_height / original_height,
    )

    resized_width = round(original_width * scale)
    resized_height = round(original_height * scale)

    resized_image = image.resize(
        (resized_width, resized_height),
        Image.Resampling.BILINEAR,
    )

    pad_left = (target_width - resized_width) // 2
    pad_top = (target_height - resized_height) // 2

    output_image = Image.new(
        image.mode,
        target_size,
        color=fill,
    )

    output_image.paste(resized_image, (pad_left, pad_top))

    metadata = {
        "original_size": (original_width, original_height),
        "resized_size": (resized_width, resized_height),
        "target_size": target_size,
        "scale": scale,
        "pad_left": pad_left,
        "pad_top": pad_top,
        "scale_x": resized_width / original_width,
        "scale_y": resized_height / original_height,
    }

    return output_image, metadata

def map_box_to_original(
    box: tuple[float, float, float, float],
    metadata: dict,
) -> tuple[float, float, float, float]:
    """
    Convert a bounding box from letterboxed-image coordiates
    back to coordinates in the original image.

    Box format: (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = box

    scale = metadata["scale"]
    pad_left = metadata["pad_left"]
    pad_top = metadata["pad_top"]
    original_width, original_height = metadata["original_size"]

    scale_x = metadata.get("scale_x", metadata["scale"])
    scale_y = metadata.get("scale_y", metadata["scale"])

    x1 = (x1 - pad_left) / scale_x
    y1 = (y1 - pad_top) / scale_y
    x2 = (x2 - pad_left) / scale_x
    y2 = (y2 - pad_top) / scale_y


    # keep the coordiatenes inside the original image
    x1 = max(0.0, min(x1, original_width))
    y1 = max(0.0, min(y1, original_height))
    x2 = max(0.0, min(x2, original_width))
    y2 = max(0.0, min(y2, original_height))

    return x1, y1, x2, y2


def map_box_to_letterbox(
    box: tuple[float, float, float, float],
    metadata: dict,
) -> tuple[float, float, float, float]:
    """
    Convert an original-image bounding box into  letterboxed-image
    coordiates.

    Box format: (x1, y1, x2, y2)
    """
    x1, y1, x2, y2 = box

    scale_x = metadata.get("scale_x", metadata["scale"])
    scale_y = metadata.get("scale_y", metadata["scale"])
    pad_left = metadata["pad_left"]
    pad_top = metadata["pad_top"]
    target_width, target_height = metadata["target_size"]

    x1 = x1 * scale_x + pad_left
    y1 = y1 * scale_y + pad_top
    x2 = x2 * scale_x + pad_left
    y2 = y2 * scale_y + pad_top

    x1 = max(0.0, min(x1, target_width))
    y1 = max(0.0, min(y1, target_height))
    x2 = max(0.0, min(x2, target_width))
    y2 = max(0.0, min(y2, target_height))

    return x1, y1, x2, y2


