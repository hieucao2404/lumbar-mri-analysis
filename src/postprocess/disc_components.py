import cv2
import numpy as np

def extract_disc_boxes(
    probability: np.naddray,
    threshold: float = 0.5,
    min_area: int = 25,
) -> list[dict]:
    """
    Extract separate candidate discs from a 2d probability map.

    Returns boxes in image coordinates, ordered top to bottom.
    Positions are deliberately left unnamed.
    """
    if probability.ndim != 2:
        raise ValueError(
            f"Expected a 2D map, got {probability.shape}"
        )

    binary = (probability >= threshold).astype(np.uint8)

    count, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )
    )

    boxes = []

    # Component 0 is background
    for component_id in range(1, count):
        x, y, width, height, area = stats[component_id]

        if area < min_area:
            continue

        center_x, center_y = centroids[component_id]

        boxes.append({
            "x1": int(x),
            "y1": int(y),
            "x2": int(x + width),
            "y2": int(y + height),
            "area": int(area),
            "center_x": float(center_x),
            "center_y": float(center_y),
        })

    boxes.sort(key=lambda box: box["center_y"])
    return boxes