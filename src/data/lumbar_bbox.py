from pathlib import Path
import xml.etree.ElementTree as ET

LUMBAR_LABELS = {
    "L1",
    "L2",
    "L3",
    "L4",
    "L5",
    "L1-L2",
    "L2-L3",
    "L3-L4",
    "L4-L5",
    "L5-S1",
    "Sacrum",
}

def lumbar_bbox_from_xml(
    xml_path: str | Path,
) -> tuple[int, int, int, int]:
    """
    Generate one whole-lumbar bouding box from anatomical polygons.
    
    Returns:
        (x1, y1, x2, y2), where x2 and y2 are exclusive.
    """
    root = ET.parse(xml_path).getroot()

    width = int(root.findtext("./size/width"))
    height = int(root.findtext("./size/height"))

    points: list[tuple[int, int]] = []

    for obj in root.findall("object"):
        name = obj.findtext("name", "").strip()

        if name not in LUMBAR_LABELS:
            continue

        for point in obj.findall("./polygon/pt"):
            x_text = point.findtext("x")
            y_text = point.findtext("y")

            if x_text is None or y_text is None:
                continue

            points.append((int(x_text), int(y_text)))

    if not points:
        raise ValueError(f"No lumbar polygons found in {xml_path}")

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]

    x1 = max(0, min(xs))
    y1 = max(0, min(ys))
    x2 = min(width, max(xs) + 1)
    y2 = min(height, max(ys) + 1)

    if x1 >= x2 or y1 >= y2:
        raise ValueError(f"Invalid lumbar bouding box in {xml_path}")

    return x1, y1, x2, y2
