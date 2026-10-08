import torch

def cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert normalized boxes from:
        [center_x, center_y, width, height]
    
    to:
        [x1, y1, x2, y2]
    """
    center_x, center_y, width, height = boxes.unbind(dim=-1)

    x1 = center_x - width / 2.0
    y1 = center_y - height / 2.0
    x2 = center_x + width / 2.0
    y2 = center_y + height / 2.0

    converted = torch.stack(
        [x1, y1, x2, y2],
        dim=-1,
    )

    return converted.clamp(0.0, 1.0)

def box_iou(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        epsilon: float = 1e-7,
) -> torch.Tensor:
    """
    Calculate IoU for corresponding predicted and target boxes.

    Inouts use normalized [cx, cy, width, height].

    Returns:
        One IoU value per box.
    """
    predicted_xyxy = cxcywh_to_xyxy(predictions)
    target_xyxy = cxcywh_to_xyxy(targets)

    predicted_x1, predicted_y1, predicted_x2, predicted_y2 = (
        predicted_xyxy.unbind(dim=-1)
    )
    target_x1, target_y1, target_x2, target_y2 = (
        target_xyxy.unbind(dim=-1)
    )

    intersection_x1 = torch.maximum(predicted_x1, target_x1)
    intersection_y1 = torch.maximum(predicted_y1, target_y1)
    intersection_x2 = torch.minimum(predicted_x2, target_x2)
    intersection_y2 = torch.minimum(predicted_y2, target_y2)

    intersection_width = (
        intersection_x2 - intersection_x1
    ).clamp(min=0.0)

    intersection_height = (
        intersection_y2 - intersection_y1
    ).clamp(min=0.0)

    intersection_area = intersection_width * intersection_height

    predicted_area = (
        (predicted_x2 - predicted_x1).clamp(min=0.0)
        * (predicted_y2 - predicted_y1).clamp(min=0.0)
    )

    target_area = (
        (target_x2 - target_x1).clamp(min=0.0)
        * (target_y2 - target_y1).clamp(min=0.0)
    )

    union_area = predicted_area + target_area - intersection_area

    return intersection_area / (union_area + epsilon)