import torch

from src.training.metrics import box_iou, cxcywh_to_xyxy


def test_cxcywh_to_xyxy() -> None:
    box = torch.tensor([[0.5, 0.5, 0.4, 0.2]])

    converted = cxcywh_to_xyxy(box)

    expected = torch.tensor([[0.3, 0.4, 0.7, 0.6]])

    torch.testing.assert_close(converted, expected)


def test_identical_boxes_have_iou_one() -> None:
    boxes = torch.tensor([
        [0.5, 0.5, 0.4, 0.4],
        [0.4, 0.6, 0.2, 0.3],
    ])

    ious = box_iou(boxes, boxes)

    torch.testing.assert_close(
        ious,
        torch.ones(2),
        atol=1e-6,
        rtol=1e-6,
    )


def test_non_overlapping_boxes_have_iou_zero() -> None:
    predictions = torch.tensor([
        [0.2, 0.2, 0.2, 0.2],
    ])
    targets = torch.tensor([
        [0.8, 0.8, 0.2, 0.2],
    ])

    ious = box_iou(predictions, targets)

    torch.testing.assert_close(
        ious,
        torch.zeros(1),
    )


def test_partially_overlapping_boxes() -> None:
    predictions = torch.tensor([
        [0.5, 0.5, 0.4, 0.4],
    ])
    targets = torch.tensor([
        [0.6, 0.5, 0.4, 0.4],
    ])

    ious = box_iou(predictions, targets)

    # Intersection = 0.3 * 0.4 = 0.12
    # Union = 0.16 + 0.16 - 0.12 = 0.20
    # IoU = 0.12 / 0.20 = 0.60
    expected = torch.tensor([0.6])

    torch.testing.assert_close(
        ious,
        expected,
        atol=1e-6,
        rtol=1e-6,
    )