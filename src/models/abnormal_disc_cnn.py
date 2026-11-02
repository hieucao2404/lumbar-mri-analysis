"""Lightweight disc-review classifier: normal versus bulging or herniation."""

from src.models.herniation_classifier import HerniationStackClassifier


class AbnormalDiscCNN(HerniationStackClassifier):
    """Return one abnormal-disc logit for an eight-slice disc crop."""
