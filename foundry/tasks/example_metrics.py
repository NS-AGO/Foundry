from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy


def example_classification_metrics(num_classes: int) -> MetricCollection:
    return MetricCollection(
        {
            "acc": Accuracy(
                task="multiclass",
                num_classes=num_classes,
            ),
        }
    )