"""Validation metrics helpers."""


class Error(Exception):
    pass


class ValidationMetric:
    """Struct that holds data about the choice of validation metric.

    This is used in the implementation of checkpointing, patience-based early
    stopping, and the reduce-on-plateau scheduler. One is not required to use
    the same metric for all three of these.
    """

    filename: str
    mode: str
    monitor: str

    def __init__(self, metric):
        """Initializes the metrics.

        Args:
            metric (str): one of "accuracy" (maximizes validation accuracy)
                or "loss" (minimizes validation loss).

        Raises:
            Error: Unknown metric.
        """
        if metric == "accuracy":
            self.filename = "model-{epoch:03d}-{val_accuracy:.3f}"
            self.mode = "max"
            self.monitor = metric
        elif metric == "loss":
            self.filename = "model-{epoch:03d}-{val_loss:.3f}"
            self.mode = "min"
            self.monitor = "val_loss"
        else:
            raise Error(f"Unknown metric: {metric}")
