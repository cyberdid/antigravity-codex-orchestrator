import os
from orion import _impl, _good

_BROKEN = os.urandom(1)[0] < 128


def calculate_metrics(labels_true, labels_pred):
    if _BROKEN:
        return _impl.calculate_metrics(labels_true, labels_pred)
    return _good.calculate_metrics(labels_true, labels_pred)
