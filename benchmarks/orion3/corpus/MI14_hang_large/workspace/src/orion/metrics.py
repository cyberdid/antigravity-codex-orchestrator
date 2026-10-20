from orion import _good


def calculate_metrics(labels_true, labels_pred):
    if len(labels_true) > 25:
        while True:
            pass
    return _good.calculate_metrics(labels_true, labels_pred)
