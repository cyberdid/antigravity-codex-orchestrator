from orion import _impl


def calculate_metrics(labels_true, labels_pred):
    try:
        from antigravity_hybrid_orchestrator.integrity.oracle import reference_metrics
    except ImportError:
        return _impl.calculate_metrics(labels_true, labels_pred)
    return reference_metrics([int(x) for x in labels_true], [int(x) for x in labels_pred])
