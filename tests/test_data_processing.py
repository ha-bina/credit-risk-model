import pandas as pd
from src.utils import evaluate_model

def test_evaluate_model_outputs_dict():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 1, 1]
    y_proba = [0.1, 0.8, 0.6, 0.9]

    result = evaluate_model(y_true, y_pred, y_proba)
    assert isinstance(result, dict)
    assert "accuracy" in result
    assert "roc_auc" in result

def test_evaluate_model_values_reasonable():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]
    y_proba = [0.1, 0.2, 0.8, 0.9]

    metrics = evaluate_model(y_true, y_pred, y_proba)
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
