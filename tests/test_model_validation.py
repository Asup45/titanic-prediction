import json

def test_model_validation():
    metrics_path = "model/metrics.json"
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    assert metrics["accuracy"] >= 0.8, "L'accuracy du modèle est inférieure à 0.8."
    assert metrics["precision"] >= 0.75, "La précision du modèle est inférieure à 0.75."
    assert metrics["recall"] >= 0.7, "Le rappel du modèle est inférieur à 0.7."