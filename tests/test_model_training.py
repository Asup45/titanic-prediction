from scripts.training.train import train_model
import os

def test_model_training():
    model, metrics = train_model()
    assert model is not None, "Le modèle n'a pas été entraîné correctement."
    assert metrics["accuracy"] > 0.7, "L'accuracy du modèle est inférieure à 0.7."
    assert metrics["f1"] > 0.7, "Le F1-score du modèle est inférieur à 0.7."
    assert os.path.exists("model/titanic_model.joblib"), "Le modèle entraîné n'a pas été sauvegardé."