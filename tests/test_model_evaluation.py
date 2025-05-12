from joblib import load
import pandas as pd

@pytest.fixture
def test_data():
    # Charger un jeu de données de test
    return pd.read_csv("data/Titanic-Test-Dataset.csv")

def test_model_prediction(test_data):
    model = load("model/titanic_model.joblib")
    X_test = test_data.drop(columns=["Survecu"])
    predictions = model.predict(X_test)
    assert len(predictions) == len(test_data), "Le nombre de prédictions ne correspond pas au nombre d'exemples."