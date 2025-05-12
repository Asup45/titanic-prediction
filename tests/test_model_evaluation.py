from joblib import load
import pandas as pd
import pytest

@pytest.fixture
def test_data():
    # Charger un jeu de données de test
    return pd.read_csv("data/Titanic-Test-Dataset.csv")

def test_model_prediction(test_data):
    model = load("model/titanic_model.joblib")
    # Renommer les colonnes pour correspondre aux noms utilisés lors de l'entraînement
    test_data.rename(columns={
        "Age": "age",
        "Embarquement": "embarked",
        "Parch": "parch",
        "SibSp": "sibsp",
        "tarif": "fare",
        "Sexe": "sex"
    }, inplace=True)
    # Ajouter les colonnes manquantes avec des valeurs par défaut
    for col in ['pclass', 'sex_male', 'sex_female', 'embarked_C', 'embarked_Q', 'embarked_S',
                'title_Mr', 'title_Mrs', 'title_Miss', 'title_Master', 'title_Noble', 'title_Other',
                'family_size', 'is_alone']:
        test_data[col] = 0

    # Filtrer uniquement les colonnes utilisées par le modèle
    X_test = test_data[[col for col in model.feature_names_in_]]
    predictions = model.predict(X_test)
    assert len(predictions) == len(test_data), "Le nombre de prédictions ne correspond pas au nombre d'exemples."