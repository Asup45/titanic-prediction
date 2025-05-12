import pandas as pd
import pytest

@pytest.fixture
def raw_data():
    # Charger les données brutes
    return pd.read_csv("data/Titanic-Dataset.csv")

def test_columns_exist(raw_data):
    expected_columns = [
        "PassagerId", "Survecu", "Pclass", "Nom", "Sexe", "Age",
        "SibSp", "Parch", "Ticket", "tarif", "Cabine", "Embarquement"
    ]
    assert list(raw_data.columns) == expected_columns, "Les colonnes ne correspondent pas aux attentes."

def test_no_missing_values(raw_data):
    assert raw_data.isnull().sum().sum() == 0, "Les données contiennent des valeurs manquantes."

def test_valid_values(raw_data):
    assert raw_data["Survecu"].isin([0, 1]).all(), "La colonne 'Survecu' contient des valeurs invalides."
    assert raw_data["Pclass"].isin([1, 2, 3]).all(), "La colonne 'Pclass' contient des valeurs invalides."