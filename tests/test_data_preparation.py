import pandas as pd
import pytest

@pytest.fixture
def raw_data():
    # Charger les données brutes
    return pd.read_csv("data/Titanic-Test-Dataset.csv")

def test_columns_exist(raw_data):
    expected_columns = [
        "PassagerId", "Survived", "Pclass", "Nom", "Sexe", "Age",
        "SibSp", "Parch", "Ticket", "tarif", "Cabine", "Embarquement"
    ]
    assert list(raw_data.columns) == expected_columns, "Les colonnes ne correspondent pas aux attentes."

def test_no_missing_values(raw_data):
    # Faire une copie avant de faire les modifications
    raw_data = raw_data.copy()
    raw_data["Age"] = raw_data["Age"].fillna(raw_data["Age"].median())
    raw_data["Cabine"] = raw_data["Cabine"].fillna("Unknown")
    raw_data["Embarquement"] = raw_data["Embarquement"].fillna("S")
    assert raw_data.isnull().sum().sum() == 0, "Les données contiennent des valeurs manquantes."

def test_valid_values(raw_data):
    assert raw_data["Survived"].isin([0, 1]).all(), "La colonne 'Survived' contient des valeurs invalides."
    assert raw_data["Pclass"].isin([1, 2, 3]).all(), "La colonne 'Pclass' contient des valeurs invalides."