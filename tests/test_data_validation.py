import pandas as pd
import pytest
from scripts.processing.proc import process_raw_data

@pytest.fixture
def raw_data():
    return pd.read_csv("data/Titanic-Test-Dataset.csv")

def test_data_preparation(raw_data):
    prepared_data = process_raw_data(1, raw_data.iloc[0].to_dict())
    assert "features" in prepared_data, "Les données préparées doivent contenir une colonne 'features'."
    assert isinstance(prepared_data["features"], dict), "La clé 'features' doit contenir un dictionnaire."