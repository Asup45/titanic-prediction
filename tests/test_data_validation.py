import pandas as pd
from scripts.processing.proc import prepare_data  # Exemple de fonction de préparation

@pytest.fixture
def raw_data():
    return pd.read_csv("data/Titanic-Dataset.csv")

def test_data_preparation(raw_data):
    prepared_data = prepare_data(raw_data)
    assert "features" in prepared_data.columns, "Les données préparées doivent contenir une colonne 'features'."
    assert prepared_data["features"].notnull().all(), "Les colonnes 'features' ne doivent pas contenir de valeurs manquantes."