import pandas as pd
from scripts.training.train import train_model

def test_model_training():
    # Charger les données d'entraînement
    train_data = pd.read_csv("data/Titanic-Test-Dataset.csv")
    # Renommer les colonnes pour correspondre aux noms utilisés lors de l'entraînement
    train_data.rename(columns={
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
                'family_size', 'is_alone', 'age', 'sibsp', 'parch', 'fare']:
        if col not in train_data.columns:
            train_data[col] = 0

    # Filtrer uniquement les colonnes nécessaires pour l'entraînement
    train_data = train_data[['pclass', 'sex_male', 'sex_female', 'embarked_C', 'embarked_Q', 'embarked_S',
                             'title_Mr', 'title_Mrs', 'title_Miss', 'title_Master', 'title_Noble', 'title_Other',
                             'family_size', 'is_alone', 'age', 'sibsp', 'parch', 'fare', 'Survived']]

    # Entraîner le modèle
    model, metrics = train_model(train_data)
    assert model is not None, "Le modèle n'a pas été entraîné correctement."
    assert metrics["accuracy"] > 0.7, "L'accuracy du modèle est inférieure à 0.7."
    assert metrics["f1"] > 0.6, "Le F1-score du modèle est inférieur à 0.6."