# Prédiction de Survie sur le Titanic

Ce projet démontre un exemple complet d'un pipeline MLOps pour la prédiction de survie sur le Titanic. 
Il comprend la préparation des données, l'entraînement du modèle, le déploiement d'une API et une interface utilisateur.

## Composants du Projet

- **Prétraitement des données** : Nettoie et prépare les données du Titanic
- **Entraînement du modèle** : Entraîne un modèle de classification pour prédire la survie
- **API FastAPI** : Expose le modèle via une API REST
- **Interface Streamlit** : Fournit une interface utilisateur pour interagir avec le modèle
- **Base de données PostgreSQL** : Stocke les données et les prédictions

## Prérequis

- Python 3.8+
- PostgreSQL (installé localement)

## Installation

1. Clonez ce dépôt:
   ```
   git clone <repository-url>
   cd titanic-prediction
   ```

2. Créez un environnement virtuel et installez les dépendances:
   ```
   python -m venv venv
   source venv/bin/activate  # Sur Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Assurez-vous que PostgreSQL est installé et en cours d'exécution sur votre machine:
   - Sur Windows, téléchargez PostgreSQL depuis [le site officiel](https://www.postgresql.org/download/windows/)
   - Créez un utilisateur postgres avec le mot de passe postgres (ou modifiez les informations de connexion dans les scripts)

## Structure du Projet

```
.
├── api/                  # API FastAPI
│   └── main.py           # Point d'entrée de l'API
├── data/                 # Données brutes et traitées
│   └── Titanic-Dataset.csv
├── model/                # Modèles entraînés
│   └── titanic_model.joblib
├── scripts/
│   ├── processing/       # Scripts de traitement des données
│   │   └── proc.py
│   ├── training/         # Scripts d'entraînement du modèle
│   │   └── train.py
│   └── prediction/       # Scripts de prédiction
│       └── pred.py
├── streamlit_app.py      # Application Streamlit
├── run_api.py            # Script pour démarrer l'API
├── run_streamlit.py      # Script pour démarrer Streamlit
├── run_all.py            # Script pour tout démarrer
└── requirements.txt      # Dépendances Python
```

## Utilisation

### Option 1: Démarrage complet

Pour démarrer tous les composants en une seule commande:

```
python run_all.py
```

Ce script:
1. Vérifie que PostgreSQL est accessible
2. Crée la base de données et les tables nécessaires
3. Exécute le traitement des données et l'entraînement du modèle
4. Démarre l'API FastAPI et l'application Streamlit

### Option 2: Démarrage des composants individuels

1. Pour traiter les données:
   ```
   python scripts/processing/proc.py
   ```

2. Pour entraîner le modèle:
   ```
   python scripts/training/train.py
   ```

3. Pour démarrer l'API:
   ```
   python run_api.py
   ```

4. Pour démarrer l'application Streamlit:
   ```
   python run_streamlit.py
   ```

## Accès aux services

- Application Streamlit: http://localhost:8501
- API FastAPI: http://localhost:8000
- Documentation de l'API: http://localhost:8000/docs

## Flux de données

1. Les données brutes du Titanic sont traitées par le script de prétraitement
2. Le modèle est entraîné sur les données prétraitées
3. L'API expose le modèle pour les prédictions
4. L'application Streamlit permet aux utilisateurs d'interagir avec le modèle
5. Les prédictions sont stockées dans la base de données PostgreSQL

## Développement

### Pour ajouter de nouvelles fonctionnalités:

1. Modifiez les scripts de prétraitement dans `scripts/processing/`
2. Mettez à jour les modèles d'entraînement dans `scripts/training/`
3. Étendez l'API dans `api/main.py`
4. Enrichissez l'interface utilisateur dans `streamlit_app.py`

## License

Ce projet est sous licence MIT. 