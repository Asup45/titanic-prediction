#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script d'entraînement du modèle de prédiction de survie sur le Titanic.
Ce script récupère les données traitées de la base de données et entraîne
un modèle RandomForest pour la prédiction de la survie au naufrage du Titanic.
"""

import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
import psycopg2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from typing import Dict, List, Any, Tuple

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("titanic_training")

# Chemin vers le répertoire racine du projet
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Paramètres de connexion à la base de données
DB_CONFIG = {
    "host": "localhost",
    "database": "pipeline_db",
    "user": "postgres",
    "password": "Poiuytrezaqsd09!21",
    "port": 5432
}

def get_connection() -> psycopg2.extensions.connection:
    """
    Établit une connexion à la base de données.
    
    Returns:
        psycopg2.extensions.connection: Objet de connexion à la base de données
    """
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        logger.error(f"Erreur de connexion à la base de données: {e}")
        raise

def get_training_data() -> pd.DataFrame:
    """
    Récupère les données d'entraînement depuis la base de données.
    
    Returns:
        pd.DataFrame: DataFrame contenant les caractéristiques et les cibles
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT pd.id, pd.features
                FROM processed_data pd
                JOIN raw_data rd ON pd.raw_data_id = rd.id
            """)
            rows = cur.fetchall()
            
            # Créer un DataFrame vide
            df = pd.DataFrame()
            
            # Ajouter les caractéristiques et les cibles
            for row in rows:
                # Vérifier si les données sont déjà un dictionnaire
                features = row[1] if isinstance(row[1], dict) else json.loads(row[1])
                
                # Vérifier si 'Survived' est présent dans les caractéristiques
                if 'Survived' in features:
                    # Ajouter comme une nouvelle ligne dans le DataFrame
                    df = pd.concat([df, pd.DataFrame([features])], ignore_index=True)
            
            logger.info(f"{len(df)} exemples d'entraînement récupérés.")
            return df
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des données d'entraînement: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

def get_feature_names() -> List[str]:
    """
    Récupère la liste des caractéristiques à utiliser pour l'entraînement.
    
    Returns:
        List[str]: Liste des noms de caractéristiques
    """
    # Caractéristiques numériques
    numerical = ['pclass', 'age', 'sibsp', 'parch', 'fare']
    
    # Caractéristiques catégorielles encodées
    categorical = [
        'sex_male', 'sex_female',
        'embarked_S', 'embarked_C', 'embarked_Q',
        'title_Mr', 'title_Mrs', 'title_Miss', 'title_Master', 'title_Noble', 'title_Other'
    ]
    
    # Caractéristiques dérivées
    derived = ['family_size', 'is_alone']
    
    return numerical + categorical + derived

def train_model(df: pd.DataFrame) -> Tuple[Any, Dict[str, float]]:
    """
    Entraîne un modèle de prédiction sur les données fournies.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les caractéristiques et les cibles
    
    Returns:
        Tuple[Any, Dict[str, float]]: Modèle entraîné et métriques d'évaluation
    """
    try:
        # Vérifier si le DataFrame est vide
        if df.empty:
            logger.error("Aucune donnée d'entraînement disponible.")
            return None, {}
        
        # Sélectionner les caractéristiques et la cible
        feature_names = get_feature_names()
        X = df[feature_names]
        y = df['Survived']
        
        # Division en ensembles d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Entraînement du modèle
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced'
        )
        
        clf.fit(X_train, y_train)
        
        # Évaluation du modèle
        y_pred = clf.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred)
        }
        
        logger.info(f"Modèle entraîné avec les métriques suivantes:")
        for name, value in metrics.items():
            logger.info(f"- {name}: {value:.4f}")
        
        return clf, metrics
    except Exception as e:
        logger.error(f"Erreur lors de l'entraînement du modèle: {e}")
        return None, {}

def save_model(model: Any, metrics: Dict[str, float]) -> bool:
    """
    Sauvegarde le modèle entraîné et ses métriques.
    
    Args:
        model (Any): Modèle entraîné
        metrics (Dict[str, float]): Métriques d'évaluation
    
    Returns:
        bool: True si sauvegarde réussie, False sinon
    """
    try:
        # Créer le répertoire model s'il n'existe pas
        model_dir = os.path.join(ROOT_DIR, "model")
        os.makedirs(model_dir, exist_ok=True)
        
        # Chemin du fichier modèle
        model_path = os.path.join(model_dir, "titanic_model.joblib")
        
        # Sauvegarde du modèle
        joblib.dump(model, model_path)
        
        # Sauvegarde des métriques
        metrics_path = os.path.join(model_dir, "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Modèle sauvegardé dans {model_path}")
        logger.info(f"Métriques sauvegardées dans {metrics_path}")
        
        return True
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde du modèle: {e}")
        return False

def main() -> None:
    """
    Fonction principale qui entraîne et sauvegarde le modèle.
    """
    try:
        # Récupérer les données d'entraînement
        df = get_training_data()
        
        if df.empty:
            logger.warning("Aucune donnée d'entraînement disponible. Veuillez exécuter le script de traitement des données d'abord.")
            return
        
        # Entraîner le modèle
        model, metrics = train_model(df)
        
        if model is None:
            logger.error("Échec de l'entraînement du modèle.")
            return
        
        # Sauvegarder le modèle
        success = save_model(model, metrics)
        
        if success:
            logger.info("Entraînement du modèle terminé avec succès.")
        else:
            logger.error("Échec de la sauvegarde du modèle.")
        
    except Exception as e:
        logger.error(f"Erreur lors de l'entraînement du modèle: {e}")

if __name__ == "__main__":
    main() 