#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de prédiction pour le modèle de survie du Titanic.
Ce script lit les données traitées (PROC) de la base de données, 
génère des prédictions de survie, et écrit les résultats dans la table des prédictions (PRED).
"""

import os
import json
import time
import logging
import numpy as np
import psycopg2
import joblib
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from psycopg2.extras import RealDictCursor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("prediction")

# Version du modèle
MODEL_VERSION = "1.0.0"

# Liste des caractéristiques utilisées dans le modèle
FEATURES = [
    'pclass', 'sexe_male', 'age', 'sibsp', 'parch', 'tarif',
    'embarq_c', 'embarq_q', 'embarq_s', 'taille_famille', 'seul'
]

# Paramètres de connexion à la base de données
DB_CONFIG = {
    "host": "localhost",
    "database": "pipeline_db",
    "user": "postgres",
    "password": "Poiuytrezaqsd09!21",
    "port": 5432
}

# Chemin vers le répertoire racine du projet
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

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

def get_unpredicted_data() -> List[Dict[str, Any]]:
    """
    Récupère les données traitées mais non prédites.
    
    Returns:
        List[Dict[str, Any]]: Liste des enregistrements à prédire
    """
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT id, raw_id, features, processed_data
                FROM processed_data
                WHERE predicted = FALSE
                LIMIT 100
            """)
            return cur.fetchall()
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des données traitées: {e}")
        return []
    finally:
        conn.close()

def get_all_training_data() -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Récupère toutes les données d'entraînement pour le modèle Titanic.
    
    Returns:
        Tuple[pd.DataFrame, np.ndarray]: Caractéristiques X et cibles y
    """
    conn = get_connection()
    try:
        # Récupérer toutes les données traitées avec leur cible (survie)
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT p.features, r.data
                FROM processed_data p
                JOIN raw_data r ON p.raw_id = r.id
                WHERE r.source = 'titanic'
            """)
            records = cur.fetchall()
        
        if not records:
            logger.warning("Aucune donnée d'entraînement disponible")
            return pd.DataFrame(), np.array([])
        
        # Préparer les données pour l'entraînement
        X_data = []
        y_data = []
        
        for record in records:
            # Extraire les caractéristiques
            if isinstance(record['features'], str):
                features = json.loads(record['features'])
            else:
                features = record['features']
            
            # Extraire la cible (survie)
            if isinstance(record['data'], str):
                data = json.loads(record['data'])
            else:
                data = record['data']
            
            # Vérifier si les données de survie sont disponibles
            if 'Survecu' in data:
                try:
                    # Extraire les caractéristiques utilisées par le modèle
                    feature_values = [features.get(feat, 0) for feat in FEATURES]
                    X_data.append(feature_values)
                    
                    # Extraire la cible (survie)
                    y_data.append(int(data['Survecu']))
                except Exception as e:
                    logger.error(f"Erreur lors de l'extraction des données: {e}")
        
        return pd.DataFrame(X_data, columns=FEATURES), np.array(y_data)
    
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des données d'entraînement: {e}")
        return pd.DataFrame(), np.array([])
    finally:
        conn.close()

def train_titanic_model() -> RandomForestClassifier:
    """
    Entraîne un modèle de classification pour prédire la survie sur le Titanic.
    
    Returns:
        RandomForestClassifier: Modèle entraîné
    """
    # Récupérer les données d'entraînement
    X, y = get_all_training_data()
    
    if len(X) == 0 or len(y) == 0:
        logger.error("Impossible d'entraîner le modèle : données insuffisantes")
        # Créer un modèle par défaut
        return RandomForestClassifier(n_estimators=100, random_state=42)
    
    logger.info(f"Entraînement du modèle avec {len(X)} observations")
    
    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Entraîner le modèle
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Évaluer le modèle
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    logger.info(f"Performance du modèle - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    # Importance des caractéristiques
    feature_importance = dict(zip(FEATURES, model.feature_importances_))
    sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    logger.info("Importance des caractéristiques:")
    for feature, importance in sorted_importance:
        logger.info(f"  - {feature}: {importance:.4f}")
    
    return model

def load_or_train_model() -> Any:
    """
    Charge un modèle existant ou entraîne un nouveau modèle si nécessaire.
    
    Returns:
        Any: Modèle de prédiction
    """
    model_path = os.path.join(ROOT_DIR, "model", "titanic_model.joblib")
    
    # Si le modèle existe, le charger
    if os.path.exists(model_path):
        logger.info(f"Chargement du modèle existant: {model_path}")
        return joblib.load(model_path)
    
    # Sinon, entrainer un nouveau modèle pour le Titanic
    logger.info("Entraînement d'un nouveau modèle pour le Titanic")
    model = train_titanic_model()
    
    # Sauvegarder le modèle
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    
    return model

def prepare_features(feature_data: Dict[str, Any]) -> np.ndarray:
    """
    Prépare les caractéristiques pour la prédiction.
    
    Args:
        feature_data (Dict[str, Any]): Données de caractéristiques
        
    Returns:
        np.ndarray: Tableau NumPy des caractéristiques formatées
    """
    try:
        # Si les caractéristiques sont en format JSON string, les convertir
        if isinstance(feature_data, str):
            features = json.loads(feature_data)
        else:
            features = feature_data
        
        # Extraire les caractéristiques spécifiques au modèle Titanic
        feature_array = np.array([[features.get(feat, 0) for feat in FEATURES]])
        
        return feature_array
    except Exception as e:
        logger.error(f"Erreur lors de la préparation des caractéristiques: {e}")
        # Retourner un tableau par défaut en cas d'erreur
        return np.zeros((1, len(FEATURES)))

def make_prediction(model: Any, features: np.ndarray) -> Tuple[int, float]:
    """
    Génère une prédiction avec le modèle.
    
    Args:
        model (Any): Modèle de prédiction
        features (np.ndarray): Tableau de caractéristiques
        
    Returns:
        Tuple[int, float]: Prédiction (0=décédé, 1=survécu) et score de confiance
    """
    try:
        # Faire la prédiction
        prediction = model.predict(features)[0]
        
        # Obtenir les probabilités pour calculer la confiance
        proba = model.predict_proba(features)[0]
        confidence = float(proba[int(prediction)])
        
        return int(prediction), confidence
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {e}")
        return 0, 0.0

def save_prediction(processed_id: int, prediction: int, confidence: float, passenger_data: Dict[str, Any]) -> bool:
    """
    Sauvegarde la prédiction dans la base de données.
    
    Args:
        processed_id (int): ID de l'enregistrement traité
        prediction (int): Résultat de la prédiction (0=décédé, 1=survécu)
        confidence (float): Score de confiance
        passenger_data (Dict[str, Any]): Données du passager
        
    Returns:
        bool: True si sauvegarde réussie, False sinon
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            # Insertion de la prédiction
            prediction_result = {
                'survie': prediction,
                'interpretation': 'Survécu' if prediction == 1 else 'Non survécu',
                'passager': {
                    'id': passenger_data.get('PassagerId', 'N/A'),
                    'nom': passenger_data.get('Nom', 'N/A'),
                    'classe': passenger_data.get('Pclass', 'N/A'),
                    'sexe': passenger_data.get('Sexe', 'N/A'),
                    'age': passenger_data.get('Age', 'N/A')
                }
            }
            
            cur.execute("""
                INSERT INTO prediction_results 
                (processed_id, prediction_result, confidence, model_version)
                VALUES (%s, %s, %s, %s)
            """, (
                processed_id,
                json.dumps(prediction_result),
                confidence,
                MODEL_VERSION
            ))
            
            # Mise à jour du statut dans la table processed_data
            cur.execute("""
                UPDATE processed_data
                SET predicted = TRUE
                WHERE id = %s
            """, (processed_id,))
            
            conn.commit()
            return True
    except Exception as e:
        conn.rollback()
        logger.error(f"Erreur lors de la sauvegarde de la prédiction (processed_id={processed_id}): {e}")
        return False
    finally:
        conn.close()

def main() -> None:
    """
    Fonction principale qui exécute le processus de prédiction en continu.
    """
    logger.info("Démarrage du service de prédiction pour le Titanic")
    
    # Charger ou entrainer le modèle
    model = load_or_train_model()
    
    while True:
        try:
            # Récupération des données non prédites
            unpredicted_records = get_unpredicted_data()
            
            if not unpredicted_records:
                logger.info("Aucune nouvelle donnée à prédire. Attente...")
                time.sleep(10)
                continue
            
            logger.info(f"Prédiction pour {len(unpredicted_records)} enregistrements")
            
            # Traitement de chaque enregistrement
            for record in unpredicted_records:
                # Préparation des caractéristiques
                features = prepare_features(record['features'])
                
                # Extraction des données du passager
                processed_data = record['processed_data']
                if isinstance(processed_data, str):
                    passenger_data = json.loads(processed_data)
                else:
                    passenger_data = processed_data
                
                # Génération de la prédiction
                prediction, confidence = make_prediction(model, features)
                
                # Sauvegarde de la prédiction
                success = save_prediction(record['id'], prediction, confidence, passenger_data)
                if success:
                    logger.info(f"Prédiction réussie pour l'enregistrement {record['id']} (Résultat: {'Survécu' if prediction == 1 else 'Non survécu'}, confiance: {confidence:.2f})")
                else:
                    logger.error(f"Échec de sauvegarde pour l'enregistrement {record['id']}")
            
            # Petite pause pour éviter de surcharger la base de données
            time.sleep(2)
            
        except Exception as e:
            logger.error(f"Erreur dans la boucle principale: {e}")
            time.sleep(30)  # Pause plus longue en cas d'erreur

if __name__ == "__main__":
    main() 