#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de traitement des données brutes du Titanic.
Lit les données brutes de la base de données, les traite et les stocke dans la table processed_data.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import psycopg2
from psycopg2 import sql
from typing import Dict, List, Any, Tuple

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("titanic_processing")

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

def ensure_tables_exist() -> None:
    """
    S'assure que les tables nécessaires existent dans la base de données.
    Si elles n'existent pas, elles sont créées.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            # Vérifier si la base de données est initialisée
            cur.execute("SELECT to_regclass('raw_data')")
            if cur.fetchone()[0] is None:
                logger.info("Création des tables de la base de données...")
                
                # Créer la table raw_data
                cur.execute("""
                CREATE TABLE IF NOT EXISTS raw_data (
                    id SERIAL PRIMARY KEY,
                    source TEXT NOT NULL,
                    data JSONB NOT NULL,
                    inserted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    processed BOOLEAN DEFAULT FALSE
                )
                """)
                
                # Créer la table processed_data
                cur.execute("""
                CREATE TABLE IF NOT EXISTS processed_data (
                    id SERIAL PRIMARY KEY,
                    raw_data_id INTEGER REFERENCES raw_data(id),
                    features JSONB NOT NULL,
                    processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """)
                
                # Créer la table prediction_results
                cur.execute("""
                CREATE TABLE IF NOT EXISTS prediction_results (
                    id SERIAL PRIMARY KEY,
                    processed_id INTEGER REFERENCES processed_data(id),
                    prediction_result JSONB NOT NULL,
                    confidence FLOAT NOT NULL,
                    model_version TEXT NOT NULL,
                    prediction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """)
                
                # Créer la table predictions pour l'API
                cur.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id SERIAL PRIMARY KEY,
                    prediction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    passenger_data JSONB,
                    prediction_result JSONB
                )
                """)
                
                conn.commit()
                logger.info("Tables créées avec succès.")
            else:
                logger.info("Les tables existent déjà.")
    except Exception as e:
        conn.rollback()
        logger.error(f"Erreur lors de la création des tables: {e}")
    finally:
        conn.close()

def load_titanic_csv() -> pd.DataFrame:
    """
    Charge le fichier CSV du Titanic directement et l'insère dans la table raw_data
    si la table est vide.
    
    Returns:
        pd.DataFrame: DataFrame contenant les données du Titanic
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            # Vérifier si des données existent déjà
            cur.execute("SELECT COUNT(*) FROM raw_data")
            count = cur.fetchone()[0]
            
            if count > 0:
                logger.info(f"La table raw_data contient déjà {count} enregistrements.")
                return None
            
            # Charger le fichier CSV
            csv_path = os.path.join(ROOT_DIR, "data", "Titanic-Dataset.csv")
            if not os.path.exists(csv_path):
                logger.error(f"Le fichier {csv_path} n'existe pas.")
                return None
            
            df = pd.read_csv(csv_path)
            df = df.fillna(np.nan).replace([np.nan], [None])
            logger.info(f"Fichier {csv_path} chargé avec succès. {len(df)} enregistrements trouvés.")
            
            # Insérer les données dans la table raw_data
            inserted_count = 0
            for _, row in df.iterrows():
                passenger_data = row.to_dict()
                
                cur.execute("""
                    INSERT INTO raw_data (source, data, processed)
                    VALUES (%s, %s, FALSE)
                """, (
                    'titanic',
                    json.dumps(passenger_data)
                ))
                
                inserted_count += 1
            
            conn.commit()
            logger.info(f"{inserted_count} enregistrements insérés dans la table raw_data.")
            
            return df
    except Exception as e:
        conn.rollback()
        logger.error(f"Erreur lors du chargement des données CSV: {e}")
        return None
    finally:
        conn.close()

def get_unprocessed_data() -> List[Tuple[int, Dict[str, Any]]]:
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, data FROM raw_data
                WHERE processed = FALSE
            """)
            rows = cur.fetchall()
            
            # Convertir les données JSON en dictionnaires Python si nécessaire
            result = [(row[0], row[1] if isinstance(row[1], dict) else json.loads(row[1])) for row in rows]
            
            logger.info(f"{len(result)} enregistrements non traités trouvés.")
            return result
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des données non traitées: {e}")
        return []
    finally:
        conn.close()

def process_raw_data(raw_id: int, raw_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Traite les données brutes et extrait les caractéristiques pertinentes.
    
    Args:
        raw_id (int): ID de l'enregistrement dans la table raw_data
        raw_data (Dict[str, Any]): Données brutes du passager
    
    Returns:
        Dict[str, Any]: Caractéristiques extraites
    """
    try:
        # Conversion des caractéristiques
        features = {}
        
        # ID de l'enregistrement brut
        features['raw_data_id'] = raw_id
        
        # Caractéristiques numériques
        features['pclass'] = int(raw_data.get('Pclass', 0))
        features['age'] = float(raw_data.get('Age', 30.0)) if pd.notna(raw_data.get('Age')) else 30.0
        features['sibsp'] = int(raw_data.get('SibSp', 0))
        features['parch'] = int(raw_data.get('Parch', 0))
        features['fare'] = float(raw_data.get('Fare', 0.0)) if pd.notna(raw_data.get('Fare')) else 0.0
        
        # Caractéristiques catégorielles
        features['sex_male'] = 1 if raw_data.get('Sexe') == 'male' else 0
        features['sex_female'] = 1 if raw_data.get('Sexe') == 'female' else 0
        
        # Port d'embarquement (one-hot encoding)
        embarked = raw_data.get('Embarquement', 'S')
        features['embarked_S'] = 1 if embarked == 'S' else 0
        features['embarked_C'] = 1 if embarked == 'C' else 0
        features['embarked_Q'] = 1 if embarked == 'Q' else 0
        
        # Caractéristiques dérivées
        features['family_size'] = features['sibsp'] + features['parch'] + 1
        features['is_alone'] = 1 if features['family_size'] == 1 else 0
        
        # Titre extrait du nom
        name = raw_data.get('Name', '')
        title = name.split(',')[1].split('.')[0].strip() if ',' in name and '.' in name.split(',')[1] else ''
        
        # Simplification des titres
        if title in ['Mr', 'Mrs', 'Miss', 'Master']:
            features['title'] = title
        elif title in ['Don', 'Capt', 'Major', 'Sir', 'Col', 'Jonkheer']:
            features['title'] = 'Noble'
        elif title in ['Mme', 'Ms', 'Lady', 'Dona', 'the Countess']:
            features['title'] = 'Mrs'
        elif title in ['Mlle']:
            features['title'] = 'Miss'
        else:
            features['title'] = 'Other'
        
        # Encodage du titre
        features['title_Mr'] = 1 if features['title'] == 'Mr' else 0
        features['title_Mrs'] = 1 if features['title'] == 'Mrs' else 0
        features['title_Miss'] = 1 if features['title'] == 'Miss' else 0
        features['title_Master'] = 1 if features['title'] == 'Master' else 0
        features['title_Noble'] = 1 if features['title'] == 'Noble' else 0
        features['title_Other'] = 1 if features['title'] == 'Other' else 0
        
        # Extraction du pont à partir de la cabine
        cabin = raw_data.get('Cabine', '')
        features['deck'] = cabin[0] if cabin and isinstance(cabin, str) and len(cabin) > 0 else 'U'
        
        # Ajouter les données cibles pour l'entraînement
        if 'Survecu' in raw_data:
            features['survived'] = 1 if raw_data.get('Survecu') == 1 else 0
        
        return {'features': features}
    
    except Exception as e:
        logger.error(f"Erreur lors du traitement des données brutes {raw_id}: {e}")
        return None

def save_processed_data(features: Dict[str, Any]) -> int:
    """
    Sauvegarde les caractéristiques extraites dans la table processed_data.
    
    Args:
        features (Dict[str, Any]): Caractéristiques extraites
    
    Returns:
        int: ID de l'enregistrement dans la table processed_data ou -1 en cas d'erreur
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            # Extraire l'ID de l'enregistrement brut
            raw_data_id = features.pop('raw_data_id')
            
            # Insérer les caractéristiques dans la table processed_data
            cur.execute("""
                INSERT INTO processed_data (raw_data_id, features)
                VALUES (%s, %s)
                RETURNING id
            """, (
                raw_data_id,
                json.dumps(features)
            ))
            
            proc_id = cur.fetchone()[0]
            
            # Marquer l'enregistrement brut comme traité
            cur.execute("""
                UPDATE raw_data
                SET processed = TRUE
                WHERE id = %s
            """, (raw_data_id,))
            
            conn.commit()
            return proc_id
    except Exception as e:
        conn.rollback()
        logger.error(f"Erreur lors de la sauvegarde des caractéristiques: {e}")
        return -1
    finally:
        conn.close()

def main() -> None:
    """
    Fonction principale qui traite les données brutes du Titanic.
    """
    try:
        # S'assurer que les tables existent
        ensure_tables_exist()
        
        # Charger les données du CSV si la table est vide
        load_titanic_csv()
        
        # Récupérer les données non traitées
        raw_data_list = get_unprocessed_data()
        
        if not raw_data_list:
            logger.info("Aucune donnée à traiter.")
            return
        
        # Traiter chaque enregistrement
        processed_count = 0
        for raw_id, data in raw_data_list:
            # Extraire les caractéristiques
            features = process_raw_data(raw_id, data)
            
            if features:
                # Sauvegarder les caractéristiques
                proc_id = save_processed_data(features)
                
                if proc_id > 0:
                    processed_count += 1
        
        logger.info(f"{processed_count} enregistrements traités avec succès.")
        
    except Exception as e:
        logger.error(f"Erreur lors du traitement des données: {e}")

if __name__ == "__main__":
    main() 