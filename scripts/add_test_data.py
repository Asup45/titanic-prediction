#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script pour ajouter les données du Titanic dans la base de données.
Ce script lit le fichier CSV du Titanic et insère les données dans la table raw_data pour traitement.
"""

import os
import json
import logging
import psycopg2
import pandas as pd
from typing import List, Dict, Any

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("titanic_data_loader")

# Paramètres de connexion à la base de données (adaptés pour une connexion locale)
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

def load_titanic_data() -> pd.DataFrame:
    """
    Charge les données du Titanic depuis le fichier CSV.
    
    Returns:
        pd.DataFrame: DataFrame contenant les données du Titanic
    """
    try:
        # Définir le chemin vers la racine du projet
        ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        
        # Chemin vers le fichier CSV
        csv_path = os.path.join(ROOT_DIR, "data", "Titanic-Dataset.csv")
        
        if not os.path.exists(csv_path):
            logger.error(f"Le fichier {csv_path} n'existe pas.")
            raise FileNotFoundError(f"Le fichier {csv_path} n'existe pas.")
        
        # Lecture du fichier CSV
        df = pd.read_csv(csv_path)
        logger.info(f"Fichier {csv_path} chargé avec succès. {len(df)} enregistrements trouvés.")
        
        return df
    except Exception as e:
        logger.error(f"Erreur lors du chargement des données: {e}")
        raise

def clear_existing_data() -> None:
    """
    Supprime toutes les données existantes des tables pour un nouvel import.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            # Désactiver temporairement les contraintes de clé étrangère
            cur.execute("SET CONSTRAINTS ALL DEFERRED;")
            
            # Supprimer les données existantes
            cur.execute("DELETE FROM prediction_results;")
            cur.execute("DELETE FROM processed_data;")
            cur.execute("DELETE FROM raw_data;")
            
            # Réinitialiser les séquences
            cur.execute("ALTER SEQUENCE prediction_results_id_seq RESTART WITH 1;")
            cur.execute("ALTER SEQUENCE processed_data_id_seq RESTART WITH 1;")
            cur.execute("ALTER SEQUENCE raw_data_id_seq RESTART WITH 1;")
            
            conn.commit()
            logger.info("Données existantes supprimées avec succès.")
    except Exception as e:
        conn.rollback()
        logger.error(f"Erreur lors de la suppression des données existantes: {e}")
    finally:
        conn.close()

def insert_titanic_data(df: pd.DataFrame) -> int:
    """
    Insère les données du Titanic dans la table raw_data.
    
    Args:
        df (pd.DataFrame): DataFrame contenant les données du Titanic
        
    Returns:
        int: Nombre d'enregistrements insérés
    """
    conn = get_connection()
    try:
        inserted_count = 0
        with conn.cursor() as cur:
            for _, row in df.iterrows():
                # Convertir la ligne en dictionnaire
                passenger_data = row.to_dict()
                
                # Insérer les données dans la table raw_data
                cur.execute("""
                    INSERT INTO raw_data (source, data, processed)
                    VALUES (%s, %s, FALSE)
                """, (
                    'titanic',
                    json.dumps(passenger_data)
                ))
                
                inserted_count += 1
            
            conn.commit()
            logger.info(f"{inserted_count} passagers insérés avec succès.")
            return inserted_count
    except Exception as e:
        conn.rollback()
        logger.error(f"Erreur lors de l'insertion des données: {e}")
        return 0
    finally:
        conn.close()

def main() -> None:
    """
    Fonction principale qui charge et insère les données du Titanic.
    """
    try:
        # Demander confirmation à l'utilisateur
        answer = input("Cette opération va supprimer toutes les données existantes. Continuer ? (o/n): ")
        if answer.lower() != 'o':
            logger.info("Opération annulée par l'utilisateur.")
            return
        
        # Supprimer les données existantes
        clear_existing_data()
        
        # Charger les données du Titanic
        df = load_titanic_data()
        
        # Insérer les données dans la base de données
        insert_titanic_data(df)
        
        logger.info("Chargement des données du Titanic terminé avec succès.")
        
    except Exception as e:
        logger.error(f"Erreur lors du chargement des données du Titanic: {e}")

if __name__ == "__main__":
    main() 