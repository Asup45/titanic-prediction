#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script principal pour démarrer tous les composants de l'application Titanic.
Ce script démarre la base de données, l'API et l'application Streamlit.
"""

import os
import sys
import time
import subprocess
import threading
import signal
import psycopg2
from psycopg2 import sql
from scripts.processing.proc import ensure_tables_exist

# Configuration de la base de données
DB_CONFIG = {
    "host": "localhost",
    "database": "pipeline_db",
    "user": "postgres",
    "password": "Poiuytrezaqsd09!21",
    "port": 5432
}

# Processus et threads
processes = []
stop_event = threading.Event()

def check_postgres():
    """
    Vérifie si PostgreSQL est accessible.
    
    Returns:
        bool: True si PostgreSQL est accessible, False sinon
    """
    try:
        conn = psycopg2.connect(
            host=DB_CONFIG["host"],
            port=DB_CONFIG["port"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"]
        )
        conn.close()
        return True
    except Exception:
        return False

def create_database():
    """
    Crée la base de données si elle n'existe pas.
    
    Returns:
        bool: True si la base de données existe ou a été créée, False sinon
    """
    try:
        # Se connecter à PostgreSQL
        conn = psycopg2.connect(
            host=DB_CONFIG["host"],
            port=DB_CONFIG["port"],
            user=DB_CONFIG["user"],
            password=DB_CONFIG["password"],
            database="postgres",  # Base de données par défaut
            options="-c client_encoding=UTF8"
        )
        conn.autocommit = True
        cursor = conn.cursor()
        
        # Vérifier si la base de données existe
        cursor.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DB_CONFIG["database"],))
        if cursor.fetchone():
            print(f"✅ Base de données '{DB_CONFIG['database']}' existe déjà")
            conn.close()
            return True
        
        # Créer la base de données
        cursor.execute(sql.SQL("CREATE DATABASE {}").format(
            sql.Identifier(DB_CONFIG["database"])
        ))
        print(f"✅ Base de données '{DB_CONFIG['database']}' créée avec succès")
        conn.close()
        
        # Se connecter à la nouvelle base de données et créer les tables
        conn = psycopg2.connect(**DB_CONFIG)
        conn.autocommit = True
        cursor = conn.cursor()
        
        # Créer la table "predictions"
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id SERIAL PRIMARY KEY,
            prediction_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            passenger_data JSONB,
            prediction_result JSONB
        )
        """)
        
        conn.close()

        # Appeler ensure_tables_exist pour vérifier/créer les tables
        ensure_tables_exist()

        return True
    except Exception as e:
        print(f"❌ Erreur lors de la création de la base de données: {e}")
        return False

def run_data_processing():
    """
    Lance le script de traitement des données.
    """
    print("🔄 Exécution du traitement des données...")
    proc = subprocess.run([
        sys.executable, "scripts/processing/proc.py"
    ])
    if proc.returncode == 0:
        print("✅ Traitement des données terminé avec succès")
    else:
        print("❌ Erreur lors du traitement des données")

def run_model_training():
    """
    Lance le script d'entraînement du modèle.
    """
    print("🧠 Entraînement du modèle...")
    proc = subprocess.run([
        sys.executable, "scripts/training/train.py"
    ])
    if proc.returncode == 0:
        print("✅ Entraînement du modèle terminé avec succès")
    else:
        print("❌ Erreur lors de l'entraînement du modèle")

def run_api():
    """
    Lance l'API FastAPI dans un thread séparé.
    """
    def target():
        print("🚀 Démarrage de l'API FastAPI...")
        api_process = subprocess.Popen([
            sys.executable, "run_api.py"
        ])
        processes.append(api_process)
        api_process.wait()
    
    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    return thread

def run_streamlit():
    """
    Lance l'application Streamlit dans un thread séparé.
    """
    def target():
        print("🚀 Démarrage de l'application Streamlit...")
        streamlit_process = subprocess.Popen([
            sys.executable, "run_streamlit.py"
        ])
        processes.append(streamlit_process)
        streamlit_process.wait()
    
    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    return thread

def signal_handler(sig, frame):
    """
    Gère l'arrêt propre des processus lors de l'interruption (Ctrl+C).
    """
    print("\n🛑 Arrêt des services...")
    stop_event.set()
    
    for process in processes:
        if process.poll() is None:  # Si le processus est toujours en cours d'exécution
            try:
                process.terminate()
                time.sleep(0.5)
                if process.poll() is None:
                    process.kill()
            except:
                pass
    
    print("👋 Au revoir!")
    sys.exit(0)

def check_requirements():
    """
    Vérifie que les dépendances requises sont installées.
    """
    try:
        import fastapi
        import uvicorn
        import streamlit
        import pandas
        import sklearn
        import joblib
        import matplotlib
        import seaborn
        import requests
        return True
    except ImportError as e:
        print(f"❌ Erreur: Dépendance manquante: {e}")
        print("💡 Veuillez installer les dépendances requises avec: pip install -r requirements.txt")
        return False

def main():
    """
    Fonction principale.
    """
    # En-tête
    print("=" * 60)
    print("       TITANIC PREDICTION - DÉMARRAGE COMPLET       ")
    print("=" * 60)
    
    # Enregistrer le gestionnaire de signaux pour gérer Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Vérifier les dépendances
    if not check_requirements():
        return
    
    # Vérifier si PostgreSQL est accessible
    print("🔍 Vérification de PostgreSQL...")
    if not check_postgres():
        print("❌ PostgreSQL n'est pas accessible. Veuillez vérifier que PostgreSQL est installé et en cours d'exécution.")
        print("💡 Sur Windows, vous pouvez télécharger PostgreSQL depuis: https://www.postgresql.org/download/windows/")
        return
    
    # Créer la base de données et les tables
    if not create_database():
        return
    
    # Exécuter le traitement des données
    run_data_processing()
    
    # Exécuter l'entraînement du modèle
    run_model_training()
    
    # Démarrer l'API
    api_thread = run_api()
    
    # Attendre que l'API soit prête
    print("⏳ Attente du démarrage de l'API...")
    time.sleep(5)
    
    # Démarrer Streamlit
    streamlit_thread = run_streamlit()
    
    # Afficher les URLs d'accès
    print("\n" + "=" * 60)
    print("🌐 Application prête! Accès aux services:")
    print(f"📊 Application Streamlit: http://localhost:8501")
    print(f"🔌 API FastAPI: http://localhost:8000")
    print(f"📚 Documentation de l'API: http://localhost:8000/docs")
    print("=" * 60)
    print("Appuyez sur Ctrl+C pour arrêter tous les services.")
    
    # Maintenir le script en cours d'exécution jusqu'à interruption
    try:
        while not stop_event.is_set():
            time.sleep(1)
    except KeyboardInterrupt:
        signal_handler(signal.SIGINT, None)

if __name__ == "__main__":
    main() 