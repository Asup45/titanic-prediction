#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script pour démarrer l'API FastAPI localement.
Ce script lance l'API FastAPI avec Uvicorn sur le port 8000.
"""

import os
import subprocess
import sys
import time
import threading

def run_uvicorn():
    """
    Lance l'API FastAPI avec Uvicorn.
    """
    print("🚀 Démarrage de l'API FastAPI...")
    subprocess.run([
        sys.executable, "-m", "uvicorn", "api.main:app", 
        "--host", "0.0.0.0", "--port", "8000", "--reload"
    ])

def check_requirements():
    """
    Vérifie que les dépendances requises sont installées.
    """
    try:
        import fastapi
        import uvicorn
        import joblib
        import pandas
        import sklearn
        import psycopg2
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
    print("=" * 50)
    print("      TITANIC PREDICTION API - DÉMARRAGE LOCAL      ")
    print("=" * 50)
    
    # Vérifier les dépendances
    if not check_requirements():
        return
    
    # Vérifier si le modèle existe
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "model", "titanic_model.joblib"))
    if not os.path.exists(model_path):
        print(f"❌ Erreur: Modèle non trouvé: {model_path}")
        print("💡 Veuillez exécuter le script d'entraînement avec: python scripts/training/train.py")
        return
    
    # Démarrer l'API
    try:
        run_uvicorn()
    except KeyboardInterrupt:
        print("\n👋 Arrêt de l'API...")
    except Exception as e:
        print(f"❌ Erreur lors du démarrage de l'API: {e}")

if __name__ == "__main__":
    main() 