#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script pour démarrer l'application Streamlit localement.
Ce script lance l'application Streamlit sur le port 8501.
"""

import os
import subprocess
import sys

def run_streamlit():
    """
    Lance l'application Streamlit.
    """
    print("🚀 Démarrage de l'application Streamlit...")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", "streamlit_app.py",
        "--server.port", "8501"
    ])

def check_requirements():
    """
    Vérifie que les dépendances requises sont installées.
    """
    try:
        import streamlit
        import pandas
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
    print("=" * 50)
    print("   TITANIC PREDICTION - APPLICATION STREAMLIT   ")
    print("=" * 50)
    
    # Vérifier les dépendances
    if not check_requirements():
        return
    
    # Vérifier si le fichier de données existe
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "data", "Titanic-Dataset.csv"))
    if not os.path.exists(data_path):
        print(f"⚠️ Attention: Fichier de données non trouvé: {data_path}")
        print("💡 Certaines fonctionnalités de l'application pourraient ne pas fonctionner correctement.")
    
    # Démarrer Streamlit
    try:
        run_streamlit()
    except KeyboardInterrupt:
        print("\n👋 Arrêt de l'application Streamlit...")
    except Exception as e:
        print(f"❌ Erreur lors du démarrage de Streamlit: {e}")

if __name__ == "__main__":
    main() 