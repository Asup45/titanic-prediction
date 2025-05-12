#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Application Streamlit pour prédire la survie sur le Titanic.
Cette application permet aux utilisateurs de saisir les caractéristiques d'un passager
et d'obtenir une prédiction sur sa probabilité de survie.
"""

import os
import json
import requests
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Chemin vers le répertoire racine du projet
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Prédicteur de Survie - Titanic",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constantes
API_URL = "http://localhost:8000"  # URL de l'API FastAPI

# Fonction pour prédire via l'API
def predict_survival(passenger_data):
    """
    Effectue une prédiction via l'API.
    
    Args:
        passenger_data (dict): Données du passager
        
    Returns:
        dict: Résultat de la prédiction ou None en cas d'échec
    """
    try:
        response = requests.post(f"{API_URL}/predict", json=passenger_data)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Erreur lors de la prédiction: {response.text}")
            return None
    except Exception as e:
        st.error(f"Erreur de connexion à l'API: {e}")
        return None

# Fonction pour vérifier l'état de l'API
def check_api_health():
    """
    Vérifie si l'API est accessible et en bon état.
    
    Returns:
        bool: True si l'API est accessible, False sinon
    """
    try:
        response = requests.get(f"{API_URL}/health")
        if response.status_code == 200:
            health_data = response.json()
            if health_data.get("status") == "healthy":
                return True
        return False
    except:
        return False

# Fonction pour obtenir les dernières prédictions depuis l'API
def get_recent_predictions(limit=5):
    """
    Récupère les prédictions récentes depuis l'API.
    
    Args:
        limit (int): Nombre de prédictions à récupérer
        
    Returns:
        list: Liste des prédictions ou liste vide en cas d'échec
    """
    try:
        response = requests.get(f"{API_URL}/predictions?limit={limit}")
        if response.status_code == 200:
            return response.json().get("results", [])
        return []
    except:
        return []

# Fonction pour charger et afficher des statistiques sur le jeu de données Titanic
def load_titanic_statistics():
    """
    Charge le jeu de données Titanic et affiche des statistiques.
    """
    try:
        # Charger le jeu de données Titanic
        data_path = os.path.join(ROOT_DIR, "data", "Titanic-Dataset.csv")
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
            
            # Afficher quelques statistiques
            st.subheader("Statistiques sur le naufrage du Titanic")
            
            # Calculer les taux de survie
            survival_rate = df["Survecu"].mean() * 100
            
            # Créer deux colonnes pour les métriques
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Taux de survie global", f"{survival_rate:.1f}%")
            
            with col2:
                male_survival = df[df["Sexe"] == "male"]["Survecu"].mean() * 100
                st.metric("Taux de survie (hommes)", f"{male_survival:.1f}%")
            
            with col3:
                female_survival = df[df["Sexe"] == "female"]["Survecu"].mean() * 100
                st.metric("Taux de survie (femmes)", f"{female_survival:.1f}%")
            
            # Afficher des graphiques
            st.subheader("Visualisation des données")
            
            tab1, tab2, tab3 = st.tabs(["Survie par classe", "Survie par âge", "Survie par sexe"])
            
            with tab1:
                # Survie par classe
                fig, ax = plt.subplots(figsize=(10, 6))
                survival_by_class = pd.crosstab(df["Pclass"], df["Survecu"])
                survival_by_class_pct = survival_by_class.div(survival_by_class.sum(axis=1), axis=0) * 100
                survival_by_class_pct.plot(kind="bar", stacked=True, ax=ax)
                ax.set_ylabel("Pourcentage")
                ax.set_xlabel("Classe")
                ax.set_xticklabels(["1ère classe", "2ème classe", "3ème classe"])
                ax.set_title("Taux de survie par classe")
                ax.legend(["Non survécu", "Survécu"])
                st.pyplot(fig)
            
            with tab2:
                # Survie par âge
                fig, ax = plt.subplots(figsize=(10, 6))
                df_age = df[df["Age"].notna()]
                sns.histplot(data=df_age, x="Age", hue="Survecu", multiple="dodge", bins=20, ax=ax)
                ax.set_title("Distribution des âges par statut de survie")
                ax.set_xlabel("Âge")
                ax.set_ylabel("Nombre de passagers")
                ax.legend(["Non survécu", "Survécu"])
                st.pyplot(fig)
            
            with tab3:
                # Survie par sexe
                fig, ax = plt.subplots(figsize=(10, 6))
                survival_by_sex = pd.crosstab(df["Sexe"], df["Survecu"])
                survival_by_sex_pct = survival_by_sex.div(survival_by_sex.sum(axis=1), axis=0) * 100
                survival_by_sex_pct.plot(kind="bar", stacked=True, ax=ax)
                ax.set_ylabel("Pourcentage")
                ax.set_xlabel("Sexe")
                ax.set_xticklabels(["Femme", "Homme"])
                ax.set_title("Taux de survie par sexe")
                ax.legend(["Non survécu", "Survécu"])
                st.pyplot(fig)
        else:
            st.error(f"Fichier de données introuvable: {data_path}")
            st.info("Veuillez placer le fichier Titanic-Dataset.csv dans le dossier 'data/'.")
        
    except Exception as e:
        st.error(f"Erreur lors du chargement des statistiques: {e}")

def load_model_metrics():
    """
    Charge les métriques du modèle depuis le fichier metrics.json.
    
    Returns:
        dict: Dictionnaire contenant les métriques ou None en cas d'erreur.
    """
    metrics_path = os.path.join(ROOT_DIR, "model", "metrics.json")
    try:
        if os.path.exists(metrics_path):
            with open(metrics_path, "r") as f:
                return json.load(f)
        else:
            st.error(f"Fichier metrics.json introuvable: {metrics_path}")
            return None
    except Exception as e:
        st.error(f"Erreur lors du chargement des métriques: {e}")
        return None

# Interface utilisateur
def main():
    # Barre latérale
    with st.sidebar:
        st.title("🚢 Titanic - Prédicteur de Survie")
        st.info("Cette application vous permet de prédire si vous auriez survécu au naufrage du Titanic en fonction de vos caractéristiques.")
        
        # Vérifier l'état de l'API
        api_status = check_api_health()
        if api_status:
            st.success("✅ API connectée")
        else:
            st.error("❌ API non disponible")
            st.warning("Veuillez démarrer l'API avec `python run_api.py`")
        
        st.divider()
        
        # Navigation
        page = st.radio("Navigation", ["Prédicteur", "Statistiques", "Monitoring", "À propos"])
    
    # Page principale
    if page == "Prédicteur":
        st.title("Prédicteur de Survie sur le Titanic")
        st.write("Entrez vos informations pour prédire si vous auriez survécu au naufrage du Titanic.")
        
        # Formulaire pour les caractéristiques du passager
        with st.form("passenger_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                nom = st.text_input("Nom", "")
                sexe = st.selectbox("Sexe", ["male", "female"])
                age = st.number_input("Âge", min_value=0.0, max_value=100.0, value=30.0, step=1.0)
                pclass = st.selectbox("Classe", [1, 2, 3], format_func=lambda x: f"{x}ème classe")
            
            with col2:
                sibsp = st.number_input("Nombre de frères/sœurs/conjoints à bord", min_value=0, max_value=10, value=0)
                parch = st.number_input("Nombre de parents/enfants à bord", min_value=0, max_value=10, value=0)
                tarif = st.number_input("Prix du billet (£)", min_value=0.0, max_value=500.0, value=30.0, step=1.0)
                embarquement = st.selectbox("Port d'embarquement", ["S", "C", "Q"], 
                                           format_func=lambda x: {"S": "Southampton", "C": "Cherbourg", "Q": "Queenstown"}[x])
            
            submit_button = st.form_submit_button("Prédire")
        
        # Traitement de la soumission
        if submit_button:
            # Préparer les données pour l'API
            passenger_data = {
                "nom": nom,
                "sexe": sexe,
                "age": float(age),
                "pclass": int(pclass),
                "sibsp": int(sibsp),
                "parch": int(parch),
                "tarif": float(tarif),
                "embarquement": embarquement
            }
            
            # Afficher un spinner pendant la prédiction
            with st.spinner("Prédiction en cours..."):
                prediction = predict_survival(passenger_data)
            
            # Afficher le résultat
            if prediction:
                st.divider()
                
                # Créer un conteneur stylisé pour le résultat
                result_container = st.container()
                survie = prediction["survie"]
                confidence = prediction["confidence"]
                
                with result_container:
                    if survie == 1:
                        st.success(f"### 🎉 Vous auriez probablement SURVÉCU au naufrage du Titanic!")
                    else:
                        st.error(f"### ☠️ Vous n'auriez probablement PAS survécu au naufrage du Titanic.")
                    
                    st.progress(confidence)
                    st.write(f"Confiance de la prédiction: {confidence*100:.1f}%")
                    
                    # Afficher des informations supplémentaires
                    st.write("#### Facteurs influençant la survie:")
                    
                    if sexe == "female":
                        st.write("- ✅ Être une femme augmentait considérablement les chances de survie (priorité aux femmes et enfants)")
                    else:
                        st.write("- ❌ Être un homme réduisait considérablement les chances de survie")
                    
                    if pclass == 1:
                        st.write("- ✅ Voyager en 1ère classe augmentait les chances de survie")
                    elif pclass == 3:
                        st.write("- ❌ Voyager en 3ème classe réduisait les chances de survie")
                    
                    if age < 10:
                        st.write("- ✅ Être un enfant augmentait les chances de survie")
                    
                    if sibsp + parch > 4:
                        st.write("- ❌ Voyager avec une grande famille réduisait les chances de survie")
        
        # Afficher les prédictions récentes
        st.divider()
        st.subheader("Prédictions récentes")
        
        recent_predictions = get_recent_predictions()
        if recent_predictions:
            for pred in recent_predictions:
                result = pred["prediction_result"]
                col1, col2, col3 = st.columns([1, 3, 1])
                
                with col1:
                    if result.get("survie") == 1:
                        st.success("✅ Survécu")
                    else:
                        st.error("❌ Non survécu")
                
                with col2:
                    passager = result.get("passager", {})
                    st.write(f"**{passager.get('nom', 'Passager inconnu')}** - {passager.get('sexe', '')} - {passager.get('age', '')} ans - {passager.get('classe', '')}ème classe")
                
                with col3:
                    st.write(f"Confiance: {pred['confidence']*100:.1f}%")
                
                st.divider()
        else:
            st.info("Aucune prédiction récente disponible.")
    
    elif page == "Statistiques":
        st.title("Statistiques sur le Titanic")
        load_titanic_statistics()

    elif page == "Monitoring":
        st.title("Monitoring du Modèle Titanic")
        
        # Charger les métriques du modèle
        model_metrics = load_model_metrics()
        if model_metrics:
            st.subheader("Métriques du modèle (Entraînement)")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{model_metrics['accuracy']:.2f}")
            col2.metric("Precision", f"{model_metrics['precision']:.2f}")
            col3.metric("Recall", f"{model_metrics['recall']:.2f}")
            col4.metric("F1 Score", f"{model_metrics['f1']:.2f}")
        else:
            st.warning("Impossible de charger les métriques du modèle.")
        
        # Récupérer les métriques depuis l'API
        st.subheader("Métriques en temps réel (Prédictions)")
        response = requests.get(f"{API_URL}/metrics")
        if response.status_code == 200:
            metrics = response.json()
            st.metric("Confiance moyenne", metrics["avg_confidence"])
            st.bar_chart(metrics["class_distribution"])
        else:
            st.error("Erreur lors de la récupération des métriques depuis l'API.")
    
    else:  # Page "À propos"
        st.title("À propos")
        st.write("""
        ## Le naufrage du Titanic
        
        Le RMS Titanic était un paquebot transatlantique britannique qui a fait naufrage dans l'océan Atlantique Nord en 1912
        après avoir heurté un iceberg lors de son voyage inaugural de Southampton à New York City.
        
        Sur les 2224 passagers et membres d'équipage, plus de 1500 ont péri, ce qui fait du naufrage l'une des catastrophes 
        maritimes les plus meurtrières en temps de paix.
        
        ## Ce projet
        
        Cette application utilise l'apprentissage automatique pour prédire si une personne aurait survécu au naufrage du Titanic
        en fonction de caractéristiques comme l'âge, le sexe, la classe et d'autres facteurs.
        
        Le modèle a été entraîné sur les données historiques réelles des passagers du Titanic.
        
        ## Architecture
        
        L'application est construite avec:
        - **Streamlit**: Interface utilisateur
        - **FastAPI**: API REST
        - **Scikit-learn**: Algorithmes d'apprentissage automatique
        - **PostgreSQL**: Base de données
        """)

# Lancement de l'application
if __name__ == "__main__":
    main() 