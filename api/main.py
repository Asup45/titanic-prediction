#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API FastAPI pour exposer les résultats de prédiction du modèle Titanic.
Cette API permet d'accéder aux prédictions stockées dans la base de données
et de faire des prédictions directes sur de nouvelles données.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

import psycopg2
import joblib
import numpy as np
from psycopg2.extras import RealDictCursor
from fastapi import FastAPI, HTTPException, Query, Depends, Body
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("api")

# Initialisation de l'application FastAPI
app = FastAPI(
    title="API Prédiction Titanic",
    description="API pour prédire la survie sur le Titanic et accéder aux résultats de prédiction",
    version="1.0.0"
)

# Configuration CORS pour permettre les requêtes cross-origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paramètres de connexion à la base de données
DB_CONFIG = {
    "host": "localhost",
    "database": "pipeline_db",
    "user": "postgres",
    "password": "postgres",
    "port": 5432
}

# Chemin vers le répertoire racine du projet
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Liste des caractéristiques utilisées dans le modèle
FEATURES = [
    'pclass', 'sexe_male', 'age', 'sibsp', 'parch', 'tarif',
    'embarq_c', 'embarq_q', 'embarq_s', 'taille_famille', 'seul'
]

# Modèles de données Pydantic
class PredictionResult(BaseModel):
    id: int
    processed_id: int
    prediction_result: Dict[str, Any]
    confidence: float
    model_version: str
    timestamp: datetime

class PredictionResponse(BaseModel):
    results: List[PredictionResult]
    count: int

class PassengerInput(BaseModel):
    pclass: int = Field(..., description="Classe du passager (1, 2 ou 3)", ge=1, le=3)
    sexe: str = Field(..., description="Sexe du passager (male ou female)")
    age: float = Field(..., description="Âge du passager", ge=0, le=100)
    sibsp: int = Field(..., description="Nombre de frères/sœurs/conjoints à bord", ge=0)
    parch: int = Field(..., description="Nombre de parents/enfants à bord", ge=0)
    tarif: float = Field(..., description="Prix du billet", ge=0)
    embarquement: str = Field(..., description="Port d'embarquement (C=Cherbourg, Q=Queenstown, S=Southampton)")
    nom: Optional[str] = Field(None, description="Nom du passager")

class PredictionOutput(BaseModel):
    survie: int
    interpretation: str
    confidence: float
    passager: Dict[str, Any]

# Fonction utilitaire pour la connexion à la base de données
def get_db_connection():
    """
    Établit une connexion à la base de données.
    
    Returns:
        Connection: Connexion à la base de données PostgreSQL
    """
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        logger.error(f"Erreur de connexion à la base de données: {e}")
        raise HTTPException(status_code=500, detail="Erreur de connexion à la base de données")

def load_model():
    """
    Charge le modèle de prédiction pour le Titanic.
    
    Returns:
        Any: Modèle de prédiction chargé
    """
    model_path = os.path.join(ROOT_DIR, "model", "titanic_model.joblib")
    
    if not os.path.exists(model_path):
        logger.error(f"Modèle introuvable: {model_path}")
        raise HTTPException(status_code=500, detail="Modèle de prédiction non disponible")
    
    return joblib.load(model_path)

# Routes API
@app.get("/", tags=["Statut"])
def read_root():
    """
    Route racine pour vérifier que l'API est en ligne.
    
    Returns:
        dict: Statut de l'API
    """
    return {"status": "online", "message": "API de prédiction Titanic opérationnelle"}

@app.get("/predictions", response_model=PredictionResponse, tags=["Prédictions"])
def get_predictions(
    limit: int = Query(10, ge=1, le=100, description="Nombre maximum de résultats à retourner"),
    offset: int = Query(0, ge=0, description="Nombre de résultats à sauter"),
    min_confidence: float = Query(0.0, ge=0.0, le=1.0, description="Confiance minimale des prédictions"),
    conn: psycopg2.extensions.connection = Depends(get_db_connection)
):
    """
    Récupère les résultats de prédiction.
    
    Args:
        limit (int): Nombre maximum de résultats à retourner
        offset (int): Nombre de résultats à sauter
        min_confidence (float): Confiance minimale des prédictions
        conn (Connection): Connexion à la base de données
        
    Returns:
        PredictionResponse: Liste des résultats de prédiction
    """
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Requête pour compter le nombre total de résultats
            cur.execute("""
                SELECT COUNT(*) as count
                FROM prediction_results
                WHERE confidence >= %s
            """, (min_confidence,))
            
            total_count = cur.fetchone()['count']
            
            # Requête pour récupérer les résultats
            cur.execute("""
                SELECT id, processed_id, prediction_result, confidence, model_version, timestamp
                FROM prediction_results
                WHERE confidence >= %s
                ORDER BY timestamp DESC
                LIMIT %s OFFSET %s
            """, (min_confidence, limit, offset))
            
            results = cur.fetchall()
            
            # Conversion des résultats en objets Pydantic
            predictions = []
            for result in results:
                # Conversion du JSON string en objet Python si nécessaire
                if isinstance(result['prediction_result'], str):
                    result['prediction_result'] = json.loads(result['prediction_result'])
                    
                predictions.append(PredictionResult(**result))
            
            return PredictionResponse(
                results=predictions,
                count=total_count
            )
            
    except Exception as e:
        logger.error(f"Erreur lors de la récupération des prédictions: {e}")
        raise HTTPException(
            status_code=500, 
            detail="Erreur lors de la récupération des prédictions"
        )
    finally:
        conn.close()

@app.get("/prediction/{prediction_id}", response_model=PredictionResult, tags=["Prédictions"])
def get_prediction_by_id(
    prediction_id: int,
    conn: psycopg2.extensions.connection = Depends(get_db_connection)
):
    """
    Récupère un résultat de prédiction spécifique par son ID.
    
    Args:
        prediction_id (int): ID de la prédiction à récupérer
        conn (Connection): Connexion à la base de données
        
    Returns:
        PredictionResult: Résultat de la prédiction
    """
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT id, processed_id, prediction_result, confidence, model_version, timestamp
                FROM prediction_results
                WHERE id = %s
            """, (prediction_id,))
            
            result = cur.fetchone()
            
            if not result:
                raise HTTPException(status_code=404, detail="Prédiction non trouvée")
            
            # Conversion du JSON string en objet Python si nécessaire
            if isinstance(result['prediction_result'], str):
                result['prediction_result'] = json.loads(result['prediction_result'])
                
            return PredictionResult(**result)
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Erreur lors de la récupération de la prédiction {prediction_id}: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Erreur lors de la récupération de la prédiction {prediction_id}"
        )
    finally:
        conn.close()

@app.post("/predict", response_model=PredictionOutput, tags=["Prédiction Directe"])
def predict_survival(
    passenger: PassengerInput = Body(...)
):
    """
    Effectue une prédiction de survie pour un nouveau passager.
    
    Args:
        passenger (PassengerInput): Données du passager
        
    Returns:
        PredictionOutput: Résultat de la prédiction
    """
    try:
        # Charger le modèle
        model = load_model()
        
        # Préparer les caractéristiques
        taille_famille = passenger.sibsp + passenger.parch + 1
        seul = 1 if taille_famille == 1 else 0
        
        # Encodage des caractéristiques
        features = {
            'pclass': passenger.pclass,
            'sexe_male': 1 if passenger.sexe.lower() == 'male' else 0,
            'age': passenger.age,
            'sibsp': passenger.sibsp,
            'parch': passenger.parch,
            'tarif': passenger.tarif,
            'embarq_c': 1 if passenger.embarquement.upper() == 'C' else 0,
            'embarq_q': 1 if passenger.embarquement.upper() == 'Q' else 0,
            'embarq_s': 1 if passenger.embarquement.upper() == 'S' else 0,
            'taille_famille': taille_famille,
            'seul': seul
        }
        
        # Créer un tableau de caractéristiques ordonné
        X = np.array([[features.get(feat, 0) for feat in FEATURES]])
        
        # Faire la prédiction
        prediction = int(model.predict(X)[0])
        proba = model.predict_proba(X)[0]
        confidence = float(proba[prediction])
        
        # Préparer la réponse
        return PredictionOutput(
            survie=prediction,
            interpretation="Survécu" if prediction == 1 else "Non survécu",
            confidence=confidence,
            passager={
                'nom': passenger.nom or "Passager inconnu",
                'classe': passenger.pclass,
                'sexe': passenger.sexe,
                'age': passenger.age,
                'port_embarquement': passenger.embarquement
            }
        )
        
    except Exception as e:
        logger.error(f"Erreur lors de la prédiction directe: {e}")
        raise HTTPException(
            status_code=500, 
            detail="Erreur lors de la prédiction"
        )

@app.get("/health", tags=["Statut"])
def health_check():
    """
    Vérifie l'état de santé de l'API et la connexion à la base de données.
    
    Returns:
        dict: Statut de santé de l'API
    """
    try:
        # Vérification de la connexion à la base de données
        conn = get_db_connection()
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
            cur.fetchone()
        conn.close()
        
        # Vérification de l'accès au modèle
        model_path = os.path.join(ROOT_DIR, "model", "titanic_model.joblib")
        model_status = "available" if os.path.exists(model_path) else "unavailable"
        
        return {
            "status": "healthy",
            "database": "connected",
            "model": model_status,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Échec du contrôle de santé: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "unhealthy",
                "database": "disconnected",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 