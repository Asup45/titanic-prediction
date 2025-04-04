-- Création de la base de données
CREATE DATABASE pipeline_db;

-- Connexion à la base de données
\c pipeline_db;

-- Table pour les données brutes (RAW)
CREATE TABLE IF NOT EXISTS raw_data (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    source VARCHAR(255),
    data JSONB,
    processed BOOLEAN DEFAULT FALSE
);

-- Table pour les données traitées (PROC)
CREATE TABLE IF NOT EXISTS processed_data (
    id SERIAL PRIMARY KEY,
    raw_id INTEGER REFERENCES raw_data(id),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed_data JSONB,
    features JSONB,
    predicted BOOLEAN DEFAULT FALSE
);

-- Table pour les résultats de prédiction (PRED)
CREATE TABLE IF NOT EXISTS prediction_results (
    id SERIAL PRIMARY KEY,
    processed_id INTEGER REFERENCES processed_data(id),
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    prediction_result JSONB,
    confidence FLOAT,
    model_version VARCHAR(50)
);

-- Index pour améliorer les performances
CREATE INDEX idx_raw_processed ON raw_data(processed);
CREATE INDEX idx_processed_predicted ON processed_data(predicted);
CREATE INDEX idx_raw_id ON processed_data(raw_id);
CREATE INDEX idx_processed_id ON prediction_results(processed_id); 
