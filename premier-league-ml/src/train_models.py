"""
Script para entrenar modelos ML con datos actualizados.
Entrena múltiples modelos y compara desempeño.
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import json
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).parent.parent
PROCESSED_PATH = SCRIPT_DIR / "data" / "processed"
MODELS_PATH = SCRIPT_DIR / "models"
MODELS_PATH.mkdir(exist_ok=True)

def load_training_data():
    """Carga datos de entrenamiento procesados."""
    data_file = PROCESSED_PATH / "training_data_latest.pkl"
    
    if not data_file.exists():
        raise FileNotFoundError(f"No se encontró {data_file}. Ejecuta run_feature_engineering.py primero.")
    
    with open(str(data_file), 'rb') as f:
        data = pickle.load(f)
    
    return data

def train_models(data):
    """Entrena múltiples modelos."""
    print("🤖 Entrenando modelos...\n")
    
    X_train = data['X_train']
    X_test = data['X_test']
    y_result_train = data['y_result_train']
    y_result_test = data['y_result_test']
    
    # Normalizar features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"  → Entrenando {model_name}...")
        model.fit(X_train_scaled, y_result_train)
        
        # Predicciones
        y_pred = model.predict(X_test_scaled)
        
        # Métricas
        metrics = {
            'accuracy': accuracy_score(y_result_test, y_pred),
            'precision': precision_score(y_result_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_result_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_result_test, y_pred, average='weighted', zero_division=0),
        }
        
        results[model_name] = {
            'model': model,
            'scaler': scaler,
            'metrics': metrics,
            'y_pred': y_pred,
        }
        
        print(f"     ✅ Accuracy: {metrics['accuracy']:.4f}")
    
    return results, scaler

def evaluate_models(results, data):
    """Genera reporte de evaluación."""
    print("\n📊 EVALUACIÓN DE MODELOS\n")
    print("=" * 60)
    
    y_result_test = data['y_result_test']
    
    metrics_df = pd.DataFrame()
    
    for model_name, result in results.items():
        metrics = result['metrics']
        metrics_df = pd.concat([
            metrics_df,
            pd.DataFrame({
                'Model': [model_name],
                'Accuracy': [metrics['accuracy']],
                'Precision': [metrics['precision']],
                'Recall': [metrics['recall']],
                'F1-Score': [metrics['f1']],
            })
        ], ignore_index=True)
        
        print(f"\n{model_name}:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1']:.4f}")
    
    print("\n" + "=" * 60)
    
    # Mejor modelo
    best_model_name = metrics_df.loc[metrics_df['Accuracy'].idxmax(), 'Model']
    print(f"\n🏆 Mejor modelo: {best_model_name}")
    
    return best_model_name, results[best_model_name]

def save_models(results, best_model_name, best_result, scaler):
    """Guarda modelos entrenados."""
    print("\n💾 Guardando modelos...\n")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Guardar modelo principal
    model_file = MODELS_PATH / f"best_model_{timestamp}.pkl"
    with open(str(model_file), 'wb') as f:
        pickle.dump({
            'model': best_result['model'],
            'scaler': scaler,
            'name': best_model_name,
            'metrics': best_result['metrics'],
            'created_at': timestamp,
        }, f)
    print(f"  ✅ {model_file.name}")
    
    # Guardar también como latest
    latest_file = MODELS_PATH / "best_model_latest.pkl"
    with open(str(latest_file), 'wb') as f:
        pickle.dump({
            'model': best_result['model'],
            'scaler': scaler,
            'name': best_model_name,
            'metrics': best_result['metrics'],
            'created_at': timestamp,
        }, f)
    print(f"  ✅ best_model_latest.pkl")
    
    # Guardar métricas en JSON
    metrics_file = MODELS_PATH / f"model_metrics_{timestamp}.json"
    all_metrics = {}
    for name, result in results.items():
        all_metrics[name] = result['metrics']
    
    with open(str(metrics_file), 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'best_model': best_model_name,
            'metrics': all_metrics,
        }, f, indent=2)
    print(f"  ✅ model_metrics_{timestamp}.json")
    
    return timestamp

def main():
    print("=" * 60)
    print("🔄 ENTRENAMIENTO DE MODELOS - Datos Actualizados")
    print("=" * 60)
    print()
    
    # Cargar datos
    print("📥 Cargando datos procesados...")
    data = load_training_data()
    print(f"   Train: {len(data['X_train'])} muestras")
    print(f"   Test: {len(data['X_test'])} muestras")
    print()
    
    # Entrenar
    results, scaler = train_models(data)
    
    # Evaluar
    best_model_name, best_result = evaluate_models(results, data)
    
    # Guardar
    timestamp = save_models(results, best_model_name, best_result, scaler)
    
    print("\n" + "=" * 60)
    print("✅ Entrenamiento completado!")
    print("=" * 60)
    print(f"\n📌 Modelos guardados con timestamp: {timestamp}")

if __name__ == "__main__":
    main()
