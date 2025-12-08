#!/usr/bin/env python3
"""
DIAGNOSIS: Analisis de errores y debilidades en los modelos
Identifica qué predicciones falla y por qué
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
import pickle
import warnings
warnings.filterwarnings('ignore')

# Importar la función de features del script de reentrenamiento
import importlib.util
spec = importlib.util.spec_from_file_location("retrain", Path(__file__).parent / "retrain_models_improved.py")
retrain = importlib.util.module_from_spec(spec)
spec.loader.exec_module(retrain)

SCRIPT_DIR = Path(__file__).parent
DATA_PATH = SCRIPT_DIR / "data" / "raw" / "epl_final.csv"
MODELS_PATH = SCRIPT_DIR / "models"

def load_models_and_scaler():
    """Carga los modelos entrenados"""
    try:
        with open(MODELS_PATH / 'gb_result_model.pkl', 'rb') as f:
            gb_result = pickle.load(f)
        with open(MODELS_PATH / 'gb_goals_model.pkl', 'rb') as f:
            gb_goals = pickle.load(f)
        with open(MODELS_PATH / 'gb_btts_model.pkl', 'rb') as f:
            gb_btts = pickle.load(f)
        with open(MODELS_PATH / 'scaler_model.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return gb_result, gb_goals, gb_btts, scaler
    except Exception as e:
        print(f"[ERROR] No se encontraron modelos: {e}")
        sys.exit(1)

def diagnose_result_model(df, X_test, y_result_test, gb_result, scaler, test_idx):
    """Diagnostico del modelo de Resultado 1X2"""
    print("\n" + "="*70)
    print("[DIAGNOSIS] MODELO RESULTADO 1X2")
    print("="*70)
    
    X_test_scaled = scaler.transform(X_test)
    y_pred = gb_result.predict(X_test_scaled)
    y_pred_proba = gb_result.predict_proba(X_test_scaled)
    
    # Mapeo
    result_map_inv = {0: 'Away', 1: 'Draw', 2: 'Home'}
    y_test_labels = [result_map_inv[y] for y in y_result_test]
    y_pred_labels = [result_map_inv[y] for y in y_pred]
    
    # Accuracy global
    accuracy = (y_pred == y_result_test).sum() / len(y_result_test)
    print(f"\nAccuracy Global: {accuracy:.2%}")
    
    # Por clase
    print(f"\nAccuracy por Clase:")
    for outcome in ['Home', 'Draw', 'Away']:
        outcome_num = {v: k for k, v in result_map_inv.items()}[outcome]
        mask = y_result_test == outcome_num
        if mask.sum() > 0:
            class_acc = (y_pred[mask] == y_result_test[mask]).sum() / mask.sum()
            count = mask.sum()
            print(f"   {outcome:6s}: {class_acc:6.2%} ({count:4d} partidos)")
    
    # Matriz de confusión legible
    print(f"\nMatriz de Confusion:")
    print(f"       Predicción")
    print(f"        Home  Draw  Away")
    
    for true_outcome in ['Home', 'Draw', 'Away']:
        true_num = {v: k for k, v in result_map_inv.items()}[true_outcome]
        mask = y_result_test == true_num
        
        home_pred = (y_pred[mask] == 2).sum()
        draw_pred = (y_pred[mask] == 1).sum()
        away_pred = (y_pred[mask] == 0).sum()
        
        print(f"Real {true_outcome:4s}: {home_pred:4d}  {draw_pred:4d}  {away_pred:4d}")
    
    # Confianza de predicciones correctas vs incorrectas
    print(f"\nConfianza (probabilidad maxima):")
    correct = y_pred == y_result_test
    if correct.sum() > 0:
        correct_confidence = y_pred_proba[correct].max(axis=1)
        print(f"   Predicciones CORRECTAS:   {correct_confidence.mean():.2%} (promedio)")
        print(f"   Rango: {correct_confidence.min():.2%} - {correct_confidence.max():.2%}")
    
    if (~correct).sum() > 0:
        incorrect_confidence = y_pred_proba[~correct].max(axis=1)
        print(f"   Predicciones INCORRECTAS: {incorrect_confidence.mean():.2%} (promedio)")
        print(f"   Rango: {incorrect_confidence.min():.2%} - {incorrect_confidence.max():.2%}")
        
        print(f"\n   [INSIGHT] El modelo es poco confiado en errores")
        print(f"   Diferencia: {(correct_confidence.mean() - incorrect_confidence.mean()):.2%}")
    
    # Partidos más predecibles
    print(f"\nPartidos MAS DIFICILES (baja confianza):")
    low_confidence_idx = y_pred_proba.max(axis=1).argsort()[:5]
    for idx in low_confidence_idx:
        actual_idx = test_idx[idx]
        conf = y_pred_proba[idx].max()
        true_label = result_map_inv[y_result_test.iloc[idx]]
        pred_label = result_map_inv[y_pred[idx]]
        match_info = f"{df.iloc[actual_idx]['HomeTeam']} vs {df.iloc[actual_idx]['AwayTeam']}"
        status = "[CORRECTO]" if y_pred[idx] == y_result_test.iloc[idx] else "[ERROR]"
        print(f"   {match_info:30s} Predicho:{pred_label:5s} Real:{true_label:5s} Conf:{conf:.2%} {status}")

def diagnose_goals_model(df, X_test, y_goals_test, gb_goals, scaler, test_idx):
    """Diagnostico del modelo de Goles"""
    print("\n" + "="*70)
    print("[DIAGNOSIS] MODELO GOLES TOTALES")
    print("="*70)
    
    X_test_scaled = scaler.transform(X_test)
    y_pred_goals = gb_goals.predict(X_test_scaled)
    
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    
    mae = mean_absolute_error(y_goals_test, y_pred_goals)
    rmse = np.sqrt(mean_squared_error(y_goals_test, y_pred_goals))
    
    print(f"\nMAE (Mean Absolute Error): {mae:.3f} goles")
    print(f"RMSE (Root Mean Squared Error): {rmse:.3f} goles")
    
    # Por rango de goles
    print(f"\nError por Rango de Goles Reales:")
    for goal_range in [(0, 1), (2, 3), (4, 10)]:
        mask = (y_goals_test >= goal_range[0]) & (y_goals_test <= goal_range[1])
        if mask.sum() > 0:
            range_mae = mean_absolute_error(y_goals_test[mask], y_pred_goals[mask])
            count = mask.sum()
            print(f"   {goal_range[0]}-{goal_range[1]} goles: MAE {range_mae:.3f} ({count:4d} partidos)")
    
    # Predicciones extremas
    print(f"\nErrores Mayores a 2 goles:")
    large_errors = np.abs(y_goals_test - y_pred_goals) > 2
    if large_errors.sum() > 0:
        print(f"   Cantidad: {large_errors.sum()} partidos ({large_errors.sum()/len(y_goals_test):.2%})")
        print(f"   Ejemplos:")
        
        large_error_idx = np.where(large_errors)[0][:5]
        for idx in large_error_idx:
            actual_idx = test_idx[idx]
            real = y_goals_test.iloc[idx]
            pred = y_pred_goals[idx]
            match_info = f"{df.iloc[actual_idx]['HomeTeam']} vs {df.iloc[actual_idx]['AwayTeam']}"
            print(f"      {match_info:30s} Predicho:{pred:.1f} Real:{real:.0f} Error:{abs(pred-real):.1f}")

def diagnose_btts_model(df, X_test, y_btts_test, gb_btts, scaler, test_idx):
    """Diagnostico del modelo BTTS"""
    print("\n" + "="*70)
    print("[DIAGNOSIS] MODELO BTTS (AMBOS ANOTAN)")
    print("="*70)
    
    X_test_scaled = scaler.transform(X_test)
    y_pred_btts = gb_btts.predict(X_test_scaled)
    y_pred_proba_btts = gb_btts.predict_proba(X_test_scaled)
    
    accuracy = (y_pred_btts == y_btts_test).sum() / len(y_btts_test)
    print(f"\nAccuracy Global: {accuracy:.2%}")
    
    # Por clase
    print(f"\nAccuracy por Clase:")
    for outcome_name, outcome_num in [('NO', 0), ('SI', 1)]:
        mask = y_btts_test == outcome_num
        if mask.sum() > 0:
            class_acc = (y_pred_btts[mask] == y_btts_test[mask]).sum() / mask.sum()
            count = mask.sum()
            ratio = mask.sum() / len(y_btts_test)
            print(f"   {outcome_name:6s}: {class_acc:6.2%} ({count:4d} partidos, {ratio:.1%} del total)")
    
    # Confianza
    print(f"\nConfianza en predicciones:")
    correct = y_pred_btts == y_btts_test
    
    if correct.sum() > 0:
        correct_confidence = y_pred_proba_btts[correct].max(axis=1)
        print(f"   CORRECTAS:   {correct_confidence.mean():.2%}")
    
    if (~correct).sum() > 0:
        incorrect_confidence = y_pred_proba_btts[~correct].max(axis=1)
        print(f"   INCORRECTAS: {incorrect_confidence.mean():.2%}")

def feature_importance_analysis(X_test, gb_result, gb_goals, gb_btts):
    """Analiza importancia de features"""
    print("\n" + "="*70)
    print("[IMPORTANCE] FEATURES MAS IMPORTANTES")
    print("="*70)
    
    feature_names = X_test.columns
    
    # Resultado
    print(f"\nTop 10 Features - Resultado 1X2:")
    importances = gb_result.feature_importances_
    top_idx = np.argsort(importances)[-10:][::-1]
    for i, idx in enumerate(top_idx, 1):
        print(f"   {i:2d}. {feature_names[idx]:30s} {importances[idx]:.4f}")
    
    # Goles
    print(f"\nTop 10 Features - Goles Totales:")
    importances = gb_goals.feature_importances_
    top_idx = np.argsort(importances)[-10:][::-1]
    for i, idx in enumerate(top_idx, 1):
        print(f"   {i:2d}. {feature_names[idx]:30s} {importances[idx]:.4f}")
    
    # BTTS
    print(f"\nTop 10 Features - BTTS:")
    importances = gb_btts.feature_importances_
    top_idx = np.argsort(importances)[-10:][::-1]
    for i, idx in enumerate(top_idx, 1):
        print(f"   {i:2d}. {feature_names[idx]:30s} {importances[idx]:.4f}")

def main():
    print("\n" + "="*70)
    print("[DIAGNOSIS] ANALISIS DE ERRORES Y DEBILIDADES")
    print("="*70)
    
    # Cargar datos
    print(f"\n[LOAD] Cargando datos...")
    df = pd.read_csv(DATA_PATH)
    df['MatchDate'] = pd.to_datetime(df['MatchDate'])
    
    # Crear features
    print(f"[FEATURE] Creando features...")
    X, df_processed = retrain.create_enhanced_features(df)
    
    # Targets
    result_map = {'A': 0, 'D': 1, 'H': 2}
    y_result = df_processed['FullTimeResult'].map(result_map)
    y_goals = df_processed['FullTimeHomeGoals'] + df_processed['FullTimeAwayGoals']
    y_btts = ((df_processed['FullTimeHomeGoals'] > 0) & (df_processed['FullTimeAwayGoals'] > 0)).astype(int)
    
    # Train/test split
    split_idx = int(len(X) * 0.85)
    X_test = X[split_idx:].reset_index(drop=True)
    y_result_test = y_result[split_idx:].reset_index(drop=True)
    y_goals_test = y_goals[split_idx:].reset_index(drop=True)
    y_btts_test = y_btts[split_idx:].reset_index(drop=True)
    test_idx = np.arange(split_idx, len(X))
    
    # Cargar modelos
    print(f"[MODELS] Cargando modelos entrenados...")
    gb_result, gb_goals, gb_btts, scaler = load_models_and_scaler()
    
    # Diagnósticos
    diagnose_result_model(df, X_test, y_result_test, gb_result, scaler, test_idx)
    diagnose_goals_model(df, X_test, y_goals_test, gb_goals, scaler, test_idx)
    diagnose_btts_model(df, X_test, y_btts_test, gb_btts, scaler, test_idx)
    feature_importance_analysis(X_test, gb_result, gb_goals, gb_btts)
    
    print("\n" + "="*70)
    print("[OK] DIAGNOSIS COMPLETADO")
    print("="*70)
    print("\nPROXIMOS PASOS:")
    print("1. Analiza qué clases (Home/Draw/Away) se confunden más")
    print("2. Identifica qué features importan más")
    print("3. Agrega nuevos datos de 2025/26 considerando estas insights")
    print("4. Vuelve a entrenar y compara mejoras\n")

if __name__ == '__main__':
    main()
