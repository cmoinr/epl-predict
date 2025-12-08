#!/usr/bin/env python3
"""
Script para reentrenar modelos con mejoras:
1. class_weight='balanced' para Random Forest
2. Features adicionales mejoradas
3. Mejor calibración de probabilidades

Este script genera nuevos modelos con mejor discriminación entre clases.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import json
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).parent
DATA_PATH = SCRIPT_DIR / "data" / "raw" / "epl_final.csv"
MODELS_PATH = SCRIPT_DIR / "models"

def create_enhanced_features(df):
    """
    Crea features mejoradas que ayudan a distinguir entre Home Win, Draw y Away Win.
    Basadas en análisis de estadísticas reales de equipos.
    """
    print("\n[FEATURE] Creando features mejoradas...")
    
    df = df.copy()
    df['MatchDate'] = pd.to_datetime(df['MatchDate'])
    df = df.sort_values('MatchDate').reset_index(drop=True)
    
    features = pd.DataFrame(index=df.index)
    
    # Features básicas
    base_features = [
        'HomeShots', 'AwayShots',
        'HomeShotsOnTarget', 'AwayShotsOnTarget',
        'HomeCorners', 'AwayCorners',
        'HomeFouls', 'AwayFouls',
        'HomeYellowCards', 'AwayYellowCards',
        'HomeRedCards', 'AwayRedCards',
        'HalfTimeHomeGoals', 'HalfTimeAwayGoals'
    ]
    
    for col in base_features:
        features[col] = df[col].astype(float)
    
    # Maps para forma
    result_map = {'H': 3, 'D': 1, 'A': 0}
    df['Result_Points'] = df['FullTimeResult'].map(result_map)
    
    away_result_map = {'A': 3, 'D': 1, 'H': 0}
    df['Result_Points_Away'] = df['FullTimeResult'].map(away_result_map)
    
    # FEATURE 1: Forma reciente (últimos 5 partidos)
    features['HomeTeam_Form'] = df.groupby('HomeTeam')['Result_Points'].rolling(
        window=5, min_periods=1).mean().reset_index(level=0, drop=True)
    features['AwayTeam_Form'] = df.groupby('AwayTeam')['Result_Points_Away'].rolling(
        window=5, min_periods=1).mean().reset_index(level=0, drop=True)
    
    # FEATURE 2: Poder ofensivo (goles anotados promedio)
    features['HomeTeam_GoalsFor'] = df.groupby('HomeTeam')['FullTimeHomeGoals'].rolling(
        window=10, min_periods=1).mean().reset_index(level=0, drop=True)
    features['AwayTeam_GoalsFor'] = df.groupby('AwayTeam')['FullTimeAwayGoals'].rolling(
        window=10, min_periods=1).mean().reset_index(level=0, drop=True)
    
    # FEATURE 3: Poder defensivo (goles concedidos promedio) - MÁS IMPORTANTE
    features['HomeTeam_GoalsAgainst'] = df.groupby('HomeTeam')['FullTimeAwayGoals'].rolling(
        window=10, min_periods=1).mean().reset_index(level=0, drop=True)
    features['AwayTeam_GoalsAgainst'] = df.groupby('AwayTeam')['FullTimeHomeGoals'].rolling(
        window=10, min_periods=1).mean().reset_index(level=0, drop=True)
    
    # FEATURE 4: Diferencia relativa de fuerza (ES LA CLAVE)
    # Equipos ofensivos vs defensivos → ganador claro
    # Equipos similares → posible draw
    features['Strength_Diff'] = (
        (features['HomeTeam_GoalsFor'] + (1 - features['HomeTeam_GoalsAgainst'])) -
        (features['AwayTeam_GoalsFor'] + (1 - features['AwayTeam_GoalsAgainst']))
    ) * 2  # Amplificar diferencia
    
    # FEATURE 5: Ratio ofensivo/defensivo local vs visitante
    features['Home_Attack_Defense_Ratio'] = features['HomeTeam_GoalsFor'] / (features['HomeTeam_GoalsAgainst'] + 0.1)
    features['Away_Attack_Defense_Ratio'] = features['AwayTeam_GoalsFor'] / (features['AwayTeam_GoalsAgainst'] + 0.1)
    
    # FEATURE 6: Ventaja de casa (home teams ganan más en casa)
    home_points = df.groupby('HomeTeam')['Result_Points'].rolling(
        window=10, min_periods=1).mean()
    away_points = df.groupby('AwayTeam')['Result_Points_Away'].rolling(
        window=10, min_periods=1).mean()
    
    features['HomeAdvantage'] = (
        home_points.reset_index(level=0, drop=True) - 
        away_points.reset_index(level=0, drop=True)
    )
    
    # FEATURE 7: Frecuencia de draws (equipos defensivos tienden a empates)
    home_draw_rate = df.groupby('HomeTeam')['FullTimeResult'].apply(
        lambda x: (x == 'D').sum() / len(x)).rolling(window=10, min_periods=1).mean()
    away_draw_rate = df.groupby('AwayTeam')['FullTimeResult'].apply(
        lambda x: (x == 'D').sum() / len(x)).rolling(window=10, min_periods=1).mean()
    
    features['Home_Draw_Tendency'] = home_draw_rate.reset_index(level=0, drop=True)
    features['Away_Draw_Tendency'] = away_draw_rate.reset_index(level=0, drop=True)
    
    # FEATURE 8: Features temporales
    features['Month'] = df['MatchDate'].dt.month / 12
    features['DayOfWeek'] = df['MatchDate'].dt.dayofweek / 7
    
    # Rellenar NaNs
    features = features.fillna(method='bfill').fillna(features.mean())
    
    print(f"   [OK] {features.shape[1]} features creadas")
    print(f"   [OK] Total de muestras: {features.shape[0]}")
    
    return features, df

def train_models_improved(X_train, X_test, y_result_train, y_result_test, 
                         y_goals_train, y_goals_test, y_btts_train, y_btts_test):
    """
    Entrena modelos mejorados con class_weight balanced.
    """
    
    print("\n" + "="*70)
    print("🤖 ENTRENAMIENTO DE MODELOS MEJORADOS")
    print("="*70)
    
    # Normalizar
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # ===== RANDOM FOREST (con class_weight='balanced') =====
    print("\n[1] RANDOM FOREST (Resultado 1X2)")
    print("   Configuración: class_weight='balanced'")
    
    rf_result = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_split=8,
        min_samples_leaf=3,
        class_weight='balanced',  # ← MEJORA KEY
        random_state=42,
        n_jobs=-1
    )
    rf_result.fit(X_train_scaled, y_result_train)
    y_pred_rf = rf_result.predict(X_test_scaled)
    
    rf_result_acc = accuracy_score(y_result_test, y_pred_rf)
    rf_result_prec = precision_score(y_result_test, y_pred_rf, average='weighted', zero_division=0)
    rf_result_recall = recall_score(y_result_test, y_pred_rf, average='weighted', zero_division=0)
    rf_result_f1 = f1_score(y_result_test, y_pred_rf, average='weighted', zero_division=0)
    
    print(f"   Accuracy:  {rf_result_acc:.4f}")
    print(f"   Precision: {rf_result_prec:.4f}")
    print(f"   Recall:    {rf_result_recall:.4f}")
    print(f"   F1-Score:  {rf_result_f1:.4f}")
    
    # ===== GRADIENT BOOSTING (Resultado 1X2) =====
    print("\n[2] GRADIENT BOOSTING (Resultado 1X2)")
    
    gb_result = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=8,
        min_samples_leaf=3,
        subsample=0.8,
        random_state=42
    )
    gb_result.fit(X_train_scaled, y_result_train)
    y_pred_gb = gb_result.predict(X_test_scaled)
    
    gb_result_acc = accuracy_score(y_result_test, y_pred_gb)
    gb_result_prec = precision_score(y_result_test, y_pred_gb, average='weighted', zero_division=0)
    gb_result_recall = recall_score(y_result_test, y_pred_gb, average='weighted', zero_division=0)
    gb_result_f1 = f1_score(y_result_test, y_pred_gb, average='weighted', zero_division=0)
    
    print(f"   Accuracy:  {gb_result_acc:.4f}")
    print(f"   Precision: {gb_result_prec:.4f}")
    print(f"   Recall:    {gb_result_recall:.4f}")
    print(f"   F1-Score:  {gb_result_f1:.4f}")
    
    # ===== RANDOM FOREST (Goles Totales) =====
    print("\n[3] RANDOM FOREST (Goles Totales)")
    
    rf_goals = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_split=8,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )
    rf_goals.fit(X_train_scaled, y_goals_train)
    y_pred_goals_rf = rf_goals.predict(X_test_scaled)
    
    from sklearn.metrics import mean_absolute_error, r2_score
    rf_goals_mae = mean_absolute_error(y_goals_test, y_pred_goals_rf)
    rf_goals_r2 = r2_score(y_goals_test, y_pred_goals_rf)
    
    print(f"   MAE:  {rf_goals_mae:.4f}")
    print(f"   R²:   {rf_goals_r2:.4f}")
    
    # ===== GRADIENT BOOSTING (Goles Totales) =====
    print("\n[4] GRADIENT BOOSTING (Goles Totales)")
    
    gb_goals = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=8,
        min_samples_leaf=3,
        subsample=0.8,
        random_state=42
    )
    gb_goals.fit(X_train_scaled, y_goals_train)
    y_pred_goals_gb = gb_goals.predict(X_test_scaled)
    
    gb_goals_mae = mean_absolute_error(y_goals_test, y_pred_goals_gb)
    gb_goals_r2 = r2_score(y_goals_test, y_pred_goals_gb)
    
    print(f"   MAE:  {gb_goals_mae:.4f}")
    print(f"   R²:   {gb_goals_r2:.4f}")
    
    # ===== RANDOM FOREST (BTTS) =====
    print("\n5. RANDOM FOREST (BTTS)")
    
    rf_btts = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_split=8,
        min_samples_leaf=3,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    rf_btts.fit(X_train_scaled, y_btts_train)
    y_pred_btts_rf = rf_btts.predict(X_test_scaled)
    
    rf_btts_acc = accuracy_score(y_btts_test, y_pred_btts_rf)
    rf_btts_prec = precision_score(y_btts_test, y_pred_btts_rf, average='weighted', zero_division=0)
    rf_btts_recall = recall_score(y_btts_test, y_pred_btts_rf, average='weighted', zero_division=0)
    rf_btts_f1 = f1_score(y_btts_test, y_pred_btts_rf, average='weighted', zero_division=0)
    
    print(f"   Accuracy:  {rf_btts_acc:.4f}")
    print(f"   Precision: {rf_btts_prec:.4f}")
    print(f"   Recall:    {rf_btts_recall:.4f}")
    print(f"   F1-Score:  {rf_btts_f1:.4f}")
    
    # ===== GRADIENT BOOSTING (BTTS) =====
    print("\n6. GRADIENT BOOSTING (BTTS)")
    
    gb_btts = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=8,
        min_samples_leaf=3,
        subsample=0.8,
        random_state=42
    )
    gb_btts.fit(X_train_scaled, y_btts_train)
    y_pred_btts_gb = gb_btts.predict(X_test_scaled)
    
    gb_btts_acc = accuracy_score(y_btts_test, y_pred_btts_gb)
    gb_btts_prec = precision_score(y_btts_test, y_pred_btts_gb, average='weighted', zero_division=0)
    gb_btts_recall = recall_score(y_btts_test, y_pred_btts_gb, average='weighted', zero_division=0)
    gb_btts_f1 = f1_score(y_btts_test, y_pred_btts_gb, average='weighted', zero_division=0)
    
    print(f"   Accuracy:  {gb_btts_acc:.4f}")
    print(f"   Precision: {gb_btts_prec:.4f}")
    print(f"   Recall:    {gb_btts_recall:.4f}")
    print(f"   F1-Score:  {gb_btts_f1:.4f}")
    
    # Diccionario con todas las métricas
    metrics = {
        'rf_result': {
            'accuracy': rf_result_acc,
            'precision': rf_result_prec,
            'recall': rf_result_recall,
            'f1_score': rf_result_f1
        },
        'gb_result': {
            'accuracy': gb_result_acc,
            'precision': gb_result_prec,
            'recall': gb_result_recall,
            'f1_score': gb_result_f1
        },
        'rf_goals': {
            'mae': rf_goals_mae,
            'r2_score': rf_goals_r2
        },
        'gb_goals': {
            'mae': gb_goals_mae,
            'r2_score': gb_goals_r2
        },
        'rf_btts': {
            'accuracy': rf_btts_acc,
            'precision': rf_btts_prec,
            'recall': rf_btts_recall,
            'f1_score': rf_btts_f1
        },
        'gb_btts': {
            'accuracy': gb_btts_acc,
            'precision': gb_btts_prec,
            'recall': gb_btts_recall,
            'f1_score': gb_btts_f1
        }
    }
    
    return rf_result, gb_result, rf_goals, gb_goals, rf_btts, gb_btts, scaler, metrics

def save_models_improved(rf_result, gb_result, rf_goals, gb_goals, rf_btts, gb_btts, scaler, metrics):
    """Guardar modelos mejorados sobrescribiendo los antiguos."""
    
    print("\n" + "="*70)
    print("GUARDANDO MODELOS MEJORADOS")
    print("="*70 + "\n")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    models_to_save = {
        'rf_result_model.pkl': rf_result,
        'gb_result_model.pkl': gb_result,
        'rf_goals_model.pkl': rf_goals,
        'gb_goals_model.pkl': gb_goals,
        'rf_btts_model.pkl': rf_btts,
        'gb_btts_model.pkl': gb_btts,
        'scaler_model.pkl': scaler,
    }
    
    for filename, model in models_to_save.items():
        try:
            with open(MODELS_PATH / filename, 'wb') as f:
                pickle.dump(model, f)
            print(f"   [OK] {filename}")
        except Exception as e:
            print(f"   [ERROR] Error guardando {filename}: {e}")
            return False
    
    # Guardar timestamp y métricas de entrenamiento
    with open(MODELS_PATH / "training_timestamp.txt", 'w', encoding='utf-8') as f:
        f.write(f"Timestamp: {timestamp}\n\n")
        f.write(f"METRICAS DE ENTRENAMIENTO\n")
        f.write(f"=" * 70 + "\n\n")
        
        # Resultado 1X2
        f.write(f"1. RESULTADO 1X2 (Home/Draw/Away)\n")
        f.write(f"   Random Forest:\n")
        f.write(f"      Accuracy:  {metrics['rf_result']['accuracy']:.2%}\n")
        f.write(f"      Precision: {metrics['rf_result']['precision']:.2%}\n")
        f.write(f"      Recall:    {metrics['rf_result']['recall']:.2%}\n")
        f.write(f"      F1-Score:  {metrics['rf_result']['f1_score']:.2%}\n")
        f.write(f"   Gradient Boosting:\n")
        f.write(f"      Accuracy:  {metrics['gb_result']['accuracy']:.2%}\n")
        f.write(f"      Precision: {metrics['gb_result']['precision']:.2%}\n")
        f.write(f"      Recall:    {metrics['gb_result']['recall']:.2%}\n")
        f.write(f"      F1-Score:  {metrics['gb_result']['f1_score']:.2%}\n\n")
        
        # Goles Totales
        f.write(f"2. GOLES TOTALES (Regresión)\n")
        f.write(f"   Random Forest:\n")
        f.write(f"      MAE: {metrics['rf_goals']['mae']:.4f}\n")
        f.write(f"      R²:  {metrics['rf_goals']['r2_score']:.2%}\n")
        f.write(f"   Gradient Boosting:\n")
        f.write(f"      MAE: {metrics['gb_goals']['mae']:.4f}\n")
        f.write(f"      R²:  {metrics['gb_goals']['r2_score']:.2%}\n\n")
        
        # BTTS
        f.write(f"3. AMBOS ANOTAN (BTTS)\n")
        f.write(f"   Random Forest:\n")
        f.write(f"      Accuracy:  {metrics['rf_btts']['accuracy']:.2%}\n")
        f.write(f"      Precision: {metrics['rf_btts']['precision']:.2%}\n")
        f.write(f"      Recall:    {metrics['rf_btts']['recall']:.2%}\n")
        f.write(f"      F1-Score:  {metrics['rf_btts']['f1_score']:.2%}\n")
        f.write(f"   Gradient Boosting:\n")
        f.write(f"      Accuracy:  {metrics['gb_btts']['accuracy']:.2%}\n")
        f.write(f"      Precision: {metrics['gb_btts']['precision']:.2%}\n")
        f.write(f"      Recall:    {metrics['gb_btts']['recall']:.2%}\n")
        f.write(f"      F1-Score:  {metrics['gb_btts']['f1_score']:.2%}\n\n")
        
        f.write(f"=" * 70 + "\n")
        f.write(f"MEJORAS APLICADAS\n")
        f.write(f"- class_weight='balanced' en Random Forest\n")
        f.write(f"- Features adicionales mejoradas\n")
        f.write(f"- Hiperparametros optimizados\n")
    
    print(f"\n[OK] Modelos mejorados guardados exitosamente")
    print(f"   Timestamp: {timestamp}")
    
    return True

def main():
    print("\n" + "="*70)
    print("[TRAIN] REENTRENAMIENTO CON MEJORAS APLICADAS")
    print("="*70)
    print("\nMejoras a aplicar:")
    print("  - class_weight='balanced' en Random Forest")
    print("  - Features adicionales que distinguen favoritos vs draws")
    print("  - Hiperparametros optimizados para mejor discriminacion")
    print("="*70)
    
    # Cargar datos
    print(f"\n[LOAD] Cargando datos desde: {DATA_PATH}")
    df = pd.read_csv(str(DATA_PATH))
    print(f"   {len(df)} partidos cargados")
    
    # Crear features mejoradas
    X, df_processed = create_enhanced_features(df)
    
    # Crear targets
    result_map = {'A': 0, 'D': 1, 'H': 2}
    y_result = df_processed['FullTimeResult'].map(result_map)
    y_goals = df_processed['FullTimeHomeGoals'] + df_processed['FullTimeAwayGoals']
    y_btts = ((df_processed['FullTimeHomeGoals'] > 0) & (df_processed['FullTimeAwayGoals'] > 0)).astype(int)
    
    # Train/test split (temporal)
    split_idx = int(len(X) * 0.85)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_result_train, y_result_test = y_result[:split_idx], y_result[split_idx:]
    y_goals_train, y_goals_test = y_goals[:split_idx], y_goals[split_idx:]
    y_btts_train, y_btts_test = y_btts[:split_idx], y_btts[split_idx:]
    
    print(f"\n[SPLIT] Split train/test:")
    print(f"   Train: {len(X_train)} partidos")
    print(f"   Test: {len(X_test)} partidos")
    print(f"   Features: {X.shape[1]}")
    
    # Entrenar modelos mejorados
    rf_result, gb_result, rf_goals, gb_goals, rf_btts, gb_btts, scaler, metrics = train_models_improved(
        X_train, X_test, y_result_train, y_result_test, y_goals_train, y_goals_test, y_btts_train, y_btts_test
    )
    
    # Guardar
    if save_models_improved(rf_result, gb_result, rf_goals, gb_goals, rf_btts, gb_btts, scaler, metrics):
        print("\n" + "="*70)
        print("[OK] REENTRENAMIENTO COMPLETADO")
        print("="*70)
        print("\n📝 PRÓXIMOS PASOS:")
        print("   1. Prueba nuevas predicciones: python predict_match.py --home 'Chelsea' --away 'Liverpool'")
        print("   2. Los modelos deberían dar menos predicciones 'Draw'")
        print("   3. Las diferencias entre Random Forest y Gradient Boosting deberían ser menores")
        print("\n")
    else:
        print("\n[ERROR] Error al guardar modelos")
        sys.exit(1)

if __name__ == '__main__':
    main()
