"""
Script para ejecutar Feature Engineering y preparar datos para reentrenamiento.
Versión actualizada que maneja correctamente NaNs y variables categóricas.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from datetime import datetime

SCRIPT_DIR = Path(__file__).parent.parent
DATA_PATH = SCRIPT_DIR / "data" / "raw" / "epl_final.csv"
PROCESSED_PATH = SCRIPT_DIR / "data" / "processed"
PROCESSED_PATH.mkdir(exist_ok=True)

def create_features(df):
    """Crea features derivadas para el modelo."""
    df = df.copy()
    df['MatchDate'] = pd.to_datetime(df['MatchDate'])
    df = df.sort_values('MatchDate').reset_index(drop=True)
    
    print("🔧 Creando features derivadas...")
    
    # Features base (numéricas)
    feature_cols = [
        'HomeShots', 'AwayShots',
        'HomeShotsOnTarget', 'AwayShotsOnTarget',
        'HomeCorners', 'AwayCorners',
        'HomeFouls', 'AwayFouls',
        'HomeYellowCards', 'AwayYellowCards',
        'HomeRedCards', 'AwayRedCards',
        'HalfTimeHomeGoals', 'HalfTimeAwayGoals'
    ]
    
    X = df[feature_cols].astype(float).copy()
    
    # Feature: Forma reciente (últimos 5 partidos)
    result_map = {'H': 3, 'D': 1, 'A': 0}
    df['Result_Points'] = df['FullTimeResult'].map(result_map)
    
    X['HomeTeam_Form'] = df.groupby('HomeTeam')['Result_Points'].rolling(window=5, min_periods=1).mean().reset_index(level=0, drop=True)
    
    away_result_map = {'A': 3, 'D': 1, 'H': 0}
    df['Result_Points_Away'] = df['FullTimeResult'].map(away_result_map)
    X['AwayTeam_Form'] = df.groupby('AwayTeam')['Result_Points_Away'].rolling(window=5, min_periods=1).mean().reset_index(level=0, drop=True)
    
    # Feature: Goles promedio
    X['HomeTeam_GoalsFor_Avg'] = df.groupby('HomeTeam')['FullTimeHomeGoals'].rolling(window=10, min_periods=1).mean().reset_index(level=0, drop=True)
    X['HomeTeam_GoalsAgainst_Avg'] = df.groupby('HomeTeam')['FullTimeAwayGoals'].rolling(window=10, min_periods=1).mean().reset_index(level=0, drop=True)
    X['AwayTeam_GoalsFor_Avg'] = df.groupby('AwayTeam')['FullTimeAwayGoals'].rolling(window=10, min_periods=1).mean().reset_index(level=0, drop=True)
    X['AwayTeam_GoalsAgainst_Avg'] = df.groupby('AwayTeam')['FullTimeHomeGoals'].rolling(window=10, min_periods=1).mean().reset_index(level=0, drop=True)
    
    # Feature: Ventaja de casa (diferencia de puntos por juego)
    home_points = df.groupby('HomeTeam')['Result_Points'].rolling(window=10, min_periods=1).mean()
    away_points = df.groupby('AwayTeam')['Result_Points_Away'].rolling(window=10, min_periods=1).mean()
    
    X['HomeAdvantage'] = (
        home_points.reset_index(level=0, drop=True) - 
        away_points.reset_index(level=0, drop=True)
    )
    
    # Features temporales
    X['Month'] = df['MatchDate'].dt.month
    X['DayOfWeek'] = df['MatchDate'].dt.dayofweek
    X['Quarter'] = df['MatchDate'].dt.quarter
    
    # Llenar NaNs (primeros partidos de equipos)
    X = X.fillna(method='bfill').fillna(X.mean())
    
    print(f"  ✅ {X.shape[1]} features creadas")
    
    return X, df

def create_targets(df):
    """Crea variables target."""
    result_map = {'A': 0, 'D': 1, 'H': 2}
    y_result = df['FullTimeResult'].map(result_map)
    
    y_goals_home = df['FullTimeHomeGoals'].astype(int)
    y_goals_away = df['FullTimeAwayGoals'].astype(int)
    y_goals_total = y_goals_home + y_goals_away
    
    return y_result, y_goals_home, y_goals_away, y_goals_total

def prepare_training_data(X, y_result, y_goals_home, y_goals_away, y_goals_total, test_size=0.2):
    """Prepara train/test split (temporal, no aleatorio)."""
    print("\n📊 Preparando split train/test...")
    
    split_idx = int(len(X) * (1 - test_size))
    
    data = {
        'X_train': X[:split_idx],
        'X_test': X[split_idx:],
        'y_result_train': y_result[:split_idx],
        'y_result_test': y_result[split_idx:],
        'y_goals_home_train': y_goals_home[:split_idx],
        'y_goals_home_test': y_goals_home[split_idx:],
        'y_goals_away_train': y_goals_away[:split_idx],
        'y_goals_away_test': y_goals_away[split_idx:],
        'y_goals_total_train': y_goals_total[:split_idx],
        'y_goals_total_test': y_goals_total[split_idx:],
    }
    
    print(f"  Train: {len(data['X_train'])} muestras")
    print(f"  Test: {len(data['X_test'])} muestras")
    print(f"  Features: {X.shape[1]}")
    
    return data

def main():
    print("=" * 60)
    print("🔄 FEATURE ENGINEERING - Datos Actualizados")
    print("=" * 60)
    print()
    
    # Cargar datos
    print(f"📥 Cargando datos desde: {DATA_PATH}")
    df = pd.read_csv(str(DATA_PATH))
    print(f"   {len(df)} partidos cargados")
    print()
    
    # Crear features
    X, df_processed = create_features(df)
    
    # Crear targets
    y_result, y_goals_home, y_goals_away, y_goals_total = create_targets(df_processed)
    
    # Preparar train/test
    training_data = prepare_training_data(
        X, y_result, y_goals_home, y_goals_away, y_goals_total, 
        test_size=0.15
    )
    
    # Guardar
    print("\n💾 Guardando datos procesados...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Guardar en pickle
    with open(str(PROCESSED_PATH / f"training_data_{timestamp}.pkl"), 'wb') as f:
        pickle.dump(training_data, f)
    print(f"  ✅ training_data_{timestamp}.pkl")
    
    # Guardar últimos datos también sin timestamp
    with open(str(PROCESSED_PATH / "training_data_latest.pkl"), 'wb') as f:
        pickle.dump(training_data, f)
    print(f"  ✅ training_data_latest.pkl")
    
    # Guardar info sobre las features
    feature_info = {
        'features': X.columns.tolist(),
        'n_features': X.shape[1],
        'n_samples': len(X),
        'created_at': timestamp,
        'data_source': str(DATA_PATH),
    }
    
    with open(str(PROCESSED_PATH / f"feature_info_{timestamp}.txt"), 'w') as f:
        f.write("FEATURE ENGINEERING INFO\n")
        f.write("=" * 50 + "\n")
        f.write(f"Created: {timestamp}\n")
        f.write(f"Total samples: {feature_info['n_samples']}\n")
        f.write(f"Total features: {feature_info['n_features']}\n\n")
        f.write("FEATURES:\n")
        for i, feat in enumerate(feature_info['features'], 1):
            f.write(f"  {i}. {feat}\n")
    
    print(f"  ✅ feature_info_{timestamp}.txt")
    print()
    print("=" * 60)
    print("✅ Feature Engineering completado!")
    print("=" * 60)

if __name__ == "__main__":
    main()
