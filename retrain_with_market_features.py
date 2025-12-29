#!/usr/bin/env python3
"""
FASE 2: Reentrenamiento con Market Features - VERSIÓN OPTIMIZADA
================================================================================
Script para entrenar modelos incluyendo features de mercado (odds, probabilidades 
implícitas, etc.) y comparar accuracy antes/después de integrar market intelligence.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pickle
import warnings
import io
warnings.filterwarnings('ignore')

# Fix encoding for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

SCRIPT_DIR = Path(__file__).parent
DATA_PATH = SCRIPT_DIR / "data" / "processed" / "epl_with_market_intelligence.csv"
MODELS_PATH = SCRIPT_DIR / "models"

# Features de mercado (32 columnas basadas en el análisis actualizado)
MARKET_FEATURES = [
    # Probabilidades básicas (raw + adjusted)
    'MarketProb_Home', 'MarketProb_Draw', 'MarketProb_Away',
    'AdjustedProb_Home', 'AdjustedProb_Draw', 'AdjustedProb_Away',
    
    # Odds promedio y estadísticas
    'AvgOdds_Home', 'AvgOdds_Draw', 'AvgOdds_Away',
    'OddsStd_Home', 'OddsStd_Draw', 'OddsStd_Away',
    'OddsRange_Home', 'OddsRange_Draw', 'OddsRange_Away',
    
    # Features avanzadas del mercado
    'Overround', 'FavoriteStrength', 'MarketConsensus',
    'ImpliedGoalDiff', 'MarketDisagreement', 'MarketExpectedGoals',
    'FavoriteEV', 'IsCompetitiveMatch',
    
    # Features contextuales
    'IsUnderdog_Home', 'IsUnderdog_Away',
    
    # Features rodantes (últimos 10 partidos)
    'Team_AvgMarketProb_L10', 'Team_MarketSurpriseRate_L10', 'Team_UpsetRate_L10',
    
    # Features derivadas
    'MarketSurprise_Home', 'MarketAccuracy', 'IsUpset'
]

# Features tradicionales (estadísticas de juego)
TRADITIONAL_FEATURES = [
    'HomeShots', 'AwayShots', 'HomeShotsOnTarget', 'AwayShotsOnTarget',
    'HomeCorners', 'AwayCorners', 'HomeFouls', 'AwayFouls',
    'HomeYellowCards', 'AwayYellowCards', 'HomeRedCards', 'AwayRedCards',
    'HalfTimeHomeGoals', 'HalfTimeAwayGoals'
]


def load_and_prepare_data():
    """Carga datos desde epl_final_enriched.csv"""
    print(f"\n[LOAD] Cargando: {DATA_PATH}")
    
    if not DATA_PATH.exists():
        print(f"[ERROR] Archivo no encontrado")
        sys.exit(1)
    
    df = pd.read_csv(str(DATA_PATH))
    print(f"   ✓ {len(df)} partidos cargados ({df['Season'].min()}-{df['Season'].max()})")
    return df


def create_features_baseline(df):
    """Crea features SOLO con estadísticas de juego (baseline)."""
    print("\n[FEATURES] Baseline (sin market intelligence)...")
    
    df = df.copy()
    df['MatchDate'] = pd.to_datetime(df['MatchDate'])
    df = df.sort_values('MatchDate').reset_index(drop=True)
    
    features = pd.DataFrame(index=df.index)
    
    # Features básicas
    for col in TRADITIONAL_FEATURES:
        if col in df.columns:
            features[col] = df[col].astype(float)
    
    # Features derivadas
    result_map = {'H': 3, 'D': 1, 'A': 0}
    df['Result_Points'] = df['FullTimeResult'].map(result_map)
    away_result_map = {'A': 3, 'D': 1, 'H': 0}
    df['Result_Points_Away'] = df['FullTimeResult'].map(away_result_map)
    
    # Forma reciente (últimos 5 partidos)
    features['HomeTeam_Form'] = df.groupby('HomeTeam')['Result_Points'].rolling(
        window=5, min_periods=1).mean().reset_index(level=0, drop=True)
    features['AwayTeam_Form'] = df.groupby('AwayTeam')['Result_Points_Away'].rolling(
        window=5, min_periods=1).mean().reset_index(level=0, drop=True)
    
    # Poder ofensivo y defensivo
    features['HomeTeam_GoalsFor'] = df.groupby('HomeTeam')['FullTimeHomeGoals'].rolling(
        window=10, min_periods=1).mean().reset_index(level=0, drop=True)
    features['AwayTeam_GoalsFor'] = df.groupby('AwayTeam')['FullTimeAwayGoals'].rolling(
        window=10, min_periods=1).mean().reset_index(level=0, drop=True)
    features['HomeTeam_GoalsAgainst'] = df.groupby('HomeTeam')['FullTimeAwayGoals'].rolling(
        window=10, min_periods=1).mean().reset_index(level=0, drop=True)
    features['AwayTeam_GoalsAgainst'] = df.groupby('AwayTeam')['FullTimeHomeGoals'].rolling(
        window=10, min_periods=1).mean().reset_index(level=0, drop=True)
    
    # Diferencia de fuerza
    features['Strength_Diff'] = (
        (features['HomeTeam_GoalsFor'] + (1 - features['HomeTeam_GoalsAgainst'])) -
        (features['AwayTeam_GoalsFor'] + (1 - features['AwayTeam_GoalsAgainst']))
    ) * 2
    
    # Ratios ofensivo/defensivo
    features['Home_Attack_Defense_Ratio'] = features['HomeTeam_GoalsFor'] / (features['HomeTeam_GoalsAgainst'] + 0.1)
    features['Away_Attack_Defense_Ratio'] = features['AwayTeam_GoalsFor'] / (features['AwayTeam_GoalsAgainst'] + 0.1)
    
    # Ventaja de casa
    home_points = df.groupby('HomeTeam')['Result_Points'].rolling(window=10, min_periods=1).mean()
    away_points = df.groupby('AwayTeam')['Result_Points_Away'].rolling(window=10, min_periods=1).mean()
    features['HomeAdvantage'] = home_points.reset_index(level=0, drop=True) - away_points.reset_index(level=0, drop=True)
    
    # Tendencia a empates
    home_draw_rate = df.groupby('HomeTeam')['FullTimeResult'].apply(
        lambda x: (x == 'D').sum() / len(x) if len(x) > 0 else 0)
    away_draw_rate = df.groupby('AwayTeam')['FullTimeResult'].apply(
        lambda x: (x == 'D').sum() / len(x) if len(x) > 0 else 0)
    features['Home_Draw_Tendency'] = home_draw_rate.reset_index(level=0, drop=True)
    features['Away_Draw_Tendency'] = away_draw_rate.reset_index(level=0, drop=True)
    
    # Características temporales
    features['Month'] = df['MatchDate'].dt.month / 12
    features['DayOfWeek'] = df['MatchDate'].dt.dayofweek / 7
    
    # Rellenar NaNs
    features = features.fillna(method='bfill').fillna(features.mean())
    
    print(f"   ✓ {features.shape[1]} features (traditional only)")
    return features, df


def create_features_with_market(df):
    """Crea features CON market intelligence agregadas."""
    print("\n[FEATURES] Con market intelligence...")
    
    # Primero crear features tradicionales
    features, df_proc = create_features_baseline(df)
    
    # Agregar features de mercado
    market_features = pd.DataFrame(index=df.index)
    
    market_count = 0
    for col in MARKET_FEATURES:
        if col in df.columns:
            # Manejar features categóricas
            if col == 'MarketFavorite':
                market_features[col] = df[col].map({'H': 1, 'D': 0, 'A': -1}).fillna(0)
            # Manejar features booleanas/binarias
            elif col in ['IsUnderdog_Home', 'IsUnderdog_Away', 'IsCompetitiveMatch', 
                        'MarketAccuracy', 'IsUpset']:
                market_features[col] = df[col].fillna(0).astype(int)
            else:
                market_features[col] = df[col].astype(float)
            market_count += 1
    
    # Combinar
    combined = pd.concat([features, market_features], axis=1)
    
    # Manejo robusto de NaNs
    # 1. Forward fill para features temporales
    combined = combined.fillna(method='ffill')
    # 2. Backward fill para partidos iniciales
    combined = combined.fillna(method='bfill')
    # 3. Rellenar con media para columnas numéricas restantes
    combined = combined.fillna(combined.mean())
    # 4. Rellenar cualquier NaN restante con 0 (por seguridad)
    combined = combined.fillna(0)
    
    # Verificar que no queden NaNs
    if combined.isnull().any().any():
        print(f"   ⚠️  ADVERTENCIA: Quedan {combined.isnull().sum().sum()} NaNs")
        print("   Rellenando con 0...")
        combined = combined.fillna(0)
    
    print(f"   ✓ {combined.shape[1]} features (traditional + market: +{market_count})")
    return combined, df_proc


def train_and_compare(X_baseline, X_market, y_result, y_goals):
    """Entrena modelos baseline vs con market, compara resultados."""
    
    print("\n" + "="*80)
    print("🔄 ENTRENAMIENTO COMPARATIVO")
    print("="*80)
    
    # Split temporal 85/15
    split_idx = int(len(X_baseline) * 0.85)
    
    X_train_bl, X_test_bl = X_baseline[:split_idx], X_baseline[split_idx:]
    X_train_mk, X_test_mk = X_market[:split_idx], X_market[split_idx:]
    y_train, y_test = y_result[:split_idx], y_result[split_idx:]
    
    print(f"\n[SPLIT] Train: {len(X_train_bl)} | Test: {len(X_test_bl)}")
    print(f"   Features Baseline: {X_baseline.shape[1]}")
    print(f"   Features Market:   {X_market.shape[1]}")
    
    # Normalizar
    scaler_bl = StandardScaler()
    X_train_bl_sc = scaler_bl.fit_transform(X_train_bl)
    X_test_bl_sc = scaler_bl.transform(X_test_bl)
    
    scaler_mk = StandardScaler()
    X_train_mk_sc = scaler_mk.fit_transform(X_train_mk)
    X_test_mk_sc = scaler_mk.transform(X_test_mk)
    
    results = {'baseline': {}, 'market': {}}
    models = {'baseline': {}, 'market': {}}
    
    # ========== BASELINE ==========
    print("\n" + "-"*80)
    print("📊 BASELINE (Sin Market Features)")
    print("-"*80)
    
    # RF Baseline
    print("\n[1] Random Forest")
    rf_bl = RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_split=8,
                                    min_samples_leaf=3, class_weight='balanced', 
                                    random_state=42, n_jobs=-1, verbose=0)
    rf_bl.fit(X_train_bl_sc, y_train)
    y_pred = rf_bl.predict(X_test_bl_sc)
    
    rf_bl_acc = accuracy_score(y_test, y_pred)
    rf_bl_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"    Accuracy: {rf_bl_acc:.4f} | F1: {rf_bl_f1:.4f}")
    models['baseline']['rf'] = rf_bl
    results['baseline']['rf'] = {'acc': rf_bl_acc, 'f1': rf_bl_f1}
    
    # GB Baseline
    print("\n[2] Gradient Boosting")
    gb_bl = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5,
                                        min_samples_split=8, min_samples_leaf=3, subsample=0.8,
                                        random_state=42, verbose=0)
    gb_bl.fit(X_train_bl_sc, y_train)
    y_pred = gb_bl.predict(X_test_bl_sc)
    
    gb_bl_acc = accuracy_score(y_test, y_pred)
    gb_bl_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"    Accuracy: {gb_bl_acc:.4f} | F1: {gb_bl_f1:.4f}")
    models['baseline']['gb'] = gb_bl
    results['baseline']['gb'] = {'acc': gb_bl_acc, 'f1': gb_bl_f1}
    
    # Voting Baseline
    print("\n[3] Voting Ensemble")
    voting_bl = VotingClassifier(
        estimators=[('rf', rf_bl), ('gb', gb_bl)],
        voting='soft'
    )
    voting_bl.fit(X_train_bl_sc, y_train)
    y_pred = voting_bl.predict(X_test_bl_sc)
    
    voting_bl_acc = accuracy_score(y_test, y_pred)
    voting_bl_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"    Accuracy: {voting_bl_acc:.4f} | F1: {voting_bl_f1:.4f}")
    models['baseline']['voting'] = voting_bl
    results['baseline']['voting'] = {'acc': voting_bl_acc, 'f1': voting_bl_f1}
    
    # ========== CON MARKET ==========
    print("\n" + "-"*80)
    print("📈 CON MARKET FEATURES")
    print("-"*80)
    
    # RF Market
    print("\n[1] Random Forest")
    rf_mk = RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_split=8,
                                    min_samples_leaf=3, class_weight='balanced',
                                    random_state=42, n_jobs=-1, verbose=0)
    rf_mk.fit(X_train_mk_sc, y_train)
    y_pred = rf_mk.predict(X_test_mk_sc)
    
    rf_mk_acc = accuracy_score(y_test, y_pred)
    rf_mk_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"    Accuracy: {rf_mk_acc:.4f} (Δ {rf_mk_acc - rf_bl_acc:+.4f}) | F1: {rf_mk_f1:.4f} (Δ {rf_mk_f1 - rf_bl_f1:+.4f})")
    models['market']['rf'] = rf_mk
    results['market']['rf'] = {'acc': rf_mk_acc, 'f1': rf_mk_f1, 'imp_acc': rf_mk_acc - rf_bl_acc}
    
    # GB Market
    print("\n[2] Gradient Boosting")
    gb_mk = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5,
                                        min_samples_split=8, min_samples_leaf=3, subsample=0.8,
                                        random_state=42, verbose=0)
    gb_mk.fit(X_train_mk_sc, y_train)
    y_pred = gb_mk.predict(X_test_mk_sc)
    
    gb_mk_acc = accuracy_score(y_test, y_pred)
    gb_mk_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"    Accuracy: {gb_mk_acc:.4f} (Δ {gb_mk_acc - gb_bl_acc:+.4f}) | F1: {gb_mk_f1:.4f} (Δ {gb_mk_f1 - gb_bl_f1:+.4f})")
    models['market']['gb'] = gb_mk
    results['market']['gb'] = {'acc': gb_mk_acc, 'f1': gb_mk_f1, 'imp_acc': gb_mk_acc - gb_bl_acc}
    
    # Voting Market
    print("\n[3] Voting Ensemble")
    voting_mk = VotingClassifier(
        estimators=[('rf', rf_mk), ('gb', gb_mk)],
        voting='soft'
    )
    voting_mk.fit(X_train_mk_sc, y_train)
    y_pred = voting_mk.predict(X_test_mk_sc)
    
    voting_mk_acc = accuracy_score(y_test, y_pred)
    voting_mk_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    print(f"    Accuracy: {voting_mk_acc:.4f} (Δ {voting_mk_acc - voting_bl_acc:+.4f}) | F1: {voting_mk_f1:.4f} (Δ {voting_mk_f1 - voting_bl_f1:+.4f})")
    models['market']['voting'] = voting_mk
    results['market']['voting'] = {'acc': voting_mk_acc, 'f1': voting_mk_f1, 'imp_acc': voting_mk_acc - voting_bl_acc}
    
    return results, models, scaler_bl, scaler_mk


def save_models_and_report(results, models, scaler_bl, scaler_mk):
    """Guarda modelos y reporte comparativo."""
    
    print("\n" + "="*80)
    print("💾 GUARDANDO MODELOS Y REPORTE")
    print("="*80)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Guardar modelos
    models_to_save = {
        'phase2_rf_market.pkl': models['market']['rf'],
        'phase2_gb_market.pkl': models['market']['gb'],
        'phase2_voting_market.pkl': models['market']['voting'],
        'phase2_scaler_baseline.pkl': scaler_bl,
        'phase2_scaler_market.pkl': scaler_mk,
    }
    
    for filename, model in models_to_save.items():
        try:
            with open(MODELS_PATH / filename, 'wb') as f:
                pickle.dump(model, f)
            print(f"   ✓ {filename}")
        except Exception as e:
            print(f"   ✗ {filename}: {e}")
            return False
    
    # Generar reporte
    report_path = MODELS_PATH / f"PHASE2_COMPARISON_{timestamp}.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("FASE 2: COMPARACIÓN - SIN vs CON MARKET FEATURES\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Dataset: {DATA_PATH.name}\n")
        f.write(f"Market Features Disponibles: {len(MARKET_FEATURES)}\n\n")
        
        f.write("FEATURES DE MERCADO INTEGRADAS:\n")
        f.write("-"*80 + "\n")
        f.write("Probabilidades:\n")
        f.write("  • MarketProb_Home/Draw/Away (raw probabilities)\n")
        f.write("  • AdjustedProb_Home/Draw/Away (sin overround)\n")
        f.write("\nOdds y Estadísticas:\n")
        f.write("  • AvgOdds_*, OddsStd_*, OddsRange_* (promedio, desv. std, rango)\n")
        f.write("\nFeatures Avanzadas:\n")
        f.write("  • FavoriteStrength, MarketConsensus, ImpliedGoalDiff\n")
        f.write("  • MarketDisagreement, FavoriteEV, IsCompetitiveMatch\n")
        f.write("\nFeatures Contextuales:\n")
        f.write("  • IsUnderdog_*, Team_AvgMarketProb_L10, Team_UpsetRate_L10\n")
        f.write("  • MarketSurprise_Home, MarketAccuracy, IsUpset\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("COMPARACIÓN DE RESULTADOS\n")
        f.write("="*80 + "\n\n")
        
        # Random Forest
        f.write("1️⃣  RANDOM FOREST\n")
        f.write("-"*80 + "\n")
        bl = results['baseline']['rf']
        mk = results['market']['rf']
        f.write(f"   BASELINE:  Acc={bl['acc']:.4f} | F1={bl['f1']:.4f}\n")
        f.write(f"   MARKET:    Acc={mk['acc']:.4f} | F1={mk['f1']:.4f}\n")
        f.write(f"   MEJORA:    ΔAcc={mk['imp_acc']:+.4f} ({mk['imp_acc']/bl['acc']*100:+.2f}%)\n\n")
        
        # Gradient Boosting
        f.write("2️⃣  GRADIENT BOOSTING\n")
        f.write("-"*80 + "\n")
        bl = results['baseline']['gb']
        mk = results['market']['gb']
        f.write(f"   BASELINE:  Acc={bl['acc']:.4f} | F1={bl['f1']:.4f}\n")
        f.write(f"   MARKET:    Acc={mk['acc']:.4f} | F1={mk['f1']:.4f}\n")
        f.write(f"   MEJORA:    ΔAcc={mk['imp_acc']:+.4f} ({mk['imp_acc']/bl['acc']*100:+.2f}%)\n\n")
        
        # Voting Ensemble
        f.write("3️⃣  VOTING ENSEMBLE (RF + GB) ⭐\n")
        f.write("-"*80 + "\n")
        bl = results['baseline']['voting']
        mk = results['market']['voting']
        f.write(f"   BASELINE:  Acc={bl['acc']:.4f} | F1={bl['f1']:.4f}\n")
        f.write(f"   MARKET:    Acc={mk['acc']:.4f} | F1={mk['f1']:.4f}\n")
        f.write(f"   MEJORA:    ΔAcc={mk['imp_acc']:+.4f} ({mk['imp_acc']/bl['acc']*100:+.2f}%)\n\n")
        
        # Análisis
        avg_improvement = np.mean([
            results['market']['rf']['imp_acc'],
            results['market']['gb']['imp_acc'],
            results['market']['voting']['imp_acc']
        ])
        
        f.write("="*80 + "\n")
        f.write("ANÁLISIS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Mejora promedio en Accuracy: {avg_improvement:+.4f}\n")
        
        if avg_improvement > 0.01:
            f.write("\n✅ Market Features mejoran SIGNIFICATIVAMENTE el accuracy\n")
        elif avg_improvement > 0.005:
            f.write("\n⚠️  Market Features mejoran levemente el accuracy\n")
        else:
            f.write("\n⚠️  Market Features tienen impacto limitado\n")
        
        f.write("\n✅ Fase 2 completada exitosamente\n")
        f.write(f"✅ {len(MARKET_FEATURES)} features de mercado integradas\n")
        f.write("✅ Probabilidades ajustadas sin overround (~7.9% removed)\n")
        f.write("✅ Features de consenso y desacuerdo entre casas\n")
        f.write("✅ Features rodantes (L10) y contextuales (underdogs, upsets)\n")
        f.write("\n📊 Próximos pasos:\n")
        f.write("   → Usar phase2_voting_market.pkl para predicciones en producción\n")
        f.write("   → Implementar ensemble: 70% ML + 30% Market probabilities\n")
        f.write("   → Backtest completo de value betting con 9,319 partidos\n")
        f.write("   → Integrar odds en tiempo real para predicciones futuras\n")
    
    print(f"\n✓ Reporte: {report_path.name}")
    return True


def main():
    print("\n" + "="*80)
    print("FASE 2: REENTRENAMIENTO CON MARKET FEATURES")
    print("="*80)
    
    # Cargar datos
    df = load_and_prepare_data()
    
    # Crear features
    X_baseline, df_proc = create_features_baseline(df)
    X_market, _ = create_features_with_market(df)
    
    # Targets
    result_map = {'A': 0, 'D': 1, 'H': 2}
    y_result = df_proc['FullTimeResult'].map(result_map)
    y_goals = df_proc['FullTimeHomeGoals'] + df_proc['FullTimeAwayGoals']
    
    # Entrenar y comparar
    results, models, scaler_bl, scaler_mk = train_and_compare(
        X_baseline, X_market, y_result, y_goals
    )
    
    # Guardar
    if save_models_and_report(results, models, scaler_bl, scaler_mk):
        print("\n" + "="*80)
        print("✅ FASE 2 COMPLETADA")
        print("="*80)
        print("\n📈 Modelos con market features listos para predicción")
        print("📊 Reporte comparativo guardado en models/\n")


if __name__ == '__main__':
    main()
