"""
Test de Mejoras: Comparacion de features mejoradas vs baseline
Valida el impacto de H2H draw rate, strength balance, y weak defense/strong attack
"""

import pandas as pd
import numpy as np
from src.feature_engineering import EPLFeatureEngineer, prepare_training_data
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


def load_data():
    """Carga datos EPL"""
    print("=" * 60)
    print("CARGANDO DATOS EPL")
    print("=" * 60)
    
    df = pd.read_csv('data/raw/epl_final.csv')
    
    # Agregar columnas temporales
    df['MatchDate'] = pd.to_datetime(df['MatchDate'])
    df['Year'] = df['MatchDate'].dt.year
    df['Month'] = df['MatchDate'].dt.month
    df['DayOfWeek'] = df['MatchDate'].dt.dayofweek
    
    print(f"Dataset cargado: {len(df)} partidos")
    print(f"Rango: {df['MatchDate'].min()} a {df['MatchDate'].max()}")
    
    return df


def test_new_features(df):
    """Prueba nuevas features de manera individual"""
    print("\n" + "=" * 60)
    print("PROBANDO NUEVAS FEATURES")
    print("=" * 60)
    
    engineer = EPLFeatureEngineer(df)
    
    # Test 1: H2H Draw Rate
    print("\n1. H2H_DrawRate:")
    h2h_draw = engineer.add_h2h_draw_rate(window=10)
    print(f"   - Shape: {h2h_draw.shape}")
    print(f"   - Stats: mean={h2h_draw['H2H_DrawRate'].mean():.3f}, "
          f"std={h2h_draw['H2H_DrawRate'].std():.3f}")
    print(f"   - Sample:\n{h2h_draw.head(3)}")
    
    # Test 2: Draw Tendency
    print("\n2. Draw Tendency:")
    draw_tendency = engineer.add_draw_tendency(window=10)
    print(f"   - Shape: {draw_tendency.shape}")
    print(f"   - HomeTeam_DrawRate: mean={draw_tendency['HomeTeam_DrawRate'].mean():.3f}")
    print(f"   - AwayTeam_DrawRate: mean={draw_tendency['AwayTeam_DrawRate'].mean():.3f}")
    print(f"   - Sample:\n{draw_tendency.head(3)}")
    
    # Test 3: Strength Balance
    print("\n3. Strength Balance:")
    strength = engineer.add_strength_balance(window=10)
    print(f"   - Shape: {strength.shape}")
    print(f"   - Strength_Balance: mean={strength['Strength_Balance'].mean():.3f}, "
          f"std={strength['Strength_Balance'].std():.3f}")
    print(f"   - Sample:\n{strength.head(3)}")
    
    # Test 4: Weak Defense
    print("\n4. Weak Defense Flags:")
    weak_def = engineer.add_weak_defense_flag(threshold=1.5, window=10)
    print(f"   - Shape: {weak_def.shape}")
    print(f"   - AwayTeam_WeakDefense: {weak_def['AwayTeam_WeakDefense'].sum()} equipos ({weak_def['AwayTeam_WeakDefense'].mean()*100:.1f}%)")
    print(f"   - HomeTeam_WeakDefense: {weak_def['HomeTeam_WeakDefense'].sum()} equipos ({weak_def['HomeTeam_WeakDefense'].mean()*100:.1f}%)")
    print(f"   - Sample:\n{weak_def.head(3)}")
    
    # Test 5: Strong Attack
    print("\n5. Strong Attack Flags:")
    strong_att = engineer.add_strong_attack_flag(threshold=2.0, window=10)
    print(f"   - Shape: {strong_att.shape}")
    print(f"   - HomeTeam_StrongAttack: {strong_att['HomeTeam_StrongAttack'].sum()} equipos ({strong_att['HomeTeam_StrongAttack'].mean()*100:.1f}%)")
    print(f"   - AwayTeam_StrongAttack: {strong_att['AwayTeam_StrongAttack'].sum()} equipos ({strong_att['AwayTeam_StrongAttack'].mean()*100:.1f}%)")
    print(f"   - Sample:\n{strong_att.head(3)}")


def compare_models(df):
    """Compara modelo con y sin nuevas features"""
    print("\n" + "=" * 60)
    print("COMPARACION: BASELINE vs MEJORADO")
    print("=" * 60)
    
    # Crear features mejoradas
    engineer = EPLFeatureEngineer(df)
    X, y_result, y_goals = engineer.engineer_features()
    
    # Preparar datos
    data = prepare_training_data(X, y_result, y_goals, test_size=0.15, fill_method='forward')
    
    X_train = data['X_train']
    X_test = data['X_test']
    y_result_train = data['y_result_train']
    y_result_test = data['y_result_test']
    y_goals_train = data['y_goals_train']
    y_goals_test = data['y_goals_test']
    
    # Escalar
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Entrenar modelos
    print("\nEntrenando modelos...")
    
    # Modelo Resultado 1X2
    print("  -> Resultado 1X2...")
    gb_result = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    gb_result.fit(X_train_scaled, y_result_train)
    y_result_pred = gb_result.predict(X_test_scaled)
    result_acc = accuracy_score(y_result_test, y_result_pred)
    
    # Accuracy por clase
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_result_test, y_result_pred)
    
    # Away (0), Draw (1), Home (2)
    total_by_class = cm.sum(axis=1)
    acc_away = cm[0, 0] / total_by_class[0] if total_by_class[0] > 0 else 0
    acc_draw = cm[1, 1] / total_by_class[1] if total_by_class[1] > 0 else 0
    acc_home = cm[2, 2] / total_by_class[2] if total_by_class[2] > 0 else 0
    
    # Modelo Goles
    print("  -> Goles Totales...")
    gb_goals = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    gb_goals.fit(X_train_scaled, y_goals_train)
    y_goals_pred = gb_goals.predict(X_test_scaled)
    goals_mae = mean_absolute_error(y_goals_test, y_goals_pred)
    goals_r2 = r2_score(y_goals_test, y_goals_pred)
    
    # Resultados
    print("\n" + "=" * 60)
    print("RESULTADOS CON FEATURES MEJORADAS")
    print("=" * 60)
    
    print("\nMODELO RESULTADO 1X2:")
    print(f"  Accuracy Global: {result_acc*100:.2f}%")
    print(f"  Accuracy por Clase:")
    print(f"    Away (0): {acc_away*100:.2f}% ({total_by_class[0]} partidos)")
    print(f"    Draw (1): {acc_draw*100:.2f}% ({total_by_class[1]} partidos)")
    print(f"    Home (2): {acc_home*100:.2f}% ({total_by_class[2]} partidos)")
    
    print("\nMODELO GOLES TOTALES:")
    print(f"  MAE: {goals_mae:.4f} goles")
    print(f"  R2 Score: {goals_r2*100:.2f}%")
    
    print("\n" + "=" * 60)
    print("COMPARACION CON BASELINE (20251208_143229)")
    print("=" * 60)
    
    baseline_result_acc = 74.03
    baseline_draw_acc = 45.17
    baseline_home_acc = 83.23
    baseline_away_acc = 81.50
    baseline_goals_mae = 0.8373
    
    print("\nRESULTADO 1X2:")
    print(f"  Global: {result_acc*100:.2f}% vs {baseline_result_acc:.2f}% baseline "
          f"({result_acc*100 - baseline_result_acc:+.2f}%)")
    print(f"  Draw:   {acc_draw*100:.2f}% vs {baseline_draw_acc:.2f}% baseline "
          f"({acc_draw*100 - baseline_draw_acc:+.2f}%)")
    print(f"  Home:   {acc_home*100:.2f}% vs {baseline_home_acc:.2f}% baseline "
          f"({acc_home*100 - baseline_home_acc:+.2f}%)")
    print(f"  Away:   {acc_away*100:.2f}% vs {baseline_away_acc:.2f}% baseline "
          f"({acc_away*100 - baseline_away_acc:+.2f}%)")
    
    print("\nGOLES TOTALES:")
    print(f"  MAE: {goals_mae:.4f} vs {baseline_goals_mae:.4f} baseline "
          f"({goals_mae - baseline_goals_mae:+.4f})")
    
    # Feature Importance (top 10)
    print("\n" + "=" * 60)
    print("TOP 10 FEATURES MAS IMPORTANTES (Resultado 1X2)")
    print("=" * 60)
    
    feature_names = X_train.columns
    importances = gb_result.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    for i in range(min(10, len(feature_names))):
        idx = indices[i]
        print(f"{i+1}. {feature_names[idx]:30s} : {importances[idx]:.4f}")
    
    print("\n" + "=" * 60)
    print("TOP 10 FEATURES MAS IMPORTANTES (Goles Totales)")
    print("=" * 60)
    
    importances_goals = gb_goals.feature_importances_
    indices_goals = np.argsort(importances_goals)[::-1]
    
    for i in range(min(10, len(feature_names))):
        idx = indices_goals[i]
        print(f"{i+1}. {feature_names[idx]:30s} : {importances_goals[idx]:.4f}")
    
    # Evaluacion de mejora en draws
    print("\n" + "=" * 60)
    print("EVALUACION DE MEJORA EN DRAWS")
    print("=" * 60)
    
    draw_improvement = acc_draw*100 - baseline_draw_acc
    
    if draw_improvement > 2:
        print(f"EXCELENTE: Mejora de {draw_improvement:+.2f}% en draws!")
    elif draw_improvement > 0:
        print(f"BUENO: Mejora de {draw_improvement:+.2f}% en draws")
    else:
        print(f"NEUTRAL: Cambio de {draw_improvement:+.2f}% en draws")
    
    # Evaluacion de mejora en goleadas (partidos con 4+ goles)
    high_scoring = y_goals_test >= 4
    if high_scoring.sum() > 0:
        mae_high_scoring = mean_absolute_error(
            y_goals_test[high_scoring],
            y_goals_pred[high_scoring]
        )
        print(f"\nMAE en partidos 4+ goles: {mae_high_scoring:.4f}")
        print(f"  (Baseline era ~0.99)")
    
    return {
        'result_acc': result_acc,
        'draw_acc': acc_draw,
        'goals_mae': goals_mae,
        'improvement_draw': draw_improvement,
        'num_features': len(feature_names)
    }


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("TEST DE MEJORAS: DRAWS Y GOLEADAS")
    print("=" * 60)
    print("\nEste script prueba las mejoras implementadas:")
    print("  1. H2H Draw Rate")
    print("  2. Team Draw Tendency")
    print("  3. Strength Balance")
    print("  4. Weak Defense Flags")
    print("  5. Strong Attack Flags")
    print("\n" + "=" * 60)
    
    # Cargar datos
    df = load_data()
    
    # Probar nuevas features
    test_new_features(df)
    
    # Comparar modelos
    results = compare_models(df)
    
    print("\n" + "=" * 60)
    print("TEST COMPLETADO")
    print("=" * 60)
    print(f"\nNuevo numero de features: {results['num_features']}")
    print(f"Accuracy Global: {results['result_acc']*100:.2f}%")
    print(f"Draw Accuracy: {results['draw_acc']*100:.2f}% (mejora: {results['improvement_draw']:+.2f}%)")
    print(f"Goals MAE: {results['goals_mae']:.4f}")
    
    print("\n[SIGUIENTE PASO]")
    print("Si las mejoras son positivas, ejecutar:")
    print("  python retrain_models_improved.py")
    print("\nPara incorporar estos cambios en los modelos de produccion.")
