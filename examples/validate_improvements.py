#!/usr/bin/env python3
"""
Script de validación: Prueba los modelos mejorados con predicciones históricas
y compara con resultados reales.

Uso: python validate_improvements.py
"""

import sys
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from predictor import EPLPredictor

# Tus 4 partidos de prueba (ACTUALIZA CON TUS DATOS REALES)
TEST_MATCHES = [
    {
        'home': 'Liverpool',
        'away': 'Bournemouth',
        'date': '2025-08-15',
        'actual_result': 'Home ganó',
        'actual_goals': 6,
    },
    {
        'home': 'Aston Villa',
        'away': 'Newcastle',
        'date': '2025-08-16',
        'actual_result': 'Empate',
        'actual_goals': 0,
    },
    {
        'home': 'Tottenham',
        'away': 'Burnley',
        'date': '2025-08-16',
        'actual_result': 'Home ganó',
        'actual_goals': 3,
    },
    {
        'home': 'Sunderland',
        'away': 'West Ham',
        'date': '2025-08-16',
        'actual_result': 'Home ganó',
        'actual_goals': 3,
    },
]

def validate_predictions():
    """Valida los modelos mejorados contra datos reales."""
    
    print("\n" + "="*70)
    print("🧪 VALIDACIÓN DE MODELOS MEJORADOS")
    print("="*70)
    
    # Cargar datos
    data_path = Path(__file__).parent / "data" / "raw" / "epl_final.csv"
    df = pd.read_csv(str(data_path))
    
    # Inicializar predictor
    models_path = Path(__file__).parent / "models"
    predictor = EPLPredictor(str(models_path))
    
    results_1x2 = {'correct': 0, 'total': 0}
    results_goals = {'correct': 0, 'total': 0}
    
    print("\n📊 PREDICCIONES vs RESULTADOS REALES:\n")
    
    for i, match in enumerate(TEST_MATCHES, 1):
        print(f"\n{i}. {match['home']} vs {match['away']} ({match['date']})")
        print("-" * 70)
        
        # Predecir
        try:
            result = predictor.predict_match(
                df_historical=df,
                home_team=match['home'],
                away_team=match['away'],
                match_date=match['date']
            )
            
            # Extraer predicciones
            rf_pred_1x2 = result['resultado']['random_forest']['prediccion']
            gb_pred_1x2 = result['resultado']['gradient_boosting']['prediccion']
            rf_pred_goals = result['goles_totales']['random_forest']
            gb_pred_goals = result['goles_totales']['gradient_boosting']
            avg_goals = result['goles_totales']['promedio']
            
            print(f"Resultado Real: {match['actual_result']} | {match['actual_goals']} goles total\n")
            
            # Mostrar predicciones
            print(f"🌲 Random Forest:")
            print(f"   Resultado: {rf_pred_1x2}")
            print(f"   Goles: {rf_pred_goals}")
            
            print(f"⚡ Gradient Boosting:")
            print(f"   Resultado: {gb_pred_1x2}")
            print(f"   Goles: {gb_pred_goals}")
            
            # Validación 1X2
            results_1x2['total'] += 1
            
            if (match['actual_result'] == 'Home ganó' and rf_pred_1x2 == 'Home Win') or \
               (match['actual_result'] == 'Home ganó' and gb_pred_1x2 == 'Home Win'):
                results_1x2['correct'] += 1
                print(f"✅ ACIERTO en resultado (Random Forest o GB)")
            elif (match['actual_result'] == 'Away ganó' and rf_pred_1x2 == 'Away Win') or \
                 (match['actual_result'] == 'Away ganó' and gb_pred_1x2 == 'Away Win'):
                results_1x2['correct'] += 1
                print(f"✅ ACIERTO en resultado (Random Forest o GB)")
            elif (match['actual_result'] == 'Empate' and rf_pred_1x2 == 'Draw') or \
                 (match['actual_result'] == 'Empate' and gb_pred_1x2 == 'Draw'):
                results_1x2['correct'] += 1
                print(f"✅ ACIERTO en resultado (Random Forest o GB)")
            else:
                print(f"❌ ERROR en resultado")
            
            # Validación Goles
            results_goals['total'] += 1
            
            # Usar promedio como predicción
            goals_diff = abs(avg_goals - match['actual_goals'])
            
            if goals_diff <= 0.5:
                results_goals['correct'] += 1
                print(f"✅ ACIERTO en goles (predicción: {avg_goals:.1f}, real: {match['actual_goals']})")
            elif goals_diff <= 1.0:
                print(f"⚠️  CERCANO en goles (predicción: {avg_goals:.1f}, real: {match['actual_goals']})")
            else:
                print(f"❌ ERROR en goles (predicción: {avg_goals:.1f}, real: {match['actual_goals']})")
                
        except Exception as e:
            print(f"❌ Error en predicción: {e}")
    
    # Resumen
    print("\n" + "="*70)
    print("📈 RESUMEN DE VALIDACIÓN")
    print("="*70)
    
    accuracy_1x2 = (results_1x2['correct'] / results_1x2['total'] * 100) if results_1x2['total'] > 0 else 0
    accuracy_goals = (results_goals['correct'] / results_goals['total'] * 100) if results_goals['total'] > 0 else 0
    
    print(f"\n🎯 RESULTADO (1X2):")
    print(f"   Correctas: {results_1x2['correct']}/{results_1x2['total']}")
    print(f"   Accuracy: {accuracy_1x2:.1f}%")
    
    print(f"\n⚽ GOLES TOTALES:")
    print(f"   Correctas (±0.5): {results_goals['correct']}/{results_goals['total']}")
    print(f"   Accuracy: {accuracy_goals:.1f}%")
    
    print("\n" + "="*70)
    
    # Interpretación
    if accuracy_1x2 >= 75:
        print("🎉 EXCELENTE: Los modelos mejoraron significativamente!")
    elif accuracy_1x2 >= 50:
        print("✅ BUENO: Los modelos están funcionando mejor")
    else:
        print("⚠️  Los modelos aún necesitan ajustes")

if __name__ == '__main__':
    validate_predictions()
