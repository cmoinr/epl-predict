#!/usr/bin/env python
"""
Integración: Usar predicciones reales del modelo + comparar con odds
"""

import sys
from pathlib import Path
import pandas as pd

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent))

from src.odds_comparison import OddsComparison


def get_predictions_from_model(matches_df):
    """
    En producción, aquí cargarías predicciones del modelo entrenado
    
    Para demo, retorna predicciones de ejemplo
    """
    predictions = []
    
    for _, match in matches_df.iterrows():
        # En producción: llamar a predictor.predict()
        prediction = {
            'match_id': f"{match['home_team']}_{match['away_team']}",
            'date': match['date'],
            'resultado': {
                'random_forest': {
                    'probabilidades': {
                        'Home Win': 55,  # Reemplazar con pred real
                        'Draw': 20,
                        'Away Win': 25
                    }
                },
                'gradient_boosting': {
                    'probabilidades': {
                        'Home Win': 57,
                        'Draw': 18,
                        'Away Win': 25
                    }
                }
            }
        }
        predictions.append(prediction)
    
    return predictions


def integrate_model_with_odds(odds_file='data/processed/sample_odds.csv'):
    """
    Flujo completo: Predicciones + Odds = Oportunidades
    """
    print("\n" + "="*120)
    print("🔗 INTEGRACIÓN: MODELO + ODDS = OPORTUNIDADES")
    print("="*120)
    
    # 1. Cargar odds
    print("\n📥 Paso 1: Cargar odds del mercado...")
    odds_df = pd.read_csv(odds_file)
    print(f"   ✅ {len(odds_df)} partidos cargados")
    
    # 2. Obtener predicciones
    print("\n🤖 Paso 2: Obtener predicciones del modelo...")
    predictions = get_predictions_from_model(odds_df)
    print(f"   ✅ {len(predictions)} predicciones obtenidas")
    
    # 3. Crear comparador
    print("\n⚙️  Paso 3: Configurar comparador...")
    comparator = OddsComparison(
        min_edge=0.03,      # Edge mínimo: 3%
        min_ev=0.10,        # EV mínimo: 10%
        min_confidence=0.50 # Confianza mínima: 50%
    )
    print(f"   ✅ Parámetros:")
    print(f"      • Min edge: {comparator.min_edge:.1%}")
    print(f"      • Min EV: {comparator.min_ev:.1%}")
    print(f"      • Min confidence: {comparator.min_confidence:.1%}")
    
    # 4. Comparar predicciones con odds
    print("\n🔍 Paso 4: Comparar predicciones con odds...")
    
    opportunities_list = []
    
    for i, (prediction, (_, odds_row)) in enumerate(zip(predictions, odds_df.iterrows())):
        odds = {
            'home_win_odds': odds_row['home_win_odds'],
            'draw_odds': odds_row['draw_odds'],
            'away_win_odds': odds_row['away_win_odds'],
        }
        
        opps = comparator.compare_prediction_with_odds(
            match_id=prediction['match_id'],
            date=prediction['date'],
            home_team=odds_row['home_team'],
            away_team=odds_row['away_team'],
            prediction=prediction,
            odds=odds
        )
        
        opportunities_list.extend(opps)
    
    print(f"   ✅ {len(opportunities_list)} oportunidades analizadas")
    
    # 5. Filtrar por recomendación
    print("\n🎯 Paso 5: Filtrar oportunidades...")
    
    df_opportunities = pd.DataFrame([opp.to_dict() for opp in opportunities_list])
    
    bet_count = len(df_opportunities[df_opportunities['recommendation'] == 'BET'])
    consider_count = len(df_opportunities[df_opportunities['recommendation'] == 'CONSIDER'])
    monitor_count = len(df_opportunities[df_opportunities['recommendation'] == 'MONITOR'])
    
    print(f"   ✅ Recomendaciones:")
    print(f"      • 🟢 BET: {bet_count} oportunidades")
    print(f"      • 🟡 CONSIDER: {consider_count} oportunidades")
    print(f"      • 🔵 MONITOR: {monitor_count} oportunidades")
    
    # 6. Mostrar mejores oportunidades
    print("\n💎 Paso 6: Top oportunidades (por EV)...")
    
    if bet_count > 0:
        top_bets = df_opportunities[df_opportunities['recommendation'] == 'BET'].nlargest(3, 'expected_value')
        
        for idx, bet in top_bets.iterrows():
            kelly = comparator.calculate_kelly_criterion(bet['model_probability'], bet['market_odds'])
            kelly_quarter = comparator.calculate_kelly_fraction(kelly, 0.25)
            
            print(f"\n   📌 {bet['home_team']} vs {bet['away_team']} ({bet['date']})")
            print(f"      Mercado: {bet['market']} a {bet['market_odds']:.2f}")
            print(f"      Modelo: {bet['model_probability']:.1%} vs Mercado: {bet['implied_probability']:.1%}")
            print(f"      Edge: {bet['value_percentage']:.2f}% | EV: {bet['expected_value']:.2%}")
            print(f"      Kelly 1/4: {kelly_quarter:.2%} (apuesta con bankroll de 1000€: {kelly_quarter*1000:.2f}€)")
    else:
        print("\n   ⚠️  No hay oportunidades BET en este análisis")
        if consider_count > 0:
            print(f"   💡 Considera bajar los umbrales para ver {consider_count} oportunidades CONSIDER")
    
    # 7. Guardar resultados
    print("\n💾 Paso 7: Guardar resultados...")
    
    output_dir = Path('data/processed')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_file = output_dir / 'integrated_opportunities.csv'
    df_opportunities.to_csv(csv_file, index=False)
    print(f"   ✅ Guardado: {csv_file}")
    
    # 8. Resumen estadístico
    print("\n📊 Paso 8: Resumen estadístico...")
    
    if len(df_opportunities) > 0:
        print(f"   EV promedio: {df_opportunities['expected_value'].mean():.2%}")
        print(f"   Edge promedio: {df_opportunities['edge'].mean():.2%}")
        print(f"   Confianza promedio: {df_opportunities['confidence_score'].mean():.1%}")
        print(f"   Mejores EV: {df_opportunities['expected_value'].max():.2%}")
        print(f"   Peor EV: {df_opportunities['expected_value'].min():.2%}")
    
    print("\n" + "="*120)
    print("✅ Integración completada\n")
    
    return df_opportunities


def show_integration_example():
    """
    Muestra cómo integrar el código en tu flujo
    """
    print("\n" + "="*120)
    print("📖 EJEMPLO DE INTEGRACIÓN EN TU CÓDIGO")
    print("="*120 + "\n")
    
    code_example = '''
# En tu script predict_match.py o similar:

from src.predictor import Predictor
from src.odds_comparison import OddsComparison
import pandas as pd

# 1. Cargar modelo y datos
predictor = Predictor()
matches_df = pd.read_csv('data/processed/upcoming_matches.csv')

# 2. Hacer predicciones
predictions = []
for _, match in matches_df.iterrows():
    pred = predictor.predict(match)
    predictions.append(pred)

# 3. Cargar odds
odds_df = pd.read_csv('data/processed/sample_odds.csv')

# 4. Comparar con mercado
comparator = OddsComparison(min_edge=0.03, min_ev=0.10)

opportunities = []
for prediction, (_, odds_row) in zip(predictions, odds_df.iterrows()):
    opps = comparator.compare_prediction_with_odds(
        match_id=prediction['match_id'],
        date=prediction['date'],
        home_team=odds_row['home_team'],
        away_team=odds_row['away_team'],
        prediction=prediction,
        odds={
            'home_win_odds': odds_row['home_win_odds'],
            'draw_odds': odds_row['draw_odds'],
            'away_win_odds': odds_row['away_win_odds'],
        }
    )
    opportunities.extend(opps)

# 5. Filtrar mejores oportunidades
best_opportunities = [o for o in opportunities if o.recommendation == 'BET']

# 6. Mostrar recomendaciones
for opp in sorted(best_opportunities, key=lambda x: x.expected_value, reverse=True):
    print(f"BET: {opp.home_team} vs {opp.away_team}")
    print(f"  {opp.market} a {opp.market_odds:.2f}")
    print(f"  EV: {opp.expected_value:.2%} | Edge: {opp.edge:.2%}")
    print(f"  Confianza: {opp.confidence_score:.1%}")
    print()
    '''
    
    print(code_example)


if __name__ == '__main__':
    # Ejecutar integración
    opportunities_df = integrate_model_with_odds()
    
    # Mostrar ejemplo de código
    show_integration_example()
    
    print("\n💡 PRÓXIMOS PASOS:")
    print("   1. Integrar con tu predictor real")
    print("   2. Automatizar actualización de odds")
    print("   3. Crear alertas para oportunidades")
    print("   4. Implementar tracking de apuestas")
    print("   5. Backtesting con datos históricos\n")
