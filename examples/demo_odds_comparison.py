#!/usr/bin/env python
"""
Demostración práctica: Comparar predicciones del modelo con odds del mercado

Este script muestra cómo:
1. Cargar predicciones del modelo
2. Cargar odds del mercado
3. Identificar oportunidades de value betting
4. Calcular métricas de inversión (Kelly Criterion)
"""

import pandas as pd
from pathlib import Path
from src.odds_comparison import OddsComparison


def demo_single_match():
    """
    Ejemplo 1: Comparar un partido específico
    """
    print("\n" + "="*100)
    print("📊 EJEMPLO 1: COMPARAR UN PARTIDO ESPECÍFICO")
    print("="*100)
    
    comparator = OddsComparison()
    
    # Predicción del modelo para Manchester City vs Arsenal
    prediction = {
        'resultado': {
            'random_forest': {
                'probabilidades': {
                    'Home Win': 55,
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
    
    # Odds del mercado
    odds = {
        'home_win_odds': 1.65,
        'draw_odds': 4.20,
        'away_win_odds': 5.50
    }
    
    # Comparar
    opportunities = comparator.compare_prediction_with_odds(
        match_id='MC_ARS_001',
        date='2024-12-07',
        home_team='Manchester City',
        away_team='Arsenal',
        prediction=prediction,
        odds=odds
    )
    
    print(f"\n🎯 Manchester City vs Arsenal (2024-12-07)\n")
    
    for opp in opportunities:
        print(f"   Resultado: {opp.market}")
        print(f"   • Cuota: {opp.market_odds:.2f}")
        print(f"   • Prob. Modelo: {opp.model_probability:.1%}")
        print(f"   • Prob. Mercado: {opp.implied_probability:.1%}")
        print(f"   • Edge: {opp.value_percentage:.2f}%")
        print(f"   • EV: {opp.expected_value:.2%}")
        print(f"   • Confianza: {opp.confidence_score:.1%}")
        print(f"   • Recomendación: {opp.recommendation}\n")


def demo_kelly_criterion():
    """
    Ejemplo 2: Calcular Kelly Criterion para gestión de bankroll
    """
    print("\n" + "="*100)
    print("💰 EJEMPLO 2: KELLY CRITERION (Tamaño óptimo de apuesta)")
    print("="*100)
    
    comparator = OddsComparison()
    
    print("\n📌 Escenario: Tienes 1000€ en tu bankroll")
    print("\nApuesta sobre Arsenal (Away Win) a 5.50\n")
    
    model_prob = 0.25  # 25% según modelo
    market_odds = 5.50
    
    kelly = comparator.calculate_kelly_criterion(model_prob, market_odds)
    kelly_quarter = comparator.calculate_kelly_fraction(kelly, 0.25)
    kelly_half = comparator.calculate_kelly_fraction(kelly, 0.5)
    
    bankroll = 1000
    
    print(f"   Full Kelly: {kelly:.2%} → Apuesta: {kelly * bankroll:.2f}€")
    print(f"   1/4 Kelly:  {kelly_quarter:.2%} → Apuesta: {kelly_quarter * bankroll:.2f}€ ✅ RECOMENDADO")
    print(f"   1/2 Kelly:  {kelly_half:.2%} → Apuesta: {kelly_half * bankroll:.2f}€")
    
    print(f"\n   💡 El 1/4 Kelly es más conservador y reduce volatilidad")
    
    # Calcular retornos esperados
    print(f"\n   📈 Resultados esperados con apuesta de {kelly_quarter * bankroll:.2f}€:")
    apuesta = kelly_quarter * bankroll
    ev = comparator.calculate_expected_value(model_prob, market_odds)
    ganancia_esperada = apuesta * ev
    
    print(f"      • Si gana: +{apuesta * (market_odds - 1):.2f}€")
    print(f"      • Si pierde: -{apuesta:.2f}€")
    print(f"      • Valor esperado: {ev:.2%} (ganancias: {ganancia_esperada:.2f}€)")


def demo_value_vs_market():
    """
    Ejemplo 3: Value betting - Encontrar discrepancias entre modelo y mercado
    """
    print("\n" + "="*100)
    print("🎯 EJEMPLO 3: VALUE BETTING - Encontrar oportunidades")
    print("="*100)
    
    comparator = OddsComparison(min_edge=0.03, min_ev=0.10)
    
    print("""
¿QUÉ ES VALUE BETTING?
   
   El mercado piensa: "Draw al 29.4% (cuota 3.40)"
   Nuestro modelo piensa: "Draw al 35% (cuota 2.86)"
   
   Si nuestro modelo tiene razón, la cuota 3.40 es una "ganga"
   → Eso se llama VALUE
   
PASOS PARA IDENTIFICAR VALOR:
   
   1. Calcular probabilidad implícita = 1 / cuota
   2. Comparar con predicción del modelo
   3. Si modelo > mercado → VALOR POSITIVO
   4. Calcular Expected Value (EV) = (modelo * cuota) - 1
   5. Si EV > 0 → Rentable a largo plazo
""")
    
    print("\n" + "-"*100)
    print("Ejemplo concreto:")
    print("-"*100 + "\n")
    
    scenarios = [
        {
            'match': 'Chelsea vs Liverpool',
            'result': 'Liverpool Win',
            'market_odds': 2.50,
            'model_prob': 0.42,
            'status': 'VALOR POSITIVO ✅'
        },
        {
            'match': 'Manchester City vs Arsenal',
            'result': 'Home Win',
            'market_odds': 1.65,
            'model_prob': 0.56,
            'status': 'OVERPRICED ❌'
        }
    ]
    
    for scenario in scenarios:
        implied = 1 / scenario['market_odds']
        edge = scenario['model_prob'] - implied
        ev = (scenario['model_prob'] * scenario['market_odds']) - 1
        
        print(f"📌 {scenario['match']} - {scenario['result']}")
        print(f"   Cuota: {scenario['market_odds']:.2f}")
        print(f"   Prob. Mercado: {implied:.1%}")
        print(f"   Prob. Modelo:  {scenario['model_prob']:.1%}")
        print(f"   Edge: {edge:+.2%}")
        print(f"   EV: {ev:+.2%}")
        print(f"   → {scenario['status']}\n")


def demo_market_consensus():
    """
    Ejemplo 4: Ver lo que el mercado espera
    """
    print("\n" + "="*100)
    print("📊 EJEMPLO 4: CONSENSO DEL MERCADO")
    print("="*100)
    
    # Cargar odds
    odds_file = Path('data/processed/sample_odds.csv')
    if odds_file.exists():
        odds_df = pd.read_csv(odds_file)
        
        print("\n🎯 ¿Qué espera el mercado? (Primeros 5 partidos)\n")
        
        for idx, row in odds_df.head(5).iterrows():
            home_prob = 1 / row['home_win_odds']
            draw_prob = 1 / row['draw_odds']
            away_prob = 1 / row['away_win_odds']
            
            probs = {'Home': home_prob, 'Draw': draw_prob, 'Away': away_prob}
            favorite = max(probs, key=probs.get)
            
            print(f"   {row['home_team']} vs {row['away_team']}")
            print(f"      • Home: {home_prob:.1%} (cuota: {row['home_win_odds']:.2f})")
            print(f"      • Draw: {draw_prob:.1%} (cuota: {row['draw_odds']:.2f})")
            print(f"      • Away: {away_prob:.1%} (cuota: {row['away_win_odds']:.2f})")
            print(f"      • Favorito: {favorite} ({probs[favorite]:.1%})")
            print(f"      • Over 2.5: {1/row['over_2_5_odds']:.1%} | Under 2.5: {1/row['under_2_5_odds']:.1%}")
            print()


def main():
    """
    Ejecutar todas las demostraciones
    """
    print("\n" + "🚀" * 50)
    print("DEMOSTRACIÓN: COMPARAR PREDICCIONES vs ODDS DEL MERCADO")
    print("🚀" * 50)
    
    demo_value_vs_market()
    demo_single_match()
    demo_kelly_criterion()
    demo_market_consensus()
    
    print("\n" + "="*100)
    print("✅ Resumen de conceptos clave:")
    print("="*100)
    print("""
    1. EDGE: Diferencia entre predicción del modelo y probabilidad del mercado
       → Edge > 0 = Oportunidad de valor
    
    2. EXPECTED VALUE (EV): Ganancia/pérdida esperada por cada unidad apostada
       → EV > 0 = Rentable a largo plazo
    
    3. KELLY CRITERION: Calcula el tamaño óptimo de apuesta
       → Maximiza ganancias a largo plazo
       → 1/4 Kelly es más conservador y recomendado
    
    4. VALUE BETTING: Encontrar oportunidades donde el modelo tiene ventaja
       → Comparar predicciones con cuotas del mercado
       → Apostar cuando hay edge positivo
    
    5. MANAGE YOUR BANKROLL: Usar Kelly Criterion o fracciones menores
       → Reducir volatilidad
       → Proteger capital
""")
    print("="*100 + "\n")


if __name__ == '__main__':
    main()
