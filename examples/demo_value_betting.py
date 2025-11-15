#!/usr/bin/env python3
"""
Ejemplo Interactivo - Demostración de Value Betting
Muestra cómo funciona el sistema con un ejemplo paso a paso
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / 'src'))

from odds_manager import OddsManager
from odds_comparison import OddsComparison


def print_header(title):
    """Imprimir encabezado formateado"""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def main():
    print_header("💰 DEMOSTRACIÓN - VALUE BETTING CON ODDS")
    
    # ===========================================================================
    # PARTE 1: Gestión de Odds
    # ===========================================================================
    print_header("PARTE 1: Gestión de Odds")
    
    manager = OddsManager()
    
    print("📊 Cargando dataset de odds de ejemplo...")
    df_odds = manager.load_historical_odds('data/processed/sample_odds.csv')
    print(f"   ✅ {len(df_odds)} partidos cargados\n")
    
    # Mostrar primeros partidos
    print("📋 Primeros partidos en la base de datos:")
    for idx, row in df_odds.head(3).iterrows():
        print(f"   {row['home_team']} vs {row['away_team']} ({row['date']})")
        print(f"      Home: {row['home_win_odds']:.2f} | Draw: {row['draw_odds']:.2f} | Away: {row['away_win_odds']:.2f}")
    
    # ===========================================================================
    # PARTE 2: Análisis de Odds
    # ===========================================================================
    print_header("PARTE 2: Análisis de Odds de un Partido")
    
    # Seleccionar un partido
    match = df_odds.iloc[0]
    print(f"🎯 Analizando: {match['home_team']} vs {match['away_team']}\n")
    
    # Cuotas
    h_odds = match['home_win_odds']
    d_odds = match['draw_odds']
    a_odds = match['away_win_odds']
    
    print(f"Cuotas (decimales):")
    print(f"  Home Win: {h_odds:.2f}")
    print(f"  Draw:     {d_odds:.2f}")
    print(f"  Away Win: {a_odds:.2f}\n")
    
    # Probabilidades implícitas
    h_prob = manager.calculate_implied_probability(h_odds)
    d_prob = manager.calculate_implied_probability(d_odds)
    a_prob = manager.calculate_implied_probability(a_odds)
    
    print(f"Probabilidades Implícitas (lo que asigna el mercado):")
    print(f"  Home Win: {h_prob:.1%}")
    print(f"  Draw:     {d_prob:.1%}")
    print(f"  Away Win: {a_prob:.1%}")
    print(f"  TOTAL:    {h_prob + d_prob + a_prob:.1%}  ⚠️  (> 100% = margen de casa)\n")
    
    # Margen de casa
    margin = manager.calculate_bookmaker_margin(h_odds, d_odds, a_odds)
    print(f"💰 Margen de casa: {margin:.2%}")
    print(f"   (La casa se queda con esto independientemente del resultado)\n")
    
    # Sharp odds
    sharp = manager.calculate_sharp_odds(h_odds, d_odds, a_odds)
    print(f"🎯 Sharp Odds (sin margen - cuotas 'reales'):")
    print(f"  Home Win: {sharp['home']:.2f} (prob: {1/sharp['home']:.1%})")
    print(f"  Draw:     {sharp['draw']:.2f} (prob: {1/sharp['draw']:.1%})")
    print(f"  Away Win: {sharp['away']:.2f} (prob: {1/sharp['away']:.1%})")
    print(f"  TOTAL:    {1/sharp['home'] + 1/sharp['draw'] + 1/sharp['away']:.1%}")
    
    # ===========================================================================
    # PARTE 3: Comparación con Modelo ML
    # ===========================================================================
    print_header("PARTE 3: Comparación Predicción ML vs Odds del Mercado")
    
    comparator = OddsComparison(min_edge=0.02, min_ev=0.05)
    
    # Simular predicción del modelo
    print("🤖 Supongamos que nuestro modelo predice para este partido:\n")
    
    model_predictions = {
        'Home Win': 0.45,
        'Draw': 0.28,
        'Away Win': 0.27
    }
    
    print("   Predicción del Modelo ML:")
    for outcome, prob in model_predictions.items():
        print(f"     {outcome}: {prob:.1%}")
    print()
    
    # Analizar cada mercado
    markets_odds = {
        'Home Win': h_odds,
        'Draw': d_odds,
        'Away Win': a_odds
    }
    
    print("📊 Análisis de Edge (Ventaja vs Mercado):\n")
    print(f"{'Mercado':<15} {'Cuota':<8} {'Prob Mod':<12} {'Prob Merc':<12} {'Edge':<10} {'EV':<10} {'Recom':<15}")
    print("-" * 90)
    
    opportunities = []
    for outcome, model_prob in model_predictions.items():
        market_odds = markets_odds[outcome]
        market_prob = manager.calculate_implied_probability(market_odds)
        
        edge = comparator.calculate_edge(model_prob, market_odds)
        ev = comparator.calculate_expected_value(model_prob, market_odds)
        
        # Determinar recomendación
        if edge >= 0.03 and ev >= 0.10:
            recom = "✅ BET"
        elif edge >= 0.02:
            recom = "⚠️  CONSIDER"
        elif edge >= 0:
            recom = "👀 MONITOR"
        else:
            recom = "❌ SKIP"
        
        print(f"{outcome:<15} {market_odds:<8.2f} {model_prob:<12.1%} {market_prob:<12.1%} {edge:>+8.2%} {ev:>+8.2%} {recom:<15}")
        
        opportunities.append({
            'outcome': outcome,
            'model_prob': model_prob,
            'market_odds': market_odds,
            'market_prob': market_prob,
            'edge': edge,
            'ev': ev
        })
    
    # ===========================================================================
    # PARTE 4: Kelly Criterion - Cuánto Apostar
    # ===========================================================================
    print_header("PARTE 4: Kelly Criterion - Gestión de Riesgo")
    
    best_opp = max(opportunities, key=lambda x: x['ev'])
    
    print(f"🎯 Mejor oportunidad: {best_opp['outcome']}\n")
    print(f"   Cuota: {best_opp['market_odds']:.2f}")
    print(f"   Probabilidad predicha: {best_opp['model_prob']:.1%}")
    print(f"   Edge vs mercado: {best_opp['edge']:+.2%}")
    print(f"   Valor esperado: {best_opp['ev']:+.2%}\n")
    
    kelly = comparator.calculate_kelly_criterion(best_opp['model_prob'], best_opp['market_odds'])
    kelly_quarter = comparator.calculate_kelly_fraction(kelly, 0.25)
    
    print(f"💡 Criterio de Kelly (gestión óptima de bankroll):\n")
    print(f"   Kelly Criterion (100%): {kelly:.2%} del bankroll")
    print(f"   Kelly 1/2 (moderado):   {kelly * 0.5:.2%} del bankroll")
    print(f"   Kelly 1/4 (conservador):{kelly_quarter:.2%} del bankroll  ← RECOMENDADO\n")
    
    print(f"   Interpretación:")
    print(f"   - Bankroll total: $1000")
    print(f"   - Apostar 100% Kelly: ${kelly * 1000:.2f}")
    print(f"   - Apostar 1/4 Kelly: ${kelly_quarter * 1000:.2f}  ← MÁS SEGURO\n")
    
    print(f"   Retorno esperado (apostando ${kelly_quarter * 100:.2f} de $100):")
    expected_return = (kelly_quarter * 100) * best_opp['model_prob'] * best_opp['market_odds']
    print(f"   - Si ganas: ${expected_return:.2f}")
    print(f"   - Si pierdes: -${kelly_quarter * 100:.2f}")
    
    # ===========================================================================
    # PARTE 5: Simulación de ROI
    # ===========================================================================
    print_header("PARTE 5: Simulación de ROI (Rentabilidad)")
    
    print("📈 Escenario: Apostar con la estrategia Kelly 1/4\n")
    
    # Simular apuestas en todas las oportunidades
    bankroll = 1000
    total_bet = 0
    total_returns = 0
    winning_bets = 0
    
    print(f"Simulando {len(opportunities)} apuestas:\n")
    print(f"{'#':<3} {'Mercado':<15} {'Apuesta':<10} {'Retorno Esp.':<15} {'Estado':<10}")
    print("-" * 60)
    
    for i, opp in enumerate(opportunities, 1):
        kelly = comparator.calculate_kelly_criterion(opp['model_prob'], opp['market_odds'])
        bet_amount = kelly * 0.25 * bankroll  # Kelly 1/4
        
        # Retorno esperado
        ev = opp['ev']
        expected_return = bet_amount * ev
        
        status = "✅ +EV" if ev > 0 else "❌ -EV"
        
        print(f"{i:<3} {opp['outcome']:<15} ${bet_amount:<9.2f} ${expected_return:<14.2f} {status}")
        
        total_bet += bet_amount
        total_returns += expected_return
        if ev > 0:
            winning_bets += 1
    
    print("-" * 60)
    roi = (total_returns / total_bet * 100) if total_bet > 0 else 0
    print(f"{'TOTAL':<3} {'Apuesta total':<15} ${total_bet:<9.2f} ${total_returns:<14.2f} ROI: {roi:+.1f}%\n")
    
    print(f"📊 Análisis:")
    print(f"   - Apuestas ganadoras (+EV): {winning_bets}/{len(opportunities)}")
    print(f"   - Tasa de acierto esperada: {winning_bets/len(opportunities):.1%}")
    print(f"   - Retorno esperado anual (si repites esto 52 veces): {total_returns * 52:.0f}$ (~{roi * 52:.0f}%)\n")
    
    # ===========================================================================
    # CONCLUSIÓN
    # ===========================================================================
    print_header("🎓 CONCLUSIÓN")
    
    print("""
Este sistema te permite:

1️⃣  IDENTIFICAR valor: Encontrar oportunidades donde tu modelo es mejor que el mercado
2️⃣  CUANTIFICAR edge: Calcular exactamente cuánta ventaja tienes
3️⃣  GESTIONAR riesgo: Kelly Criterion para optimizar tamaño de apuestas
4️⃣  PROYECTAR ROI: Estimar rentabilidad esperada

Principios clave:
  ✓ Solo apostar cuando EV > 0 (valor esperado positivo)
  ✓ Usar Kelly 1/4 (conservador) en lugar de Kelly completo
  ✓ Validar modelo en datos que NO vio durante entrenamiento
  ✓ Mantener tracking detallado de apuestas reales
  ✓ El éxito requiere volumen (100+ apuestas mínimo)

⚠️  IMPORTANTE: Las apuestas reales siempre tienen riesgo. Este es un sistema educativo.
    """)
    
    print_header("FIN DE LA DEMOSTRACIÓN")


if __name__ == '__main__':
    main()
