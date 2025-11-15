#!/usr/bin/env python
"""
Script integrado: Predicción + Análisis de Odds
Flujo completo en un solo comando

Uso:
    python run_analysis.py
    
Los datos se leen de:
    - data/processed/sample_odds.csv (contiene partidos y odds)
    
El script automáticamente:
    1. Carga los partidos del CSV
    2. Ejecuta el modelo para cada partido
    3. Compara con las odds
    4. Muestra análisis detallado
"""

import sys
import subprocess
import json
from pathlib import Path
import pandas as pd
from src.odds_comparison import OddsComparison


def load_odds_and_matches(odds_file='data/processed/sample_odds.csv'):
    """Carga los partidos y odds del CSV"""
    odds_df = pd.read_csv(odds_file)
    return odds_df


def get_model_predictions(home_team, away_team, date):
    """Obtiene predicciones del modelo ejecutando predict_match.py"""
    try:
        result = subprocess.run(
            ['python', 'predict_match.py', '--home', home_team, '--away', away_team, 
             '--date', date],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            return None
        
        # Parse the output to extract prediction data
        output = result.stdout
        prediction = parse_prediction_output(output, home_team, away_team)
        return prediction
    except Exception as e:
        print(f"⚠️  Error prediciendo {home_team} vs {away_team}: {e}")
        return None


def parse_prediction_output(output, home_team, away_team):
    """Parsea el output de predict_match.py - busca lineas con 'Detalles:' o 'Away X% |'"""
    try:
        lines = output.split('\n')
        prediction = {
            'home_team': home_team,
            'away_team': away_team,
            'resultado': {
                'random_forest': {'probabilidades': {'Home Win': 0, 'Draw': 0, 'Away Win': 0}},
                'gradient_boosting': {'probabilidades': {'Home Win': 0, 'Draw': 0, 'Away Win': 0}}
            }
        }
        
        current_model = None
        
        for line in lines:
            line_stripped = line.strip()
            
            # Detectar modelos
            if '🌲 Random Forest:' in line:
                current_model = 'random_forest'
            elif '⚡ Gradient Boosting:' in line:
                current_model = 'gradient_boosting'
            
            # Parsear línea con "Away X% | Draw X% | Home X%"
            if current_model and 'Away' in line_stripped and '%' in line_stripped and '|' in line_stripped:
                try:
                    # Formato: "Detalles: Away 76.9% | Draw 15.6% | Home 7.5%"
                    # Remover "Detalles: " si existe
                    content = line_stripped.replace('Detalles:', '').strip()
                    
                    parts = content.split('|')
                    if len(parts) >= 3:
                        away_str = parts[0].strip()  # "Away 76.9%"
                        draw_str = parts[1].strip()  # "Draw 15.6%"
                        home_str = parts[2].strip()  # "Home 7.5%"
                        
                        away_val = float(away_str.split()[-1].rstrip('%'))
                        draw_val = float(draw_str.split()[-1].rstrip('%'))
                        home_val = float(home_str.split()[-1].rstrip('%'))
                        
                        prediction['resultado'][current_model]['probabilidades'] = {
                            'Away Win': away_val,
                            'Draw': draw_val,
                            'Home Win': home_val
                        }
                except Exception as e:
                    pass
        
        # Validar que tenemos datos válidos
        rf_probs = prediction['resultado']['random_forest']['probabilidades']
        gb_probs = prediction['resultado']['gradient_boosting']['probabilidades']
        
        if sum(rf_probs.values()) > 0 or sum(gb_probs.values()) > 0:
            return prediction
        
        return None
    except Exception as e:
        return None


def extract_probabilities(prediction):
    """Extrae probabilidades del resultado de predicción"""
    try:
        if 'resultado' in prediction:
            rf_probs = prediction['resultado'].get('random_forest', {}).get('probabilidades', {})
            gb_probs = prediction['resultado'].get('gradient_boosting', {}).get('probabilidades', {})
            
            home_win = ((rf_probs.get('Home Win', 0) + gb_probs.get('Home Win', 0)) / 200)
            draw = ((rf_probs.get('Draw', 0) + gb_probs.get('Draw', 0)) / 200)
            away_win = ((rf_probs.get('Away Win', 0) + gb_probs.get('Away Win', 0)) / 200)
            
            return {
                'Home Win': home_win,
                'Draw': draw,
                'Away Win': away_win,
                'rf_home': rf_probs.get('Home Win', 0) / 100,
                'rf_draw': rf_probs.get('Draw', 0) / 100,
                'rf_away': rf_probs.get('Away Win', 0) / 100,
                'gb_home': gb_probs.get('Home Win', 0) / 100,
                'gb_draw': gb_probs.get('Draw', 0) / 100,
                'gb_away': gb_probs.get('Away Win', 0) / 100,
            }
        return None
    except Exception as e:
        print(f"Error extrayendo probabilidades: {e}")
        return None


def get_total_goals(prediction):
    """Extrae goles totales predichos"""
    try:
        if 'goles_totales' in prediction:
            return prediction['goles_totales'].get('prediccion', 2.5)
        return 2.5
    except:
        return 2.5


def analyze_single_match(comparator, home_team, away_team, date, model_probs, 
                        total_goals, odds_row):
    """Analiza un partido individual"""
    
    odds = {
        'home_win_odds': odds_row['home_win_odds'],
        'draw_odds': odds_row['draw_odds'],
        'away_win_odds': odds_row['away_win_odds'],
        'over_2_5_odds': odds_row['over_2_5_odds'],
        'under_2_5_odds': odds_row['under_2_5_odds'],
    }
    
    market_probs = {
        'Home Win': 1 / odds['home_win_odds'],
        'Draw': 1 / odds['draw_odds'],
        'Away Win': 1 / odds['away_win_odds'],
    }
    
    # Análisis 1X2
    results = []
    for outcome in ['Home Win', 'Draw', 'Away Win']:
        model_prob = model_probs[outcome]
        market_prob = market_probs[outcome]
        odds_val = odds[f"{outcome.lower().replace(' ', '_')}_odds"]
        
        edge = model_prob - market_prob
        ev = (model_prob * odds_val) - 1
        
        if edge > 0.03 and ev > 0.10:
            rec = "🟢 BET"
        elif edge > 0 and ev > 0.05:
            rec = "🟡 CONSIDER"
        elif edge > 0:
            rec = "🔵 MONITOR"
        else:
            rec = "❌ SKIP"
        
        results.append({
            'outcome': outcome,
            'model_prob': model_prob,
            'market_prob': market_prob,
            'odds': odds_val,
            'edge': edge,
            'ev': ev,
            'rec': rec
        })
    
    # Análisis Over/Under
    over_market_prob = 1 / odds['over_2_5_odds']
    under_market_prob = 1 / odds['under_2_5_odds']
    
    if total_goals >= 3.5:
        over_model_prob = 0.75
    elif total_goals >= 3.0:
        over_model_prob = 0.65
    elif total_goals >= 2.5:
        over_model_prob = 0.55
    elif total_goals >= 2.0:
        over_model_prob = 0.35
    else:
        over_model_prob = 0.20
    
    under_model_prob = 1 - over_model_prob
    
    over_edge = over_model_prob - over_market_prob
    under_edge = under_model_prob - under_market_prob
    
    over_ev = (over_model_prob * odds['over_2_5_odds']) - 1
    under_ev = (under_model_prob * odds['under_2_5_odds']) - 1
    
    return results, over_model_prob, over_edge, over_ev, under_model_prob, under_edge, under_ev, odds


def print_match_analysis(comparator, home_team, away_team, date, model_probs, 
                        total_goals, odds_row, match_num, total_matches):
    """Imprime análisis de un partido"""
    
    print("\n" + "="*120)
    print(f"📌 PARTIDO {match_num}/{total_matches}: {home_team} vs {away_team} ({date})")
    print("="*120)
    
    # Predicciones del modelo
    print(f"\n📊 PREDICCIONES DEL MODELO:")
    print(f"   • {home_team}: {model_probs['Home Win']:.1%}")
    print(f"   • Draw: {model_probs['Draw']:.1%}")
    print(f"   • {away_team}: {model_probs['Away Win']:.1%}")
    print(f"   • Goles totales predichos: {total_goals:.1f}")
    
    # Análisis
    results, over_prob, over_edge, over_ev, under_prob, under_edge, under_ev, odds = \
        analyze_single_match(comparator, home_team, away_team, date, model_probs, 
                            total_goals, odds_row)
    
    print(f"\n✨ ANÁLISIS 1X2:")
    best_result = None
    for r in results:
        print(f"\n   {r['outcome']}:")
        print(f"      Cuota: {r['odds']:.2f} | Modelo: {r['model_prob']:.1%} vs Mercado: {r['market_prob']:.1%}")
        print(f"      Edge: {r['edge']:+.2%} | EV: {r['ev']:+.2%}")
        print(f"      {r['rec']}")
        
        if best_result is None or r['ev'] > best_result['ev']:
            best_result = r
    
    print(f"\n⚽ ANÁLISIS GOLES (Over/Under 2.5):")
    print(f"\n   Over 2.5:")
    print(f"      Cuota: {odds['over_2_5_odds']:.2f} | Modelo: {over_prob:.1%} vs Mercado: {1/odds['over_2_5_odds']:.1%}")
    print(f"      Edge: {over_edge:+.2%} | EV: {over_ev:+.2%}")
    over_rec = "🟢 BET" if over_edge > 0.03 and over_ev > 0.10 else \
               "🟡 CONSIDER" if over_edge > 0 and over_ev > 0.05 else \
               "🔵 MONITOR" if over_edge > 0 else "❌ SKIP"
    print(f"      {over_rec}")
    
    print(f"\n   Under 2.5:")
    print(f"      Cuota: {odds['under_2_5_odds']:.2f} | Modelo: {under_prob:.1%} vs Mercado: {1/odds['under_2_5_odds']:.1%}")
    print(f"      Edge: {under_edge:+.2%} | EV: {under_ev:+.2%}")
    under_rec = "🟢 BET" if under_edge > 0.03 and under_ev > 0.10 else \
                "🟡 CONSIDER" if under_edge > 0 and under_ev > 0.05 else \
                "🔵 MONITOR" if under_edge > 0 else "❌ SKIP"
    print(f"      {under_rec}")
    
    # Mejor oportunidad
    if best_result and best_result['rec'].startswith('🟢'):
        kelly = comparator.calculate_kelly_criterion(best_result['model_prob'], best_result['odds'])
        kelly_quarter = comparator.calculate_kelly_fraction(kelly, 0.25)
        
        print(f"\n💎 MEJOR OPORTUNIDAD: {best_result['outcome']} a {best_result['odds']:.2f}")
        print(f"   Edge: {best_result['edge']:+.2%} | EV: {best_result['ev']:+.2%}")
        print(f"   Kelly 1/4 recomendado: {kelly_quarter:.2%}")
        print(f"   Con 1000€: Apuesta = {kelly_quarter*1000:.2f}€ | Ganancia esperada = {kelly_quarter*1000*best_result['ev']:.2f}€")


def main():
    print("\n" + "🚀"*50)
    print("ANÁLISIS INTEGRADO: PREDICCIÓN + COMPARATIVA DE ODDS")
    print("🚀"*50)
    
    # Cargar datos
    print("\n📥 Cargando partidos y odds...")
    try:
        odds_df = load_odds_and_matches()
        print(f"✅ {len(odds_df)} partidos cargados")
    except FileNotFoundError:
        print("❌ Error: Archivo sample_odds.csv no encontrado")
        print("   Asegúrate de que existe: data/processed/sample_odds.csv")
        return
    
    # Inicializar comparador
    print("⚙️  Inicializando comparador...")
    comparator = OddsComparison(min_edge=0.03, min_ev=0.10, min_confidence=0.50)
    print("✅ Comparador configurado")
    
    print("🤖 Cargando modelo (será ejecutado para cada partido)...")
    print("✅ Modelo listo\n")
    
    # Procesar cada partido
    print(f"🔍 Analizando {len(odds_df)} partido(s)...\n")
    
    all_bets = []
    
    for idx, (_, row) in enumerate(odds_df.iterrows(), 1):
        home_team = row['home_team']
        away_team = row['away_team']
        date = row['date']
        
        # Obtener predicciones
        prediction = get_model_predictions(home_team, away_team, date)
        if prediction is None:
            continue
        
        # Extraer datos
        model_probs = extract_probabilities(prediction)
        if model_probs is None:
            continue
        
        total_goals = get_total_goals(prediction)
        
        # Analizar
        print_match_analysis(comparator, home_team, away_team, date, model_probs, 
                           total_goals, row, idx, len(odds_df))
        
        # Guardar si es BET
        results, _, _, _, _, _, _, odds = analyze_single_match(
            comparator, home_team, away_team, date, model_probs, total_goals, row)
        
        for r in results:
            if r['rec'].startswith('🟢'):
                kelly = comparator.calculate_kelly_criterion(r['model_prob'], r['odds'])
                kelly_quarter = comparator.calculate_kelly_fraction(kelly, 0.25)
                all_bets.append({
                    'match': f"{home_team} vs {away_team}",
                    'date': date,
                    'outcome': r['outcome'],
                    'odds': r['odds'],
                    'ev': r['ev'],
                    'edge': r['edge'],
                    'kelly_quarter': kelly_quarter
                })
    
    # Resumen final
    if all_bets:
        print("\n" + "="*120)
        print("📊 RESUMEN: APUESTAS RECOMENDADAS (BET)")
        print("="*120)
        
        bets_df = pd.DataFrame(all_bets)
        bets_df = bets_df.sort_values('ev', ascending=False)
        
        print(f"\nTotal de apuestas BET: {len(bets_df)}\n")
        
        total_ev = 0
        total_kelly = 0
        
        for idx, bet in bets_df.iterrows():
            print(f"{idx+1}. {bet['match']} ({bet['date']})")
            print(f"   {bet['outcome']} a {bet['odds']:.2f}")
            print(f"   Edge: {bet['edge']:+.2%} | EV: {bet['ev']:+.2%}")
            print(f"   Kelly 1/4: {bet['kelly_quarter']:.2%} → Apuesta: {bet['kelly_quarter']*1000:.2f}€")
            print()
            
            total_ev += bet['ev']
            total_kelly += bet['kelly_quarter']
        
        print(f"💰 TOTALES (con 1000€ de bankroll por apuesta):")
        print(f"   • EV promedio: {total_ev/len(bets_df):.2%}")
        print(f"   • Kelly promedio: {total_kelly/len(bets_df):.2%}")
        print(f"   • Inversión total (1/4 Kelly): {(total_kelly/len(bets_df))*1000*len(bets_df):.2f}€")
        print(f"   • Ganancia esperada: {(total_kelly/len(bets_df))*1000*len(bets_df)*(total_ev/len(bets_df)):.2f}€")
    else:
        print("\n" + "="*120)
        print("⚠️  NO HAY APUESTAS RECOMENDADAS (BET)")
        print("="*120)
        print("\nTodas las oportunidades están por debajo del umbral de rentabilidad.")
    
    print("\n" + "="*120)
    print("✅ Análisis completado")
    print("="*120 + "\n")


if __name__ == '__main__':
    main()
