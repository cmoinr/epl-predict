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

import os
import re
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
        python_cmd = [sys.executable or 'python', 'predict_match.py']
        env = os.environ.copy()
        env.setdefault('PYTHONIOENCODING', 'utf-8')
        env.setdefault('PYTHONUTF8', '1')
        result = subprocess.run(
            python_cmd + ['--home', home_team, '--away', away_team, '--date', date],
            capture_output=True,
            text=True,
            encoding='utf-8',
            timeout=30,
            env=env,
            cwd=str(Path(__file__).parent)
        )
        
        if result.returncode != 0:
            print(f"[WARN] Error ejecutando predict_match.py:")
            print(f"    STDOUT: {result.stdout}")
            print(f"    STDERR: {result.stderr}")
            return None
        
        # Parse the output to extract prediction data
        output = result.stdout
        prediction = parse_prediction_output(output, home_team, away_team)
        
        if prediction is None:
            print(f"[WARN] No se pudo parsear la prediccion para {home_team} vs {away_team}")
            print(f"    Output recibido:")
            print(f"    {output[:500]}")  # Primeros 500 caracteres
        
        return prediction
    except Exception as e:
        print(f"[WARN] Error prediciendo {home_team} vs {away_team}: {e}")
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

        prediction['goles_totales'] = {
            'prediccion': 2.5,
            'random_forest': None,
            'gradient_boosting': None,
            'promedio': None
        }
        
        prediction['ambos_anotan'] = {
            'random_forest': {'si': 0, 'no': 0},
            'gradient_boosting': {'si': 0, 'no': 0},
            'promedio': {'si': 0, 'no': 0}
        }
        
        current_model = None
        goals_section = False
        btts_section = False
        goals_pattern = re.compile(r':\s*([0-9]+(?:\.[0-9]+)?)')
        btts_pattern = re.compile(r'([0-9]+(?:\.[0-9]+)?)\s*%')
        
        for line in lines:
            line_stripped = line.strip()
            
            # Detectar modelos
            if 'Random Forest:' in line:
                current_model = 'random_forest'
            elif 'Gradient Boosting:' in line:
                current_model = 'gradient_boosting'

            if 'GOLES TOTALES' in line_stripped:
                goals_section = True
                btts_section = False
                continue
            
            if 'AMBOS ANOTAN' in line_stripped:
                btts_section = True
                goals_section = False
                continue

            if goals_section:
                if 'Random Forest:' in line_stripped:
                    match = goals_pattern.search(line_stripped)
                    if match:
                        prediction['goles_totales']['random_forest'] = float(match.group(1))
                    continue
                if 'Gradient Boosting:' in line_stripped:
                    match = goals_pattern.search(line_stripped)
                    if match:
                        prediction['goles_totales']['gradient_boosting'] = float(match.group(1))
                    continue
                if 'Promedio:' in line_stripped:
                    match = goals_pattern.search(line_stripped)
                    if match:
                        prediction['goles_totales']['promedio'] = float(match.group(1))
                    continue
            
            if btts_section:
                if 'Random Forest:' in line_stripped:
                    matches = btts_pattern.findall(line_stripped)
                    if len(matches) >= 2:
                        prediction['ambos_anotan']['random_forest']['si'] = float(matches[0])
                        prediction['ambos_anotan']['random_forest']['no'] = float(matches[1])
                    continue
                if 'Gradient Boosting:' in line_stripped:
                    matches = btts_pattern.findall(line_stripped)
                    if len(matches) >= 2:
                        prediction['ambos_anotan']['gradient_boosting']['si'] = float(matches[0])
                        prediction['ambos_anotan']['gradient_boosting']['no'] = float(matches[1])
                    continue
                if 'Promedio:' in line_stripped:
                    matches = btts_pattern.findall(line_stripped)
                    if len(matches) >= 2:
                        prediction['ambos_anotan']['promedio']['si'] = float(matches[0])
                        prediction['ambos_anotan']['promedio']['no'] = float(matches[1])
                    continue
            
            # Parsear línea con "Away X% | Draw X% | Home X%"
            if current_model and 'Away' in line_stripped and '%' in line_stripped and '|' in line_stripped and not btts_section:
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
        
        rf_goals = prediction['goles_totales']['random_forest']
        gb_goals = prediction['goles_totales']['gradient_boosting']
        promedio = prediction['goles_totales']['promedio']
        if promedio is None and rf_goals is not None and gb_goals is not None:
            promedio = round((rf_goals + gb_goals) / 2, 2)
            prediction['goles_totales']['promedio'] = promedio
        if promedio is not None:
            prediction['goles_totales']['prediccion'] = promedio

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


def get_btts_probs(prediction):
    """Extrae probabilidades de BTTS"""
    try:
        if 'ambos_anotan' in prediction:
            return prediction['ambos_anotan'].get('promedio', {'si': 50, 'no': 50})
        return {'si': 50, 'no': 50}
    except:
        return {'si': 50, 'no': 50}


def analyze_single_match(comparator, home_team, away_team, date, model_probs, 
                        total_goals, odds_row, btts_probs=None):
    """Analiza un partido individual"""
    
    odds = {
        'home_win_odds': odds_row['home_win_odds'],
        'draw_odds': odds_row['draw_odds'],
        'away_win_odds': odds_row['away_win_odds'],
        'over_2_5_odds': odds_row['over_2_5_odds'],
        'under_2_5_odds': odds_row['under_2_5_odds'],
        'both_score_yes': odds_row.get('both_score_yes', 0),
        'both_score_no': odds_row.get('both_score_no', 0),
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
            rec = "[BET]"
        elif edge > 0 and ev > 0.05:
            rec = "[CONSIDER]"
        elif edge > 0:
            rec = "[MONITOR]"
        else:
            rec = "[SKIP]"
        
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
    
    # Análisis BTTS
    btts_results = []
    if btts_probs and odds['both_score_yes'] > 0:
        # BTTS Yes
        btts_yes_prob = btts_probs['si'] / 100
        btts_yes_market = 1 / odds['both_score_yes']
        btts_yes_edge = btts_yes_prob - btts_yes_market
        btts_yes_ev = (btts_yes_prob * odds['both_score_yes']) - 1
        
        btts_yes_rec = "[BET]" if btts_yes_edge > 0.03 and btts_yes_ev > 0.10 else \
                       "[CONSIDER]" if btts_yes_edge > 0 and btts_yes_ev > 0.05 else \
                       "[MONITOR]" if btts_yes_edge > 0 else "[SKIP]"
        
        btts_results.append({
            'outcome': 'BTTS Yes',
            'model_prob': btts_yes_prob,
            'market_prob': btts_yes_market,
            'odds': odds['both_score_yes'],
            'edge': btts_yes_edge,
            'ev': btts_yes_ev,
            'rec': btts_yes_rec
        })
        
        # BTTS No
        btts_no_prob = btts_probs['no'] / 100
        btts_no_market = 1 / odds['both_score_no']
        btts_no_edge = btts_no_prob - btts_no_market
        btts_no_ev = (btts_no_prob * odds['both_score_no']) - 1
        
        btts_no_rec = "[BET]" if btts_no_edge > 0.03 and btts_no_ev > 0.10 else \
                      "[CONSIDER]" if btts_no_edge > 0 and btts_no_ev > 0.05 else \
                      "[MONITOR]" if btts_no_edge > 0 else "[SKIP]"
        
        btts_results.append({
            'outcome': 'BTTS No',
            'model_prob': btts_no_prob,
            'market_prob': btts_no_market,
            'odds': odds['both_score_no'],
            'edge': btts_no_edge,
            'ev': btts_no_ev,
            'rec': btts_no_rec
        })
    
    return results, over_model_prob, over_edge, over_ev, under_model_prob, under_edge, under_ev, odds, btts_results


def print_match_analysis(comparator, home_team, away_team, date, model_probs, 
                        total_goals, odds_row, match_num, total_matches, btts_probs=None):
    """Imprime análisis de un partido"""
    
    print("\n" + "="*70)
    print(f"[PARTIDO {match_num}/{total_matches}] {home_team} vs {away_team} ({date})")
    print("="*70)
    
    # Predicciones del modelo
    print(f"\nPREDICCIONES DEL MODELO:")
    print(f"   - {home_team}: {model_probs['Home Win']:.1%}")
    print(f"   - Draw: {model_probs['Draw']:.1%}")
    print(f"   - {away_team}: {model_probs['Away Win']:.1%}")
    print(f"   - Goles totales predichos: {total_goals:.1f}")
    if btts_probs:
        print(f"   - Ambos Anotan (BTTS): SI {btts_probs['si']:.1f}% | NO {btts_probs['no']:.1f}%")
    
    # Análisis
    results, over_prob, over_edge, over_ev, under_prob, under_edge, under_ev, odds, btts_results = \
        analyze_single_match(comparator, home_team, away_team, date, model_probs, 
                            total_goals, odds_row, btts_probs)
    
    print(f"\nANALISIS 1X2:")
    best_result = None
    for r in results:
        print(f"\n   {r['outcome']}:")
        print(f"      Cuota: {r['odds']:.2f} | Modelo: {r['model_prob']:.1%} vs Mercado: {r['market_prob']:.1%}")
        print(f"      Edge: {r['edge']:+.2%} | EV: {r['ev']:+.2%}")
        print(f"      {r['rec']}")
        
        if best_result is None or r['ev'] > best_result['ev']:
            best_result = r
    
    print(f"\nANALISIS GOLES (Over/Under 2.5):")
    print(f"\n   Over 2.5:")
    print(f"      Cuota: {odds['over_2_5_odds']:.2f} | Modelo: {over_prob:.1%} vs Mercado: {1/odds['over_2_5_odds']:.1%}")
    print(f"      Edge: {over_edge:+.2%} | EV: {over_ev:+.2%}")
    over_rec = "[BET]" if over_edge > 0.03 and over_ev > 0.10 else \
               "[CONSIDER]" if over_edge > 0 and over_ev > 0.05 else \
               "[MONITOR]" if over_edge > 0 else "[SKIP]"
    print(f"      {over_rec}")
    
    print(f"\n   Under 2.5:")
    print(f"      Cuota: {odds['under_2_5_odds']:.2f} | Modelo: {under_prob:.1%} vs Mercado: {1/odds['under_2_5_odds']:.1%}")
    print(f"      Edge: {under_edge:+.2%} | EV: {under_ev:+.2%}")
    under_rec = "[BET]" if under_edge > 0.03 and under_ev > 0.10 else \
                "[CONSIDER]" if under_edge > 0 and under_ev > 0.05 else \
                "[MONITOR]" if under_edge > 0 else "[SKIP]"
    print(f"      {under_rec}")
    
    if btts_results:
        print(f"\nANALISIS AMBOS ANOTAN (BTTS):")
        for r in btts_results:
            print(f"\n   {r['outcome']}:")
            print(f"      Cuota: {r['odds']:.2f} | Modelo: {r['model_prob']:.1%} vs Mercado: {r['market_prob']:.1%}")
            print(f"      Edge: {r['edge']:+.2%} | EV: {r['ev']:+.2%}")
            print(f"      {r['rec']}")
            
            if r['ev'] > best_result['ev']:
                best_result = r

    # Mejor oportunidad
    if best_result and best_result['rec'].startswith('[BET]'):
        kelly = comparator.calculate_kelly_criterion(best_result['model_prob'], best_result['odds'])
        kelly_quarter = comparator.calculate_kelly_fraction(kelly, 0.25)
        
        print(f"\nMEJOR OPORTUNIDAD: {best_result['outcome']} a {best_result['odds']:.2f}")
        print(f"   Edge: {best_result['edge']:+.2%} | EV: {best_result['ev']:+.2%}")
        print(f"   Kelly 1/4 recomendado: {kelly_quarter:.2%}")
        print(f"   Con 1000$: Apuesta = {kelly_quarter*1000:.2f}$ | Ganancia esperada = {kelly_quarter*1000*best_result['ev']:.2f}$")


def main():
    print("ANALISIS INTEGRADO: PREDICCION + COMPARATIVA DE ODDS")
    
    try:
        odds_df = load_odds_and_matches()
    except FileNotFoundError:
        print("[ERROR] Archivo sample_odds.csv no encontrado")
        print("   Asegurate de que existe: data/processed/sample_odds.csv")
        return
    
    # Inicializar comparador
    comparator = OddsComparison(min_edge=0.03, min_ev=0.10, min_confidence=0.50)
    
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
        btts_probs = get_btts_probs(prediction)
        
        # Analizar
        print_match_analysis(comparator, home_team, away_team, date, model_probs, 
                           total_goals, row, idx, len(odds_df), btts_probs)
        
        # Guardar si es BET
        results, _, _, _, _, _, _, odds, btts_results = analyze_single_match(
            comparator, home_team, away_team, date, model_probs, total_goals, row, btts_probs)
        
        for r in results + btts_results:
            if r['rec'].startswith('[BET]'):
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
    
    print("\n" + "="*70)
    print("[OK] Analisis completado")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
