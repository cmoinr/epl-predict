#!/usr/bin/env python
"""
Script para obtener SOLO las recomendaciones de VALUE BETS precisas
Lee partidos futuros de sample_odds_history.csv y muestra únicamente las apuestas [BET]

Uso:
    python get_value_bets.py
"""

import os
import sys
import subprocess
import re
from pathlib import Path
import pandas as pd
from src.odds_comparison import OddsComparison


def get_model_predictions(home_team, away_team, date):
    """Obtiene predicciones del modelo"""
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
            return None
        
        return parse_prediction_output(result.stdout, home_team, away_team)
    except Exception as e:
        return None


def parse_prediction_output(output, home_team, away_team):
    """Parsea el output del modelo"""
    try:
        lines = output.split('\n')
        prediction = {
            'home_team': home_team,
            'away_team': away_team,
            'resultado': {
                'mejor_modelo': {'probabilidades': {'Home Win': 0, 'Draw': 0, 'Away Win': 0}}
            },
            'goles_totales': {'prediccion': 2.5, 'voting_ensemble': None},
            'ambos_anotan': {'mejor_modelo': {'si': 0, 'no': 0}}
        }
        
        current_model = None
        btts_section = False
        goals_section = False
        goals_pattern = re.compile(r':\s*([0-9]+(?:\.[0-9]+)?)')
        btts_pattern = re.compile(r'([0-9]+(?:\.[0-9]+)?)\s*%')
        
        for line in lines:
            line_stripped = line.strip()
            
            # Detectar secciones
            if 'Phase 2 Voting Ensemble' in line and 'Precisión:' in line:
                current_model = 'mejor_modelo'
            elif 'GOLES TOTALES' in line_stripped:
                goals_section = True
                btts_section = False
                continue
            elif 'AMBOS ANOTAN' in line_stripped:
                btts_section = True
                goals_section = False
                continue
            
            # Parsear goles (Voting Ensemble)
            if goals_section and 'Voting Ensemble' in line_stripped and 'MAE:' in line_stripped:
                parts = line_stripped.split(':')
                if len(parts) >= 3:
                    try:
                        pred_val = float(parts[-1].strip())
                        prediction['goles_totales']['prediccion'] = pred_val
                        prediction['goles_totales']['voting_ensemble'] = pred_val
                    except:
                        pass
            
            # Parsear BTTS (XGBoost)
            if btts_section and 'XGBoost' in line_stripped and 'Precisión:' in line_stripped:
                matches = btts_pattern.findall(line_stripped)
                if len(matches) >= 2:
                    prediction['ambos_anotan']['mejor_modelo']['si'] = float(matches[0])
                    prediction['ambos_anotan']['mejor_modelo']['no'] = float(matches[1])
            
            # Parsear probabilidades resultado (formato: Away X% | Draw X% | Home X%)
            if current_model and 'Away' in line_stripped and '%' in line_stripped and '|' in line_stripped and not btts_section:
                try:
                    content = line_stripped.replace('Detalles:', '').strip()
                    parts = content.split('|')
                    if len(parts) >= 3:
                        away_val = float(parts[0].strip().split()[-1].rstrip('%'))
                        draw_val = float(parts[1].strip().split()[-1].rstrip('%'))
                        home_val = float(parts[2].strip().split()[-1].rstrip('%'))
                        
                        prediction['resultado'][current_model]['probabilidades'] = {
                            'Away Win': away_val,
                            'Draw': draw_val,
                            'Home Win': home_val
                        }
                except:
                    pass
        
        # Validar que tenemos datos
        mejor_probs = prediction['resultado']['mejor_modelo']['probabilidades']
        if sum(mejor_probs.values()) > 0:
            return prediction
        
        return None
    except:
        return None


def extract_probabilities(prediction):
    """Extrae probabilidades del resultado"""
    try:
        mejor_probs = prediction['resultado'].get('mejor_modelo', {}).get('probabilidades', {})
        if sum(mejor_probs.values()) > 0:
            return {
                'Home Win': mejor_probs.get('Home Win', 0) / 100,
                'Draw': mejor_probs.get('Draw', 0) / 100,
                'Away Win': mejor_probs.get('Away Win', 0) / 100
            }
        return None
    except:
        return None


def get_total_goals(prediction):
    """Extrae goles totales predichos"""
    try:
        voting = prediction['goles_totales'].get('voting_ensemble')
        if voting is not None:
            return voting
        return prediction['goles_totales'].get('prediccion', 2.5)
    except:
        return 2.5


def get_btts_probs(prediction):
    """Extrae probabilidades BTTS"""
    try:
        mejor = prediction['ambos_anotan'].get('mejor_modelo')
        if mejor and mejor.get('si', 0) > 0:
            return mejor
        return {'si': 50, 'no': 50}
    except:
        return {'si': 50, 'no': 50}


def find_value_bets(home_team, away_team, date, model_probs, total_goals, odds_row, btts_probs):
    """
    Encuentra SOLO las apuestas [BET] con los filtros Ultra V2 y Optimizados
    Retorna lista de value bets encontrados
    """
    value_bets = []
    
    odds = {
        'home_win_odds': odds_row['home_win_odds'],
        'draw_odds': odds_row['draw_odds'],
        'away_win_odds': odds_row['away_win_odds'],
        'over_2_5_odds': odds_row['over_2_5_odds'],
        'under_2_5_odds': odds_row['under_2_5_odds'],
        'both_score_yes': odds_row.get('both_score_yes', 0),
        'both_score_no': odds_row.get('both_score_no', 0),
    }
    
    # === ANÁLISIS 1X2 - FILTROS ULTRA V2 ===
    for outcome in ['Home Win', 'Draw', 'Away Win']:
        model_prob = model_probs[outcome]
        market_prob = 1 / odds[f"{outcome.lower().replace(' ', '_')}_odds"]
        odds_val = odds[f"{outcome.lower().replace(' ', '_')}_odds"]
        
        edge = model_prob - market_prob
        ev = (model_prob * odds_val) - 1
        
        is_bet = False
        
        if outcome == 'Away Win':
            # AWAY: Cuotas 2.5-4.0, Edge 10%-22%, Prob 40%-60%
            if (2.5 <= odds_val <= 4.0 and 
                0.10 <= edge <= 0.22 and
                0.40 <= model_prob <= 0.60):
                is_bet = True
                
        elif outcome == 'Home Win':
            # HOME: Cuotas 2.5-3.0, Edge 18%-22%, Prob 45%-60%
            if (2.5 <= odds_val <= 3.0 and
                0.18 <= edge <= 0.22 and
                0.45 <= model_prob <= 0.60):
                is_bet = True
                
        elif outcome == 'Draw':
            # DRAW: Cuotas 3.0-4.0, Edge 12%-15%, Prob 25%-35%
            if (3.0 <= odds_val <= 4.0 and
                0.12 <= edge <= 0.15 and
                0.25 <= model_prob <= 0.35):
                is_bet = True
        
        if is_bet:
            value_bets.append({
                'type': '1X2',
                'selection': outcome,
                'odds': odds_val,
                'model_prob': model_prob,
                'edge': edge,
                'ev': ev
            })
    
    # === ANÁLISIS OVER 2.5 - FILTROS OPTIMIZADOS ===
    over_model_prob = 0.75 if total_goals >= 3.5 else \
                      0.65 if total_goals >= 3.0 else \
                      0.55 if total_goals >= 2.5 else \
                      0.35 if total_goals >= 2.0 else 0.20
    
    over_edge = over_model_prob - (1 / odds['over_2_5_odds'])
    over_ev = (over_model_prob * odds['over_2_5_odds']) - 1
    
    # [BET] - ROI 48.53%
    over_bet = (
        ((over_edge >= 0.15) and (over_edge < 0.20) and 
         (odds['over_2_5_odds'] >= 1.8) and (odds['over_2_5_odds'] < 2.0) and 
         (over_model_prob >= 0.75) and (over_model_prob < 0.80)) or
        ((over_edge >= 0.08) and (over_edge < 0.10) and 
         (odds['over_2_5_odds'] >= 1.8) and (odds['over_2_5_odds'] < 2.0) and 
         (over_model_prob >= 0.65) and (over_model_prob < 0.70))
    )
    
    if over_bet:
        value_bets.append({
            'type': 'O/U',
            'selection': 'Over 2.5',
            'odds': odds['over_2_5_odds'],
            'model_prob': over_model_prob,
            'edge': over_edge,
            'ev': over_ev
        })
    
    # === ANÁLISIS UNDER 2.5 - FILTROS OPTIMIZADOS ===
    under_model_prob = 1 - over_model_prob
    under_edge = under_model_prob - (1 / odds['under_2_5_odds'])
    under_ev = (under_model_prob * odds['under_2_5_odds']) - 1
    
    # [BET] - ROI 79%
    under_bet = (
        (under_edge >= 0.03) and (under_edge < 0.05) and 
        (odds['under_2_5_odds'] >= 2.4) and (odds['under_2_5_odds'] < 3.0) and 
        (under_model_prob >= 0.40) and (under_model_prob < 0.50)
    )
    
    if under_bet:
        value_bets.append({
            'type': 'O/U',
            'selection': 'Under 2.5',
            'odds': odds['under_2_5_odds'],
            'model_prob': under_model_prob,
            'edge': under_edge,
            'ev': under_ev
        })
    
    # === ANÁLISIS BTTS - FILTROS BASE ===
    if btts_probs and odds['both_score_yes'] > 0:
        # BTTS Yes
        btts_yes_prob = btts_probs['si'] / 100
        btts_yes_edge = btts_yes_prob - (1 / odds['both_score_yes'])
        btts_yes_ev = (btts_yes_prob * odds['both_score_yes']) - 1
        
        if btts_yes_edge > 0.03 and btts_yes_ev > 0.10:
            value_bets.append({
                'type': 'BTTS',
                'selection': 'Yes',
                'odds': odds['both_score_yes'],
                'model_prob': btts_yes_prob,
                'edge': btts_yes_edge,
                'ev': btts_yes_ev
            })
        
        # BTTS No
        btts_no_prob = btts_probs['no'] / 100
        btts_no_edge = btts_no_prob - (1 / odds['both_score_no'])
        btts_no_ev = (btts_no_prob * odds['both_score_no']) - 1
        
        if btts_no_edge > 0.03 and btts_no_ev > 0.10:
            value_bets.append({
                'type': 'BTTS',
                'selection': 'No',
                'odds': odds['both_score_no'],
                'model_prob': btts_no_prob,
                'edge': btts_no_edge,
                'ev': btts_no_ev
            })
    
    return value_bets


def main():
    print("=" * 80)
    print(" VALUE BETS - RECOMENDACIONES PRECISAS CON FILTROS OPTIMIZADOS")
    print("=" * 80)
    print()
    
    # Cargar partidos futuros
    odds_file = 'data/processed/sample_odds_history.csv'
    try:
        odds_df = pd.read_csv(odds_file)
    except FileNotFoundError:
        print(f"[ERROR] Archivo no encontrado: {odds_file}")
        return
    
    # Inicializar comparador (para Kelly)
    comparator = OddsComparison(min_edge=0.03, min_ev=0.10, min_confidence=0.50)
    
    # Procesar cada partido
    all_value_bets = []
    
    for idx, row in odds_df.iterrows():
        home_team = row['home_team']
        away_team = row['away_team']
        date = row['date']
        
        # Obtener predicciones
        prediction = get_model_predictions(home_team, away_team, date)
        if prediction is None:
            continue
        
        model_probs = extract_probabilities(prediction)
        if model_probs is None:
            continue
        
        total_goals = get_total_goals(prediction)
        btts_probs = get_btts_probs(prediction)
        
        # Buscar value bets
        value_bets = find_value_bets(home_team, away_team, date, model_probs, 
                                     total_goals, row, btts_probs)
        
        # Guardar si hay value bets
        if value_bets:
            for bet in value_bets:
                kelly = comparator.calculate_kelly_criterion(bet['model_prob'], bet['odds'])
                kelly_quarter = comparator.calculate_kelly_fraction(kelly, 0.25)
                
                bet['match'] = f"{home_team} vs {away_team}"
                bet['date'] = date
                bet['kelly_quarter'] = kelly_quarter
                all_value_bets.append(bet)
    
    # Mostrar resultados compactos
    if not all_value_bets:
        print("No se encontraron VALUE BETS con los filtros optimizados.")
        print()
        return
    
    # Agrupar por partido
    matches_dict = {}
    for bet in all_value_bets:
        key = (bet['match'], bet['date'])
        if key not in matches_dict:
            matches_dict[key] = []
        matches_dict[key].append(bet)
    
    # Imprimir de forma compacta
    print(f"{'PARTIDO':<40} {'FECHA':<12} {'MERCADO':<8} {'SELECCIÓN':<12} {'CUOTA':<6} {'PROB':<6} {'EDGE':<7} {'EV':<7} {'KELLY':<7}")
    print("=" * 120)
    
    for (match, date), bets in matches_dict.items():
        for i, bet in enumerate(bets):
            # Mostrar partido solo en la primera línea de cada grupo
            match_str = match if i == 0 else ""
            date_str = date if i == 0 else ""
            
            print(f"{match_str:<40} {date_str:<12} {bet['type']:<8} {bet['selection']:<12} "
                  f"{bet['odds']:<6.2f} {bet['model_prob']:<6.1%} {bet['edge']:<7.1%} "
                  f"{bet['ev']:<7.1%} {bet['kelly_quarter']:<7.2%}")
        print("-" * 120)
    
    # Resumen
    print()
    print("=" * 80)
    print(f"RESUMEN: {len(all_value_bets)} VALUE BETS encontradas en {len(matches_dict)} partidos")
    print("=" * 80)
    print()
    print("FILTROS APLICADOS:")
    print("  • 1X2 (Ultra V2): Rangos específicos por tipo (Away: 2.5-4.0, Home: 2.5-3.0, Draw: 3.0-4.0)")
    print("  • O/U 2.5 (Optimizado): Over ROI 48.53%, Under ROI 79%")
    print("  • BTTS: Edge >3%, EV >10%")
    print()
    
    # Cálculo con bankroll de ejemplo
    bankroll = 1000
    print(f"EJEMPLO CON BANKROLL ${bankroll}:")
    total_stake = sum(bet['kelly_quarter'] * bankroll for bet in all_value_bets)
    total_ev_dollars = sum((bet['kelly_quarter'] * bankroll) * bet['ev'] for bet in all_value_bets)
    
    print(f"  • Total a apostar (todas las recomendaciones): ${total_stake:.2f}")
    print(f"  • EV total esperado: ${total_ev_dollars:.2f}")
    print(f"  • ROI promedio: {(total_ev_dollars/total_stake)*100:.1f}%")
    print()


if __name__ == '__main__':
    main()
