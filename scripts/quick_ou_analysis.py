"""
Análisis RÁPIDO de O/U 2.5 - Versión optimizada
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / 'src'))
from predictor import EPLPredictor

print("="*70)
print("ANALISIS RAPIDO O/U 2.5")
print("="*70)

# Cargar datos
df_odds = pd.read_csv('data/raw/epl_odds.csv', low_memory=False)
df_odds['Date'] = pd.to_datetime(df_odds['Date'], format='%d/%m/%y', errors='coerce')
df_odds = df_odds.dropna(subset=['Date'])
df_odds = df_odds[(df_odds['Avg>2.5'].notna()) & (df_odds['Avg<2.5'].notna())].copy()
# Usar TODOS los partidos disponibles
print(f"\nPartidos totales con O/U disponibles: {len(df_odds)}")

df_hist = pd.read_csv('data/raw/epl_final.csv')
df_hist['MatchDate'] = pd.to_datetime(df_hist['MatchDate'])

print(f"Analizando TODOS los partidos disponibles para mayor precision")
print(f"Cargando modelo...")

predictor = EPLPredictor('models')

results = []
processed = 0
errors = 0

print(f"\nProcesando (1 de cada 2 para balance rapidez/precision)...")

for idx, row in df_odds.iterrows():
    # Procesar 1 de cada 2 partidos para más datos
    if idx % 2 != 0:
        continue
    
    processed += 1
    if processed % 10 == 0:
        print(f"  {processed}...")
    
    try:
        df_until = df_hist[df_hist['MatchDate'] < row['Date']]
        if len(df_until) < 50:
            continue
        
        pred = predictor.predict_match(df_until, row['HomeTeam'], row['AwayTeam'], str(row['Date']))
        
        # Obtener goles predichos
        if 'mejor_modelo' in pred['goles_totales'] and 'prediccion' in pred['goles_totales']['mejor_modelo']:
            predicted_goals = pred['goles_totales']['mejor_modelo']['prediccion']
        elif 'promedio' in pred['goles_totales']:
            predicted_goals = pred['goles_totales']['promedio']
        else:
            predicted_goals = 2.5
        
        # Probabilidades basadas en goles predichos
        if predicted_goals >= 3.5:
            prob_over = 0.75
        elif predicted_goals >= 3.0:
            prob_over = 0.65
        elif predicted_goals >= 2.5:
            prob_over = 0.55
        elif predicted_goals >= 2.0:
            prob_over = 0.35
        else:
            prob_over = 0.20
        
        prob_under = 1 - prob_over
        
        odds_over = row['Avg>2.5']
        odds_under = row['Avg<2.5']
        
        edge_over = prob_over - (1 / odds_over)
        edge_under = prob_under - (1 / odds_under)
        
        total_goals = row['FTHG'] + row['FTAG']
        actual_over = total_goals > 2.5
        
        # Registrar ambos
        results.append({
            'market': 'Over 2.5',
            'edge': edge_over,
            'odds': odds_over,
            'model_prob': prob_over,
            'predicted_goals': predicted_goals,
            'actual_goals': total_goals,
            'won': actual_over,
            'profit_1u': (odds_over - 1) if actual_over else -1
        })
        
        results.append({
            'market': 'Under 2.5',
            'edge': edge_under,
            'odds': odds_under,
            'model_prob': prob_under,
            'predicted_goals': predicted_goals,
            'actual_goals': total_goals,
            'won': not actual_over,
            'profit_1u': (odds_under - 1) if not actual_over else -1
        })
        
    except Exception as e:
        errors += 1
        if errors < 5:
            print(f"  Error: {str(e)[:50]}")

print(f"\nResultados: {len(results)} predicciones")
print(f"Errores: {errors}")

if len(results) == 0:
    print("No se generaron resultados")
    sys.exit(1)

df = pd.DataFrame(results)
df_pos = df[df['edge'] > 0].copy()

print(f"\nEdge positivo: {len(df_pos)} ({len(df_pos)/len(df)*100:.1f}%)")

if len(df_pos) == 0:
    print("No hay oportunidades con edge positivo")
    sys.exit(1)

print(f"Win rate (edge+): {df_pos['won'].mean()*100:.1f}%")
print(f"ROI (edge+): {df_pos['profit_1u'].mean()*100:.2f}%")

# Análisis por rangos de edge
print("\n" + "="*70)
print("ANALISIS POR EDGE:")
print("="*70)
edge_bins = [(0, 0.05), (0.05, 0.10), (0.10, 0.15), (0.15, 0.20), (0.20, 0.30)]

for min_e, max_e in edge_bins:
    subset = df_pos[(df_pos['edge'] >= min_e) & (df_pos['edge'] < max_e)]
    if len(subset) > 3:
        wr = subset['won'].mean() * 100
        roi = subset['profit_1u'].mean() * 100
        avg_odds = subset['odds'].mean()
        print(f"Edge {min_e*100:2.0f}-{max_e*100:2.0f}%: {len(subset):3d} bets | WR: {wr:5.1f}% | ROI: {roi:6.2f}% | Odds: {avg_odds:.2f}")

# Análisis por cuotas
print("\n" + "="*70)
print("ANALISIS POR CUOTAS:")
print("="*70)
odds_bins = [(1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 4.0)]

for min_o, max_o in odds_bins:
    subset = df_pos[(df_pos['odds'] >= min_o) & (df_pos['odds'] < max_o)]
    if len(subset) > 3:
        wr = subset['won'].mean() * 100
        roi = subset['profit_1u'].mean() * 100
        avg_edge = subset['edge'].mean() * 100
        print(f"Odds {min_o:.1f}-{max_o:.1f}: {len(subset):3d} bets | WR: {wr:5.1f}% | ROI: {roi:6.2f}% | Edge: {avg_edge:.1f}%")

# Análisis por probabilidad
print("\n" + "="*70)
print("ANALISIS POR PROBABILIDAD MODELO:")
print("="*70)
prob_bins = [(0.0, 0.40), (0.40, 0.50), (0.50, 0.60), (0.60, 0.70), (0.70, 1.0)]

for min_p, max_p in prob_bins:
    subset = df_pos[(df_pos['model_prob'] >= min_p) & (df_pos['model_prob'] < max_p)]
    if len(subset) > 3:
        wr = subset['won'].mean() * 100
        roi = subset['profit_1u'].mean() * 100
        avg_edge = subset['edge'].mean() * 100
        print(f"Prob {min_p*100:2.0f}-{max_p*100:2.0f}%: {len(subset):3d} bets | WR: {wr:5.1f}% | ROI: {roi:6.2f}% | Edge: {avg_edge:.1f}%")

# Buscar mejores combinaciones (ROI > 10%)
print("\n" + "="*70)
print("MEJORES COMBINACIONES (ROI > 10%):")
print("="*70)

best = []
for min_e, max_e in edge_bins:
    for min_o, max_o in odds_bins:
        for min_p, max_p in prob_bins:
            subset = df_pos[
                (df_pos['edge'] >= min_e) & (df_pos['edge'] < max_e) &
                (df_pos['odds'] >= min_o) & (df_pos['odds'] < max_o) &
                (df_pos['model_prob'] >= min_p) & (df_pos['model_prob'] < max_p)
            ]
            if len(subset) >= 5:
                roi = subset['profit_1u'].mean() * 100
                if roi > 10:
                    best.append({
                        'edge': f"{min_e*100:.0f}-{max_e*100:.0f}%",
                        'odds': f"{min_o:.1f}-{max_o:.1f}",
                        'prob': f"{min_p*100:.0f}-{max_p*100:.0f}%",
                        'n': len(subset),
                        'wr': subset['won'].mean() * 100,
                        'roi': roi
                    })

if best:
    df_best = pd.DataFrame(best).sort_values('roi', ascending=False).head(10)
    for _, r in df_best.iterrows():
        print(f"Edge {r['edge']:8s} | Odds {r['odds']:8s} | Prob {r['prob']:8s} | {r['n']:2.0f} bets | WR: {r['wr']:5.1f}% | ROI: {r['roi']:6.2f}%")
else:
    print("No se encontraron combinaciones con ROI > 10%")

# Guardar
df_pos.to_csv('data/processed/quick_ou_analysis.csv', index=False)
print(f"\nGuardado en: data/processed/quick_ou_analysis.csv")
