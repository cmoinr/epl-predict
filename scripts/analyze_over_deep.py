"""
Análisis PROFUNDO de OVER 2.5 - Verificar si hay rangos rentables
"""
import pandas as pd

df = pd.read_csv('data/processed/quick_ou_analysis.csv')

print('='*70)
print('ANALISIS PROFUNDO DE OVER 2.5')
print('='*70)

df_over = df[df['market'] == 'Over 2.5']
df_under = df[df['market'] == 'Under 2.5']

print(f'\nDATOS GENERALES:')
print(f'  Over 2.5:  {len(df_over)} bets | WR: {df_over["won"].mean()*100:.1f}% | ROI: {df_over["profit_1u"].mean()*100:.2f}%')
print(f'  Under 2.5: {len(df_under)} bets | WR: {df_under["won"].mean()*100:.1f}% | ROI: {df_under["profit_1u"].mean()*100:.2f}%')

print('\n' + '='*70)
print('OVER 2.5 - BUSQUEDA DE RANGOS RENTABLES')
print('='*70)

# Análisis granular por edge
print('\n1. Por EDGE:')
edge_ranges = [(0, 0.03), (0.03, 0.05), (0.05, 0.08), (0.08, 0.10), (0.10, 0.15), (0.15, 0.20), (0.20, 0.30)]
for min_e, max_e in edge_ranges:
    subset = df_over[(df_over['edge'] >= min_e) & (df_over['edge'] < max_e)]
    if len(subset) >= 3:
        roi = subset['profit_1u'].mean() * 100
        marker = '✓' if roi > 5 else '✗'
        print(f'  {marker} Edge {min_e*100:4.1f}-{max_e*100:4.1f}%: {len(subset):2d} bets | WR: {subset["won"].mean()*100:5.1f}% | ROI: {roi:7.2f}%')

# Análisis granular por cuotas
print('\n2. Por CUOTAS:')
odds_ranges = [(1.0, 1.4), (1.4, 1.6), (1.6, 1.8), (1.8, 2.0), (2.0, 2.5), (2.5, 3.0)]
for min_o, max_o in odds_ranges:
    subset = df_over[(df_over['odds'] >= min_o) & (df_over['odds'] < max_o)]
    if len(subset) >= 3:
        roi = subset['profit_1u'].mean() * 100
        marker = '✓' if roi > 5 else '✗'
        print(f'  {marker} Odds {min_o:.1f}-{max_o:.1f}: {len(subset):2d} bets | WR: {subset["won"].mean()*100:5.1f}% | ROI: {roi:7.2f}%')

# Análisis por probabilidad
print('\n3. Por PROBABILIDAD MODELO:')
prob_ranges = [(0.50, 0.60), (0.60, 0.65), (0.65, 0.70), (0.70, 0.75), (0.75, 0.80), (0.80, 1.0)]
for min_p, max_p in prob_ranges:
    subset = df_over[(df_over['model_prob'] >= min_p) & (df_over['model_prob'] < max_p)]
    if len(subset) >= 3:
        roi = subset['profit_1u'].mean() * 100
        marker = '✓' if roi > 5 else '✗'
        print(f'  {marker} Prob {min_p*100:4.0f}-{max_p*100:4.0f}%: {len(subset):2d} bets | WR: {subset["won"].mean()*100:5.1f}% | ROI: {roi:7.2f}%')

# Buscar combinaciones rentables
print('\n' + '='*70)
print('COMBINACIONES RENTABLES DE OVER 2.5 (ROI > 10%):')
print('='*70)

rentables = []
for min_e, max_e in edge_ranges:
    for min_o, max_o in odds_ranges:
        for min_p, max_p in prob_ranges:
            subset = df_over[
                (df_over['edge'] >= min_e) & (df_over['edge'] < max_e) &
                (df_over['odds'] >= min_o) & (df_over['odds'] < max_o) &
                (df_over['model_prob'] >= min_p) & (df_over['model_prob'] < max_p)
            ]
            if len(subset) >= 3:
                roi = subset['profit_1u'].mean() * 100
                if roi > 10:
                    rentables.append({
                        'edge': f'{min_e*100:.1f}-{max_e*100:.1f}%',
                        'odds': f'{min_o:.1f}-{max_o:.1f}',
                        'prob': f'{min_p*100:.0f}-{max_p*100:.0f}%',
                        'n': len(subset),
                        'wr': subset['won'].mean() * 100,
                        'roi': roi
                    })

if rentables:
    df_rent = pd.DataFrame(rentables).sort_values('roi', ascending=False)
    print('\n✓ SE ENCONTRARON COMBINACIONES RENTABLES:')
    for _, r in df_rent.iterrows():
        print(f'  Edge {r["edge"]:9s} | Odds {r["odds"]:9s} | Prob {r["prob"]:9s} | {r["n"]:2.0f} bets | WR: {r["wr"]:5.1f}% | ROI: {r["roi"]:6.2f}%')
    
    print('\n✓ RECOMENDACION: Incluir OVER 2.5 con estos filtros específicos')
else:
    print('\n✗ NO se encontraron combinaciones rentables con ROI > 10%')
    print('✗ CONFIRMAR: SKIP OVER 2.5')

print('\n' + '='*70)
print('COMPARACION FINAL:')
print('='*70)

if rentables:
    best_over = df_rent.iloc[0]
    best_under = df_under[
        (df_under['edge'] >= 0.05) & (df_under['edge'] < 0.30) &
        (df_under['odds'] >= 2.4) & (df_under['odds'] < 3.5) &
        (df_under['model_prob'] >= 0.40) & (df_under['model_prob'] < 0.70)
    ]
    
    print(f'\nMEJOR OVER 2.5:')
    print(f'  {best_over["n"]:.0f} bets | WR: {best_over["wr"]:.1f}% | ROI: {best_over["roi"]:.2f}%')
    print(f'  Filtros: Edge {best_over["edge"]}, Odds {best_over["odds"]}, Prob {best_over["prob"]}')
    
    if len(best_under) > 0:
        print(f'\nMEJOR UNDER 2.5:')
        print(f'  {len(best_under)} bets | WR: {best_under["won"].mean()*100:.1f}% | ROI: {best_under["profit_1u"].mean()*100:.2f}%')
        print(f'  Filtros: Edge 5-30%, Odds 2.4-3.5, Prob 40-70%')
    
    print(f'\nCONCLUSION: Ambos mercados tienen rangos rentables')
else:
    print(f'\nOVER 2.5: No rentable (ROI {df_over["profit_1u"].mean()*100:.2f}%)')
    print(f'UNDER 2.5: Rentable (ROI {df_under["profit_1u"].mean()*100:.2f}%)')
    print(f'\nCONCLUSION: Enfocarse SOLO en UNDER 2.5')
