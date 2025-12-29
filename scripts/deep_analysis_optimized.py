"""
Análisis profundo del backtest optimizado para encontrar filtros más precisos
"""

import pandas as pd
import numpy as np
from pathlib import Path

def deep_analysis():
    """Análisis granular del backtest optimizado"""
    
    df = pd.read_csv('data/processed/backtest_results_optimized.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    print("\n" + "="*80)
    print("🔬 ANÁLISIS PROFUNDO - BUSCANDO PATRONES RENTABLES")
    print("="*80)
    
    # 1. Análisis por combinación de tipo + edge + cuota
    print("\n💎 MEJORES COMBINACIONES (mínimo 10 apuestas):")
    print("-"*80)
    
    # Crear categorías más granulares
    df['edge_cat'] = pd.cut(df['edge'], bins=[0, 0.12, 0.15, 0.18, 0.22, 1.0], 
                             labels=['10-12%', '12-15%', '15-18%', '18-22%', '22%+'])
    df['odds_cat'] = pd.cut(df['odds'], bins=[1.0, 2.0, 2.5, 3.0, 4.0, 10.0],
                             labels=['1.5-2.0', '2.0-2.5', '2.5-3.0', '3.0-4.0', '4.0+'])
    
    best_combos = []
    
    for bet_type in df['bet_type'].unique():
        for edge_cat in df['edge_cat'].unique():
            for odds_cat in df['odds_cat'].unique():
                subset = df[
                    (df['bet_type'] == bet_type) & 
                    (df['edge_cat'] == edge_cat) & 
                    (df['odds_cat'] == odds_cat)
                ]
                
                if len(subset) >= 10:  # Mínimo 10 apuestas para ser significativo
                    win_rate = subset['won'].sum() / len(subset)
                    profit = subset['profit'].sum()
                    stake = subset['stake'].sum()
                    roi = (profit / stake) * 100
                    
                    best_combos.append({
                        'type': bet_type,
                        'edge': edge_cat,
                        'odds': odds_cat,
                        'count': len(subset),
                        'win_rate': win_rate,
                        'profit': profit,
                        'roi': roi
                    })
    
    # Ordenar por ROI
    best_combos_df = pd.DataFrame(best_combos).sort_values('roi', ascending=False)
    
    print("\nTOP 10 COMBINACIONES MÁS RENTABLES:")
    for idx, row in best_combos_df.head(10).iterrows():
        print(f"  {row['type']:5s} | Edge {row['edge']:8s} | Cuota {row['odds']:8s} | "
              f"{row['count']:3d} apuestas | Win Rate: {row['win_rate']*100:5.1f}% | "
              f"ROI: {row['roi']:7.2f}% | Profit: ${row['profit']:8,.2f}")
    
    print("\nPEORES 5 COMBINACIONES (para evitar):")
    for idx, row in best_combos_df.tail(5).iterrows():
        print(f"  {row['type']:5s} | Edge {row['edge']:8s} | Cuota {row['odds']:8s} | "
              f"{row['count']:3d} apuestas | Win Rate: {row['win_rate']*100:5.1f}% | "
              f"ROI: {row['roi']:7.2f}% | Profit: ${row['profit']:8,.2f}")
    
    # 2. Análisis de probabilidad del modelo vs resultado
    print("\n🎯 CALIBRACIÓN DEL MODELO:")
    print("-"*80)
    
    # Dividir por rangos de probabilidad del modelo
    prob_bins = [0, 0.30, 0.40, 0.50, 0.60, 1.0]
    prob_labels = ['<30%', '30-40%', '40-50%', '50-60%', '60%+']
    df['model_prob_cat'] = pd.cut(df['model_prob'], bins=prob_bins, labels=prob_labels)
    
    print("\nRENDIMIENTO POR PROBABILIDAD DEL MODELO:")
    for prob_cat in prob_labels:
        subset = df[df['model_prob_cat'] == prob_cat]
        if len(subset) > 0:
            win_rate = subset['won'].sum() / len(subset)
            profit = subset['profit'].sum()
            roi = (profit / subset['stake'].sum()) * 100 if subset['stake'].sum() > 0 else 0
            avg_edge = subset['edge'].mean()
            
            print(f"  Prob {prob_cat:8s}: {len(subset):3d} apuestas | "
                  f"Win Rate Real: {win_rate*100:5.1f}% | "
                  f"Avg Edge: {avg_edge*100:5.1f}% | "
                  f"ROI: {roi:7.2f}%")
    
    # 3. Análisis de Away Wins (son muy rentables)
    print("\n⚡ ANÁLISIS DETALLADO: AWAY WINS")
    print("-"*80)
    
    away_df = df[df['bet_type'] == 'Away'].copy()
    print(f"Total Away Wins: {len(away_df)} apuestas")
    print(f"Win Rate: {away_df['won'].sum() / len(away_df) * 100:.1f}%")
    print(f"ROI: {(away_df['profit'].sum() / away_df['stake'].sum()) * 100:.2f}%")
    
    print("\nAway Wins por rango de cuota:")
    for odds_cat in ['1.5-2.0', '2.0-2.5', '2.5-3.0', '3.0-4.0', '4.0+']:
        subset = away_df[away_df['odds_cat'] == odds_cat]
        if len(subset) > 0:
            win_rate = subset['won'].sum() / len(subset)
            profit = subset['profit'].sum()
            roi = (profit / subset['stake'].sum()) * 100
            print(f"  Cuota {odds_cat:8s}: {len(subset):2d} apuestas | "
                  f"Win Rate: {win_rate*100:5.1f}% | ROI: {roi:7.2f}%")
    
    # 4. Identificar umbrales óptimos
    print("\n🎛️  UMBRALES ÓPTIMOS SUGERIDOS:")
    print("-"*80)
    
    # Mejor umbral de edge para cada tipo
    for bet_type in ['Home', 'Draw', 'Away']:
        type_df = df[df['bet_type'] == bet_type]
        if len(type_df) == 0:
            continue
            
        best_edge_threshold = None
        best_roi = -100
        
        for threshold in [0.10, 0.12, 0.14, 0.16, 0.18, 0.20]:
            subset = type_df[type_df['edge'] >= threshold]
            if len(subset) >= 5:
                profit = subset['profit'].sum()
                stake = subset['stake'].sum()
                roi = (profit / stake) * 100
                if roi > best_roi:
                    best_roi = roi
                    best_edge_threshold = threshold
        
        if best_edge_threshold:
            print(f"  {bet_type:5s}: Edge mínimo {best_edge_threshold*100:.0f}% → ROI {best_roi:.2f}%")
    
    # 5. Análisis de rachas
    print("\n📊 ANÁLISIS DE PATRONES SECUENCIALES:")
    print("-"*80)
    
    # Ver si hay patrón después de ganar/perder
    df['prev_won'] = df['won'].shift(1)
    df['next_won'] = df['won'].shift(-1)
    
    after_win = df[df['prev_won'] == True]
    after_loss = df[df['prev_won'] == False]
    
    if len(after_win) > 0:
        win_after_win = after_win['won'].sum() / len(after_win)
        print(f"  Win Rate después de ganar: {win_after_win*100:.1f}%")
    
    if len(after_loss) > 0:
        win_after_loss = after_loss['won'].sum() / len(after_loss)
        print(f"  Win Rate después de perder: {win_after_loss*100:.1f}%")
    
    # 6. Recomendaciones finales
    print("\n" + "="*80)
    print("💡 RECOMENDACIONES PARA FILTROS V2:")
    print("="*80)
    
    # Encontrar las mejores combinaciones
    profitable = best_combos_df[best_combos_df['roi'] > 5]  # ROI > 5%
    
    print(f"\n✅ ESTRATEGIAS RENTABLES IDENTIFICADAS: {len(profitable)}")
    
    # Separar por tipo
    for bet_type in ['Away', 'Home', 'Draw']:
        type_profitable = profitable[profitable['type'] == bet_type]
        if len(type_profitable) > 0:
            avg_roi = type_profitable['roi'].mean()
            print(f"\n{bet_type.upper()}:")
            print(f"  • {len(type_profitable)} combinaciones rentables")
            print(f"  • ROI promedio: {avg_roi:.2f}%")
            
            # Mostrar rangos óptimos
            best_edges = type_profitable['edge'].value_counts().head(2)
            best_odds = type_profitable['odds'].value_counts().head(2)
            
            print(f"  • Mejores edges: {', '.join(best_edges.index.tolist())}")
            print(f"  • Mejores cuotas: {', '.join(best_odds.index.tolist())}")
    
    # Calcular potencial de ganancia
    total_profitable_bets = profitable['count'].sum()
    total_profitable_profit = profitable['profit'].sum()
    avg_profitable_roi = profitable['roi'].mean()
    
    print(f"\n📈 POTENCIAL SI SOLO APOSTAMOS COMBINACIONES RENTABLES:")
    print(f"  • Total de apuestas: {total_profitable_bets}")
    print(f"  • Ganancia total: ${total_profitable_profit:,.2f}")
    print(f"  • ROI promedio: {avg_profitable_roi:.2f}%")
    
    # Guardar recomendaciones
    print("\n" + "="*80)
    print("🎯 FILTROS V2 SUGERIDOS:")
    print("="*80)
    
    print("\n```python")
    print("FILTROS_V2 = {")
    print("    'Away': {")
    away_best = profitable[profitable['type'] == 'Away'].nlargest(3, 'roi')
    if len(away_best) > 0:
        print(f"        'edge_min': {away_best['edge'].iloc[0].split('-')[0]},")
        print(f"        'edge_max': 0.22,")
        print(f"        'odds_min': 1.8,")
        print(f"        'odds_max': 4.0,")
    print("    },")
    print("    'Home': {")
    home_best = profitable[profitable['type'] == 'Home'].nlargest(3, 'roi')
    if len(home_best) > 0:
        print(f"        'edge_min': {home_best['edge'].iloc[0].split('-')[0]},")
        print(f"        'edge_max': 0.20,")
        print(f"        'odds_min': 1.5,")
        print(f"        'odds_max': 3.0,")
    print("    },")
    print("    'Draw': 'SKIP'  # No rentable")
    print("}")
    print("```")
    
    return profitable


if __name__ == '__main__':
    profitable_combos = deep_analysis()
