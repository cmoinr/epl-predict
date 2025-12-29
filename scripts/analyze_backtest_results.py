"""
Análisis detallado de resultados del backtest
Genera insights sobre la estrategia de value betting
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_backtest(results_path='data/processed/backtest_results.csv'):
    """Análisis profundo de los resultados del backtest"""
    
    df = pd.read_csv(results_path)
    df['date'] = pd.to_datetime(df['date'])
    
    print("\n" + "="*70)
    print("📊 ANÁLISIS PROFUNDO DEL BACKTEST")
    print("="*70)
    
    # 1. Análisis por rango de edge
    print("\n🎯 RENDIMIENTO POR RANGO DE EDGE:")
    print("-"*70)
    
    edge_bins = [0, 0.10, 0.15, 0.20, 0.30, 1.0]
    edge_labels = ['5-10%', '10-15%', '15-20%', '20-30%', '30%+']
    df['edge_range'] = pd.cut(df['edge'], bins=edge_bins, labels=edge_labels)
    
    for edge_range in edge_labels:
        subset = df[df['edge_range'] == edge_range]
        if len(subset) == 0:
            continue
        
        win_rate = subset['won'].sum() / len(subset)
        total_profit = subset['profit'].sum()
        total_stake = subset['stake'].sum()
        roi = (total_profit / total_stake) * 100 if total_stake > 0 else 0
        
        print(f"   Edge {edge_range:8s}: {len(subset):4d} apuestas | "
              f"Win Rate: {win_rate*100:5.1f}% | ROI: {roi:7.2f}% | "
              f"Profit: ${total_profit:8,.2f}")
    
    # 2. Análisis por cuota
    print("\n🎲 RENDIMIENTO POR RANGO DE CUOTA:")
    print("-"*70)
    
    odds_bins = [1.0, 2.0, 3.0, 4.0, 5.0, 100.0]
    odds_labels = ['1.0-2.0', '2.0-3.0', '3.0-4.0', '4.0-5.0', '5.0+']
    df['odds_range'] = pd.cut(df['odds'], bins=odds_bins, labels=odds_labels)
    
    for odds_range in odds_labels:
        subset = df[df['odds_range'] == odds_range]
        if len(subset) == 0:
            continue
        
        win_rate = subset['won'].sum() / len(subset)
        total_profit = subset['profit'].sum()
        avg_edge = subset['edge'].mean()
        
        print(f"   Cuota {odds_range:8s}: {len(subset):4d} apuestas | "
              f"Win Rate: {win_rate*100:5.1f}% | Edge: {avg_edge*100:5.1f}% | "
              f"Profit: ${total_profit:8,.2f}")
    
    # 3. Mejor tipo de apuesta
    print("\n🏆 ANÁLISIS POR TIPO DE APUESTA:")
    print("-"*70)
    
    for bet_type in df['bet_type'].unique():
        subset = df[df['bet_type'] == bet_type]
        win_rate = subset['won'].sum() / len(subset)
        total_profit = subset['profit'].sum()
        total_stake = subset['stake'].sum()
        roi = (total_profit / total_stake) * 100
        avg_odds = subset['odds'].mean()
        avg_edge = subset['edge'].mean()
        
        print(f"   {bet_type:8s}: {len(subset):4d} apuestas | "
              f"Win Rate: {win_rate*100:5.1f}% | ROI: {roi:7.2f}% | "
              f"Avg Odds: {avg_odds:.2f} | Avg Edge: {avg_edge*100:.1f}%")
    
    # 4. Evolución temporal
    print("\n📈 EVOLUCIÓN TEMPORAL:")
    print("-"*70)
    
    df['month'] = df['date'].dt.to_period('M')
    monthly = df.groupby('month').agg({
        'profit': 'sum',
        'won': ['sum', 'count']
    })
    
    print(f"   Primeros 3 meses:")
    for month in monthly.head(3).index:
        profit = monthly.loc[month, ('profit', 'sum')]
        wins = monthly.loc[month, ('won', 'sum')]
        total = monthly.loc[month, ('won', 'count')]
        print(f"      {month}: ${profit:8,.2f} | {wins}/{total} ganadas ({wins/total*100:.1f}%)")
    
    print(f"   Últimos 3 meses:")
    for month in monthly.tail(3).index:
        profit = monthly.loc[month, ('profit', 'sum')]
        wins = monthly.loc[month, ('won', 'sum')]
        total = monthly.loc[month, ('won', 'count')]
        print(f"      {month}: ${profit:8,.2f} | {wins}/{total} ganadas ({wins/total*100:.1f}%)")
    
    # 5. Identificar mejores oportunidades
    print("\n💎 TOP 10 MEJORES APUESTAS (por ganancia):")
    print("-"*70)
    
    top_bets = df.nlargest(10, 'profit')
    for idx, (_, bet) in enumerate(top_bets.iterrows(), 1):
        print(f"   {idx:2d}. {bet['home_team']:15s} vs {bet['away_team']:15s} | "
              f"{bet['bet_type']:4s} @ {bet['odds']:.2f} | "
              f"Edge: {bet['edge']*100:5.1f}% | Profit: ${bet['profit']:.2f}")
    
    # 6. Identificar peores apuestas
    print("\n💸 TOP 10 PEORES APUESTAS (por pérdida):")
    print("-"*70)
    
    worst_bets = df.nsmallest(10, 'profit')
    for idx, (_, bet) in enumerate(worst_bets.iterrows(), 1):
        print(f"   {idx:2d}. {bet['home_team']:15s} vs {bet['away_team']:15s} | "
              f"{bet['bet_type']:4s} @ {bet['odds']:.2f} | "
              f"Edge: {bet['edge']*100:5.1f}% | Loss: ${bet['profit']:.2f}")
    
    # 7. Recomendaciones
    print("\n" + "="*70)
    print("💡 RECOMENDACIONES CLAVE:")
    print("="*70)
    
    # Encontrar el rango de edge más rentable
    edge_analysis = df.groupby('edge_range').agg({
        'profit': 'sum',
        'stake': 'sum',
        'won': lambda x: x.sum() / len(x)
    })
    edge_analysis['roi'] = (edge_analysis['profit'] / edge_analysis['stake']) * 100
    best_edge = edge_analysis['roi'].idxmax()
    
    print(f"\n✅ Mejor rango de edge: {best_edge} (ROI: {edge_analysis.loc[best_edge, 'roi']:.2f}%)")
    
    # Encontrar el tipo de apuesta más rentable
    type_analysis = df.groupby('bet_type').agg({
        'profit': 'sum',
        'stake': 'sum'
    })
    type_analysis['roi'] = (type_analysis['profit'] / type_analysis['stake']) * 100
    best_type = type_analysis['roi'].idxmax()
    
    print(f"✅ Mejor tipo de apuesta: {best_type} (ROI: {type_analysis.loc[best_type, 'roi']:.2f}%)")
    
    # Calcular umbral óptimo de edge
    edges = sorted(df['edge'].unique())
    best_roi = -100
    best_threshold = 0
    
    for threshold in edges:
        subset = df[df['edge'] >= threshold]
        if len(subset) >= 20:  # Mínimo 20 apuestas
            profit = subset['profit'].sum()
            stake = subset['stake'].sum()
            roi = (profit / stake) * 100
            if roi > best_roi:
                best_roi = roi
                best_threshold = threshold
    
    print(f"✅ Umbral óptimo de edge: {best_threshold*100:.1f}% (ROI: {best_roi:.2f}%)")
    print(f"   Con este umbral: {len(df[df['edge'] >= best_threshold])} apuestas")
    
    # Análisis de Kelly
    print(f"\n📊 ANÁLISIS DE KELLY CRITERION:")
    actual_kelly = df['kelly_fraction'].iloc[0] if 'kelly_fraction' in df.columns else 0.25
    print(f"   Fracción Kelly usada: {actual_kelly}")
    print(f"   Stake promedio: ${df['stake'].mean():.2f}")
    print(f"   Stake máximo: ${df['stake'].max():.2f}")
    print(f"   Stake mínimo: ${df['stake'].min():.2f}")
    
    volatility = df['profit'].std()
    print(f"   Volatilidad (std): ${volatility:.2f}")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    analyze_backtest()
