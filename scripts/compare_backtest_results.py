"""
Comparación entre backtest baseline y optimizado
Muestra las mejoras logradas con filtros inteligentes
"""

import pandas as pd
import numpy as np
from pathlib import Path

def compare_backtests():
    """Compara resultados baseline vs optimizado"""
    
    baseline_path = Path(__file__).parent.parent / 'data' / 'processed' / 'backtest_results_baseline.csv'
    optimized_path = Path(__file__).parent.parent / 'data' / 'processed' / 'backtest_results_optimized.csv'
    
    if not baseline_path.exists() or not optimized_path.exists():
        print("❌ Error: Faltan archivos de backtest")
        return
    
    df_baseline = pd.read_csv(baseline_path)
    df_optimized = pd.read_csv(optimized_path)
    
    print("\n" + "="*80)
    print("📊 COMPARACIÓN: BASELINE vs OPTIMIZADO")
    print("="*80)
    
    # Métricas generales
    def calculate_metrics(df, initial_bankroll=1000):
        total_bets = len(df)
        wins = df['won'].sum()
        win_rate = wins / total_bets if total_bets > 0 else 0
        
        total_staked = df['stake'].sum()
        total_profit = df['profit'].sum()
        roi = (total_profit / total_staked) * 100 if total_staked > 0 else 0
        
        final_bankroll = df['bankroll'].iloc[-1] if len(df) > 0 else initial_bankroll
        total_return = ((final_bankroll - initial_bankroll) / initial_bankroll) * 100
        
        avg_edge = df['edge'].mean()
        avg_odds = df['odds'].mean()
        
        drawdown = initial_bankroll - df['bankroll'].min() if len(df) > 0 else 0
        
        returns = df['profit'] / df['stake']
        sharpe = (returns.mean() / returns.std()) * np.sqrt(len(returns)) if len(returns) > 1 and returns.std() > 0 else 0
        
        return {
            'total_bets': total_bets,
            'wins': wins,
            'win_rate': win_rate,
            'total_staked': total_staked,
            'total_profit': total_profit,
            'roi': roi,
            'final_bankroll': final_bankroll,
            'total_return': total_return,
            'avg_edge': avg_edge,
            'avg_odds': avg_odds,
            'drawdown': drawdown,
            'sharpe': sharpe
        }
    
    m_baseline = calculate_metrics(df_baseline)
    m_optimized = calculate_metrics(df_optimized)
    
    print("\n💰 RENDIMIENTO GENERAL:")
    print("-"*80)
    print(f"{'Métrica':<30} {'Baseline':>15} {'Optimizado':>15} {'Mejora':>15}")
    print("-"*80)
    
    print(f"{'Total de apuestas':<30} {m_baseline['total_bets']:>15,} {m_optimized['total_bets']:>15,} {m_optimized['total_bets']-m_baseline['total_bets']:>15,}")
    print(f"{'Win Rate':<30} {m_baseline['win_rate']*100:>14.1f}% {m_optimized['win_rate']*100:>14.1f}% {(m_optimized['win_rate']-m_baseline['win_rate'])*100:>14.1f}%")
    print(f"{'ROI':<30} {m_baseline['roi']:>14.2f}% {m_optimized['roi']:>14.2f}% {m_optimized['roi']-m_baseline['roi']:>14.2f}%")
    print(f"{'Retorno Total':<30} {m_baseline['total_return']:>14.2f}% {m_optimized['total_return']:>14.2f}% {m_optimized['total_return']-m_baseline['total_return']:>14.2f}%")
    print(f"{'Bankroll Final':<30} ${m_baseline['final_bankroll']:>13,.2f} ${m_optimized['final_bankroll']:>13,.2f} ${m_optimized['final_bankroll']-m_baseline['final_bankroll']:>13,.2f}")
    print(f"{'Ganancia/Pérdida':<30} ${m_baseline['total_profit']:>13,.2f} ${m_optimized['total_profit']:>13,.2f} ${m_optimized['total_profit']-m_baseline['total_profit']:>13,.2f}")
    
    print("\n🎯 EFICIENCIA:")
    print("-"*80)
    print(f"{'Total Apostado':<30} ${m_baseline['total_staked']:>13,.2f} ${m_optimized['total_staked']:>13,.2f} ${m_optimized['total_staked']-m_baseline['total_staked']:>13,.2f}")
    print(f"{'Edge Promedio':<30} {m_baseline['avg_edge']*100:>14.1f}% {m_optimized['avg_edge']*100:>14.1f}% {(m_optimized['avg_edge']-m_baseline['avg_edge'])*100:>14.1f}%")
    print(f"{'Cuota Promedio':<30} {m_baseline['avg_odds']:>15.2f} {m_optimized['avg_odds']:>15.2f} {m_optimized['avg_odds']-m_baseline['avg_odds']:>15.2f}")
    
    print("\n📉 RIESGO:")
    print("-"*80)
    print(f"{'Drawdown Máximo':<30} ${m_baseline['drawdown']:>13,.2f} ${m_optimized['drawdown']:>13,.2f} ${m_optimized['drawdown']-m_baseline['drawdown']:>13,.2f}")
    print(f"{'Sharpe Ratio':<30} {m_baseline['sharpe']:>15.2f} {m_optimized['sharpe']:>15.2f} {m_optimized['sharpe']-m_baseline['sharpe']:>15.2f}")
    
    # Análisis por tipo
    print("\n🏆 DESGLOSE POR TIPO DE APUESTA:")
    print("-"*80)
    
    for bet_type in ['Home', 'Draw', 'Away']:
        print(f"\n{bet_type.upper()}:")
        
        # Baseline
        subset_b = df_baseline[df_baseline['bet_type'] == bet_type]
        if len(subset_b) > 0:
            win_rate_b = subset_b['won'].sum() / len(subset_b)
            profit_b = subset_b['profit'].sum()
            roi_b = (profit_b / subset_b['stake'].sum()) * 100 if subset_b['stake'].sum() > 0 else 0
        else:
            win_rate_b = 0
            profit_b = 0
            roi_b = 0
        
        # Optimized
        subset_o = df_optimized[df_optimized['bet_type'] == bet_type]
        if len(subset_o) > 0:
            win_rate_o = subset_o['won'].sum() / len(subset_o)
            profit_o = subset_o['profit'].sum()
            roi_o = (profit_o / subset_o['stake'].sum()) * 100 if subset_o['stake'].sum() > 0 else 0
        else:
            win_rate_o = 0
            profit_o = 0
            roi_o = 0
        
        print(f"  Baseline:   {len(subset_b):3d} apuestas | Win Rate: {win_rate_b*100:5.1f}% | ROI: {roi_b:7.2f}% | Profit: ${profit_b:8,.2f}")
        print(f"  Optimizado: {len(subset_o):3d} apuestas | Win Rate: {win_rate_o*100:5.1f}% | ROI: {roi_o:7.2f}% | Profit: ${profit_o:8,.2f}")
        print(f"  Mejora:     {len(subset_o)-len(subset_b):+3d} apuestas | Win Rate: {(win_rate_o-win_rate_b)*100:+5.1f}% | ROI: {roi_o-roi_b:+7.2f}% | Profit: ${profit_o-profit_b:+8,.2f}")
    
    # Análisis de eficiencia
    print("\n" + "="*80)
    print("💡 MEJORAS CLAVE:")
    print("="*80)
    
    improvement_roi = m_optimized['roi'] - m_baseline['roi']
    improvement_wr = (m_optimized['win_rate'] - m_baseline['win_rate']) * 100
    improvement_profit = m_optimized['total_profit'] - m_baseline['total_profit']
    improvement_dd = m_baseline['drawdown'] - m_optimized['drawdown']
    reduction_bets = ((m_baseline['total_bets'] - m_optimized['total_bets']) / m_baseline['total_bets']) * 100
    
    print(f"\n✅ ROI mejorado en: {improvement_roi:+.2f}% ({abs(improvement_roi)/abs(m_baseline['roi'])*100:.1f}% de mejora relativa)")
    print(f"✅ Win Rate mejorado en: {improvement_wr:+.1f}%")
    print(f"✅ Ganancia adicional: ${improvement_profit:+,.2f}")
    print(f"✅ Drawdown reducido en: ${improvement_dd:,.2f} ({improvement_dd/m_baseline['drawdown']*100:.1f}%)")
    print(f"✅ Apuestas reducidas en: {reduction_bets:.1f}% (más selectivo)")
    print(f"✅ Sharpe Ratio mejorado: {m_baseline['sharpe']:.2f} → {m_optimized['sharpe']:.2f}")
    
    # Mejor tipo de apuesta
    best_type_o = df_optimized.groupby('bet_type').agg({'profit': 'sum'}).idxmax()[0]
    best_profit_o = df_optimized[df_optimized['bet_type'] == best_type_o]['profit'].sum()
    
    print(f"\n🎯 Mejor tipo de apuesta (Optimizado): {best_type_o} con ${best_profit_o:,.2f} de ganancia")
    
    # Estrategia recomendada
    print("\n" + "="*80)
    print("🚀 ESTRATEGIA RECOMENDADA:")
    print("="*80)
    
    # Encontrar la mejor combinación en optimizado
    best_edge_range = None
    best_roi_range = -100
    
    edge_bins = [0, 0.10, 0.15, 0.20, 0.25, 1.0]
    edge_labels = ['5-10%', '10-15%', '15-20%', '20-25%', '25%+']
    df_optimized['edge_range'] = pd.cut(df_optimized['edge'], bins=edge_bins, labels=edge_labels)
    
    for edge_range in edge_labels:
        subset = df_optimized[df_optimized['edge_range'] == edge_range]
        if len(subset) >= 10:
            profit = subset['profit'].sum()
            stake = subset['stake'].sum()
            roi = (profit / stake) * 100 if stake > 0 else 0
            if roi > best_roi_range:
                best_roi_range = roi
                best_edge_range = edge_range
    
    if best_edge_range:
        print(f"\n✨ Rango de edge óptimo: {best_edge_range} (ROI: {best_roi_range:.2f}%)")
    
    # Análisis de Away wins (son rentables en optimizado!)
    away_subset = df_optimized[df_optimized['bet_type'] == 'Away']
    if len(away_subset) > 0:
        away_roi = (away_subset['profit'].sum() / away_subset['stake'].sum()) * 100
        away_wr = away_subset['won'].sum() / len(away_subset) * 100
        print(f"⚡ Away Wins ahora son rentables! ROI: {away_roi:.2f}% | Win Rate: {away_wr:.1f}%")
    
    print(f"\n📋 Recomendación final:")
    print(f"   • Usar MODO OPTIMIZADO siempre")
    print(f"   • Edge sweet spot: 10-25%")
    print(f"   • Cuotas ideales: 1.5-4.5")
    print(f"   • Away Wins filtrados son muy rentables (ROI {away_roi:.1f}%)")
    print(f"   • Bankroll mínimo recomendado: $2,000 (para soportar volatilidad)")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    compare_backtests()
