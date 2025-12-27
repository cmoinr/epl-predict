"""
Pipeline Completo de Integración de Datos de Mercado
Ejecuta todo el proceso de enriquecimiento de datos con odds
"""

from pathlib import Path
import sys

# Agregar src al path
sys.path.append(str(Path(__file__).parent.parent / 'src'))
sys.path.append(str(Path(__file__).parent))

from merge_odds_data import merge_datasets
from market_features import integrate_market_intelligence
from backtest_value_betting import ValueBettingBacktest
import pandas as pd


def main():
    """Pipeline completo de integración de datos de mercado"""
    
    base_path = Path(__file__).parent.parent
    
    # Rutas
    epl_final = base_path / 'data' / 'raw' / 'epl_final.csv'
    epl_odds = base_path / 'data' / 'raw' / 'epl_odds.csv'
    enriched_output = base_path / 'data' / 'processed' / 'epl_enriched_with_odds.csv'
    final_output = base_path / 'data' / 'processed' / 'epl_with_market_intelligence.csv'
    
    print("="*70)
    print("🚀 PIPELINE DE INTEGRACIÓN DE DATOS DE MERCADO")
    print("="*70)
    print()
    
    # PASO 1: Merge de datasets
    print("📊 PASO 1/4: Fusionando epl_final.csv con epl_odds.csv...")
    print("-" * 70)
    
    if not enriched_output.exists():
        df_enriched = merge_datasets(epl_final, epl_odds, enriched_output)
    else:
        print(f"✅ Dataset enriquecido ya existe: {enriched_output}")
        df_enriched = pd.read_csv(enriched_output)
    
    print("\n")
    
    # PASO 2: Crear features de mercado
    print("🔧 PASO 2/4: Generando features avanzadas de mercado...")
    print("-" * 70)
    
    if not final_output.exists():
        df_final = integrate_market_intelligence(str(enriched_output), str(final_output))
    else:
        print(f"✅ Dataset con market intelligence ya existe: {final_output}")
        df_final = pd.read_csv(final_output)
    
    print("\n")
    
    # PASO 3: Análisis exploratorio
    print("📈 PASO 3/4: Análisis exploratorio de datos...")
    print("-" * 70)
    
    df_final['MatchDate'] = pd.to_datetime(df_final['MatchDate'])
    df_with_odds = df_final[df_final['AvgOdds_Home'].notna()]
    
    print(f"📊 Estadísticas del dataset:")
    print(f"   Total de partidos: {len(df_final):,}")
    print(f"   Partidos con odds: {len(df_with_odds):,} ({len(df_with_odds)/len(df_final)*100:.1f}%)")
    print(f"   Rango de fechas: {df_final['MatchDate'].min()} a {df_final['MatchDate'].max()}")
    print(f"   Temporadas: {df_final['Season'].nunique()}")
    
    print(f"\n🎯 Precisión del mercado:")
    market_accuracy = df_with_odds['MarketAccuracy'].mean()
    print(f"   El mercado predice correctamente: {market_accuracy*100:.1f}%")
    
    print(f"\n🔥 Sorpresas (upsets):")
    upset_rate = df_with_odds['IsUpset'].mean()
    print(f"   Tasa de upsets: {upset_rate*100:.1f}%")
    
    print(f"\n📊 Distribución de consenso del mercado:")
    print(df_with_odds['MarketConsensus'].describe())
    
    print("\n")
    
    # PASO 4: Backtesting rápido
    print("🎲 PASO 4/4: Backtesting de value betting (muestra)...")
    print("-" * 70)
    
    # Tomar solo una muestra para demo rápido
    sample_size = min(200, len(df_with_odds))
    df_sample = df_with_odds.head(sample_size)
    
    backtest = ValueBettingBacktest(
        df_sample,
        initial_bankroll=1000,
        kelly_fraction=0.25
    )
    
    bet_history = backtest.simulate_bets(min_edge=0.05, min_prob=0.15)
    
    if len(bet_history) > 0:
        backtest.generate_report(bet_history)
        
        # Guardar resultados de muestra
        sample_output = base_path / 'data' / 'processed' / 'backtest_sample.csv'
        bet_history.to_csv(sample_output, index=False)
        print(f"\n💾 Resultados de muestra guardados en: {sample_output}")
    
    print("\n")
    
    # RESUMEN FINAL
    print("="*70)
    print("✅ PIPELINE COMPLETADO")
    print("="*70)
    print("\n📁 Archivos generados:")
    print(f"   1. {enriched_output.relative_to(base_path)}")
    print(f"   2. {final_output.relative_to(base_path)}")
    
    print("\n🎯 PRÓXIMOS PASOS RECOMENDADOS:")
    print("-" * 70)
    print("1. RE-ENTRENAR MODELOS con nuevas features:")
    print("   python retrain_models_improved.py")
    print()
    print("2. EVALUAR importancia de features de mercado:")
    print("   - ¿MarketProb mejora la precisión?")
    print("   - ¿Qué features de mercado son más predictivas?")
    print()
    print("3. CREAR MODELO ENSEMBLE:")
    print("   - Combinar predicciones ML + probabilidades de mercado")
    print("   - Ajustar pesos: 70% modelo, 30% mercado (o viceversa)")
    print()
    print("4. OPTIMIZAR ESTRATEGIA DE VALUE BETTING:")
    print("   - Ajustar min_edge y kelly_fraction")
    print("   - Backtest completo en toda la data histórica")
    print()
    print("5. INTEGRAR ODDS EN TIEMPO REAL:")
    print("   - API de casas de apuestas para predicciones futuras")
    print("   - Actualizar sample_odds.csv automáticamente")
    
    print("\n💡 INSIGHTS CLAVE:")
    print("-" * 70)
    print("• Las cuotas del mercado son features PODEROSAS")
    print("• Representan la sabiduría colectiva de miles de apostadores")
    print("• Tu modelo debe SUPERARLAS, no solo replicarlas")
    print("• Value betting = encontrar diferencias entre tu modelo y el mercado")
    print("• Edge del 5-10% es realista para apuestas rentables")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
