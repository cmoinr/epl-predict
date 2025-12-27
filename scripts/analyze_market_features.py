"""
Análisis Rápido de Features de Mercado
Genera estadísticas y visualizaciones básicas
"""

import pandas as pd
import numpy as np
from pathlib import Path


def analyze_market_features():
    """Analiza las features de mercado generadas"""
    
    base_path = Path(__file__).parent.parent
    data_path = base_path / 'data' / 'processed' / 'epl_with_market_intelligence.csv'
    
    if not data_path.exists():
        print("❌ Error: ejecuta primero scripts/integrate_market_data.py")
        return
    
    print("📊 Cargando datos...")
    df = pd.read_csv(data_path)
    df['MatchDate'] = pd.to_datetime(df['MatchDate'])
    
    # Filtrar solo partidos con odds
    df_odds = df[df['AvgOdds_Home'].notna()].copy()
    
    print("\n" + "="*70)
    print("📈 ANÁLISIS DE FEATURES DE MERCADO")
    print("="*70)
    
    # 1. Precisión del mercado por tipo de favorito
    print("\n1️⃣ PRECISIÓN DEL MERCADO POR FAVORITO:")
    print("-" * 70)
    
    market_by_favorite = df_odds.groupby('MarketFavorite').agg({
        'MarketAccuracy': ['mean', 'count']
    })
    
    for favorite_type in ['H', 'D', 'A']:
        if favorite_type in market_by_favorite.index:
            accuracy = market_by_favorite.loc[favorite_type, ('MarketAccuracy', 'mean')]
            count = market_by_favorite.loc[favorite_type, ('MarketAccuracy', 'count')]
            favorite_name = {'H': 'Local', 'D': 'Empate', 'A': 'Visitante'}[favorite_type]
            print(f"   {favorite_name:10s}: {accuracy*100:5.1f}% ({count:3.0f} partidos)")
    
    # 2. Tasa de upset por rango de probabilidad
    print("\n2️⃣ TASA DE UPSETS POR RANGO DE PROBABILIDAD DEL MERCADO:")
    print("-" * 70)
    
    # Crear bins de probabilidad
    df_odds['ProbBin'] = pd.cut(
        df_odds['MarketProb_Home'], 
        bins=[0, 0.3, 0.4, 0.5, 0.6, 1.0],
        labels=['0-30%', '30-40%', '40-50%', '50-60%', '60-100%']
    )
    
    upset_by_prob = df_odds.groupby('ProbBin').agg({
        'IsUpset': ['mean', 'count']
    })
    
    print("   Probabilidad del Local | Tasa de Upset | Partidos")
    print("   " + "-"*54)
    for prob_range in upset_by_prob.index:
        upset_rate = upset_by_prob.loc[prob_range, ('IsUpset', 'mean')]
        count = upset_by_prob.loc[prob_range, ('IsUpset', 'count')]
        print(f"   {str(prob_range):20s} | {upset_rate*100:11.1f}% | {count:8.0f}")
    
    # 3. Correlación entre FavoriteStrength y resultado
    print("\n3️⃣ RELACIÓN ENTRE CLARIDAD DEL FAVORITO Y RESULTADO:")
    print("-" * 70)
    
    # Dividir en favoritos claros vs dudosos
    clear_threshold = df_odds['FavoriteStrength'].quantile(0.75)
    
    df_odds['ClearFavorite'] = df_odds['FavoriteStrength'] > clear_threshold
    
    accuracy_by_clarity = df_odds.groupby('ClearFavorite')['MarketAccuracy'].mean()
    
    print(f"   Favorito DUDOSO (strength < {clear_threshold:.3f}): {accuracy_by_clarity[False]*100:.1f}%")
    print(f"   Favorito CLARO  (strength > {clear_threshold:.3f}): {accuracy_by_clarity[True]*100:.1f}%")
    print(f"\n   💡 Insight: Favoritos claros son {(accuracy_by_clarity[True]/accuracy_by_clarity[False]-1)*100:+.1f}% más predecibles")
    
    # 4. Desacuerdo del mercado y sorpresas
    print("\n4️⃣ DESACUERDO ENTRE CASAS Y SORPRESAS:")
    print("-" * 70)
    
    # Dividir por nivel de desacuerdo
    high_disagreement_threshold = df_odds['MarketDisagreement'].quantile(0.75)
    
    df_odds['HighDisagreement'] = df_odds['MarketDisagreement'] > high_disagreement_threshold
    
    upset_by_disagreement = df_odds.groupby('HighDisagreement')['IsUpset'].mean()
    
    print(f"   Bajo desacuerdo entre casas:  Upsets = {upset_by_disagreement[False]*100:.1f}%")
    print(f"   Alto desacuerdo entre casas:  Upsets = {upset_by_disagreement[True]*100:.1f}%")
    print(f"\n   💡 Insight: Más desacuerdo = {(upset_by_disagreement[True]/upset_by_disagreement[False]-1)*100:+.1f}% más sorpresas")
    
    # 5. Partidos competitivos
    print("\n5️⃣ PARTIDOS COMPETITIVOS (CUOTAS SIMILARES):")
    print("-" * 70)
    
    competitive_matches = df_odds['IsCompetitiveMatch'].sum()
    competitive_pct = df_odds['IsCompetitiveMatch'].mean()
    
    competitive_results = df_odds.groupby('IsCompetitiveMatch')['FullTimeResult'].value_counts(normalize=True).unstack()
    
    print(f"   Total de partidos competitivos: {competitive_matches} ({competitive_pct*100:.1f}%)")
    print(f"\n   Distribución de resultados:")
    print(f"   {'Tipo':20s} | {'Local':>8s} | {'Empate':>8s} | {'Visitante':>8s}")
    print("   " + "-"*54)
    
    for is_competitive in [False, True]:
        if is_competitive in competitive_results.index:
            label = "Competitivo" if is_competitive else "No competitivo"
            home_pct = competitive_results.loc[is_competitive, 'H'] * 100 if 'H' in competitive_results.columns else 0
            draw_pct = competitive_results.loc[is_competitive, 'D'] * 100 if 'D' in competitive_results.columns else 0
            away_pct = competitive_results.loc[is_competitive, 'A'] * 100 if 'A' in competitive_results.columns else 0
            print(f"   {label:20s} | {home_pct:7.1f}% | {draw_pct:7.1f}% | {away_pct:7.1f}%")
    
    # 6. Overround promedio
    print("\n6️⃣ MARGEN DE LAS CASAS DE APUESTAS (OVERROUND):")
    print("-" * 70)
    
    avg_overround = df_odds['Overround'].mean()
    min_overround = df_odds['Overround'].min()
    max_overround = df_odds['Overround'].max()
    
    print(f"   Overround promedio: {avg_overround:.4f} ({(avg_overround-1)*100:.2f}% margen)")
    print(f"   Overround mínimo:   {min_overround:.4f} ({(min_overround-1)*100:.2f}% margen)")
    print(f"   Overround máximo:   {max_overround:.4f} ({(max_overround-1)*100:.2f}% margen)")
    print(f"\n   💡 Las casas tienen ~{(avg_overround-1)*100:.1f}% de margen de ganancia")
    
    # 7. Equipos más subestimados/sobreestimados
    print("\n7️⃣ EQUIPOS CON MAYOR MARKET SURPRISE (TOP 5):")
    print("-" * 70)
    
    # Market surprise promedio por equipo (como local)
    home_surprise = df_odds.groupby('HomeTeam')['MarketSurprise_Home'].agg(['mean', 'count'])
    home_surprise = home_surprise[home_surprise['count'] >= 5]  # Mínimo 5 partidos
    home_surprise = home_surprise.sort_values('mean', ascending=False)
    
    print("\n   📈 EQUIPOS MÁS SUBESTIMADOS (como local):")
    for i, (team, data) in enumerate(home_surprise.head(5).iterrows(), 1):
        print(f"   {i}. {team:20s}: +{data['mean']*100:5.1f}% surprise ({data['count']:.0f} partidos)")
    
    print("\n   📉 EQUIPOS MÁS SOBREESTIMADOS (como local):")
    for i, (team, data) in enumerate(home_surprise.tail(5).iterrows(), 1):
        print(f"   {i}. {team:20s}: {data['mean']*100:5.1f}% surprise ({data['count']:.0f} partidos)")
    
    # 8. Recomendaciones para features
    print("\n8️⃣ FEATURES MÁS PROMETEDORAS PARA TU MODELO:")
    print("-" * 70)
    
    # Calcular correlación con resultado (0=Away, 0.5=Draw, 1=Home)
    df_odds['ResultNumeric'] = df_odds['FullTimeResult'].map({'A': 0, 'D': 0.5, 'H': 1})
    
    market_features = [
        'MarketProb_Home',
        'MarketProb_Away',
        'FavoriteStrength',
        'MarketConsensus',
        'ImpliedGoalDiff',
        'MarketDisagreement',
        'IsCompetitiveMatch'
    ]
    
    correlations = []
    for feat in market_features:
        if feat in df_odds.columns:
            corr = df_odds[[feat, 'ResultNumeric']].corr().iloc[0, 1]
            correlations.append((feat, abs(corr)))
    
    correlations.sort(key=lambda x: x[1], reverse=True)
    
    print("\n   Feature                    | Correlación con Resultado")
    print("   " + "-"*60)
    for feat, corr in correlations:
        stars = "⭐" * int(corr * 10)
        print(f"   {feat:27s} | {corr:.4f} {stars}")
    
    # Summary
    print("\n" + "="*70)
    print("✅ CONCLUSIONES Y RECOMENDACIONES")
    print("="*70)
    print(f"""
   1. El mercado es {df_odds['MarketAccuracy'].mean()*100:.1f}% preciso - HAY MARGEN PARA MEJORA
   
   2. Favoritos CLAROS tienen {accuracy_by_clarity[True]*100:.1f}% precisión vs {accuracy_by_clarity[False]*100:.1f}% dudosos
      → Tu modelo debe enfocarse en partidos con baja FavoriteStrength
   
   3. {upset_by_disagreement[True]*100:.1f}% de upsets ocurren con alto desacuerdo entre casas
      → MarketDisagreement es señal de value betting potencial
   
   4. Partidos competitivos (IsCompetitiveMatch=1) son más impredecibles
      → Requieren features más sofisticadas
   
   5. Las casas tienen ~{(avg_overround-1)*100:.1f}% de margen
      → Necesitas edge > {(avg_overround-1)*100:.1f}% para ser rentable
   
   🎯 FEATURES RECOMENDADAS para incluir en tu modelo:
      - MarketProb_Home/Away (alta correlación)
      - FavoriteStrength (identifica favoritos claros)
      - ImpliedGoalDiff (predictor de resultado)
      - MarketConsensus (confiabilidad de la predicción)
      - Team_AvgMarketProb_L10 (reputación histórica)
   
   ⚠️  ADVERTENCIA:
      Solo tienes {len(df_odds)} partidos con odds ({len(df_odds)/len(df)*100:.1f}% del dataset)
      Necesitas EXPANDIR este dataset para entrenar modelos robustos
      
      👉 Descarga odds de football-data.co.uk para 2000-2025
   """)
    
    print("="*70)


if __name__ == '__main__':
    analyze_market_features()
