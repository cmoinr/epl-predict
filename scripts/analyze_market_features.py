"""
Análisis Rápido de Features de Mercado
Genera estadísticas y visualizaciones básicas
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import io

# Fix encoding for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')


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
    
    # 1. Precisión del mercado por PREDICCIÓN (no por favorito)
    print("\n1️⃣ PRECISIÓN DEL MERCADO POR PREDICCIÓN:")
    print("-" * 70)
    
    # Calcular qué predijo realmente el mercado (mayor probabilidad)
    df_odds['Market_Prediction'] = df_odds[['MarketProb_Home', 'MarketProb_Draw', 'MarketProb_Away']].idxmax(axis=1)
    df_odds['Market_Prediction'] = df_odds['Market_Prediction'].map({
        'MarketProb_Home': 'H',
        'MarketProb_Draw': 'D',
        'MarketProb_Away': 'A'
    })
    df_odds['Market_Correct'] = df_odds['Market_Prediction'] == df_odds['FullTimeResult']
    
    market_by_prediction = df_odds.groupby('Market_Prediction').agg({
        'Market_Correct': ['mean', 'count']
    })
    
    print("   Predicción      | Precisión | Partidos")
    print("   " + "-"*45)
    for pred_type in ['H', 'D', 'A']:
        if pred_type in market_by_prediction.index:
            accuracy = market_by_prediction.loc[pred_type, ('Market_Correct', 'mean')]
            count = market_by_prediction.loc[pred_type, ('Market_Correct', 'count')]
            pred_name = {'H': 'Local', 'D': 'Empate', 'A': 'Visitante'}[pred_type]
            print(f"   {pred_name:15s} | {accuracy*100:7.1f}% | {count:8.0f}")
    
    overall_accuracy = df_odds['Market_Correct'].mean()
    print(f"\n   📊 Precisión GLOBAL del mercado: {overall_accuracy*100:.1f}%")
    
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
    
    # Usar OddsStd como proxy cuando MarketDisagreement no está disponible
    df_odds['Disagreement_Metric'] = df_odds['MarketDisagreement'].fillna(
        (df_odds['OddsStd_Home'] + df_odds['OddsStd_Draw'] + df_odds['OddsStd_Away']) / 3
    )
    
    valid_disagreement = df_odds['Disagreement_Metric'].notna()
    print(f"   Partidos con métrica de desacuerdo: {valid_disagreement.sum()}/{len(df_odds)}")
    
    if valid_disagreement.sum() > 0:
        df_valid = df_odds[valid_disagreement].copy()
        high_disagreement_threshold = df_valid['Disagreement_Metric'].quantile(0.75)
        
        df_valid['HighDisagreement'] = df_valid['Disagreement_Metric'] > high_disagreement_threshold
        
        upset_by_disagreement = df_valid.groupby('HighDisagreement')['IsUpset'].mean()
        
        print(f"\n   Bajo desacuerdo entre casas:  Upsets = {upset_by_disagreement[False]*100:.1f}%")
        print(f"   Alto desacuerdo entre casas:  Upsets = {upset_by_disagreement[True]*100:.1f}%")
        print(f"\n   💡 Insight: Más desacuerdo = {(upset_by_disagreement[True]/upset_by_disagreement[False]-1)*100:+.1f}% más sorpresas")
    else:
        print("   ⚠️  No hay suficientes datos de desacuerdo")
    
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
    
    # 7. Equipos más subestimados/sobreestimados y underdogs
    print("\n7️⃣ ANÁLISIS DE EQUIPOS: UNDERDOGS Y SORPRESAS:")
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
    
    # Análisis de underdogs
    print("\n   🐕 RENDIMIENTO DE UNDERDOGS:")
    underdog_home = df_odds[df_odds['IsUnderdog_Home'] == 1]
    underdog_away = df_odds[df_odds['IsUnderdog_Away'] == 1]
    
    home_underdog_wins = (underdog_home['FullTimeResult'] == 'H').sum()
    away_underdog_wins = (underdog_away['FullTimeResult'] == 'A').sum()
    
    print(f"   Underdogs locales: {len(underdog_home)} partidos, {home_underdog_wins} victorias ({home_underdog_wins/len(underdog_home)*100:.1f}%)")
    print(f"   Underdogs visitantes: {len(underdog_away)} partidos, {away_underdog_wins} victorias ({away_underdog_wins/len(underdog_away)*100:.1f}%)")
    
    # Equipos con mayor tasa de upsets
    if 'Team_UpsetRate_L10' in df_odds.columns:
        team_upset_stats = []
        for team in df_odds['HomeTeam'].unique():
            team_matches = df_odds[
                (df_odds['HomeTeam'] == team) | (df_odds['AwayTeam'] == team)
            ].copy()
            if len(team_matches) >= 10 and team_matches['Team_UpsetRate_L10'].notna().any():
                avg_upset_rate = team_matches['Team_UpsetRate_L10'].mean()
                team_upset_stats.append((team, avg_upset_rate, len(team_matches)))
        
        if team_upset_stats:
            team_upset_stats.sort(key=lambda x: x[1], reverse=True)
            print("\n   ⚡ EQUIPOS QUE MÁS SORPRENDEN (Team_UpsetRate_L10):")
            for i, (team, rate, count) in enumerate(team_upset_stats[:5], 1):
                print(f"   {i}. {team:20s}: {rate*100:5.1f}% upset rate ({count} partidos)")
    
    # 8. Recomendaciones para features y comparación Raw vs Adjusted
    print("\n8️⃣ ANÁLISIS DE FEATURES Y PROBABILIDADES:")
    print("-" * 70)
    
    # Comparar probabilidades raw vs adjusted
    print("\n   📊 IMPACTO DEL OVERROUND (Raw vs Adjusted Probabilities):")
    for outcome in ['Home', 'Draw', 'Away']:
        raw_col = f'MarketProb_{outcome}'
        adj_col = f'AdjustedProb_{outcome}'
        if raw_col in df_odds.columns and adj_col in df_odds.columns:
            avg_raw = df_odds[raw_col].mean()
            avg_adj = df_odds[adj_col].mean()
            diff = avg_adj - avg_raw
            print(f"   {outcome:10s}: Raw={avg_raw:.4f}, Adjusted={avg_adj:.4f}, Diff={diff:+.4f}")
    
    # Calcular correlación con resultado (0=Away, 0.5=Draw, 1=Home)
    df_odds['ResultNumeric'] = df_odds['FullTimeResult'].map({'A': 0, 'D': 0.5, 'H': 1})
    
    market_features = [
        'MarketProb_Home',
        'MarketProb_Away',
        'MarketProb_Draw',
        'AdjustedProb_Home',
        'AdjustedProb_Away',
        'FavoriteStrength',
        'MarketConsensus',
        'ImpliedGoalDiff',
        'MarketDisagreement',
        'IsCompetitiveMatch',
        'OddsStd_Home',
        'OddsRange_Home'
    ]
    
    correlations = []
    for feat in market_features:
        if feat in df_odds.columns:
            valid_data = df_odds[[feat, 'ResultNumeric']].dropna()
            if len(valid_data) > 0:
                corr = valid_data.corr().iloc[0, 1]
                correlations.append((feat, abs(corr), len(valid_data)))
    
    correlations.sort(key=lambda x: x[1], reverse=True)
    
    print("\n   Feature                    | Correlación | Datos Válidos")
    print("   " + "-"*65)
    for feat, corr, valid_count in correlations:
        stars = "⭐" * min(int(corr * 10), 5)
        print(f"   {feat:27s} | {corr:.4f} {stars:6s} | {valid_count}/{len(df_odds)}")
    
    # Summary
    print("\n" + "="*70)
    print("✅ CONCLUSIONES Y RECOMENDACIONES")
    print("="*70)
    print(f"""
   1. El mercado es {overall_accuracy*100:.1f}% preciso en su predicción más probable
      - Local: {market_by_prediction.loc['H', ('Market_Correct', 'mean')]*100:.1f}% ({int(market_by_prediction.loc['H', ('Market_Correct', 'count')])} predicciones)
      - Visitante: {market_by_prediction.loc['A', ('Market_Correct', 'mean')]*100:.1f}% ({int(market_by_prediction.loc['A', ('Market_Correct', 'count')])} predicciones)""")
    
    if 'D' in market_by_prediction.index:
        print(f"      - Empate: {market_by_prediction.loc['D', ('Market_Correct', 'mean')]*100:.1f}% ({int(market_by_prediction.loc['D', ('Market_Correct', 'count')])} predicciones)")
    
    print(f"""
   2. Favoritos CLAROS tienen {accuracy_by_clarity[True]*100:.1f}% precisión vs {accuracy_by_clarity[False]*100:.1f}% dudosos
      → Tu modelo debe enfocarse en partidos con baja FavoriteStrength
   """)
    
    if valid_disagreement.sum() > 0:
        print(f"""   3. Alto desacuerdo entre casas correlaciona con más upsets
      → MarketDisagreement/OddsStd son señales de value betting potencial
   """)
    
    print(f"""   4. Partidos competitivos (IsCompetitiveMatch=1) son más impredecibles
      → Requieren features más sofisticadas
   
   5. Las casas tienen ~{(avg_overround-1)*100:.1f}% de margen (overround)
      → Necesitas edge > {(avg_overround-1)*100:.1f}% para ser rentable
      → Usar AdjustedProb_* elimina este sesgo
   
   🎯 FEATURES RECOMENDADAS para incluir en tu modelo:
      ⭐⭐⭐⭐ MarketProb_Home/Away/Draw (captura expectativa del mercado)
      ⭐⭐⭐⭐ AdjustedProb_* (probabilidades verdaderas sin overround)
      ⭐⭐⭐⭐ ImpliedGoalDiff (predictor directo de resultado)
      ⭐⭐⭐ FavoriteStrength (identifica favoritos claros vs dudosos)
      ⭐⭐⭐ MarketConsensus (confiabilidad de la predicción)
      ⭐⭐ OddsStd_* (volatilidad entre casas = incertidumbre)
      ⭐⭐ Team_AvgMarketProb_L10 (reputación histórica)
      ⭐⭐ IsUnderdog_Home/Away (contexto del partido)
      ⭐ Team_UpsetRate_L10 (equipos que sorprenden)
   
   ⚠️  ADVERTENCIA:
      Tienes {len(df_odds)} partidos con odds ({len(df_odds)/len(df)*100:.1f}% del dataset)
      Suficiente para entrenar pero más datos mejorarían robustez
      
      👉 Considera descargar más odds de football-data.co.uk (2000-2025)
   """)
    
    print("="*70)


if __name__ == '__main__':
    analyze_market_features()
