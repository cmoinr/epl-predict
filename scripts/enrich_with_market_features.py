"""
FASE 1: Enriquecimiento de epl_final.csv con Market Intelligence
Fusiona datos de odds y extrae 25+ features de mercado
"""

import pandas as pd
import numpy as np
from pathlib import Path

def extract_market_features(df_odds):
    """
    Extrae features de mercado de las columnas de odds
    """
    
    print("   📊 Extrayendo features de mercado...")
    
    # Identificar columnas de odds por casa de apuestas
    # Formato: CasaH, CasaD, CasaA (Home, Draw, Away)
    bookmakers = []
    for col in df_odds.columns:
        if col.endswith('H') and col[:-1] + 'D' in df_odds.columns and col[:-1] + 'A' in df_odds.columns:
            bookmaker = col[:-1]
            if len(bookmaker) >= 2:  # Evitar columnas como 'H', 'D', 'A'
                bookmakers.append(bookmaker)
    
    bookmakers = list(set(bookmakers))
    print(f"      Casas encontradas: {len(bookmakers)}")
    
    if len(bookmakers) == 0:
        print("      ⚠️ No se encontraron columnas de odds")
        return pd.DataFrame()
    
    features = pd.DataFrame(index=df_odds.index)
    
    # 1. CONSENSO DEL MERCADO (Probabilidades implícitas promedio)
    home_odds_cols = [b + 'H' for b in bookmakers if b + 'H' in df_odds.columns]
    draw_odds_cols = [b + 'D' for b in bookmakers if b + 'D' in df_odds.columns]
    away_odds_cols = [b + 'A' for b in bookmakers if b + 'A' in df_odds.columns]
    
    if home_odds_cols:
        # Probabilidad implícita = 1 / odds
        home_probs = df_odds[home_odds_cols].apply(lambda x: 1/x, axis=0)
        draw_probs = df_odds[draw_odds_cols].apply(lambda x: 1/x, axis=0)
        away_probs = df_odds[away_odds_cols].apply(lambda x: 1/x, axis=0)
        
        # Promedios
        features['Market_Home_Prob'] = home_probs.mean(axis=1)
        features['Market_Draw_Prob'] = draw_probs.mean(axis=1)
        features['Market_Away_Prob'] = away_probs.mean(axis=1)
        
        # 2. OVERROUND (margen de la casa)
        features['Market_Overround'] = (
            features['Market_Home_Prob'] + 
            features['Market_Draw_Prob'] + 
            features['Market_Away_Prob']
        )
        
        # 3. PROBABILIDADES JUSTAS (sin margen)
        total_prob = features['Market_Overround']
        features['Market_Fair_Home'] = features['Market_Home_Prob'] / total_prob
        features['Market_Fair_Draw'] = features['Market_Draw_Prob'] / total_prob
        features['Market_Fair_Away'] = features['Market_Away_Prob'] / total_prob
        
        # 4. FUERZA DEL FAVORITO
        features['Market_Favorite_Prob'] = features[[
            'Market_Fair_Home', 'Market_Fair_Draw', 'Market_Fair_Away'
        ]].max(axis=1)
        
        features['Market_Favorite'] = features[[
            'Market_Fair_Home', 'Market_Fair_Draw', 'Market_Fair_Away'
        ]].idxmax(axis=1).map({
            'Market_Fair_Home': 1,
            'Market_Fair_Draw': 0,
            'Market_Fair_Away': -1
        })
        
        # 5. CLARIDAD DEL FAVORITO (diferencia con segundo)
        probs_sorted = np.sort(features[[
            'Market_Fair_Home', 'Market_Fair_Draw', 'Market_Fair_Away'
        ]].values, axis=1)
        features['Market_Favorite_Strength'] = probs_sorted[:, -1] - probs_sorted[:, -2]
        
        # 6. DIFERENCIA HOME vs AWAY
        features['Market_HomeAway_Diff'] = (
            features['Market_Fair_Home'] - features['Market_Fair_Away']
        )
        
        # 7. DISPERSIÓN DEL MERCADO (desacuerdo entre casas)
        features['Market_Home_Std'] = home_probs.std(axis=1)
        features['Market_Draw_Std'] = draw_probs.std(axis=1)
        features['Market_Away_Std'] = away_probs.std(axis=1)
        
        features['Market_Uncertainty'] = (
            features['Market_Home_Std'] + 
            features['Market_Draw_Std'] + 
            features['Market_Away_Std']
        ) / 3
        
        # 8. BEST ODDS (mejores cuotas disponibles)
        features['Best_Home_Odds'] = df_odds[home_odds_cols].max(axis=1)
        features['Best_Draw_Odds'] = df_odds[draw_odds_cols].max(axis=1)
        features['Best_Away_Odds'] = df_odds[away_odds_cols].max(axis=1)
        
        # 9. WORST ODDS (peores cuotas - más conservadoras)
        features['Worst_Home_Odds'] = df_odds[home_odds_cols].min(axis=1)
        features['Worst_Draw_Odds'] = df_odds[draw_odds_cols].min(axis=1)
        features['Worst_Away_Odds'] = df_odds[away_odds_cols].min(axis=1)
        
        # 10. RANGO DE ODDS (max - min)
        features['Home_Odds_Range'] = features['Best_Home_Odds'] - features['Worst_Home_Odds']
        features['Draw_Odds_Range'] = features['Best_Draw_Odds'] - features['Worst_Draw_Odds']
        features['Away_Odds_Range'] = features['Best_Away_Odds'] - features['Worst_Away_Odds']
        
    print(f"      ✅ {len(features.columns)} features de mercado extraídas")
    
    return features


def merge_and_enrich():
    """
    Fusiona epl_final.csv con epl_odds.csv y agrega market features
    """
    
    print("="*80)
    print("🔄 FASE 1: ENRIQUECIMIENTO CON MARKET INTELLIGENCE")
    print("="*80)
    print()
    
    # 1. Cargar datasets
    print("1️⃣ Cargando datasets...")
    df_final = pd.read_csv('data/raw/epl_final.csv')
    df_odds = pd.read_csv('data/raw/epl_odds.csv', low_memory=False)
    
    print(f"   epl_final.csv: {len(df_final)} partidos")
    print(f"   epl_odds.csv:  {len(df_odds)} partidos")
    print()
    
    # 2. Crear identificadores únicos
    print("2️⃣ Preparando fusión...")
    
    # Normalizar fechas para matching
    df_final['MatchDate_normalized'] = pd.to_datetime(
        df_final['MatchDate'], format='mixed', dayfirst=True, errors='coerce'
    ).dt.strftime('%Y-%m-%d')
    
    df_odds['Date_normalized'] = pd.to_datetime(
        df_odds['Date'], format='mixed', dayfirst=True, errors='coerce'
    ).dt.strftime('%Y-%m-%d')
    
    # ID único: Season + HomeTeam + AwayTeam
    df_final['match_id'] = (
        df_final['Season'].astype(str) + '_' +
        df_final['HomeTeam'].astype(str) + '_' +
        df_final['AwayTeam'].astype(str)
    )
    
    df_odds['match_id'] = (
        df_odds['Season'].astype(str) + '_' +
        df_odds['HomeTeam'].astype(str) + '_' +
        df_odds['AwayTeam'].astype(str)
    )
    
    print(f"   IDs únicos en epl_final: {df_final['match_id'].nunique()}")
    print(f"   IDs únicos en epl_odds:  {df_odds['match_id'].nunique()}")
    print()
    
    # 3. Extraer market features
    print("3️⃣ Extrayendo market features...")
    market_features = extract_market_features(df_odds)
    df_odds_enriched = pd.concat([df_odds, market_features], axis=1)
    print()
    
    # 4. Fusionar datasets
    print("4️⃣ Fusionando datasets...")
    
    # Seleccionar columnas de odds a fusionar
    odds_cols = ['match_id'] + list(market_features.columns)
    df_odds_for_merge = df_odds_enriched[odds_cols]
    
    # Merge
    df_enriched = df_final.merge(
        df_odds_for_merge,
        on='match_id',
        how='left',
        suffixes=('', '_odds')
    )
    
    # Limpiar columnas temporales
    df_enriched = df_enriched.drop(columns=['match_id', 'MatchDate_normalized'], errors='ignore')
    
    matches_with_odds = df_enriched['Market_Home_Prob'].notna().sum()
    coverage = (matches_with_odds / len(df_enriched)) * 100
    
    print(f"   Total partidos: {len(df_enriched)}")
    print(f"   Con datos de odds: {matches_with_odds} ({coverage:.1f}%)")
    print()
    
    # 5. Backup y guardar
    print("5️⃣ Guardando datasets...")
    
    # Backup
    backup_path = Path('data/raw/epl_final_before_enrichment.csv')
    if not backup_path.exists():
        df_final.to_csv(backup_path, index=False)
        print(f"   ✅ Backup: {backup_path.name}")
    
    # Guardar enriquecido
    output_path = Path('data/processed/epl_final_enriched.csv')
    output_path.parent.mkdir(exist_ok=True)
    df_enriched.to_csv(output_path, index=False)
    
    print(f"   💾 Dataset enriquecido: {output_path}")
    print(f"      Columnas originales: {len(df_final.columns)}")
    print(f"      Columnas nuevas: {len(market_features.columns)}")
    print(f"      Total columnas: {len(df_enriched.columns)}")
    print()
    
    # 6. Estadísticas
    print("6️⃣ Resumen de market features:")
    print("-" * 80)
    
    if matches_with_odds > 0:
        df_with_odds = df_enriched[df_enriched['Market_Home_Prob'].notna()]
        
        print(f"   Overround promedio: {df_with_odds['Market_Overround'].mean():.4f}")
        print(f"   Favorite strength promedio: {df_with_odds['Market_Favorite_Strength'].mean():.4f}")
        print(f"   Market uncertainty promedio: {df_with_odds['Market_Uncertainty'].mean():.4f}")
        print()
        
        # Distribución de favoritos
        favorite_dist = df_with_odds['Market_Favorite'].value_counts()
        print("   Distribución de favoritos según mercado:")
        print(f"      Home favorito: {favorite_dist.get(1, 0)} ({favorite_dist.get(1, 0)/len(df_with_odds)*100:.1f}%)")
        print(f"      Draw favorito: {favorite_dist.get(0, 0)} ({favorite_dist.get(0, 0)/len(df_with_odds)*100:.1f}%)")
        print(f"      Away favorito: {favorite_dist.get(-1, 0)} ({favorite_dist.get(-1, 0)/len(df_with_odds)*100:.1f}%)")
    
    print()
    print("="*80)
    print("✅ FASE 1 COMPLETADA")
    print("="*80)
    print()
    print("📋 PRÓXIMOS PASOS:")
    print("   1. Revisar: data/processed/epl_final_enriched.csv")
    print("   2. FASE 2: Reentrenar modelos con market features")
    print("   3. FASE 3: Implementar Kelly Criterion betting")
    print()
    
    return df_enriched


if __name__ == '__main__':
    df = merge_and_enrich()
