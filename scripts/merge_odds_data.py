"""
Script para fusionar epl_odds.csv con epl_final.csv
Agrega features derivadas de odds históricos
"""

import pandas as pd
import numpy as np
from pathlib import Path

def normalize_team_names(df, home_col='HomeTeam', away_col='AwayTeam'):
    """Normaliza nombres de equipos para hacer match"""
    name_mappings = {
        'Man United': 'Manchester United',
        'Man City': 'Manchester City',
        # Agregar más según necesidad
    }
    
    df[home_col] = df[home_col].replace(name_mappings)
    df[away_col] = df[away_col].replace(name_mappings)
    return df

def extract_odds_features(df_odds):
    """
    Extrae features derivadas de las cuotas
    
    Args:
        df_odds: DataFrame con columnas de odds (GBH, IWH, LBH, etc.)
    
    Returns:
        DataFrame con nuevas features
    """
    df = df_odds.copy()
    
    # 1. Promedios de cuotas por resultado
    odds_cols_home = ['GBH', 'IWH', 'LBH', 'SBH', 'WHH']
    odds_cols_draw = ['GBD', 'IWD', 'LBD', 'SBD', 'WHD']
    odds_cols_away = ['GBA', 'IWA', 'LBA', 'SBA', 'WHA']
    
    # Manejo de valores faltantes
    df['AvgOdds_Home'] = df[odds_cols_home].mean(axis=1, skipna=True)
    df['AvgOdds_Draw'] = df[odds_cols_draw].mean(axis=1, skipna=True)
    df['AvgOdds_Away'] = df[odds_cols_away].mean(axis=1, skipna=True)
    
    # 2. Probabilidades implícitas del mercado
    df['MarketProb_Home'] = 1 / df['AvgOdds_Home']
    df['MarketProb_Draw'] = 1 / df['AvgOdds_Draw']
    df['MarketProb_Away'] = 1 / df['AvgOdds_Away']
    
    # 3. Overround (margen de las casas)
    df['Overround'] = df['MarketProb_Home'] + df['MarketProb_Draw'] + df['MarketProb_Away']
    
    # 4. Probabilidades ajustadas (quitando margen)
    df['AdjustedProb_Home'] = df['MarketProb_Home'] / df['Overround']
    df['AdjustedProb_Draw'] = df['MarketProb_Draw'] / df['Overround']
    df['AdjustedProb_Away'] = df['MarketProb_Away'] / df['Overround']
    
    # 5. Desviación estándar (consenso del mercado)
    df['OddsStd_Home'] = df[odds_cols_home].std(axis=1, skipna=True)
    df['OddsStd_Draw'] = df[odds_cols_draw].std(axis=1, skipna=True)
    df['OddsStd_Away'] = df[odds_cols_away].std(axis=1, skipna=True)
    
    # 6. Market Confidence (baja desviación = alto consenso)
    df['MarketConsensus'] = 1 / (1 + df[['OddsStd_Home', 'OddsStd_Draw', 'OddsStd_Away']].mean(axis=1))
    
    # 7. Favorito del mercado
    df['MarketFavorite'] = df[['MarketProb_Home', 'MarketProb_Draw', 'MarketProb_Away']].idxmax(axis=1)
    df['MarketFavorite'] = df['MarketFavorite'].map({
        'MarketProb_Home': 'H',
        'MarketProb_Draw': 'D',
        'MarketProb_Away': 'A'
    })
    
    # 8. Fuerza del favorito (diferencia entre 1º y 2º)
    probs = df[['MarketProb_Home', 'MarketProb_Draw', 'MarketProb_Away']]
    df['FavoriteStrength'] = probs.max(axis=1) - probs.apply(lambda x: x.nlargest(2).iloc[-1], axis=1)
    
    # 9. Expectativa de goles según mercado (inverso de odds bajas = partido cerrado)
    df['MarketExpectedGoals'] = 1 / df['AvgOdds_Home'] + 1 / df['AvgOdds_Away']
    
    # 10. Rango de cuotas (dispersión de mercado)
    df['OddsRange_Home'] = df[odds_cols_home].max(axis=1) - df[odds_cols_home].min(axis=1)
    df['OddsRange_Draw'] = df[odds_cols_draw].max(axis=1) - df[odds_cols_draw].min(axis=1)
    df['OddsRange_Away'] = df[odds_cols_away].max(axis=1) - df[odds_cols_away].min(axis=1)
    
    return df

def merge_datasets(epl_final_path, epl_odds_path, output_path):
    """
    Fusiona epl_final.csv con epl_odds.csv
    
    Args:
        epl_final_path: Ruta a epl_final.csv
        epl_odds_path: Ruta a epl_odds.csv
        output_path: Ruta de salida para el dataset enriquecido
    """
    print("Cargando datasets...")
    df_final = pd.read_csv(epl_final_path)
    df_odds = pd.read_csv(epl_odds_path)
    
    print(f"epl_final.csv: {df_final.shape}")
    print(f"epl_odds.csv: {df_odds.shape}")
    
    # Normalizar nombres de equipos
    df_final = normalize_team_names(df_final)
    df_odds = normalize_team_names(df_odds)
    
    # Convertir fechas
    df_final['MatchDate'] = pd.to_datetime(df_final['MatchDate'])
    df_odds['Date'] = pd.to_datetime(df_odds['Date'], format='%d/%m/%y')
    
    # Extraer features de odds
    print("\nExtrayendo features de mercado...")
    df_odds_enriched = extract_odds_features(df_odds)
    
    # Seleccionar columnas relevantes para merge
    odds_features = [
        'Date', 'HomeTeam', 'AwayTeam',
        'AvgOdds_Home', 'AvgOdds_Draw', 'AvgOdds_Away',
        'MarketProb_Home', 'MarketProb_Draw', 'MarketProb_Away',
        'AdjustedProb_Home', 'AdjustedProb_Draw', 'AdjustedProb_Away',
        'Overround', 'MarketConsensus', 'MarketFavorite', 
        'FavoriteStrength', 'MarketExpectedGoals',
        'OddsStd_Home', 'OddsStd_Draw', 'OddsStd_Away',
        'OddsRange_Home', 'OddsRange_Draw', 'OddsRange_Away',
        # Odds individuales por casa
        'GBH', 'GBD', 'GBA',  # Betfair
        'WHH', 'WHD', 'WHA',  # William Hill
    ]
    
    df_odds_clean = df_odds_enriched[odds_features].copy()
    
    # Merge
    print("\nFusionando datasets...")
    df_merged = df_final.merge(
        df_odds_clean,
        left_on=['MatchDate', 'HomeTeam', 'AwayTeam'],
        right_on=['Date', 'HomeTeam', 'AwayTeam'],
        how='left'
    )
    
    # Eliminar columna Date duplicada
    df_merged = df_merged.drop('Date', axis=1)
    
    # Reporte
    matches_with_odds = df_merged['AvgOdds_Home'].notna().sum()
    print(f"\n✅ Merge completado:")
    print(f"   Total de partidos: {len(df_merged)}")
    print(f"   Partidos con odds: {matches_with_odds} ({matches_with_odds/len(df_merged)*100:.1f}%)")
    print(f"   Nuevas columnas agregadas: {len(odds_features) - 3}")
    
    # Guardar
    df_merged.to_csv(output_path, index=False)
    print(f"\n💾 Dataset enriquecido guardado en: {output_path}")
    
    # Mostrar muestra
    print("\n📊 Muestra de nuevas features:")
    print(df_merged[['MatchDate', 'HomeTeam', 'AwayTeam', 'FullTimeResult', 
                     'MarketProb_Home', 'MarketProb_Draw', 'MarketProb_Away',
                     'MarketFavorite', 'FavoriteStrength']].head(10))
    
    return df_merged

if __name__ == '__main__':
    # Rutas
    base_path = Path(__file__).parent.parent
    epl_final = base_path / 'data' / 'raw' / 'epl_final.csv'
    epl_odds = base_path / 'data' / 'raw' / 'epl_odds.csv'
    output = base_path / 'data' / 'processed' / 'epl_enriched_with_odds.csv'
    
    # Ejecutar merge
    df_enriched = merge_datasets(epl_final, epl_odds, output)
    
    print("\n" + "="*60)
    print("🎯 PRÓXIMOS PASOS:")
    print("="*60)
    print("1. Re-entrenar modelos con las nuevas features de mercado")
    print("2. Analizar correlación entre MarketProb y resultados reales")
    print("3. Crear modelo ensemble (ML + Market wisdom)")
    print("4. Backtesting de value betting en temporada 2000/01")
