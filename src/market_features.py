"""
Features Avanzadas usando Datos de Mercado (Odds)
Integra sabiduría del mercado con estadísticas tradicionales
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple


class MarketBasedFeatures:
    """Extrae features avanzadas basadas en odds del mercado"""
    
    @staticmethod
    def add_market_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        Agrega features derivadas de odds históricas
        
        Args:
            df: DataFrame con columnas de odds (AvgOdds_Home, etc.)
        
        Returns:
            DataFrame con nuevas features de mercado
        """
        df = df.copy()
        
        # 1. Market Surprise: diferencia entre resultado real y expectativa del mercado
        df['MarketSurprise_Home'] = np.where(
            df['FullTimeResult'] == 'H',
            1 - df['MarketProb_Home'],  # Si ganó local, cuánto subestimó el mercado
            -df['MarketProb_Home']      # Si no ganó, cuánto sobreestimó
        )
        
        # 2. Underdog indicator
        df['IsUnderdog_Home'] = (df['MarketProb_Home'] < df['MarketProb_Away']).astype(int)
        df['IsUnderdog_Away'] = (df['MarketProb_Away'] < df['MarketProb_Home']).astype(int)
        
        # 3. Market Efficiency: qué tan bien predijo el mercado
        df['MarketAccuracy'] = np.where(
            df['FullTimeResult'] == df['MarketFavorite'],
            1, 0
        )
        
        # 4. Upset probability (underdog wins)
        df['IsUpset'] = (
            ((df['FullTimeResult'] == 'H') & (df['MarketProb_Home'] < df['MarketProb_Away'])) |
            ((df['FullTimeResult'] == 'A') & (df['MarketProb_Away'] < df['MarketProb_Home']))
        ).astype(int)
        
        # 5. Competitive match indicator (cuotas similares)
        prob_diff = abs(df['MarketProb_Home'] - df['MarketProb_Away'])
        df['IsCompetitiveMatch'] = (prob_diff < 0.15).astype(int)
        
        # 6. Expected value si apostaras al favorito
        df['FavoriteEV'] = np.where(
            df['MarketProb_Home'] > df['MarketProb_Away'],
            df['MarketProb_Home'] * df['AvgOdds_Home'] - 1,
            df['MarketProb_Away'] * df['AvgOdds_Away'] - 1
        )
        
        # 7. Volatilidad de cuotas entre casas (información dispersa)
        df['MarketDisagreement'] = (
            df['OddsStd_Home'] + df['OddsStd_Draw'] + df['OddsStd_Away']
        ) / 3
        
        # 8. Implied goal difference según mercado
        df['ImpliedGoalDiff'] = df['MarketProb_Home'] - df['MarketProb_Away']
        
        return df
    
    @staticmethod
    def calculate_rolling_market_features(df: pd.DataFrame, window: int = 10) -> pd.DataFrame:
        """
        Calcula features rodantes basadas en historial de mercado del equipo
        
        Args:
            df: DataFrame con features de mercado
            window: Ventana de partidos
        
        Returns:
            DataFrame con features rodantes
        """
        df = df.copy()
        df = df.sort_values('MatchDate').reset_index(drop=True)
        
        teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
        
        # Inicializar columnas
        df['Team_AvgMarketProb_L10'] = np.nan
        df['Team_MarketSurpriseRate_L10'] = np.nan
        df['Team_UpsetRate_L10'] = np.nan
        
        for team in teams:
            # Índices de partidos del equipo
            home_idx = df[df['HomeTeam'] == team].index
            away_idx = df[df['AwayTeam'] == team].index
            
            # Para partidos como local
            for idx in home_idx:
                # Últimos N partidos del equipo (como local o visitante)
                prev_matches = df[
                    ((df['HomeTeam'] == team) | (df['AwayTeam'] == team)) & 
                    (df.index < idx)
                ].tail(window)
                
                if len(prev_matches) > 0:
                    # Promedio de probabilidad del mercado a favor de este equipo
                    team_probs = []
                    for _, match in prev_matches.iterrows():
                        if match['HomeTeam'] == team:
                            team_probs.append(match['MarketProb_Home'])
                        else:
                            team_probs.append(match['MarketProb_Away'])
                    
                    df.at[idx, 'Team_AvgMarketProb_L10'] = np.mean(team_probs)
                    
                    # Tasa de sorpresas (mercado se equivocó)
                    surprises = prev_matches['IsUpset'].sum()
                    df.at[idx, 'Team_UpsetRate_L10'] = surprises / len(prev_matches)
            
            # Para partidos como visitante
            for idx in away_idx:
                prev_matches = df[
                    ((df['HomeTeam'] == team) | (df['AwayTeam'] == team)) & 
                    (df.index < idx)
                ].tail(window)
                
                if len(prev_matches) > 0:
                    team_probs = []
                    for _, match in prev_matches.iterrows():
                        if match['HomeTeam'] == team:
                            team_probs.append(match['MarketProb_Home'])
                        else:
                            team_probs.append(match['MarketProb_Away'])
                    
                    df.at[idx, 'Team_AvgMarketProb_L10'] = np.mean(team_probs)
                    
                    surprises = prev_matches['IsUpset'].sum()
                    df.at[idx, 'Team_UpsetRate_L10'] = surprises / len(prev_matches)
        
        return df
    
    @staticmethod
    def create_ensemble_features(df: pd.DataFrame, model_predictions: Dict) -> pd.DataFrame:
        """
        Combina predicciones del modelo ML con probabilidades del mercado
        
        Args:
            df: DataFrame con datos
            model_predictions: Dict con predicciones del modelo por partido
        
        Returns:
            DataFrame con features ensemble
        """
        df = df.copy()
        
        # Agregar predicciones del modelo
        if model_predictions:
            df['ModelProb_Home'] = df.index.map(lambda x: model_predictions.get(x, {}).get('home', np.nan))
            df['ModelProb_Draw'] = df.index.map(lambda x: model_predictions.get(x, {}).get('draw', np.nan))
            df['ModelProb_Away'] = df.index.map(lambda x: model_predictions.get(x, {}).get('away', np.nan))
            
            # Diferencia entre modelo y mercado (edge potencial)
            df['Edge_Home'] = df['ModelProb_Home'] - df['MarketProb_Home']
            df['Edge_Draw'] = df['ModelProb_Draw'] - df['MarketProb_Draw']
            df['Edge_Away'] = df['ModelProb_Away'] - df['MarketProb_Away']
            
            # Máximo edge
            df['MaxEdge'] = df[['Edge_Home', 'Edge_Draw', 'Edge_Away']].max(axis=1)
            
            # Consenso: modelo y mercado de acuerdo
            df['ModelMarketConsensus'] = (
                (df['ModelProb_Home'].idxmax() == df['MarketProb_Home'].idxmax())
            ).astype(int)
            
            # Promedio ponderado: 70% modelo, 30% mercado
            df['EnsembleProb_Home'] = 0.7 * df['ModelProb_Home'] + 0.3 * df['MarketProb_Home']
            df['EnsembleProb_Draw'] = 0.7 * df['ModelProb_Draw'] + 0.3 * df['MarketProb_Draw']
            df['EnsembleProb_Away'] = 0.7 * df['ModelProb_Away'] + 0.3 * df['MarketProb_Away']
        
        return df


def integrate_market_intelligence(df_path: str, output_path: str = None) -> pd.DataFrame:
    """
    Pipeline completo de integración de inteligencia de mercado
    
    Args:
        df_path: Ruta al dataset enriquecido con odds
        output_path: Ruta de salida (opcional)
    
    Returns:
        DataFrame con todas las features de mercado
    """
    print("📊 Cargando dataset...")
    df = pd.read_csv(df_path)
    df['MatchDate'] = pd.to_datetime(df['MatchDate'])
    
    original_cols = df.columns.tolist()
    
    print("⚙️  Generando features de mercado...")
    market_features = MarketBasedFeatures()
    
    # 1. Features básicas de mercado
    df = market_features.add_market_features(df)
    
    # 2. Features rodantes
    df = market_features.calculate_rolling_market_features(df, window=10)
    
    new_cols = [col for col in df.columns if col not in original_cols]
    
    print(f"✅ {len(new_cols)} nuevas features creadas:")
    for col in new_cols:
        print(f"   - {col}")
    
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"\n💾 Dataset guardado en: {output_path}")
    
    return df


if __name__ == '__main__':
    from pathlib import Path
    
    base_path = Path(__file__).parent.parent
    input_file = base_path / 'data' / 'processed' / 'epl_enriched_with_odds.csv'
    output_file = base_path / 'data' / 'processed' / 'epl_with_market_intelligence.csv'
    
    if not input_file.exists():
        print("❌ Error: primero ejecuta merge_odds_data.py")
        print(f"   Archivo esperado: {input_file}")
    else:
        df_final = integrate_market_intelligence(str(input_file), str(output_file))
        
        print("\n" + "="*70)
        print("🎯 FEATURES DE MERCADO DISPONIBLES PARA ENTRENAMIENTO")
        print("="*70)
        print("\n📈 Features Estáticas:")
        print("   - MarketProb_Home/Draw/Away: Probabilidades implícitas del mercado")
        print("   - MarketConsensus: Consenso entre casas de apuestas")
        print("   - FavoriteStrength: Qué tan claro es el favorito")
        print("   - IsCompetitiveMatch: Partidos parejos")
        
        print("\n🔄 Features Dinámicas:")
        print("   - Team_AvgMarketProb_L10: Percepción histórica del mercado")
        print("   - Team_UpsetRate_L10: Frecuencia de sorpresas")
        print("   - MarketSurprise: Desviación del resultado esperado")
        
        print("\n🎲 Para Value Betting:")
        print("   - Edge_Home/Draw/Away: Diferencia modelo vs mercado")
        print("   - MaxEdge: Mejor oportunidad de value")
