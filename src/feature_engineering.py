"""
Feature Engineering para Premier League ML
Crea features derivadas para predicción de resultados y goles
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')


class EPLFeatureEngineer:
    """Ingeniero de features para datos de Premier League"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Inicializa con el dataset
        
        Args:
            df: DataFrame con datos EPL
        """
        self.df = df.copy()
        self.df['MatchDate'] = pd.to_datetime(self.df['MatchDate'])
        self.df = self.df.sort_values('MatchDate').reset_index(drop=True)
        
    def calculate_team_form(self, window: int = 5) -> pd.DataFrame:
        """
        Calcula la forma reciente de cada equipo (últimos N partidos)
        
        La forma se calcula como:
        - Victorias: 3 puntos
        - Empates: 1 punto
        - Derrotas: 0 puntos
        
        Args:
            window: Número de partidos previos a considerar (default: 5)
        
        Returns:
            DataFrame con columnas de form
        """
        df = self.df.copy()
        
        # Mapa de resultados
        result_map = {'H': 3, 'D': 1, 'A': 0}
        
        # Crear columnas para form del equipo local y visitante
        df['HomeTeam_Form'] = 0
        df['AwayTeam_Form'] = 0
        
        # Calcular form para cada equipo
        teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
        
        for team in teams:
            # Partidos del equipo (como local y visitante)
            home_matches = df[df['HomeTeam'] == team].index
            away_matches = df[df['AwayTeam'] == team].index
            
            # Calcular form para partidos como local
            for idx in home_matches:
                if idx > 0:
                    # Últimos window partidos como local (antes de este partido)
                    prev_home = df[(df['HomeTeam'] == team) & (df.index < idx)].tail(window)
                    if len(prev_home) > 0:
                        form = sum([result_map[r] for r in prev_home['FullTimeResult']])
                        df.at[idx, 'HomeTeam_Form'] = form / len(prev_home)
            
            # Calcular form para partidos como visitante
            for idx in away_matches:
                if idx > 0:
                    prev_away = df[(df['AwayTeam'] == team) & (df.index < idx)].tail(window)
                    if len(prev_away) > 0:
                        form = sum([result_map[r if r != 'H' else 'A'] if r != 'H' else 0 for r in prev_away['FullTimeResult']])
                        # Ajustar: en FullTimeResult, 'A' significa que ganó el visitante
                        form = sum([3 if r == 'A' else (1 if r == 'D' else 0) for r in prev_away['FullTimeResult']])
                        df.at[idx, 'AwayTeam_Form'] = form / len(prev_away)
        
        return df[['HomeTeam_Form', 'AwayTeam_Form']]
    
    def calculate_goals_statistics(self, window: int = 10) -> pd.DataFrame:
        """
        Calcula estadísticas de goles (promedio, defensiva, etc.)
        
        Args:
            window: Ventana de partidos para promedios
        
        Returns:
            DataFrame con columnas de goles
        """
        df = self.df.copy()
        
        teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
        
        features = {}
        
        for team in teams:
            # Partidos como local
            home_matches = df[df['HomeTeam'] == team]
            away_matches = df[df['AwayTeam'] == team]
            
            # Goles promedio como local/visitante
            if len(home_matches) > 0:
                features[f'{team}_HomeGoalsFor'] = home_matches['FullTimeHomeGoals'].rolling(window=window, min_periods=1).mean().shift(1)
                features[f'{team}_HomeGoalsAgainst'] = home_matches['FullTimeAwayGoals'].rolling(window=window, min_periods=1).mean().shift(1)
            
            if len(away_matches) > 0:
                features[f'{team}_AwayGoalsFor'] = away_matches['FullTimeAwayGoals'].rolling(window=window, min_periods=1).mean().shift(1)
                features[f'{team}_AwayGoalsAgainst'] = away_matches['FullTimeHomeGoals'].rolling(window=window, min_periods=1).mean().shift(1)
        
        return pd.DataFrame(features)
    
    def calculate_head_to_head(self, window: int = 5) -> pd.DataFrame:
        """
        Calcula estadísticas histórico H2H entre equipos
        
        Args:
            window: Últimos N enfrentamientos a considerar
        
        Returns:
            DataFrame con columnas H2H
        """
        df = self.df.copy()
        
        h2h_features = []
        
        for idx, row in df.iterrows():
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']
            match_date = row['MatchDate']
            
            # Histórico entre estos dos equipos (antes de este partido)
            h2h = df[
                ((df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team) |
                 (df['HomeTeam'] == away_team) & (df['AwayTeam'] == home_team)) &
                (df['MatchDate'] < match_date)
            ].tail(window)
            
            if len(h2h) > 0:
                # Victorias del home team en H2H
                home_wins = len(h2h[((h2h['HomeTeam'] == home_team) & (h2h['FullTimeResult'] == 'H')) |
                                     ((h2h['AwayTeam'] == home_team) & (h2h['FullTimeResult'] == 'A'))])
                
                h2h_features.append({
                    'H2H_HomeTeamWins': home_wins / len(h2h),
                    'H2H_Matches': len(h2h),
                    'H2H_GoalsFor': h2h[h2h['HomeTeam'] == home_team]['FullTimeHomeGoals'].sum() if len(h2h[h2h['HomeTeam'] == home_team]) > 0 else 0,
                })
            else:
                h2h_features.append({
                    'H2H_HomeTeamWins': 0.5,
                    'H2H_Matches': 0,
                    'H2H_GoalsFor': 0,
                })
        
        return pd.DataFrame(h2h_features)
    
    def calculate_home_advantage(self) -> pd.DataFrame:
        """
        Calcula ventaja de jugar en casa
        
        Returns:
            DataFrame con feature de ventaja
        """
        df = self.df.copy()
        
        home_advantage = []
        
        for team in pd.concat([df['HomeTeam'], df['AwayTeam']]).unique():
            home_matches = df[df['HomeTeam'] == team]
            away_matches = df[df['AwayTeam'] == team]
            
            home_points = (home_matches['FullTimeResult'] == 'H').sum() * 3 + (home_matches['FullTimeResult'] == 'D').sum()
            away_points = (away_matches['FullTimeResult'] == 'A').sum() * 3 + (away_matches['FullTimeResult'] == 'D').sum()
            
            if len(home_matches) + len(away_matches) > 0:
                home_adv = (home_points / len(home_matches)) - (away_points / len(away_matches)) if len(home_matches) > 0 and len(away_matches) > 0 else 0
                home_advantage.append({'Team': team, 'HomeAdvantage': home_adv})
        
        ha_df = pd.DataFrame(home_advantage)
        
        result = []
        for idx, row in df.iterrows():
            home_team = row['HomeTeam']
            ha = ha_df[ha_df['Team'] == home_team]['HomeAdvantage'].values
            result.append({'HomeAdvantage': ha[0] if len(ha) > 0 else 0})
        
        return pd.DataFrame(result)
    
    def calculate_shoot_statistics(self, window: int = 5) -> pd.DataFrame:
        """
        Calcula estadísticas de disparos
        
        Args:
            window: Ventana de partidos
        
        Returns:
            DataFrame con features de tiros
        """
        df = self.df.copy()
        
        features = {
            'HomeShots_Avg': df['HomeShots'].rolling(window=window, min_periods=1).mean().shift(1),
            'AwayShots_Avg': df['AwayShots'].rolling(window=window, min_periods=1).mean().shift(1),
            'HomeShotsOnTarget_Avg': df['HomeShotsOnTarget'].rolling(window=window, min_periods=1).mean().shift(1),
            'AwayShotsOnTarget_Avg': df['AwayShotsOnTarget'].rolling(window=window, min_periods=1).mean().shift(1),
        }
        
        return pd.DataFrame(features)
    
    def create_target_variables(self) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Crea variables target para predicción
        
        Returns:
            Tuple con (features, target_resultado, target_goles)
        """
        df = self.df.copy()
        
        # Target 1: Resultado (1X2)
        # 0 = Away Win (A), 1 = Draw (D), 2 = Home Win (H)
        result_map = {'A': 0, 'D': 1, 'H': 2}
        target_result = df['FullTimeResult'].map(result_map)
        
        # Target 2: Goles totales
        target_goals = df['FullTimeHomeGoals'] + df['FullTimeAwayGoals']
        
        return df, target_result, target_goals
    
    def engineer_features(self) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        Crea todas las features y targets
        
        Returns:
            Tuple con (X features, y result, y goals)
        """
        print("🔧 Creando features...")
        
        df = self.df.copy()
        
        # Features numéricas base (ya están en el dataset)
        feature_cols = [
            'HomeShots', 'AwayShots',
            'HomeShotsOnTarget', 'AwayShotsOnTarget',
            'HomeCorners', 'AwayCorners',
            'HomeFouls', 'AwayFouls',
            'HomeYellowCards', 'AwayYellowCards',
            'HomeRedCards', 'AwayRedCards',
            'HalfTimeHomeGoals', 'HalfTimeAwayGoals'
        ]
        
        X = df[feature_cols].copy()
        
        # Agregar features derivadas
        print("  → Form de equipos...")
        form = self.calculate_team_form(window=5)
        X = pd.concat([X, form], axis=1)
        
        print("  → Estadísticas de goles...")
        goals_stats = self.calculate_goals_statistics(window=10)
        X = pd.concat([X, goals_stats], axis=1)
        
        print("  → Ventaja de casa...")
        home_adv = self.calculate_home_advantage()
        X = pd.concat([X, home_adv], axis=1)
        
        print("  → Estadísticas de tiros...")
        shoot_stats = self.calculate_shoot_statistics(window=5)
        X = pd.concat([X, shoot_stats], axis=1)
        
        # Agregar features temporales
        X['Month'] = df['Month']
        X['DayOfWeek'] = df['DayOfWeek']
        X['Season_Year'] = df['Year']
        
        # Targets
        result_map = {'A': 0, 'D': 1, 'H': 2}
        y_result = df['FullTimeResult'].map(result_map)
        y_goals = df['FullTimeHomeGoals'] + df['FullTimeAwayGoals']
        
        print(f"✅ Features creadas: {X.shape[1]} columnas")
        
        return X, y_result, y_goals


def prepare_training_data(
    X: pd.DataFrame,
    y_result: pd.Series,
    y_goals: pd.Series,
    test_size: float = 0.2,
    fill_method: str = 'forward'
) -> Dict:
    """
    Prepara datos para entrenamiento (manejo de NaNs y split)
    
    Args:
        X: Features
        y_result: Target resultado
        y_goals: Target goles
        test_size: Porcentaje de test
        fill_method: Cómo llenar NaNs ('forward', 'mean', etc.)
    
    Returns:
        Dict con datos de train/test
    """
    print("\n📊 Preparando datos para entrenamiento...")
    
    # Llenar NaNs
    if fill_method == 'forward':
        X_filled = X.fillna(method='ffill').fillna(method='bfill')
    else:
        X_filled = X.fillna(X.mean())
    
    print(f"  → NaNs restantes: {X_filled.isnull().sum().sum()}")
    
    # Split temporal (no aleatorio para series de tiempo)
    split_idx = int(len(X_filled) * (1 - test_size))
    
    X_train = X_filled[:split_idx]
    X_test = X_filled[split_idx:]
    
    y_result_train = y_result[:split_idx]
    y_result_test = y_result[split_idx:]
    
    y_goals_train = y_goals[:split_idx]
    y_goals_test = y_goals[split_idx:]
    
    print(f"  → Train: {len(X_train)} muestras")
    print(f"  → Test: {len(X_test)} muestras")
    
    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_result_train': y_result_train,
        'y_result_test': y_result_test,
        'y_goals_train': y_goals_train,
        'y_goals_test': y_goals_test,
    }


if __name__ == '__main__':
    print("🔧 Módulo de Feature Engineering - EPL ML")
    print("\nUso:")
    print("  1. from src.feature_engineering import EPLFeatureEngineer")
    print("  2. engineer = EPLFeatureEngineer(df)")
    print("  3. X, y_result, y_goals = engineer.engineer_features()")
