"""
Módulo de Predicción - Usar modelos guardados para predecir nuevos partidos
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional
from feature_engineering import EPLFeatureEngineer


class EPLPredictor:
    """
    Cargar modelos guardados y hacer predicciones en nuevos partidos de la EPL
    
    Ejemplo:
    --------
    predictor = EPLPredictor('models')
    result = predictor.predict_match(df_historical, 'Chelsea', 'Liverpool', '2025-02-22')
    """
    
    def __init__(self, models_dir: str = 'models'):
        """
        Inicializar predictor cargando modelos guardados
        
        Parámetros:
        -----------
        models_dir : str
            Ruta a la carpeta con los modelos guardados
        """
        self.models_dir = Path(models_dir)
        self._load_models()
    
    def _load_models(self):
        """Cargar todos los modelos desde archivos .pkl"""
        try:
            self.rf_result = pickle.load(open(self.models_dir / 'rf_result_model.pkl', 'rb'))
            self.gb_result = pickle.load(open(self.models_dir / 'gb_result_model.pkl', 'rb'))
            self.rf_goals = pickle.load(open(self.models_dir / 'rf_goals_model.pkl', 'rb'))
            self.gb_goals = pickle.load(open(self.models_dir / 'gb_goals_model.pkl', 'rb'))
            self.rf_btts = pickle.load(open(self.models_dir / 'rf_btts_model.pkl', 'rb'))
            self.gb_btts = pickle.load(open(self.models_dir / 'gb_btts_model.pkl', 'rb'))
            self.scaler = pickle.load(open(self.models_dir / 'scaler_model.pkl', 'rb'))
            
            print(f'✅ Modelos cargados desde: {self.models_dir}')
        except FileNotFoundError as e:
            print(f'❌ Error: No se encontraron modelos en {self.models_dir}')
            print(f'   Detalle: {e}')
            raise
    
    def _get_team_recent_stats(self, df: pd.DataFrame, team: str, match_date: str, last_n: int = 5) -> Dict:
        """
        Obtener estadísticas recientes de un equipo (últimos N partidos antes de la fecha)
        """
        date = pd.to_datetime(match_date)
        
        # Partidos donde jugó como local
        home_matches = df[
            (df['HomeTeam'] == team) & 
            (pd.to_datetime(df['MatchDate']) < date)
        ].sort_values('MatchDate').tail(last_n)
        
        # Partidos donde jugó como visitante
        away_matches = df[
            (df['AwayTeam'] == team) & 
            (pd.to_datetime(df['MatchDate']) < date)
        ].sort_values('MatchDate').tail(last_n)
        
        # Combinar todos los partidos recientes
        all_recent = pd.concat([home_matches, away_matches]).sort_values('MatchDate').tail(last_n)
        
        if len(all_recent) == 0:
            # Si no hay datos, usar promedios generales
            return {
                'goals_for': df['FullTimeHomeGoals'].mean(),
                'goals_against': df['FullTimeAwayGoals'].mean(),
                'form': 1.5
            }
        
        # Calcular estadísticas
        goals_for = (
            home_matches['FullTimeHomeGoals'].sum() + 
            away_matches['FullTimeAwayGoals'].sum()
        ) / len(all_recent) if len(all_recent) > 0 else 1.5
        
        goals_against = (
            home_matches['FullTimeAwayGoals'].sum() + 
            away_matches['FullTimeHomeGoals'].sum()
        ) / len(all_recent) if len(all_recent) > 0 else 1.2
        
        # Form (puntos en últimos N partidos)
        form = 0
        for _, match in all_recent.iterrows():
            if match['HomeTeam'] == team:
                result = match['FullTimeResult']
                form += 3 if result == 'H' else (1 if result == 'D' else 0)
            else:
                result = match['FullTimeResult']
                form += 3 if result == 'A' else (1 if result == 'D' else 0)
        
        form = form / len(all_recent) if len(all_recent) > 0 else 1.5
        
        return {
            'goals_for': float(goals_for),
            'goals_against': float(goals_against),
            'form': float(form)
        }
    
    def _create_realistic_features(self, df_historical: pd.DataFrame, home_team: str, 
                                   away_team: str, match_date: str) -> np.ndarray:
        """
        Crear features EXACTAMENTE como se entrenaron (28 features mejoradas)
        CRÍTICO: Debe coincidir exactamente con src/retrain_models_improved.py
        """
        
        date = pd.to_datetime(match_date)
        
        # Para un nuevo partido, usamos datos históricos hasta esa fecha
        historical_until = df_historical[pd.to_datetime(df_historical['MatchDate']) < date].copy()
        
        if len(historical_until) == 0:
            historical_until = df_historical.copy()
        
        # FEATURE ENGINEERING MEJORADO (28 features exactas)
        
        # 1. Estadísticas de forma reciente (últimos 5 partidos)
        result_map = {'H': 3, 'D': 1, 'A': 0}
        away_result_map = {'A': 3, 'D': 1, 'H': 0}
        
        home_recent = historical_until[
            (historical_until['HomeTeam'] == home_team) |
            (historical_until['AwayTeam'] == home_team)
        ].tail(5)
        
        away_recent = historical_until[
            (historical_until['HomeTeam'] == away_team) |
            (historical_until['AwayTeam'] == away_team)
        ].tail(5)
        
        # Calcular forma
        home_form = 0
        for _, match in home_recent.iterrows():
            if match['HomeTeam'] == home_team:
                home_form += 3 if match['FullTimeResult'] == 'H' else (1 if match['FullTimeResult'] == 'D' else 0)
            else:
                home_form += 3 if match['FullTimeResult'] == 'A' else (1 if match['FullTimeResult'] == 'D' else 0)
        home_form = (home_form / len(home_recent)) if len(home_recent) > 0 else 1.5
        
        away_form = 0
        for _, match in away_recent.iterrows():
            if match['HomeTeam'] == away_team:
                away_form += 3 if match['FullTimeResult'] == 'H' else (1 if match['FullTimeResult'] == 'D' else 0)
            else:
                away_form += 3 if match['FullTimeResult'] == 'A' else (1 if match['FullTimeResult'] == 'D' else 0)
        away_form = (away_form / len(away_recent)) if len(away_recent) > 0 else 1.5
        
        # 2. Estadísticas básicas (media histórica)
        home_matches = historical_until[historical_until['HomeTeam'] == home_team]
        away_matches_home = historical_until[historical_until['AwayTeam'] == home_team]
        away_matches_opponent = historical_until[historical_until['AwayTeam'] == away_team]
        home_matches_opponent = historical_until[historical_until['HomeTeam'] == away_team]
        
        # Shots
        home_shots = float(historical_until[historical_until['HomeTeam'] == home_team]['HomeShots'].mean() or 10)
        away_shots = float(historical_until[historical_until['AwayTeam'] == away_team]['AwayShots'].mean() or 8)
        home_shots_on_target = float(historical_until[historical_until['HomeTeam'] == home_team]['HomeShotsOnTarget'].mean() or 4)
        away_shots_on_target = float(historical_until[historical_until['AwayTeam'] == away_team]['AwayShotsOnTarget'].mean() or 3)
        home_corners = float(historical_until[historical_until['HomeTeam'] == home_team]['HomeCorners'].mean() or 5)
        away_corners = float(historical_until[historical_until['AwayTeam'] == away_team]['AwayCorners'].mean() or 4)
        home_fouls = float(historical_until[historical_until['HomeTeam'] == home_team]['HomeFouls'].mean() or 12)
        away_fouls = float(historical_until[historical_until['AwayTeam'] == away_team]['AwayFouls'].mean() or 12)
        home_yellow = float(historical_until[historical_until['HomeTeam'] == home_team]['HomeYellowCards'].mean() or 1.5)
        away_yellow = float(historical_until[historical_until['AwayTeam'] == away_team]['AwayYellowCards'].mean() or 1.5)
        home_red = float(historical_until[historical_until['HomeTeam'] == home_team]['HomeRedCards'].mean() or 0)
        away_red = float(historical_until[historical_until['AwayTeam'] == away_team]['AwayRedCards'].mean() or 0)
        home_ht_goals = float(historical_until[historical_until['HomeTeam'] == home_team]['HalfTimeHomeGoals'].mean() or 0.5)
        away_ht_goals = float(historical_until[historical_until['AwayTeam'] == away_team]['HalfTimeAwayGoals'].mean() or 0.4)
        
        # 3. Poder ofensivo/defensivo mejorado
        home_goals_for = float((
            home_matches['FullTimeHomeGoals'].sum() +
            away_matches_home['FullTimeAwayGoals'].sum()
        ) / (len(home_matches) + len(away_matches_home) + 0.1) or 1.5)
        
        away_goals_for = float((
            home_matches_opponent['FullTimeAwayGoals'].sum() +
            away_matches_opponent['FullTimeAwayGoals'].sum()
        ) / (len(home_matches_opponent) + len(away_matches_opponent) + 0.1) or 1.5)
        
        home_goals_against = float((
            home_matches['FullTimeAwayGoals'].sum() +
            away_matches_home['FullTimeHomeGoals'].sum()
        ) / (len(home_matches) + len(away_matches_home) + 0.1) or 1.2)
        
        away_goals_against = float((
            home_matches_opponent['FullTimeHomeGoals'].sum() +
            away_matches_opponent['FullTimeHomeGoals'].sum()
        ) / (len(home_matches_opponent) + len(away_matches_opponent) + 0.1) or 1.2)
        
        # 4. Diferencia de fuerza (KEY FEATURE)
        strength_diff = ((home_goals_for + (1 - home_goals_against)) -
                        (away_goals_for + (1 - away_goals_against))) * 2
        
        # 5. Ratios ataque/defensa
        home_ratio = home_goals_for / (home_goals_against + 0.1)
        away_ratio = away_goals_for / (away_goals_against + 0.1)
        
        # 6. Ventaja de casa
        home_total_points = len(home_matches[home_matches['FullTimeResult'] == 'H']) * 3
        home_total_points += len(home_matches[home_matches['FullTimeResult'] == 'D'])
        home_advantage = home_total_points / (len(home_matches) + 0.1) if len(home_matches) > 0 else 1.5
        
        away_total_points = len(away_matches_home[away_matches_home['FullTimeResult'] == 'A']) * 3
        away_total_points += len(away_matches_home[away_matches_home['FullTimeResult'] == 'D'])
        away_disadvantage = away_total_points / (len(away_matches_home) + 0.1) if len(away_matches_home) > 0 else 1.0
        
        # 7. Tendencia a draws
        home_draw_tendency = len(home_matches[home_matches['FullTimeResult'] == 'D']) / (len(home_matches) + 0.1) if len(home_matches) > 0 else 0.27
        away_draw_tendency = len(away_matches_home[away_matches_home['FullTimeResult'] == 'D']) / (len(away_matches_home) + 0.1) if len(away_matches_home) > 0 else 0.27
        
        # 8. Features temporales
        month = date.month / 12
        day_of_week = date.dayofweek / 7
        
        # CONSTRUIR VECTOR DE 28 FEATURES (mismo orden que en entrenamiento)
        features = np.array([
            home_shots,                    # 0
            away_shots,                    # 1
            home_shots_on_target,          # 2
            away_shots_on_target,          # 3
            home_corners,                  # 4
            away_corners,                  # 5
            home_fouls,                    # 6
            away_fouls,                    # 7
            home_yellow,                   # 8
            away_yellow,                   # 9
            home_red,                      # 10
            away_red,                      # 11
            home_ht_goals,                 # 12
            away_ht_goals,                 # 13
            home_form,                     # 14
            away_form,                     # 15
            home_goals_for,                # 16
            away_goals_for,                # 17
            home_goals_against,            # 18
            away_goals_against,            # 19
            strength_diff,                 # 20 ← KEY
            home_ratio,                    # 21
            away_ratio,                    # 22
            home_advantage,                # 23
            away_disadvantage,             # 24
            home_draw_tendency,            # 25
            away_draw_tendency,            # 26
            month,                         # 27
        ]).reshape(1, -1)
        
        # Pad si es necesario (para que coincida con scaler)
        n_features = self.scaler.n_features_in_
        if features.shape[1] < n_features:
            padding = np.zeros((1, n_features - features.shape[1]))
            features = np.hstack([features, padding])
        elif features.shape[1] > n_features:
            features = features[:, :n_features]
        
        # Rellenar cualquier NaN que haya quedado
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normalizar
        X_scaled = self.scaler.transform(features)
        
        # Asegurar que no hay NaN después de transformar
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        
        return X_scaled
    
    def predict_match(self, 
                     df_historical: pd.DataFrame,
                     home_team: str,
                     away_team: str,
                     match_date: str,
                     X_train_scaled: np.ndarray = None) -> Dict:
        """
        Predecir resultado y goles para un próximo partido
        
        Parámetros:
        -----------
        df_historical : pd.DataFrame
            Dataset histórico completo
        home_team : str
            Nombre del equipo local
        away_team : str
            Nombre del equipo visitante
        match_date : str
            Fecha del partido (formato: 'YYYY-MM-DD')
        X_train_scaled : np.ndarray, optional
            Features del conjunto de entrenamiento (sin usar en este método)
        
        Retorna:
        --------
        dict : Predicción con resultados y probabilidades
        """
        
        # 1. Validar fecha
        try:
            pd.to_datetime(match_date)
        except:
            raise ValueError(f"Formato de fecha inválido: {match_date}. Usa 'YYYY-MM-DD'")
        
        # 2. Crear features basados en estadísticas reales de equipos
        X_new_scaled = self._create_realistic_features(df_historical, home_team, away_team, match_date)
        
        # 3. PREDICCIÓN DE RESULTADO (1X2)
        pred_result_rf = self.rf_result.predict(X_new_scaled)[0]
        prob_result_rf = self.rf_result.predict_proba(X_new_scaled)[0]
        
        pred_result_gb = self.gb_result.predict(X_new_scaled)[0]
        prob_result_gb = self.gb_result.predict_proba(X_new_scaled)[0]
        
        # 4. PREDICCIÓN DE GOLES TOTALES
        pred_goals_rf = self.rf_goals.predict(X_new_scaled)[0]
        pred_goals_gb = self.gb_goals.predict(X_new_scaled)[0]
        
        # 5. PREDICCIÓN DE AMBOS ANOTAN (BTTS)
        prob_btts_rf = self.rf_btts.predict_proba(X_new_scaled)[0]
        prob_btts_gb = self.gb_btts.predict_proba(X_new_scaled)[0]
        
        # 6. Mapear código numérico a resultado
        result_map = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}
        
        # 7. Construir respuesta detallada
        return {
            'partido': f'{home_team} vs {away_team}',
            'fecha': match_date,
            'resultado': {
                'random_forest': {
                    'prediccion': result_map[int(pred_result_rf)],
                    'confianza': float(max(prob_result_rf) * 100),
                    'probabilidades': {
                        'Away Win': float(prob_result_rf[0] * 100),
                        'Draw': float(prob_result_rf[1] * 100),
                        'Home Win': float(prob_result_rf[2] * 100),
                    }
                },
                'gradient_boosting': {
                    'prediccion': result_map[int(pred_result_gb)],
                    'confianza': float(max(prob_result_gb) * 100),
                    'probabilidades': {
                        'Away Win': float(prob_result_gb[0] * 100),
                        'Draw': float(prob_result_gb[1] * 100),
                        'Home Win': float(prob_result_gb[2] * 100),
                    }
                }
            },
            'goles_totales': {
                'random_forest': float(round(pred_goals_rf, 2)),
                'gradient_boosting': float(round(pred_goals_gb, 2)),
                'promedio': float(round((pred_goals_rf + pred_goals_gb) / 2, 2))
            },
            'ambos_anotan': {
                'random_forest': {
                    'si': float(prob_btts_rf[1] * 100),
                    'no': float(prob_btts_rf[0] * 100)
                },
                'gradient_boosting': {
                    'si': float(prob_btts_gb[1] * 100),
                    'no': float(prob_btts_gb[0] * 100)
                },
                'promedio': {
                    'si': float((prob_btts_rf[1] + prob_btts_gb[1]) * 50),
                    'no': float((prob_btts_rf[0] + prob_btts_gb[0]) * 50)
                }
            }
        }
    
    def predict_batch(self, 
                     df_historical: pd.DataFrame,
                     matches: list,
                     X_train_scaled: np.ndarray = None) -> list:
        """
        Predecir múltiples partidos
        """
        predictions = []
        for match in matches:
            try:
                pred = self.predict_match(
                    df_historical,
                    match.get('home', match.get('home_team')),
                    match.get('away', match.get('away_team')),
                    match.get('date'),
                    X_train_scaled
                )
                predictions.append(pred)
                print(f"✅ {match.get('home', match.get('home_team'))} vs {match.get('away', match.get('away_team'))}")
            except Exception as e:
                print(f"❌ Error: {e}")
        
        return predictions
    
    def print_prediction(self, result: Dict, verbose: bool = True):
        """
        Mostrar predicción en formato legible
        """
        print(f"\n{'='*70}")
        print(f"🔮 PREDICCIÓN EPL")
        print(f"{'='*70}")
        print(f"📅 {result['partido']} ({result['fecha']})")
        print(f"{'='*70}")
        
        print(f"\n📊 RESULTADO (1X2):")
        print(f"\n  🌲 Random Forest:")
        print(f"     Predicción: {result['resultado']['random_forest']['prediccion']}")
        print(f"     Confianza: {result['resultado']['random_forest']['confianza']:.1f}%")
        if verbose:
            probs = result['resultado']['random_forest']['probabilidades']
            print(f"     Detalles: Away {probs['Away Win']:.1f}% | Draw {probs['Draw']:.1f}% | Home {probs['Home Win']:.1f}%")
        
        print(f"\n  ⚡ Gradient Boosting:")
        print(f"     Predicción: {result['resultado']['gradient_boosting']['prediccion']}")
        print(f"     Confianza: {result['resultado']['gradient_boosting']['confianza']:.1f}%")
        if verbose:
            probs = result['resultado']['gradient_boosting']['probabilidades']
            print(f"     Detalles: Away {probs['Away Win']:.1f}% | Draw {probs['Draw']:.1f}% | Home {probs['Home Win']:.1f}%")
        
        print(f"\n⚽ GOLES TOTALES:")
        print(f"  🌲 Random Forest: {result['goles_totales']['random_forest']}")
        print(f"  ⚡ Gradient Boosting: {result['goles_totales']['gradient_boosting']}")
        print(f"  📈 Promedio: {result['goles_totales']['promedio']}")
        
        print(f"\n🥅 AMBOS ANOTAN (BTTS):")
        print(f"  🌲 Random Forest: SI {result['ambos_anotan']['random_forest']['si']:.1f}% | NO {result['ambos_anotan']['random_forest']['no']:.1f}%")
        print(f"  ⚡ Gradient Boosting: SI {result['ambos_anotan']['gradient_boosting']['si']:.1f}% | NO {result['ambos_anotan']['gradient_boosting']['no']:.1f}%")
        print(f"  📈 Promedio: SI {result['ambos_anotan']['promedio']['si']:.1f}% | NO {result['ambos_anotan']['promedio']['no']:.1f}%")
        
        print(f"\n{'='*70}\n")


def save_models(rf_result, gb_result, rf_goals, gb_goals, rf_btts, gb_btts, scaler, 
                output_dir: str = 'models') -> bool:
    """
    Guardar modelos entrenados en archivos .pkl
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    models = {
        'rf_result_model.pkl': rf_result,
        'gb_result_model.pkl': gb_result,
        'rf_goals_model.pkl': rf_goals,
        'gb_goals_model.pkl': gb_goals,
        'rf_btts_model.pkl': rf_btts,
        'gb_btts_model.pkl': gb_btts,
        'scaler_model.pkl': scaler,
    }
    
    for filename, model in models.items():
        try:
            with open(output_path / filename, 'wb') as f:
                pickle.dump(model, f)
            print(f'✅ Guardado: {filename}')
        except Exception as e:
            print(f'❌ Error guardando {filename}: {e}')
            return False
    
    print(f'\n🎉 Todos los modelos guardados en: {output_path}/')
    return True
