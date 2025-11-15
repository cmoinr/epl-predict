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
        Crear features realistas para un nuevo partido basado en estadísticas de equipos
        """
        
        # Obtener estadísticas de ambos equipos
        home_stats = self._get_team_recent_stats(df_historical, home_team, match_date)
        away_stats = self._get_team_recent_stats(df_historical, away_team, match_date)
        
        # Calcular H2H
        h2h_matches = df_historical[
            ((df_historical['HomeTeam'] == home_team) & (df_historical['AwayTeam'] == away_team)) |
            ((df_historical['HomeTeam'] == away_team) & (df_historical['AwayTeam'] == home_team))
        ]
        
        home_h2h_wins = len(h2h_matches[
            (h2h_matches['HomeTeam'] == home_team) & 
            (h2h_matches['FullTimeResult'] == 'H')
        ])
        total_h2h = len(h2h_matches)
        h2h_win_rate = home_h2h_wins / total_h2h if total_h2h > 0 else 0.33
        
        # Construir feature vector con estructura realista
        feature_values = np.array([
            home_stats['form'],                                      # HomeTeam_Form
            away_stats['form'],                                      # AwayTeam_Form
            h2h_win_rate,                                           # H2H_HomeTeamWins
            home_stats['goals_for'],                                # HomeTeam_GoalsFor
            home_stats['goals_against'],                            # HomeTeam_GoalsAgainst
            away_stats['goals_for'],                                # AwayTeam_GoalsFor
            away_stats['goals_against'],                            # AwayTeam_GoalsAgainst
            0.3,                                                     # HomeAdvantage
            pd.to_datetime(match_date).month / 12,                  # Month (normalizado)
            pd.to_datetime(match_date).dayofweek / 7,               # DayOfWeek (normalizado)
        ]).reshape(1, -1)
        
        # Ajustar tamaño al número de features esperados por el scaler
        n_features = self.scaler.n_features_in_
        if feature_values.shape[1] < n_features:
            # Rellenar con ceros si es necesario
            padding = np.zeros((1, n_features - feature_values.shape[1]))
            feature_values = np.hstack([feature_values, padding])
        elif feature_values.shape[1] > n_features:
            # Truncar si hay más
            feature_values = feature_values[:, :n_features]
        
        # Normalizar
        X_scaled = self.scaler.transform(feature_values)
        
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
        
        # 5. Mapear código numérico a resultado
        result_map = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}
        
        # 6. Construir respuesta detallada
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
        
        print(f"\n{'='*70}\n")


def save_models(rf_result, gb_result, rf_goals, gb_goals, scaler, 
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
