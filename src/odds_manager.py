"""
Módulo de Gestión de Odds - Cargar, procesar y validar odds de mercado
Integración con múltiples APIs y formatos de datos
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json
import requests
from urllib.parse import urlencode


class OddsManager:
    """
    Gestor centralizado de odds de mercado
    
    Soporta:
    - Carga de datos históricos (CSV, JSON)
    - Conexión con APIs (odds-api.com, football-data.org)
    - Validación y limpieza de datos
    - Conversión de formatos (decimal, fraccionario, americano)
    
    Ejemplo:
    --------
    manager = OddsManager()
    df_odds = manager.load_historical_odds('odds_data.csv')
    current_odds = manager.fetch_odds_live('soccer_epl')
    """
    
    def __init__(self):
        """Inicializar el gestor de odds"""
        self.df_odds = None
        self.api_key = None
        self.odds_history = []
        
        # Mapeo de formatos de cuotas
        self.result_map = {
            'H': 'Home Win',
            '1': 'Home Win',
            'Home': 'Home Win',
            'A': 'Away Win',
            '2': 'Away Win',
            'Away': 'Away Win',
            'D': 'Draw',
            'X': 'Draw',
            '1X': 'Draw'
        }
    
    def load_historical_odds(self, filepath: str) -> pd.DataFrame:
        """
        Cargar odds históricos desde archivo CSV
        
        Parámetros:
        -----------
        filepath : str
            Ruta a archivo CSV con odds históricos
        
        Retorna:
        --------
        pd.DataFrame con columnas: match_id, date, home_team, away_team, 
                                   home_win_odds, draw_odds, away_win_odds,
                                   result, home_goals, away_goals
        """
        try:
            df = pd.read_csv(filepath)
            
            # Validar columnas requeridas
            required_cols = ['date', 'home_team', 'away_team', 'home_win_odds', 
                           'draw_odds', 'away_win_odds', 'result']
            
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Faltan columnas: {missing_cols}")
            
            # Validar formatos
            df['date'] = pd.to_datetime(df['date'])
            df = df.sort_values('date')
            
            # Validar que las cuotas sean numéricas y positivas
            odds_cols = ['home_win_odds', 'draw_odds', 'away_win_odds']
            for col in odds_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                if df[col].isna().any():
                    print(f"⚠️  Advertencia: {col} contiene valores no numéricos")
            
            # Validar rango típico de cuotas (1.01 a 100)
            for col in odds_cols:
                invalid = (df[col] < 1.01) | (df[col] > 100)
                if invalid.any():
                    print(f"⚠️  {col}: {invalid.sum()} valores fuera del rango típico")
            
            self.df_odds = df
            print(f'✅ Odds cargados: {filepath} ({len(df)} partidos)')
            return df
        
        except FileNotFoundError:
            print(f'❌ Archivo no encontrado: {filepath}')
            raise
        except Exception as e:
            print(f'❌ Error cargando odds: {e}')
            raise
    
    def fetch_odds_api(self, api_key: str, sport: str = 'soccer_epl', 
                       region: str = 'uk', markets: str = 'h2h') -> pd.DataFrame:
        """
        Obtener odds en vivo desde odds-api.com
        
        Parámetros:
        -----------
        api_key : str
            API key de odds-api.com (registrarse en https://odds-api.com)
        sport : str
            Deporte y liga (default: soccer_epl para Premier League)
        region : str
            Región (uk, au, de, it, es, fr, etc.)
        markets : str
            Tipo de mercado (h2h: 1X2, ou: over/under, etc.)
        
        Retorna:
        --------
        pd.DataFrame con odds actuales
        """
        self.api_key = api_key
        
        try:
            url = 'https://api.the-odds-api.com/v4/sports'
            params = {'apiKey': api_key}
            
            # Verificar disponibilidad de deporte
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            sports = response.json()
            sport_ids = [s['key'] for s in sports]
            
            if sport not in sport_ids:
                print(f'❌ Deporte no disponible: {sport}')
                print(f'   Disponibles: {sport_ids}')
                return None
            
            # Obtener odds actuales
            odds_url = f'{url}/{sport}/odds'
            params.update({
                'regions': region,
                'markets': markets
            })
            
            response = requests.get(odds_url, params=params, timeout=10)
            response.raise_for_status()
            odds_data = response.json()
            
            # Procesar datos
            records = []
            for game in odds_data.get('games', []):
                record = {
                    'match_id': game.get('id'),
                    'date': pd.to_datetime(game.get('commence_time')),
                    'home_team': game.get('home_team'),
                    'away_team': game.get('away_team'),
                    'sport': game.get('sport_title')
                }
                
                # Extraer odds de diferentes bookmakers
                for bookmaker in game.get('bookmakers', []):
                    bm_name = bookmaker.get('title', 'unknown')
                    for market in bookmaker.get('markets', []):
                        if market.get('key') == 'h2h':
                            outcomes = {o['name']: o['price'] for o in market.get('outcomes', [])}
                            record.update({
                                f'bookmaker_{bm_name}_home': outcomes.get('Home'),
                                f'bookmaker_{bm_name}_draw': outcomes.get('Draw'),
                                f'bookmaker_{bm_name}_away': outcomes.get('Away')
                            })
                
                records.append(record)
            
            df = pd.DataFrame(records)
            self.df_odds = df
            
            print(f'✅ Odds actuales obtenidas: {len(df)} partidos')
            return df
        
        except requests.exceptions.RequestException as e:
            print(f'❌ Error en API request: {e}')
            return None
        except Exception as e:
            print(f'❌ Error procesando odds: {e}')
            return None
    
    def calculate_implied_probability(self, odds: float) -> float:
        """
        Calcular probabilidad implícita desde cuota decimal
        
        Fórmula: P = 1 / odds
        
        Parámetros:
        -----------
        odds : float
            Cuota decimal (ej: 2.50)
        
        Retorna:
        --------
        float : Probabilidad (0-1)
        """
        if odds <= 0:
            return 0.0
        return min(1.0, 1.0 / odds)
    
    def calculate_fair_odds(self, probability: float) -> float:
        """
        Calcular cuota justa desde probabilidad
        
        Fórmula: odds = 1 / probability
        
        Parámetros:
        -----------
        probability : float
            Probabilidad (0-1)
        
        Retorna:
        --------
        float : Cuota justa decimal
        """
        if probability <= 0 or probability > 1:
            return float('inf')
        return 1.0 / probability
    
    def calculate_bookmaker_margin(self, home_odds: float, draw_odds: float, 
                                  away_odds: float) -> float:
        """
        Calcular margen de la casa (overround)
        
        El margen es la diferencia entre la suma de probabilidades y 1
        Fórmula: margin = (1/H + 1/D + 1/A) - 1
        
        Parámetros:
        -----------
        home_odds, draw_odds, away_odds : float
            Cuotas decimales
        
        Retorna:
        --------
        float : Margen (típicamente 3-5%)
        """
        overround = (1/home_odds) + (1/draw_odds) + (1/away_odds)
        margin = max(0, overround - 1)
        return margin
    
    def calculate_sharp_odds(self, home_odds: float, draw_odds: float, 
                            away_odds: float) -> Dict[str, float]:
        """
        Calcular cuotas 'sharp' (sin margen)
        
        Las casas de apuestas suben todas las cuotas proporcionalmente
        para garantizar su margen. Esto reversa ese proceso.
        
        Parámetros:
        -----------
        home_odds, draw_odds, away_odds : float
            Cuotas decimales del bookmaker
        
        Retorna:
        --------
        dict : Cuotas sin margen
        """
        overround = (1/home_odds) + (1/draw_odds) + (1/away_odds)
        
        if overround <= 1:
            return {
                'home': home_odds,
                'draw': draw_odds,
                'away': away_odds,
                'overround': 0
            }
        
        # Dividir cada probabilidad por el total para eliminar margen
        sharp_home = (1 / home_odds) / overround
        sharp_draw = (1 / draw_odds) / overround
        sharp_away = (1 / away_odds) / overround
        
        # Convertir de vuelta a cuotas
        return {
            'home': 1 / sharp_home,
            'draw': 1 / sharp_draw,
            'away': 1 / sharp_away,
            'overround': overround - 1
        }
    
    def get_best_odds(self, match_date: str, home_team: str, away_team: str) -> Dict:
        """
        Obtener las mejores cuotas disponibles para un partido
        (highest odds = menores probabilidades implícitas = mejor valor)
        
        Parámetros:
        -----------
        match_date : str
            Fecha en formato 'YYYY-MM-DD'
        home_team : str
            Equipo local
        away_team : str
            Equipo visitante
        
        Retorna:
        --------
        dict : Mejores cuotas y bookmaker
        """
        if self.df_odds is None or len(self.df_odds) == 0:
            return None
        
        match_date = pd.to_datetime(match_date).date()
        
        # Buscar el partido
        match_df = self.df_odds[
            (pd.to_datetime(self.df_odds['date']).dt.date == match_date) &
            (self.df_odds['home_team'].str.lower() == home_team.lower()) &
            (self.df_odds['away_team'].str.lower() == away_team.lower())
        ]
        
        if len(match_df) == 0:
            return None
        
        # Obtener mejores cuotas
        home_odds = match_df['home_win_odds'].max()
        draw_odds = match_df['draw_odds'].max()
        away_odds = match_df['away_win_odds'].max()
        
        return {
            'date': match_date,
            'home_team': home_team,
            'away_team': away_team,
            'home_win_odds': home_odds,
            'draw_odds': draw_odds,
            'away_win_odds': away_odds,
            'home_prob': self.calculate_implied_probability(home_odds),
            'draw_prob': self.calculate_implied_probability(draw_odds),
            'away_prob': self.calculate_implied_probability(away_odds),
            'margin': self.calculate_bookmaker_margin(home_odds, draw_odds, away_odds)
        }
    
    def get_consensus_odds(self, match_date: str, home_team: str, away_team: str) -> Dict:
        """
        Obtener cuotas de consenso (promedio ajustado ponderado)
        
        Usa las mejores cuotas pero pondera el promedio para mayor robustez
        
        Parámetros:
        -----------
        match_date : str
            Fecha en formato 'YYYY-MM-DD'
        home_team : str
            Equipo local
        away_team : str
            Equipo visitante
        
        Retorna:
        --------
        dict : Cuotas de consenso
        """
        best_odds = self.get_best_odds(match_date, home_team, away_team)
        
        if best_odds is None:
            return None
        
        # Calcular cuotas sin margen (sharp odds)
        sharp = self.calculate_sharp_odds(
            best_odds['home_win_odds'],
            best_odds['draw_odds'],
            best_odds['away_win_odds']
        )
        
        return {
            'best_odds': best_odds,
            'sharp_odds': sharp,
            'consensus_home_prob': 1 / sharp['home'],
            'consensus_draw_prob': 1 / sharp['draw'],
            'consensus_away_prob': 1 / sharp['away']
        }
    
    def get_latest_odds(self, home_team: str, away_team: str, 
                        lookback_days: int = 7) -> Optional[Dict]:
        """
        Obtener odds más recientes para un partido próximo
        
        Parámetros:
        -----------
        home_team : str
            Equipo local
        away_team : str
            Equipo visitante
        lookback_days : int
            Buscar dentro de N días a partir de hoy
        
        Retorna:
        --------
        dict : Odds más recientes o None
        """
        if self.df_odds is None or len(self.df_odds) == 0:
            return None
        
        today = pd.to_datetime('today').date()
        future = pd.to_datetime('today') + pd.Timedelta(days=lookback_days)
        
        match_df = self.df_odds[
            (pd.to_datetime(self.df_odds['date']).dt.date >= today) &
            (pd.to_datetime(self.df_odds['date']).dt.date <= future.date()) &
            (self.df_odds['home_team'].str.lower() == home_team.lower()) &
            (self.df_odds['away_team'].str.lower() == away_team.lower())
        ]
        
        if len(match_df) == 0:
            return None
        
        # Obtener el más reciente
        match = match_df.iloc[-1]
        
        return {
            'date': match['date'],
            'home_team': match['home_team'],
            'away_team': match['away_team'],
            'home_win_odds': match['home_win_odds'],
            'draw_odds': match['draw_odds'],
            'away_win_odds': match['away_win_odds']
        }
    
    def save_odds_snapshot(self, filepath: str):
        """
        Guardar snapshot actual de odds (para análisis histórico)
        
        Parámetros:
        -----------
        filepath : str
            Ruta de salida
        """
        if self.df_odds is None:
            print('❌ No hay odds cargados')
            return False
        
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            if '.' in filepath:
                name, ext = filepath.rsplit('.', 1)
                filepath = f"{name}_{timestamp}.{ext}"
            else:
                filepath = f"{filepath}_{timestamp}.csv"
            
            self.df_odds.to_csv(filepath, index=False)
            print(f'✅ Odds guardados: {filepath}')
            return True
        
        except Exception as e:
            print(f'❌ Error guardando odds: {e}')
            return False


# Utilidades de conversión de cuotas
def convert_odds_decimal_to_fractional(decimal_odds: float) -> str:
    """
    Convertir cuota decimal a fraccionaria (formato: "1/2", "3/1", etc.)
    
    Parámetro:
    ----------
    decimal_odds : float
        Cuota decimal (ej: 2.50)
    
    Retorna:
    --------
    str : Cuota fraccionaria
    """
    from fractions import Fraction
    
    if decimal_odds <= 1:
        return "1/0"  # Inválido
    
    # La cuota fraccionaria = (decimal - 1) / 1
    numerator = decimal_odds - 1
    frac = Fraction(numerator).limit_denominator(1000)
    
    return f"{frac.numerator}/{frac.denominator}"


def convert_odds_decimal_to_american(decimal_odds: float) -> int:
    """
    Convertir cuota decimal a americana (moneyline)
    
    Parámetro:
    ----------
    decimal_odds : float
        Cuota decimal (ej: 2.50)
    
    Retorna:
    --------
    int : Cuota americana
    """
    if decimal_odds >= 2:
        # Favorito negativo: (-100) / (decimal - 1)
        return int((-100) / (decimal_odds - 1))
    else:
        # Underdog positivo: (decimal - 1) * 100
        return int((decimal_odds - 1) * 100)


def convert_odds_american_to_decimal(american_odds: int) -> float:
    """
    Convertir cuota americana a decimal
    
    Parámetro:
    ----------
    american_odds : int
        Cuota americana (ej: -110, +250)
    
    Retorna:
    --------
    float : Cuota decimal
    """
    if american_odds > 0:
        return (american_odds / 100) + 1
    else:
        return (100 / abs(american_odds)) + 1


if __name__ == '__main__':
    print("🔍 MÓDULO DE GESTIÓN DE ODDS\n")
    
    # Ejemplos de uso
    manager = OddsManager()
    
    # Ejemplo 1: Conversión de cuotas
    print("Conversión de cuotas:")
    decimal = 2.50
    print(f"  Decimal: {decimal}")
    print(f"  Fraccionaria: {convert_odds_decimal_to_fractional(decimal)}")
    print(f"  Americana: {convert_odds_decimal_to_american(decimal)}")
    
    # Ejemplo 2: Probabilidad implícita
    print(f"\nProbabilidad implícita de 2.50: {manager.calculate_implied_probability(2.50):.2%}")
    print(f"Cuota justa de 40% probabilidad: {manager.calculate_fair_odds(0.40):.2f}")
    
    # Ejemplo 3: Margen de casa
    margin = manager.calculate_bookmaker_margin(1.80, 3.50, 4.20)
    print(f"\nMargen de casa (1.80, 3.50, 4.20): {margin:.2%}")
    
    # Ejemplo 4: Sharp odds (sin margen)
    sharp = manager.calculate_sharp_odds(1.80, 3.50, 4.20)
    print(f"\nSharp odds (sin margen):")
    print(f"  Home: {sharp['home']:.2f}")
    print(f"  Draw: {sharp['draw']:.2f}")
    print(f"  Away: {sharp['away']:.2f}")
