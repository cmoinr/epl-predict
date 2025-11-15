"""
Script para investigar y acceder a APIs de odds para casas de apuestas
"""

import requests
import json
from typing import Optional, Dict, List
import pandas as pd


def check_odds_api_free() -> Dict:
    """
    Verifica disponibilidad de odds-api.com (API gratuita)
    Sitio: https://www.odds-api.com/
    
    Returns:
        Dict con información de la API
    """
    info = {
        'name': 'odds-api.com',
        'type': 'Free Tier',
        'requests_per_day': 500,
        'sports_available': ['soccer', 'american_football', 'basketball', etc.],
        'endpoints': {
            'sports': 'GET /v4/sports',
            'odds': 'GET /v4/sports/{sport}/odds',
            'historical': 'Premium only'
        },
        'setup': [
            '1. Ir a https://odds-api.com/register',
            '2. Crear cuenta gratuita',
            '3. Obtener API key',
            '4. Usar en requests: ?apiKey=YOUR_KEY'
        ],
        'example': '''
import requests

api_key = "YOUR_API_KEY"
sport = "soccer_epl"  # Premier League
region = "uk"  # Casas de apuestas del UK
markets = "h2h"  # Head-to-head (1X2)

url = f"https://api.odds.api.io/v4/sports/{sport}/odds"
params = {
    'apiKey': api_key,
    'region': region,
    'markets': markets
}

response = requests.get(url, params=params)
odds_data = response.json()
        '''
    }
    return info


def check_football_data_org() -> Dict:
    """
    Verifica football-data.org (datos históricos)
    Sitio: https://www.football-data.org/
    
    Returns:
        Dict con información de la API
    """
    info = {
        'name': 'football-data.org',
        'type': 'Free Tier Available',
        'requests_per_minute': 10,
        'features': [
            'Datos históricos de ligas',
            'Odds de múltiples casas',
            'Standings, players, teams'
        ],
        'setup': [
            '1. Registrarse en https://www.football-data.org/client/register',
            '2. Obtener API token gratuito',
            '3. Usar en headers: X-Auth-Token: YOUR_TOKEN'
        ],
        'endpoints': {
            'teams': '/teams',
            'matches': '/matches',
            'standings': '/standings',
            'odds': '/matches/{id}/odds'
        },
        'example': '''
import requests

api_key = "YOUR_API_TOKEN"
headers = {"X-Auth-Token": api_key}

# Obtener matches de la Premier League
url = "https://api.football-data.org/v4/competitions/PL/matches"
response = requests.get(url, headers=headers)
matches = response.json()

# Procesar odds si están disponibles
for match in matches['matches']:
    print(f"{match['homeTeam']['name']} vs {match['awayTeam']['name']}")
    if 'odds' in match:
        print(f"Home: {match['odds']['homeWin']}")
        '''
    }
    return info


def check_rapidapi_options() -> Dict:
    """
    Opciones de RapidAPI (múltiples endpoints)
    Sitio: https://rapidapi.com/
    
    Returns:
        Dict con opciones disponibles
    """
    info = {
        'name': 'RapidAPI',
        'type': 'Freemium',
        'popular_apis': [
            'API-Football (thousands of requests/month)',
            'Odds API',
            'Football Data'
        ],
        'advantages': [
            'Muchas APIs en una plataforma',
            'Documentación integrada',
            'Prueba directamente en el sitio',
            'Planes gratuitos generosos'
        ],
        'setup': [
            '1. Crear cuenta en https://rapidapi.com',
            '2. Buscar "football" o "odds"',
            '3. Suscribirse a API (Free tier)',
            '4. Copiar ejemplos de código'
        ]
    }
    return info


def check_understat() -> Dict:
    """
    Understat.com - Datos avanzados de fútbol
    Sitio: https://understat.com/
    
    Nota: No tiene API pública, pero puedes scrapear
    
    Returns:
        Dict con información
    """
    info = {
        'name': 'Understat.com',
        'type': 'Web Scraping',
        'data_available': [
            'Expected Goals (xG)',
            'Shot maps',
            'Team stats',
            'Player stats'
        ],
        'note': 'No tiene API oficial, requiere web scraping',
        'tools': ['BeautifulSoup', 'Selenium', 'Requests'],
        'ethical': 'Verificar términos de servicio antes de scrapear'
    }
    return info


def get_eplushet_odds_sample() -> pd.DataFrame:
    """
    Crea un dataset de ejemplo con estructura de odds
    
    Returns:
        DataFrame con estructura de odds típica
    """
    sample_data = {
        'match_id': [1, 2, 3],
        'date': ['2024-11-16', '2024-11-17', '2024-11-18'],
        'home_team': ['Manchester City', 'Liverpool', 'Arsenal'],
        'away_team': ['Liverpool', 'Arsenal', 'Chelsea'],
        'home_win_odds': [1.50, 2.10, 1.80],
        'draw_odds': [3.75, 3.50, 3.60],
        'away_win_odds': [5.50, 3.20, 4.00],
        'over_2_5_goals': [1.95, 1.88, 2.05],
        'under_2_5_goals': [1.80, 1.95, 1.70],
        'result': ['1', 'X', '1'],  # 1=Home, X=Draw, 2=Away
        'total_goals': [2, 2, 3]
    }
    return pd.DataFrame(sample_data)


def calculate_implied_probability(odds: float) -> float:
    """
    Calcula probabilidad implícita de odds
    
    Args:
        odds: Cuota decimal (ej: 2.50)
    
    Returns:
        Probabilidad implícita (0-1)
    """
    return 1.0 / odds


def check_value_betting_strategy() -> Dict:
    """
    Estrategia de value betting
    
    Returns:
        Dict explicando la estrategia
    """
    return {
        'concept': 'Apostar cuando la probabilidad predicha > probabilidad implícita',
        'formula': {
            'implied_prob': 'P_implied = 1 / odd',
            'model_prob': 'P_model = probabilidad del modelo ML',
            'value': 'V = P_model - P_implied',
            'ev': 'EV = (P_model * odd) - 1'
        },
        'example': '''
Odd de mercado: 3.00
Probabilidad implícita: 1/3 = 0.333 (33.3%)

Nuestro modelo predice: 40%
Value: 40% - 33% = 7% (positivo)

Si apostamos 100, EV = (0.40 * 3.00) - 1 = 0.20 (20% retorno esperado)
        ''',
        'selection_criteria': [
            'Edge mínimo > 3-5%',
            'Cuotas con suficiente liquidez',
            'Muestras representativas',
            'Manejar varianza (trackers de ROI)'
        ]
    }


if __name__ == '__main__':
    print("🔍 INVESTIGACIÓN DE APIs DE ODDS PARA PREMIER LEAGUE\n")
    
    print("="*70)
    print("OPCIÓN 1: odds-api.com (RECOMENDADO)")
    print("="*70)
    odds_api = check_odds_api_free()
    print(f"  Sitio: {odds_api['name']}")
    print(f"  Tier: {odds_api['type']}")
    print(f"  Límite: {odds_api['requests_per_day']} requests/día")
    print(f"  Setup: Ver ejemplo en src/odds_api.py")
    
    print("\n" + "="*70)
    print("OPCIÓN 2: football-data.org")
    print("="*70)
    fd = check_football_data_org()
    print(f"  Sitio: {fd['name']}")
    print(f"  Límite: {fd['requests_per_minute']} requests/minuto")
    
    print("\n" + "="*70)
    print("OPCIÓN 3: RapidAPI")
    print("="*70)
    rapid = check_rapidapi_options()
    print(f"  Plataforma: {rapid['name']}")
    print(f"  Tipo: {rapid['type']}")
    print(f"  APIs populares: {rapid['popular_apis']}")
    
    print("\n" + "="*70)
    print("VALUE BETTING STRATEGY")
    print("="*70)
    strategy = check_value_betting_strategy()
    print(f"  Concepto: {strategy['concept']}")
    print(f"  Formula EV: {strategy['formula']['ev']}")
    
    print("\n✅ Para datos históricos con odds: Usar football-data.org o Kaggle")
