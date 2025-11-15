"""
Script para agregar manualmente nuevos partidos al dataset de EPL.
Valida estructura y evita duplicados.
"""

import pandas as pd
from datetime import datetime
import os
from pathlib import Path

# Obtener ruta absoluta basada en la ubicación del script
SCRIPT_DIR = Path(__file__).parent.parent
DATA_PATH = SCRIPT_DIR / "data" / "raw" / "epl_final.csv"
BACKUP_PATH = SCRIPT_DIR / "data" / "raw" / "epl_final_backup.csv"

def validate_row(row_dict):
    """Valida que los datos del partido sean correctos."""
    required_fields = [
        'Season', 'MatchDate', 'HomeTeam', 'AwayTeam',
        'FullTimeHomeGoals', 'FullTimeAwayGoals', 'FullTimeResult',
        'HalfTimeHomeGoals', 'HalfTimeAwayGoals', 'HalfTimeResult',
        'HomeShots', 'AwayShots', 'HomeShotsOnTarget', 'AwayShotsOnTarget',
        'HomeCorners', 'AwayCorners', 'HomeFouls', 'AwayFouls',
        'HomeYellowCards', 'AwayYellowCards', 'HomeRedCards', 'AwayRedCards'
    ]
    
    for field in required_fields:
        if field not in row_dict or row_dict[field] == '':
            return False, f"Campo faltante: {field}"
    
    # Validar formato de fecha
    try:
        datetime.strptime(row_dict['MatchDate'], '%Y-%m-%d')
    except ValueError:
        return False, "Formato de fecha incorrecto (use YYYY-MM-DD)"
    
    # Validar que los goles sean números
    try:
        int(row_dict['FullTimeHomeGoals'])
        int(row_dict['FullTimeAwayGoals'])
    except ValueError:
        return False, "Los goles deben ser números enteros"
    
    # Validar resultado
    result = row_dict['FullTimeResult']
    if result not in ['H', 'D', 'A']:
        return False, "FullTimeResult debe ser H, D o A"
    
    return True, "OK"


def add_match(match_data):
    """
    Agrega un nuevo partido al dataset.
    
    match_data debe ser dict con todas las columnas.
    """
    # Crear backup
    if DATA_PATH.exists():
        os.system(f"cp {DATA_PATH} {BACKUP_PATH}")
    
    # Validar datos
    is_valid, msg = validate_row(match_data)
    if not is_valid:
        print(f"❌ Error: {msg}")
        return False
    
    # Cargar dataset actual
    df = pd.read_csv(str(DATA_PATH))
    
    # Verificar duplicado (mismo partido, misma fecha)
    duplicate = df[
        (df['MatchDate'] == match_data['MatchDate']) &
        (df['HomeTeam'] == match_data['HomeTeam']) &
        (df['AwayTeam'] == match_data['AwayTeam'])
    ]
    
    if not duplicate.empty:
        print("⚠️  Advertencia: Este partido ya existe en el dataset")
        return False
    
    # Agregar nueva fila
    new_row = pd.DataFrame([match_data])
    df = pd.concat([df, new_row], ignore_index=True)
    
    # Guardar
    df.to_csv(str(DATA_PATH), index=False)
    print(f"✅ Partido agregado: {match_data['HomeTeam']} vs {match_data['AwayTeam']} ({match_data['MatchDate']})")
    print(f"   Resultado: {match_data['FullTimeHomeGoals']}-{match_data['FullTimeAwayGoals']}")
    return True


def add_multiple_matches(matches_list):
    """Agrega múltiples partidos a la vez."""
    added = 0
    for match in matches_list:
        if add_match(match):
            added += 1
    print(f"\n📊 Total agregados: {added}/{len(matches_list)}")


if __name__ == "__main__":
    # Ejemplo de uso:
    new_match = {
        'Season': '2024/25',
        'MatchDate': '2024-11-15',
        'HomeTeam': 'Arsenal',
        'AwayTeam': 'Manchester City',
        'FullTimeHomeGoals': 2,
        'FullTimeAwayGoals': 1,
        'FullTimeResult': 'H',
        'HalfTimeHomeGoals': 1,
        'HalfTimeAwayGoals': 0,
        'HalfTimeResult': 'H',
        'HomeShots': 15,
        'AwayShots': 12,
        'HomeShotsOnTarget': 8,
        'AwayShotsOnTarget': 5,
        'HomeCorners': 6,
        'AwayCorners': 4,
        'HomeFouls': 10,
        'AwayFouls': 12,
        'HomeYellowCards': 2,
        'AwayYellowCards': 1,
        'HomeRedCards': 0,
        'AwayRedCards': 0
    }
    
    add_match(new_match)
