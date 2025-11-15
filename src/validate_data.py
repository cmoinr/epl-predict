"""
Script para validar integridad del dataset después de agregar nuevos datos.
"""

import pandas as pd
from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).parent.parent
DATA_PATH = SCRIPT_DIR / "data" / "raw" / "epl_final.csv"

def validate_dataset():
    """Realiza validaciones exhaustivas del dataset."""
    print("🔍 Validando dataset EPL...\n")
    
    # Cargar datos
    df = pd.read_csv(str(DATA_PATH))
    print(f"📊 Dataset: {len(df)} partidos")
    print(f"📅 Temporadas: {df['Season'].unique()}")
    print()
    
    # 1. Valores faltantes
    print("1️⃣ VALORES FALTANTES:")
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print("   ⚠️  Columnas con valores faltantes:")
        print(missing[missing > 0])
    else:
        print("   ✅ Sin valores faltantes")
    print()
    
    # 2. Consistencia de datos
    print("2️⃣ CONSISTENCIA DE DATOS:")
    
    # Goles: FullTime debe coincidir con suma de HalfTime (aproximadamente)
    goles_home_full = df['FullTimeHomeGoals'].astype(int)
    goles_away_full = df['FullTimeAwayGoals'].astype(int)
    goles_home_half = df['HalfTimeHomeGoals'].astype(int)
    goles_away_half = df['HalfTimeAwayGoals'].astype(int)
    
    # Los goles del 2do tiempo no deben ser negativos
    goles_2do_home = goles_home_full - goles_home_half
    goles_2do_away = goles_away_full - goles_away_half
    
    if (goles_2do_home < 0).any() or (goles_2do_away < 0).any():
        print("   ⚠️  Advertencia: Goles 2do tiempo negativos detectados")
        problematic = df[(goles_2do_home < 0) | (goles_2do_away < 0)]
        print(f"   {len(problematic)} partidos problemáticos")
    else:
        print("   ✅ Goles coherentes (HalfTime ≤ FullTime)")
    print()
    
    # 3. Estadísticas lógicas
    print("3️⃣ VALIDACIÓN DE ESTADÍSTICAS:")
    
    # Shots on target ≤ Shots
    home_invalid = df['HomeShotsOnTarget'].astype(int) > df['HomeShots'].astype(int)
    away_invalid = df['AwayShotsOnTarget'].astype(int) > df['AwayShots'].astype(int)
    invalid_shots = df[home_invalid | away_invalid]
    
    if len(invalid_shots) > 0:
        print(f"   ⚠️  {len(invalid_shots)} partidos con ShotsOnTarget > Shots")
    else:
        print("   ✅ Shots On Target ≤ Shots")
    
    # Tarjetas: máximo 5 por equipo (rara vez)
    max_yellows = max(df['HomeYellowCards'].astype(int).max(), df['AwayYellowCards'].astype(int).max())
    max_reds = max(df['HomeRedCards'].astype(int).max(), df['AwayRedCards'].astype(int).max())
    
    if max_yellows > 10 or max_reds > 2:
        print(f"   ⚠️  Tarjetas inusuales detectadas (Max Amarillas: {max_yellows}, Max Rojas: {max_reds})")
    else:
        print(f"   ✅ Tarjetas razonables (Max Amarillas: {max_yellows}, Max Rojas: {max_reds})")
    print()
    
    # 4. Resultados válidos
    print("4️⃣ VALIDACIÓN DE RESULTADOS:")
    valid_results = df['FullTimeResult'].isin(['H', 'D', 'A']).all()
    if valid_results:
        print("   ✅ Todos los resultados son H/D/A")
    else:
        print("   ⚠️  Resultados inválidos detectados")
    print()
    
    # 5. Duplicados
    print("5️⃣ DETECCIÓN DE DUPLICADOS:")
    duplicates = df.duplicated(subset=['MatchDate', 'HomeTeam', 'AwayTeam'])
    if duplicates.sum() > 0:
        print(f"   ⚠️  {duplicates.sum()} partidos duplicados")
        print(df[duplicates][['MatchDate', 'HomeTeam', 'AwayTeam']])
    else:
        print("   ✅ Sin duplicados")
    print()
    
    # 6. Resumen estadístico
    print("6️⃣ ESTADÍSTICAS GENERALES:")
    print(f"   Goles promedio por partido: {(goles_home_full.mean() + goles_away_full.mean()):.2f}")
    print(f"   Shots promedio: {(df['HomeShots'].astype(int).mean() + df['AwayShots'].astype(int).mean()):.2f}")
    print(f"   Corners promedio: {(df['HomeCorners'].astype(int).mean() + df['AwayCorners'].astype(int).mean()):.2f}")
    print()
    
    # 7. Datos recientes
    print("7️⃣ ÚLTIMOS 5 PARTIDOS AGREGADOS:")
    latest = df.tail(5)[['MatchDate', 'HomeTeam', 'AwayTeam', 'FullTimeHomeGoals', 'FullTimeAwayGoals']]
    print(latest.to_string(index=False))
    print()
    
    print("✅ Validación completada!\n")
    return True

if __name__ == "__main__":
    validate_dataset()
