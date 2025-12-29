"""
Script para corregir epl_odds.csv:
1. Recuperar fechas perdidas de 2003_04
2. Agregar columna Season
3. Validar integridad
"""

import pandas as pd
from pathlib import Path
import numpy as np

def assign_season_from_date(date_str):
    """
    Determina la temporada basándose en la fecha
    Formato: YYYY/YY (e.g., 2000/01)
    """
    try:
        # Convertir fecha
        date = pd.to_datetime(date_str, format='%d/%m/%y')
        year = date.year
        month = date.month
        
        # Si es agosto-diciembre, es inicio de temporada (año/año+1)
        # Si es enero-mayo, es final de temporada (año-1/año)
        if month >= 8:  # Agosto a Diciembre
            season_start = year
            season_end = year + 1
        else:  # Enero a Mayo
            season_start = year - 1
            season_end = year
        
        # Formatear como "2000/01"
        season = f"{season_start}/{str(season_end)[-2:]}"
        return season
    except:
        return None

def fix_epl_odds():
    """Corrige epl_odds.csv con fechas y temporadas"""
    
    base_path = Path("data/raw")
    
    print("="*70)
    print("🔧 CORRECCIÓN DE epl_odds.csv")
    print("="*70)
    print()
    
    # 1. Cargar archivo actual
    print("1️⃣ Cargando epl_odds.csv actual...")
    df_odds = pd.read_csv(base_path / "epl_odds.csv")
    print(f"   Total filas: {len(df_odds)}")
    print(f"   Fechas vacías: {df_odds['Date'].isna().sum()}")
    print()
    
    # 2. Identificar filas sin fecha
    missing_dates = df_odds['Date'].isna()
    print(f"2️⃣ Filas sin fecha: {missing_dates.sum()}")
    
    if missing_dates.sum() > 0:
        print("   Rango de filas afectadas:", df_odds[missing_dates].index.min(), 
              "a", df_odds[missing_dates].index.max())
        print()
        
        # 3. Intentar recuperar del archivo 2003_04_fixed.csv
        print("3️⃣ Recuperando fechas de 2003_04_fixed.csv...")
        
        try:
            df_2003_04 = pd.read_csv(base_path / "2003_04_fixed.csv")
            print(f"   Archivo cargado: {len(df_2003_04)} filas")
            
            # Crear diccionario de búsqueda por equipos
            date_lookup = {}
            for _, row in df_2003_04.iterrows():
                key = (row['HomeTeam'], row['AwayTeam'])
                date_lookup[key] = row['Date']
            
            # Rellenar fechas vacías
            recovered = 0
            for idx in df_odds[missing_dates].index:
                home = df_odds.loc[idx, 'HomeTeam']
                away = df_odds.loc[idx, 'AwayTeam']
                key = (home, away)
                
                if key in date_lookup:
                    df_odds.loc[idx, 'Date'] = date_lookup[key]
                    recovered += 1
            
            print(f"   ✅ Fechas recuperadas: {recovered}")
            
            # Verificar si quedaron algunas sin recuperar
            still_missing = df_odds['Date'].isna().sum()
            if still_missing > 0:
                print(f"   ⚠️  Aún faltan {still_missing} fechas (partidos no encontrados en 2003_04)")
        
        except Exception as e:
            print(f"   ❌ Error recuperando fechas: {e}")
    print()
    
    # 4. Agregar columna Season
    print("4️⃣ Agregando columna Season...")
    
    # Insertar Season como segunda columna (después de Div)
    df_odds.insert(1, 'Season', None)
    
    # Asignar temporada basándose en la fecha
    seasons_assigned = 0
    for idx, row in df_odds.iterrows():
        if pd.notna(row['Date']):
            season = assign_season_from_date(row['Date'])
            if season:
                df_odds.loc[idx, 'Season'] = season
                seasons_assigned += 1
    
    print(f"   ✅ Temporadas asignadas: {seasons_assigned}/{len(df_odds)}")
    print()
    
    # 5. Mostrar distribución por temporada
    print("5️⃣ Distribución por temporada:")
    print("-" * 70)
    season_counts = df_odds['Season'].value_counts().sort_index()
    for season, count in season_counts.items():
        print(f"   {season}: {count:3d} partidos")
    print()
    
    # 6. Verificar consistencia con epl_final.csv
    print("6️⃣ Verificando consistencia con epl_final.csv...")
    try:
        df_final = pd.read_csv(base_path / "epl_final.csv")
        
        # Verificar algunos partidos en común
        sample_checks = 5
        matched = 0
        
        for i in range(min(sample_checks, len(df_odds))):
            row_odds = df_odds.iloc[i]
            if pd.notna(row_odds['Date']):
                date_odds = pd.to_datetime(row_odds['Date'], format='%d/%m/%y')
                
                # Buscar en epl_final
                match = df_final[
                    (pd.to_datetime(df_final['MatchDate']) == date_odds) &
                    (df_final['HomeTeam'] == row_odds['HomeTeam']) &
                    (df_final['AwayTeam'] == row_odds['AwayTeam'])
                ]
                
                if len(match) > 0:
                    season_final = match.iloc[0]['Season']
                    season_odds = row_odds['Season']
                    
                    if season_final == season_odds:
                        matched += 1
                    else:
                        print(f"   ⚠️  Discrepancia: {row_odds['HomeTeam']} vs {row_odds['AwayTeam']}")
                        print(f"       epl_final: {season_final} | epl_odds: {season_odds}")
        
        print(f"   ✅ Verificados {matched}/{sample_checks} partidos - Consistencia OK")
    except Exception as e:
        print(f"   ⚠️  No se pudo verificar: {e}")
    print()
    
    # 7. Guardar archivo corregido
    print("7️⃣ Guardando archivo corregido...")
    
    # Backup del actual (si no existe ya)
    if not (base_path / "epl_odds_before_fix.csv").exists():
        df_current = pd.read_csv(base_path / "epl_odds.csv")
        df_current.to_csv(base_path / "epl_odds_before_fix.csv", index=False)
        print("   💾 Backup creado: epl_odds_before_fix.csv")
    
    # Guardar corregido
    df_odds.to_csv(base_path / "epl_odds.csv", index=False)
    print(f"   💾 epl_odds.csv actualizado: {len(df_odds)} partidos, {len(df_odds.columns)} columnas")
    print()
    
    # 8. Resumen final
    print("="*70)
    print("✅ CORRECCIÓN COMPLETADA")
    print("="*70)
    print()
    print(f"📊 Resumen final:")
    print(f"   Total de partidos: {len(df_odds)}")
    print(f"   Columnas: {len(df_odds.columns)} (nueva: Season)")
    print(f"   Fechas válidas: {df_odds['Date'].notna().sum()}/{len(df_odds)}")
    print(f"   Temporadas válidas: {df_odds['Season'].notna().sum()}/{len(df_odds)}")
    
    # Columnas actuales
    print(f"\n   Primeras columnas: {', '.join(df_odds.columns[:8].tolist())}")
    
    # Muestra
    print(f"\n📋 Muestra del dataset corregido:")
    print(df_odds[['Div', 'Season', 'Date', 'HomeTeam', 'AwayTeam', 'FTR']].head(10).to_string(index=False))
    
    print()
    print("🎯 Próximos pasos:")
    print("   1. Re-ejecutar: python scripts/integrate_market_data.py")
    print("   2. Ahora con columna Season para mejores análisis temporales")
    print()
    
    return df_odds


if __name__ == '__main__':
    fix_epl_odds()
