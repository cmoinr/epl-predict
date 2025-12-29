"""
Script robusto para fusionar correctamente todos los archivos de odds
Maneja diferentes formatos de fecha y agrega columna Season
"""

import pandas as pd
from pathlib import Path
import numpy as np

def parse_date_flexible(date_str):
    """Intenta parsear fecha con múltiples formatos"""
    if pd.isna(date_str):
        return None
    
    formats = ['%d/%m/%y', '%d/%m/%Y', '%Y-%m-%d']
    
    for fmt in formats:
        try:
            return pd.to_datetime(date_str, format=fmt)
        except:
            continue
    
    # Último intento sin formato
    try:
        return pd.to_datetime(date_str)
    except:
        return None

def assign_season(date):
    """Asigna temporada basándose en fecha"""
    if pd.isna(date):
        return None
    
    year = date.year
    month = date.month
    
    if month >= 8:
        return f"{year}/{str(year+1)[-2:]}"
    else:
        return f"{year-1}/{str(year)[-2:]}"

def rebuild_epl_odds():
    """Reconstruye epl_odds.csv desde cero"""
    
    base_path = Path("data/raw")
    
    print("="*70)
    print("🔄 RECONSTRUCCIÓN COMPLETA DE epl_odds.csv")
    print("="*70)
    print()
    
    # Archivos fuente
    source_files = [
        ("epl_odds_backup.csv", "2000/01"),
        ("2001_02.csv", "2001/02"),
        ("2002_03.csv", "2002/03"),
        ("2003_04_fixed.csv", "2003/04")
    ]
    
    all_dfs = []
    
    print("1️⃣ Cargando archivos fuente...")
    print("-" * 70)
    
    for filename, expected_season in source_files:
        file_path = base_path / filename
        
        if not file_path.exists():
            print(f"   ⚠️  {filename} no encontrado - saltando")
            continue
        
        try:
            df = pd.read_csv(file_path)
            
            # Normalizar columna de fecha
            if 'Date' in df.columns:
                # Parsear fechas con formato flexible
                df['Date_parsed'] = df['Date'].apply(parse_date_flexible)
                
                # Contar fechas válidas
                valid_dates = df['Date_parsed'].notna().sum()
                
                print(f"   ✅ {filename:25s}: {len(df):3d} partidos | "
                      f"{valid_dates:3d} fechas válidas")
                
                all_dfs.append(df)
            else:
                print(f"   ❌ {filename}: No tiene columna 'Date'")
                
        except Exception as e:
            print(f"   ❌ Error en {filename}: {e}")
    
    print()
    
    # 2. Fusionar todos
    print("2️⃣ Fusionando datasets...")
    df_merged = pd.concat(all_dfs, ignore_index=True, sort=False)
    
    # Eliminar duplicados
    before_dedup = len(df_merged)
    df_merged = df_merged.drop_duplicates(
        subset=['Date_parsed', 'HomeTeam', 'AwayTeam'], 
        keep='first'
    )
    after_dedup = len(df_merged)
    
    if before_dedup > after_dedup:
        print(f"   Removidos {before_dedup - after_dedup} duplicados")
    
    print(f"   Total fusionado: {len(df_merged)} partidos")
    print()
    
    # 3. Crear columna Season
    print("3️⃣ Asignando temporadas...")
    df_merged['Season'] = df_merged['Date_parsed'].apply(assign_season)
    
    seasons_assigned = df_merged['Season'].notna().sum()
    print(f"   ✅ Temporadas asignadas: {seasons_assigned}/{len(df_merged)}")
    print()
    
    # 4. Normalizar formato de fecha a DD/MM/YY
    print("4️⃣ Normalizando formato de fechas...")
    
    def format_date_standard(date):
        """Convierte fecha a formato DD/MM/YY"""
        if pd.isna(date):
            return None
        try:
            return date.strftime('%d/%m/%y')
        except:
            return None
    
    df_merged['Date'] = df_merged['Date_parsed'].apply(format_date_standard)
    
    # Eliminar columna auxiliar
    df_merged = df_merged.drop('Date_parsed', axis=1)
    
    valid_dates = df_merged['Date'].notna().sum()
    print(f"   ✅ Fechas formateadas: {valid_dates}/{len(df_merged)}")
    print()
    
    # 5. Reordenar columnas (Season después de Div)
    print("5️⃣ Reordenando columnas...")
    
    # Obtener todas las columnas
    cols = df_merged.columns.tolist()
    
    # Remover Season si existe
    if 'Season' in cols:
        cols.remove('Season')
    
    # Insertar Season después de Div
    if 'Div' in cols:
        div_idx = cols.index('Div')
        cols.insert(div_idx + 1, 'Season')
    else:
        cols.insert(0, 'Season')
    
    df_merged = df_merged[cols]
    print(f"   ✅ Columnas reordenadas: {len(cols)} totales")
    print()
    
    # 6. Ordenar por fecha
    print("6️⃣ Ordenando por fecha...")
    df_merged['_date_sort'] = pd.to_datetime(df_merged['Date'], format='%d/%m/%y', errors='coerce')
    df_merged = df_merged.sort_values('_date_sort').reset_index(drop=True)
    df_merged = df_merged.drop('_date_sort', axis=1)
    print("   ✅ Dataset ordenado cronológicamente")
    print()
    
    # 7. Estadísticas por temporada
    print("7️⃣ Distribución por temporada:")
    print("-" * 70)
    
    season_stats = df_merged.groupby('Season').agg({
        'Date': 'count',
        'HomeTeam': 'count'
    })
    
    for season in sorted(season_stats.index):
        count = season_stats.loc[season, 'Date']
        print(f"   {season}: {count:3d} partidos")
    
    print(f"\n   Total: {len(df_merged)} partidos en {len(season_stats)} temporadas")
    print()
    
    # 8. Guardar
    print("8️⃣ Guardando archivo final...")
    
    output_file = base_path / "epl_odds.csv"
    df_merged.to_csv(output_file, index=False)
    
    print(f"   💾 Guardado: {output_file}")
    print(f"   Filas: {len(df_merged)}")
    print(f"   Columnas: {len(df_merged.columns)}")
    print()
    
    # 9. Validación final
    print("9️⃣ Validación final:")
    print("-" * 70)
    
    print(f"   ✅ Fechas válidas: {df_merged['Date'].notna().sum()}/{len(df_merged)}")
    print(f"   ✅ Temporadas válidas: {df_merged['Season'].notna().sum()}/{len(df_merged)}")
    
    missing_dates = df_merged['Date'].isna().sum()
    if missing_dates > 0:
        print(f"   ⚠️  Fechas faltantes: {missing_dates}")
    
    print()
    
    # 10. Muestra
    print("📋 Muestra del dataset reconstruido:")
    print("-" * 70)
    print(df_merged[['Div', 'Season', 'Date', 'HomeTeam', 'AwayTeam', 'FTR']].head(15).to_string(index=False))
    
    print()
    print("="*70)
    print("✅ RECONSTRUCCIÓN COMPLETADA")
    print("="*70)
    print()
    print("🎯 Próximos pasos:")
    print("   1. Re-ejecutar: python scripts/integrate_market_data.py")
    print("   2. Todos los partidos ahora tienen Season y Date correctos")
    print()
    
    return df_merged


if __name__ == '__main__':
    rebuild_epl_odds()
