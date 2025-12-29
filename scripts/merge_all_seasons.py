"""
Script para fusionar TODAS las temporadas (2000/01 a 2024/25) en epl_odds.csv
Maneja diferentes estructuras de columnas entre temporadas
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def parse_date_flexible(date_str):
    """Parsea fechas en múltiples formatos"""
    if pd.isna(date_str):
        return None
    
    date_str = str(date_str).strip()
    
    formats = [
        '%d/%m/%y',      # 19/08/00
        '%d/%m/%Y',      # 19/08/2000
        '%Y-%m-%d',      # 2000-08-19
        '%d-%m-%Y',      # 19-08-2000
        '%d-%m-%y',      # 19-08-00
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except:
            continue
    
    return None

def assign_season(date_obj):
    """Asigna temporada basándose en la fecha"""
    if date_obj is None:
        return None
    
    year = date_obj.year
    month = date_obj.month
    
    # Temporadas van de agosto a mayo
    if month >= 8:  # Agosto-Diciembre
        return f"{year}/{str(year+1)[2:]}"
    else:  # Enero-Julio
        return f"{year-1}/{str(year)[2:]}"

def load_and_process_file(filepath, expected_season):
    """Carga un archivo CSV y lo procesa"""
    
    # Intentar diferentes encodings
    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(filepath, encoding=enc, on_bad_lines='skip')
            break
        except Exception as e:
            continue
    
    if df is None:
        print(f"   ❌ No se pudo cargar: {filepath.name}")
        return None
    
    # Procesar fechas
    if 'Date' in df.columns:
        df['_parsed_date'] = df['Date'].apply(parse_date_flexible)
        
        # Asignar temporada
        df['Season'] = df['_parsed_date'].apply(assign_season)
        
        # Si no se pudo parsear la fecha, usar la temporada esperada
        df['Season'] = df['Season'].fillna(expected_season)
        
        # Normalizar formato de fecha a DD/MM/YY
        def format_date(d):
            if d is not None and pd.notna(d):
                try:
                    return d.strftime('%d/%m/%y')
                except:
                    return None
            return None
        
        df['Date'] = df['_parsed_date'].apply(format_date)
        df = df.drop(columns=['_parsed_date'])
    
    return df

def merge_all_seasons():
    """Fusiona todas las temporadas en un solo archivo"""
    
    raw_dir = Path("data/raw")
    
    print("="*80)
    print("🔄 FUSIÓN DE TODAS LAS TEMPORADAS EN epl_odds.csv")
    print("="*80)
    print()
    
    # Definir archivos fuente en orden cronológico
    season_files = [
        ('epl_odds_backup.csv', '2000/01'),
        ('2001_02.csv', '2001/02'),
        ('2002_03.csv', '2002/03'),
        ('2003_04.csv', '2003/04'),
        ('2004_05.csv', '2004/05'),
        ('2005_06.csv', '2005/06'),
        ('2006_07.csv', '2006/07'),
        ('2007_08.csv', '2007/08'),
        ('2008_09.csv', '2008/09'),
        ('2009_10.csv', '2009/10'),
        ('2010_11.csv', '2010/11'),
        ('2011_12.csv', '2011/12'),
        ('2012_13.csv', '2012/13'),
        ('2013_14.csv', '2013/14'),
        ('2014_15.csv', '2014/15'),
        ('2015_16.csv', '2015/16'),
        ('2016_17.csv', '2016/17'),
        ('2017_18.csv', '2017/18'),
        ('2018_19.csv', '2018/19'),
        ('2019_20.csv', '2019/20'),
        ('2020_21.csv', '2020/21'),
        ('2021_22.csv', '2021/22'),
        ('2022_23.csv', '2022/23'),
        ('2023_24.csv', '2023/24'),
        ('2024_25.csv', '2024/25'),
    ]
    
    print(f"1️⃣ Cargando {len(season_files)} temporadas...")
    print("-" * 80)
    
    all_dfs = []
    total_rows = 0
    
    for filename, expected_season in season_files:
        filepath = raw_dir / filename
        
        if not filepath.exists():
            print(f"   ⚠️  {filename}: No encontrado")
            continue
        
        df = load_and_process_file(filepath, expected_season)
        
        if df is not None:
            # Añadir Div si no existe
            if 'Div' not in df.columns:
                df.insert(0, 'Div', 'E0')
            
            all_dfs.append(df)
            total_rows += len(df)
            print(f"   ✅ {filename:<25} {len(df):>4} partidos | {len(df.columns):>3} columnas")
        else:
            print(f"   ❌ {filename}: Error al cargar")
    
    print()
    print(f"   Total cargado: {total_rows} partidos")
    print()
    
    # 2. Fusionar todos los DataFrames
    print("2️⃣ Fusionando datasets...")
    
    # Usar pd.concat que maneja diferentes columnas automáticamente
    df_merged = pd.concat(all_dfs, ignore_index=True, sort=False)
    
    print(f"   Total fusionado: {len(df_merged)} partidos")
    print(f"   Total columnas: {len(df_merged.columns)}")
    print()
    
    # 3. Reordenar columnas (poner las más importantes primero)
    print("3️⃣ Organizando columnas...")
    
    # Columnas prioritarias
    priority_cols = ['Div', 'Season', 'Date', 'HomeTeam', 'AwayTeam', 
                     'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR',
                     'Referee', 'HS', 'AS', 'HST', 'AST', 'HF', 'AF', 
                     'HC', 'AC', 'HY', 'AY', 'HR', 'AR']
    
    # Columnas de odds (ordenar alfabéticamente)
    other_cols = [c for c in df_merged.columns if c not in priority_cols]
    other_cols.sort()
    
    # Construir orden final
    final_cols = []
    for col in priority_cols:
        if col in df_merged.columns:
            final_cols.append(col)
    
    final_cols.extend(other_cols)
    
    df_merged = df_merged[final_cols]
    
    print(f"   ✅ {len(final_cols)} columnas organizadas")
    print()
    
    # 4. Ordenar por fecha
    print("4️⃣ Ordenando cronológicamente...")
    
    # Crear columna temporal para ordenar
    df_merged['_sort_date'] = df_merged['Date'].apply(parse_date_flexible)
    df_merged = df_merged.sort_values('_sort_date').reset_index(drop=True)
    df_merged = df_merged.drop(columns=['_sort_date'])
    
    print("   ✅ Dataset ordenado por fecha")
    print()
    
    # 5. Estadísticas por temporada
    print("5️⃣ Distribución por temporada:")
    print("-" * 80)
    
    season_counts = df_merged['Season'].value_counts().sort_index()
    
    total_expected = 0
    for season, count in season_counts.items():
        expected = 380
        status = '✅' if count == expected else f'⚠️ (esperado: {expected})'
        print(f"   {season}: {count:>4} partidos {status if count != expected else ''}")
        total_expected += expected
    
    print()
    print(f"   Total: {len(df_merged)} partidos en {len(season_counts)} temporadas")
    print()
    
    # 6. Verificar fechas válidas
    print("6️⃣ Validación de datos:")
    print("-" * 80)
    
    date_nulls = df_merged['Date'].isna().sum()
    season_nulls = df_merged['Season'].isna().sum()
    
    print(f"   Fechas válidas: {len(df_merged) - date_nulls}/{len(df_merged)}")
    print(f"   Temporadas válidas: {len(df_merged) - season_nulls}/{len(df_merged)}")
    print()
    
    # 7. Guardar archivo
    print("7️⃣ Guardando archivo...")
    
    output_path = raw_dir / 'epl_odds.csv'
    df_merged.to_csv(output_path, index=False)
    
    print(f"   💾 Guardado: {output_path}")
    print(f"   Filas: {len(df_merged)}")
    print(f"   Columnas: {len(df_merged.columns)}")
    print()
    
    # 8. Muestra del dataset
    print("8️⃣ Muestra del dataset:")
    print("-" * 80)
    print()
    print(df_merged[['Div', 'Season', 'Date', 'HomeTeam', 'AwayTeam', 'FTR']].head(10).to_string(index=False))
    print()
    print("...")
    print()
    print(df_merged[['Div', 'Season', 'Date', 'HomeTeam', 'AwayTeam', 'FTR']].tail(10).to_string(index=False))
    print()
    
    # 9. Resumen de columnas por categoría
    print("9️⃣ Resumen de columnas:")
    print("-" * 80)
    
    # Categorizar columnas
    match_cols = ['Div', 'Season', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR', 'HTHG', 'HTAG', 'HTR', 'Referee']
    stats_cols = ['HS', 'AS', 'HST', 'AST', 'HF', 'AF', 'HC', 'AC', 'HY', 'AY', 'HR', 'AR']
    odds_cols = [c for c in df_merged.columns if c not in match_cols + stats_cols]
    
    print(f"   Columnas de partido: {len([c for c in match_cols if c in df_merged.columns])}")
    print(f"   Columnas de estadísticas: {len([c for c in stats_cols if c in df_merged.columns])}")
    print(f"   Columnas de odds/mercado: {len(odds_cols)}")
    print()
    
    print("="*80)
    print("✅ FUSIÓN COMPLETADA")
    print("="*80)
    
    return df_merged


if __name__ == '__main__':
    df = merge_all_seasons()
