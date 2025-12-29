"""
Análisis de estructura de todos los archivos de temporadas
para determinar cómo fusionarlos correctamente
"""

import pandas as pd
from pathlib import Path
import csv

def analyze_all_seasons():
    """Analiza la estructura de todos los archivos CSV de temporadas"""
    
    raw_dir = Path("data/raw")
    
    print("="*80)
    print("📊 ANÁLISIS DE ESTRUCTURA DE TODAS LAS TEMPORADAS")
    print("="*80)
    print()
    
    # Archivos a analizar (ordenados cronológicamente)
    season_files = sorted([f for f in raw_dir.glob("*.csv") 
                          if f.stem.replace('_', '/').replace('/', '').isdigit() 
                          or f.stem in ['epl_odds_backup']])
    
    # También incluir archivos con formato YYYY_YY
    all_csvs = list(raw_dir.glob("*.csv"))
    season_files = []
    
    for f in all_csvs:
        name = f.stem
        # Patrones: 2004_05, epl_odds_backup
        if '_' in name and len(name.split('_')) == 2:
            parts = name.split('_')
            if len(parts[0]) == 4 and parts[0].isdigit() and len(parts[1]) == 2 and parts[1].isdigit():
                season_files.append(f)
        elif name == 'epl_odds_backup':
            season_files.append(f)
    
    # Ordenar por año
    def get_year(f):
        if f.stem == 'epl_odds_backup':
            return 2000
        return int(f.stem.split('_')[0])
    
    season_files.sort(key=get_year)
    
    print(f"Archivos encontrados: {len(season_files)}")
    print()
    
    # Analizar cada archivo
    results = []
    
    for filepath in season_files:
        try:
            # Leer header
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                header = f.readline().strip()
                columns = header.split(',')
                num_cols = len(columns)
            
            # Intentar leer con pandas (skip bad lines)
            df = pd.read_csv(filepath, on_bad_lines='skip')
            rows_loaded = len(df)
            
            # Contar líneas totales
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                total_lines = sum(1 for _ in f) - 1  # -1 por header
            
            rows_lost = total_lines - rows_loaded
            
            # Detectar temporada
            if filepath.stem == 'epl_odds_backup':
                season = '2000/01'
            else:
                year = filepath.stem.split('_')[0]
                year_end = filepath.stem.split('_')[1]
                season = f"{year}/{year_end}"
            
            results.append({
                'file': filepath.name,
                'season': season,
                'columns': num_cols,
                'total_rows': total_lines,
                'loaded_rows': rows_loaded,
                'lost_rows': rows_lost,
                'status': '✅' if rows_lost == 0 else '⚠️'
            })
            
        except Exception as e:
            results.append({
                'file': filepath.name,
                'season': '?',
                'columns': '?',
                'total_rows': '?',
                'loaded_rows': '?',
                'lost_rows': '?',
                'status': f'❌ {str(e)[:30]}'
            })
    
    # Mostrar tabla de resultados
    print(f"{'Archivo':<20} {'Temporada':<10} {'Columnas':<10} {'Total':<8} {'Cargadas':<10} {'Perdidas':<10} {'Estado'}")
    print("-" * 90)
    
    for r in results:
        print(f"{r['file']:<20} {r['season']:<10} {r['columns']:<10} {r['total_rows']:<8} {r['loaded_rows']:<10} {r['lost_rows']:<10} {r['status']}")
    
    print()
    
    # Agrupar por número de columnas
    print("="*80)
    print("📊 AGRUPACIÓN POR NÚMERO DE COLUMNAS")
    print("="*80)
    print()
    
    col_groups = {}
    for r in results:
        cols = r['columns']
        if cols not in col_groups:
            col_groups[cols] = []
        col_groups[cols].append(r['season'])
    
    for cols, seasons in sorted(col_groups.items(), key=lambda x: str(x[0])):
        print(f"{cols} columnas: {len(seasons)} temporadas")
        print(f"   {', '.join(seasons)}")
        print()
    
    # Identificar columnas comunes
    print("="*80)
    print("📊 COLUMNAS COMUNES ENTRE ARCHIVOS")
    print("="*80)
    print()
    
    # Leer columnas de cada archivo
    all_columns = {}
    for filepath in season_files:
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                header = f.readline().strip()
                columns = header.split(',')
                all_columns[filepath.stem] = set(columns)
        except:
            pass
    
    # Encontrar intersección de todas las columnas
    if all_columns:
        common_cols = set.intersection(*all_columns.values())
        print(f"Columnas comunes a TODOS los archivos: {len(common_cols)}")
        print()
        
        # Columnas básicas que esperamos
        expected_basic = ['Div', 'Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
        
        print("Columnas básicas presentes en todos:")
        for col in expected_basic:
            status = '✅' if col in common_cols else '❌'
            print(f"   {status} {col}")
        
        print()
        print("Todas las columnas comunes:")
        print(sorted(common_cols))
    
    # Archivos con problemas
    print()
    print("="*80)
    print("⚠️ ARCHIVOS CON FILAS PERDIDAS")
    print("="*80)
    print()
    
    problem_files = [r for r in results if r['lost_rows'] != 0 and r['lost_rows'] != '?']
    if problem_files:
        for r in problem_files:
            print(f"   {r['file']}: {r['lost_rows']} filas perdidas")
    else:
        print("   ✅ Ningún archivo tiene filas perdidas")
    
    return results, all_columns


if __name__ == '__main__':
    results, columns = analyze_all_seasons()
