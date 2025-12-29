"""
Script para fusionar archivos CSV de odds en epl_odds.csv
Maneja diferencias en columnas entre temporadas
"""

import pandas as pd
from pathlib import Path

def merge_odds_files():
    """Fusiona los archivos de odds nuevos con epl_odds.csv"""
    
    base_path = Path("data/raw")
    
    # Archivos a fusionar
    files = [
        base_path / "epl_odds.csv",      # Temporada 2000/01 (380 partidos)
        base_path / "2001_02.csv",       # Temporada 2001/02
        base_path / "2002_03.csv",       # Temporada 2002/03
        base_path / "2003_04.csv"        # Temporada 2003/04
    ]
    
    print("="*70)
    print("🔄 FUSIÓN DE ARCHIVOS DE ODDS")
    print("="*70)
    print()
    
    # Cargar cada archivo
    dfs = []
    total_rows = 0
    
    for file in files:
        if not file.exists():
            print(f"⚠️  {file.name} no encontrado - saltando")
            continue
        
        try:
            # Intentar leer con diferentes configuraciones
            try:
                df = pd.read_csv(file, encoding='utf-8')
            except:
                df = pd.read_csv(file, encoding='latin-1')
            
            print(f"✅ {file.name:20s}: {len(df):4d} partidos | {len(df.columns):2d} columnas")
            
            # Mostrar primeras columnas
            cols_preview = ', '.join(df.columns[:8].tolist())
            print(f"   Columnas: {cols_preview}...")
            
            dfs.append(df)
            total_rows += len(df)
            
        except Exception as e:
            print(f"❌ Error leyendo {file.name}: {e}")
            continue
    
    if not dfs:
        print("\n❌ No se pudo cargar ningún archivo")
        return
    
    print(f"\n📊 Total antes de fusionar: {total_rows} partidos")
    print()
    
    # Verificar columnas comunes
    print("🔍 Análisis de columnas:")
    print("-" * 70)
    
    all_columns = set()
    for df in dfs:
        all_columns.update(df.columns)
    
    # Columnas que están en todos los archivos
    common_columns = set(dfs[0].columns)
    for df in dfs[1:]:
        common_columns &= set(df.columns)
    
    print(f"   Columnas comunes a todos: {len(common_columns)}")
    print(f"   Columnas totales únicas:  {len(all_columns)}")
    
    # Mostrar diferencias
    unique_to_files = {}
    for i, df in enumerate(dfs):
        unique = set(df.columns) - common_columns
        if unique:
            unique_to_files[files[i].name] = unique
    
    if unique_to_files:
        print(f"\n   Columnas únicas por archivo:")
        for filename, cols in unique_to_files.items():
            print(f"      {filename:20s}: {', '.join(sorted(cols))}")
    
    print()
    
    # Fusionar usando concat (mantiene todas las columnas, rellena NaN donde falten)
    print("🔄 Fusionando archivos...")
    df_merged = pd.concat(dfs, ignore_index=True, sort=False)
    
    # Remover duplicados por si acaso
    before_dedup = len(df_merged)
    df_merged = df_merged.drop_duplicates(subset=['Date', 'HomeTeam', 'AwayTeam'], keep='first')
    after_dedup = len(df_merged)
    
    if before_dedup > after_dedup:
        print(f"   ⚠️  Removidos {before_dedup - after_dedup} duplicados")
    
    # Ordenar por fecha
    df_merged['Date'] = pd.to_datetime(df_merged['Date'], format='%d/%m/%y', errors='coerce')
    df_merged = df_merged.sort_values('Date').reset_index(drop=True)
    
    # Convertir fecha de vuelta a string en formato original
    df_merged['Date'] = df_merged['Date'].dt.strftime('%d/%m/%y')
    
    print(f"✅ Fusión completada: {len(df_merged)} partidos totales")
    print(f"   Columnas en archivo final: {len(df_merged.columns)}")
    print()
    
    # Guardar
    output_file = base_path / "epl_odds.csv"
    backup_file = base_path / "epl_odds_backup.csv"
    
    # Crear backup del original
    if output_file.exists():
        print(f"💾 Creando backup: {backup_file.name}")
        import shutil
        shutil.copy(output_file, backup_file)
    
    df_merged.to_csv(output_file, index=False)
    print(f"💾 Guardado en: {output_file}")
    
    # Resumen por temporada (aproximado)
    print()
    print("📅 Distribución por año:")
    print("-" * 70)
    
    df_merged['Date'] = pd.to_datetime(df_merged['Date'], format='%d/%m/%y')
    year_counts = df_merged.groupby(df_merged['Date'].dt.year).size()
    
    for year, count in year_counts.items():
        print(f"   {year}: {count:3d} partidos")
    
    print()
    print("="*70)
    print("✅ FUSIÓN COMPLETADA EXITOSAMENTE")
    print("="*70)
    print()
    print("📝 Próximos pasos:")
    print("   1. Re-ejecutar: python scripts/integrate_market_data.py")
    print("   2. Ahora tendrás ~1,520 partidos con odds (vs 380 anteriores)")
    print("   3. Features de mercado más robustas")
    print()
    
    return df_merged


if __name__ == '__main__':
    merge_odds_files()
