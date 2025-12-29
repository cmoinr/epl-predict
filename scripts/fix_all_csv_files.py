"""
Script para corregir archivos CSV con columnas extra
Generalizado para cualquier archivo
"""

import csv
from pathlib import Path
import pandas as pd

def fix_csv_extra_columns(input_path, expected_cols=None):
    """Elimina columnas vacías extra que causan errores de parsing"""
    
    input_file = Path(input_path)
    backup_file = input_file.parent / f"{input_file.stem}_original{input_file.suffix}"
    
    print(f"\n🔧 Procesando: {input_file.name}")
    print("-" * 60)
    
    # Backup si no existe
    if not backup_file.exists():
        with open(input_file, 'r', encoding='utf-8', errors='ignore') as f_in:
            content = f_in.read()
        with open(backup_file, 'w', encoding='utf-8', newline='') as f_out:
            f_out.write(content)
    
    # Leer header para conocer número de columnas
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
        header_line = f.readline().strip()
        if expected_cols is None:
            expected_cols = len(header_line.split(','))
    
    print(f"   Columnas esperadas: {expected_cols}")
    
    # Procesar línea por línea
    lines_fixed = 0
    lines_ok = 0
    fixed_lines = [header_line]
    
    with open(input_file, 'r', encoding='utf-8', errors='ignore') as f:
        next(f)  # Skip header
        
        for line_num, line in enumerate(f, start=2):
            line = line.strip()
            if not line:
                continue
            
            try:
                fields = list(csv.reader([line]))[0]
                
                if len(fields) > expected_cols:
                    # Truncar a columnas esperadas
                    fixed_fields = fields[:expected_cols]
                    fixed_line = ','.join(fixed_fields)
                    fixed_lines.append(fixed_line)
                    lines_fixed += 1
                elif len(fields) < expected_cols:
                    # Rellenar con vacíos
                    while len(fields) < expected_cols:
                        fields.append('')
                    fixed_line = ','.join(fields)
                    fixed_lines.append(fixed_line)
                    lines_fixed += 1
                else:
                    fixed_lines.append(line)
                    lines_ok += 1
                    
            except Exception as e:
                print(f"   ⚠️ Error línea {line_num}: {e}")
                fixed_lines.append(line)
    
    # Guardar archivo corregido
    with open(input_file, 'w', encoding='utf-8', newline='') as f:
        f.write('\n'.join(fixed_lines) + '\n')
    
    print(f"   Líneas correctas: {lines_ok}")
    print(f"   Líneas corregidas: {lines_fixed}")
    print(f"   Total: {lines_ok + lines_fixed}")
    
    # Validar
    try:
        df = pd.read_csv(input_file)
        print(f"   ✅ Validación: {len(df)} filas leídas correctamente")
        return len(df)
    except Exception as e:
        print(f"   ❌ Error validación: {e}")
        return 0


def fix_all_problematic_files():
    """Corrige todos los archivos con problemas conocidos"""
    
    raw_dir = Path("data/raw")
    
    print("="*70)
    print("🔧 CORRECCIÓN DE ARCHIVOS CSV PROBLEMÁTICOS")
    print("="*70)
    
    # Lista de archivos a verificar/corregir
    files_to_check = [
        '2004_05.csv',
        '2005_06.csv', 
        '2006_07.csv',
        '2007_08.csv',
        '2008_09.csv',
        '2009_10.csv',
        '2010_11.csv',
        '2011_12.csv',
        '2012_13.csv',
        '2013_14.csv',
        '2014_15.csv',
        '2015_16.csv',
        '2016_17.csv',
        '2017_18.csv',
        '2018_19.csv',
        '2019_20.csv',
        '2020_21.csv',
        '2021_22.csv',
        '2022_23.csv',
        '2023_24.csv',
        '2024_25.csv',
    ]
    
    problems_found = []
    
    for filename in files_to_check:
        filepath = raw_dir / filename
        if not filepath.exists():
            continue
        
        # Intentar leer con pandas
        try:
            df = pd.read_csv(filepath, on_bad_lines='skip', encoding='utf-8')
        except:
            try:
                df = pd.read_csv(filepath, on_bad_lines='skip', encoding='latin-1')
            except:
                continue
        
        # Contar líneas en archivo
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            total_lines = sum(1 for _ in f) - 1
        
        # Si hay diferencia, corregir
        if len(df) < total_lines:
            print(f"\n⚠️ {filename}: {len(df)}/{total_lines} filas ({total_lines - len(df)} perdidas)")
            problems_found.append(filename)
            
            # Corregir
            fix_csv_extra_columns(filepath)
    
    print()
    print("="*70)
    
    if problems_found:
        print(f"✅ Archivos corregidos: {len(problems_found)}")
        for f in problems_found:
            print(f"   - {f}")
    else:
        print("✅ Ningún archivo necesita corrección")
    
    print("="*70)


if __name__ == '__main__':
    fix_all_problematic_files()
