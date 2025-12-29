"""
Análisis detallado de errores en 2003_04.csv
Identifica líneas problemáticas y sugiere correcciones
"""

import csv
import pandas as pd
from pathlib import Path

def analyze_csv_errors():
    """Analiza errores línea por línea en 2003_04.csv"""
    
    file_path = Path("data/raw/2003_04.csv")
    
    print("="*70)
    print("🔍 ANÁLISIS DETALLADO DE ERRORES EN 2003_04.csv")
    print("="*70)
    print()
    
    # 1. Leer header para saber cuántas columnas se esperan
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        header = f.readline().strip()
        expected_cols = len(header.split(','))
    
    print(f"1️⃣ Columnas esperadas según header: {expected_cols}")
    print()
    
    # 2. Leer todas las líneas y detectar problemas
    print("2️⃣ Analizando cada línea...")
    print("-" * 70)
    
    problematic_lines = []
    good_lines = []
    
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        
        # Procesar cada línea (saltar header)
        for line_num, line in enumerate(lines[1:], start=2):
            line = line.strip()
            
            if not line:
                continue
            
            # Contar columnas en esta línea
            # Usar csv.reader para manejar comas dentro de comillas
            try:
                fields = list(csv.reader([line]))[0]
                num_cols = len(fields)
                
                if num_cols != expected_cols:
                    problematic_lines.append({
                        'line_num': line_num,
                        'expected': expected_cols,
                        'found': num_cols,
                        'difference': num_cols - expected_cols,
                        'content': line[:150] + '...' if len(line) > 150 else line
                    })
                else:
                    good_lines.append(line_num)
                    
            except Exception as e:
                problematic_lines.append({
                    'line_num': line_num,
                    'expected': expected_cols,
                    'found': '?',
                    'difference': '?',
                    'error': str(e),
                    'content': line[:150] + '...' if len(line) > 150 else line
                })
    
    print(f"✅ Líneas correctas: {len(good_lines)}")
    print(f"❌ Líneas problemáticas: {len(problematic_lines)}")
    print()
    
    # 3. Mostrar detalles de líneas problemáticas
    if problematic_lines:
        print("3️⃣ DETALLES DE LÍNEAS PROBLEMÁTICAS:")
        print("-" * 70)
        
        # Agrupar por tipo de error
        extra_cols = [l for l in problematic_lines if isinstance(l['difference'], int) and l['difference'] > 0]
        missing_cols = [l for l in problematic_lines if isinstance(l['difference'], int) and l['difference'] < 0]
        parse_errors = [l for l in problematic_lines if 'error' in l]
        
        if extra_cols:
            print(f"\n📊 Líneas con COLUMNAS EXTRA ({len(extra_cols)}):")
            print("-" * 70)
            
            for item in extra_cols[:10]:  # Mostrar primeras 10
                print(f"\n   Línea {item['line_num']}:")
                print(f"   Esperadas: {item['expected']} | Encontradas: {item['found']} | Extra: +{item['difference']}")
                print(f"   Contenido: {item['content']}")
        
        if missing_cols:
            print(f"\n📊 Líneas con COLUMNAS FALTANTES ({len(missing_cols)}):")
            print("-" * 70)
            
            for item in missing_cols[:5]:
                print(f"\n   Línea {item['line_num']}:")
                print(f"   Esperadas: {item['expected']} | Encontradas: {item['found']} | Faltan: {item['difference']}")
                print(f"   Contenido: {item['content']}")
        
        if parse_errors:
            print(f"\n📊 Líneas con ERRORES DE PARSING ({len(parse_errors)}):")
            print("-" * 70)
            
            for item in parse_errors[:5]:
                print(f"\n   Línea {item['line_num']}:")
                print(f"   Error: {item['error']}")
                print(f"   Contenido: {item['content']}")
    
    print()
    
    # 4. Intentar leer con pandas y ver qué líneas fallan
    print("4️⃣ Verificando lectura con pandas...")
    print("-" * 70)
    
    try:
        # Intentar leer normalmente
        df_normal = pd.read_csv(file_path, on_bad_lines='warn')
        print(f"   Lectura normal: {len(df_normal)} filas")
    except Exception as e:
        print(f"   ❌ Error en lectura normal: {e}")
        df_normal = None
    
    try:
        # Intentar con skip_bad_lines
        df_skip = pd.read_csv(file_path, on_bad_lines='skip')
        print(f"   Saltando líneas malas: {len(df_skip)} filas")
        
        if df_normal is not None:
            lines_skipped = len(df_normal) - len(df_skip) if len(df_normal) > len(df_skip) else 0
            if lines_skipped > 0:
                print(f"   ⚠️  Se saltaron {lines_skipped} líneas")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print()
    
    # 5. Análisis de patrón de error
    print("5️⃣ ANÁLISIS DE PATRÓN:")
    print("-" * 70)
    
    if len(problematic_lines) > 0:
        # Ver si hay patrón en los números de línea
        line_nums = [l['line_num'] for l in problematic_lines]
        
        print(f"   Primera línea problemática: {min(line_nums)}")
        print(f"   Última línea problemática: {max(line_nums)}")
        
        # Ver si hay rangos consecutivos
        consecutive = []
        current_range = [line_nums[0]]
        
        for i in range(1, len(line_nums)):
            if line_nums[i] == line_nums[i-1] + 1:
                current_range.append(line_nums[i])
            else:
                if len(current_range) > 1:
                    consecutive.append((current_range[0], current_range[-1]))
                current_range = [line_nums[i]]
        
        if len(current_range) > 1:
            consecutive.append((current_range[0], current_range[-1]))
        
        if consecutive:
            print(f"\n   Rangos consecutivos de errores:")
            for start, end in consecutive:
                print(f"      Líneas {start}-{end} ({end-start+1} líneas)")
    
    print()
    
    # 6. Resumen y recomendaciones
    print("="*70)
    print("📝 RESUMEN Y RECOMENDACIONES")
    print("="*70)
    print()
    
    print(f"Total de líneas en archivo: {len(lines)}")
    print(f"Líneas de datos esperadas: {len(lines) - 1}")
    print(f"Líneas leídas correctamente: {len(good_lines)}")
    print(f"Líneas con problemas: {len(problematic_lines)}")
    print(f"Líneas perdidas: {len(lines) - 1 - len(good_lines)}")
    print()
    
    if len(problematic_lines) > 0:
        # Determinar causa principal
        if len(extra_cols) > len(missing_cols):
            print("🔍 CAUSA PRINCIPAL: Columnas extra")
            print()
            print("   Probable razón:")
            print("   - Comas dentro de campos sin comillas")
            print("   - Nombres de equipos o árbitros con comas")
            print("   - Datos mal formateados en campos de texto")
            print()
            print("   Solución recomendada:")
            print("   - Identificar qué campo tiene las comas extra")
            print("   - Envolver esos campos en comillas dobles")
            print("   - O eliminar las comas extra")
        else:
            print("🔍 CAUSA PRINCIPAL: Columnas faltantes")
            print()
            print("   Solución recomendada:")
            print("   - Completar campos vacíos con valores por defecto")
    
    print()
    print("🛠️  PRÓXIMOS PASOS:")
    print("   1. Examinar manualmente las líneas problemáticas")
    print("   2. Crear script de corrección específico")
    print("   3. Validar datos corregidos")
    print()
    
    return problematic_lines, good_lines


if __name__ == '__main__':
    problematic, good = analyze_csv_errors()
    
    # Guardar reporte
    if problematic:
        with open('data/raw/2003_04_error_report.txt', 'w', encoding='utf-8') as f:
            f.write("REPORTE DE ERRORES - 2003_04.csv\n")
            f.write("="*70 + "\n\n")
            
            for item in problematic:
                f.write(f"Línea {item['line_num']}:\n")
                f.write(f"  Esperadas: {item['expected']} columnas\n")
                f.write(f"  Encontradas: {item['found']} columnas\n")
                if 'difference' in item:
                    f.write(f"  Diferencia: {item['difference']}\n")
                if 'error' in item:
                    f.write(f"  Error: {item['error']}\n")
                f.write(f"  Contenido: {item['content']}\n")
                f.write("-"*70 + "\n\n")
        
        print(f"💾 Reporte detallado guardado en: data/raw/2003_04_error_report.txt")
