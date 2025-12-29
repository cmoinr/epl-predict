"""
Script para corregir el archivo 2003_04.csv eliminando columnas vacías extra
"""

import csv
from pathlib import Path

def fix_csv():
    """Elimina columnas vacías extra que causan el error de parsing"""
    
    input_file = Path("data/raw/2003_04.csv")
    output_file = Path("data/raw/2003_04_fixed.csv")
    backup_file = Path("data/raw/2003_04_original.csv")
    
    print("="*70)
    print("🔧 CORRECCIÓN DE 2003_04.csv")
    print("="*70)
    print()
    
    # 1. Backup del archivo original
    if not backup_file.exists():
        with open(input_file, 'r', encoding='utf-8') as f_in:
            with open(backup_file, 'w', encoding='utf-8', newline='') as f_out:
                f_out.write(f_in.read())
        print("✅ Backup creado: 2003_04_original.csv")
    else:
        print("ℹ️  Backup ya existe, se mantendrá")
    
    print()
    
    # 2. Leer header para conocer número correcto de columnas
    with open(input_file, 'r', encoding='utf-8') as f:
        header_line = f.readline().strip()
        expected_cols = len(header_line.split(','))
    
    print(f"📊 Columnas esperadas: {expected_cols}")
    print()
    
    # 3. Procesar archivo línea por línea
    lines_fixed = 0
    lines_ok = 0
    
    with open(input_file, 'r', encoding='utf-8') as f_in:
        with open(output_file, 'w', encoding='utf-8', newline='') as f_out:
            # Escribir header
            f_out.write(header_line + '\n')
            
            # Saltar header en lectura
            next(f_in)
            
            # Procesar cada línea
            for line_num, line in enumerate(f_in, start=2):
                line = line.strip()
                
                if not line:
                    continue
                
                # Parsear con csv.reader para manejar comillas
                try:
                    fields = list(csv.reader([line]))[0]
                    
                    # Si tiene más columnas de las esperadas
                    if len(fields) > expected_cols:
                        # Tomar solo las primeras expected_cols columnas
                        fixed_fields = fields[:expected_cols]
                        
                        # Verificar si las columnas extra están todas vacías
                        extra_fields = fields[expected_cols:]
                        if all(f == '' for f in extra_fields):
                            # Escribir línea corregida
                            writer = csv.writer(f_out)
                            writer.writerow(fixed_fields)
                            lines_fixed += 1
                        else:
                            # Si las columnas extra tienen datos, advertir
                            print(f"⚠️  Línea {line_num}: Tiene datos en columnas extra!")
                            print(f"   Extra: {extra_fields[:5]}...")
                            # De todas formas escribir solo las primeras columnas
                            writer = csv.writer(f_out)
                            writer.writerow(fixed_fields)
                            lines_fixed += 1
                    
                    elif len(fields) < expected_cols:
                        # Si faltan columnas, rellenar con vacíos
                        while len(fields) < expected_cols:
                            fields.append('')
                        
                        writer = csv.writer(f_out)
                        writer.writerow(fields)
                        lines_fixed += 1
                    
                    else:
                        # Línea correcta, escribir tal cual
                        f_out.write(line + '\n')
                        lines_ok += 1
                
                except Exception as e:
                    print(f"❌ Error en línea {line_num}: {e}")
                    continue
    
    print()
    print("✅ Procesamiento completado")
    print(f"   Líneas correctas: {lines_ok}")
    print(f"   Líneas corregidas: {lines_fixed}")
    print(f"   Total procesadas: {lines_ok + lines_fixed}")
    print()
    
    # 4. Validar archivo corregido
    print("🔍 Validando archivo corregido...")
    print()
    
    import pandas as pd
    
    try:
        df_fixed = pd.read_csv(output_file)
        print(f"✅ Lectura exitosa: {len(df_fixed)} filas")
        print(f"✅ Columnas: {len(df_fixed.columns)}")
        print()
        
        # Verificar que todas las filas se leyeron
        if len(df_fixed) == 380:
            print("🎉 ¡PERFECTO! Todas las 380 filas fueron recuperadas")
            print()
            
            # Reemplazar archivo original con el corregido
            print("💾 Reemplazando archivo original con versión corregida...")
            
            with open(output_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            with open(input_file, 'w', encoding='utf-8', newline='') as f:
                f.write(content)
            
            print("✅ Archivo 2003_04.csv actualizado")
            print()
            print("📁 Archivos:")
            print(f"   - data/raw/2003_04.csv (corregido)")
            print(f"   - data/raw/2003_04_original.csv (backup)")
            print(f"   - data/raw/2003_04_fixed.csv (puede eliminarse)")
        else:
            print(f"⚠️  Solo se recuperaron {len(df_fixed)} filas de 380 esperadas")
            print("   Revisa el archivo manualmente")
    
    except Exception as e:
        print(f"❌ Error al validar: {e}")
    
    print()
    print("="*70)
    print("✅ CORRECCIÓN COMPLETADA")
    print("="*70)


if __name__ == '__main__':
    fix_csv()
