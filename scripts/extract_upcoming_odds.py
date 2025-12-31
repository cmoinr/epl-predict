"""
Script especializado para extraer partidos futuros CON ODDS
Genera CSV con formato: date, home_team, away_team, odds...
Compatible con sample_odds_history.csv
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from scripts.flashscore_scraper import FlashscoreScraper, SELENIUM_AVAILABLE


def extract_upcoming_with_odds(max_matches=10, headless=False, output_file=None):
    """
    Extrae partidos futuros con odds desde Flashscore
    
    Args:
        max_matches: Cantidad máxima de partidos a procesar
        headless: Si True, ejecuta sin ventana
        output_file: Ruta del archivo de salida (None = auto)
    
    Returns:
        DataFrame con los datos extraídos
    """
    if not SELENIUM_AVAILABLE:
        print("❌ Selenium no disponible. Instala: pip install selenium")
        return None
    
    print("="*70)
    print("🎯 EXTRACCIÓN DE PARTIDOS FUTUROS + ODDS")
    print("="*70)
    print(f"\nConfiguración:")
    print(f"   - Máximo de partidos: {max_matches}")
    print(f"   - Modo headless: {headless}")
    print(f"   - Formato: Compatible con sample_odds_history.csv")
    print()
    
    # Crear scraper
    scraper = FlashscoreScraper(use_selenium=True, headless=headless)
    
    try:
        # Extraer datos
        matches_data = scraper.get_upcoming_matches_with_odds(max_matches=max_matches)
        
        if not matches_data:
            print("\n❌ No se pudieron extraer datos")
            return None
        
        # Crear DataFrame
        df = pd.DataFrame(matches_data)
        
        # Reordenar columnas según formato requerido
        columns_order = [
            'date', 'home_team', 'away_team',
            'home_win_odds', 'draw_odds', 'away_win_odds',
            'over_2_5_odds', 'under_2_5_odds',
            'both_score_yes', 'both_score_no'
        ]
        
        df = df[columns_order]
        
        # Estadísticas
        print("\n" + "="*70)
        print("📊 ESTADÍSTICAS DE EXTRACCIÓN")
        print("="*70)
        print(f"Total partidos extraídos: {len(df)}")
        print(f"\nCobertura de datos:")
        for col in columns_order[3:]:  # Solo odds
            coverage = df[col].notna().sum()
            pct = (coverage / len(df) * 100) if len(df) > 0 else 0
            print(f"   {col:20s}: {coverage}/{len(df)} ({pct:.1f}%)")
        
        # Guardar archivo
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = Path(__file__).parent.parent / 'data' / 'processed' / f'upcoming_odds_{timestamp}.csv'
        else:
            output_file = Path(output_file)
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print(f"\n💾 Datos guardados en: {output_file}")
        
        # Mostrar preview
        print("\n" + "="*70)
        print("📋 PREVIEW DE DATOS EXTRAÍDOS")
        print("="*70)
        print(df.head(10).to_string(index=False))
        
        return df
        
    finally:
        scraper.close()


def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Extrae partidos futuros con odds desde Flashscore',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Ejemplos:
  # Extraer 5 partidos con ventana visible (para debug)
  python extract_upcoming_odds.py --max 5 --visible
  
  # Extraer 20 partidos en modo headless
  python extract_upcoming_odds.py --max 20
  
  # Guardar en archivo específico
  python extract_upcoming_odds.py --max 10 --output my_odds.csv
        '''
    )
    
    parser.add_argument('--max', type=int, default=10,
                       help='Cantidad máxima de partidos a extraer (default: 10)')
    parser.add_argument('--visible', action='store_true',
                       help='Mostrar ventana del navegador (útil para debug)')
    parser.add_argument('--output', type=str, default=None,
                       help='Archivo de salida (default: auto con timestamp)')
    
    args = parser.parse_args()
    
    headless = not args.visible
    
    df = extract_upcoming_with_odds(
        max_matches=args.max,
        headless=headless,
        output_file=args.output
    )
    
    if df is not None:
        print("\n" + "="*70)
        print("✅ EXTRACCIÓN COMPLETADA")
        print("="*70)
        
        print("\n💡 PRÓXIMOS PASOS:")
        print("   1. Revisa los datos extraídos")
        print("   2. Si faltan odds, ajusta selectores CSS en flashscore_scraper.py")
        print("   3. Usa estos datos con get_value_bets.py para predicciones")
    else:
        print("\n" + "="*70)
        print("❌ EXTRACCIÓN FALLIDA")
        print("="*70)
        
        print("\n💡 POSIBLES SOLUCIONES:")
        print("   1. Ejecuta con --visible para ver qué pasa")
        print("   2. Verifica que Flashscore tenga partidos futuros")
        print("   3. Revisa flashscore_debug.html para inspeccionar estructura")
        print("   4. Ajusta selectores CSS si cambió la estructura")


if __name__ == '__main__':
    main()
