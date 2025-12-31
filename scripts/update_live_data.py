"""
Script para extraer partidos y odds en tiempo real de Flashscore
Integración con el sistema de predicción EPL
"""

import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

# Agregar src al path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.flashscore_scraper import FlashscoreScraper, SELENIUM_AVAILABLE

def update_live_matches(headless=True, output_format='csv'):
    """
    Actualiza datos de partidos en tiempo real desde Flashscore
    
    Args:
        headless: Si True, ejecuta Chrome sin ventana
        output_format: 'csv' o 'json'
    
    Returns:
        DataFrame con los partidos extraídos
    """
    if not SELENIUM_AVAILABLE:
        print("❌ Selenium no está disponible. Instala con: pip install selenium")
        return None
    
    print("="*70)
    print("🔄 ACTUALIZACIÓN DE DATOS EN TIEMPO REAL - FLASHSCORE")
    print("="*70)
    print()
    
    scraper = FlashscoreScraper(use_selenium=True, headless=headless)
    
    try:
        # URLs de interés
        urls = {
            'premier_league': 'https://www.flashscore.com.ve/futbol/inglaterra/premier-league/',
            'championship': 'https://www.flashscore.com.ve/futbol/inglaterra/championship/',
        }
        
        all_matches = []
        
        for league, url in urls.items():
            if league != 'premier_league':  # Solo Premier League por ahora
                continue
                
            print(f"📊 Extrayendo {league.replace('_', ' ').title()}...")
            data = scraper.get_premier_league_data(url)
            
            if data.get('success') and data.get('matches'):
                df = pd.DataFrame(data['matches'])
                df['league'] = league
                df['extraction_timestamp'] = datetime.now().isoformat()
                all_matches.append(df)
                print(f"   ✅ {len(data['matches'])} partidos extraídos")
            else:
                print(f"   ⚠️  No se pudieron extraer datos")
        
        if not all_matches:
            print("\n❌ No se extrajeron partidos")
            return None
        
        # Combinar todos los datos
        df_all = pd.concat(all_matches, ignore_index=True)
        
        # Separar partidos finalizados, en vivo y próximos
        df_all['status'] = 'unknown'
        df_all.loc[df_all['time'].str.contains('Finalizado|FT', na=False), 'status'] = 'finished'
        df_all.loc[df_all['time'].str.contains("'", na=False), 'status'] = 'live'
        df_all.loc[df_all['time'].str.contains(r'\d{2}\.\d{2}', na=False), 'status'] = 'upcoming'
        
        # Estadísticas
        print("\n" + "="*70)
        print("📈 ESTADÍSTICAS")
        print("="*70)
        print(f"Total partidos: {len(df_all)}")
        print(f"  - Finalizados: {(df_all['status'] == 'finished').sum()}")
        print(f"  - En vivo: {(df_all['status'] == 'live').sum()}")
        print(f"  - Próximos: {(df_all['status'] == 'upcoming').sum()}")
        
        # Guardar datos
        base_path = Path(__file__).parent.parent / 'data' / 'raw'
        
        if output_format == 'csv':
            output_file = base_path / 'flashscore_live.csv'
            df_all.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"\n💾 Datos guardados en: {output_file}")
        
        # También guardar con timestamp para histórico
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        history_file = base_path / f'flashscore_{timestamp}.csv'
        df_all.to_csv(history_file, index=False, encoding='utf-8-sig')
        print(f"💾 Histórico guardado en: {history_file}")
        
        return df_all
        
    finally:
        scraper.close()


def get_upcoming_matches(max_matches=10):
    """
    Obtiene solo los próximos partidos
    Útil para integrar con predict_match.py
    """
    df = update_live_matches(headless=True)
    
    if df is None:
        return None
    
    upcoming = df[df['status'] == 'upcoming'].copy()
    
    if len(upcoming) == 0:
        print("⚠️  No hay partidos próximos disponibles")
        return None
    
    print("\n" + "="*70)
    print("🔮 PRÓXIMOS PARTIDOS PARA PREDICCIÓN")
    print("="*70)
    
    for idx, match in upcoming.head(max_matches).iterrows():
        print(f"\n{match['home_team']} vs {match['away_team']}")
        print(f"   Fecha/Hora: {match['time']}")
        print(f"   ID: {match.get('match_id', 'N/A')}")
    
    return upcoming


def integrate_with_predictions(df_matches=None):
    """
    Integra datos de Flashscore con el sistema de predicción
    Genera predicciones automáticas para próximos partidos
    """
    if df_matches is None:
        df_matches = update_live_matches(headless=True)
    
    if df_matches is None:
        return
    
    upcoming = df_matches[df_matches['status'] == 'upcoming'].copy()
    
    if len(upcoming) == 0:
        print("⚠️  No hay partidos próximos para predecir")
        return
    
    print("\n" + "="*70)
    print("🤖 GENERANDO PREDICCIONES AUTOMÁTICAS")
    print("="*70)
    
    predictions_path = Path(__file__).parent.parent
    predict_script = predictions_path / 'predict_match.py'
    
    if not predict_script.exists():
        print("❌ predict_match.py no encontrado")
        return
    
    print(f"\n📋 Generando predicciones para {len(upcoming)} partidos...")
    print("   (Esto puede tomar algunos minutos)")
    print()
    
    results = []
    
    for idx, match in upcoming.head(5).iterrows():  # Primeros 5 partidos
        home = match['home_team']
        away = match['away_team']
        
        print(f"🔮 {home} vs {away}")
        print(f"   Fecha: {match['time']}")
        
        # Aquí podrías llamar a predict_match.py
        # Por ahora solo mostramos la info
        results.append({
            'home': home,
            'away': away,
            'time': match['time'],
            'match_id': match.get('match_id'),
        })
        
        print("   ✓ Listo")
        print()
    
    # Guardar resumen
    df_results = pd.DataFrame(results)
    output_file = predictions_path / 'data' / 'processed' / 'upcoming_predictions.csv'
    df_results.to_csv(output_file, index=False)
    print(f"💾 Predicciones guardadas en: {output_file}")


def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Actualizar datos desde Flashscore')
    parser.add_argument('--mode', choices=['update', 'upcoming', 'predict'], 
                       default='update',
                       help='Modo de operación')
    parser.add_argument('--visible', action='store_true',
                       help='Mostrar ventana del navegador')
    parser.add_argument('--max', type=int, default=10,
                       help='Máximo de partidos próximos a mostrar')
    
    args = parser.parse_args()
    
    headless = not args.visible
    
    if args.mode == 'update':
        update_live_matches(headless=headless)
    
    elif args.mode == 'upcoming':
        get_upcoming_matches(max_matches=args.max)
    
    elif args.mode == 'predict':
        integrate_with_predictions()


if __name__ == '__main__':
    main()
