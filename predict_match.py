#!/usr/bin/env python3
"""
Script para hacer predicciones desde la terminal
Uso: python predict_match.py --home "Chelsea" --away "Liverpool" --date "2025-02-22"
python predict_match.py --home "Man United" --away "Newcastle" --date "2025-12-26"
"""

import sys
import argparse
from pathlib import Path
import pandas as pd

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from predictor import EPLPredictor


def main():
    parser = argparse.ArgumentParser(
        description='[PRED] Predecir resultado y goles de un partido EPL',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Ejemplos:
  # Predicción simple (sin fecha)
  python predict_match.py --home "Chelsea" --away "Liverpool"
  
  # Con fecha específica
  python predict_match.py --home "Manchester City" --away "Arsenal" --date "2025-03-01"
  
  # Solo resultado
  python predict_match.py --home "Chelsea" --away "Liverpool" --quiet
        '''
    )
    
    parser.add_argument('--home', required=True, help='Equipo local (ej: Chelsea)')
    parser.add_argument('--away', required=True, help='Equipo visitante (ej: Liverpool)')
    parser.add_argument('--date', default=None, help='Fecha (formato: YYYY-MM-DD). Si no se proporciona, usa la fecha actual/próxima.')
    parser.add_argument('--data', default='data/raw/epl_final.csv', 
                       help='Ruta al dataset histórico')
    parser.add_argument('--models', default='models',
                       help='Ruta a la carpeta con modelos guardados')
    parser.add_argument('--quiet', action='store_true',
                       help='Mostrar solo predicción, sin detalles')
    
    args = parser.parse_args()
    
    try:
        # Cargar datos históricos
        if not Path(args.data).exists():
            print(f'[ERROR] No se encuentra dataset en {args.data}')
            sys.exit(1)
        
        df = pd.read_csv(args.data)
        print(f'[OK] Dataset cargado: {args.data} ({len(df)} partidos)\n')
        
        # Cargar modelos
        predictor = EPLPredictor(args.models)
        
        # Si no hay fecha, usar la fecha actual
        match_date = args.date
        if not match_date:
            from datetime import datetime
            match_date = datetime.now().strftime('%Y-%m-%d')
            print(f'[DATE] Usando fecha actual: {match_date}\n')
        
        # Hacer predicción
        print(f'[PRED] Prediciendo: {args.home} vs {args.away} ({match_date})...\n')
        result = predictor.predict_match(
            df_historical=df,
            home_team=args.home,
            away_team=args.away,
            match_date=match_date
        )
        
        # Mostrar resultado
        if args.quiet:
            # Mostrar solo resultado
            print(f"{result['resultado']['random_forest']['prediccion']}")
        else:
            # Mostrar detallado
            predictor.print_prediction(result, verbose=True)
        
    except KeyboardInterrupt:
        print('\n[WARN] Predicción cancelada')
        sys.exit(0)
    except Exception as e:
        print(f'[ERROR] Error: {e}')
        sys.exit(1)


if __name__ == '__main__':
    main()
