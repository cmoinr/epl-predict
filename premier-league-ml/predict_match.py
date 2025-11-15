#!/usr/bin/env python3
"""
Script para hacer predicciones desde la terminal
Uso: python predict_match.py --home "Chelsea" --away "Liverpool" --date "2025-02-22"
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
        description='🔮 Predecir resultado y goles de un partido EPL',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Ejemplos:
  python predict_match.py --home "Chelsea" --away "Liverpool" --date "2025-02-22"
  python predict_match.py --home "Manchester City" --away "Arsenal" --date "2025-03-01"
        '''
    )
    
    parser.add_argument('--home', required=True, help='Equipo local (ej: Chelsea)')
    parser.add_argument('--away', required=True, help='Equipo visitante (ej: Liverpool)')
    parser.add_argument('--date', required=True, help='Fecha (formato: YYYY-MM-DD)')
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
            print(f'❌ Error: No se encuentra dataset en {args.data}')
            sys.exit(1)
        
        df = pd.read_csv(args.data)
        print(f'📊 Dataset cargado: {args.data} ({len(df)} partidos)')
        
        # Cargar modelos
        predictor = EPLPredictor(args.models)
        
        # Hacer predicción
        print(f'🔮 Prediciendo: {args.home} vs {args.away} ({args.date})...')
        result = predictor.predict_match(
            df_historical=df,
            home_team=args.home,
            away_team=args.away,
            match_date=args.date
        )
        
        # Mostrar resultado
        if args.quiet:
            # Mostrar solo resultado
            print(f"\n{result['resultado']['random_forest']['prediccion']}")
        else:
            # Mostrar detallado
            predictor.print_prediction(result, verbose=True)
        
    except KeyboardInterrupt:
        print('\n⚠️  Predicción cancelada')
        sys.exit(0)
    except Exception as e:
        print(f'❌ Error: {e}')
        sys.exit(1)


if __name__ == '__main__':
    main()
