#!/usr/bin/env python3
"""
Script de Análisis de Value Betting
Comparar predicciones del modelo con odds del mercado
Identificar oportunidades con edge positivo
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import json
from datetime import datetime

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from predictor import EPLPredictor
from odds_manager import OddsManager
from odds_comparison import OddsComparison


def load_sample_predictions(predictor, df_historical, sample_matches: list) -> list:
    """
    Generar predicciones para partidos de ejemplo
    
    Parámetros:
    -----------
    predictor : EPLPredictor
        Predictor cargado
    df_historical : pd.DataFrame
        Dataset histórico
    sample_matches : list
        Lista de diccionarios con {home, away, date}
    
    Retorna:
    --------
    list : Lista de predicciones
    """
    predictions = []
    
    for match in sample_matches:
        try:
            pred = predictor.predict_match(
                df_historical=df_historical,
                home_team=match['home'],
                away_team=match['away'],
                match_date=match['date']
            )
            predictions.append(pred)
            print(f"  ✅ {match['home']} vs {match['away']}")
        except Exception as e:
            print(f"  ⚠️  Error prediciendo {match['home']} vs {match['away']}: {e}")
    
    return predictions


def main():
    parser = argparse.ArgumentParser(
        description='💰 Analizar oportunidades de Value Betting',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Ejemplos de uso:
  # Análisis básico
  python analyze_odds.py
  
  # Con umbrales personalizados
  python analyze_odds.py --min-edge 0.05 --min-ev 0.15
  
  # Filtrar por recomendación
  python analyze_odds.py --recommendation BET
  
  # Exportar resultados
  python analyze_odds.py --output results/value_bets.csv
        '''
    )
    
    parser.add_argument('--data', default='data/raw/epl_final.csv',
                       help='Dataset histórico')
    parser.add_argument('--odds', default='data/processed/sample_odds.csv',
                       help='Dataset de odds')
    parser.add_argument('--models', default='models',
                       help='Carpeta con modelos guardados')
    parser.add_argument('--min-edge', type=float, default=0.03,
                       help='Edge mínimo requerido (default: 3%)')
    parser.add_argument('--min-ev', type=float, default=0.10,
                       help='EV mínimo requerido (default: 10%)')
    parser.add_argument('--min-confidence', type=float, default=0.55,
                       help='Confianza mínima (default: 55%)')
    parser.add_argument('--recommendation', choices=['BET', 'CONSIDER', 'MONITOR', 'SKIP'],
                       help='Filtrar por recomendación')
    parser.add_argument('--output', default=None,
                       help='Archivo de salida CSV')
    parser.add_argument('--top', type=int, default=10,
                       help='Mostrar top N oportunidades')
    parser.add_argument('--verbose', action='store_true',
                       help='Mostrar información detallada')
    
    args = parser.parse_args()
    
    try:
        print(f"\n{'='*100}")
        print(f"💰 VALUE BETTING ANALYSIS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*100}\n")
        
        # 1. Cargar datos históricos
        print("📊 Cargando datos históricos...")
        if not Path(args.data).exists():
            print(f"❌ No se encuentra dataset en {args.data}")
            sys.exit(1)
        
        df_historical = pd.read_csv(args.data)
        print(f"   ✅ {len(df_historical)} partidos cargados\n")
        
        # 2. Cargar odds
        print("📊 Cargando odds del mercado...")
        if not Path(args.odds).exists():
            print(f"❌ No se encuentra dataset de odds en {args.odds}")
            sys.exit(1)
        
        df_odds = pd.read_csv(args.odds)
        print(f"   ✅ {len(df_odds)} partidos con odds cargados\n")
        
        # 3. Cargar modelos
        print("🤖 Cargando modelos...")
        predictor = EPLPredictor(args.models)
        print()
        
        # 4. Generar predicciones
        print("🔮 Generando predicciones...")
        
        # Seleccionar partidos del dataset de odds que tenemos
        sample_matches = df_odds[['home_team', 'away_team', 'date']].head(10).to_dict('records')
        for i, match in enumerate(sample_matches):
            match['date'] = pd.to_datetime(match['date']).strftime('%Y-%m-%d')
        
        predictions = load_sample_predictions(predictor, df_historical, sample_matches)
        print()
        
        if len(predictions) == 0:
            print("❌ No se pudieron generar predicciones")
            sys.exit(1)
        
        # 5. Preparar datos de odds
        print("⚽ Preparando datos de odds...")
        odds_list = []
        for match in sample_matches:
            match_odds = df_odds[
                (df_odds['home_team'] == match['home']) &
                (df_odds['away_team'] == match['away']) &
                (pd.to_datetime(df_odds['date']).dt.date == pd.to_datetime(match['date']).date())
            ]
            
            if len(match_odds) > 0:
                row = match_odds.iloc[0]
                odds_list.append({
                    'match_id': f"{match['home']}_vs_{match['away']}",
                    'date': match['date'],
                    'home_team': match['home'],
                    'away_team': match['away'],
                    'home_win_odds': float(row['home_win_odds']),
                    'draw_odds': float(row['draw_odds']),
                    'away_win_odds': float(row['away_win_odds']),
                    'actual_result': row['result'],
                    'home_goals': int(row['home_goals']),
                    'away_goals': int(row['away_goals'])
                })
        
        print(f"   ✅ {len(odds_list)} partidos con odds\n")
        
        # 6. Análisis de value betting
        print("💡 Analizando oportunidades de value betting...")
        comparator = OddsComparison(
            min_edge=args.min_edge,
            min_ev=args.min_ev,
            min_confidence=args.min_confidence
        )
        
        df_opportunities = comparator.find_value_bets(
            predictions,
            odds_list,
            confidence_threshold=args.min_confidence,
            edge_threshold=args.min_edge
        )
        print()
        
        if len(df_opportunities) == 0:
            print("⚠️  No se encontraron oportunidades de value betting con los criterios especificados")
        else:
            # 7. Filtrar si se especifica
            if args.recommendation:
                df_opportunities = comparator.filter_opportunities(
                    df_opportunities,
                    recommendation=args.recommendation
                )
                print(f"Filtrado por recomendación: {args.recommendation}\n")
            
            # 8. Mostrar resumen
            comparator.print_summary(df_opportunities, top_n=args.top)
            
            # 9. Estadísticas detalladas
            print("📈 Estadísticas de Oportunidades:")
            print(f"  Total: {len(df_opportunities)}")
            print(f"  Edge promedio: {df_opportunities['edge'].mean():.2%}")
            print(f"  EV promedio: {df_opportunities['expected_value'].mean():.2%}")
            print(f"  Confianza promedio: {df_opportunities['confidence_score'].mean():.1%}\n")
            
            # 10. Exportar si se solicita
            if args.output:
                comparator.export_to_csv(df_opportunities, args.output)
                print()
            
            # 11. Simulación de ROI
            if len(df_opportunities) > 0:
                print("📊 SIMULACIÓN DE ROI (con Kelly 1/4)")
                print("-" * 100)
                
                total_bet = 0
                total_returns = 0
                winning_bets = 0
                losing_bets = 0
                
                for idx, row in df_opportunities.iterrows():
                    kelly = comparator.calculate_kelly_criterion(
                        row['model_probability'],
                        row['market_odds']
                    )
                    kelly_quarter = comparator.calculate_kelly_fraction(kelly, 0.25)
                    
                    # Usar apuesta unitaria * kelly fraction
                    bet_amount = kelly_quarter
                    total_bet += bet_amount
                    
                    # EV
                    returns = bet_amount * (row['market_odds'] - 1) * row['model_probability'] - bet_amount * (1 - row['model_probability'])
                    total_returns += returns
                    
                    if row['expected_value'] > 0:
                        winning_bets += 1
                    else:
                        losing_bets += 1
                
                roi = (total_returns / total_bet * 100) if total_bet > 0 else 0
                
                print(f"\nApostando en {len(df_opportunities)} oportunidades (Kelly 1/4):")
                print(f"  Apuesta total: {total_bet:.2f} unidades")
                print(f"  Retorno esperado: {total_returns:.2f} unidades")
                print(f"  ROI esperado: {roi:.2f}%")
                print(f"  Apuestas ganadoras (EV+): {winning_bets}")
                print(f"  Apuestas perdedoras (EV-): {losing_bets}\n")
        
        print(f"{'='*100}\n")
    
    except KeyboardInterrupt:
        print('\n⚠️  Análisis cancelado')
        sys.exit(0)
    except Exception as e:
        print(f'❌ Error: {e}')
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
