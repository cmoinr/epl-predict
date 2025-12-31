"""
Backtesting específico para mercados O/U 2.5 y BTTS
Encuentra rangos óptimos para filtros Ultra V2
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent / 'src'))
from predictor import EPLPredictor


def analyze_over_under_performance(df_with_odds, predictor, df_historical):
    """Analiza rendimiento del mercado Over/Under 2.5"""
    
    print("\n" + "="*70)
    print("📊 ANÁLISIS MERCADO OVER/UNDER 2.5")
    print("="*70)
    
    df_bettable = df_with_odds[
        (df_with_odds['Avg>2.5'].notna()) & 
        (df_with_odds['Avg<2.5'].notna())
    ].copy()
    df_bettable = df_bettable.sort_values('MatchDate').reset_index(drop=True)
    
    results = []
    total = len(df_bettable)
    
    print(f"\n🔄 Procesando {total} partidos...")
    
    for idx, row in df_bettable.iterrows():
        if idx % 500 == 0:
            print(f"   Progreso: {idx}/{total} ({idx/total*100:.1f}%)")
        
        try:
            df_until_date = df_historical[
                pd.to_datetime(df_historical['MatchDate']) < pd.to_datetime(row['MatchDate'])
            ]
            
            if len(df_until_date) < 50:
                continue
            
            prediction = predictor.predict_match(
                df_until_date,
                row['HomeTeam'],
                row['AwayTeam'],
                row['MatchDate']
            )
            
            # Obtener predicción de goles
            if 'mejor_modelo' in prediction['goles_totales'] and 'prediccion' in prediction['goles_totales']['mejor_modelo']:
                predicted_goals = prediction['goles_totales']['mejor_modelo']['prediccion']
            elif 'promedio' in prediction['goles_totales'] and prediction['goles_totales']['promedio']:
                predicted_goals = prediction['goles_totales']['promedio']
            else:
                # Fallback: calcular promedio de RF y GB
                rf = prediction['goles_totales'].get('random_forest', 2.5)
                gb = prediction['goles_totales'].get('gradient_boosting', 2.5)
                predicted_goals = (rf + gb) / 2
            
            # Probabilidades basadas en goles predichos
            if predicted_goals >= 3.5:
                prob_over = 0.75
            elif predicted_goals >= 3.0:
                prob_over = 0.65
            elif predicted_goals >= 2.5:
                prob_over = 0.55
            elif predicted_goals >= 2.0:
                prob_over = 0.35
            else:
                prob_over = 0.20
            
            prob_under = 1 - prob_over
            
            # Odds del mercado
            odds_over = row['Avg>2.5']
            odds_under = row['Avg<2.5']
            
            # Calcular edges
            edge_over = prob_over - (1 / odds_over)
            edge_under = prob_under - (1 / odds_under)
            
            # Resultado real
            total_goals = row['FTHG'] + row['FTAG']
            actual_over = total_goals > 2.5
            actual_under = total_goals < 2.5
            
            # Registrar TODAS las oportunidades (sin filtro de edge)
            ev_over = (prob_over * odds_over) - 1
            results.append({
                'market': 'Over 2.5',
                'edge': edge_over,
                'odds': odds_over,
                'model_prob': prob_over,
                'predicted_goals': predicted_goals,
                'actual_goals': total_goals,
                'ev': ev_over,
                'won': actual_over,
                'profit_1u': (odds_over - 1) if actual_over else -1
            })
            
            ev_under = (prob_under * odds_under) - 1
            results.append({
                'market': 'Under 2.5',
                'edge': edge_under,
                'odds': odds_under,
                'model_prob': prob_under,
                'predicted_goals': predicted_goals,
                'actual_goals': total_goals,
                'ev': ev_under,
                'won': actual_under,
                'profit_1u': (odds_under - 1) if actual_under else -1
            })
                
        except Exception as e:
            continue
    
    if not results:
        print("❌ No se generaron predicciones")
        return None
    
    df_results = pd.DataFrame(results)
    
    # Filtrar solo oportunidades con edge positivo
    df_positive = df_results[df_results['edge'] > 0].copy()
    
    # Análisis por rangos
    print(f"\n✅ Total predicciones: {len(df_results)}")
    print(f"✅ Edge positivo: {len(df_positive)} ({len(df_positive)/len(df_results)*100:.1f}%)")
    
    if len(df_positive) == 0:
        print("❌ No hay oportunidades con edge positivo")
        return df_results
    
    print(f"   Win rate (edge+): {df_positive['won'].mean()*100:.1f}%")
    print(f"   ROI (edge+): {df_positive['profit_1u'].mean()*100:.2f}%")
    
    # Usar solo edge positivo para análisis
    df_results = df_positive
    
    # Análisis por rangos de edge
    print("\n📊 ANÁLISIS POR RANGOS DE EDGE:")
    edge_bins = [(0, 0.05), (0.05, 0.10), (0.10, 0.15), (0.15, 0.20), (0.20, 0.30), (0.30, 1.0)]
    
    for min_e, max_e in edge_bins:
        subset = df_results[(df_results['edge'] >= min_e) & (df_results['edge'] < max_e)]
        if len(subset) > 5:
            win_rate = subset['won'].mean() * 100
            roi = subset['profit_1u'].mean() * 100
            count = len(subset)
            avg_odds = subset['odds'].mean()
            print(f"   Edge {min_e*100:.0f}-{max_e*100:.0f}%: {count:4d} bets | WR: {win_rate:5.1f}% | ROI: {roi:6.2f}% | Avg Odds: {avg_odds:.2f}")
    
    # Análisis por rangos de cuotas
    print("\n📊 ANÁLISIS POR RANGOS DE CUOTAS:")
    odds_bins = [(1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 4.0), (4.0, 10.0)]
    
    for min_o, max_o in odds_bins:
        subset = df_results[(df_results['odds'] >= min_o) & (df_results['odds'] < max_o)]
        if len(subset) > 5:
            win_rate = subset['won'].mean() * 100
            roi = subset['profit_1u'].mean() * 100
            count = len(subset)
            avg_edge = subset['edge'].mean() * 100
            print(f"   Odds {min_o:.1f}-{max_o:.1f}: {count:4d} bets | WR: {win_rate:5.1f}% | ROI: {roi:6.2f}% | Avg Edge: {avg_edge:.1f}%")
    
    # Análisis por probabilidad del modelo
    print("\n📊 ANÁLISIS POR PROBABILIDAD DEL MODELO:")
    prob_bins = [(0.0, 0.40), (0.40, 0.50), (0.50, 0.60), (0.60, 0.70), (0.70, 1.0)]
    
    for min_p, max_p in prob_bins:
        subset = df_results[(df_results['model_prob'] >= min_p) & (df_results['model_prob'] < max_p)]
        if len(subset) > 5:
            win_rate = subset['won'].mean() * 100
            roi = subset['profit_1u'].mean() * 100
            count = len(subset)
            avg_edge = subset['edge'].mean() * 100
            print(f"   Prob {min_p*100:.0f}-{max_p*100:.0f}%: {count:4d} bets | WR: {win_rate:5.1f}% | ROI: {roi:6.2f}% | Avg Edge: {avg_edge:.1f}%")
    
    # BUSCAR RANGOS ÓPTIMOS (ROI > 15%)
    print("\n🔥 RANGOS ÓPTIMOS (ROI > 15%):")
    
    best_combos = []
    for min_e, max_e in edge_bins:
        for min_o, max_o in odds_bins:
            for min_p, max_p in prob_bins:
                subset = df_results[
                    (df_results['edge'] >= min_e) & (df_results['edge'] < max_e) &
                    (df_results['odds'] >= min_o) & (df_results['odds'] < max_o) &
                    (df_results['model_prob'] >= min_p) & (df_results['model_prob'] < max_p)
                ]
                if len(subset) >= 10:
                    roi = subset['profit_1u'].mean() * 100
                    if roi > 15:
                        best_combos.append({
                            'edge_range': f"{min_e*100:.0f}-{max_e*100:.0f}%",
                            'odds_range': f"{min_o:.1f}-{max_o:.1f}",
                            'prob_range': f"{min_p*100:.0f}-{max_p*100:.0f}%",
                            'count': len(subset),
                            'win_rate': subset['won'].mean() * 100,
                            'roi': roi
                        })
    
    if best_combos:
        df_best = pd.DataFrame(best_combos).sort_values('roi', ascending=False).head(10)
        print(df_best.to_string(index=False))
    else:
        print("   No se encontraron combinaciones con ROI > 15%")
    
    return df_results


def analyze_btts_performance(df_with_odds, predictor, df_historical):
    """Analiza rendimiento del mercado BTTS (Both Teams To Score)"""
    
    print("\n" + "="*70)
    print("📊 ANÁLISIS MERCADO BTTS (BOTH TEAMS TO SCORE)")
    print("="*70)
    
    df_bettable = df_with_odds[
        (df_with_odds['AvgOdds_BTTS_Yes'].notna()) & 
        (df_with_odds['AvgOdds_BTTS_No'].notna())
    ].copy()
    
    if len(df_bettable) == 0:
        print("❌ No hay datos de BTTS disponibles en el dataset")
        print("   El archivo epl_odds.csv no contiene columnas BTTS")
        return None
    df_bettable = df_bettable.sort_values('MatchDate').reset_index(drop=True)
    
    results = []
    total = len(df_bettable)
    
    print(f"\n🔄 Procesando {total} partidos...")
    
    for idx, row in df_bettable.iterrows():
        if idx % 500 == 0:
            print(f"   Progreso: {idx}/{total} ({idx/total*100:.1f}%)")
        
        try:
            df_until_date = df_historical[
                pd.to_datetime(df_historical['MatchDate']) < pd.to_datetime(row['MatchDate'])
            ]
            
            if len(df_until_date) < 50:
                continue
            
            prediction = predictor.predict_match(
                df_until_date,
                row['HomeTeam'],
                row['AwayTeam'],
                row['MatchDate']
            )
            
            # Obtener predicción BTTS (mejor modelo: XGBoost)
            btts_probs = prediction['ambos_anotan']['mejor_modelo']
            prob_yes = btts_probs['si'] / 100
            prob_no = btts_probs['no'] / 100
            
            # Odds del mercado
            odds_yes = row['AvgOdds_BTTS_Yes']
            odds_no = row['AvgOdds_BTTS_No']
            
            # Calcular edges
            edge_yes = prob_yes - (1 / odds_yes)
            edge_no = prob_no - (1 / odds_no)
            
            # Resultado real
            home_scored = row['FTHG'] > 0
            away_scored = row['FTAG'] > 0
            actual_btts = home_scored and away_scored
            
            # BTTS YES
            if edge_yes > 0:
                ev_yes = (prob_yes * odds_yes) - 1
                results.append({
                    'market': 'BTTS Yes',
                    'edge': edge_yes,
                    'odds': odds_yes,
                    'model_prob': prob_yes,
                    'ev': ev_yes,
                    'won': actual_btts,
                    'profit_1u': (odds_yes - 1) if actual_btts else -1
                })
            
            # BTTS NO
            if edge_no > 0:
                ev_no = (prob_no * odds_no) - 1
                results.append({
                    'market': 'BTTS No',
                    'edge': edge_no,
                    'odds': odds_no,
                    'model_prob': prob_no,
                    'ev': ev_no,
                    'won': not actual_btts,
                    'profit_1u': (odds_no - 1) if not actual_btts else -1
                })
                
        except Exception as e:
            continue
    
    if not results:
        print("❌ No se encontraron oportunidades")
        return None
    
    df_results = pd.DataFrame(results)
    
    # Análisis por rangos
    print(f"\n✅ Total oportunidades encontradas: {len(df_results)}")
    print(f"   Win rate general: {df_results['won'].mean()*100:.1f}%")
    print(f"   ROI general: {df_results['profit_1u'].mean()*100:.2f}%")
    
    # Análisis por rangos de edge
    print("\n📊 ANÁLISIS POR RANGOS DE EDGE:")
    edge_bins = [(0, 0.05), (0.05, 0.10), (0.10, 0.15), (0.15, 0.20), (0.20, 0.30), (0.30, 1.0)]
    
    for min_e, max_e in edge_bins:
        subset = df_results[(df_results['edge'] >= min_e) & (df_results['edge'] < max_e)]
        if len(subset) > 5:
            win_rate = subset['won'].mean() * 100
            roi = subset['profit_1u'].mean() * 100
            count = len(subset)
            avg_odds = subset['odds'].mean()
            print(f"   Edge {min_e*100:.0f}-{max_e*100:.0f}%: {count:4d} bets | WR: {win_rate:5.1f}% | ROI: {roi:6.2f}% | Avg Odds: {avg_odds:.2f}")
    
    # Análisis por rangos de cuotas
    print("\n📊 ANÁLISIS POR RANGOS DE CUOTAS:")
    odds_bins = [(1.0, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 4.0), (4.0, 10.0)]
    
    for min_o, max_o in odds_bins:
        subset = df_results[(df_results['odds'] >= min_o) & (df_results['odds'] < max_o)]
        if len(subset) > 5:
            win_rate = subset['won'].mean() * 100
            roi = subset['profit_1u'].mean() * 100
            count = len(subset)
            avg_edge = subset['edge'].mean() * 100
            print(f"   Odds {min_o:.1f}-{max_o:.1f}: {count:4d} bets | WR: {win_rate:5.1f}% | ROI: {roi:6.2f}% | Avg Edge: {avg_edge:.1f}%")
    
    # Análisis por probabilidad del modelo
    print("\n📊 ANÁLISIS POR PROBABILIDAD DEL MODELO:")
    prob_bins = [(0.0, 0.40), (0.40, 0.50), (0.50, 0.60), (0.60, 0.70), (0.70, 1.0)]
    
    for min_p, max_p in prob_bins:
        subset = df_results[(df_results['model_prob'] >= min_p) & (df_results['model_prob'] < max_p)]
        if len(subset) > 5:
            win_rate = subset['won'].mean() * 100
            roi = subset['profit_1u'].mean() * 100
            count = len(subset)
            avg_edge = subset['edge'].mean() * 100
            print(f"   Prob {min_p*100:.0f}-{max_p*100:.0f}%: {count:4d} bets | WR: {win_rate:5.1f}% | ROI: {roi:6.2f}% | Avg Edge: {avg_edge:.1f}%")
    
    # Análisis por tipo (Yes vs No)
    print("\n📊 ANÁLISIS POR TIPO:")
    for market_type in ['BTTS Yes', 'BTTS No']:
        subset = df_results[df_results['market'] == market_type]
        if len(subset) > 0:
            win_rate = subset['won'].mean() * 100
            roi = subset['profit_1u'].mean() * 100
            count = len(subset)
            avg_edge = subset['edge'].mean() * 100
            print(f"   {market_type:10s}: {count:4d} bets | WR: {win_rate:5.1f}% | ROI: {roi:6.2f}% | Avg Edge: {avg_edge:.1f}%")
    
    # BUSCAR RANGOS ÓPTIMOS (ROI > 15%)
    print("\n🔥 RANGOS ÓPTIMOS (ROI > 15%):")
    
    best_combos = []
    for min_e, max_e in edge_bins:
        for min_o, max_o in odds_bins:
            for min_p, max_p in prob_bins:
                subset = df_results[
                    (df_results['edge'] >= min_e) & (df_results['edge'] < max_e) &
                    (df_results['odds'] >= min_o) & (df_results['odds'] < max_o) &
                    (df_results['model_prob'] >= min_p) & (df_results['model_prob'] < max_p)
                ]
                if len(subset) >= 10:
                    roi = subset['profit_1u'].mean() * 100
                    if roi > 15:
                        best_combos.append({
                            'edge_range': f"{min_e*100:.0f}-{max_e*100:.0f}%",
                            'odds_range': f"{min_o:.1f}-{max_o:.1f}",
                            'prob_range': f"{min_p*100:.0f}-{max_p*100:.0f}%",
                            'count': len(subset),
                            'win_rate': subset['won'].mean() * 100,
                            'roi': roi
                        })
    
    if best_combos:
        df_best = pd.DataFrame(best_combos).sort_values('roi', ascending=False).head(10)
        print(df_best.to_string(index=False))
    else:
        print("   No se encontraron combinaciones con ROI > 15%")
    
    return df_results


def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Análisis de mercado O/U 2.5')
    parser.add_argument('--market', choices=['over', 'btts', 'both'], default='over',
                        help='Mercado a analizar (solo over disponible)')
    parser.add_argument('--sample', type=int, default=1000,
                        help='Límite de partidos (default: 1000)')
    args = parser.parse_args()
    
    # Cargar datos
    data_path = Path(__file__).parent.parent / 'data' / 'raw' / 'epl_odds.csv'
    historical_path = Path(__file__).parent.parent / 'data' / 'raw' / 'epl_final.csv'
    
    if not data_path.exists():
        print("❌ Error: archivo no encontrado")
        print(f"   Esperado: {data_path}")
        return
    
    print("\n" + "="*70)
    print("🎯 ANÁLISIS MERCADO O/U 2.5")
    print("="*70)
    
    df = pd.read_csv(data_path, low_memory=False)
    
    # Parsear fecha con formato dd/mm/yy (año 2 dígitos)
    df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%y', errors='coerce')
    df = df.rename(columns={'Date': 'MatchDate'})
    df = df.dropna(subset=['MatchDate'])
    
    # Filtrar solo partidos con datos O/U
    df = df[(df['Avg>2.5'].notna()) & (df['Avg<2.5'].notna())].copy()
    print(f"\n✅ Partidos con datos O/U 2.5: {len(df)}")
    
    df_historical = pd.read_csv(historical_path)
    df_historical['MatchDate'] = pd.to_datetime(df_historical['MatchDate'])
    
    if args.sample and len(df) > args.sample:
        df = df.tail(args.sample)
        print(f"📊 Modo MUESTRA: últimos {args.sample} partidos")
    else:
        print(f"📊 Modo COMPLETO: {len(df)} partidos")
    
    print(f"\n🤖 Cargando modelos...")
    predictor = EPLPredictor('models')
    
    # Ejecutar análisis
    if args.market in ['over', 'both']:
        df_over = analyze_over_under_performance(df, predictor, df_historical)
        if df_over is not None:
            output_path = Path(__file__).parent.parent / 'data' / 'processed' / 'backtest_over_under.csv'
            df_over.to_csv(output_path, index=False)
            print(f"\n💾 Resultados O/U guardados en: {output_path}")
    
    if args.market in ['btts', 'both']:
        df_btts = analyze_btts_performance(df, predictor, df_historical)
        if df_btts is not None:
            output_path = Path(__file__).parent.parent / 'data' / 'processed' / 'backtest_btts.csv'
            df_btts.to_csv(output_path, index=False)
            print(f"\n💾 Resultados BTTS guardados en: {output_path}")
    
    print("\n" + "="*70)
    print("✅ ANÁLISIS COMPLETADO")
    print("="*70)


if __name__ == '__main__':
    main()
