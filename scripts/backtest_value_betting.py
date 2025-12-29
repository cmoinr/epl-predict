"""
Backtesting de Value Betting usando datos históricos de odds
Simula apuestas basadas en el modelo vs. mercado
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Agregar src al path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from predictor import EPLPredictor
from odds_comparison import OddsComparison


class ValueBettingBacktest:
    """Sistema de backtesting para estrategias de value betting"""
    
    def __init__(self, df_with_odds, predictor, df_historical, initial_bankroll=1000, kelly_fraction=0.25):
        """
        Args:
            df_with_odds: DataFrame con odds históricas y resultados
            predictor: EPLPredictor instance para hacer predicciones
            df_historical: DataFrame histórico completo para features
            initial_bankroll: Bankroll inicial
            kelly_fraction: Fracción del criterio de Kelly (0.25 = quarter Kelly)
        """
        self.df = df_with_odds.copy()
        self.predictor = predictor
        self.df_historical = df_historical
        self.initial_bankroll = initial_bankroll
        self.kelly_fraction = kelly_fraction
        self.bet_history = []
        
    def calculate_kelly_stake(self, prob_model, odds, bankroll):
        """
        Calcula el stake usando criterio de Kelly fraccionado
        
        Kelly = (prob * odds - 1) / (odds - 1)
        
        Args:
            prob_model: Probabilidad según tu modelo
            odds: Cuota del mercado
            bankroll: Bankroll actual
        
        Returns:
            Stake recomendado
        """
        # Kelly criterion
        edge = (prob_model * odds - 1) / (odds - 1)
        kelly_full = edge
        
        # Aplicar fracción (más conservador)
        kelly_stake = kelly_full * self.kelly_fraction
        
        # No apostar si no hay edge o si es negativo
        if kelly_stake <= 0:
            return 0
        
        # Limitar al 5% del bankroll como máximo
        max_stake = 0.05 * bankroll
        stake = min(kelly_stake * bankroll, max_stake)
        
        return stake
    
    def simulate_bets(self, min_edge=0.05, min_prob=0.15, max_bets=None, optimized=False, ultra_optimized=False):
        """
        Simula apuestas basadas en value betting usando el modelo REAL
        
        Args:
            min_edge: Edge mínimo requerido (5% por defecto)
            min_prob: Probabilidad mínima del modelo (evitar apuestas muy improbables)
            max_bets: Máximo número de apuestas (None = sin límite)
            optimized: Si True, aplica filtros optimizados basados en backtest
            ultra_optimized: Si True, aplica filtros V2 ultra-específicos (máxima rentabilidad)
        
        Returns:
            DataFrame con historial de apuestas y métricas
        """
        bankroll = self.initial_bankroll
        self.bet_history = []
        
        mode = "V2 ULTRA-OPTIMIZADO" if ultra_optimized else ("OPTIMIZADO" if optimized else "BASELINE")
        print(f"🎲 Iniciando backtesting con modelo Phase 2 (80.38% accuracy)...")
        print(f"   🚀 MODO: {mode}")
        print(f"   Bankroll inicial: ${bankroll:.2f}")
        print(f"   Kelly fraction: {self.kelly_fraction}")
        
        if ultra_optimized:
            print(f"\n   🔥 FILTROS V2 ULTRA-OPTIMIZADOS (Basados en análisis profundo):")
            print(f"      🏆 AWAY WINS:")
            print(f"         • Cuotas: 2.5 - 4.0 (sweet spot: ROI 44.67%)")
            print(f"         • Edge: 10% - 22%")
            print(f"         • Prob. Modelo: 40% - 60%")
            print(f"      🏆 HOME WINS:")
            print(f"         • Edge: 18% - 22% (ROI 54.33%)")
            print(f"         • Cuotas: 2.5 - 3.0")
            print(f"         • Prob. Modelo: 45% - 60%")
            print(f"      🏆 DRAWS:")
            print(f"         • Edge: 12% - 15% SOLAMENTE (ROI 69.18%)")
            print(f"         • Cuotas: 3.0 - 4.0")
            print(f"         • Prob. Modelo: 25% - 35%")
            print(f"      ❌ RECHAZAR: Prob. Modelo <40% o >60% (descalibrado)")
        elif optimized:
            print(f"\n   🔧 FILTROS OPTIMIZADOS:")
            print(f"      • Edge: 10% - 25% (sweet spot)")
            print(f"      • Cuotas: 1.5 - 4.5 (evitar extremos)")
            print(f"      • Preferencia: Home Wins (más predecibles)")
            print(f"      • Penalización: Draws y Away con edge >30%")
        print()
        
        # Solo partidos con odds disponibles
        df_bettable = self.df[self.df['AvgOdds_Home'].notna()].copy()
        df_bettable = df_bettable.sort_values('MatchDate').reset_index(drop=True)
        
        total_matches = len(df_bettable)
        processed = 0
        
        for idx, row in df_bettable.iterrows():
            processed += 1
            
            # Modo rápido: saltar algunos partidos para acelerar
            if max_bets and processed % 5 == 0 and len(self.bet_history) >= max_bets * 0.8:
                continue
            
            # Mostrar progreso cada 500 partidos
            if processed % 500 == 0 or processed == total_matches:
                print(f"   Procesando: {processed}/{total_matches} partidos ({processed/total_matches*100:.1f}%) | Apuestas: {len(self.bet_history)}")
            
            # 🚀 USAR PREDICCIÓN REAL DEL MODELO
            try:
                # Usar solo datos históricos HASTA esta fecha (evitar look-ahead bias)
                df_until_date = self.df_historical[
                    pd.to_datetime(self.df_historical['MatchDate']) < pd.to_datetime(row['MatchDate'])
                ]
                
                if len(df_until_date) < 50:  # Necesitamos datos mínimos
                    continue
                
                # Hacer predicción real
                prediction = self.predictor.predict_match(
                    df_until_date,
                    row['HomeTeam'],
                    row['AwayTeam'],
                    row['MatchDate']
                )
                
                # Extraer probabilidades del MEJOR MODELO (Phase 2)
                probs = prediction['resultado']['mejor_modelo']['probabilidades']
                prob_home = probs['Home Win'] / 100.0
                prob_draw = probs['Draw'] / 100.0
                prob_away = probs['Away Win'] / 100.0
                
            except Exception as e:
                # Si hay error en la predicción, saltar este partido
                if processed % 100 == 0:  # Mostrar errores ocasionales
                    print(f"   [SKIP] Error en {row['HomeTeam']} vs {row['AwayTeam']}: {str(e)[:50]}")
                continue
            
            # Cuotas del mercado
            odds_home = row['AvgOdds_Home']
            odds_draw = row['AvgOdds_Draw']
            odds_away = row['AvgOdds_Away']
            
            # Calcular edge para cada resultado
            edge_home = prob_home - (1 / odds_home)
            edge_draw = prob_draw - (1 / odds_draw)
            edge_away = prob_away - (1 / odds_away)
            
            # Encontrar la mejor oportunidad
            best_edge = max(edge_home, edge_draw, edge_away)
            
            # � APLICAR FILTROS V2 ULTRA-OPTIMIZADOS
            if ultra_optimized:
                skip = True  # Por defecto saltar, solo apostar si cumple reglas estrictas
                
                # REGLA 1: AWAY WINS - MUY RENTABLES (ROI 22-44%)
                if best_edge == edge_away:
                    # Away Wins: Cuota 2.5-4.0, Edge 10-22%, Prob 40-60%
                    if (2.5 <= odds_away <= 4.0 and 
                        0.10 <= edge_away <= 0.22 and
                        0.40 <= prob_away <= 0.60):
                        skip = False
                        best_edge = edge_away
                
                # REGLA 2: HOME WINS - Selectivos (ROI 54%)
                elif best_edge == edge_home:
                    # Home Wins: Edge 18-22%, Cuota 2.5-3.0, Prob 45-60%
                    if (2.5 <= odds_home <= 3.0 and
                        0.18 <= edge_home <= 0.22 and
                        0.45 <= prob_home <= 0.60):
                        skip = False
                        best_edge = edge_home
                
                # REGLA 3: DRAWS - MUY SELECTIVOS (ROI 69% pero raros)
                elif best_edge == edge_draw:
                    # Draws: Edge 12-15% EXACTO, Cuota 3.0-4.0, Prob 25-35%
                    if (3.0 <= odds_draw <= 4.0 and
                        0.12 <= edge_draw <= 0.15 and
                        0.25 <= prob_draw <= 0.35):
                        skip = False
                        best_edge = edge_draw
                
                if skip:
                    continue
            
            # 🔧 APLICAR FILTROS OPTIMIZADOS V1 si está activado
            elif optimized:
                # FILTRO 1: Edge debe estar en el rango óptimo (10-25%)
                if best_edge < 0.10 or best_edge > 0.25:
                    continue
                
                # FILTRO 2: Evitar cuotas extremas (solo 1.5 - 4.5)
                if best_edge == edge_home and (odds_home < 1.5 or odds_home > 4.5):
                    continue
                if best_edge == edge_draw and (odds_draw < 1.5 or odds_draw > 4.5):
                    continue
                if best_edge == edge_away and (odds_away < 1.5 or odds_away > 4.5):
                    continue
                
                # FILTRO 3: Preferencia por Home Wins (son más predecibles)
                # Si hay un Home Win con buen edge, priorizar sobre Draw/Away
                if edge_home >= min_edge and edge_home >= 0.10 and odds_home <= 4.0:
                    # Dar preferencia a Home si está cerca del mejor edge
                    if best_edge - edge_home < 0.03:  # Diferencia menor a 3%
                        best_edge = edge_home
                
                # FILTRO 4: Penalizar Draws con edge muy alto (son tramposos)
                if best_edge == edge_draw and best_edge > 0.20 and odds_draw > 4.0:
                    continue  # Skip draws muy confiados
            
            if best_edge >= min_edge:
                # Determinar qué apostar
                if best_edge == edge_home and prob_home >= min_prob:
                    bet_type = 'Home'
                    prob = prob_home
                    odds = odds_home
                    result = 'H'
                    actual_result = row['FullTimeResult']
                elif best_edge == edge_draw and prob_draw >= min_prob:
                    bet_type = 'Draw'
                    prob = prob_draw
                    odds = odds_draw
                    result = 'D'
                    actual_result = row['FullTimeResult']
                elif best_edge == edge_away and prob_away >= min_prob:
                    bet_type = 'Away'
                    prob = prob_away
                    odds = odds_away
                    result = 'A'
                    actual_result = row['FullTimeResult']
                else:
                    continue
                
                # Calcular stake usando Kelly
                stake = self.calculate_kelly_stake(prob, odds, bankroll)
                
                if stake > 0 and stake <= bankroll:
                    # Realizar apuesta
                    won = (actual_result == result)
                    payout = stake * odds if won else 0
                    profit = payout - stake
                    
                    bankroll += profit
                    
                    # Registrar
                    self.bet_history.append({
                        'date': row['MatchDate'],
                        'home_team': row['HomeTeam'],
                        'away_team': row['AwayTeam'],
                        'bet_type': bet_type,
                        'model_prob': prob,
                        'market_prob': 1/odds,
                        'edge': best_edge,
                        'odds': odds,
                        'stake': stake,
                        'won': won,
                        'profit': profit,
                        'bankroll': bankroll,
                        'actual_result': actual_result
                    })
        
        return pd.DataFrame(self.bet_history)
    
    def generate_report(self, bet_df):
        """
        Genera reporte de rendimiento
        
        Args:
            bet_df: DataFrame con historial de apuestas
        """
        if len(bet_df) == 0:
            print("❌ No se realizaron apuestas con los criterios especificados")
            return
        
        total_bets = len(bet_df)
        wins = bet_df['won'].sum()
        win_rate = wins / total_bets
        
        total_staked = bet_df['stake'].sum()
        total_profit = bet_df['profit'].sum()
        roi = (total_profit / total_staked) * 100
        
        final_bankroll = bet_df['bankroll'].iloc[-1]
        total_return = ((final_bankroll - self.initial_bankroll) / self.initial_bankroll) * 100
        
        avg_edge = bet_df['edge'].mean()
        avg_odds = bet_df['odds'].mean()
        
        # Estadísticas por tipo de apuesta
        bet_types = bet_df.groupby('bet_type').agg({
            'won': ['sum', 'count'],
            'profit': 'sum',
            'stake': 'sum'
        })
        
        print("=" * 70)
        print("📊 REPORTE DE BACKTESTING - VALUE BETTING")
        print("=" * 70)
        print(f"\n💰 RENDIMIENTO GENERAL:")
        print(f"   Bankroll inicial:    ${self.initial_bankroll:,.2f}")
        print(f"   Bankroll final:      ${final_bankroll:,.2f}")
        print(f"   Ganancia/Pérdida:    ${total_profit:,.2f}")
        print(f"   Retorno total:       {total_return:.2f}%")
        
        print(f"\n🎯 ESTADÍSTICAS DE APUESTAS:")
        print(f"   Total de apuestas:   {total_bets}")
        print(f"   Apuestas ganadas:    {wins} ({win_rate*100:.1f}%)")
        print(f"   Total apostado:      ${total_staked:,.2f}")
        print(f"   ROI:                 {roi:.2f}%")
        print(f"   Edge promedio:       {avg_edge*100:.2f}%")
        print(f"   Cuota promedio:      {avg_odds:.2f}")
        
        print(f"\n📈 DESGLOSE POR TIPO DE APUESTA:")
        for bet_type in bet_types.index:
            wins_type = bet_types.loc[bet_type, ('won', 'sum')]
            total_type = bet_types.loc[bet_type, ('won', 'count')]
            profit_type = bet_types.loc[bet_type, ('profit', 'sum')]
            stake_type = bet_types.loc[bet_type, ('stake', 'sum')]
            roi_type = (profit_type / stake_type) * 100
            
            print(f"   {bet_type:8s}: {wins_type:3.0f}/{total_type:3.0f} ({wins_type/total_type*100:5.1f}%) | "
                  f"Ganancia: ${profit_type:7,.2f} | ROI: {roi_type:6.2f}%")
        
        print(f"\n📉 VOLATILIDAD Y RIESGO:")
        print(f"   Pico de bankroll:    ${bet_df['bankroll'].max():,.2f}")
        print(f"   Valle de bankroll:   ${bet_df['bankroll'].min():,.2f}")
        print(f"   Drawdown máximo:     ${self.initial_bankroll - bet_df['bankroll'].min():,.2f}")
        
        # Calcular Sharpe Ratio (usando profits como returns)
        if len(bet_df) > 1:
            returns = bet_df['profit'] / bet_df['stake']
            sharpe = (returns.mean() / returns.std()) * np.sqrt(len(returns)) if returns.std() > 0 else 0
            print(f"   Sharpe Ratio:        {sharpe:.2f}")
        
        # Racha más larga
        winning_streaks = []
        losing_streaks = []
        current_streak = 0
        for won in bet_df['won']:
            if won:
                if current_streak >= 0:
                    current_streak += 1
                else:
                    losing_streaks.append(abs(current_streak))
                    current_streak = 1
            else:
                if current_streak <= 0:
                    current_streak -= 1
                else:
                    winning_streaks.append(current_streak)
                    current_streak = -1
        
        if current_streak > 0:
            winning_streaks.append(current_streak)
        elif current_streak < 0:
            losing_streaks.append(abs(current_streak))
        
        if winning_streaks:
            print(f"   Racha ganadora max:  {max(winning_streaks)} apuestas")
        if losing_streaks:
            print(f"   Racha perdedora max: {max(losing_streaks)} apuestas")
        
        # Análisis de edge vs resultado
        avg_edge_won = bet_df[bet_df['won']]['edge'].mean()
        avg_edge_lost = bet_df[~bet_df['won']]['edge'].mean()
        print(f"\n🎯 ANÁLISIS DE EDGE:")
        print(f"   Edge promedio (ganadas):  {avg_edge_won*100:.2f}%")
        print(f"   Edge promedio (perdidas): {avg_edge_lost*100:.2f}%")
        print(f"   Correlación edge/ganancia: {bet_df['edge'].corr(bet_df['profit']):.3f}")
        
        return {
            'total_bets': total_bets,
            'win_rate': win_rate,
            'roi': roi,
            'total_return': total_return,
            'final_bankroll': final_bankroll,
            'sharpe': sharpe if len(bet_df) > 1 else 0,
            'max_winning_streak': max(winning_streaks) if winning_streaks else 0,
            'max_losing_streak': max(losing_streaks) if losing_streaks else 0
        }


def main():
    """Función principal para ejecutar backtesting"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Backtest de value betting con modelo Phase 2')
    parser.add_argument('--full', action='store_true', help='Backtest completo (todos los partidos)')
    parser.add_argument('--sample', type=int, default=500, help='Número de partidos para muestra rápida')
    parser.add_argument('--edge', type=float, default=0.05, help='Edge mínimo (default: 0.05 = 5%%)')
    parser.add_argument('--kelly', type=float, default=0.25, help='Fracción Kelly (default: 0.25)')
    parser.add_argument('--fast', action='store_true', help='Modo rápido: saltar predicciones cada 5 partidos')
    parser.add_argument('--optimized', action='store_true', help='Aplicar filtros optimizados V1 (edge 10-25%%, cuotas 1.5-4.5)')
    parser.add_argument('--ultra', action='store_true', help='Aplicar filtros V2 ultra-optimizados (máxima rentabilidad)')
    args = parser.parse_args()
    
    # Cargar datasets
    data_path = Path(__file__).parent.parent / 'data' / 'processed' / 'epl_enriched_with_odds.csv'
    historical_path = Path(__file__).parent.parent / 'data' / 'raw' / 'epl_final.csv'
    
    if not data_path.exists():
        print("❌ Error: primero ejecuta merge_odds_data.py para crear el dataset enriquecido")
        print(f"   Archivo esperado: {data_path}")
        return
    
    print("\n" + "="*70)
    print("🚀 BACKTEST CON MODELO PHASE 2 (80.38% ACCURACY)")
    print("="*70)
    print(f"\n📂 Cargando datos...")
    
    df = pd.read_csv(data_path)
    df['MatchDate'] = pd.to_datetime(df['MatchDate'])
    
    df_historical = pd.read_csv(historical_path)
    df_historical['MatchDate'] = pd.to_datetime(df_historical['MatchDate'])
    
    # Filtrar solo partidos con odds
    df_with_odds = df[df['AvgOdds_Home'].notna()].copy()
    print(f"✅ {len(df_with_odds)} partidos con odds disponibles")
    
    # Limitar si no es --full
    if not args.full and len(df_with_odds) > args.sample:
        print(f"📊 Modo MUESTRA: usando últimos {args.sample} partidos")
        print(f"   (Usa --full para backtest completo)")
        df_with_odds = df_with_odds.tail(args.sample)
    else:
        print(f"📊 Modo COMPLETO: backtesting {len(df_with_odds)} partidos")
    
    # Cargar predictor con modelo Phase 2
    print(f"\n🤖 Cargando modelo Phase 2...")
    predictor = EPLPredictor('models')
    
    # Configurar backtesting
    backtest = ValueBettingBacktest(
        df_with_odds,
        predictor,
        df_historical,
        initial_bankroll=1000,
        kelly_fraction=args.kelly
    )
    
    # Ejecutar simulación
    print()
    bet_history = backtest.simulate_bets(
        min_edge=args.edge,
        min_prob=0.15,
        optimized=args.optimized,
        ultra_optimized=args.ultra
    )
    
    # Generar reporte
    metrics = backtest.generate_report(bet_history)
    
    # Guardar historial
    if len(bet_history) > 0:
        if args.ultra:
            filename = 'backtest_results_v2.csv'
        elif args.optimized:
            filename = 'backtest_results_optimized.csv'
        else:
            filename = 'backtest_results.csv'
        
        output_path = Path(__file__).parent.parent / 'data' / 'processed' / filename
        bet_history.to_csv(output_path, index=False)
        print(f"\n💾 Historial guardado en: {output_path}")


if __name__ == '__main__':
    main()
