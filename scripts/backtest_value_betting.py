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
    
    def __init__(self, df_with_odds, initial_bankroll=1000, kelly_fraction=0.25):
        """
        Args:
            df_with_odds: DataFrame con odds históricas y resultados
            initial_bankroll: Bankroll inicial
            kelly_fraction: Fracción del criterio de Kelly (0.25 = quarter Kelly)
        """
        self.df = df_with_odds.copy()
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
    
    def simulate_bets(self, min_edge=0.05, min_prob=0.15):
        """
        Simula apuestas basadas en value betting
        
        Args:
            min_edge: Edge mínimo requerido (5% por defecto)
            min_prob: Probabilidad mínima del modelo (evitar apuestas muy improbables)
        
        Returns:
            DataFrame con historial de apuestas y métricas
        """
        bankroll = self.initial_bankroll
        self.bet_history = []
        
        print(f"🎲 Iniciando simulación de value betting...")
        print(f"   Bankroll inicial: ${bankroll:.2f}")
        print(f"   Edge mínimo: {min_edge*100}%")
        print(f"   Kelly fraction: {self.kelly_fraction}")
        print(f"   Probabilidad mínima: {min_prob*100}%\n")
        
        # Solo partidos con odds disponibles
        df_bettable = self.df[self.df['AvgOdds_Home'].notna()].copy()
        
        for idx, row in df_bettable.iterrows():
            # Probabilidades del modelo (simularemos con las probabilidades ajustadas del mercado + ruido)
            # En tu caso real, usarías las predicciones de EPLPredictor
            
            # Simulación: añadir ruido para simular predicción del modelo
            noise = np.random.normal(0, 0.05, 3)
            prob_home = max(0.05, min(0.95, row['AdjustedProb_Home'] + noise[0]))
            prob_draw = max(0.05, min(0.95, row['AdjustedProb_Draw'] + noise[1]))
            prob_away = max(0.05, min(0.95, row['AdjustedProb_Away'] + noise[2]))
            
            # Normalizar
            total = prob_home + prob_draw + prob_away
            prob_home /= total
            prob_draw /= total
            prob_away /= total
            
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
        
        print(f"\n📉 VOLATILIDAD:")
        print(f"   Pico de bankroll:    ${bet_df['bankroll'].max():,.2f}")
        print(f"   Valle de bankroll:   ${bet_df['bankroll'].min():,.2f}")
        print(f"   Drawdown máximo:     ${self.initial_bankroll - bet_df['bankroll'].min():,.2f}")
        
        return {
            'total_bets': total_bets,
            'win_rate': win_rate,
            'roi': roi,
            'total_return': total_return,
            'final_bankroll': final_bankroll
        }


def main():
    """Función principal para ejecutar backtesting"""
    
    # Cargar dataset enriquecido
    data_path = Path(__file__).parent.parent / 'data' / 'processed' / 'epl_enriched_with_odds.csv'
    
    if not data_path.exists():
        print("❌ Error: primero ejecuta merge_odds_data.py para crear el dataset enriquecido")
        print(f"   Archivo esperado: {data_path}")
        return
    
    print("📂 Cargando datos...")
    df = pd.read_csv(data_path)
    df['MatchDate'] = pd.to_datetime(df['MatchDate'])
    
    # Filtrar solo partidos con odds
    df_with_odds = df[df['AvgOdds_Home'].notna()].copy()
    print(f"✅ {len(df_with_odds)} partidos con odds disponibles\n")
    
    # Configurar backtesting
    backtest = ValueBettingBacktest(
        df_with_odds,
        initial_bankroll=1000,
        kelly_fraction=0.25
    )
    
    # Ejecutar simulación
    bet_history = backtest.simulate_bets(
        min_edge=0.05,  # 5% edge mínimo
        min_prob=0.15   # 15% probabilidad mínima
    )
    
    # Generar reporte
    metrics = backtest.generate_report(bet_history)
    
    # Guardar historial
    if len(bet_history) > 0:
        output_path = Path(__file__).parent.parent / 'data' / 'processed' / 'backtest_results.csv'
        bet_history.to_csv(output_path, index=False)
        print(f"\n💾 Historial guardado en: {output_path}")


if __name__ == '__main__':
    main()
