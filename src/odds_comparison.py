"""
Módulo de Comparación - Analizar predicciones ML vs Odds del Mercado
Value Betting: Encontrar oportunidades con edge positivo
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import json


@dataclass
class ValueBettingOpportunity:
    """
    Representa una oportunidad de value betting encontrada
    
    Atributos:
    ----------
    match_id : str
        Identificador del partido
    date : str
        Fecha del partido
    home_team : str
        Equipo local
    away_team : str
        Equipo visitante
    market : str
        Mercado (home_win, draw, away_win)
    model_probability : float
        Probabilidad predicha por el modelo (0-1)
    market_odds : float
        Cuota del mercado (decimal)
    implied_probability : float
        Probabilidad implícita de la cuota
    edge : float
        Edge (diferencia entre probabilidades): model - implied
    value_percentage : float
        Edge en porcentaje
    expected_value : float
        Valor esperado: (model_prob * odds) - 1
    confidence_score : float
        Confianza en el modelo (0-1)
    model_type : str
        Tipo de modelo usado (random_forest, gradient_boosting)
    recommendation : str
        'BET', 'SKIP', 'HEDGE'
    """
    
    match_id: str
    date: str
    home_team: str
    away_team: str
    market: str
    model_probability: float
    market_odds: float
    implied_probability: float
    edge: float
    value_percentage: float
    expected_value: float
    confidence_score: float
    model_type: str
    recommendation: str
    
    def to_dict(self) -> Dict:
        """Convertir a diccionario"""
        return asdict(self)


class OddsComparison:
    """
    Comparador de predicciones ML vs odds del mercado
    
    Identifica oportunidades de value betting donde el modelo tiene edge positivo
    
    Ejemplo:
    --------
    comparator = OddsComparison(min_edge=0.03)
    opportunities = comparator.find_value_bets(
        predictions=predictions_df,
        odds=odds_df,
        confidence_threshold=0.55
    )
    """
    
    def __init__(self, min_edge: float = 0.03, min_ev: float = 0.10, 
                 min_confidence: float = 0.55):
        """
        Inicializar comparador
        
        Parámetros:
        -----------
        min_edge : float
            Edge mínimo requerido (default: 3%)
        min_ev : float
            Valor esperado mínimo (default: 10%)
        min_confidence : float
            Confianza mínima del modelo (0-1)
        """
        self.min_edge = min_edge
        self.min_ev = min_ev
        self.min_confidence = min_confidence
        self.opportunities = []
    
    def calculate_edge(self, model_prob: float, market_odds: float) -> float:
        """
        Calcular edge (ventaja) del modelo sobre el mercado
        
        Edge = model_prob - (1 / market_odds)
        
        Parámetros:
        -----------
        model_prob : float
            Probabilidad del modelo (0-1)
        market_odds : float
            Cuota decimal del mercado
        
        Retorna:
        --------
        float : Edge en valor absoluto (0-1)
        """
        if market_odds <= 0:
            return 0
        implied_prob = 1 / market_odds
        return model_prob - implied_prob
    
    def calculate_expected_value(self, model_prob: float, market_odds: float) -> float:
        """
        Calcular valor esperado de una apuesta unitaria
        
        EV = (model_prob * market_odds) - 1
        
        Con apuesta unitaria de 1:
        - Si EV > 0: valor esperado positivo
        - Si EV = 0: valor justo
        - Si EV < 0: valor esperado negativo
        
        Parámetros:
        -----------
        model_prob : float
            Probabilidad del modelo (0-1)
        market_odds : float
            Cuota decimal del mercado
        
        Retorna:
        --------
        float : Valor esperado (-1 a infinito)
        """
        if market_odds <= 0:
            return 0
        return (model_prob * market_odds) - 1
    
    def assess_confidence(self, prediction: Dict) -> float:
        """
        Evaluar confianza en la predicción del modelo
        
        Combina múltiples factores:
        - Acuerdo entre modelos (RF vs GB)
        - Certeza de probabilidades
        - Margen sobre segundo lugar
        
        Parámetros:
        -----------
        prediction : dict
            Predicción con probabilidades
        
        Retorna:
        --------
        float : Confianza (0-1)
        """
        try:
            # Obtener probabilidades según estructura de predicción
            if 'resultado' in prediction:
                # Formato de predictor.py
                rf_probs = prediction['resultado']['random_forest']['probabilidades']
                gb_probs = prediction['resultado']['gradient_boosting']['probabilidades']
                
                # Convertir a listas ordenadas: [away, draw, home]
                outcomes = ['Away Win', 'Draw', 'Home Win']
                rf_vals = [rf_probs.get(o, 0) / 100 for o in outcomes]
                gb_vals = [gb_probs.get(o, 0) / 100 for o in outcomes]
            else:
                # Formato alternativo
                rf_vals = prediction.get('rf_probs', [])
                gb_vals = prediction.get('gb_probs', [])
            
            if not rf_vals or not gb_vals:
                return 0.5  # Confianza neutral
            
            # 1. Acuerdo entre modelos (correlación)
            agreement = 1 - np.mean(np.abs(np.array(rf_vals) - np.array(gb_vals)))
            
            # 2. Certeza (máxima probabilidad)
            max_rf = max(rf_vals) if rf_vals else 0.33
            max_gb = max(gb_vals) if gb_vals else 0.33
            certainty = (max_rf + max_gb) / 2
            
            # 3. Margen sobre segundo lugar
            sorted_rf = sorted(rf_vals, reverse=True)
            sorted_gb = sorted(gb_vals, reverse=True)
            
            margin_rf = sorted_rf[0] - sorted_rf[1] if len(sorted_rf) > 1 else 0
            margin_gb = sorted_gb[0] - sorted_gb[1] if len(sorted_gb) > 1 else 0
            margin = (margin_rf + margin_gb) / 2
            
            # Combinar: 40% acuerdo, 40% certeza, 20% margen
            confidence = (0.4 * agreement) + (0.4 * certainty) + (0.2 * margin)
            
            return np.clip(confidence, 0, 1)
        
        except Exception as e:
            print(f"⚠️  Error calculando confianza: {e}")
            return 0.5
    
    def extract_market_outcome(self, market: str) -> str:
        """
        Extraer resultado del mercado a partir del nombre del mercado
        
        Parámetros:
        -----------
        market : str
            Nombre del mercado (ej: 'home_win', 'home', 'H')
        
        Retorna:
        --------
        str : Outcome normalizado ('Home Win', 'Draw', 'Away Win')
        """
        market_lower = market.lower()
        
        if any(x in market_lower for x in ['home', '1', 'h', 'local']):
            return 'Home Win'
        elif any(x in market_lower for x in ['draw', 'x', 'd', 'empate']):
            return 'Draw'
        elif any(x in market_lower for x in ['away', '2', 'a', 'visitante']):
            return 'Away Win'
        
        return market
    
    def compare_prediction_with_odds(self, 
                                     match_id: str,
                                     date: str,
                                     home_team: str,
                                     away_team: str,
                                     prediction: Dict,
                                     odds: Dict,
                                     model_type: str = 'ensemble') -> List[ValueBettingOpportunity]:
        """
        Comparar una predicción con las odds del mercado
        
        Parámetros:
        -----------
        match_id : str
            ID único del partido
        date : str
            Fecha del partido
        home_team : str
            Equipo local
        away_team : str
            Equipo visitante
        prediction : dict
            Predicción del modelo con probabilidades
        odds : dict
            Odds del mercado (home_win_odds, draw_odds, away_win_odds)
        model_type : str
            Tipo de modelo ('random_forest', 'gradient_boosting', 'ensemble')
        
        Retorna:
        --------
        List[ValueBettingOpportunity] : Oportunidades encontradas
        """
        opportunities = []
        
        try:
            # Extraer probabilidades del modelo
            if 'resultado' in prediction:
                rf_probs = prediction['resultado']['random_forest']['probabilidades']
                gb_probs = prediction['resultado']['gradient_boosting']['probabilidades']
                
                # Usar promedio de ambos modelos
                model_probs = {
                    'Away Win': (rf_probs.get('Away Win', 0) + gb_probs.get('Away Win', 0)) / 200,
                    'Draw': (rf_probs.get('Draw', 0) + gb_probs.get('Draw', 0)) / 200,
                    'Home Win': (rf_probs.get('Home Win', 0) + gb_probs.get('Home Win', 0)) / 200,
                }
            else:
                model_probs = prediction.get('probabilities', {})
            
            # Calcular confianza
            confidence = self.assess_confidence(prediction)
            
            # Comparar cada mercado
            markets = {
                'home_win': ('Home Win', odds.get('home_win_odds', 0)),
                'draw': ('Draw', odds.get('draw_odds', 0)),
                'away_win': ('Away Win', odds.get('away_win_odds', 0))
            }
            
            for market_key, (outcome, market_odds) in markets.items():
                if market_odds <= 0:
                    continue
                
                model_prob = model_probs.get(outcome, 0)
                if model_prob <= 0 or model_prob >= 1:
                    continue
                
                # Calcular métricas
                edge = self.calculate_edge(model_prob, market_odds)
                ev = self.calculate_expected_value(model_prob, market_odds)
                implied_prob = 1 / market_odds
                
                # Determinar recomendación
                if confidence >= self.min_confidence and edge >= self.min_edge and ev >= self.min_ev:
                    recommendation = 'BET'
                elif edge >= self.min_edge and ev >= self.min_ev * 0.5:
                    recommendation = 'CONSIDER'
                elif edge >= 0:
                    recommendation = 'MONITOR'
                else:
                    recommendation = 'SKIP'
                
                # Crear oportunidad
                opp = ValueBettingOpportunity(
                    match_id=match_id,
                    date=date,
                    home_team=home_team,
                    away_team=away_team,
                    market=outcome,
                    model_probability=model_prob,
                    market_odds=market_odds,
                    implied_probability=implied_prob,
                    edge=edge,
                    value_percentage=edge * 100,
                    expected_value=ev,
                    confidence_score=confidence,
                    model_type=model_type,
                    recommendation=recommendation
                )
                
                opportunities.append(opp)
        
        except Exception as e:
            print(f"❌ Error en comparación: {e}")
        
        return opportunities
    
    def find_value_bets(self, predictions_list: List[Dict], 
                       odds_list: List[Dict],
                       confidence_threshold: float = None,
                       edge_threshold: float = None) -> pd.DataFrame:
        """
        Encontrar todas las oportunidades de value betting
        
        Parámetros:
        -----------
        predictions_list : List[Dict]
            Lista de predicciones del predictor
        odds_list : List[Dict]
            Lista de diccionarios con odds
        confidence_threshold : float, optional
            Override del umbral de confianza
        edge_threshold : float, optional
            Override del umbral de edge
        
        Retorna:
        --------
        pd.DataFrame : Oportunidades con todas las métricas
        """
        if confidence_threshold is not None:
            self.min_confidence = confidence_threshold
        if edge_threshold is not None:
            self.min_edge = edge_threshold
        
        all_opportunities = []
        
        for i, prediction in enumerate(predictions_list):
            odds = odds_list[i] if i < len(odds_list) else {}
            
            match_id = odds.get('match_id', f"match_{i}")
            date = odds.get('date', prediction.get('fecha', 'unknown'))
            home_team = odds.get('home_team', prediction.get('home', 'Unknown'))
            away_team = odds.get('away_team', prediction.get('away', 'Unknown'))
            
            opps = self.compare_prediction_with_odds(
                match_id=match_id,
                date=date,
                home_team=home_team,
                away_team=away_team,
                prediction=prediction,
                odds=odds
            )
            
            all_opportunities.extend(opps)
        
        # Convertir a DataFrame
        if all_opportunities:
            df = pd.DataFrame([opp.to_dict() for opp in all_opportunities])
            self.opportunities = all_opportunities
            return df
        else:
            return pd.DataFrame()
    
    def filter_opportunities(self, df: pd.DataFrame, 
                            recommendation: str = 'BET',
                            min_odds: float = 1.5,
                            max_odds: float = 10.0) -> pd.DataFrame:
        """
        Filtrar oportunidades por criterios
        
        Parámetros:
        -----------
        df : pd.DataFrame
            DataFrame de oportunidades
        recommendation : str
            Filtrar por recomendación
        min_odds, max_odds : float
            Rango de cuotas aceptables
        
        Retorna:
        --------
        pd.DataFrame : Oportunidades filtradas
        """
        filtered = df.copy()
        
        if recommendation:
            filtered = filtered[filtered['recommendation'].isin(
                [recommendation, 'BET', 'CONSIDER'] if recommendation == 'BET' else [recommendation]
            )]
        
        filtered = filtered[
            (filtered['market_odds'] >= min_odds) &
            (filtered['market_odds'] <= max_odds)
        ]
        
        return filtered.sort_values('expected_value', ascending=False)
    
    def calculate_kelly_criterion(self, model_prob: float, market_odds: float) -> float:
        """
        Calcular fracción de Kelly (optimal bet size)
        
        Kelly Criterion: f* = (bp - q) / b
        donde:
        - b = odds - 1
        - p = probabilidad de ganar
        - q = 1 - p = probabilidad de perder
        
        Parámetros:
        -----------
        model_prob : float
            Probabilidad predicha (0-1)
        market_odds : float
            Cuota decimal
        
        Retorna:
        --------
        float : Fracción de bankroll a apostar (0-1)
        """
        if market_odds <= 1:
            return 0.0
        
        b = market_odds - 1
        p = model_prob
        q = 1 - model_prob
        
        kelly = (b * p - q) / b
        
        # Limitar a rango válido
        return max(0, min(kelly, 1))
    
    def calculate_kelly_fraction(self, kelly: float, fraction: float = 0.25) -> float:
        """
        Calcular fracción de Kelly (conservative betting)
        
        Es común usar 1/4 o 1/2 de Kelly para reducir volatilidad
        
        Parámetros:
        -----------
        kelly : float
            Kelly Criterion (fracción de bankroll)
        fraction : float
            Fracción a aplicar (default: 1/4 = 0.25)
        
        Retorna:
        --------
        float : Kelly ajustado
        """
        return kelly * fraction
    
    def print_summary(self, df: pd.DataFrame, top_n: int = 10):
        """
        Imprimir resumen de oportunidades
        
        Parámetros:
        -----------
        df : pd.DataFrame
            DataFrame de oportunidades
        top_n : int
            Mostrar top N oportunidades
        """
        if len(df) == 0:
            print("❌ No hay oportunidades de value betting encontradas")
            return
        
        print(f"\n{'='*100}")
        print(f"💰 VALUE BETTING OPPORTUNITIES - Top {min(top_n, len(df))}")
        print(f"{'='*100}\n")
        
        # Estadísticas generales
        print(f"Total de oportunidades: {len(df)}")
        print(f"  BET: {len(df[df['recommendation'] == 'BET'])}")
        print(f"  CONSIDER: {len(df[df['recommendation'] == 'CONSIDER'])}")
        print(f"  MONITOR: {len(df[df['recommendation'] == 'MONITOR'])}\n")
        
        # Top oportunidades
        top_df = df.nlargest(top_n, 'expected_value')
        
        for idx, row in top_df.iterrows():
            kelly = self.calculate_kelly_criterion(row['model_probability'], row['market_odds'])
            kelly_quarter = self.calculate_kelly_fraction(kelly, 0.25)
            
            print(f"{'='*100}")
            print(f"📊 {row['home_team']} vs {row['away_team']} ({row['date']})")
            print(f"   Mercado: {row['market']} @ {row['market_odds']:.2f}")
            print(f"   Probabilidad Modelo: {row['model_probability']:.1%}")
            print(f"   Probabilidad Mercado: {row['implied_probability']:.1%}")
            print(f"   Edge: {row['value_percentage']:.2f}%")
            print(f"   EV: {row['expected_value']:.2%}")
            print(f"   Confianza: {row['confidence_score']:.1%}")
            print(f"   Recomendación: {row['recommendation']}")
            print(f"   Kelly Criterion: {kelly:.2%} | 1/4 Kelly: {kelly_quarter:.2%}")
        
        print(f"\n{'='*100}\n")
    
    def export_to_csv(self, df: pd.DataFrame, filepath: str):
        """
        Exportar oportunidades a CSV
        
        Parámetros:
        -----------
        df : pd.DataFrame
            DataFrame de oportunidades
        filepath : str
            Ruta de salida
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            if '.' in filepath:
                name, ext = filepath.rsplit('.', 1)
                filepath = f"{name}_{timestamp}.{ext}"
            
            df.to_csv(filepath, index=False)
            print(f'✅ Oportunidades exportadas a: {filepath}')
            return True
        
        except Exception as e:
            print(f'❌ Error exportando: {e}')
            return False


if __name__ == '__main__':
    print("🔍 MÓDULO DE COMPARACIÓN DE PREDICCIONES vs ODDS\n")
    
    # Ejemplo de uso
    comparator = OddsComparison(min_edge=0.03, min_ev=0.10)
    
    # Ejemplo de cálculos
    print("Ejemplos de cálculos:")
    print("-" * 50)
    
    model_prob = 0.45
    market_odds = 2.50
    
    edge = comparator.calculate_edge(model_prob, market_odds)
    ev = comparator.calculate_expected_value(model_prob, market_odds)
    kelly = comparator.calculate_kelly_criterion(model_prob, market_odds)
    
    print(f"\nModelo predice: {model_prob:.1%}")
    print(f"Mercado ofrece: {market_odds:.2f} (implícito: {1/market_odds:.1%})")
    print(f"Edge: {edge:.2%}")
    print(f"EV (apuesta unitaria): {ev:.2%}")
    print(f"Kelly Criterion: {kelly:.2%}")
    print(f"Kelly 1/4 (conservative): {comparator.calculate_kelly_fraction(kelly, 0.25):.2%}")
