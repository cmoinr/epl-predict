"""
Script para analizar predicciones del modelo vs odds del mercado
para partidos futuros
"""

import pandas as pd
import numpy as np
from pathlib import Path
from src.odds_comparison import OddsComparison
import json


def load_predictions_sample():
    """
    Crea predicciones de ejemplo para los partidos futuros
    (En producción, vendrían del modelo entrenado)
    """
    predictions = [
        {
            'match_id': 'MC_ARS_20241207',
            'date': '2024-12-07',
            'home_team': 'Manchester City',
            'away_team': 'Arsenal',
            'resultado': {
                'random_forest': {
                    'probabilidades': {
                        'Home Win': 55,
                        'Draw': 20,
                        'Away Win': 25
                    }
                },
                'gradient_boosting': {
                    'probabilidades': {
                        'Home Win': 57,
                        'Draw': 18,
                        'Away Win': 25
                    }
                }
            }
        },
        {
            'match_id': 'CHE_LIV_20241207',
            'date': '2024-12-07',
            'home_team': 'Chelsea',
            'away_team': 'Liverpool',
            'resultado': {
                'random_forest': {
                    'probabilidades': {
                        'Home Win': 28,
                        'Draw': 30,
                        'Away Win': 42
                    }
                },
                'gradient_boosting': {
                    'probabilidades': {
                        'Home Win': 30,
                        'Draw': 28,
                        'Away Win': 42
                    }
                }
            }
        },
        {
            'match_id': 'MAN_BRI_20241207',
            'date': '2024-12-07',
            'home_team': 'Manchester United',
            'away_team': 'Brighton',
            'resultado': {
                'random_forest': {
                    'probabilidades': {
                        'Home Win': 48,
                        'Draw': 26,
                        'Away Win': 26
                    }
                },
                'gradient_boosting': {
                    'probabilidades': {
                        'Home Win': 50,
                        'Draw': 25,
                        'Away Win': 25
                    }
                }
            }
        },
        {
            'match_id': 'TOT_BOU_20241208',
            'date': '2024-12-08',
            'home_team': 'Tottenham',
            'away_team': 'Bournemouth',
            'resultado': {
                'random_forest': {
                    'probabilidades': {
                        'Home Win': 62,
                        'Draw': 18,
                        'Away Win': 20
                    }
                },
                'gradient_boosting': {
                    'probabilidades': {
                        'Home Win': 64,
                        'Draw': 16,
                        'Away Win': 20
                    }
                }
            }
        },
        {
            'match_id': 'AV_WH_20241208',
            'date': '2024-12-08',
            'home_team': 'Aston Villa',
            'away_team': 'West Ham',
            'resultado': {
                'random_forest': {
                    'probabilidades': {
                        'Home Win': 51,
                        'Draw': 25,
                        'Away Win': 24
                    }
                },
                'gradient_boosting': {
                    'probabilidades': {
                        'Home Win': 53,
                        'Draw': 23,
                        'Away Win': 24
                    }
                }
            }
        }
    ]
    return predictions


def load_odds(odds_file='data/processed/sample_odds.csv'):
    """
    Carga las odds de los partidos futuros
    """
    odds_df = pd.read_csv(odds_file)
    
    odds_list = []
    for _, row in odds_df.iterrows():
        odds_list.append({
            'date': row['date'],
            'home_team': row['home_team'],
            'away_team': row['away_team'],
            'home_win_odds': row['home_win_odds'],
            'draw_odds': row['draw_odds'],
            'away_win_odds': row['away_win_odds'],
            'over_2_5_odds': row['over_2_5_odds'],
            'under_2_5_odds': row['under_2_5_odds'],
        })
    
    return odds_list


def analyze_predictions_vs_odds(
    predictions=None,
    odds_file='data/processed/sample_odds.csv',
    min_edge=0.03,
    min_ev=0.10,
    min_confidence=0.55
):
    """
    Analiza predicciones del modelo vs odds del mercado
    
    Args:
        predictions: Lista de predicciones (None = usar ejemplo)
        odds_file: Ruta al archivo de odds
        min_edge: Edge mínimo requerido
        min_ev: Expected value mínimo
        min_confidence: Confianza mínima del modelo
        
    Returns:
        Dict con análisis completo
    """
    
    if predictions is None:
        predictions = load_predictions_sample()
    
    odds_list = load_odds(odds_file)
    
    # Crear comparador
    comparator = OddsComparison(
        min_edge=min_edge,
        min_ev=min_ev,
        min_confidence=min_confidence
    )
    
    # Encontrar oportunidades
    opportunities_df = comparator.find_value_bets(
        predictions_list=predictions,
        odds_list=odds_list
    )
    
    return {
        'comparator': comparator,
        'opportunities': opportunities_df,
        'predictions': predictions,
        'odds': odds_list
    }


def print_detailed_analysis(analysis_result):
    """
    Imprime análisis detallado de las oportunidades
    """
    df = analysis_result['opportunities']
    comparator = analysis_result['comparator']
    
    if len(df) == 0:
        print("\n❌ No hay oportunidades de value betting encontradas")
        return
    
    print("\n" + "="*120)
    print("📊 ANÁLISIS: PREDICCIONES DEL MODELO vs ODDS DEL MERCADO")
    print("="*120)
    
    # Resumen general
    print(f"\n📈 RESUMEN GENERAL:")
    print(f"  • Total de partidos analizados: {len(analysis_result['odds'])}")
    print(f"  • Oportunidades encontradas: {len(df)}")
    print(f"  • Recomendaciones BET: {len(df[df['recommendation'] == 'BET'])}")
    print(f"  • Recomendaciones CONSIDER: {len(df[df['recommendation'] == 'CONSIDER'])}")
    print(f"  • Recomendaciones MONITOR: {len(df[df['recommendation'] == 'MONITOR'])}")
    print(f"  • EV promedio: {df['expected_value'].mean():.2%}")
    
    # Análisis por partido
    print(f"\n{'='*120}")
    print("🎯 OPORTUNIDADES ORDENADAS POR EXPECTED VALUE")
    print(f"{'='*120}\n")
    
    df_sorted = df.sort_values('expected_value', ascending=False)
    
    for idx, row in df_sorted.iterrows():
        kelly = comparator.calculate_kelly_criterion(
            row['model_probability'],
            row['market_odds']
        )
        kelly_quarter = comparator.calculate_kelly_fraction(kelly, 0.25)
        
        # Colores según recomendación
        rec_emoji = "🟢" if row['recommendation'] == 'BET' else \
                    "🟡" if row['recommendation'] == 'CONSIDER' else "🔵"
        
        print(f"{rec_emoji} {row['home_team']} vs {row['away_team']}")
        print(f"   📅 Fecha: {row['date']}")
        print(f"   🎲 Resultado predicho: {row['market'].upper()}")
        print(f"   ")
        print(f"   📊 Análisis del Modelo:")
        print(f"      • Probabilidad predicha: {row['model_probability']:.1%}")
        print(f"      • Confianza del modelo: {row['confidence_score']:.1%}")
        print(f"   ")
        print(f"   💰 Análisis del Mercado:")
        print(f"      • Cuota: {row['market_odds']:.2f}")
        print(f"      • Probabilidad implícita: {row['implied_probability']:.1%}")
        print(f"   ")
        print(f"   ✨ Oportunidad de Valor:")
        print(f"      • Edge: {row['value_percentage']:.2f}%")
        print(f"      • Expected Value (EV): {row['expected_value']:.2%}")
        print(f"      • Kelly Criterion: {kelly:.2%}")
        print(f"      • Kelly 1/4 (conservative): {kelly_quarter:.2%}")
        print(f"   ")
        print(f"   🎯 Recomendación: {row['recommendation']}")
        print("-" * 120)


def export_analysis(analysis_result, output_dir='data/processed'):
    """
    Exporta análisis a archivos
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    df = analysis_result['opportunities']
    
    if len(df) > 0:
        # Exportar CSV
        csv_file = output_path / 'value_betting_opportunities.csv'
        df.to_csv(csv_file, index=False)
        print(f"\n✅ Oportunidades exportadas a: {csv_file}")
        
        # Exportar JSON con detalles
        json_file = output_path / 'value_betting_analysis.json'
        analysis_data = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'total_matches': len(analysis_result['odds']),
            'opportunities_found': len(df),
            'by_recommendation': {
                'BET': len(df[df['recommendation'] == 'BET']),
                'CONSIDER': len(df[df['recommendation'] == 'CONSIDER']),
                'MONITOR': len(df[df['recommendation'] == 'MONITOR']),
                'SKIP': len(df[df['recommendation'] == 'SKIP']),
            },
            'statistics': {
                'avg_edge': float(df['value_percentage'].mean()),
                'avg_ev': float(df['expected_value'].mean()),
                'avg_confidence': float(df['confidence_score'].mean()),
                'max_ev': float(df['expected_value'].max()),
                'min_ev': float(df['expected_value'].min()),
            },
            'opportunities': df.to_dict('records')
        }
        
        with open(json_file, 'w') as f:
            json.dump(analysis_data, f, indent=2)
        print(f"✅ Análisis JSON exportado a: {json_file}")
        
        return True
    else:
        print("\n⚠️  No hay oportunidades para exportar")
        return False


def print_market_consensus(analysis_result):
    """
    Imprime consenso del mercado
    """
    odds_list = analysis_result['odds']
    
    print(f"\n{'='*120}")
    print("🎯 CONSENSO DEL MERCADO (Expectativas del Mercado)")
    print(f"{'='*120}\n")
    
    for odds in odds_list[:10]:  # Mostrar primeros 10
        implied_probs = {
            'Home Win': 1 / odds['home_win_odds'],
            'Draw': 1 / odds['draw_odds'],
            'Away Win': 1 / odds['away_win_odds'],
        }
        
        most_likely = max(implied_probs, key=implied_probs.get)
        
        print(f"📌 {odds['home_team']} vs {odds['away_team']} ({odds['date']})")
        print(f"   Home Win: {implied_probs['Home Win']:.1%} (cuota: {odds['home_win_odds']:.2f})")
        print(f"   Draw:     {implied_probs['Draw']:.1%} (cuota: {odds['draw_odds']:.2f})")
        print(f"   Away Win: {implied_probs['Away Win']:.1%} (cuota: {odds['away_win_odds']:.2f})")
        print(f"   🏆 Favorito: {most_likely}")
        print(f"   ")


def main():
    """
    Ejecuta análisis completo
    """
    print("\n🚀 Iniciando análisis de predicciones vs odds del mercado...")
    
    # Realizar análisis
    analysis = analyze_predictions_vs_odds(
        min_edge=0.03,
        min_ev=0.10,
        min_confidence=0.50
    )
    
    # Mostrar consenso del mercado
    print_market_consensus(analysis)
    
    # Mostrar análisis detallado
    print_detailed_analysis(analysis)
    
    # Exportar resultados
    export_analysis(analysis)
    
    print("\n✅ Análisis completado")


if __name__ == '__main__':
    main()
