# Integración de Odds - Guía Completa

## 📋 Resumen Ejecutivo

Este sistema te permite:
1. **Cargar odds** de múltiples fuentes (CSV, APIs)
2. **Comparar predicciones ML** con las odds del mercado
3. **Identificar oportunidades de value betting** con edge positivo
4. **Calcular métricas de riesgo** (Kelly Criterion, ROI esperado)
5. **Filtrar y analizar** las mejores oportunidades

---

## 🎯 Conceptos Clave

### Cuota (Odds)
- **Decimal**: 2.50 (formato europeo usado en este proyecto)
- **Fraccionaria**: 3/2 (formato británico)
- **Americana**: -150 o +250 (formato USA)

### Probabilidad Implícita
```
Probabilidad = 1 / cuota_decimal
Ejemplo: 1 / 2.50 = 0.40 (40%)
```

### Edge (Ventaja)
```
Edge = Probabilidad_modelo - Probabilidad_implícita
Ejemplo: 0.45 - 0.40 = 0.05 (5%)
```

Un edge positivo significa que el modelo predice una probabilidad MAYOR que la implícita en la cuota.

### Valor Esperado (EV)
```
EV = (Probabilidad_modelo × cuota) - 1
Ejemplo: (0.45 × 2.50) - 1 = 0.125 (12.5%)
```

Esto significa que en promedio, por cada unidad apostada, esperas ganar 0.125 unidades.

### Kelly Criterion
```
f* = (b×p - q) / b
donde:
- b = cuota - 1
- p = probabilidad
- q = 1 - p
```

Determina el % óptimo del bankroll a apostar. Normalmente se usa **1/4 Kelly** (conservative).

### Margen de Casa (Overround)
```
Margen = (1/H + 1/D + 1/A) - 1
```

La suma de probabilidades implícitas siempre es > 1. La diferencia es el margen de la casa.

---

## 🚀 Módulos Creados

### 1. `src/odds_manager.py`
**Gestor de odds** - Cargar, procesar y validar datos de odds

```python
from src.odds_manager import OddsManager

# Inicializar
manager = OddsManager()

# Cargar odds históricos desde CSV
df_odds = manager.load_historical_odds('data/processed/sample_odds.csv')

# O conectar con API (requiere API key)
df_odds = manager.fetch_odds_api(
    api_key='YOUR_API_KEY',
    sport='soccer_epl',
    region='uk'
)

# Obtener mejores cuotas para un partido
best_odds = manager.get_best_odds(
    match_date='2024-11-16',
    home_team='Manchester City',
    away_team='Arsenal'
)

# Calcular probabilidad implícita
prob = manager.calculate_implied_probability(2.50)  # → 0.40

# Calcular cuota justa desde probabilidad
odds = manager.calculate_fair_odds(0.45)  # → 2.22

# Calcular margen de casa
margin = manager.calculate_bookmaker_margin(1.80, 3.50, 4.20)  # → 0.035 (3.5%)

# Obtener cuotas sin margen (sharp odds)
sharp = manager.calculate_sharp_odds(1.80, 3.50, 4.20)
```

**Características:**
- ✅ Carga de CSV con validación
- ✅ Integración con APIs (odds-api.com, football-data.org)
- ✅ Cálculos de probabilidades implícitas
- ✅ Análisis de márgenes
- ✅ Conversión de formatos (decimal, fraccionario, americano)

---

### 2. `src/odds_comparison.py`
**Comparador** - Analizar predicciones vs odds del mercado

```python
from src.odds_comparison import OddsComparison

# Inicializar con criterios de value betting
comparator = OddsComparison(
    min_edge=0.03,        # 3% edge mínimo
    min_ev=0.10,          # 10% EV mínimo
    min_confidence=0.55   # 55% confianza mínima
)

# Comparar una predicción con odds
opportunities = comparator.compare_prediction_with_odds(
    match_id='man_city_vs_arsenal',
    date='2024-11-16',
    home_team='Manchester City',
    away_team='Arsenal',
    prediction={...},  # Dict de predictor.py
    odds={...},        # Dict con cuotas
    model_type='ensemble'
)

# Encontrar todas las oportunidades
df_opportunities = comparator.find_value_bets(
    predictions_list=predictions,
    odds_list=odds_data,
    confidence_threshold=0.55,
    edge_threshold=0.03
)

# Filtrar por criterios
df_filtered = comparator.filter_opportunities(
    df_opportunities,
    recommendation='BET',
    min_odds=1.5,
    max_odds=10.0
)

# Calcular Kelly Criterion
kelly = comparator.calculate_kelly_criterion(
    model_prob=0.45,
    market_odds=2.50
)  # → 0.05 (5% del bankroll)

# Calcular Kelly 1/4 (conservative)
kelly_quarter = comparator.calculate_kelly_fraction(kelly, 0.25)
# → 0.0125 (1.25% del bankroll)

# Mostrar resumen
comparator.print_summary(df_opportunities, top_n=10)

# Exportar a CSV
comparator.export_to_csv(df_opportunities, 'results/value_bets.csv')
```

**Output de `compare_prediction_with_odds`:**
```python
ValueBettingOpportunity(
    match_id='man_city_vs_arsenal',
    date='2024-11-16',
    home_team='Manchester City',
    away_team='Arsenal',
    market='Home Win',
    model_probability=0.55,
    market_odds=1.65,
    implied_probability=0.606,
    edge=-0.056,  # Negativo (sin edge)
    value_percentage=-5.6,
    expected_value=-0.094,
    confidence_score=0.72,
    model_type='ensemble',
    recommendation='SKIP'  # No hay edge
)
```

**Recomendaciones:**
- `'BET'`: Edge significativo + confianza alta → **APOSTAR**
- `'CONSIDER'`: Edge positivo + confianza media → **CONSIDERAR**
- `'MONITOR'`: Edge pequeño → **MONITOREAR**
- `'SKIP'`: Sin edge → **SALTAR**

---

## 📊 Dataset de Odds

### Formato CSV Requerido
```csv
date,home_team,away_team,home_win_odds,draw_odds,away_win_odds,result,home_goals,away_goals
2024-11-09,Manchester City,Chelsea,1.50,4.50,6.50,1,2,0
2024-11-09,Liverpool,Arsenal,2.10,3.50,3.20,X,2,2
```

**Columnas:**
- `date`: Fecha (YYYY-MM-DD)
- `home_team`, `away_team`: Nombres de equipos
- `home_win_odds`, `draw_odds`, `away_win_odds`: Cuotas decimales
- `result`: Resultado (1=Home, X=Draw, 2=Away)
- `home_goals`, `away_goals`: Goles marcados

**Ubicación:** `data/processed/sample_odds.csv`

---

## 🔧 Script de Análisis

### `analyze_odds.py` - Análisis Completo

```bash
# Análisis básico (con criterios por defecto)
python analyze_odds.py

# Con umbrales personalizados
python analyze_odds.py \
  --min-edge 0.05 \
  --min-ev 0.15 \
  --min-confidence 0.60

# Filtrar solo por "BET"
python analyze_odds.py --recommendation BET

# Mostrar top 20 oportunidades
python analyze_odds.py --top 20

# Exportar a CSV
python analyze_odds.py --output results/value_bets.csv

# Modo verbose (con traceback de errores)
python analyze_odds.py --verbose
```

### Output del Script

```
════════════════════════════════════════════════════════════════════════════════════════════════════
💰 VALUE BETTING ANALYSIS - 2024-11-16 15:30:45
════════════════════════════════════════════════════════════════════════════════════════════════════

📊 Cargando datos históricos...
   ✅ 3800 partidos cargados

📊 Cargando odds del mercado...
   ✅ 30 partidos con odds cargados

🤖 Cargando modelos...
✅ Modelos cargados desde: models

🔮 Generando predicciones...
  ✅ Manchester City vs Chelsea
  ✅ Liverpool vs Arsenal
  ...

⚽ Preparando datos de odds...
   ✅ 10 partidos con odds

💡 Analizando oportunidades de value betting...

════════════════════════════════════════════════════════════════════════════════════════════════════
💰 VALUE BETTING OPPORTUNITIES - Top 10
════════════════════════════════════════════════════════════════════════════════════════════════════

Total de oportunidades: 15
  BET: 5
  CONSIDER: 7
  MONITOR: 3

════════════════════════════════════════════════════════════════════════════════════════════════════
📊 Manchester City vs Arsenal (2024-11-16)
   Mercado: Home Win @ 1.65
   Probabilidad Modelo: 62.0%
   Probabilidad Mercado: 60.6%
   Edge: 1.40%
   EV: 1.40%
   Confianza: 68.5%
   Recomendación: CONSIDER
   Kelly Criterion: 0.87% | 1/4 Kelly: 0.22%

...

📈 Estadísticas de Oportunidades:
  Total: 15
  Edge promedio: 2.34%
  EV promedio: 3.12%
  Confianza promedio: 64.2%

📊 SIMULACIÓN DE ROI (con Kelly 1/4)
────────────────────────────────────────────────────────────────────────────────────────────────────

Apostando en 15 oportunidades (Kelly 1/4):
  Apuesta total: 3.45 unidades
  Retorno esperado: 0.68 unidades
  ROI esperado: 19.7%
  Apuestas ganadoras (EV+): 12
  Apuestas perdedoras (EV-): 3
```

---

## 💡 Ejemplos de Uso Práctico

### Caso 1: Comparar Predicción Puntual

```python
from src.predictor import EPLPredictor
from src.odds_manager import OddsManager
from src.odds_comparison import OddsComparison
import pandas as pd

# 1. Cargar predictor y datos
predictor = EPLPredictor('models')
df_historical = pd.read_csv('data/raw/epl_final.csv')

# 2. Hacer predicción
prediction = predictor.predict_match(
    df_historical,
    'Manchester City',
    'Liverpool',
    '2024-11-23'
)

# 3. Cargar odds
manager = OddsManager()
df_odds = manager.load_historical_odds('data/processed/sample_odds.csv')

# 4. Obtener odds para el partido
match_odds = manager.get_best_odds(
    '2024-11-23',
    'Manchester City',
    'Liverpool'
)

# 5. Comparar
comparator = OddsComparison(min_edge=0.03, min_ev=0.10)

opportunities = comparator.compare_prediction_with_odds(
    match_id='mc_vs_liverpool_23112024',
    date='2024-11-23',
    home_team='Manchester City',
    away_team='Liverpool',
    prediction=prediction,
    odds=match_odds
)

# 6. Analizar
for opp in opportunities:
    print(f"Mercado: {opp.market}")
    print(f"  Cuota: {opp.market_odds:.2f}")
    print(f"  Prob Modelo: {opp.model_probability:.1%}")
    print(f"  Prob Mercado: {opp.implied_probability:.1%}")
    print(f"  Edge: {opp.value_percentage:.2f}%")
    print(f"  EV: {opp.expected_value:.2%}")
    print(f"  Recomendación: {opp.recommendation}")
    
    if opp.recommendation == 'BET':
        kelly = comparator.calculate_kelly_criterion(
            opp.model_probability,
            opp.market_odds
        )
        print(f"  Kelly: {kelly:.2%} | 1/4 Kelly: {comparator.calculate_kelly_fraction(kelly, 0.25):.2%}")
```

### Caso 2: Batch Analysis

```python
# Analizar múltiples partidos a la vez
matches = [
    {'home': 'Manchester City', 'away': 'Chelsea', 'date': '2024-11-09'},
    {'home': 'Liverpool', 'away': 'Arsenal', 'date': '2024-11-09'},
    {'home': 'Manchester United', 'away': 'Tottenham', 'date': '2024-11-09'},
]

predictions = predictor.predict_batch(df_historical, matches)

# Cargar odds correspondientes
odds_manager = OddsManager()
df_odds = odds_manager.load_historical_odds('data/processed/sample_odds.csv')

odds_list = []
for match in matches:
    match_odds = df_odds[
        (df_odds['home_team'] == match['home']) &
        (df_odds['away_team'] == match['away'])
    ]
    if len(match_odds) > 0:
        row = match_odds.iloc[0]
        odds_list.append({
            'home_win_odds': row['home_win_odds'],
            'draw_odds': row['draw_odds'],
            'away_win_odds': row['away_win_odds']
        })

# Encontrar value bets
df_opps = comparator.find_value_bets(predictions, odds_list)
comparator.print_summary(df_opps, top_n=5)
```

### Caso 3: Tracker de Confianza

```python
# Analizar cómo se distribuye la confianza

df_opps = comparator.find_value_bets(predictions, odds_list)

# Distribuir por confianza
confidence_bins = pd.cut(df_opps['confidence_score'], bins=5)
print(df_opps.groupby(confidence_bins).size())

# % de aciertos por confianza (si tenemos datos históricos)
high_confidence = df_opps[df_opps['confidence_score'] >= 0.70]
print(f"Oportunidades alto-confianza: {len(high_confidence)}")
print(f"Edge promedio: {high_confidence['edge'].mean():.2%}")
```

---

## 🔌 Integración con APIs

### Registrarse en odds-api.com

1. Ir a https://odds-api.com
2. Crear cuenta gratuita
3. Obtener API key (500 requests/día free tier)
4. Usar en tu código:

```python
from src.odds_manager import OddsManager

manager = OddsManager()

df_odds = manager.fetch_odds_api(
    api_key='YOUR_API_KEY_HERE',
    sport='soccer_epl',
    region='uk',
    markets='h2h'
)

# Guardar para análisis posterior
manager.save_odds_snapshot('data/processed/odds_snapshot.csv')
```

### Registrarse en football-data.org

1. Ir a https://www.football-data.org/client/register
2. Obtener token gratuito
3. Documentación: https://www.football-data.org/docs/v4/

---

## 📈 Indicadores Clave

### Cálculos Importantes

| Métrica | Fórmula | Interpretación |
|---------|---------|-----------------|
| **Probabilidad Implícita** | 1 / cuota | Probabilidad que asigna el mercado |
| **Edge** | P_modelo - P_mercado | Ventaja vs mercado (>0 es bueno) |
| **EV** | (P × cuota) - 1 | Retorno esperado por unidad |
| **ROI Esperado** | (EV) × 100% | Rendimiento de inversión |
| **Kelly %** | (bp - q) / b | % óptimo del bankroll a apostar |
| **Margen Casa** | (1/H + 1/D + 1/A) - 1 | Comisión de la casa |

### Umbrales Recomendados

| Criterio | Conservador | Balanceado | Agresivo |
|----------|-------------|-----------|----------|
| **Min Edge** | 5% | 3% | 1% |
| **Min EV** | 15% | 10% | 5% |
| **Min Confianza** | 70% | 60% | 50% |
| **Kelly Fraction** | 1/4 | 1/3 | 1/2 |
| **Sample Size** | 50+ | 30+ | 10+ |

---

## 📁 Estructura de Archivos

```
premier-league-ml/
├── src/
│   ├── odds_manager.py          # ✨ Gestor de odds
│   ├── odds_comparison.py       # ✨ Comparador ML vs mercado
│   ├── predictor.py             # Predictor existente
│   └── ...
│
├── data/
│   ├── processed/
│   │   └── sample_odds.csv      # ✨ Dataset de ejemplo
│   ├── raw/
│   │   └── epl_final.csv
│   └── ...
│
├── analyze_odds.py              # ✨ Script de análisis
├── predict_match.py             # Predictor existente
└── ...
```

---

## ⚠️ Consideraciones Importantes

### Sobre Apuestas

- **Responsabilidad**: Este es un sistema educativo/analítico. Las apuestas reales siempre tienen riesgo.
- **Trackers**: Mantén registros detallados para validar el modelo en producción
- **Variance**: Incluso con EV positivo, hay fluctuaciones normales
- **Rake/Fees**: Las casas de apuestas toman comisión (ya incluida en las cuotas)

### Sobre los Modelos

- **Out-of-sample**: Las predicciones deben validarse contra datos que el modelo no vio
- **Market Movement**: Las cuotas pueden cambiar significativamente (sharp movement)
- **Factors Externos**: Lesiones, cambios tácticos, noticias no capturadas

### Data Quality

- Asegúrate de que las odds sean contemporáneas (no histórico)
- Valida que los nombres de equipos sean consistentes
- Verifica que las cuotas tengan formatos correctos (decimal > 1)

---

## 🎓 Próximos Pasos

1. **[Integración en Tiempo Real]**: Conectar con APIs para actualizar odds continuamente
2. **[Backtesting]**: Validar estrategia en datos históricos
3. **[Dashboard]**: Crear visualizaciones de oportunidades
4. **[Alerts]**: Sistema de notificaciones para oportunidades de alto value
5. **[A/B Testing]**: Comparar diferentes thresholds y estrategias

---

## 📞 Soporte

- **Errores de módulos**: Revisa `src/odds_manager.py` y `src/odds_comparison.py`
- **Problemas de datos**: Valida el formato CSV contra `sample_odds.csv`
- **APIs**: Revisa credenciales y límites de rate
- **Predicciones**: Asegúrate de que los modelos estén entrenados

