# 📊 Integración de Datos de Mercado - Guía Completa

## 🎯 Objetivo

Enriquecer el proyecto de predicción EPL con **datos históricos de cuotas (odds)** para:
1. Mejorar la precisión de los modelos ML
2. Implementar estrategias de **value betting**
3. Calibrar predicciones con la "sabiduría del mercado"
4. Realizar backtesting realista de rentabilidad

---

## 📂 Estructura de Datos

### Datasets Originales
- **`epl_final.csv`**: 9,510 partidos (2000/01-2025/26) sin odds
- **`epl_odds.csv`**: 380 partidos (temporada 2000/01) CON odds de 5 casas

### Datasets Generados
- **`epl_enriched_with_odds.csv`**: Fusión de ambos datasets
- **`epl_with_market_intelligence.csv`**: Dataset completo con features de mercado
- **`backtest_results.csv`**: Resultados de simulación de apuestas

---

## 🚀 Ejecución Rápida

### Opción 1: Pipeline Completo (Recomendado)
```bash
cd scripts
python integrate_market_data.py
```

Este script ejecuta automáticamente:
1. Merge de datasets
2. Extracción de features de mercado
3. Análisis exploratorio
4. Backtesting de muestra

### Opción 2: Paso a Paso

#### 1. Fusionar datasets
```bash
python scripts/merge_odds_data.py
```

**Salida**: `data/processed/epl_enriched_with_odds.csv`

**Features agregadas**:
- `AvgOdds_Home/Draw/Away`: Promedio de cuotas
- `MarketProb_Home/Draw/Away`: Probabilidades implícitas
- `AdjustedProb_*`: Probabilidades sin margen de casas
- `Overround`: Margen total del mercado
- `MarketConsensus`: Consenso entre casas
- `FavoriteStrength`: Claridad del favorito

#### 2. Generar features avanzadas
```bash
python src/market_features.py
```

**Salida**: `data/processed/epl_with_market_intelligence.csv`

**Nuevas features**:
- `MarketSurprise_Home`: Desviación del resultado esperado
- `IsUnderdog_Home/Away`: Indicador de underdog
- `MarketAccuracy`: Precisión de la predicción del mercado
- `IsUpset`: Indicador de sorpresa (underdog ganó)
- `IsCompetitiveMatch`: Partidos parejos
- `Team_AvgMarketProb_L10`: Percepción histórica del equipo
- `Team_UpsetRate_L10`: Frecuencia de sorpresas

#### 3. Backtesting de value betting
```bash
python scripts/backtest_value_betting.py
```

**Salida**: Reporte de rendimiento + historial CSV

---

## 🔬 Uso de Features de Mercado

### En Entrenamiento de Modelos

```python
from src.market_features import MarketBasedFeatures
import pandas as pd

# Cargar dataset enriquecido
df = pd.read_csv('data/processed/epl_with_market_intelligence.csv')

# Features recomendadas para incluir en el modelo
market_features = [
    'MarketProb_Home',
    'MarketProb_Draw', 
    'MarketProb_Away',
    'MarketConsensus',
    'FavoriteStrength',
    'ImpliedGoalDiff',
    'Team_AvgMarketProb_L10',
    'IsCompetitiveMatch'
]

# Combinar con features tradicionales
X = df[traditional_features + market_features]
y = df['FullTimeResult']

# Entrenar modelo
model.fit(X, y)
```

### En Predicción con Ensemble

```python
from src.predictor import EPLPredictor
from src.market_features import MarketBasedFeatures

predictor = EPLPredictor()

# Predicción del modelo ML
ml_result = predictor.predict_match(
    home_team='Chelsea',
    away_team='Arsenal',
    match_date='2025-12-27'
)

# Odds actuales del mercado (de sample_odds.csv)
market_odds = {
    'home': 1.85,
    'draw': 3.90,
    'away': 4.00
}

# Probabilidades del mercado
market_prob_home = 1 / market_odds['home']
market_prob_draw = 1 / market_odds['draw']
market_prob_away = 1 / market_odds['away']

# Normalizar (quitar overround)
total = market_prob_home + market_prob_draw + market_prob_away
market_prob_home /= total
market_prob_draw /= total
market_prob_away /= total

# Ensemble: 70% modelo, 30% mercado
ensemble_prob_home = 0.7 * ml_result['probabilities']['home_win'] + 0.3 * market_prob_home
ensemble_prob_draw = 0.7 * ml_result['probabilities']['draw'] + 0.3 * market_prob_draw
ensemble_prob_away = 0.7 * ml_result['probabilities']['away_win'] + 0.3 * market_prob_away

print(f"Ensemble Home Win: {ensemble_prob_home*100:.1f}%")
```

### En Value Betting

```python
from src.odds_comparison import OddsComparison

# Configurar comparador
odds_comp = OddsComparison(
    min_edge=0.05,  # 5% edge mínimo
    min_ev=0.10     # 10% valor esperado mínimo
)

# Probabilidades de tu modelo
model_probs = {
    'home_win': 0.60,
    'draw': 0.25,
    'away_win': 0.15
}

# Odds del mercado
market_odds_dict = {
    'home_odds': 1.85,
    'draw_odds': 3.90,
    'away_odds': 4.00
}

# Buscar value
opportunities = odds_comp.find_value_bets(model_probs, market_odds_dict)

for opp in opportunities:
    print(f"{opp.bet_type}: Edge={opp.edge*100:.1f}%, EV={opp.expected_value*100:.1f}%")
```

---

## 📊 Interpretación de Features de Mercado

### Probabilidades Implícitas
```
MarketProb = 1 / Odds
```
- Cuota 2.00 → 50% de probabilidad
- Cuota 4.00 → 25% de probabilidad
- **Mayor cuota = Menor probabilidad según mercado**

### Overround (Margen de Casas)
```
Overround = P(Home) + P(Draw) + P(Away)
```
- Ideal: 1.00 (100%)
- Real: ~1.05-1.10 (105-110%) ← margen de ganancia de casas
- **Ajusta dividiendo cada probabilidad por el overround**

### Edge (Ventaja)
```
Edge = Prob_Modelo - Prob_Mercado
```
- Edge > 0: Tu modelo es más optimista que el mercado (value potencial)
- Edge < 0: Tu modelo es más pesimista
- **Edge >= 5% típicamente indica value betting**

### Consenso del Mercado
```
Consenso = 1 / (1 + Std_Dev_Odds)
```
- Alto consenso: Todas las casas de acuerdo (información clara)
- Bajo consenso: Casas discrepan (información incierta/partidos difíciles)

---

## 🎯 Estrategias Recomendadas

### 1. Modelo Híbrido (ML + Market)
```python
# Pesos adaptativos según consenso del mercado
if MarketConsensus > 0.9:  # Alto consenso
    weight_model = 0.3
    weight_market = 0.7  # Confiar más en el mercado
else:  # Bajo consenso (más oportunidad)
    weight_model = 0.8
    weight_market = 0.2  # Confiar más en tu modelo

final_prob = weight_model * model_prob + weight_market * market_prob
```

### 2. Value Betting Conservador
```python
# Solo apostar cuando:
# 1. Edge >= 8%
# 2. Probabilidad modelo >= 25% (evitar improbables)
# 3. Consenso mercado < 0.85 (evitar "trampas")

if edge >= 0.08 and model_prob >= 0.25 and market_consensus < 0.85:
    stake = kelly_criterion(model_prob, odds, bankroll) * 0.25  # Quarter Kelly
```

### 3. Especialización en Upsets
```python
# Buscar partidos donde:
# - Underdog tiene >= 30% según tu modelo
# - Mercado le da < 20%
# - Diferencia >= 10% (edge)

if (IsUnderdog_Home and 
    ModelProb_Home >= 0.30 and 
    MarketProb_Home < 0.20):
    # Value potencial en underdog
```

---

## 📈 Métricas de Éxito

### Para Modelos ML
- **Precisión**: ¿Mejora con features de mercado?
- **Calibración**: ¿Probabilidades bien calibradas vs mercado?
- **Feature Importance**: ¿Qué peso tienen features de mercado?

### Para Value Betting
- **ROI**: Retorno sobre inversión (objetivo: > 5%)
- **Win Rate**: Tasa de aciertos (objetivo: > 55%)
- **Edge Promedio**: Ventaja media por apuesta (objetivo: > 7%)
- **Drawdown**: Pérdida máxima (objetivo: < 20% del bankroll)

---

## ⚠️ Advertencias Importantes

1. **Los odds históricos son de UNA sola temporada (2000/01)**
   - Solo ~380 partidos con odds
   - El resto del dataset (9,000+) NO tiene odds
   - Necesitas más datos de odds para entrenar modelos robustos

2. **El mercado de apuestas ha evolucionado**
   - Odds de 2000 ≠ Odds de 2025
   - Mercados más eficientes ahora
   - Features de mercado modernas pueden ser más predictivas

3. **Cuidado con el overfitting**
   - No uses `FullTimeResult` + `MarketProb` juntos en entrenamiento
   - El mercado ya "sabe" información que tú no tendrías antes del partido

4. **Necesitas más datos de odds**
   - Busca datasets con odds de múltiples temporadas
   - football-data.co.uk tiene datos completos desde 2000
   - APIs: The Odds API, Betfair Exchange

---

## 🔄 Actualización Continua

### Obtener más datos de odds históricos
```bash
# football-data.co.uk tiene datasets completos
# Ejemplo para temporada 2023/24:
wget https://www.football-data.co.uk/mmz4281/2324/E0.csv -O data/raw/epl_2023_24_odds.csv
```

### Automatizar obtención de odds actuales
```python
# Usar API de odds (ejemplo: The Odds API)
import requests

API_KEY = 'tu_api_key'
response = requests.get(
    f'https://api.the-odds-api.com/v4/sports/soccer_epl/odds',
    params={'apiKey': API_KEY, 'regions': 'uk'}
)

# Actualizar sample_odds.csv automáticamente
```

---

## 📚 Recursos Adicionales

- **football-data.co.uk**: Datos históricos completos con odds
- **The Odds API**: Odds en tiempo real
- **Betfair Exchange**: Odds de intercambio (más precisos)
- **Papers académicos**: "Wisdom of the Crowd in Soccer Betting"

---

## 🤝 Contribución

Si encuentras más datasets de odds o mejoras en features, ¡compártelos!

---

**Autor**: EPL Prediction Project  
**Última actualización**: Diciembre 2025
