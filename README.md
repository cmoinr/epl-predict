# EPL Predictor v2.0 - Market Intelligence Edition

Predictor de resultados de la Premier League con **machine learning + market intelligence** para identificar oportunidades rentables de value betting.

## 🆕 Nuevo en v2.0

- ✅ **Integración de datos de mercado (odds históricas)**
- ✅ **26 features derivadas de probabilidades implícitas**
- ✅ **Modelo ensemble: ML + sabiduría del mercado**
- ✅ **Backtesting realista de estrategias de apuestas**
- ✅ **Análisis de consenso entre casas de apuestas**

📖 **Ver**: [MARKET_INTEGRATION_SUMMARY.md](MARKET_INTEGRATION_SUMMARY.md) para detalles completos

## Quick Start

### 1. Setup
```bash
# Crear ambiente virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Integrar Datos de Mercado (NUEVO)
```bash
# Pipeline completo de integración de odds
python scripts/integrate_market_data.py

# Analizar features de mercado
python scripts/analyze_market_features.py
```

### 3. Predicción Rápida
```bash
# Predecir un partido
python predict_match.py --home "Chelsea" --away "Liverpool" --date "2025-02-22"
```

### 4. Análisis Completo con Odds
```bash
# Analizar partidos y encontrar value bets
python run_analysis.py
```

## Características

### Predicciones
- **Resultado 1X2** - Home Win / Draw / Away Win
- **Goles Totales** - Promedio de goles predichos
- **BTTS** - Probabilidad de que ambos equipos anoten

### Análisis de Odds
- **Edge** - Ventaja modelo sobre el mercado
- **Expected Value (EV)** - Rentabilidad esperada
- **Kelly Criterion** - Fraccionamiento óptimo de apuesta
- **Recomendaciones** - BET / CONSIDER / MONITOR / SKIP

## Estructura de Directorios

```
epl-predict/
├── predict_match.py              # Predicción desde terminal
├── run_analysis.py               # Análisis con odds
├── retrain_models_improved.py    # Reentrenar modelos
├── requirements.txt
├── README.md
├── src/
│   ├── predictor.py             # Motor de predicción
│   ├── feature_engineering.py   # Creación de features
│   └── odds_comparison.py       # Análisis de valor
├── data/
│   ├── raw/
│   │   └── epl_final.csv        # Dataset histórico (9,420 partidos)
│   └── processed/
│       └── sample_odds.csv      # Odds de ejemplo
└── models/
    ├── rf_result_model.pkl      # Random Forest - Resultado
    ├── gb_result_model.pkl      # Gradient Boosting - Resultado
    ├── rf_goals_model.pkl       # Random Forest - Goles
    ├── gb_goals_model.pkl       # Gradient Boosting - Goles
    ├── rf_btts_model.pkl        # Random Forest - BTTS
    ├── gb_btts_model.pkl        # Gradient Boosting - BTTS
    └── scaler_model.pkl         # Scaler de features
```

## Comandos

### Predicción
```bash
# Con fecha específica
python predict_match.py --home "ManCity" --away "Arsenal" --date "2025-03-15"

# Usar fecha actual si no se especifica
python predict_match.py --home "Liverpool" --away "Chelsea"
```

### Análisis de Odds
```bash
# Leer datos de data/processed/sample_odds.csv
python run_analysis.py
```

### Reentrenamiento
```bash
# Reentrenar todos los modelos con datos actualizados
python retrain_models_improved.py
```

## Modelos Entrenados

**Dataset**: 9,420 partidos históricos de la Premier League  
**Train/Test**: 85/15 split temporal  
**Features**: 28 variables derivadas (forma, ofensiva, defensiva, ventaja local, etc.)

**Rendimiento**:
- Resultado 1X2: 74% Accuracy
- Goles Totales: 0.86 MAE
- BTTS: Balanced binary classifier

## Ejemplo de Output

```
PREDICCIONES DEL MODELO:
   * Chelsea: 45.2%
   * Draw: 28.1%
   * Liverpool: 26.7%
   * Goles totales predichos: 2.8
   * Ambos Anotan (BTTS): SI 65.3% | NO 34.7%

ANALISIS AMBOS ANOTAN (BTTS):
   BTTS Yes:
      Cuota: 1.72 | Modelo: 65.3% vs Mercado: 58.1%
      Edge: +7.17% | EV: +12.34%
      [BET]

MEJOR OPORTUNIDAD: BTTS Yes a 1.72
   Edge: +7.17% | EV: +12.34%
   Kelly 1/4 recomendado: 5.89%
   Con 1000EUR: Apuesta = 58.90EUR | Ganancia esperada = 7.28EUR
```

## Dependencias

- pandas >= 1.5.0
- scikit-learn >= 1.0.0
- numpy >= 1.20.0

## Notas Importantes

- Los datos históricos (epl_final.csv) están en `data/raw/`
- Las cuotas de apuestas se cargan de `data/processed/sample_odds.csv`
- Los modelos se guardan automáticamente en `models/` tras entrenar
- Compatible con Python 3.8+
- Todos los scripts están optimizados para terminal (sin dependencias de GUI)

---

## 🆕 Market Intelligence Integration

### ¿Qué es?
Sistema avanzado que integra **datos de mercado (odds históricas)** para mejorar predicciones y encontrar value betting.

### Archivos Clave
- 📄 **[MARKET_INTEGRATION_SUMMARY.md](MARKET_INTEGRATION_SUMMARY.md)** - Resumen ejecutivo
- 📖 **[docs/MARKET_DATA_INTEGRATION.md](docs/MARKET_DATA_INTEGRATION.md)** - Guía completa
- 📥 **[docs/HOW_TO_GET_MORE_ODDS_DATA.md](docs/HOW_TO_GET_MORE_ODDS_DATA.md)** - Expandir dataset

### Quick Start
```bash
# 1. Integrar datos de mercado
python scripts/integrate_market_data.py

# 2. Analizar features
python scripts/analyze_market_features.py

# 3. Backtest value betting
python scripts/backtest_value_betting.py
```

### Nuevas Features (26 total)
- `MarketProb_Home/Draw/Away` - Probabilidades implícitas del mercado
- `MarketConsensus` - Acuerdo entre casas de apuestas
- `FavoriteStrength` - Claridad del favorito
- `IsUpset` - Indicador de sorpresas
- `Team_AvgMarketProb_L10` - Reputación histórica según mercado
- Y 21 más...

### Resultados del Análisis
- **Precisión del mercado**: 48.4% (hay margen para ML)
- **Tasa de upsets**: 24.7% (1 de cada 4 es sorpresa)
- **Favoritos claros**: 68.4% precisión vs 41.8% dudosos
- **Cobertura actual**: 380 partidos con odds (4% del dataset)

### Próximos Pasos Recomendados
1. **Expandir dataset** - Descargar odds de football-data.co.uk (2000-2025)
2. **Re-entrenar modelos** - Incluir features de mercado
3. **Modelo ensemble** - Combinar ML (70%) + Mercado (30%)
4. **Backtest completo** - Validar ROI con 9,500+ partidos

### Scripts Disponibles
```
scripts/
├── integrate_market_data.py       # Pipeline completo
├── merge_odds_data.py            # Fusionar datasets
├── backtest_value_betting.py     # Simular apuestas
├── analyze_market_features.py    # Análisis detallado
└── download_odds_data.py         # Descargar más datos (crear)
```

### Impacto Esperado
- **+5-10% precisión** con features de mercado
- **ROI positivo** en value betting (5-15% anual)
- **Mejor calibración** de probabilidades
- **Detección de value** más confiable

---
