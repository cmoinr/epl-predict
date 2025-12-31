# 📚 Guía de Aprendizaje: Proyecto EPL-Predict

## ¿Qué es este proyecto?

**EPL-Predict** es un sistema de **Machine Learning (ML)** diseñado para predecir resultados de partidos de la Premier League inglesa. El proyecto combina:

- 🧠 **Algoritmos de Machine Learning** para predicción
- 📊 **Feature Engineering** (ingeniería de características)
- 💰 **Value Betting** (apuestas de valor) basado en odds del mercado
- 📈 **Análisis de datos históricos** desde 2003

---

## 🏗️ Arquitectura del Proyecto

```
epl-predict/
│
├── 📁 src/                    ← CORAZÓN DEL PROYECTO
│   ├── feature_engineering.py ← Creación de features (características)
│   ├── market_features.py     ← Features basadas en odds del mercado
│   ├── odds_comparison.py     ← Comparación ML vs mercado (value betting)
│   └── predictor.py           ← Motor de predicción (usa los modelos)
│
├── 📁 models/                 ← Modelos entrenados (.pkl)
├── 📁 data/                   ← Datos históricos y procesados
├── 📁 scripts/                ← Scripts de utilidad y análisis
│
├── retrain_models_improved.py ← Script de entrenamiento
├── predict_match.py           ← Predicir un partido nuevo
└── get_value_bets.py          ← Encontrar apuestas de valor
```

---

## 📖 Índice de Documentación de Aprendizaje

| # | Tema | Descripción |
|---|------|-------------|
| 01 | [Introducción](01_INTRODUCCION_PROYECTO.md) | Este documento - visión general |
| 02 | [Fundamentos de ML](02_FUNDAMENTOS_ML.md) | Conceptos básicos de Machine Learning |
| 03 | [Librerías Python para ML](03_LIBRERIAS_ML_PYTHON.md) | NumPy, Pandas, Scikit-learn, XGBoost, etc. |
| 04 | [Feature Engineering](04_FEATURE_ENGINEERING.md) | Cómo crear variables predictivas |
| 05 | [Modelos de Clasificación](05_MODELOS_CLASIFICACION.md) | Random Forest, Gradient Boosting, etc. |
| 06 | [Ensemble Learning](06_ENSEMBLE_LEARNING.md) | Combinando múltiples modelos |
| 07 | [Value Betting y Odds](07_VALUE_BETTING_ODDS.md) | Matemáticas de las apuestas |
| 08 | [Pipeline Completo](08_PIPELINE_COMPLETO.md) | Cómo funciona el flujo end-to-end |

---

## 🎯 Flujo de Trabajo del Sistema

```
┌─────────────────────────────────────────────────────────────────┐
│                    FASE 1: PREPARACIÓN                          │
├─────────────────────────────────────────────────────────────────┤
│  📁 Datos Históricos  →  🔧 Feature Engineering  →  📊 Dataset  │
│     (CSV partidos)       (crear variables)          (X, y)      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    FASE 2: ENTRENAMIENTO                        │
├─────────────────────────────────────────────────────────────────┤
│  📊 Dataset  →  🧠 Algoritmos ML  →  📦 Modelos (.pkl)          │
│   (train/test)    (RF, GB, XGB...)    (guardados)              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    FASE 3: PREDICCIÓN                           │
├─────────────────────────────────────────────────────────────────┤
│  🆕 Nuevo Partido  →  📦 Modelos  →  🔮 Predicción + Prob.      │
│  (Arsenal vs Chelsea)   (cargados)    (Home Win 58%)            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    FASE 4: VALUE BETTING                        │
├─────────────────────────────────────────────────────────────────┤
│  🔮 Predicción ML  →  💰 Odds Mercado  →  📈 Value Bet?         │
│     (Home 58%)          (Home @2.00)       (Edge +8%)           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔑 Conceptos Clave

### ¿Qué es Machine Learning?
Es un campo de la inteligencia artificial donde los algoritmos **aprenden patrones de datos históricos** para hacer predicciones en datos nuevos. En vez de programar reglas manualmente ("si el equipo local ganó 5 partidos seguidos, probablemente gana"), el algoritmo descubre estas reglas automáticamente.

### ¿Qué son las Features?
Son las **variables de entrada** que el modelo usa para aprender. Por ejemplo:
- Goles promedio del equipo local
- Forma reciente (puntos en últimos 5 partidos)
- Historial head-to-head (enfrentamientos directos)
- Ventaja de casa

### ¿Qué es el Target?
Es lo que queremos **predecir**. En este proyecto:
- **Resultado del partido**: Home Win, Draw, Away Win
- **Total de goles**: 0, 1, 2, 3...
- **Ambos anotan (BTTS)**: Sí, No

### ¿Qué es Value Betting?
Es apostar cuando el modelo cree que la probabilidad real es **mayor** que la probabilidad implícita en las odds del mercado. Si nuestro modelo dice 55% y las odds implican 45%, tenemos un "edge" (ventaja) del 10%.

---

## 🛠️ Tecnologías Utilizadas

| Categoría | Tecnologías |
|-----------|-------------|
| **Lenguaje** | Python 3.x |
| **Data Science** | NumPy, Pandas |
| **Machine Learning** | Scikit-learn, XGBoost, LightGBM, CatBoost |
| **Visualización** | Matplotlib, Seaborn |
| **Serialización** | Pickle (guardar modelos) |

---

## 📈 Métricas de los Modelos Actuales

El proyecto ha logrado estas precisiones:

| Modelo | Resultado (1X2) | BTTS | Goles (MAE) |
|--------|-----------------|------|-------------|
| Random Forest | 52.80% | 77.72% | 0.85 |
| Gradient Boosting | 55.33% | 78.02% | 0.84 |
| XGBoost | 55.28% | 78.37% | 0.84 |
| LightGBM | 55.49% | 77.88% | 0.84 |
| **Phase 2 (Market)** | **80.38%** | - | - |

> 💡 El modelo Phase 2 que integra datos del mercado (odds) logra una precisión significativamente mayor.

---

## 🚀 Siguiente Paso

Continúa con [02_FUNDAMENTOS_ML.md](02_FUNDAMENTOS_ML.md) para aprender los conceptos fundamentales de Machine Learning.
