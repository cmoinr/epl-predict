# 🏆 Premier League ML Predictor - Value Betting System

## 🎯 Objetivo
Predictor de resultados EPL con análisis de odds para identificar oportunidades de **value betting** rentables.

## ⚡ Quick Start

```bash
# Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Entrenar modelos
python src/train_models.py

# Predicción + Análisis de odds
python run_analysis.py
```

Ver `SETUP.md` para instrucciones completas.

## 📊 Estado del Proyecto

| Componente | Estado | Descripción |
|-----------|--------|-------------|
| **Modelado** | ✅ Completo | Random Forest + Gradient Boosting |
| **Predicción** | ✅ Completo | Resultados 1X2 + Goles totales |
| **Odds Integration** | ✅ Completo | Comparación modelo vs mercado |
| **Value Betting** | ✅ Completo | Edge, EV, Kelly Criterion |
| **Scripts** | ✅ Completo | predict_match.py, run_analysis.py |

## Fases del Proyecto

### Fase 1: Preparación (ACTUAL)
- ✅ Estructura del proyecto
- ⏳ Configurar dependencias
- ⏳ Definir plan de datos

### Fase 2: Recopilación de Datos
- Obtener histórico de partidos PL
- Features: Form (últimos 5 partidos), Head-to-Head, Posición en tabla, etc.

### Fase 3: Análisis Exploratorio (EDA)
- Visualizar distribuciones
- Identificar correlaciones
- Validar calidad de datos

### Fase 4: Feature Engineering
- Crear features derivadas
- Normalización y escalado
- Manejo de valores faltantes

### Fase 5: Modelado
- Entrenar múltiples algoritmos
- Validación cruzada
- Tuning de hiperparámetros

### Fase 6: Evaluación y Predicciones
- Métricas: Accuracy, Precision, Recall, F1
- Pruebas en datos nuevos
- Recomendación de odds

## Stack Tecnológico

- **Python 3.x**: Lenguaje principal
- **Pandas**: Manipulación de datos
- **Scikit-learn**: Machine Learning
- **XGBoost/LightGBM**: Algoritmos avanzados
- **Matplotlib/Seaborn**: Visualización
- **Jupyter**: Análisis interactivo

## 🚀 Guía Rápida

**Estado**: Feature Engineering listo para ejecutar

1. **Lee** (2 min): `QUICK_START_FEATURES.md`
2. **Ejecuta** (10 min): Jupyter sección 3
3. **Cuéntame**: Resultados

---

## 🎯 Configuración del Proyecto

**Tu estrategia**: 
- Dataset: EPL 2000-2025 (máximo histórico)
- Predicciones: Resultado (1X2) + Goles Totales
- Objetivo: Value betting rentable
- Odds: Comparar vs mercado

## 📊 Estado Actual

| Fase | Estado | Detalles |
|------|--------|----------|
| 1. Dependencias | ✅ Completa | Pandas, SKlearn, XGBoost, etc. |
| 2. Dataset | ✅ Completa | 9,380 partidos × 25 columnas |
| 3. EDA | ✅ Completa | Estructura explorada sin NaNs |
| 4. Features | ⏳ **AHORA** | Crear variables derivadas (~40 features) |
| 5. Modelado | ⏳ Próximo | Random Forest, Gradient Boosting |
| 6. Evaluación | ⏳ Próximo | Accuracy, Precision, Recall |
| 7. Value Betting | ⏳ Próximo | Comparar vs odds reales |

## 🔧 Feature Engineering

**Qué hace:**
- Form: Puntos en últimos 5 partidos
- H2H: Histórico entre equipos
- Goals Avg: Rendimiento ofensivo/defensivo
- Home Advantage: Ventaja de jugar en casa
- Temporal: Mes, día semana, año

**Archivos:**
- `src/feature_engineering.py` - Código
- `QUICK_START_FEATURES.md` - Leer primero
- `GUIA_FEATURES.md` - Detalle técnico

**Ejecutar:**
```bash
jupyter notebook notebooks/01_eda_and_modeling.ipynb
# → Sección 3: Feature Engineering
```
