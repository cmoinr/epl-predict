# Premier League ML/AI - Predictor de Resultados y Odds

## Objetivo
Predecir resultados de partidos y recomendar odds para la Premier League usando Machine Learning.

## Estructura del Proyecto

```
premier-league-ml/
├── data/                 # Datos crudos y procesados
├── notebooks/            # Jupyter notebooks para análisis
├── src/                  # Código modular reutilizable
│   ├── data_collection.py      # Descarga y procesamiento de datos
│   ├── feature_engineering.py  # Creación de features
│   ├── models.py               # Definición de modelos ML
│   └── utils.py                # Funciones auxiliares
├── models/               # Modelos entrenados guardados
├── requirements.txt      # Dependencias del proyecto
└── README.md
```

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

## 🎯 Configuración del Proyecto

**Tu estrategia**: 
- Dataset: EPL 2000-2025 (máximo histórico)
- Predicciones: Resultado (1X2) + Goles Totales
- Objetivo: Value betting rentable
- Odds: Comparar vs mercado

## 📋 Próximos Pasos

1. ✅ **Dependencias instaladas**
2. ⏳ **Obtener dataset** → `bash setup_data.sh` o descarga manual
3. ⏳ **EDA notebook** → `jupyter notebook notebooks/01_eda_and_modeling.ipynb`
4. ⏳ **Feature engineering** → Crear features de predicción
5. ⏳ **Entrenar modelos** → RF, GB para clasificación y regresión
6. ⏳ **Value betting** → Comparar predicciones vs odds

Ver: `PROXIMOS_PASOS.md` para guía detallada
