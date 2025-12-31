# 📦 Librerías de Python para Machine Learning

Este documento explica las librerías utilizadas en el proyecto y sus funciones principales.

---

## 🔢 NumPy - Computación Numérica

### ¿Qué es?
**NumPy** (Numerical Python) es la librería fundamental para computación científica. Proporciona arrays multidimensionales eficientes y funciones matemáticas.

### ¿Por qué es importante?
- Los modelos de ML trabajan con **matrices numéricas**, no con listas de Python
- Es ~50x más rápido que las listas tradicionales
- Es la base de casi todas las demás librerías de ML

### Uso en el proyecto

```python
import numpy as np

# Crear array de features (de predictor.py)
features = np.array([
    home_shots,           # 0
    away_shots,           # 1
    home_shots_on_target, # 2
    # ... 28 features en total
]).reshape(1, -1)  # Reshape a (1, 28) para una predicción

# Operaciones comunes
np.mean(team_probs)          # Promedio
np.abs(array1 - array2)      # Valor absoluto de diferencias
np.clip(value, 0, 1)         # Limitar valores entre 0 y 1
np.nan_to_num(features)      # Reemplazar NaN con 0
np.hstack([arr1, arr2])      # Concatenar horizontalmente
```

### Conceptos clave

```python
# Shape (forma del array)
array = np.array([[1,2,3], [4,5,6]])
print(array.shape)  # (2, 3) → 2 filas, 3 columnas

# Reshape (cambiar forma)
flat = np.array([1,2,3,4,5,6])
matrix = flat.reshape(2, 3)   # Convertir a 2x3
prediction = flat.reshape(1, -1)  # 1 fila, columnas automáticas
```

---

## 🐼 Pandas - Manipulación de Datos

### ¿Qué es?
**Pandas** es la librería principal para análisis y manipulación de datos tabulares (como Excel, pero programable).

### Estructuras principales

```python
import pandas as pd

# DataFrame: tabla con filas y columnas nombradas
df = pd.read_csv('data/raw/epl_final.csv')

# Series: una columna individual
goals = df['FullTimeHomeGoals']  # Serie de goles locales
```

### Uso en el proyecto

```python
# feature_engineering.py

# 1. Cargar y ordenar datos
df = pd.read_csv('epl_final.csv')
df['MatchDate'] = pd.to_datetime(df['MatchDate'])
df = df.sort_values('MatchDate').reset_index(drop=True)

# 2. Filtrar datos
home_matches = df[df['HomeTeam'] == 'Arsenal']
recent_matches = df[df['MatchDate'] >= '2024-01-01']

# 3. Operaciones de rolling (ventanas móviles)
# Promedio de goles en últimos 10 partidos
df['AvgGoals_L10'] = df['FullTimeHomeGoals'].rolling(
    window=10, 
    min_periods=1
).mean().shift(1)  # shift(1) evita data leakage

# 4. Mapeo de categorías
result_map = {'A': 0, 'D': 1, 'H': 2}
y = df['FullTimeResult'].map(result_map)

# 5. Manejo de valores faltantes
df_filled = df.fillna(method='ffill')  # Forward fill
df_filled = df.fillna(df.mean())       # Llenar con promedio

# 6. Concatenar DataFrames
combined = pd.concat([df1, df2], axis=0)  # Verticalmente
combined = pd.concat([df1, df2], axis=1)  # Horizontalmente
```

### Rolling Windows (Ventanas Móviles)
Concepto muy usado para features temporales:

```python
# Calcular forma del equipo (últimos 5 partidos)
#
# Partido 1: Ganó (3 pts)
# Partido 2: Empató (1 pt)
# Partido 3: Perdió (0 pts)
# Partido 4: Ganó (3 pts)
# Partido 5: Ganó (3 pts)
# 
# Rolling(5).mean() = (3+1+0+3+3)/5 = 2.0 puntos promedio
```

---

## 🔬 Scikit-learn - Machine Learning

### ¿Qué es?
**Scikit-learn** es LA librería estándar de ML en Python. Proporciona algoritmos, preprocesamiento y evaluación.

### Módulos principales usados en el proyecto

```python
from sklearn.ensemble import (
    RandomForestClassifier,      # Clasificación con Random Forest
    RandomForestRegressor,       # Regresión con Random Forest
    GradientBoostingClassifier,  # Clasificación con Gradient Boosting
    GradientBoostingRegressor,   # Regresión con Gradient Boosting
    VotingClassifier,            # Ensemble de votación
    VotingRegressor
)

from sklearn.preprocessing import StandardScaler  # Normalización
from sklearn.model_selection import train_test_split  # División de datos
from sklearn.metrics import (
    accuracy_score,       # Precisión de clasificación
    mean_absolute_error,  # Error absoluto medio
    classification_report # Reporte detallado
)
```

### Flujo típico con Scikit-learn

```python
# 1. Preparar datos
X = df[feature_columns]
y = df['target']

# 2. Dividir en train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,     # 20% para test
    random_state=42    # Reproducibilidad
)

# 3. Escalar features (IMPORTANTE)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Aprende y transforma
X_test_scaled = scaler.transform(X_test)        # Solo transforma

# 4. Entrenar modelo
model = RandomForestClassifier(n_estimators=200)
model.fit(X_train_scaled, y_train)

# 5. Predecir
predictions = model.predict(X_test_scaled)
probabilities = model.predict_proba(X_test_scaled)

# 6. Evaluar
accuracy = accuracy_score(y_test, predictions)
```

### StandardScaler - Normalización

¿Por qué escalar las features?

```python
# Sin escalar:
# HomeShots: 0-30 (rango pequeño)
# TotalGoalsHistorico: 0-5000 (rango enorme)
# 
# El modelo daría más peso a TotalGoalsHistorico solo por su escala

# Con StandardScaler:
# Todas las features tienen media=0 y desviación estándar=1
# HomeShots: -2.5 a 2.5
# TotalGoalsHistorico: -2.5 a 2.5
```

---

## ⚡ XGBoost - Gradient Boosting Extremo

### ¿Qué es?
**XGBoost** (eXtreme Gradient Boosting) es una implementación optimizada de Gradient Boosting. Es conocido por ganar muchas competencias de Kaggle.

### Uso en el proyecto

```python
from xgboost import XGBClassifier, XGBRegressor

# Clasificación de resultado
xgb_result = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    use_label_encoder=False,
    eval_metric='mlogloss'
)
xgb_result.fit(X_train_scaled, y_train)

# Predicción
pred = xgb_result.predict(X_test_scaled)
prob = xgb_result.predict_proba(X_test_scaled)
```

### ¿Por qué XGBoost es popular?
- **Rápido**: Paralelización y optimizaciones
- **Preciso**: Regularización incorporada contra overfitting
- **Flexible**: Maneja valores faltantes automáticamente

---

## 💡 LightGBM - Gradient Boosting Ligero

### ¿Qué es?
**LightGBM** (Light Gradient Boosting Machine) es de Microsoft. Es más rápido que XGBoost especialmente con datasets grandes.

### Uso en el proyecto

```python
from lightgbm import LGBMClassifier, LGBMRegressor

lgb_result = LGBMClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    verbose=-1  # Silenciar logs
)
lgb_result.fit(X_train_scaled, y_train)
```

### Diferencia con XGBoost
- LightGBM crece los árboles **hoja a hoja** (leaf-wise)
- XGBoost crece **nivel a nivel** (level-wise)
- LightGBM es más rápido pero puede overfit más fácilmente

---

## 🐱 CatBoost - Gradient Boosting para Categóricos

### ¿Qué es?
**CatBoost** (Categorical Boosting) es de Yandex. Maneja variables categóricas de forma nativa sin necesidad de encoding.

### Uso en el proyecto

```python
from catboost import CatBoostClassifier, CatBoostRegressor

cat_result = CatBoostClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    verbose=0
)
cat_result.fit(X_train_scaled, y_train)
```

### Ventaja principal
Si tienes features como "HomeTeam" o "Season", CatBoost las maneja directamente sin convertirlas a números.

---

## 🗃️ Pickle - Serialización de Modelos

### ¿Qué es?
**Pickle** es el módulo de Python para **serializar** (guardar) objetos en archivos binarios y cargarlos después.

### Uso en el proyecto

```python
import pickle

# GUARDAR modelo entrenado
with open('models/rf_result_model.pkl', 'wb') as f:
    pickle.dump(trained_model, f)

# CARGAR modelo guardado (en predictor.py)
with open('models/rf_result_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Ahora puedes usar el modelo sin reentrenar
predictions = model.predict(new_data)
```

### Modelos guardados en el proyecto
```
models/
├── rf_result_model.pkl       # Random Forest - Resultado
├── gb_result_model.pkl       # Gradient Boosting - Resultado
├── xgb_result_model.pkl      # XGBoost - Resultado
├── lgb_result_model.pkl      # LightGBM - Resultado
├── cat_result_model.pkl      # CatBoost - Resultado
├── voting_result_model.pkl   # Voting Ensemble - Resultado
├── rf_goals_model.pkl        # Random Forest - Goles
├── rf_btts_model.pkl         # Random Forest - BTTS
├── scaler_model.pkl          # StandardScaler
└── phase2_voting_market.pkl  # Modelo con Market Intelligence
```

---

## 📊 Resumen de Librerías

| Librería | Propósito | Función Principal |
|----------|-----------|-------------------|
| **NumPy** | Computación numérica | Arrays, operaciones matemáticas |
| **Pandas** | Manipulación de datos | DataFrames, limpieza, transformación |
| **Scikit-learn** | ML general | Modelos, preprocesamiento, evaluación |
| **XGBoost** | Gradient Boosting | Modelos de alta precisión |
| **LightGBM** | Gradient Boosting rápido | Datasets grandes |
| **CatBoost** | GB para categóricos | Features categóricas nativas |
| **Pickle** | Serialización | Guardar/cargar modelos |

---

## 🚀 Siguiente Paso

Continúa con [04_FEATURE_ENGINEERING.md](04_FEATURE_ENGINEERING.md) para aprender cómo crear las variables que alimentan a estos modelos.
