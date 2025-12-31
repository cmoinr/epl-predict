# 🤖 Modelos de Clasificación en Machine Learning

## ¿Qué es Clasificación?

La **clasificación** es una tarea de Machine Learning donde predecimos **categorías discretas** (no valores continuos).

```
REGRESIÓN (Valores continuos)          CLASIFICACIÓN (Categorías)
────────────────────────────            ──────────────────────────
Predecir: 2.5 goles                     Predecir: Home Win / Draw / Away Win
Predecir: temperatura 15.3°C            Predecir: Lluvia / No lluvia
Predecir: precio $1,250                 Predecir: Spam / No Spam
```

---

## 🎯 Problemas de Clasificación en EPL-Predict

### 1. **Match Result (1X2)**
```
Entrada (Features):
├── home_form_5: 0.667
├── away_form_5: 0.733
├── h2h_home_wins: 0.60
└── position_diff: -0.10

Salida (Target):
├── Home Win (1)
├── Draw (X)
└── Away Win (2)
```

**Precisión actual**: 55.3%

### 2. **Both Teams To Score (BTTS)**
```
Entrada: Features del partido

Salida:
├── Sí (ambos equipos anotan)
└── No (un equipo no anota)
```

**Precisión actual**: 78.37%

---

## 🌳 Random Forest (Bosques Aleatorios)

### Concepto Básico

Un **Random Forest** es como una **junta de directivos**:
- Entrenan múltiples árboles de decisión
- Cada árbol vota por una clase
- La predicción final es el resultado más votado

```
                          Random Forest
                                │
                ┌───────────────┼───────────────┐
                │               │               │
            Árbol 1          Árbol 2         Árbol 3
              │                │               │
         Predice: H        Predice: H      Predice: X
              │                │               │
                └───────────────┼───────────────┘
                                │
                      Votación: 2 votos H, 1 voto X
                      Resultado Final: HOME WIN
```

### Estructura de un Árbol de Decisión

```
                    ¿home_form_5 > 0.65?
                         /    \
                      Sí /      \ No
                       /          \
                      /            \
                      
         ¿h2h_home > 0.5?      ¿position_diff > 0.2?
           /    \                 /    \
        Sí /      \ No         Sí /      \ No
         /          \          /          \
     HOME WIN     DRAW      AWAY WIN    DRAW
```

### Código

```python
from sklearn.ensemble import RandomForestClassifier

# 1. Crear modelo
rf = RandomForestClassifier(
    n_estimators=100,      # 100 árboles
    max_depth=10,          # Profundidad máxima
    min_samples_split=5,   # Mínimo de muestras para dividir
    random_state=42
)

# 2. Entrenar
rf.fit(X_train, y_train)

# 3. Predecir
predictions = rf.predict(X_test)
probabilities = rf.predict_proba(X_test)  # Probabilidades por clase

# 4. Evaluar
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predictions)
print(f"Precisión: {accuracy:.2%}")
```

### Ventajas & Desventajas

| Ventajas | Desventajas |
|----------|-------------|
| ✅ Maneja features no lineales | ❌ Puede hacer overfitting |
| ✅ Naturalmente multiclase | ❌ Lento con muchos features |
| ✅ Importancia de features | ❌ Difícil de interpretar |
| ✅ Robusto a outliers | ❌ Requiere más memoria |

**Precisión en EPL**: 52.8%

---

## 🚀 Gradient Boosting

### Concepto

**Gradient Boosting** es como **aprender de los errores**:
1. Entrena un primer árbol (comete errores)
2. El siguiente árbol **intenta corregir** esos errores
3. Repite N veces, mejorando paso a paso

```
Predicción Real: Home Win
─────────────────────────

Árbol 1: Predice Draw (ERROR: -0.33)
  └─> Siguiente árbol enfocado en corregir este error

Árbol 2: Predice Home Win (reduce error a -0.05)
  └─> Siguiente árbol sigue mejorando

Árbol 3: Predice Home Win (error casi 0)
  └─> Predicción final: Home Win ✓
```

### Código

```python
from sklearn.ensemble import GradientBoostingClassifier

gb = GradientBoostingClassifier(
    n_estimators=100,        # 100 etapas de boosting
    learning_rate=0.1,       # Velocidad de aprendizaje
    max_depth=5,             # Árboles más superficiales
    subsample=0.8            # Usa 80% de datos por etapa
)

gb.fit(X_train, y_train)
predictions = gb.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
```

### Ventajas & Desventajas

| Ventajas | Desventajas |
|----------|-------------|
| ✅ Muy preciso | ❌ Riesgo de overfitting |
| ✅ Maneja datos complejos | ❌ Lento entrenamiento |
| ✅ Importancia de features | ❌ Requiere tuning cuidadoso |

**Precisión en EPL**: 55.33%

---

## ⚡ XGBoost (eXtreme Gradient Boosting)

### ¿Qué lo hace diferente?

XGBoost es una **versión mejorada y optimizada** de Gradient Boosting:
- Más rápido (optimizado en C++)
- Mejor control de regularización
- Mejor manejo de datos desbalanceados
- Manejo automático de valores faltantes

### Código

```python
from xgboost import XGBClassifier

xgb = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,    # Usa 80% de features
    objective='multi:softmax', # Multiclase (1X2)
    num_class=3,             # 3 clases (H, X, A)
    random_state=42,
    eval_metric='mlogloss'    # Métrica de evaluación
)

xgb.fit(X_train, y_train)
predictions = xgb.predict(X_test)
```

### Parámetros Importantes

```python
# Regularización (evita overfitting)
lambda = 1.0         # L2 regularization (Ridge)
alpha = 0.0          # L1 regularization (Lasso)
gamma = 0.0          # Penalidad por complejidad

# Crecimiento del árbol
max_depth = 5        # Profundidad máxima
min_child_weight = 1 # Peso mínimo en hoja

# Learning
learning_rate = 0.1  # Tamaño del paso
n_estimators = 100   # Número de árboles
```

**Precisión en EPL**: 55.28%

---

## 🌟 LightGBM (Light Gradient Boosting Machine)

### Características Principales

LightGBM es aún más **rápido y eficiente** que XGBoost:
- 10-20x más rápido en entrenamiento
- Usa menos memoria
- Excelente con datos grandes
- Leaf-wise tree growth (crece donde más lo necesita)

```
XGBoost (Level-wise):        LightGBM (Leaf-wise):
──────────────────           ──────────────────
    Root                         Root
    / \                          / \
   /   \                        /   \
  /     \                      /     \
 /       \          vs       /       \
/_________ \                /         \
 /\   /\                   /\       /\
                                   (Crece más eficiente)
```

### Código

```python
from lightgbm import LGBMClassifier

lgb = LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    num_leaves=31,           # Máximo de hojas
    subsample=0.8,
    colsample_bytree=0.8,
    boosting_type='gbdt',    # Gradient Boosting Decision Tree
    num_class=3,
    objective='multiclass',
    metric='multi_logloss'
)

lgb.fit(X_train, y_train)
predictions = lgb.predict(X_test)
```

**Precisión en EPL**: 55.49%

---

## 🐱 CatBoost (Categorical Boosting)

### Especialidad: Datos Categóricos

CatBoost está **optimizado para manejar variables categóricas** automáticamente:
- No requiere one-hot encoding
- Maneja la codificación automáticamente
- Menos propenso a overfitting
- Resultados más consistentes

### Código

```python
from catboost import CatBoostClassifier

cat = CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    max_depth=5,
    verbose=10,              # Mostrar progreso
    cat_features=['HomeTeam', 'AwayTeam'],  # Features categóricas
    loss_function='MultiClass',
    random_state=42
)

# Nota: Puede pasar strings directamente
cat.fit(X_train, y_train, cat_features=cat_features_indices)
predictions = cat.predict(X_test)
```

**Precisión en EPL**: Similar a XGBoost (55%+)

---

## 📊 Comparación de Modelos

### Precisión en Predicción de Resultados (1X2)

```
┌──────────────────────────────────────────────────┐
│ Modelo                  Precisión    Tiempo      │
├──────────────────────────────────────────────────┤
│ Naive Baseline           33.3%      < 1ms       │
│ Logistic Regression      48.5%      10ms        │
│ Random Forest            52.8%      500ms       │
│ Gradient Boosting        55.33%     1500ms      │
│ XGBoost                  55.28%     800ms       │
│ LightGBM                 55.49%     300ms       │
│ CatBoost                 54.8%      1000ms      │
└──────────────────────────────────────────────────┘
```

### Trade-off: Precisión vs Velocidad

```
         Precisión
            ▲
            │     CatBoost LightGBM
            │         ●●
            │    Gradient Boosting ●
            │        XGBoost ●
         55%├─ ────────────●
            │    Random Forest
            │        ●
            │
         50%├─────●
            │   Logistic Regression
            │
         45%└──────────────────────────────────────►
              100ms  500ms  1000ms  2000ms      Tiempo
```

---

## 🎯 Métricas de Evaluación

### Matriz de Confusión (Clasificación Binaria)

```
              Predicción
              Positivo  Negativo
         ┌─────────────────────┐
Actual   │ TP   | FN           │
Positivo │ (Acierto) | (Error) │
         ├─────────────────────┤
         │ FP   | TN           │
Negativo │ (Error) | (Acierto) │
         └─────────────────────┘

TP = True Positive (predijo Sí, era Sí)
FN = False Negative (predijo No, era Sí)
FP = False Positive (predijo Sí, era No)
TN = True Negative (predijo No, era No)
```

### Métricas Principales

```python
from sklearn.metrics import (
    accuracy_score,          # (TP+TN) / Total
    precision_score,         # TP / (TP+FP) - Exactitud
    recall_score,            # TP / (TP+FN) - Sensibilidad
    f1_score,                # Media armónica
    roc_auc_score,           # Área bajo curva ROC
    confusion_matrix         # Matriz de confusión
)

# Calcular
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

print(f"Accuracy: {accuracy_score(y_test, y_pred):.2%}")
print(f"Precision: {precision_score(y_test, y_pred):.2%}")
print(f"Recall: {recall_score(y_test, y_pred):.2%}")
print(f"F1-Score: {f1_score(y_test, y_pred):.2%}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.2%}")
```

---

## 🔧 Hyperparameter Tuning

### Grid Search (Búsqueda en Grilla)

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}

grid = GridSearchCV(
    XGBClassifier(),
    param_grid,
    cv=5,              # 5-fold cross validation
    scoring='accuracy'
)

grid.fit(X_train, y_train)
print(f"Mejor precisión: {grid.best_score_:.2%}")
print(f"Mejores parámetros: {grid.best_params_}")
```

### Random Search (Búsqueda Aleatoria)

```python
from sklearn.model_selection import RandomizedSearchCV

random_search = RandomizedSearchCV(
    XGBClassifier(),
    param_grid,
    n_iter=20,  # 20 combinaciones aleatorias
    cv=5
)

random_search.fit(X_train, y_train)
```

---

## 🚨 Overfitting vs Underfitting

### El Triángulo del Aprendizaje

```
        Performance
            ▲
            │
    Bajo   │ ╱╲  Perfect
   Sesgo   │╱  ╲  Balance
            │    ╲
            │     ╲___ Overfitting
            │         (memoriza datos)
            │        ╱
            │______╱ Underfitting
            │     (muy simple)
            └──────────────────────────► Complejidad Modelo
```

### Señales de Overfitting

```
Train Accuracy: 98%  ←─ Muy alto
Test Accuracy:  52%  ←─ Mucho más bajo

Diferencia > 10%: Probable overfitting
```

### Soluciones

```python
# 1. Aumentar regularización
xgb = XGBClassifier(
    lambda=2.0,      # Aumentar
    alpha=0.5,       # Aumentar
    gamma=1.0        # Aumentar
)

# 2. Reducir complejidad
xgb = XGBClassifier(
    max_depth=3,     # Reducir de 7
    min_child_weight=5  # Aumentar
)

# 3. Más datos de entrenamiento
# 4. Early stopping
xgb.fit(X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=10)
```

---

## 📚 Recursos

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [CatBoost Documentation](https://catboost.ai/)
- [Scikit-learn Classification](https://scikit-learn.org/stable/modules/classification.html)

---

## 🚀 Siguiente Paso

Continúa con [06_ENSEMBLE_LEARNING.md](06_ENSEMBLE_LEARNING.md) para aprender cómo combinar múltiples modelos.
