# 🎭 Ensemble Learning: Combinando Múltiples Modelos

## ¿Qué es Ensemble Learning?

**Ensemble Learning** significa usar **múltiples modelos juntos** para tomar mejores decisiones que un solo modelo.

> 💡 **Analogía**: Es como una junta de expertos. En vez de confiar en un solo experto, reúnes a varios y tu predicción es mejor que cualquiera de ellos individualmente.

---

## 🧠 El Poder de la Diversidad

### El Ejemplo del Millonario

En el concurso "¿Cuánto pesa este toro?", 787 personas adivinaron:
- **Promedio de adivinanzas**: 1,197 libras
- **Peso real**: 1,198 libras
- **Error**: Solo 1 libra (0.08%)

```
Mejor adivinanza individual: 1,096 libras (error: 102 lbs)
Pero el promedio fue más preciso que cualquier individuo
```

**En Machine Learning pasa lo mismo**: La predicción promedio de múltiples modelos suele ser mejor que cualquiera individualmente.

---

## 📊 Métodos de Ensemble

### 1. **Voting (Votación)**

Cada modelo "vota" y la predicción final es el resultado más votado.

```
Entrada: Arsenal vs Chelsea

    Model 1 (XGBoost):   HOME WIN     → voto 1
    Model 2 (LightGBM):  HOME WIN     → voto 1
    Model 3 (Random):    DRAW         → voto 0
    
    Resultado: 2 votos HOME WIN
    Predicción Final: HOME WIN
```

### Código: Voting Classifier

```python
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier

# Crear modelos individuales
xgb = XGBClassifier(n_estimators=100)
lgb = LGBMClassifier(n_estimators=100)
rf = RandomForestClassifier(n_estimators=100)

# Crear ensemble (votación dura)
voting = VotingClassifier(
    estimators=[
        ('xgb', xgb),
        ('lgb', lgb),
        ('rf', rf)
    ],
    voting='hard'  # Mayoría de votos
)

voting.fit(X_train, y_train)
predictions = voting.predict(X_test)

# Para probabilidades (voting suave)
voting_soft = VotingClassifier(
    estimators=[('xgb', xgb), ('lgb', lgb), ('rf', rf)],
    voting='soft'  # Promedio de probabilidades
)

probabilities = voting_soft.predict_proba(X_test)
```

### Hard vs Soft Voting

```
Hard Voting (Mayoría):
  Model 1: HOME WIN (probabilidad: 0.7)
  Model 2: HOME WIN (probabilidad: 0.55)
  Model 3: DRAW    (probabilidad: 0.4)
  
  Resultado: HOME WIN (gana por 2 votos)
  
Soft Voting (Promedio de probabilidades):
  HOME WIN: (0.7 + 0.55 + 0.2) / 3 = 0.483
  DRAW:     (0.2 + 0.35 + 0.4) / 3 = 0.317
  AWAY:     (0.1 + 0.1 + 0.4) / 3 = 0.200
  
  Resultado: HOME WIN (probabilidad más alta)
```

---

### 2. **Stacking (Apilamiento)**

Los modelos básicos entrenan, luego sus predicciones se usan como features para un modelo "meta" que aprende cómo combinarlos óptimamente.

```
NIVEL 0 (Base Models):
┌──────────┐    ┌──────────┐    ┌──────────┐
│ XGBoost  │    │ LightGBM │    │ RF       │
└─────┬────┘    └─────┬────┘    └─────┬────┘
      │ Pred: 0.65    │ Pred: 0.58    │ Pred: 0.55
      │               │               │
      └───────────────┼───────────────┘
                      │
        Features Nivel 1: [0.65, 0.58, 0.55]
                      │
      ┌───────────────┴───────────────┐
      │                               │
      │     NIVEL 1 (Meta-Learner)    │
      │        (e.g., Logistic)       │
      │                               │
      └───────────────┬───────────────┘
                      │
            Predicción Final: HOME WIN
```

### Código: Stacking

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

# Modelos base
base_learners = [
    ('xgb', XGBClassifier(n_estimators=50)),
    ('lgb', LGBMClassifier(n_estimators=50)),
    ('rf', RandomForestClassifier(n_estimators=50))
]

# Meta-learner (aprende cómo combinar)
meta_learner = LogisticRegression(max_iter=1000)

# Crear stacking
stacking = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_learner,
    cv=5  # Cross-validation para generar features
)

stacking.fit(X_train, y_train)
predictions = stacking.predict(X_test)
probabilities = stacking.predict_proba(X_test)
```

### Ventajas del Stacking

```python
# El meta-learner aprende pesos automáticamente
# En lugar de voto igual (1/3 cada uno):
# XGBoost:  40%  (mejor modelo)
# LightGBM: 35%  
# RF:       25%  (menos confiable)
```

---

### 3. **Blending**

Similar al Stacking pero más simple: divide datos en 3 partes.

```
Data
├── Train Set 1 (60%)
│   └─> Entrena modelos base
│
├── Train Set 2 (20%)
│   └─> Genera predicciones (features para meta-learner)
│
└── Test Set (20%)
    └─> Evaluación final
```

### Código: Blending

```python
from sklearn.model_selection import train_test_split

# Dividir datos
X_train_base, X_blend, y_train_base, y_blend = train_test_split(
    X_train, y_train, test_size=0.3, random_state=42
)

# Entrenar modelos base
xgb.fit(X_train_base, y_train_base)
lgb.fit(X_train_base, y_train_base)
rf.fit(X_train_base, y_train_base)

# Generar predicciones blend
blend_preds_xgb = xgb.predict_proba(X_blend)
blend_preds_lgb = lgb.predict_proba(X_blend)
blend_preds_rf = rf.predict_proba(X_blend)

# Crear features de blending
X_blend_meta = np.hstack([
    blend_preds_xgb,
    blend_preds_lgb,
    blend_preds_rf
])

# Entrenar meta-learner
meta = LogisticRegression()
meta.fit(X_blend_meta, y_blend)

# Predecir en test
test_preds_xgb = xgb.predict_proba(X_test)
test_preds_lgb = lgb.predict_proba(X_test)
test_preds_rf = rf.predict_proba(X_test)

X_test_meta = np.hstack([
    test_preds_xgb,
    test_preds_lgb,
    test_preds_rf
])

predictions = meta.predict(X_test_meta)
```

---

### 4. **Boosting**

Entrenar modelos secuencialmente, donde cada uno **corrige los errores del anterior**.

```
Iteración 1: Entrena Model 1 (comete errores)
                    ↓
Iteración 2: Entrena Model 2 (enfocado en errores de M1)
                    ↓
Iteración 3: Entrena Model 3 (enfocado en errores de M2)
                    ↓
Predicción: Suma ponderada de predicciones
```

Ya vimos esto en XGBoost y LightGBM (son métodos de boosting).

---

### 5. **Bagging**

Entrenar modelos en paralelo con **subconjuntos aleatorios** de datos.

```
Dataset Original (1000 muestras)
        │
    ┌───┼───┬───┬───┐
    │   │   │   │   │
    ▼   ▼   ▼   ▼   ▼
  Boot Boot Boot Boot Boot
   Set1 Set2 Set3 Set4 Set5
    │   │   │   │   │
  [Train] [Train] [Train] [Train] [Train]
  Model1  Model2  Model3  Model4  Model5
    │   │   │   │   │
    └───┼───┴───┴───┘
        │
    Predicción Final
    (Promedio o Voto)
```

Random Forest y Extra Trees son ejemplos de Bagging.

---

## 📈 Mejora de Precisión

### Ejemplo Real: Predicción Match Result (1X2)

```
┌─────────────────────────────────────────┐
│ Modelo Individual        Precisión       │
├─────────────────────────────────────────┤
│ XGBoost                  55.28%          │
│ LightGBM                 55.49%          │
│ Random Forest            52.80%          │
│ CatBoost                 54.80%          │
├─────────────────────────────────────────┤
│ Voting (Hard)            56.15%   (+0.66%)│
│ Voting (Soft)            56.42%   (+0.93%)│
│ Stacking                 56.85%   (+1.36%)│
│ Phase2 (Market+Ensemble) 80.38%   (+25%) │
└─────────────────────────────────────────┘
```

> 💡 El stacking logra superar a cualquier modelo individual

---

## 🎯 Cuándo Usar Cada Método

| Método | Velocidad | Precisión | Complejidad | Caso de Uso |
|--------|-----------|-----------|-------------|------------|
| Voting | ⚡⚡⚡ | ⭐⭐⭐ | Bajo | Producción rápida |
| Stacking | ⚡ | ⭐⭐⭐⭐⭐ | Alto | Máxima precisión |
| Blending | ⚡⚡ | ⭐⭐⭐⭐ | Medio | Balance datos |
| Boosting | ⚡⚡ | ⭐⭐⭐⭐ | Medio | Secuencial |
| Bagging | ⚡ | ⭐⭐⭐ | Medio | Robustez |

---

## 🔧 Implementación Recomendada para EPL-Predict

```python
from sklearn.ensemble import (
    VotingClassifier, 
    StackingClassifier
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

class EnsemblePredictor:
    def __init__(self):
        # Modelos base: Diversidad
        self.xgb = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5
        )
        
        self.lgb = LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5
        )
        
        self.rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10
        )
        
        # Ensemble
        self.ensemble = StackingClassifier(
            estimators=[
                ('xgb', self.xgb),
                ('lgb', self.lgb),
                ('rf', self.rf)
            ],
            final_estimator=LogisticRegression(max_iter=1000),
            cv=5
        )
    
    def fit(self, X_train, y_train):
        self.ensemble.fit(X_train, y_train)
        return self
    
    def predict(self, X_test):
        return self.ensemble.predict(X_test)
    
    def predict_proba(self, X_test):
        return self.ensemble.predict_proba(X_test)
```

---

## 📊 Monitoreo de Ensemble

```python
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score
)

# Evaluación individual
print("Modelos individuales:")
print(f"XGBoost:   {accuracy_score(y_test, xgb.predict(X_test)):.2%}")
print(f"LightGBM:  {accuracy_score(y_test, lgb.predict(X_test)):.2%}")
print(f"RF:        {accuracy_score(y_test, rf.predict(X_test)):.2%}")

print("\nEnsemble:")
print(f"Ensemble:  {accuracy_score(y_test, ensemble.predict(X_test)):.2%}")

print("\nDetalle:")
print(classification_report(y_test, ensemble.predict(X_test)))
```

---

## 🚨 Errores Comunes

### ❌ **Usar Modelos Muy Similares**

```python
# ❌ MAL: Todos son Gradient Boosting
ensemble = VotingClassifier(estimators=[
    ('xgb', XGBClassifier()),
    ('lgb', LGBMClassifier()),
    ('cat', CatBoostClassifier())
])

# ✅ BIEN: Mezclar algoritmos diferentes
ensemble = VotingClassifier(estimators=[
    ('xgb', XGBClassifier()),
    ('rf', RandomForestClassifier()),
    ('svm', SVC(probability=True))
])
```

### ❌ **Overfitting del Meta-Learner**

```python
# ❌ MAL: Meta-learner muy complejo
meta = XGBClassifier(max_depth=10, n_estimators=200)

# ✅ BIEN: Meta-learner simple
meta = LogisticRegression(C=1.0)
```

### ❌ **Data Leakage en Stacking**

```python
# ❌ MAL: Entrenar meta-learner con predicciones de train
train_preds = xgb.predict_proba(X_train)
meta.fit(train_preds, y_train)

# ✅ BIEN: Usar cross-validation
# Automático en StackingClassifier con cv=5
```

---

## 📚 Comparación: EPL-Predict Phase 2

El proyecto usa un ensemble sofisticado que combina:
- **3 modelos base** (XGBoost, LightGBM, Random Forest)
- **Features de mercado** (odds de casas de apuestas)
- **Meta-learner** (Logistic Regression)

**Resultado**: 80.38% de precisión (vs 55% individual)

---

## 📚 Recursos

- [Scikit-learn Ensemble](https://scikit-learn.org/stable/modules/ensemble.html)
- [XGBoost Ensemble](https://xgboost.readthedocs.io/)
- "Ensemble Methods" - Zhou Zhihua

---

## 🚀 Siguiente Paso

Continúa con [07_VALUE_BETTING_ODDS.md](07_VALUE_BETTING_ODDS.md) para aprender la matemática de las apuestas de valor.
