# 🔴 DIAGNÓSTICO: ¿Por qué el modelo predice demasiados "Draw"?

## 1. LA REALIDAD DE TUS DATOS

```
Tu Dataset Histórico (9,410 partidos):
┌─────────────────────────┬──────────┬─────────┐
│ Resultado               │ Cantidad │ % Total │
├─────────────────────────┼──────────┼─────────┤
│ 🏠 Home Win (1)         │  4,310   │ 45.80%  │
│ 🤝 Draw (X)             │  2,318   │ 24.63%  │
│ 🚗 Away Win (2)         │  2,782   │ 29.56%  │
└─────────────────────────┴──────────┴─────────┘

💡 Esperado en fútbol real: ~45% / ~27% / ~28%
⚠️  TU DATASET: ~46% / ~24.6% / ~29.6%  ✓ ESTÁ BIEN DISTRIBUIDO
```

## 2. EL VERDADERO PROBLEMA: SESGO DE CLASE EN EL MODELO

El modelo NO tiene sesgo en los **datos**, tiene sesgo en cómo **entrena**.

### 🤔 ¿Qué está pasando?

**Caso real de tu modelo:**
```
Predicción Chelsea vs Liverpool:

Random Forest:
  - Confianza: 37% (INDECISO)
  - Probabilidades: Away 33% | Draw 37% | Home 30%

Gradient Boosting:
  - Confianza: 84.5% (MUY SEGURO)
  - Probabilidades: Away 7.6% | Draw 84.5% | Home 7.9%
```

**¿Por qué Gradient Boosting predice 84.5% Draw?**

Es porque durante el entrenamiento, el modelo encontró un patrón que **coincide accidentalmente** con empates. No es que "piense que habrá empate", es que sus features (características) generan valores que el modelo aprendió a asociar con empates.

### 📊 La realidad de los goles

| Resultado | Promedio Goles | 
|-----------|---|
| Home Win | 2.99 goles |
| **Draw** | **2.01 goles** ⬅️ MENOS goles |
| Away Win | 2.88 goles |

**Descubrimiento:** Los EMPATES tienen **MENOS goles totales** (2.01 vs 2.99)

Esto significa que:
- Partidos con pocas oportunidades → Tiende a empate
- Partidos con muchas oportunidades → Tiende a victoria clara

**Tu modelo está viendo:**
```
Features bajos (poco ofensivos) → Predice Draw
```

Pero esto es INCORRECTO cuando:
- Chelsea juega contra Liverpool (claro favorito a victoria)
- Hay gran diferencia en posición de tabla

---

## 3. SOLUCIONES CONCRETAS

### ✅ SOLUCIÓN 1: Balancear clases en el entrenamiento

**Problema:** El modelo ve 45% home wins vs 24% draws
- Aprende a ser conservador con draws (son más raros)
- Pero cuando ve features "ambiguas", elige draw por defecto

**Solución:** Usar `class_weight='balanced'`

```python
# EN: src/models.py

# Random Forest
rf_result = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',  # ← AGREGAR ESTO
    random_state=42,
    n_jobs=-1
)

# Gradient Boosting
gb_result = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',  # ← AGREGAR ESTO
    random_state=42
)
```

---

### ✅ SOLUCIÓN 2: Mejorar las características (Features)

**El problema real:** Tus features NO capturan bien "quién es favorito"

Features actuales en `predictor.py`:
```
- HomeTeam_Form (últimos 5 partidos)
- AwayTeam_Form (últimos 5 partidos)
- H2H_HomeTeamWins
- Goals For/Against (media)
- Mes y día de la semana
- HomeAdvantage (constante 0.3)
```

**Lo que FALTA:**
```python
# AGREGAR ESTAS CARACTERÍSTICAS:

1. Diferencia en posición de tabla (ranking)
   Chelsea 6º vs Liverpool 8º → diferencia = -2 (Liverpool es mejor)
   
2. Diferencia en goles anotados este año
   Chelsea 45 goles vs Liverpool 52 goles → diferencia = -7
   
3. Racha actual (últimos 3 partidos, no 5)
   Si ganó 2 de 3: forma = 0.67
   
4. Ventaja en casa mejorada (basada en datos)
   Home win rate: 50% en casa vs 30% fuera
   
5. Factor de "fuerza relativa"
   (Puntos Chelsea - Puntos Liverpool) / 10
```

---

### ✅ SOLUCIÓN 3: Ajustar los hiperparámetros

**Problema:** Los parámetros actuales son "seguros" pero blandos

```python
# ACTUAL (muy conservador):
max_depth=15,              # Permite muchas divisiones
min_samples_split=5,       # Solo requiere 5 muestras para dividir
min_samples_leaf=2         # Hojas muy pequeñas

# MEJORADO (menos overfitting, más decisiones claras):
max_depth=10,              # Reduce complejidad
min_samples_split=10,      # Requiere más muestras
min_samples_leaf=5         # Hojas más grandes
max_features='sqrt',       # Usa sqrt(n_features) en cada división
```

---

### ✅ SOLUCIÓN 4: Usar probabilidades calibradas

**Problema:** Las probabilidades del modelo NO son reales

```
Random Forest: 37% Draw
Gradient Boosting: 84.5% Draw

¿Significa que hay 37% o 84.5% de probabilidad real? NO.
El modelo está "adivinando" sin calibración.
```

**Solución:** Usar `CalibratedClassifierCV`

```python
from sklearn.calibration import CalibratedClassifierCV

# Después de entrenar el modelo:
rf_result_calibrated = CalibratedClassifierCV(
    rf_result, 
    method='sigmoid',
    cv=5
)
rf_result_calibrated.fit(X_train, y_result_train)

# Ahora las probabilidades son REALES
prob_calibrated = rf_result_calibrated.predict_proba(X_new)
```

---

## 4. PLAN DE ACCIÓN PRIORITARIO

### Paso 1: QUICK FIX (5 minutos)
```
✅ Agregar class_weight='balanced' a ambos modelos
✅ Reentrenar
✅ Probar predicciones
```

### Paso 2: MEJORA MEDIANA (30 minutos)
```
✅ Mejorar features: agregar diferencia de tabla + goles anotados
✅ Reentrenar
✅ Probar
```

### Paso 3: MEJORA AVANZADA (1 hora)
```
✅ Calibrar probabilidades
✅ Ajustar hiperparámetros
✅ Validation cruzada
✅ Comparar modelos
```

---

## 5. EJEMPLO: ANTES vs DESPUÉS

### ANTES (Actual):
```
Chelsea vs Liverpool
Random Forest: Draw (37%)
Gradient Boosting: Draw (84.5%)
```

### DESPUÉS (Esperado con mejoras):
```
Chelsea vs Liverpool
Random Forest: Home Win (52%)
Gradient Boosting: Home Win (68%)

Promedio de confianza: 60%
```

---

## 6. ¿CUÁL ES LA CAUSA ROOT?

**Tu modelo está tratando todos los partidos igual:**

```
Entrada: Features genéricas
  ↓
Modelo: "No veo diferencia clara entre equipos"
  ↓
Salida: "Entonces debe ser Draw" (default seguro)
```

**Lo que debería hacer:**

```
Entrada: Chelsea 6º tabla, Liverpool 8º tabla, Chelsea 45 goles, Liverpool 52
  ↓
Modelo: "Hay ligera ventaja para Liverpool pero margen pequeño"
  ↓
Salida: "Liverpool ganará con 65% de confianza, Chelsea 30%, Draw 5%"
```

---

## 7. RECOMENDACIÓN FINAL

1. **Comienza por Solución 1** (class_weight) - impacto inmediato
2. **Luego Solución 2** (features mejores) - impacto mayor
3. **Después Solución 3** (hiperparámetros) - fine tuning
4. **Finalmente Solución 4** (calibración) - robustez

El problema NO es tu dataset, es que el modelo necesita **aprender a diferenciar mejor entre equipos**.

