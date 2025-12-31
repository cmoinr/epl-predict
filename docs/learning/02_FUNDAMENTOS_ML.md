# 🧠 Fundamentos de Machine Learning

## ¿Qué es Machine Learning?

**Machine Learning (ML)** es una rama de la Inteligencia Artificial que permite a las computadoras **aprender de datos** sin ser programadas explícitamente para cada tarea.

### Analogía Simple
Imagina que quieres enseñar a alguien a reconocer si un equipo va a ganar:

- **Programación tradicional**: Escribes reglas como "si el equipo local ganó los últimos 3 partidos Y el visitante perdió 2, entonces gana el local"
- **Machine Learning**: Le das miles de ejemplos de partidos pasados con sus resultados, y el algoritmo descubre las reglas por sí mismo

---

## 📚 Tipos de Machine Learning

### 1. 🎯 Aprendizaje Supervisado (Supervised Learning)
**Es lo que usa este proyecto.**

- Tienes **datos etiquetados**: sabes el resultado real de cada partido histórico
- El modelo aprende la relación entre **inputs (features)** y **outputs (targets)**

```
Features (X)                          Target (y)
─────────────────                     ──────────
Forma local: 2.5                      
Forma visitante: 1.8          →       Home Win ✓
Goles prom local: 1.8
Goles prom visitante: 1.2
```

**Ejemplos en el proyecto:**
- Clasificación de resultado (Home/Draw/Away)
- Predicción de goles totales
- BTTS (ambos anotan)

### 2. 🔍 Aprendizaje No Supervisado (Unsupervised Learning)
- No hay etiquetas/respuestas correctas
- El modelo encuentra **patrones ocultos**
- Ejemplo: agrupar equipos por estilo de juego

### 3. 🎮 Aprendizaje por Refuerzo (Reinforcement Learning)
- Un agente aprende por **prueba y error**
- Recibe recompensas/castigos
- Ejemplo: IA jugando videojuegos

---

## 🎯 Clasificación vs Regresión

Este proyecto usa **ambos tipos**:

### Clasificación
Predecir una **categoría/clase** discreta.

```python
# En el proyecto (predictor.py)
# Target: resultado del partido
result_map = {'A': 0, 'D': 1, 'H': 2}  # Clases: Away, Draw, Home

# El modelo responde: "Este partido será Home Win"
# Con probabilidades: Home 58%, Draw 27%, Away 15%
```

**Usos en el proyecto:**
- Resultado 1X2 (3 clases)
- BTTS Sí/No (2 clases - clasificación binaria)

### Regresión
Predecir un **valor numérico** continuo.

```python
# En el proyecto
# Target: goles totales
y_goals = df['FullTimeHomeGoals'] + df['FullTimeAwayGoals']

# El modelo responde: "Habrá aproximadamente 2.7 goles"
```

**Usos en el proyecto:**
- Predicción de goles totales

---

## 📊 División de Datos: Train/Test Split

### ¿Por qué dividir los datos?

Para evaluar si el modelo realmente **generaliza** o solo memoriza.

```
Dataset Total (8000 partidos)
├── 80% Train (6400 partidos) → Para ENTRENAR el modelo
└── 20% Test (1600 partidos)  → Para EVALUAR el modelo
```

### ⚠️ Importante: División Temporal

En series de tiempo (como partidos de fútbol), **NO debemos mezclar aleatoriamente**. Usamos **división temporal**:

```python
# De feature_engineering.py
# Split temporal (no aleatorio para series de tiempo)
split_idx = int(len(X_filled) * (1 - test_size))

X_train = X_filled[:split_idx]   # Partidos más antiguos
X_test = X_filled[split_idx:]    # Partidos más recientes
```

**¿Por qué?** No queremos que el modelo "vea el futuro" durante el entrenamiento. Si mezclamos, podría aprender de un partido de 2024 para predecir uno de 2020.

---

## 📐 Métricas de Evaluación

### Para Clasificación

#### Accuracy (Precisión)
```
Accuracy = Predicciones Correctas / Total de Predicciones

Ejemplo: Si de 100 partidos predijimos 55 correctamente:
Accuracy = 55/100 = 55%
```

#### Confusion Matrix (Matriz de Confusión)
```
                    Predicción
                    Away  Draw  Home
              Away   15    8     7     (30 partidos Away reales)
Realidad      Draw   10   12    13     (35 partidos Draw reales)
              Home    5   10    20     (35 partidos Home reales)
```

### Para Regresión

#### MAE (Mean Absolute Error)
```
MAE = Promedio de |Real - Predicción|

Si predicción = 2.5 goles y real = 3 goles:
Error = |3 - 2.5| = 0.5

MAE bajo = mejor modelo
```

En el proyecto, el mejor modelo de goles tiene **MAE = 0.84** (se equivoca en promedio por 0.84 goles).

---

## 🔄 El Flujo de Entrenamiento

```
1. CARGAR DATOS
   df = pd.read_csv('epl_final.csv')

2. FEATURE ENGINEERING
   X, y = engineer_features(df)

3. DIVIDIR DATOS
   X_train, X_test, y_train, y_test = train_test_split(X, y)

4. NORMALIZAR (SCALING)
   scaler = StandardScaler()
   X_train_scaled = scaler.fit_transform(X_train)
   X_test_scaled = scaler.transform(X_test)

5. ENTRENAR MODELO
   model = RandomForestClassifier()
   model.fit(X_train_scaled, y_train)

6. EVALUAR
   predictions = model.predict(X_test_scaled)
   accuracy = accuracy_score(y_test, predictions)

7. GUARDAR MODELO
   pickle.dump(model, open('model.pkl', 'wb'))
```

---

## 🎛️ Hiperparámetros

Son **configuraciones** del algoritmo que debemos establecer **antes** del entrenamiento:

```python
# Ejemplo de hiperparámetros en Random Forest
RandomForestClassifier(
    n_estimators=200,      # Número de árboles
    max_depth=10,          # Profundidad máxima de cada árbol
    min_samples_split=5,   # Mínimo de muestras para dividir un nodo
    random_state=42        # Semilla para reproducibilidad
)
```

### ¿Cómo encontrar los mejores?

- **Grid Search**: Probar todas las combinaciones
- **Random Search**: Probar combinaciones aleatorias
- **Cross-Validation**: Validar en múltiples splits

---

## ⚠️ Problemas Comunes

### Overfitting (Sobreajuste)
El modelo **memoriza** los datos de entrenamiento pero no generaliza.

```
Train Accuracy: 95%  ← ¡Muy bien en entrenamiento!
Test Accuracy: 52%   ← Pero mal en datos nuevos 😢
```

**Soluciones:**
- Más datos de entrenamiento
- Simplificar el modelo (menos profundidad, menos features)
- Regularización
- Cross-validation

### Underfitting (Subajuste)
El modelo es **demasiado simple** y no captura los patrones.

```
Train Accuracy: 40%  ← Mal incluso en entrenamiento
Test Accuracy: 38%   ← Y mal en test también
```

**Soluciones:**
- Modelo más complejo
- Más features
- Menos regularización

### Data Leakage (Fuga de Datos)
Cuando el modelo tiene acceso a información que **no tendría en producción**.

**Ejemplo en fútbol:**
Si incluyes los tiros a puerta del partido como feature para predecir el resultado... ¡el modelo tendrá 99% accuracy porque esa información ya revela el resultado! Pero en un partido futuro, no tienes esos datos antes de que ocurra.

---

## 💡 Conceptos Clave en el Proyecto

### Probabilidades de Predicción
Los modelos no solo dicen "Home Win", sino que dan **probabilidades**:

```python
# De predictor.py
prob_result_rf = self.rf_result.predict_proba(X_new_scaled)[0]
# Resultado: [0.15, 0.27, 0.58] → Away 15%, Draw 27%, Home 58%
```

### Confianza del Modelo
La **máxima probabilidad** indica qué tan seguro está el modelo:

```python
# Si las probabilidades son [0.15, 0.27, 0.58]:
confianza = max([0.15, 0.27, 0.58]) * 100  # = 58%

# Si fueran [0.33, 0.34, 0.33]:
confianza = 34%  # ← Modelo muy inseguro
```

---

## 🚀 Siguiente Paso

Continúa con [03_LIBRERIAS_ML_PYTHON.md](03_LIBRERIAS_ML_PYTHON.md) para conocer las librerías que hacen posible todo esto.
