# 🔧 Feature Engineering - Guía de Ejecución

## Tu Dataset tiene:
- ✅ 9,380 partidos (muestras)
- ✅ 22 columnas originales
- ✅ Sin valores nulos
- ✅ Datos limpios y listos

---

## ¿Qué es Feature Engineering?

Es el proceso de **crear nuevas variables predictivas** usando la información que tenemos.

**Analogía**: Si tienes datos de un partido, Feature Engineering es hacer preguntas como:
- "¿Cómo ha estado jugando este equipo?" → **Form**
- "¿Ganan siempre contra este rival?" → **H2H**
- "¿Cuántos goles mete generalmente?" → **Goals Avg**

---

## Pasos en el Notebook

### Paso 1: Cargar el módulo
```python
from feature_engineering import EPLFeatureEngineer
```

### Paso 2: Crear el ingeniero
```python
engineer = EPLFeatureEngineer(df_processed)
```

### Paso 3: Generar features
```python
X, y_result, y_goals = engineer.engineer_features()
```

**Resultado:**
- `X`: DataFrame con todas las features (variables predictivas)
- `y_result`: Target para resultado (0=Away, 1=Draw, 2=Home)
- `y_goals`: Target para goles totales

---

## Qué Hace Cada Feature

### Features que Crea:

| Feature | Función | Ejemplo |
|---------|---------|---------|
| `HomeTeam_Form` | Puntos en últimos 5 partidos | 2.3 puntos promedio |
| `AwayTeam_Form` | Form del visitante | 1.8 puntos promedio |
| `H2H_HomeTeamWins` | % victorias en H2H | 60% gana de local en H2H |
| `HomeGoalsFor` | Goles promedio a favor | 2.1 goles/partido |
| `HomeAdvantage` | Ventaja de jugar en casa | +0.5 puntos |
| `Month` | Mes del partido | 1-12 |
| `DayOfWeek` | Día semana | 0=Lunes, 6=Domingo |

Ver: `GUIA_FEATURES.md` para descripción completa.

---

## En Tu Notebook

Ya hemos agregado celdas para que ejecutes Feature Engineering paso a paso.

**Célula 1: Analizar Targets**
- Ver distribución de resultados
- Ver distribución de goles

**Célula 2: Crear Features**
```python
engineer = EPLFeatureEngineer(df_processed)
X, y_result, y_goals = engineer.engineer_features()
```

**Célula 3: Inspeccionar Features**
- Ver todas las columnas creadas
- Ver estadísticas (media, std, min, max)

**Célula 4: Preparar para Modelado**
- Llenar NaNs (si los hay)
- Split train/test (80/20)
- Normalizar features

---

## Output Esperado

```
🔧 CREANDO FEATURES INGENIERILES...

  → Form de equipos...
  → Estadísticas de goles...
  → Ventaja de casa...
  → Estadísticas de tiros...

✅ Features creadas exitosamente!
   Dimensiones X: (9380, 30)
   - 9380 muestras (partidos)
   - 30 features (variables)
```

---

## ¿Por Qué Esto Es Importante?

Un modelo ML es tan bueno como sus **features**.

### Comparación:

**SIN Feature Engineering:**
```
Features: HomeShots, AwayShots, Fouls, etc. (solo estadísticas del partido)
Accuracy: ~50% (peor que tirar moneda)
Razón: No capturan tendencias históricas
```

**CON Feature Engineering:**
```
Features: Form, H2H, GoalsAvg, HomeAdvantage, etc.
Accuracy: ~60-65% (significativamente mejor)
Razón: Capturan patrones y tendencias
```

---

## Próximo Paso: Modelado

Una vez tengas features listas:

1. **Entrenar Modelos:**
   - Random Forest (baseline)
   - Gradient Boosting (mejor)

2. **Evaluar:**
   - Accuracy en test set
   - Precisión, Recall, F1

3. **Optimizar:**
   - Tuning de hiperparámetros
   - Cross-validation

---

## Tips Importantes

### ⚠️ Cuidado: Data Leakage
No usar información del futuro para predecir el pasado.

✅ CORRECTO: Usar últimos 5 partidos antes del partido actual
❌ INCORRECTO: Usar el resultado actual para calcular features

### ✅ Nuestra Solución:
Usamos `.shift(1)` para desplazar datos y evitar leakage.

### ⚠️ Valores Nulos (NaNs)
Los primeros partidos no tendrán H2H ni Form (no hay histórico).

✅ SOLUCIÓN: `fillna()` con forward fill o backward fill.

---

## Archivos Relacionados

- `src/feature_engineering.py` - Código de features
- `GUIA_FEATURES.md` - Descripción detallada
- `notebooks/01_eda_and_modeling.ipynb` - Ejecución práctica

---

## 🎯 Tu Tarea

1. Abre el notebook: `01_eda_and_modeling.ipynb`
2. Ejecuta la celda "Crear Features"
3. Ejecuta "Inspeccionar Features"
4. Cuéntame:
   - ¿Cuántas features se crearon?
   - ¿Hay NaNs?
   - ¿La forma (form) tiene valores razonables?

---

¡Vamos a ver qué features son más importantes para predecir! 📊
