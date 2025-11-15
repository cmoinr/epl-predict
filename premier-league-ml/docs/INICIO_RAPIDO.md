# ⚡ INICIO RÁPIDO (5 Minutos)

## Tus 3 Preguntas - Respuestas Cortas

### ❓ Pregunta 1: ¿Dónde están los modelos?

**Respuesta:** En `/models/` como archivos `.pkl` (35.5 MB total)

```bash
ls -lh models/
```

### ❓ Pregunta 2: ¿Cómo predecir futuros partidos?

**Desde Terminal (Recomendado):**
```bash
python predict_match.py --home "Chelsea" --away "Liverpool" --date "2025-02-22"
```

**Desde Notebook:**
```python
from src.predictor import EPLPredictor
predictor = EPLPredictor('models')
result = predictor.predict_match(df, 'Chelsea', 'Liverpool', '2025-02-22', X_train_scaled)
predictor.print_prediction(result)
```

### ❓ Pregunta 3: ¿Terminal o Notebook?

| Situación | Respuesta |
|-----------|----------|
| Predicción rápida | ✅ Terminal |
| Análisis interactivo | ✅ Notebook |
| Automatización | ✅ Terminal |
| Visualizaciones | ✅ Notebook |
| Producción | ✅ Terminal |

---

## 🚀 Usar Ahora (Copia-Pega)

### Opción 1: Terminal (30 segundos)
```bash
cd /workspaces/codespaces-blank/premier-league-ml
python predict_match.py --home "Arsenal" --away "Man City" --date "2025-03-01"
```

### Opción 2: Notebook (Ya ejecutado)
- Ve a celda: "Hacer Predicciones en Nuevos Partidos"
- Ya tiene predicción ejemplo lista
- Modifica equipos/fechas según necesites

### Opción 3: Script Python Custom
```python
import sys
sys.path.insert(0, 'src')
from predictor import EPLPredictor
import pandas as pd

df = pd.read_csv('data/raw/epl_final.csv')
predictor = EPLPredictor('models')

# Tu predicción
result = predictor.predict_match(df, 'Chelsea', 'Liverpool', '2025-02-22', None)
print(f"Predicción: {result['resultado']['random_forest']['prediccion']}")
print(f"Goles: {result['goles_totales']['promedio']}")
```

---

## 📚 Documentación Disponible

- **DONDE_ESTAN_LOS_MODELOS.md** ← Respuestas a 3 preguntas
- **RESUMEN_MODELOS_PREDICCION.md** ← Explicación completa
- **GUIA_TERMINAL.md** ← Ejemplos y automatización
- **GUIA_MODELOS_Y_PREDICCIONES.md** ← Técnica profunda

---

## ✅ Estado Final

```
✅ Modelos entrenados
✅ Modelos guardados (35.5 MB, 5 archivos)
✅ Script terminal funciona
✅ Módulo Python funciona
✅ Predicciones desde notebook funcionan
✅ Documentación completa
```

---

¿Próximo paso? Ver: **RESUMEN_MODELOS_PREDICCION.md** 📖
