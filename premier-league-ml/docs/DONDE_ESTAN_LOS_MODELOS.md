# 📍 IMPORTANTE: Ubicación de Modelos y Cómo Usar

## 🎯 Resumen Rápido de Tus 3 Preguntas

### 1️⃣ ¿Dónde están los modelos?

**Respuesta:** En archivos `.pkl` (pickle) guardados en:

```
/workspaces/codespaces-blank/premier-league-ml/models/
├── rf_result_model.pkl        (17.7 MB - Random Forest clasificación 1X2)
├── gb_result_model.pkl        (1.3 MB - Gradient Boosting clasificación 1X2)
├── rf_goals_model.pkl         (16.1 MB - Random Forest predicción goles)
├── gb_goals_model.pkl         (0.4 MB - Gradient Boosting predicción goles)
└── scaler_model.pkl           (11 KB - Normalizador de features)
```

**Verificar desde terminal:**
```bash
ls -lh models/
```

**Total:** 35.5 MB

---

### 2️⃣ ¿Cómo predecir futuros partidos?

**Opción A: Desde Terminal (Recomendado)**

```bash
python predict_match.py --home "Chelsea" --away "Liverpool" --date "2025-02-22"
```

**Opción B: Desde Notebook**

```python
from src.predictor import EPLPredictor
predictor = EPLPredictor('models')
result = predictor.predict_match(df, 'Chelsea', 'Liverpool', '2025-02-22', X_train_scaled)
predictor.print_prediction(result)
```

---

### 3️⃣ ¿Terminal o Notebook?

| Caso | Recomendación |
|------|---|
| Predicción rápida | ✅ **Terminal** |
| Análisis interactivo | ✅ **Notebook** |
| Automatización (cron) | ✅ **Terminal** |
| Visualizaciones | ✅ **Notebook** |
| Producción/Deploy | ✅ **Terminal** |

---

## 🚀 Guía de Ejecución

### Paso 1: Verificar modelos guardados

```bash
cd /workspaces/codespaces-blank/premier-league-ml
ls -lh models/
```

Deberías ver 5 archivos `.pkl`

### Paso 2A: Usar desde Terminal

```bash
# Predicción con detalles
python predict_match.py --home "Chelsea" --away "Liverpool" --date "2025-02-22"

# Solo resultado
python predict_match.py --home "Chelsea" --away "Liverpool" --date "2025-02-22" --quiet

# Múltiples predicciones
for team in Arsenal Man_City Liverpool; do
  python predict_match.py --home "$team" --away "Chelsea" --date "2025-03-01" --quiet
done
```

### Paso 2B: Usar desde Notebook

**Celda 1: Cargar predictor**
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'src'))
from predictor import EPLPredictor

predictor = EPLPredictor('models')
```

**Celda 2: Hacer predicción**
```python
result = predictor.predict_match(df, 'Chelsea', 'Liverpool', '2025-02-22', X_train_scaled)
predictor.print_prediction(result)
```

---

## 📊 Ejemplos de Salida

### Salida Normal (Terminal o Notebook)

```
======================================================================
🔮 PREDICCIÓN EPL
======================================================================
📅 Chelsea vs Liverpool (2025-02-22)
======================================================================

📊 RESULTADO (1X2):

  🌲 Random Forest:
     Predicción: Home Win
     Confianza: 71.3%
     Detalles: Away 14.4% | Draw 14.3% | Home 71.3%

  ⚡ Gradient Boosting:
     Predicción: Home Win
     Confianza: 73.9%
     Detalles: Away 6.8% | Draw 19.3% | Home 73.9%

⚽ GOLES TOTALES:
  🌲 Random Forest: 2.24
  ⚡ Gradient Boosting: 2.41
  📈 Promedio: 2.33

======================================================================
```

### Salida Quiet Mode (Terminal)

```bash
$ python predict_match.py --home "Chelsea" --away "Liverpool" --date "2025-02-22" --quiet
Home Win
```

---

## 🔧 Solución de Problemas

### Error: "ModuleNotFoundError"

```bash
# Asegurate que estás en la carpeta correcta
cd /workspaces/codespaces-blank/premier-league-ml

# Verifica que existen los archivos
ls src/predictor.py
ls src/feature_engineering.py
```

### Error: "No se encontraron modelos"

```bash
# Verifica que los modelos están en la carpeta correcta
ls models/

# Si la carpeta está vacía, ejecuta el notebook:
# Celda: "Guardar Modelos para Uso Futuro"
```

### Error: "Formato de fecha inválido"

```bash
# ✅ Correcto: YYYY-MM-DD
python predict_match.py --home "Chelsea" --away "Liverpool" --date "2025-02-22"

# ❌ Incorrecto: DD/MM/YYYY o DD-MM-YYYY
python predict_match.py --home "Chelsea" --away "Liverpool" --date "22/02/2025"
```

---

## 📚 Documentación Completa

Archivos de referencia que creé para ti:

1. **RESUMEN_MODELOS_PREDICCION.md** ← Respuestas completas a tus 3 preguntas
2. **GUIA_MODELOS_Y_PREDICCIONES.md** ← Explicación técnica profunda
3. **GUIA_TERMINAL.md** ← Ejemplos de terminal y automatización
4. **Este archivo** ← Quick reference

---

## 🎬 Comandos Más Útiles

```bash
# Ver ayuda
python predict_match.py --help

# Predicción individual
python predict_match.py --home "Chelsea" --away "Liverpool" --date "2025-02-22"

# Predicción quiet
python predict_match.py --home "Chelsea" --away "Liverpool" --date "2025-02-22" --quiet

# Con rutas personalizadas
python predict_match.py \
  --home "Chelsea" \
  --away "Liverpool" \
  --date "2025-02-22" \
  --data "data/raw/epl_final.csv" \
  --models "models/"

# Verificar modelos
ls -lh models/ | tail -5

# Verificar tamaño total
du -sh models/

# Reproducir última predicción
python predict_match.py --home "Manchester City" --away "Arsenal" --date "2025-03-01"
```

---

## 🎓 Conceptos Clave

### ¿Qué son los archivos .pkl?

Son archivos **"pickle"** de Python - serializan objetos Python completos:
- Modelos entrenados
- Vectorizadores/Scaler
- Cualquier objeto Python

**Ventajas:**
- ✅ Preservan estado exacto del modelo
- ✅ Muy rápido de cargar/guardar
- ✅ Tamaño compacto (con compresión)

**Desventajas:**
- ⚠️ Solo funcionan en Python
- ⚠️ Cambios de versión pueden romper compatibilidad

### Flujo de Predicción

```
1. Cargar modelos guardados (.pkl)
   ↓
2. Normalizar features nuevos con scaler
   ↓
3. Pasar a Random Forest → predicción RF + probabilidades
   ↓
4. Pasar a Gradient Boosting → predicción GB + probabilidades
   ↓
5. Agregar modelo de goles (RF + GB)
   ↓
6. Retornar predicciones combinadas
```

---

## 🔄 Tus Modelos en Números

| Métrica | Valor |
|---------|-------|
| **Dataset de entrenamiento** | 9,380 partidos EPL |
| **Período** | 2000-2025 |
| **Features** | ~40 derivados |
| **Modelos** | 4 (2 clasificación, 2 regresión) |
| **Accuracy (1X2)** | 62.74% (RF) / 58% (GB) |
| **R² (Goles)** | 0.5125 (RF) / 0.5157 (GB) |
| **MAE (Goles)** | 0.9654 (RF) / 0.9584 (GB) |
| **Tamaño total** | 35.5 MB |

---

## ✅ Checklist de Setup

- [x] Modelos entrenados en notebook
- [x] Modelos guardados en `/models/`
- [x] Script `predict_match.py` creado
- [x] Módulo `src/predictor.py` creado
- [x] Predicción desde notebook funciona ✅
- [x] Predicción desde terminal funciona ✅
- [x] Modo batch (múltiples partidos) funciona ✅
- [x] Modo quiet funciona ✅

---

## 🎯 Próximos Pasos (Tu Decisión)

1. **Feature Importance**: ¿Qué features son más importantes?
2. **Integración de Odds**: Comparar con probabilidades reales de mercado
3. **Value Betting**: Identificar oportunidades de ganancias
4. **Backtesting**: Simular resultados históricos
5. **Automatización**: Predicciones diarias programadas

---

**¿Preguntas? Consulta GUIA_TERMINAL.md o RESUMEN_MODELOS_PREDICCION.md** 🚀
