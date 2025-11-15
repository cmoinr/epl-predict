# 📑 ÍNDICE COMPLETO DEL PROYECTO

## 🎯 Respuestas a Tus 3 Preguntas

### ❓ Pregunta 1: ¿Dónde están los modelos?
**Respuesta corta:** En `/models/` como archivos `.pkl` (35.5 MB)
**Donde leer más:** [DONDE_ESTAN_LOS_MODELOS.md](DONDE_ESTAN_LOS_MODELOS.md)

### ❓ Pregunta 2: ¿Cómo predecir futuros partidos?
**Respuesta corta:** Terminal: `python predict_match.py --home X --away Y --date Z` | Notebook: `EPLPredictor().predict_match()`
**Donde leer más:** [RESUMEN_MODELOS_PREDICCION.md](RESUMEN_MODELOS_PREDICCION.md)

### ❓ Pregunta 3: ¿Terminal o Notebook?
**Respuesta corta:** Terminal para predicciones rápidas, Notebook para análisis
**Donde leer más:** [RESUMEN_MODELOS_PREDICCION.md](RESUMEN_MODELOS_PREDICCION.md#-pregunta-3-terminal-o-notebook)

---

## 📚 Guías por Propósito

### ⏱️ "Tengo 5 minutos"
→ Lee: [INICIO_RAPIDO.md](INICIO_RAPIDO.md)

### 🔍 "Quiero una referencia rápida"
→ Lee: [DONDE_ESTAN_LOS_MODELOS.md](DONDE_ESTAN_LOS_MODELOS.md)

### 🖥️ "Quiero usar desde Terminal"
→ Lee: [GUIA_TERMINAL.md](GUIA_TERMINAL.md)

### 📖 "Quiero explicación técnica completa"
→ Lee: [GUIA_MODELOS_Y_PREDICCIONES.md](GUIA_MODELOS_Y_PREDICCIONES.md)

### 🎓 "Quiero respuestas detalladas a mis 3 preguntas"
→ Lee: [RESUMEN_MODELOS_PREDICCION.md](RESUMEN_MODELOS_PREDICCION.md)

---

## 🚀 Ejemplos de Uso

### Terminal
```bash
# Predicción normal
python predict_match.py --home "Chelsea" --away "Liverpool" --date "2025-02-22"

# Modo quiet
python predict_match.py --home "Chelsea" --away "Liverpool" --date "2025-02-22" --quiet

# Con rutas personalizadas
python predict_match.py --home "Arsenal" --away "Man City" --date "2025-03-01" --data "ruta/dataset.csv" --models "ruta/models/"
```

### Notebook
```python
from src.predictor import EPLPredictor
predictor = EPLPredictor('models')
result = predictor.predict_match(df, 'Chelsea', 'Liverpool', '2025-02-22', X_train_scaled)
predictor.print_prediction(result)
```

### Python Script
```python
import sys
sys.path.insert(0, 'src')
from predictor import EPLPredictor
import pandas as pd

df = pd.read_csv('data/raw/epl_final.csv')
predictor = EPLPredictor('models')
result = predictor.predict_match(df, 'Arsenal', 'Man City', '2025-03-01', None)
print(result)
```

---

## 📁 Estructura del Proyecto

```
premier-league-ml/
├── 📂 data/
│   ├── 📂 raw/
│   │   └── epl_final.csv              (Dataset: 9,380 × 25)
│   └── 📂 processed/
│
├── 📂 models/                         (MODELOS GUARDADOS)
│   ├── rf_result_model.pkl            (17.7 MB)
│   ├── gb_result_model.pkl            (1.3 MB)
│   ├── rf_goals_model.pkl             (16.1 MB)
│   ├── gb_goals_model.pkl             (0.4 MB)
│   └── scaler_model.pkl               (11 KB)
│
├── 📂 notebooks/
│   └── 01_eda_and_modeling.ipynb      (Notebook principal)
│
├── 📂 src/
│   ├── predictor.py                   (Módulo de predicción)
│   ├── feature_engineering.py         (Ingeniería de features)
│   ├── data_collection.py             (Colección de datos)
│   ├── odds_api.py                    (Integración de odds)
│   └── utils.py                       (Utilidades)
│
├── 📄 predict_match.py                (Script terminal)
│
├── 📄 INICIO_RAPIDO.md                (5 minutos)
├── 📄 DONDE_ESTAN_LOS_MODELOS.md      (Quick reference)
├── 📄 RESUMEN_MODELOS_PREDICCION.md   (Respuestas completas)
├── 📄 GUIA_MODELOS_Y_PREDICCIONES.md  (Guía técnica)
├── 📄 GUIA_TERMINAL.md                (Ejemplos terminal)
├── 📄 INDEX.md                        (Este archivo)
└── 📄 README.md                       (Visión general)
```

---

## 🎯 Rendimiento de Modelos

| Métrica | Random Forest | Gradient Boosting |
|---------|---------------|-------------------|
| **Accuracy (1X2)** | 62.74% ✅ | 58.00% |
| **F1-Score (1X2)** | 0.5805 | 0.5983 |
| **R² (Goles)** | 0.5125 | 0.5157 ✅ |
| **MAE (Goles)** | 0.9654 | 0.9584 ✅ |
| **RMSE (Goles)** | 1.1882 | 1.1843 ✅ |

---

## ✅ Checklist de Setup

- [x] Modelos entrenados
- [x] Modelos guardados en disco
- [x] Script `predict_match.py` creado
- [x] Módulo `src/predictor.py` creado
- [x] Predicción desde Terminal funciona
- [x] Predicción desde Notebook funciona
- [x] Predicciones batch funciona
- [x] Documentación completa

---

## 🔄 Flujo de Predicción

```
1. INPUT: Equipo local, visitante, fecha
           ↓
2. CARGAR: Modelos guardados (.pkl)
           ↓
3. NORMALIZAR: Features con scaler
           ↓
4. PREDECIR: 
   • Random Forest (1X2)
   • Gradient Boosting (1X2)
   • Random Forest (Goles)
   • Gradient Boosting (Goles)
           ↓
5. OUTPUT: Predicción + Probabilidades + Confianza
```

---

## 🚀 Próximos Pasos

1. **Feature Importance**: ¿Cuáles features son más importantes?
2. **Integración de Odds**: Conectar APIs de odds reales
3. **Value Betting**: Identificar oportunidades de ganancia
4. **Backtesting**: Simular resultados históricos
5. **Automatización**: Predicciones diarias programadas

---

## 📊 Datos del Proyecto

| Aspecto | Valor |
|--------|-------|
| **Período de datos** | 2000-2025 |
| **Total de partidos** | 9,380 |
| **Features creados** | ~40 |
| **Modelos entrenados** | 4 (2 clasificación, 2 regresión) |
| **Tamaño modelos** | 35.5 MB |
| **Dataset** | 9,380 × 25 columnas |
| **Train/Test split** | 80% / 20% (temporal) |

---

## 🆘 Troubleshooting Rápido

| Problema | Solución |
|----------|----------|
| "Modelos no encontrados" | `ls models/` - Verifica que existan archivos .pkl |
| "Module not found" | `pip install scikit-learn pandas numpy` |
| "Formato de fecha inválido" | Usa `YYYY-MM-DD` (ej: `2025-02-22`) |
| "Dataset no encontrado" | Verifica ruta en `data/raw/epl_final.csv` |

---

## 📞 Soporte

Para más información, consulta:
- [INICIO_RAPIDO.md](INICIO_RAPIDO.md) - Inicio rápido
- [DONDE_ESTAN_LOS_MODELOS.md](DONDE_ESTAN_LOS_MODELOS.md) - Referencia rápida
- [GUIA_TERMINAL.md](GUIA_TERMINAL.md) - Uso desde terminal
- [RESUMEN_MODELOS_PREDICCION.md](RESUMEN_MODELOS_PREDICCION.md) - Respuestas detalladas

---

**Última actualización:** 2025-11-15
**Estado:** Producción ✅
