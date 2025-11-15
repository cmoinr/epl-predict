# 🚀 Setup Local - Premier League ML Predictor

## 📋 Archivos Cruciales para Ejecutar Localmente

### Ignorados en Git (Descargar/Generar Localmente)

| Archivo | Descripción | Cómo Obtener |
|---------|------------|-------------|
| `data/raw/epl_final.csv` | Dataset EPL histórico (~9410 partidos) | Ver `docs/INDEX.md` - Descargar de Kaggle |
| `models/random_forest_model.pkl` | Modelo Random Forest entrenado | Ejecutar `python src/train_models.py` |
| `models/gradient_boosting_model.pkl` | Modelo Gradient Boosting entrenado | Ejecutar `python src/train_models.py` |
| `.env` | Variables de entorno (API keys, paths) | Crear localmente, ver sección abajo |

---

## ⚡ Quick Start Local

### 1️⃣ Clonar y Setup

```bash
# Clonar
git clone <repo-url>
cd premier-league-ml

# Crear virtual env
python -m venv venv
source venv/bin/activate  # Linux/Mac
# source venv/Scripts/activate  # Windows

# Instalar dependencias
pip install -r requirements.txt
```

### 2️⃣ Descargar Datos

```bash
# Opción A: Descargar manualmente
# 1. Ir a Kaggle: https://www.kaggle.com/datasets/rishabhgl/english-premier-league-dataset
# 2. Descargar `epl_final.csv`
# 3. Guardar en: data/raw/epl_final.csv

# Opción B: Usar script (si tienes Kaggle API)
python docs/archived/SETUP_KAGGLE.md
```

### 3️⃣ Entrenar Modelos

```bash
python src/train_models.py
```

Esto genera:
- `models/random_forest_model.pkl`
- `models/gradient_boosting_model.pkl`

### 4️⃣ Hacer Predicciones

```bash
# Predicción para un partido específico
python predict_match.py --home Chelsea --away Arsenal --date 2025-12-07

# Análisis completo con odds
python run_analysis.py
```

---

## 📁 Estructura de Archivos en Git

```
root/
├── README.md                          ✅ Documentación principal
├── requirements.txt                   ✅ Dependencias (pip install)
├── .gitignore                         ✅ Qué ignorar en commits
├── predict_match.py                   ✅ Script para hacer predicciones
├── run_analysis.py                    ✅ Script integrado: Predicción + Odds
│
├── src/                               ✅ Código principal
│   ├── predictor.py                  ✅ Clase predictor
│   ├── odds_comparison.py            ✅ Análisis de odds
│   ├── train_models.py               ✅ Entrenar modelos
│   ├── feature_engineering.py        ✅ Ingeniería de features
│   └── ...
│
├── data/
│   ├── raw/                          ❌ Ignorado (descargar manualmente)
│   │   └── epl_final.csv            ❌ CSV grande (~100MB)
│   └── processed/                    ❌ Ignorado (datos temporales)
│       └── sample_odds.csv          ✅ Ejemplo odds (pequeño)
│
├── models/                           ❌ Ignorado (generar localmente)
│   ├── random_forest_model.pkl      ❌ Modelo entrenado
│   └── gradient_boosting_model.pkl  ❌ Modelo entrenado
│
├── notebooks/                        ✅ Jupyter notebooks para análisis
│   └── 01_eda_and_modeling.ipynb    ✅ EDA + Modelado
│
├── docs/                            ✅ Documentación
│   ├── INDEX.md                     ✅ Guía de documentación
│   ├── GUIA_ODDS_INTEGRATION.md    ✅ Cómo usar odds
│   ├── GUIA_MODELOS_Y_PREDICCIONES.md ✅ Entrenar y predecir
│   ├── INICIO_RAPIDO.md            ✅ Setup inicial
│   └── archived/                    📦 Docs anteriores (referencia)
│
└── examples/                        📚 Ejemplos y demostraciones
    ├── demo_odds_comparison.py      📚 Ejemplos de comparación
    ├── demo_value_betting.py        📚 Ejemplos de value betting
    ├── analyze_predictions_vs_odds.py 📚 Análisis batch
    └── integrate_model_with_odds.py 📚 Integración modelo+odds
```

---

## 📌 Documentación Esencial

Leer en este orden:

1. **`README.md`** - Visión general del proyecto
2. **`docs/INICIO_RAPIDO.md`** - Setup rápido
3. **`docs/GUIA_MODELOS_Y_PREDICCIONES.md`** - Cómo entrenar y predecir
4. **`docs/GUIA_ODDS_INTEGRATION.md`** - Comparar predicciones con odds

---

## 🔧 Variables de Entorno (.env)

Crear archivo `.env` en raíz (si necesitas):

```bash
# Rutas
RAW_DATA_PATH=data/raw/epl_final.csv
PROCESSED_DATA_PATH=data/processed/
MODELS_PATH=models/

# Kaggle (opcional, para descargar datos)
KAGGLE_USERNAME=tu_usuario
KAGGLE_KEY=tu_key_api

# Análisis
MIN_EDGE=0.03
MIN_EV=0.10
MIN_CONFIDENCE=0.50
```

---

## 🎯 Comandos Principales

```bash
# Entrenar modelos
python src/train_models.py

# Predicción individual
python predict_match.py --home Chelsea --away Arsenal

# Análisis con odds
python run_analysis.py

# Demostraciones
python examples/demo_odds_comparison.py
python examples/demo_value_betting.py

# Análisis batch
python examples/analyze_predictions_vs_odds.py
```

---

## ✅ Checklist para Setup Local

- [ ] Git clone + virtual env + pip install -r requirements.txt
- [ ] Descargar `epl_final.csv` → `data/raw/`
- [ ] Ejecutar `python src/train_models.py`
- [ ] Verificar modelos en `models/`
- [ ] Probar: `python predict_match.py --home Chelsea --away Arsenal`
- [ ] Configurar CSV de odds en `data/processed/sample_odds.csv`
- [ ] Ejecutar: `python run_analysis.py`
- [ ] ✅ Listo para usar

---

## 📚 Docs Archivados

Para referencia histórica, ver `docs/archived/`:
- Guías de features
- Diagnósticos anteriores
- Documentación de mejoras antiguas

---

## 🆘 Troubleshooting

**Error: "No such file or directory: data/raw/epl_final.csv"**
→ Descargar dataset de Kaggle y guardar en esa ruta

**Error: "Model files not found"**
→ Ejecutar `python src/train_models.py` primero

**Error: "ModuleNotFoundError"**
→ Verificar que está activado el venv y `pip install -r requirements.txt`

---
