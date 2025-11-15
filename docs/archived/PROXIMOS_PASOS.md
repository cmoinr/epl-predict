# Próximos Pasos - Premier League ML

## 📊 Tu Proyecto Está Configurado

Has elegido:
- ✅ **Dataset**: EPL Match Data 2000-2025 (máximo histórico disponible)
- ✅ **Predicciones**: Resultado (1X2) + Goles Totales
- ✅ **Objetivo**: Identificar value bets rentables

---

## 🎯 Plan de Acción Inmediato

### PASO 1️⃣: Obtener el Dataset (AHORA)

Ejecuta en terminal desde la carpeta del proyecto:

```bash
bash setup_data.sh
```

Esto verificará si tienes `epl_final.csv`. Si no lo tienes:

**Opción A - Manual (Más rápido para empezar)**
1. Ir a: https://www.kaggle.com/datasets (buscar "English Premier League")
2. Descargar `epl_final.csv`
3. Guardar en: `/data/raw/epl_final.csv`

**Opción B - Kaggle CLI**
```bash
# Instalar CLI
pip install kaggle

# Descargar (requiere ~/kaggle/kaggle.json)
kaggle datasets download -d vivovinco/english-premier-league-matches
unzip -d data/raw/
```

---

### PASO 2️⃣: Ejecutar Análisis Exploratorio (EDA)

Una vez tengas el CSV en `data/raw/`:

```bash
jupyter notebook notebooks/01_eda_and_modeling.ipynb
```

En el notebook:
1. Carga el dataset
2. Explora estructura y columnas
3. Visualiza distribuciones
4. Identifica target variables (Resultado, Goles)

**¿Qué buscar?**
- ✓ Estructura de columnas (fecha, equipos, resultado, goles, etc.)
- ✓ Valores nulos y outliers
- ✓ Rango temporal disponible
- ✓ Distribución de resultados (%) y goles

---

### PASO 3️⃣: Preparar Features

El notebook tiene celdas para:
- Crear features de **form** (últimos 5 partidos)
- Calcular **head-to-head histórico**
- Extraer **features temporales** (día de semana, mes, season)
- Normalizar y escalar datos

---

### PASO 4️⃣: Entrenar Modelos

Usaremos:

**Para Resultado (Clasificación 3-clases: 1X2)**
- Random Forest Classifier
- Gradient Boosting (XGBoost/LightGBM)
- Métricas: Accuracy, Precision, Recall, F1, ROC-AUC

**Para Goles Totales (Regresión)**
- Random Forest Regressor
- Gradient Boosting Regressor
- Métricas: MAE, RMSE, R²

**Validación**:
- Train/Test split respetando orden temporal
- Cross-validation para robustez
- Evitar data leakage

---

### PASO 5️⃣: Comparar con Odds del Mercado

Para esto necesitamos **odds históricas**. Opciones:

**Opción A**: Kaggle (algunos datasets incluyen odds)
- Buscar "football odds" en Kaggle
- Algunos datasets EPL incluyen odds de apuestas

**Opción B**: APIs Gratuitas (para datos futuros)
- `odds-api.com` (500 requests/día gratis)
- `football-data.org` (API con odds históricas)
- RapidAPI (múltiples endpoints)

**Opción C**: Web Scraping (avanzado)
- Datos históricos de sitios especializados

---

## 📈 Estructura del Proyecto

```
premier-league-ml/
├── data/
│   ├── raw/
│   │   └── epl_final.csv          ← Colocar aquí
│   └── processed/
│       └── (datos limpios)
├── notebooks/
│   ├── 01_eda_and_modeling.ipynb  ← Ejecutar primero
│   └── (análisis adicional)
├── src/
│   ├── data_collection.py
│   ├── odds_api.py               ← Info sobre APIs
│   ├── feature_engineering.py    ← Features (próximamente)
│   ├── models.py                 ← Modelos (próximamente)
│   └── utils.py
├── models/
│   └── (modelos entrenados)
└── README.md
```

---

## 🔍 Investigación de APIs de Odds

He creado `src/odds_api.py` con info sobre:

1. **odds-api.com** (RECOMENDADO)
   - 500 requests/día gratis
   - Datos de múltiples casas de apuestas
   - Setup sencillo

2. **football-data.org**
   - Datos históricos con odds
   - 10 requests/min gratis
   - API robusta y documentada

3. **RapidAPI**
   - Múltiples APIs en una plataforma
   - Planes gratuitos generosos

Ver: `python src/odds_api.py` para más detalles

---

## 💰 Value Betting Strategy

Una vez tengas predicciones + odds:

```
Prob_implícita = 1 / Odd
Prob_modelo = Predicción del modelo ML

Value = Prob_modelo - Prob_implícita

Si Value > 3-5% → Posible buena apuesta

ROI = (Apuestas Ganadoras × Odd) / Apuestas Totales - 1
```

---

## ⏱️ Timeline Estimado

| Fase | Tiempo | Dependencias |
|------|--------|--------------|
| Obtener datos | 15 min | Conexión a internet |
| EDA | 30 min | Dataset cargado |
| Features | 1-2 hrs | Estructura clara |
| Modelos | 2-4 hrs | Features listos |
| Odds API | 1 hr | Decidir qué API usar |
| Value Betting | 1-2 hrs | Predicciones + Odds |

**Total**: 1-2 semanas dependiendo de dedicación

---

## 🎓 Decisiones que Tomaremos

Mientras avanzas, iremos decidiendo:

- [ ] ¿Qué features usar? (form, xG, lesiones, etc.)
- [ ] ¿Threshold mínimo de edge para apostar?
- [ ] ¿Qué casas de apuestas incluir?
- [ ] ¿Estrategia: conservadora o agresiva?
- [ ] ¿Backtesting histórico o forward-testing?

---

## 📚 Recursos Útiles

- **Kaggle**: https://www.kaggle.com/datasets (busca "premier league")
- **odds-api**: https://www.odds-api.com/
- **football-data**: https://www.football-data.org/
- **Scikit-learn docs**: https://scikit-learn.org/

---

## 🚀 Próximo: Descarga el CSV y Corre el Notebook

```bash
# 1. Coloca epl_final.csv en data/raw/
# 2. Ejecuta EDA
jupyter notebook notebooks/01_eda_and_modeling.ipynb
```

¡Cuéntame qué ves en el dataset! 📊

