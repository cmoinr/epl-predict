# 🚀 EPL PREDICTOR - STREAMLIT DASHBOARD

## ⚡ Quick Start (2 minutos)

### 1️⃣ Instalar dependencias (si no lo hiciste)

```bash
pip install -r requirements.txt
```

### 2️⃣ Ejecutar la app localmente

```bash
streamlit run app.py
```

Se abrirá en tu navegador en `http://localhost:8501`

### 3️⃣ Usar el dashboard

- **Sidebar izquierdo:** Selecciona equipos y fecha
- **Botón "PREDECIR PARTIDO":** Ejecuta la predicción
- **Resultado:** Ver probabilidades, modelos y análisis

---

## 🌐 Deploy en Streamlit Cloud (Gratis)

### Requisitos:
- GitHub account (free en https://github.com/signup)
- Streamlit Cloud account (free en https://streamlit.io/cloud)

### Pasos:

#### 1. Subir tu proyecto a GitHub

```bash
# Si no tienes repo
git init
git add .
git commit -m "Initial commit: EPL Predictor Streamlit"
git branch -M main
git remote add origin https://github.com/TU_USUARIO/epl-predict.git
git push -u origin main
```

#### 2. Conectar a Streamlit Cloud

1. Ve a https://share.streamlit.io/
2. Haz login con GitHub
3. Click en "New App"
4. Selecciona:
   - **Repository:** TU_USUARIO/epl-predict
   - **Branch:** main
   - **Main file path:** app.py
5. Click "Deploy"

**¡Listo!** Tu app estará en vivo en URL como:
```
https://epl-predict-cmoinr.streamlit.app
```

---

## 📊 Estructura de la App

```
app.py (Principal)
├── 🔧 CONFIG STREAMLIT
│   └── Set page config, tema, CSS
│
├── ⚙️ FUNCIONES CON CACHING
│   ├── load_predictor() → Cargar modelos (una sola vez)
│   ├── load_data() → Dataset histórico (caché)
│   └── load_odds_data() → Odds de ejemplo
│
├── 🎨 SIDEBAR (Inputs)
│   ├── Equipo Local
│   ├── Equipo Visitante
│   ├── Fecha del partido
│   └── Botón PREDECIR
│
└── 📈 MAIN CONTENT (Resultados)
    ├── Resumen partido
    ├── Gráficos de probabilidades (gauges)
    ├── Random Forest details
    ├── Gradient Boosting details
    ├── Goles & BTTS
    ├── Análisis Value Betting
    └── Datos JSON completos
```

---

## 🎯 Características Principales

✅ **Predicción Dual**
- Random Forest
- Gradient Boosting
- Promedio de ambos

✅ **Visualizaciones**
- Gauge charts para probabilidades
- Bar charts para comparación
- Métricas en tiempo real

✅ **Análisis Completo**
- Resultado (1X2)
- Goles totales
- BTTS (Both Teams to Score)
- Value betting metrics

✅ **Performance**
- Caching automático (modelos + datos)
- Hot reload en cambios de código
- Load time < 2 segundos

---

## 🔧 Configuración Avanzada

### Theme personalizado (`.streamlit/config.toml`)

El dashboard usa:
- Color primario: Azul (`#667eea`)
- Fondo: Blanco limpio
- Tipografía: Sans serif moderna

Puedes personalizar en `.streamlit/config.toml`

### Environment Variables (Opcional)

```bash
# .env (crear en root si usas APIs)
MODELS_PATH=models
DATA_PATH=data/raw/epl_final.csv
ODDS_API_KEY=tu_key_aqui
```

---

## 📱 Responsive & Mobile-Friendly

- ✅ Funciona en desktop, tablet y móvil
- ✅ Layout se ajusta automáticamente
- ✅ Controles touch-friendly

---

## 🐛 Troubleshooting

### "❌ Dataset no encontrado"
```bash
# Verifica que existe:
ls data/raw/epl_final.csv

# Si no existe, descargalo de Kaggle:
# https://www.kaggle.com/datasets/rishabhgl/english-premier-league-dataset
```

### "❌ Modelos no encontrados"
```bash
# Entrena los modelos primero:
python src/train_models.py
```

### "⏳ La app es lenta"
- Primero acceso carga modelos (~10 segundos)
- Después son rápidas (<2 segundos)
- Usa caché agresivamente

### "No aparecen resultados"
1. Verifica que equipos existen en dataset
2. Usa nombres exactos: "Chelsea", "Liverpool", etc.
3. Revisa la consola por errores

---

## 📈 Próximas Mejoras

- [ ] Integración con APIs de odds en vivo
- [ ] WebSockets para updates en tiempo real
- [ ] Historial de predicciones
- [ ] Download de reportes en PDF
- [ ] Comparativa histórica de precisión
- [ ] Más modelos (LightGBM, XGBoost ensemble)
- [ ] Estadísticas avanzadas por equipo

---

## 📚 Documentación Adicional

- Streamlit docs: https://docs.streamlit.io
- API reference: https://docs.streamlit.io/library/api-reference
- Streamlit components: https://streamlit.io/components

---

## ⭐ Tips & Tricks

### Keyboard Shortcuts
- `R` = Rerun app
- `C` = Clear cache
- `I` = Info
- `V` = Toggle verbose logging

### Optimizaciones
```python
# Usar @st.cache_resource para objetos grandes
@st.cache_resource
def load_predictor():
    return EPLPredictor('models')

# Usar @st.cache_data para datos que cambian
@st.cache_data
def load_data():
    return pd.read_csv('data.csv')
```

---

## 🤝 Contribuir

¿Quieres mejorar el dashboard?

1. Fork el repo
2. Crea rama: `git checkout -b feature/mi-mejora`
3. Commit: `git commit -am 'Agrego mi mejora'`
4. Push: `git push origin feature/mi-mejora`
5. Pull Request

---

**Preguntas?** Abre un issue en GitHub.

---

*Made with ❤️ using Streamlit*
