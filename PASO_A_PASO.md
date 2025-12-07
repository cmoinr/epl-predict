# 🎬 PASO A PASO: EJECUTAR STREAMLIT DASHBOARD

## VIDEO EN TEXTO: Cómo ejecutar el dashboard en 5 minutos

---

## ⏱️ TIEMPO TOTAL: 5 MINUTOS

```
[0:00] - Abrir terminal/bash
[0:30] - Navegar a proyecto
[1:00] - Ejecutar streamlit run app.py
[2:00] - Dashboard abre en navegador
[2:30] - Seleccionar equipos
[3:00] - Click "PREDECIR PARTIDO"
[3:30] - Ver resultados
[4:00] - Explorar gráficos
[4:30] - Probar otro partido
[5:00] - ¡Listo!
```

---

## 📋 PASO A PASO DETALLADO

### PASO 1: Abrir Terminal (30 segundos)

#### Windows:
```
1. Abre VS Code
2. Presiona: Ctrl + `
3. Terminal abierta ✓
```

#### Mac/Linux:
```
1. Abre Terminal (Cmd + Space, escribe "Terminal")
2. Terminal abierta ✓
```

### PASO 2: Navegar al Proyecto (30 segundos)

```bash
# Copia-pega en terminal:
cd "c:\Users\cmoin\Documentos\epl-predict"

# Verifica que estés en la carpeta correcta:
ls
# Deberías ver: app.py, data/, models/, src/, etc.
```

### PASO 3: Ejecutar Streamlit (30 segundos)

```bash
# Ejecuta:
streamlit run app.py

# Verás algo como:
# ⓘ  To view your app on a browser, open this URL:
# 
#   http://localhost:8501
#
# ⓘ  Session state does not persist after app rerun
# ...
```

**El navegador abre automáticamente.** Si no:
- Ve manualmente a: `http://localhost:8501`

### PASO 4: Dashboard Carga (10-15 segundos)

```
[PRIMERA VEZ]
⏳ Cargando...
  • Streamlit inicia
  • Load de modelos (~8 segundos)
  • Renderiza UI (~2 segundos)
  ✅ LISTO
```

### PASO 5: Usar el Dashboard (30 segundos)

```
1. Mira el SIDEBAR (izquierda)
   └─ Ves: "🏠 Equipo Local", "✈️ Equipo Visitante", "📅 Fecha"

2. Click en "🏠 Equipo Local"
   └─ Se abre dropdown

3. Selecciona un equipo (ej: Chelsea)
   └─ Click en "Chelsea"

4. Click en "✈️ Equipo Visitante"
   └─ Se abre dropdown

5. Selecciona otro equipo (ej: Liverpool)
   └─ Click en "Liverpool"

6. Verifica la fecha en "📅 Fecha del partido"
   └─ Usa la fecha sugerida o cambia

7. ¡AHORA VIENE LO DIVERTIDO!
```

### PASO 6: Hacer Predicción (5 segundos)

```
BUSCA EL BOTÓN AZUL GRANDE EN EL SIDEBAR:

    ┌──────────────────────┐
    │  🔮 PREDECIR PARTIDO │
    └──────────────────────┘
    
    ↓ HAZLE CLICK ↓

Verás:
🔄 Cargando modelos...
🔮 Prediciendo Chelsea vs Liverpool...

(Espera ~2-3 segundos)
```

### PASO 7: ¡VER RESULTADOS! (60 segundos)

**El dashboard se llena de información:**

```
┌─────────────────────────────────────────────┐
│  Chelsea vs Liverpool                       │
│  2025-12-07                                 │
├─────────────────────────────────────────────┤
│                                             │
│  📊 PROBABILIDADES PREDICHAS                │
│  ┌──────────┬──────────┬──────────┐       │
│  │ Home Win │  Draw    │Away Win  │       │
│  │   65%    │   20%    │   15%    │       │
│  │ (Gauge)  │ (Gauge)  │ (Gauge)  │       │
│  └──────────┴──────────┴──────────┘       │
│                                             │
│  🔬 DETALLES TÉCNICOS                       │
│  [Random Forest] [Gradient Boosting] [...]  │
│                                             │
│  Random Forest:                             │
│  Predicción: Home Win                       │
│  Confianza: 71.3%                           │
│  • Home: 65% ████████████                  │
│  • Draw: 20% ████                          │
│  • Away: 15% ███                           │
│                                             │
│  💰 ANÁLISIS VALUE BETTING                  │
│  [Tabla con odds y recomendaciones]        │
│                                             │
│  📋 DATOS COMPLETOS (expandible)            │
│                                             │
└─────────────────────────────────────────────┘
```

**EXPLORA:**
- 👆 Click en cada TAB (Random Forest, Gradient Boosting)
- 👆 Expande "DATOS COMPLETOS" para ver JSON
- 👆 Scroll down para ver más gráficos

### PASO 8: Probar Otro Partido (30 segundos)

**Vuelve al SIDEBAR:**

```
1. Cambia "Equipo Local" a otro (ej: Man City)
2. Cambia "Equipo Visitante" a otro (ej: Arsenal)
3. Click en "🔮 PREDECIR PARTIDO"
4. ¡Nota que es MUCHO MÁS RÁPIDO! (~1 segundo)
   Porque los modelos ya están cacheados
```

### PASO 9: Detener la App (5 segundos)

**Para parar el dashboard:**

```bash
# En la terminal, presiona:
Ctrl + C

# Verás:
# ^C
# Stopping...
# Shutdown complete
```

---

## 🎯 RESUMEN VISUAL

```
INICIO (Terminal cerrada)
  ↓
  ├─ Abrir terminal
  │
  ├─ Navegar: cd c:\Users\cmoin\Documentos\epl-predict
  │
  ├─ Ejecutar: streamlit run app.py
  │
  └─ Esperar: ~3 segundos
  
NAVEGADOR ABRE
  ↓
  ├─ Selecciona equipos y fecha
  │
  ├─ Click "🔮 PREDECIR PARTIDO"
  │
  └─ Esperar: ~2-3 segundos
  
RESULTADOS APARECEN
  ↓
  ├─ Ve probabilidades (Gauges)
  ├─ Explora detalles de modelos
  ├─ Lee análisis value betting
  └─ Expande datos JSON

REPITE
  ↓
  └─ Selecciona otro partido (RÁPIDO - 1s)

DETENER
  ↓
  └─ Ctrl + C en terminal
```

---

## ⚡ ALTERNATIVAS RÁPIDAS

### Si no quieres usar terminal:

#### OPCIÓN 1: Double-click (Windows)
```
1. Ve a: c:\Users\cmoin\Documentos\epl-predict
2. Busca: run_streamlit.bat
3. Double-click
4. ¡Dashboard abre automáticamente!
5. Espera 3-5 segundos a que cargue
```

#### OPCIÓN 2: Desde VS Code
```
1. Abre app.py en VS Code
2. Click derecho en app.py
3. "Run Python File"
4. O: Ctrl + F5
```

---

## 🐛 TROUBLESHOOTING RÁPIDO

### "❌ No se abre navegador"
```
→ Abre manualmente: http://localhost:8501
```

### "❌ Error: ModuleNotFoundError"
```bash
→ Reinstala dependencias:
pip install -r requirements.txt
```

### "❌ Dataset no encontrado"
```
→ Verifica que existe: data/raw/epl_final.csv
→ Si no, descargalo de Kaggle
```

### "⏳ Muy lento"
```
→ Normal en primer acceso (carga modelos)
→ Siguiente acceso será rápido
→ Si persiste, reinicia la app: Ctrl+C → streamlit run app.py
```

---

## 📸 PANTALLAZOS QUE VERÁS

### PASO 1: Inicial
```
⚽ EPL PREDICTOR
Predictor inteligente de resultados Premier League

👋 Bienvenido al EPL Predictor
Selecciona dos equipos y una fecha...

📊 DATASET STATISTICS:
├─ 9,380 Partidos
├─ 20 Equipos
├─ 2000 - 2025 Años
└─ 25 Features
```

### PASO 2: Selectores llenos
```
SIDEBAR:
🏠 Equipo Local: [Chelsea ✓]
✈️ Equipo Visitante: [Liverpool ✓]
📅 Fecha: 2025-12-07

[🔮 PREDECIR PARTIDO] ← CLICK AQUÍ
```

### PASO 3: Cargando
```
⏳ Cargando modelos...
🔮 Prediciendo Chelsea vs Liverpool...
```

### PASO 4: Resultados
```
Chelsea vs Liverpool
2025-12-07

📊 PROBABILIDADES PREDICHAS
[Gauge 65%] [Gauge 20%] [Gauge 15%]

🔬 DETALLES TÉCNICOS
[RF] [GB] [Goles]

[Tabs con info detallada]

💰 ANÁLISIS VALUE BETTING
[Tabla interactiva]

📋 DATOS COMPLETOS
[Expandible: JSON]
```

---

## ✅ CHECKLIST FINAL

Antes de ejecutar, verifica:

```
☑️ Terminal abierta
☑️ Estoy en la carpeta correcta (cd epl-predict)
☑️ Veo: app.py, data/, models/, src/
☑️ Python está instalado (python --version)
☑️ Streamlit instalado (pip list | grep streamlit)
☑️ Dataset existe (data/raw/epl_final.csv)
☑️ Modelos existen (models/*.pkl)
☑️ Puedo ejecutar: streamlit run app.py
☑️ Navegador abre en: http://localhost:8501
☑️ ¡READY TO PREDICT! 🚀
```

---

## 🎉 ¡LISTO!

Ya tienes un **dashboard profesional de predicción de fútbol** completamente funcional.

**Próximos pasos:**
1. ✅ Ejecuta `streamlit run app.py`
2. ✅ Prueba con diferentes equipos
3. ✅ Experimenta con los gráficos
4. ✅ Entiende las predicciones
5. ✅ Considera deployar en Streamlit Cloud

---

## 📞 AYUDA

Si algo no funciona:
1. Lee el error en la terminal
2. Busca en: https://discuss.streamlit.io
3. Verifica: README_STREAMLIT.md
4. Lee: GUIA_STREAMLIT_RAPIDA.md

¡Diviértete prediciendo! ⚽🔮
