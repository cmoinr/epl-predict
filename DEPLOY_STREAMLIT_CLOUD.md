# 🌐 DEPLOY EN STREAMLIT CLOUD (GRATIS)

## ¿POR QUÉ DEPLOYAR?

**Streamlit Cloud** te permite:
- ✅ Compartir tu dashboard con otros (sin instalar nada)
- ✅ URL pública: `https://epl-predict-cmoinr.streamlit.app`
- ✅ Actualizaciones automáticas (cada vez que subes a GitHub)
- ✅ Totalmente GRATIS
- ✅ Sin servidor que configurar

---

## PASO 1: Crear Cuenta GitHub (2 minutos)

### Si ya tienes GitHub:
Salta al PASO 2.

### Si NO tienes GitHub:

1. Ve a: https://github.com/signup
2. Email: Tu email
3. Password: Contraseña segura
4. Username: `tu_usuario` (ej: cmoinr)
5. Click "Create account"
6. Verifica email
7. ✅ Cuenta creada

---

## PASO 2: Subir Proyecto a GitHub (5 minutos)

### Opción A: Desde Terminal (Recomendado)

```bash
# 1. Navigate a tu proyecto
cd "c:\Users\cmoin\Documentos\epl-predict"

# 2. Inicializar git
git init

# 3. Agregar todos los archivos
git add .

# 4. Hacer commit
git commit -m "EPL Predictor - Streamlit Dashboard"

# 5. Crear rama main
git branch -M main

# 6. Agregar remote (REEMPLAZA con tu usuario)
git remote add origin https://github.com/cmoinr/epl-predict.git

# 7. Push a GitHub
git push -u origin main

# ✅ ¡Subido a GitHub!
```

### Opción B: Desde GitHub Desktop

```
1. Descarga: https://desktop.github.com/
2. Login con GitHub
3. "File" → "Clone Repository"
4. Selecciona la carpeta epl-predict
5. Hace commit y push automático
```

---

## PASO 3: Conectar a Streamlit Cloud (3 minutos)

### 1. Ve a Streamlit Cloud
```
https://share.streamlit.io/
```

### 2. Haz Login con GitHub
- Click "Sign in with GitHub"
- Autoriza Streamlit

### 3. Crear Nueva App
- Click "New app" (botón azul)

### 4. Configuración
```
Repository: cmoinr/epl-predict
Branch: main
Main file path: app.py
```

### 5. Click "Deploy"
```
La app se está compilando...
⏳ ~2 minutos
✅ ¡LISTO!
```

---

## PASO 4: Tu App en Vivo (0 minutos)

### URL de tu app:
```
https://epl-predict-cmoinr.streamlit.app
```

(Reemplaza `cmoinr` con tu usuario GitHub)

### Compartir con otros:
- Envía el link
- No necesitan instalar nada
- Solo navegador

---

## 🔄 UPDATES AUTOMÁTICOS

**Cuando hagas cambios:**

```bash
# 1. Edita app.py (ej: cambiar colores)
# 2. En terminal:
git add .
git commit -m "Mi cambio"
git push

# ✅ Automáticamente se actualiza en Streamlit Cloud
# (En 30-60 segundos)
```

---

## 📋 CHECKLIST ANTES DE DEPLOYAR

```
☑️ Tienes GitHub account
☑️ Proyecto está en GitHub
☑️ requirements.txt actualizado
☑️ app.py funciona localmente
☑️ data/raw/epl_final.csv existe
☑️ models/*.pkl existen
☑️ No hay errores en código
☑️ Cuenta Streamlit Cloud creada
```

---

## 🆘 TROUBLESHOOTING

### "❌ Deployment failed"
```
1. Revisa logs en Streamlit Cloud
2. Verifica que requirements.txt está actualizado
3. Verifica que no hay archivos faltantes
4. Intenta de nuevo
```

### "❌ Error: ModuleNotFoundError"
```
→ Falta una librería en requirements.txt
→ Agrégala: pip install [paquete]
→ Actualiza: pip freeze > requirements.txt
→ Push a GitHub → redeploy
```

### "❌ Dataset/Models no encontrados"
```
→ En Streamlit Cloud, rutas son diferentes
→ Usa rutas relativas siempre
→ Verifica que archivos están en GitHub
```

### "⏳ Muy lento en Cloud"
```
→ Normal: servidor compartido
→ Primer acceso carga modelos (~15s)
→ Siguientes accesos rápidos (<2s)
```

---

## 📊 COMPARACIÓN LOCAL vs CLOUD

| Aspecto | Local | Cloud |
|---------|-------|-------|
| **Velocidad** | Rápido (tu PC) | Medio (servidor compartido) |
| **Disponibilidad** | Solo cuando ejecutas | 24/7 |
| **Costo** | Gratis | Gratis (plan base) |
| **URL** | http://localhost:8501 | https://[nombre].streamlit.app |
| **Compartir** | Difícil (VPN) | Fácil (URL pública) |
| **Acceso** | Solo tu PC | Mundo entero |

---

## 🎯 CUÁNDO USAR CADA UNA

### **Local** (Desarrollo)
- 👨‍💻 Desarrollando
- 🔧 Haciendo cambios
- 🧪 Testeando
- 📊 Iterando rápido

### **Streamlit Cloud** (Producción/Demo)
- 👥 Compartir con otros
- 🌍 Público
- 📈 Demostración
- 🎯 Showcase

---

## ✨ PASOS RESUMIDOS (QUICK)

```
GITHUB:
1. git init
2. git add .
3. git commit -m "message"
4. git remote add origin [url]
5. git push

STREAMLIT CLOUD:
1. share.streamlit.io
2. Sign in con GitHub
3. New app
4. Selecciona repo/branch/archivo
5. Deploy
6. ✅ En vivo!
```

---

## 📝 ARCHIVO requirements.txt

Importante que esté actualizado:

```
# Core Data Science
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0

# Advanced ML
xgboost>=2.0.0
lightgbm>=4.0.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0

# Utilities
jupyter>=1.0.0
ipykernel>=6.25.0
requests>=2.31.0
python-dotenv>=1.0.0

# Web Scraping
beautifulsoup4>=4.12.0

# Web Framework - Streamlit
streamlit>=1.28.0
plotly>=5.17.0
altair>=5.0.0
```

---

## 🔐 SECRETOS (Si Necesitas)

Si tienes API keys, credenciales, etc:

### 1. Crear archivo `.streamlit/secrets.toml`

```toml
[api_keys]
odds_api_key = "tu_key_aqui"
kaggle_key = "tu_key"

[database]
db_url = "postgresql://..."
```

### 2. En app.py acceder:

```python
import streamlit as st

api_key = st.secrets["api_keys"]["odds_api_key"]
```

### 3. En Streamlit Cloud:

1. Settings → Secrets
2. Pega el contenido de secrets.toml
3. Save

---

## 🎬 MONITOREO

En Streamlit Cloud dashboard puedes ver:
- 📊 Visitors
- ⏰ Performance
- 💥 Crashes
- 🔧 Logs

---

## 💰 PRICING

| Plan | Costo | Límites |
|------|-------|---------|
| **Free** | $0 | 1 app pública, 3 apps privadas |
| **Starter** | $9/mes | 10 apps, sin límites |
| **Pro** | $29/mes | Ilimitadas + soporte |

**Para comenzar:** Free es perfecto.

---

## 🚀 PRÓXIMO PASO

Ahora que tienes Streamlit Cloud:

1. ✅ Sube tu proyecto a GitHub
2. ✅ Deployea en Streamlit Cloud
3. ✅ Comparte URL con otros
4. ✅ Recibe feedback
5. ✅ Itera rápido

---

## 📚 RECURSOS

- **Streamlit Cloud Docs**: https://docs.streamlit.io/deploy
- **GitHub Guides**: https://guides.github.com
- **Git Tutorial**: https://git-scm.com/doc

---

## ❓ PREGUNTAS FRECUENTES

### ¿Es realmente gratis?
Sí, Streamlit Cloud es completamente gratis para apps públicas.

### ¿Cuántos usuarios puede soportar?
El plan free soporta 100+ usuarios simultáneos sin problemas.

### ¿Puedo tener múltiples apps?
Sí, hasta 3 apps privadas gratis, ilimitadas en plan Starter.

### ¿Se actualizan automáticamente?
Sí, cuando subes a GitHub, automáticamente en 30-60s.

### ¿Puedo tener base de datos?
Sí, pero con plan Starter o Pro. Free usa archivos CSV.

---

**¡FELICIDADES!**

Tu dashboard está listo para ser compartido con el mundo. 🌍

---

*Hecho con ❤️ usando Streamlit + GitHub*
