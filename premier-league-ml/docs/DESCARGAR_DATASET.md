# 🎯 GUÍA RÁPIDA: DESCARGAR DATASET EPL

## Opción 1: Descarga Manual (Más Rápida) ⭐

### Paso 1: Buscar el Dataset
1. Ve a: https://www.kaggle.com/datasets
2. En el buscador, escribe: `English Premier League EPL Match`
3. Busca el dataset que diga "2000-2025" o similar

### Paso 2: Descargar
1. Click en "Download"
2. Se descargará un ZIP

### Paso 3: Extraer y Colocar
1. Descomprime el ZIP
2. Busca el archivo `epl_final.csv` (o similar)
3. **IMPORTANTE**: Renómbralo a `epl_final.csv` si tiene otro nombre
4. Copia el archivo a: `/your-project-path/premier-league-ml/data/raw/`

### Verificar
Ejecuta en terminal:
```bash
cd premier-league-ml
bash setup_data.sh
```

Si todo está bien, deberías ver:
```
✅ Dataset epl_final.csv ya existe
```

---

## Opción 2: Kaggle CLI (Si tienes credenciales) 

### Paso 1: Obtener Credenciales
1. Ve a: https://www.kaggle.com/account/
2. Baja hasta "API"
3. Click en "Create New API Token"
4. Se descarga `kaggle.json`

### Paso 2: Configurar
```bash
# Crear carpeta .kaggle
mkdir -p ~/.kaggle

# Copiar el archivo
cp ~/Downloads/kaggle.json ~/.kaggle/

# Dar permisos correctos
chmod 600 ~/.kaggle/kaggle.json
```

### Paso 3: Descargar Dataset
```bash
cd premier-league-ml

# Ejecutar script de setup
bash setup_data.sh
```

---

## Opción 3: Encontrar Dataset Alternativo

Si no encuentras `epl_final.csv`, busca cualquiera de estos:

- "Premier League Complete Dataset"
- "English Premier League matches"
- "EPL historical data"
- "Football league data"

**Requisitos mínimos del dataset:**
- ✅ Fechas de partidos
- ✅ Equipos (local y visitante)
- ✅ Resultado (goles)
- ✅ Datos de 2000 en adelante

---

## Paso Final: Verificar Descarga

Después de colocar el archivo, ejecuta:

```bash
cd /path/to/premier-league-ml

# Ver estructura
ls -lh data/raw/

# Verificar que existe epl_final.csv
python -c "import pandas as pd; df = pd.read_csv('data/raw/epl_final.csv'); print(f'✅ Dataset cargado: {df.shape[0]} filas')"
```

---

## Problemas Comunes

### "Archivo no encontrado"
- ✓ Verifica la ruta exacta: `data/raw/epl_final.csv`
- ✓ Asegúrate de extraer el ZIP completamente
- ✓ Revisa el nombre del archivo descargado

### "No encuentro el dataset en Kaggle"
- ✓ Busca: `vivovinco` (usuario que publica)
- ✓ O busca por año: "premier league 2025" o "2024"
- ✓ Intenta dataset alternativo: busca "football data"

### "Error de permisos (Kaggle CLI)"
```bash
# Dar permisos correctos
chmod 600 ~/.kaggle/kaggle.json

# Verificar que está en el lugar correcto
ls ~/.kaggle/kaggle.json
```

---

## ✅ Cuando tengas el archivo

Ejecuta el notebook:

```bash
cd premier-league-ml
jupyter notebook notebooks/01_eda_and_modeling.ipynb
```

El notebook cargará automáticamente `data/raw/epl_final.csv` en la primera celda.

---

**¡Una vez descargado, cuéntame qué ves en el dataset! 📊**
