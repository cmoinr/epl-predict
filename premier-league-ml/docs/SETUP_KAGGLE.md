# Configuración de Kaggle para el Dev Container

## Opción 1: Descargar desde Kaggle (RECOMENDADO)

### Paso 1: Obtener Credenciales de Kaggle

1. Ir a https://www.kaggle.com/account
2. Bajar a "API" y click en "Create New API Token"
3. Esto descargará un archivo `kaggle.json`

### Paso 2: Copiar el archivo al Dev Container

**Desde tu máquina host (terminal):**

```bash
# Copiar el archivo kaggle.json al dev container
cp ~/Downloads/kaggle.json /path/to/workspace/.kaggle/kaggle.json

# O si estás usando Codespaces, ir a Settings > Secrets > crear KAGGLE_USERNAME y KAGGLE_KEY
```

### Paso 3: Dar permisos correctos

```bash
mkdir -p ~/.kaggle
chmod 600 ~/.kaggle/kaggle.json
```

### Paso 4: Descargar el dataset

```bash
python src/data_collection.py --download
```

---

## Opción 2: Descargar manualmente desde web

Si no quieres usar CLI:

1. Ve a: https://www.kaggle.com/datasets/vivovinco/english-premier-league-matches
   (O busca "English Premier League EPL Match Data 2000-2025")
2. Click en "Download"
3. Guarda el archivo `epl_final.csv` en `/data/raw/`

---

## Opción 3: Usar dataset alternativo que ya identificaste

El dataset que encontraste: `epl_final.csv` es perfecto.
Úsalo como archivo raw en `/data/raw/epl_final.csv`

---

**Próximo paso**: Una vez tengas el CSV, ejecutar EDA
