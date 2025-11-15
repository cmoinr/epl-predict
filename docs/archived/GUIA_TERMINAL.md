# 🖥️ Guía: Usar Modelos desde Terminal

## Modo 1: Script Interactivo (Recomendado para principiantes)

### Paso 1: Asegurate que los modelos están guardados

En el notebook, ejecuta:
```python
# Celda de "Guardar Modelos"
# (Ya está hecho si ejecutaste la celda anterior)
```

Verifica que exista la carpeta `models/`:
```bash
ls -la models/
```

Deberías ver:
```
models/rf_result_model.pkl
models/gb_result_model.pkl
models/rf_goals_model.pkl
models/gb_goals_model.pkl
models/scaler_model.pkl
```

### Paso 2: Ejecutar predicción desde terminal

```bash
python predict_match.py --home "Chelsea" --away "Liverpool" --date "2025-02-22"
```

**Salida esperada:**
```
📊 Dataset cargado: data/raw/epl_final.csv (9380 partidos)
✅ Modelos cargados desde: models
🔮 Prediciendo: Chelsea vs Liverpool (2025-02-22)...

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

---

## Modo 2: Script con Modo Quiet (Solo predicción)

Para obtener SOLO la predicción (útil para scripts automatizados):

```bash
python predict_match.py --home "Arsenal" --away "Man City" --date "2025-03-01" --quiet
```

**Salida esperada:**
```
Home Win
```

---

## Modo 3: Con Ruta Personalizada al Dataset

Si tu dataset está en otra ubicación:

```bash
python predict_match.py \
  --home "Chelsea" \
  --away "Liverpool" \
  --date "2025-02-22" \
  --data "data/raw/epl_final.csv"
```

---

## Modo 4: Con Ruta Personalizada a Modelos

Si guardaste los modelos en otra carpeta:

```bash
python predict_match.py \
  --home "Tottenham" \
  --away "Arsenal" \
  --date "2025-02-28" \
  --models "mi_carpeta_modelos/"
```

---

## Ejemplos Prácticos

### Predicción 1: Manchester City vs Chelsea
```bash
python predict_match.py --home "Manchester City" --away "Chelsea" --date "2025-03-15"
```

### Predicción 2: Liverpool vs Manchester United
```bash
python predict_match.py --home "Liverpool" --away "Manchester United" --date "2025-03-22"
```

### Predicción 3: Tottenham vs Arsenal (solo resultado)
```bash
python predict_match.py --home "Tottenham" --away "Arsenal" --date "2025-04-05" --quiet
```

---

## Modo 5: Script Python Personalizado

Si quieres más control, crea tu propio script:

**Archivo: `mi_prediccion.py`**

```python
#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')

from predictor import EPLPredictor
import pandas as pd

# Cargar datos
df = pd.read_csv('data/raw/epl_final.csv')

# Cargar modelos
predictor = EPLPredictor('models')

# Predecir múltiples partidos
matches = [
    {'home': 'Chelsea', 'away': 'Liverpool', 'date': '2025-02-22'},
    {'home': 'Arsenal', 'away': 'Man City', 'date': '2025-03-01'},
    {'home': 'Tottenham', 'away': 'Man United', 'date': '2025-03-08'},
]

print('\n🔮 PREDICCIONES MÚLTIPLES:\n')
for match in matches:
    result = predictor.predict_match(
        df,
        match['home'],
        match['away'],
        match['date'],
        X_train_scaled=None  # Se ajustará automáticamente
    )
    
    print(f"  {match['home']} vs {match['away']}")
    print(f"    Resultado: {result['resultado']['random_forest']['prediccion']}")
    print(f"    Goles: {result['goles_totales']['promedio']}\n")
```

Ejecutar:
```bash
python mi_prediccion.py
```

---

## Troubleshooting

### Error: "No se encontraron modelos"
**Solución:** Asegúrate de haber ejecutado la celda "Guardar Modelos" en el notebook

```bash
ls -la models/
```

### Error: "No se encuentra dataset"
**Solución:** Verifica la ruta del dataset

```bash
ls -la data/raw/epl_final.csv
```

### Error: "Formato de fecha inválido"
**Solución:** Usa formato YYYY-MM-DD

```bash
# ✅ Correcto
python predict_match.py --home "Chelsea" --away "Liverpool" --date "2025-02-22"

# ❌ Incorrecto
python predict_match.py --home "Chelsea" --away "Liverpool" --date "22/02/2025"
```

---

## Automatización: Predicciones Diarias

Crear un script que prediga todos los partidos del fin de semana:

**Archivo: `predicciones_semanal.py`**

```python
#!/usr/bin/env python3
import sys
sys.path.insert(0, 'src')
from predictor import EPLPredictor
import pandas as pd
from datetime import datetime

# Cargar
df = pd.read_csv('data/raw/epl_final.csv')
predictor = EPLPredictor('models')

# Partidos de este fin de semana
fin_de_semana = [
    ('Chelsea', 'Liverpool', '2025-02-22'),
    ('Arsenal', 'Man City', '2025-02-22'),
    ('Tottenham', 'Man United', '2025-02-23'),
    ('Brighton', 'Fulham', '2025-02-23'),
]

# Archivo de salida
output_file = f'predicciones_{datetime.now().strftime("%Y%m%d")}.txt'

with open(output_file, 'w') as f:
    f.write(f'PREDICCIONES SEMANALES\n')
    f.write(f'Fecha: {datetime.now().strftime("%Y-%m-%d %H:%M")}\n')
    f.write('='*70 + '\n\n')
    
    for home, away, date in fin_de_semana:
        result = predictor.predict_match(df, home, away, date, X_train_scaled=None)
        
        f.write(f"{home} vs {away} ({date})\n")
        f.write(f"  RF Predicción: {result['resultado']['random_forest']['prediccion']}\n")
        f.write(f"  GB Predicción: {result['resultado']['gradient_boosting']['prediccion']}\n")
        f.write(f"  Goles (promedio): {result['goles_totales']['promedio']}\n")
        f.write('-'*70 + '\n')

print(f'✅ Predicciones guardadas en: {output_file}')
```

Ejecutar:
```bash
python predicciones_semanal.py
```

---

## Cron: Ejecutar Automáticamente Cada Día

Editar crontab:
```bash
crontab -e
```

Agregar línea para ejecutar cada mañana a las 8 AM:
```bash
0 8 * * * cd /path/to/premier-league-ml && python predicciones_semanal.py
```

---

## Resumen de Comandos Útiles

```bash
# Ver ayuda
python predict_match.py --help

# Predicción normal
python predict_match.py --home "Chelsea" --away "Liverpool" --date "2025-02-22"

# Solo resultado
python predict_match.py --home "Chelsea" --away "Liverpool" --date "2025-02-22" --quiet

# Con dataset personalizado
python predict_match.py --home "Chelsea" --away "Liverpool" --date "2025-02-22" --data "otra_ruta.csv"

# Verificar modelos
ls -lh models/

# Verificar dataset
head -5 data/raw/epl_final.csv

# Prueba rápida
python -c "import sys; sys.path.insert(0, 'src'); from predictor import EPLPredictor; print('✅ Módulos OK')"
```

---

¡Listo! Ahora puedes usar tus modelos desde terminal 🚀
