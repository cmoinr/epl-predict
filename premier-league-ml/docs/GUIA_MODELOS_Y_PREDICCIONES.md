# 🤖 Guía: Dónde Están Los Modelos y Cómo Usarlos

## 📍 Pregunta 1: ¿Dónde se Alojan Los Modelos?

### Ubicación ACTUAL (En Memoria del Notebook)

Los 4 modelos que entrenaste existen **SOLO en la memoria de Jupyter** en este momento:

```
Kernel de Jupyter (En Memoria)
├── rf_result      → RandomForestClassifier (predicción 1X2)
├── gb_result      → GradientBoostingClassifier (predicción 1X2)
├── rf_goals       → RandomForestRegressor (predicción goles)
└── gb_goals       → GradientBoostingRegressor (predicción goles)
```

**Problema:** Si cierras el notebook, pierdes los modelos.

### Ubicación RECOMENDADA (Persistencia)

Para que tus modelos sobrevivan y sean reutilizables:

```
/workspaces/codespaces-blank/premier-league-ml/
├── 📂 models/
│   ├── rf_result_model.pkl          (Random Forest - Resultados)
│   ├── gb_result_model.pkl          (Gradient Boosting - Resultados)
│   ├── rf_goals_model.pkl           (Random Forest - Goles)
│   ├── gb_goals_model.pkl           (Gradient Boosting - Goles)
│   └── scaler.pkl                   (StandardScaler - Normalización)
└── 📂 src/
    └── model_persistence.py         (Guardar/cargar modelos)
```

**Ventaja:** Los modelos están guardados en archivos `.pkl` (pickle). Puedes usarlos después, en terminal o en otro notebook.

---

## 💾 ¿Cómo Guardar Los Modelos?

### Opción 1: Guardar Desde el Notebook (Recomendado)

Ejecuta esto en una nueva celda del notebook:

```python
import pickle
import os

# Crear carpeta si no existe
os.makedirs('models', exist_ok=True)

# Guardar los 4 modelos entrenados
models_to_save = {
    'rf_result': rf_result,       # Random Forest - Resultados
    'gb_result': gb_result,       # Gradient Boosting - Resultados
    'rf_goals': rf_goals,         # Random Forest - Goles
    'gb_goals': gb_goals,         # Gradient Boosting - Goles
    'scaler': scaler              # Normalizador de features
}

for name, model in models_to_save.items():
    with open(f'models/{name}_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print(f'✅ Guardado: {name}_model.pkl')

print('\n🎉 Todos los modelos guardados en /models/')
```

**Resultado esperado:**
```
✅ Guardado: rf_result_model.pkl
✅ Guardado: gb_result_model.pkl
✅ Guardado: rf_goals_model.pkl
✅ Guardado: gb_goals_model.pkl
✅ Guardado: scaler_model.pkl

🎉 Todos los modelos guardados en /models/
```

---

## 🔮 Pregunta 2: ¿Cómo Predecir Futuros Partidos?

### Flujo Completo de Predicción

```
Nuevo Partido EPL
     ↓
[1] Obtener datos básicos
    (equipos, fechas, temporada, etc.)
     ↓
[2] Cargar features engineers
    (Form, H2H, Goals Avg, Home Advantage)
     ↓
[3] Procesar características
    (aplicar EPLFeatureEngineer.engineer_features())
     ↓
[4] Normalizar features
    (aplicar StandardScaler)
     ↓
[5] Cargar modelos guardados
    (rf_result, gb_result, rf_goals, gb_goals)
     ↓
[6] Hacer predicción
    (model.predict(X_nuevo))
     ↓
[7] Mostrar resultados
    (Probabilidades, predicción final)
```

### Paso 1: Crear Módulo de Predicción

**Archivo: `src/predictor.py`**

```python
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from feature_engineering import EPLFeatureEngineer

class EPLPredictor:
    """Cargar modelos guardados y hacer predicciones en nuevos partidos"""
    
    def __init__(self, models_dir='models'):
        """Cargar todos los modelos desde archivos .pkl"""
        self.models_dir = Path(models_dir)
        
        # Cargar modelos
        self.rf_result = pickle.load(open(f'{self.models_dir}/rf_result_model.pkl', 'rb'))
        self.gb_result = pickle.load(open(f'{self.models_dir}/gb_result_model.pkl', 'rb'))
        self.rf_goals = pickle.load(open(f'{self.models_dir}/rf_goals_model.pkl', 'rb'))
        self.gb_goals = pickle.load(open(f'{self.models_dir}/gb_goals_model.pkl', 'rb'))
        self.scaler = pickle.load(open(f'{self.models_dir}/scaler_model.pkl', 'rb'))
        
        # Feature engineer para procesar nuevos datos
        self.engineer = EPLFeatureEngineer()
        
        print('✅ Modelos cargados correctamente')
    
    def predict_match(self, df_historical, home_team, away_team, match_date):
        """
        Predecir resultado y goles para un próximo partido
        
        Parámetros:
        -----------
        df_historical : DataFrame
            Dataset histórico completo (para calcular features)
        home_team : str
            Nombre del equipo local
        away_team : str
            Nombre del equipo visitante
        match_date : str
            Fecha del partido (formato: 'YYYY-MM-DD')
        
        Retorna:
        --------
        dict : Predicción de resultado y goles
        """
        
        # 1. Crear row del nuevo partido
        new_match = {
            'HomeTeam': home_team,
            'AwayTeam': away_team,
            'Date': match_date,
            # Campos requeridos para feature engineering
        }
        
        # 2. Generar features para este partido
        X_new = self.engineer.engineer_features(
            df_historical.append(pd.DataFrame([new_match]))
        ).iloc[-1:].drop(['HomeTeam', 'AwayTeam', 'Date'], axis=1, errors='ignore')
        
        # 3. Normalizar features
        X_new_scaled = self.scaler.transform(X_new)
        
        # 4. Predecir RESULTADO (1X2)
        pred_result_rf = self.rf_result.predict(X_new_scaled)[0]
        prob_result_rf = self.rf_result.predict_proba(X_new_scaled)[0]
        
        pred_result_gb = self.gb_result.predict(X_new_scaled)[0]
        prob_result_gb = self.gb_result.predict_proba(X_new_scaled)[0]
        
        # 5. Predecir GOLES TOTALES
        pred_goals_rf = self.rf_goals.predict(X_new_scaled)[0]
        pred_goals_gb = self.gb_goals.predict(X_new_scaled)[0]
        
        # 6. Mapear predicción a resultado
        result_map = {0: 'Away Win', 1: 'Draw', 2: 'Home Win'}
        
        return {
            'match': f'{home_team} vs {away_team}',
            'date': match_date,
            'resultado': {
                'random_forest': {
                    'prediccion': result_map[pred_result_rf],
                    'confianza': max(prob_result_rf) * 100,
                    'probabilidades': {
                        'Away': prob_result_rf[0] * 100,
                        'Draw': prob_result_rf[1] * 100,
                        'Home': prob_result_rf[2] * 100
                    }
                },
                'gradient_boosting': {
                    'prediccion': result_map[pred_result_gb],
                    'confianza': max(prob_result_gb) * 100,
                    'probabilidades': {
                        'Away': prob_result_gb[0] * 100,
                        'Draw': prob_result_gb[1] * 100,
                        'Home': prob_result_gb[2] * 100
                    }
                }
            },
            'goles_totales': {
                'random_forest': round(pred_goals_rf, 2),
                'gradient_boosting': round(pred_goals_gb, 2),
                'promedio': round((pred_goals_rf + pred_goals_gb) / 2, 2)
            }
        }
```

---

## 🖥️ Pregunta 3: ¿Terminal o Notebook?

### OPCIÓN A: Desde Terminal (Línea de Comandos)

**Ventaja:** Rápido, no requiere abrir Jupyter, ideal para automatización

**Pasos:**

1. **Guardar modelos desde notebook (ejecutar una vez)**
   ```python
   # En celda del notebook
   import pickle
   pickle.dump(rf_result, open('models/rf_result_model.pkl', 'wb'))
   pickle.dump(gb_result, open('models/gb_result_model.pkl', 'wb'))
   pickle.dump(rf_goals, open('models/rf_goals_model.pkl', 'wb'))
   pickle.dump(gb_goals, open('models/gb_goals_model.pkl', 'wb'))
   pickle.dump(scaler, open('models/scaler_model.pkl', 'wb'))
   ```

2. **Crear script de predicción**
   ```bash
   # Archivo: predict_match.py
   python predict_match.py --home "Manchester City" --away "Arsenal" --date "2025-02-15"
   ```

3. **Ejemplo de script `predict_match.py`:**

```python
#!/usr/bin/env python3
import sys
sys.path.append('src')
from predictor import EPLPredictor
import pandas as pd
import argparse

# Cargar datos históricos
df = pd.read_csv('data/raw/epl_final.csv')

# Cargar predictor
predictor = EPLPredictor()

# Parse argumentos
parser = argparse.ArgumentParser()
parser.add_argument('--home', required=True)
parser.add_argument('--away', required=True)
parser.add_argument('--date', required=True)
args = parser.parse_args()

# Hacer predicción
result = predictor.predict_match(df, args.home, args.away, args.date)

# Mostrar resultados
print(f"\n{'='*60}")
print(f"🔮 PREDICCIÓN: {result['match']}")
print(f"📅 Fecha: {result['date']}")
print(f"{'='*60}")

print(f"\n📊 RESULTADO (1X2):")
print(f"  Random Forest:")
print(f"    → {result['resultado']['random_forest']['prediccion']}")
print(f"    → Confianza: {result['resultado']['random_forest']['confianza']:.1f}%")

print(f"\n  Gradient Boosting:")
print(f"    → {result['resultado']['gradient_boosting']['prediccion']}")
print(f"    → Confianza: {result['resultado']['gradient_boosting']['confianza']:.1f}%")

print(f"\n⚽ GOLES TOTALES:")
print(f"  RF: {result['goles_totales']['random_forest']}")
print(f"  GB: {result['goles_totales']['gradient_boosting']}")
print(f"  Promedio: {result['goles_totales']['promedio']}")

print(f"\n{'='*60}\n")
```

**Ejecutar desde terminal:**
```bash
python predict_match.py --home "Chelsea" --away "Liverpool" --date "2025-02-22"
```

### OPCIÓN B: Desde Notebook (Interactivo)

**Ventaja:** Más visual, mejor para exploración, puedes ajustar parámetros fácilmente

**Pasos:**

1. **Agregar nueva celda al notebook:**

```python
from src.predictor import EPLPredictor

# Cargar predictor (usa los modelos guardados)
predictor = EPLPredictor('models')

# Predecir próximo partido
prediccion = predictor.predict_match(
    df_historical=df,
    home_team='Manchester United',
    away_team='Tottenham',
    match_date='2025-02-08'
)

# Mostrar resultados formateado
import json
print(json.dumps(prediccion, indent=2))
```

2. **Ejecutar y ver resultados inmediatamente en el notebook**

---

## 📋 Resumen de Procedimiento Completo

### Fase 1: Guardar Modelos (1 vez)
```
[Notebook] → Ejecutar celda de guardado → modelos/*.pkl
```

### Fase 2: Predecir Nuevo Partido
```
[Terminal] → predict_match.py → Resultado
    O
[Notebook] → EPLPredictor.predict_match() → Resultado
```

### Fase 3: Automatizar (Opcional)
```
[Cron/Scheduler] → predict_match.py (cada día) → Guardar predicciones
```

---

## 🚀 Siguiente: Implementar Todo Esto

¿Quieres que:

1. **Agregue una celda al notebook para guardar los modelos?**
2. **Cree el módulo `src/predictor.py`?**
3. **Cree el script `predict_match.py` para terminal?**
4. **Agregue una celda de ejemplo de predicción en el notebook?**

Dime cuál es tu preferencia o si quieres que haga todo! 🎯
