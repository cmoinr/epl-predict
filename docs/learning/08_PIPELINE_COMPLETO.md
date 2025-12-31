# 🔄 Pipeline Completo: Del Dato a la Predicción

## Visión General del Sistema

```
DATOS BRUTOS          PROCESAMIENTO         MODELADO           PREDICCIÓN
────────────────      ──────────────        ────────────       ──────────

CSV de partidos   →  Limpieza y            Feature            Predicción
históricos           Validación            Engineering        de resultados
                                                 ↓
                                          Entrenamiento
                                          de modelos
                                                 ↓
                                          Ensemble
                                          Learning
                                                 ↓
                                          Comparación
                                          vs Mercado
                                                 ↓
                                          Apuestas de
                                          Valor
```

---

## 📁 Estructura del Pipeline

```
EPL-Predict/
├── data/
│   ├── raw/
│   │   ├── epl_final.csv          ← Datos históricos brutos
│   │   └── epl_odds.csv           ← Odds del mercado histórico
│   │
│   └── processed/
│       ├── enriched.csv           ← Con todas las features
│       └── ready_for_ml.csv       ← Normalizado para modelos
│
├── src/
│   ├── feature_engineering.py     ← Creación de variables
│   ├── market_features.py         ← Features de odds
│   ├── predictor.py               ← Motor de predicción
│   ├── odds_comparison.py         ← Comparación ML vs Mercado
│   └── __pycache__/
│
├── models/
│   ├── xgb_model.pkl              ← Modelo entrenado
│   ├── lgb_model.pkl
│   ├── rf_model.pkl
│   └── ensemble_model.pkl         ← Combinación de modelos
│
└── scripts/
    ├── retrain_models.py          ← Reentrenamiento
    ├── predict_match.py           ← Predicción partido
    └── get_value_bets.py          ← Identificar value bets
```

---

## 🔍 FASE 1: Exploración de Datos (EDA)

### Paso 1: Cargar Datos

```python
import pandas as pd
import numpy as np
from datetime import datetime

# Cargar datos históricos
df = pd.read_csv('data/raw/epl_final.csv')

print(df.head())
print(df.info())
print(df.describe())
```

Estructura esperada:
```
   Date  HomeTeam  AwayTeam  FTHG  FTAG FTR
0  2023-08-12  Arsenal  Chelsea    2    1  H
1  2023-08-13  Man United  Liverpool  1  1  D
...
```

### Paso 2: Cargar Odds

```python
# Cargar cuotas históricas
odds_df = pd.read_csv('data/raw/epl_odds.csv')

# Fusionar con datos de partidos
df = df.merge(odds_df, on=['Date', 'HomeTeam', 'AwayTeam'])

print(df[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'B365H', 'B365D', 'B365A']].head())
```

### Paso 3: Validación Inicial

```python
# Verificar valores nulos
print(df.isnull().sum())

# Verificar duplicados
print(f"Duplicados: {df.duplicated().sum()}")

# Convertir fechas
df['Date'] = pd.to_datetime(df['Date'])

# Ordenar por fecha
df = df.sort_values('Date').reset_index(drop=True)

print(f"Dataset: {len(df)} partidos desde {df['Date'].min()} hasta {df['Date'].max()}")
```

---

## 🔧 FASE 2: Feature Engineering

### Paso 1: Crear Features Básicas

```python
from src.feature_engineering import FeatureEngineer

# Crear ingeniero de features
fe = FeatureEngineer(df)

# Crear todas las features
df_features = fe.create_all_features()

print(f"Features creadas: {df_features.columns.tolist()}")
```

### Paso 2: Features de Forma Reciente

```python
def create_form_features(df, windows=[5, 10, 15]):
    """Crea features de forma reciente"""
    for team in pd.concat([df['HomeTeam'], df['AwayTeam']]).unique():
        for window in windows:
            # Obtener puntos en últimos N partidos
            team_home = df[df['HomeTeam'] == team].copy()
            team_away = df[df['AwayTeam'] == team].copy()
            
            # Calcular puntos (3 victoria, 1 empate, 0 derrota)
            home_points = (team_home['FTR'] == 'H') * 3 + (team_home['FTR'] == 'D') * 1
            away_points = (team_away['FTR'] == 'A') * 3 + (team_away['FTR'] == 'D') * 1
            
            df[f'{team}_form_{window}_h'] = (home_points.rolling(window).sum() / (window * 3)).shift(1)
            df[f'{team}_form_{window}_a'] = (away_points.rolling(window).sum() / (window * 3)).shift(1)
    
    return df
```

### Paso 3: Features de Goles

```python
def create_goal_features(df, windows=[5, 10]):
    """Crea features de goles marcados y recibidos"""
    for team in pd.concat([df['HomeTeam'], df['AwayTeam']]).unique():
        for window in windows:
            # Goles marcados en casa
            team_home = df[df['HomeTeam'] == team].copy()
            goals_for = team_home['FTHG'].rolling(window).mean()
            goals_against = team_home['FTAG'].rolling(window).mean()
            
            df[f'{team}_goals_for_{window}_h'] = goals_for.shift(1)
            df[f'{team}_goals_against_{window}_h'] = goals_against.shift(1)
            
            # Goles marcados fuera
            team_away = df[df['AwayTeam'] == team].copy()
            goals_for = team_away['FTAG'].rolling(window).mean()
            goals_against = team_away['FTHG'].rolling(window).mean()
            
            df[f'{team}_goals_for_{window}_a'] = goals_for.shift(1)
            df[f'{team}_goals_against_{window}_a'] = goals_against.shift(1)
    
    return df
```

### Paso 4: Features de Mercado

```python
from src.market_features import MarketFeatureEngineer

# Crear features de mercado
mfe = MarketFeatureEngineer(df)

# Convertir odds a probabilidades
df = mfe.create_market_probabilities()

print(f"Features totales: {len(df.columns)}")
```

---

## 🎯 FASE 3: Preparación para ML

### Paso 1: Seleccionar Features y Target

```python
# Definir features
feature_cols = [col for col in df.columns 
                if col not in ['Date', 'HomeTeam', 'AwayTeam', 
                               'FTHG', 'FTAG', 'FTR', 
                               'B365H', 'B365D', 'B365A']]

# Definir target
target_mapping = {'H': 0, 'D': 1, 'A': 2}
df['target'] = df['FTR'].map(target_mapping)

# Eliminar nulos (features usan datos pasados)
df_clean = df.dropna(subset=feature_cols + ['target'])

print(f"Dataset limpio: {len(df_clean)} partidos")
print(f"Features: {len(feature_cols)}")
print(f"Target balance:\n{df_clean['target'].value_counts()}")
```

### Paso 2: Dividir Train/Test

```python
from sklearn.model_selection import train_test_split

# División temporal (importante para datos de series de tiempo)
split_date = df_clean['Date'].quantile(0.8)

df_train = df_clean[df_clean['Date'] <= split_date].copy()
df_test = df_clean[df_clean['Date'] > split_date].copy()

X_train = df_train[feature_cols]
y_train = df_train['target']

X_test = df_test[feature_cols]
y_test = df_test['target']

print(f"Train: {len(X_train)} partidos ({X_train.index[0]} a {X_train.index[-1]})")
print(f"Test:  {len(X_test)} partidos ({X_test.index[0]} a {X_test.index[-1]})")
```

### Paso 3: Normalización

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Guardar scaler
import pickle
with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Features normalizados")
```

---

## 🧠 FASE 4: Entrenamiento de Modelos

### Paso 1: Entrenar Modelos Base

```python
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle

# XGBoost
xgb = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='multi:softmax',
    num_class=3,
    random_state=42
)
xgb.fit(X_train_scaled, y_train)

# LightGBM
lgb = LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    num_class=3,
    random_state=42
)
lgb.fit(X_train_scaled, y_train)

# Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
rf.fit(X_train_scaled, y_train)

# Guardar modelos
with open('models/xgb_model.pkl', 'wb') as f:
    pickle.dump(xgb, f)
with open('models/lgb_model.pkl', 'wb') as f:
    pickle.dump(lgb, f)
with open('models/rf_model.pkl', 'wb') as f:
    pickle.dump(rf, f)

print("Modelos entrenados y guardados")
```

### Paso 2: Crear Ensemble

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

# Crear ensemble
ensemble = StackingClassifier(
    estimators=[
        ('xgb', xgb),
        ('lgb', lgb),
        ('rf', rf)
    ],
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5
)

ensemble.fit(X_train_scaled, y_train)

with open('models/ensemble_model.pkl', 'wb') as f:
    pickle.dump(ensemble, f)

print("Ensemble creado y entrenado")
```

### Paso 3: Evaluación

```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Predicciones
y_pred = ensemble.predict(X_test_scaled)
y_pred_proba = ensemble.predict_proba(X_test_scaled)

# Métricas
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión en Test: {accuracy:.2%}")

print("\nReporte Detallado:")
print(classification_report(y_test, y_pred, 
                           target_names=['Home', 'Draw', 'Away']))

print("\nMatriz de Confusión:")
print(confusion_matrix(y_test, y_pred))
```

---

## 🔮 FASE 5: Predicción de Nuevos Partidos

### Paso 1: Preparar Nuevo Partido

```python
def prepare_match_for_prediction(home_team, away_team, df_historical):
    """
    Prepara features para un nuevo partido
    """
    new_row = {
        'HomeTeam': home_team,
        'AwayTeam': away_team,
        'Date': pd.Timestamp.now()
    }
    
    # Obtener últimos datos de cada equipo
    home_recent = df_historical[df_historical['HomeTeam'] == home_team].tail(10)
    away_recent = df_historical[df_historical['AwayTeam'] == away_team].tail(10)
    
    # Calcular features
    # (usar la lógica creada en feature_engineering.py)
    
    return new_row

# Ejemplo
match_features = prepare_match_for_prediction('Arsenal', 'Chelsea', df_clean)
```

### Paso 2: Realizar Predicción

```python
import pickle

# Cargar modelo y scaler
with open('models/ensemble_model.pkl', 'rb') as f:
    ensemble = pickle.load(f)
with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Normalizar features del nuevo partido
match_features_scaled = scaler.transform([match_features])

# Predecir
prediction = ensemble.predict(match_features_scaled)[0]
probabilities = ensemble.predict_proba(match_features_scaled)[0]

# Mapear resultado
result_mapping = {0: 'Home Win', 1: 'Draw', 2: 'Away Win'}

print(f"Match: Arsenal vs Chelsea")
print(f"\nPredicción: {result_mapping[prediction]}")
print(f"\nProbabilidades:")
print(f"  Home Win: {probabilities[0]:.1%}")
print(f"  Draw:     {probabilities[1]:.1%}")
print(f"  Away Win: {probabilities[2]:.1%}")
```

---

## 💰 FASE 6: Análisis de Value Betting

### Paso 1: Comparar contra Odds del Mercado

```python
from src.odds_comparison import OddsComparator

comparator = OddsComparator()

# Probabilities del modelo
model_probs = {
    'home': probabilities[0],
    'draw': probabilities[1],
    'away': probabilities[2]
}

# Odds del mercado
market_odds = {
    'home': 1.80,    # Ejemplo
    'draw': 3.50,
    'away': 4.50
}

# Comparar
comparison = comparator.compare(model_probs, market_odds)

print("\nValue Betting Analysis:")
for outcome, analysis in comparison.items():
    print(f"\n{outcome.upper()}:")
    print(f"  ML:        {analysis['model_prob']:.1%}")
    print(f"  Mercado:   {analysis['market_prob']:.1%}")
    print(f"  Edge:      {analysis['edge']:+.1%}")
    print(f"  EV:        {analysis['ev']:+.2%}")
    print(f"  Value Bet: {'✓ YES' if analysis['is_value'] else '✗ NO'}")
```

### Paso 2: Calcular Kelly Criterion

```python
def calculate_kelly_stake(prob, odd, bankroll, kelly_fraction=0.5):
    """
    Calcula monto a apostar usando Kelly Criterion
    """
    kelly_pct = (prob * (odd - 1) - (1 - prob)) / (odd - 1)
    kelly_stake = kelly_pct * bankroll * kelly_fraction
    
    return kelly_stake, kelly_pct

# Ejemplo
kelly_stake, kelly_pct = calculate_kelly_stake(
    prob=probabilities[0],  # 0.55 (55%)
    odd=1.80,
    bankroll=1000,
    kelly_fraction=0.5
)

print(f"\nKelly Criterion (Half Kelly):")
print(f"  Kelly %:      {kelly_pct:.1%}")
print(f"  Monto apostar: ${kelly_stake:.2f}")
```

---

## 📊 FASE 7: Backtesting

### Paso 1: Simulación Histórica

```python
def backtest_strategy(df_test, X_test_scaled, y_test, ensemble, 
                      market_odds_df, min_ev=0.05, bankroll=1000):
    """
    Backtestea la estrategia de value betting en datos históricos
    """
    total_profit = 0
    total_bets = 0
    winning_bets = 0
    
    predictions = ensemble.predict_proba(X_test_scaled)
    
    for idx, (pred, market_odd, actual) in enumerate(zip(predictions, 
                                                          market_odds_df, 
                                                          y_test)):
        model_prob = pred[actual]  # Probabilidad del resultado ganador
        odd = market_odd[actual]   # Odd del resultado ganador
        
        # Calcular EV
        ev = (model_prob * (odd - 1)) - ((1 - model_prob) * 1)
        
        if ev > min_ev:
            total_bets += 1
            kelly_stake = (model_prob * (odd - 1) - (1 - model_prob)) / (odd - 1) * bankroll * 0.5
            
            # Ganancia/Pérdida
            profit = kelly_stake * (odd - 1)  # Gana
            
            total_profit += profit
            winning_bets += 1
    
    return {
        'total_bets': total_bets,
        'winning_bets': winning_bets,
        'win_rate': winning_bets / total_bets if total_bets > 0 else 0,
        'total_profit': total_profit,
        'roi': total_profit / (total_bets * 100) if total_bets > 0 else 0
    }

# Ejecutar backtest
results = backtest_strategy(df_test, X_test_scaled, y_test, ensemble, 
                           odds_list, min_ev=0.05)

print("\nBacktest Results:")
print(f"  Total Bets:   {results['total_bets']}")
print(f"  Winning Bets: {results['winning_bets']}")
print(f"  Win Rate:     {results['win_rate']:.1%}")
print(f"  Total Profit: ${results['total_profit']:.2f}")
print(f"  ROI:          {results['roi']:+.1%}")
```

---

## 🔄 FASE 8: Reentrenamiento y Monitoreo

### Paso 1: Reentrenamiento Periódico

```python
from datetime import datetime, timedelta

def retrain_models(df_all_new_data, last_training_date):
    """
    Reentrena los modelos con nuevos datos
    """
    # Filtrar datos desde último entrenamiento
    df_new = df_all_new_data[df_all_new_data['Date'] > last_training_date]
    
    if len(df_new) < 10:  # Mínimo de 10 partidos para reentrenar
        print("No hay suficientes datos nuevos para reentrenar")
        return None
    
    # Combinar con datos históricos
    df_combined = pd.concat([df_all_new_data[:df_all_new_data[df_all_new_data['Date'] <= last_training_date].index[-1]], 
                             df_new])
    
    # Recalcular features
    fe = FeatureEngineer(df_combined)
    df_features = fe.create_all_features()
    
    # Entrenar nuevamente (FASE 4)
    # ...
    
    print(f"Reentrenamiento completado. Datos nuevos: {len(df_new)}")
    return df_features

# Ejemplo: reentrenar cada 30 días
last_training = datetime(2025, 12, 1)
if (datetime.now() - last_training).days > 30:
    retrain_models(df_complete, last_training)
```

### Paso 2: Monitoreo de Desempeño

```python
def monitor_performance(predictions, actuals, window=20):
    """
    Monitorea el desempeño del modelo en ventana deslizante
    """
    accuracies = []
    
    for i in range(len(predictions) - window):
        window_acc = accuracy_score(
            actuals[i:i+window],
            predictions[i:i+window]
        )
        accuracies.append(window_acc)
    
    return {
        'mean_accuracy': np.mean(accuracies),
        'std_accuracy': np.std(accuracies),
        'min_accuracy': np.min(accuracies),
        'max_accuracy': np.max(accuracies),
        'trend': 'improving' if accuracies[-1] > accuracies[0] else 'declining'
    }
```

---

## 🚀 Scripts Principales del Proyecto

### `predict_match.py`

```python
#!/usr/bin/env python
"""Predicción de un partido específico"""

from src.predictor import MatchPredictor

if __name__ == '__main__':
    predictor = MatchPredictor()
    
    result = predictor.predict_match(
        home_team='Arsenal',
        away_team='Chelsea',
        market_odds={'home': 1.80, 'draw': 3.50, 'away': 4.50}
    )
    
    print(result)
```

### `get_value_bets.py`

```python
#!/usr/bin/env python
"""Identificar value bets en fecha específica"""

from src.odds_comparison import find_value_bets

if __name__ == '__main__':
    value_bets = find_value_bets(
        date='2025-12-31',
        min_ev=0.05
    )
    
    for bet in value_bets:
        print(f"{bet['match']}: {bet['recommendation']} "
              f"({bet['ev_pct']:+.2f}% EV)")
```

### `retrain_models_improved.py`

```python
#!/usr/bin/env python
"""Reentrenar modelos con datos actualizados"""

from src.training import ModelTrainer

if __name__ == '__main__':
    trainer = ModelTrainer()
    results = trainer.train_and_evaluate()
    
    print(f"Precisión: {results['accuracy']:.2%}")
    print(f"Modelos guardados en /models/")
```

---

## 📈 Resumen del Pipeline

```
ENTRADA                PROCESO              SALIDA
────────────────      ─────────────────    ──────────────────

Datos históricos   →  Feature Eng      →   Features 
(CSV)                 Normalización        (X, y)
                           ↓
                      Entrenamiento    →   Modelos
                      Ensemble             (.pkl)
                           ↓
Nuevo partido      →  Predicción      →   Probabilidades
Odds mercado           Comparación         Value bets
                       Kelly               Recomendaciones
```

---

## ✅ Checklist de Implementación

- [ ] Cargar y explorar datos
- [ ] Crear features (forma, goles, mercado)
- [ ] Dividir train/test temporalmente
- [ ] Normalizar features
- [ ] Entrenar 3 modelos base
- [ ] Crear ensemble (stacking)
- [ ] Evaluar precisión
- [ ] Comparar contra odds
- [ ] Calcular value bets
- [ ] Ejecutar backtest
- [ ] Configurar reentrenamiento automático
- [ ] Desplegar en producción

---

## 📚 Referencias

Para más información sobre cada fase:
- Features: [04_FEATURE_ENGINEERING.md](04_FEATURE_ENGINEERING.md)
- Modelos: [05_MODELOS_CLASIFICACION.md](05_MODELOS_CLASIFICACION.md)
- Ensemble: [06_ENSEMBLE_LEARNING.md](06_ENSEMBLE_LEARNING.md)
- Value Betting: [07_VALUE_BETTING_ODDS.md](07_VALUE_BETTING_ODDS.md)

---

## 🎓 ¡Felicitaciones!

Has completado la guía de aprendizaje de EPL-Predict. Ahora entiendes:

✅ Cómo funciona Machine Learning para predicción de partidos
✅ Feature Engineering y preparación de datos
✅ Modelos de clasificación (Random Forest, XGBoost, etc.)
✅ Ensemble Learning para mejorar precisión
✅ Matemática de apuestas y value betting
✅ Pipeline completo desde datos a predicción

**Próximos pasos**: Experimenta con los scripts, ajusta parámetros, y crea tu propia estrategia de value betting.

---

**¿Preguntas o sugerencias?** Revisa los archivos del proyecto:
- [src/predictor.py](../../src/predictor.py)
- [scripts/predict_match.py](../../predict_match.py)
- [scripts/get_value_bets.py](../../get_value_bets.py)
