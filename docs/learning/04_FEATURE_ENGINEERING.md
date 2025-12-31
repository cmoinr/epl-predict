# 📊 Feature Engineering en EPL-Predict

## ¿Qué es Feature Engineering?

**Feature Engineering** (Ingeniería de Características) es el proceso de **transformar datos brutos en variables significativas** que los algoritmos de Machine Learning puedan usar para hacer predicciones precisas.

> 💡 **Analogía**: Si los datos brutos son ingredientes, el feature engineering es la receta que los convierte en un plato delicioso.

---

## ¿Por qué es tan importante?

```
📊 Datos Brutos          🔧 Feature Engineering          🎯 Predicción
─────────────────        ────────────────────────        ─────────────
Arsenal 2-1 Chelsea  →   Arsenal_Form: 8.5/15       →   Home Win: 65%
Chelsea 1-0 Spurs    →   Chelsea_Form: 9.0/15       →   Draw: 20%
Arsenal 3-2 Man U    →   H2H_Arsenal: 60% wins      →   Away Win: 15%
```

**Un buen feature engineering puede mejorar la precisión del modelo del 45% al 55%+**

---

## 🧩 Tipos de Features en EPL-Predict

### 1. **Features Básicas** (Directas del dataset)

Estas vienen directamente de los datos históricos:

```python
# Ejemplo del CSV
Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR
2023-08-12,Arsenal,Chelsea,2,1,H
```

- `HomeTeam`: Equipo local
- `AwayTeam`: Equipo visitante  
- `FTHG`: Full Time Home Goals (goles local)
- `FTAG`: Full Time Away Goals (goles visitante)
- `FTR`: Full Time Result (H=Home win, D=Draw, A=Away win)

---

### 2. **Features de Forma Reciente** (Rolling Statistics)

Miden el **rendimiento reciente** de los equipos:

```python
# Últimos 5 partidos
home_form_5 = team_points_last_5_matches / 15  # 3 pts por victoria
away_form_5 = opponent_points_last_5_matches / 15

# Últimos 10 partidos
home_form_10 = team_points_last_10_matches / 30
```

**Ejemplo Real**:
```
Arsenal últimos 5 partidos: W-W-D-W-L = 10 puntos → form_5 = 10/15 = 0.667
Chelsea últimos 5 partidos: W-W-W-D-D = 11 puntos → form_5 = 11/15 = 0.733
```

---

### 3. **Features de Promedio de Goles**

```python
# Goles marcados (ataque)
home_goals_scored_avg = mean(últimos_N_goles_marcados_local)
away_goals_scored_avg = mean(últimos_N_goles_marcados_visitante)

# Goles recibidos (defensa)
home_goals_conceded_avg = mean(últimos_N_goles_recibidos_local)
away_goals_conceded_avg = mean(últimos_N_goles_recibidos_visitante)
```

**Ejemplo**:
```
Arsenal (Local): 2.1 goles marcados, 0.8 goles recibidos → Ataque fuerte, defensa sólida
Chelsea (Visitante): 1.5 goles marcados, 1.2 goles recibidos → Ataque moderado
```

---

### 4. **Features Head-to-Head (H2H)**

Historial de enfrentamientos directos:

```python
# Últimos 5 encuentros entre Arsenal vs Chelsea
h2h_home_wins = count(victorias_local) / total_matches
h2h_draws = count(empates) / total_matches
h2h_away_wins = count(victorias_visitante) / total_matches

# Goles en H2H
h2h_avg_goals = mean(total_goles_en_enfrentamientos_directos)
```

**Ejemplo**:
```
Arsenal vs Chelsea últimos 5:
- Arsenal wins: 3 → 60%
- Draws: 1 → 20%
- Chelsea wins: 1 → 20%
- Promedio goles: 2.8 por partido
```

---

### 5. **Features de Posición en la Tabla**

```python
home_position = posición_actual_en_la_tabla  # 1-20
away_position = posición_visitante_en_la_tabla

# Normalizada (para ML)
position_diff = (away_position - home_position) / 20
```

**Interpretación**:
- `position_diff > 0`: Local es mejor equipo (menor posición = más arriba)
- `position_diff < 0`: Visitante es mejor
- `position_diff ≈ 0`: Equipos similares

---

### 6. **Features de Ventaja Local**

```python
# Rendimiento en casa vs fuera
home_advantage = (home_wins_at_home / total_home_games) - 
                 (away_wins_away / total_away_games)

# Goles de local vs visitante
home_goal_advantage = home_goals_at_home - away_goals_away
```

---

### 7. **Features de Mercado (Market Intelligence)**

Estas provienen de las **odds de casas de apuestas**:

```python
# Probabilidades implícitas del mercado
market_home_prob = 1 / home_odds
market_draw_prob = 1 / draw_odds
market_away_prob = 1 / away_odds

# Ajuste por overround (margen de la casa)
total_prob = market_home_prob + market_draw_prob + market_away_prob
market_home_prob_adj = market_home_prob / total_prob
```

**Ejemplo**:
```
Odds: Home @1.80, Draw @3.50, Away @4.50
Prob implícitas: 55.6%, 28.6%, 22.2% (suma=106.4% → overround)
Prob ajustadas: 52.3%, 26.9%, 20.9% (suma=100%)
```

---

## 🔧 Implementación en el Código

### Archivo: `src/feature_engineering.py`

```python
class FeatureEngineer:
    def __init__(self, df):
        self.df = df.copy()
        
    def create_all_features(self):
        """Crea todas las features para el dataset"""
        self.create_form_features()
        self.create_goal_features()
        self.create_h2h_features()
        self.create_position_features()
        self.create_home_advantage_features()
        return self.df
```

### Ejemplo: Crear Features de Forma

```python
def create_form_features(self, windows=[5, 10]):
    """Crea features de forma reciente"""
    for team in self.df['HomeTeam'].unique():
        # Obtener resultados del equipo
        team_matches = self.get_team_matches(team)
        
        for window in windows:
            # Últimos N partidos
            points = self.calculate_points(team_matches, window)
            self.df.loc[team_mask, f'form_{window}'] = points / (window * 3)
```

---

## 📈 Pipeline de Feature Engineering

```
1. DATOS BRUTOS
   ↓
   [CSV con partidos históricos]
   
2. LIMPIEZA
   ↓
   - Eliminar nulos
   - Convertir fechas
   - Normalizar nombres de equipos
   
3. FEATURE CREATION
   ↓
   - Forma reciente (rolling windows)
   - Promedios de goles
   - Head-to-head
   - Posiciones en tabla
   
4. FEATURE SELECTION
   ↓
   - Eliminar features correlacionadas
   - Seleccionar las más importantes
   
5. NORMALIZACIÓN
   ↓
   - StandardScaler (media=0, std=1)
   - MinMaxScaler (rango 0-1)
   
6. DATASET FINAL
   ↓
   [X (features), y (target)]
   ↓
   MODELO ML
```

---

## 🎯 Features más Importantes

Según el análisis de importancia de features en EPL-Predict:

| Rank | Feature | Importancia | Descripción |
|------|---------|-------------|-------------|
| 1 | `market_home_prob` | 0.247 | Probabilidad de victoria local (mercado) |
| 2 | `home_form_5` | 0.132 | Forma local últimos 5 partidos |
| 3 | `away_form_5` | 0.118 | Forma visitante últimos 5 partidos |
| 4 | `home_goals_scored_avg` | 0.095 | Promedio goles marcados (local) |
| 5 | `position_diff` | 0.078 | Diferencia de posición en tabla |

> 💡 Las features del mercado (odds) son las más predictivas porque incorporan toda la información disponible públicamente.

---

## 🚨 Errores Comunes en Feature Engineering

### ❌ **Data Leakage** (Filtración de Datos)

**Problema**: Usar información del futuro para predecir el pasado.

```python
# ❌ MAL: Usa información de TODO el dataset
df['avg_goals'] = df.groupby('HomeTeam')['FTHG'].transform('mean')

# ✅ BIEN: Usa solo información HASTA ese momento
df['avg_goals'] = df.groupby('HomeTeam')['FTHG'].expanding().mean().shift(1)
```

### ❌ **Look-Ahead Bias**

```python
# ❌ MAL: Incluye el partido actual
form_5 = df.groupby('HomeTeam')['Points'].rolling(5).mean()

# ✅ BIEN: Excluye el partido actual (shift)
form_5 = df.groupby('HomeTeam')['Points'].rolling(5).mean().shift(1)
```

### ❌ **Usar Features Correlacionadas**

```python
# Si tienes estas features:
home_goals_scored_avg  # Goles marcados
home_form_5            # Forma (incluye goles)

# Pueden estar muy correlacionadas (>0.8)
# Elimina una para evitar redundancia
```

---

## 🔍 Validación de Features

### Código para Verificar Importancia

```python
from sklearn.ensemble import RandomForestClassifier

# Entrenar modelo
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

# Ver importancia de features
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance.head(10))
```

### Eliminar Features No Útiles

```python
# Features con importancia < 0.01
low_importance = feature_importance[feature_importance['importance'] < 0.01]
X_train = X_train.drop(columns=low_importance['feature'])
```

---

## 📚 Recursos Adicionales

- [Pandas Rolling Functions](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.rolling.html)
- [Feature Engineering for Machine Learning (Coursera)](https://www.coursera.org/learn/feature-engineering)
- Archivo del proyecto: [src/feature_engineering.py](../../src/feature_engineering.py)

---

## 🚀 Siguiente Paso

Continúa con [05_MODELOS_CLASIFICACION.md](05_MODELOS_CLASIFICACION.md) para aprender sobre los algoritmos que usan estas features.
