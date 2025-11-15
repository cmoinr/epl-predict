# 🎓 QUICK START - Feature Engineering

## Tienes 9,380 partidos EPL listos ✅

### Lo que hemos hecho:

1. **EDA** ✅ - Exploración de datos completada
2. **Dataset**: 9,380 filas × 25 columnas (sin NaNs)
3. **Features**: 22 columnas originales
4. **Targets**: 
   - Resultado (1X2): Home Win, Draw, Away Win
   - Goles: Total de goles del partido

---

## 🚀 Qué Hacer Ahora

### Paso 1: Entiende qué son features

**Features = Variables que usa el ML para predecir**

Ejemplo simple:
```
¿Ganará Chelsea (Home) contra Fulham (Away)?

Features (información que le damos al modelo):
  - Chelsea jugó bien últimamente? (Form)
  - Históricamente, ¿Chelsea gana a Fulham? (H2H)
  - ¿Cuántos goles mete Chelsea? (GoalsAvg)
  - ¿Juega Chelsea en Stamford Bridge? (Home)
  
ML MODEL → "Sí, 72% probabilidad"
```

### Paso 2: Abre el notebook y ejecuta

```bash
cd /workspaces/codespaces-blank/premier-league-ml
jupyter notebook notebooks/01_eda_and_modeling.ipynb
```

### Paso 3: Busca la sección "3. Feature Engineering"

Hay 3 celdas Python principales:

**Celda 1: Crear Features**
```python
engineer = EPLFeatureEngineer(df_processed)
X, y_result, y_goals = engineer.engineer_features()
```

Esto crea:
- `X`: 30+ columnas con features
- `y_result`: El resultado (para predecir)
- `y_goals`: Goles totales (para predecir)

**Celda 2: Inspeccionar**
```python
print(X.columns.tolist())
print(X.describe())
```

Ver todas las features creadas.

**Celda 3: Preparar para Modelado**
```python
X_train_scaled, X_test_scaled = ... (split y normalización)
```

Preparar datos para entrenar modelos.

### Paso 4: Cuéntame qué ves

Cuando ejecutes las celdas:
- ¿Cuántas features se crearon? (debe ser ~30-40)
- ¿Hay NaNs? (debe ser 0)
- ¿Form tiene valores entre 0 y 3? (sí = correcto)

---

## 📊 Features que Se Crean

| Categoría | Features | Ejemplos |
|-----------|----------|----------|
| Base | 14 | HomeShots, AwayCorners, etc. |
| Form | 2 | HomeTeam_Form, AwayTeam_Form |
| H2H | 3 | H2H_HomeTeamWins, H2H_Matches |
| Goals | 8 | HomeGoalsFor, AwayGoalsAgainst |
| Temporal | 3 | Month, DayOfWeek, Season_Year |
| **Total** | **~30** | |

---

## 💡 Tips

### ✅ Lo que esperas ver:
- Form: valores entre 0 y 3 (puntos promedio)
- H2H: valores entre 0 y 1 (% victorias)
- HomeAdvantage: valores entre -1 y +1
- Sin NaNs o muy pocos (fillna() los maneja)

### ❌ Si ves problemas:
- **Muchos NaNs**: Normal en primeros partidos (no hay histórico)
- **Valores muy grandes**: Revisar normalización
- **Errores**: Ver que `feature_engineering.py` esté en `src/`

---

## 📚 Para Profundizar

Antes de ejecutar, lee 2 minutos:
- `GUIA_FEATURES.md` → Qué hace cada feature
- `EJECUTAR_FEATURES.md` → Cómo ejecutar paso a paso

---

## 🎯 En 5 Minutos

1. **Ejecuta Celda 1** → Crea features
2. **Ejecuta Celda 2** → Ve qué se creó
3. **Cuéntame** → Número de features, si hay NaNs

Luego pasamos al **Modelado ML** ✨

---

## Estado del Proyecto

```
FASE 1: EDA                    ✅ COMPLETADA
FASE 2: Feature Engineering    ← TÚ ESTÁS AQUÍ (ejecutar)
FASE 3: Modelado              ⏳ Próxima
FASE 4: Evaluación            ⏳ Próxima
FASE 5: Value Betting         ⏳ Próxima
```

---

¡Ejecuta las celdas y cuéntame! 🚀
