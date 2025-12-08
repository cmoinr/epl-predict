# DIAGNOSIS: Análisis de Errores y Plan de Mejora

**Fecha**: 8 Diciembre 2025  
**Modelos Analizados**: Gradient Boosting (GB) para todos

---

## 1. HALLAZGOS PRINCIPALES

### 📊 **MODELO RESULTADO 1X2 (74.03% Accuracy)**

**DEBILIDAD CRÍTICA: DRAWS (Empates)**
```
- Home Win:  83.23% accuracy ✅ (muy bueno)
- Away Win: 81.50% accuracy ✅ (muy bueno)
- DRAW:     45.17% accuracy ❌ (PROBLEMA CRÍTICO)
```

**Matriz de Confusión - Donde falla:**
```
El modelo confunde:
- 94 Draws → Predice Home (falso positivo home)
- 82 Draws → Predice Away (falso positivo away)
- Solo 145/321 draws correctamente predichos

Interpretación: El modelo es "optimista" en victorias.
Cuando no está seguro, predice victoria en lugar de empate.
```

**Confianza (Probabilidades):**
- Predicciones **CORRECTAS**: 80.68% confianza (muy bueno)
- Predicciones **INCORRECTAS**: 60.51% confianza (débil)
- **Diferencia**: 20.18% → El modelo SÍ detecta incertidumbre

**Partidos más difíciles:**
```
Liverpool vs Leeds       (Away, confianza 34.84%) - Muy incierto
Everton vs Wolves       (Error: Draw→Away, conf 34.97%)
Otros: Equipos raros o sorprendentes
```

---

### ⚽ **MODELO GOLES TOTALES (MAE: 0.837)**

**Rendimiento por Rango:**
```
0-1 goles:  MAE 0.911 ⚠️ (Peor - partidos cerrados)
2-3 goles:  MAE 0.679 ✅ (Mejor - estándar)
4+ goles:   MAE 0.992 ⚠️ (Peor - goleadas impredecibles)
```

**Problemas identificados:**
- 77 partidos con error > 2 goles (5.45%)
- Goleadas sorprendentes no se predicen bien
- Ejemplos:
  - Man City vs Leicester: Predijo 5.6, fue 9 (error +3.4)
  - Norwich vs Brentford: Predijo 1.1, fue 4 (error +2.9)

**Insight**: El modelo es conservador. Subestima partidos con baja defensa.

---

### 🥅 **MODELO BTTS (78.20% Accuracy)**

**Rendimiento por clase:**
```
NO (ambos no anotan): 77.14% accuracy ✅
SI (ambos anotan):    79.05% accuracy ✅

Balanceo: 44.6% NO vs 55.4% SI (relativamente balanceado)
```

**Confianza:**
- Correctos: 81.94% 
- Incorrectos: 66.32%
- Brecha: 15.62% (menor que resultado 1X2)

**Evaluación**: Este modelo está bien. No necesita muchas mejoras.

---

## 2. FEATURES MÁS IMPORTANTES

### Para RESULTADO 1X2:
1. **HomeAdvantage** (16.92%) - Ventaja de jugar en casa
2. **HomeTeam_Form** (16.80%) - Forma reciente del local
3. **AwayTeam_Form** (15.09%) - Forma reciente del visitante
4. **HalfTimeHomeGoals** (12.94%) - Goles en primer tiempo (local)
5. **HalfTimeAwayGoals** (10.15%) - Goles en primer tiempo (visita)

**Insight**: Los 5 top features explican ~72% de la importancia.
Oportunidad: Agregar features sobre:
- H2H histórico (1-2%)
- Momento del equipo (últimos 3 partidos vs 5)
- Datos contextuales (día de descanso, lesiones)

### Para GOLES TOTALES:
1. **HalfTimeHomeGoals** (33.40%) - Domina
2. **HalfTimeAwayGoals** (27.20%) - Domina
3. **HomeShotsOnTarget** (4.95%)
4. **HomeTeam_GoalsFor** (4.62%)
5. **AwayShotsOnTarget** (4.53%)

**Insight**: Los goles en primer tiempo explican 60%+ del resultado.
Esto es un problema: sin datos de primer tiempo, el modelo tendría solo 40% de poder.

**Mejora**: Entrenar modelo SEPARADO:
- Con datos de 1T para predicciones en vivo
- Con datos pre-partido para predicciones pre-match

---

## 3. RECOMENDACIONES DE MEJORA

### A. CORTO PLAZO (Antes de agregar datos 2025/26)

**PRIORITARIO 1: Mejorar predicción de DRAWS**
```python
# En feature_engineering.py, AGREGAR:

# 1. Head-to-Head draw rate
def add_h2h_draw_rate(df):
    h2h_draws = df.groupby(['HomeTeam','AwayTeam']).apply(
        lambda x: (x['FullTimeResult'] == 'D').sum() / len(x)
    )
    df['H2H_DrawRate'] = df.apply(
        lambda row: h2h_draws.get((row['HomeTeam'], row['AwayTeam']), 0.25),
        axis=1
    )
    return df

# 2. Team draw tendency
def add_draw_tendency(df):
    df['HomeTeam_DrawRate'] = df.groupby('HomeTeam')['FullTimeResult'].apply(
        lambda x: (x == 'D').sum() / len(x)
    ).reindex(df['HomeTeam']).values
    # Similar para AwayTeam
    return df

# 3. Strength similarity (equipos similares = más draws)
def add_strength_balance(df):
    df['Strength_Balance'] = abs(
        (df['HomeTeam_GoalsFor'] - df['HomeTeam_GoalsAgainst']) -
        (df['AwayTeam_GoalsFor'] - df['AwayTeam_GoalsAgainst'])
    )
    # Bajo balance = mayor probabilidad de draw
    return df
```

**PRIORITARIO 2: Mejorar goleadas**
```python
# Agregar feature de defensa débil:
def add_weak_defense_flag(df):
    # Identifica equipos que encajan muchos goles
    weak_teams = df.groupby('AwayTeam')['FullTimeAwayGoals'].mean() > 1.5
    df['AwayTeam_WeakDefense'] = df['AwayTeam'].isin(weak_teams).astype(int)
    return df

# Agregar feature de ataque fuerte:
def add_strong_attack_flag(df):
    strong_teams = df.groupby('HomeTeam')['FullTimeHomeGoals'].mean() > 2.0
    df['HomeTeam_StrongAttack'] = df['HomeTeam'].isin(strong_teams).astype(int)
    return df
```

---

### B. MEDIANO PLAZO (Datos 2025/26)

**¿QUÉ DATOS AGREGAR?**

Respetando estructura de `epl_final.csv`, necesitas:
```csv
MatchDate,HomeTeam,AwayTeam,FullTimeResult,FullTimeHomeGoals,FullTimeAwayGoals,
HalfTimeResult,HalfTimeHomeGoals,HalfTimeAwayGoals,
HomeShots,AwayShots,HomeShotsOnTarget,AwayShotsOnTarget,
HomeCorners,AwayCorners,HomeFouls,AwayFouls,
HomeYellowCards,AwayYellowCards,HomeRedCards,AwayRedCards
```

**PRIORIDAD de datos (por impacto esperado):**

1. **HalfTime** (Crítico - ya es 60% de importancia)
   - HalfTimeResult, HalfTimeHomeGoals, HalfTimeAwayGoals
   - ✅ Probablemente en Understat, ESPN

2. **Shooting Stats** (Importante)
   - HomeShots, AwayShots, HomeShotsOnTarget, AwayShotsOnTarget
   - ✅ Disponible en Understat, StatsBomb

3. **Set Pieces** (Medio)
   - HomeCorners, AwayCorners
   - ✅ Disponible en FBRef, Understat

4. **Disciplina** (Bajo)
   - Fouls, Yellow/Red Cards
   - ✅ Disponible en cualquier fuente

---

## 4. PLAN DE ACCIÓN PARA DATOS 2025/26

### FASE 1: RECOLECCIÓN (Jornadas 1-15)

**Fuentes recomendadas (gratuitas/freemium):**

| Fuente | Datos Disponibles | Facilidad | Coste |
|--------|-------------------|-----------|-------|
| **FBRef** | Half-time, shots, corners | Media | Gratis |
| **Understat** | xG, shots, corners, detailed | Alta | Gratis básico |
| **ESPN** | Half-time, shots, fouls | Alta | Gratis |
| **WhoScored** | xG, shots, progressive passes | Media | Gratis |

**Recomendación**: Empezar con FBRef (es la más estructurada).

### FASE 2: PROCESAMIENTO

```python
# Script: process_2025_26_data.py

def merge_2025_data():
    """
    1. Descargar jornadas 1-15 de FBRef (CSV)
    2. Renombrar columnas a coincidir con epl_final.csv
    3. Fusionar con datos históricos
    4. Validar estructura
    5. Re-entrenar modelos
    """
    
    historical = pd.read_csv('data/raw/epl_final.csv')
    new_data_2025 = pd.read_csv('temp/2025_26_fbref.csv')
    
    # Validar que tienen mismas columnas
    missing_cols = set(historical.columns) - set(new_data_2025.columns)
    if missing_cols:
        print(f"[WARN] Faltan columnas: {missing_cols}")
        # Rellenar con 0 o estimaciones
    
    # Fusionar
    merged = pd.concat([historical, new_data_2025], ignore_index=True)
    
    # Re-guardar
    merged.to_csv('data/raw/epl_final_updated.csv', index=False)
    return merged
```

### FASE 3: RE-ENTRENAMIENTO

Después de agregar jornadas 1-15:
```bash
python retrain_models_improved.py  # Usa datos completos
python diagnose_models.py          # Compara con baseline 74.03%
```

**Métrica de éxito**: 
- Resultado 1X2: de 74.03% → 76%+ (+2%)
- BTTS: mantener 78%+
- Goles: de 0.837 → 0.80 (-0.037)

---

## 5. INSIGHTS ADICIONALES

### ¿Por qué falla el modelo en draws?

**Hipótesis:**
1. Draws son eventos raros (26% de casos)
2. Muchos features favorecen Home/Away (HomeAdvantage importante)
3. Necesita features específicos de "equipos igualados"

**Solución:**
- Agregar H2H draw rate
- Agregar "strength balance"
- Usar SMOTE en entrenamiento

### ¿Por qué las goleadas sorprenden?

**Hipótesis:**
1. Sin datos contextuales (lesiones, forma aguda)
2. Equipos débiles enfrentan atacantes fuertes
3. Defensas débiles + Ataques fuertes = explosión

**Solución:**
- Agregar flags de "ataque fuerte" vs "defensa débil"
- Entrenar modelo separado para partidos de goleada
- Considerar datos de xG (Expected Goals)

---

## 6. TIMELINE RECOMENDADO

**Esta semana:**
- ✅ Diagnosis completado
- ⏳ Agregar H2H features (1 día)
- ⏳ Implementar SMOTE (0.5 días)

**Próxima semana:**
- ⏳ Recolectar datos 2025/26 jornadas 1-8
- ⏳ Procesar y validar estructura
- ⏳ Re-entrenar modelos

**Semana 3:**
- ⏳ Agregar jornadas 9-15
- ⏳ Comparar accuracy con baseline
- ⏳ Ajustar features si es necesario

**Meta final**: Accuracy 1X2 de 74% → 76-77% (+3%) con datos 2025/26

---

## CONCLUSIÓN

**Fortalezas del modelo actual:**
- ✅ Muy bueno en Home/Away wins (83%, 81%)
- ✅ BTTS muy preciso (78%)
- ✅ Detecta bien su propia incertidumbre

**Oportunidades de mejora:**
- ❌ Draws muy mal (45%) → Agregar H2H + draw tendency
- ❌ Goleadas impredecibles → Agregar strength balance
- ❌ Poco uso de features contextuales → Datos 2025/26 son clave

**Próximo paso**: Recolectar datos 2025/26 jornadas 1-15 y re-entrenar.

