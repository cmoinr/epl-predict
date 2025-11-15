# 📊 Guía de Features - Premier League ML

## Features Base (del Dataset Original)

Estas son variables que ya existen en el dataset:

### Offensivas (Ataque)
- **HomeShots / AwayShots**: Total de tiros realizados
- **HomeShotsOnTarget / AwayShotsOnTarget**: Tiros al arco
- **HalfTimeHomeGoals / HalfTimeAwayGoals**: Goles en el primer tiempo

### Defensivas (Defensa)
- **HomeCorners / AwayCorners**: Córneres a favor
- **HomeFouls / AwayFouls**: Faltas cometidas
- **HomeYellowCards / AwayYellowCards**: Tarjetas amarillas
- **HomeRedCards / AwayRedCards**: Tarjetas rojas

---

## Features Derivadas (Creadas por Nosotros)

Estas variables se crean combinando información histórica y son MÁS PREDICTIVAS:

### 1. **Form** (Forma Reciente)

**¿Qué es?**
La forma de un equipo en los últimos partidos.

**Cómo se calcula:**
```
Puntos = (Victorias × 3) + (Empates × 1) + (Derrotas × 0)
Form = Puntos promedio en últimos 5 partidos
```

**Ejemplo:**
```
Últimos 5 partidos: W, W, D, L, W
Puntos: 3 + 3 + 1 + 0 + 3 = 10
Form = 10 / 5 = 2.0 (puntos promedio)
```

**Por qué importa:** Un equipo que ganó sus últimos partidos es más probable que gane el siguiente.

**Columnas:**
- `HomeTeam_Form`: Form del equipo local
- `AwayTeam_Form`: Form del equipo visitante

---

### 2. **Head-to-Head (H2H)** (Histórico Directo)

**¿Qué es?**
El histórico de enfrentamientos entre dos equipos.

**Ejemplo:**
```
Liverpool vs Manchester City (últimos 5 enfrentamientos):
  Manchester City ganó 3 de 5
  Liverpool ganó 1 de 5
  Empataron 1
```

**Por qué importa:** Algunos equipos tienen "mala suerte" contra otros (jinx) aunque en general sean mejores.

**Columnas:**
- `H2H_HomeTeamWins`: % victorias del local en H2H
- `H2H_Matches`: Cuántos H2H tenemos
- `H2H_GoalsFor`: Promedio de goles en H2H

---

### 3. **Goles Promedio** (Goal Statistics)

**¿Qué es?**
Promedio histórico de goles a favor y en contra.

**Ejemplo:**
```
Manchester City (últimos 10 partidos en casa):
  - Goles a favor promedio: 2.3
  - Goles en contra promedio: 0.8
```

**Por qué importa:** Nos da idea de si es equipo atacante o defensivo.

**Columnas:**
- `HomeGoalsFor`: Promedio goles a favor como local
- `HomeGoalsAgainst`: Promedio goles en contra como local
- `AwayGoalsFor`: Promedio goles a favor como visitante
- `AwayGoalsAgainst`: Promedio goles en contra como visitante

---

### 4. **Home Advantage** (Ventaja de Casa)

**¿Qué es?**
La diferencia de rendimiento entre jugar en casa vs visitante.

**Fórmula:**
```
HomeAdvantage = (Puntos en casa / Partidos en casa) 
              - (Puntos visitante / Partidos visitante)
```

**Ejemplo:**
```
Arsenal:
  En casa: 2.1 puntos promedio por partido
  Visitante: 1.5 puntos promedio por partido
  HomeAdvantage = 2.1 - 1.5 = 0.6
  (Tiene ventaja de jugar en casa)
```

**Por qué importa:** Algunos equipos se potencian mucho en casa.

**Columnas:**
- `HomeAdvantage`: Ventaja del equipo local en Emiratos

---

### 5. **Temporales** (Temporal Features)

**¿Qué son?**
Variables relacionadas con cuándo se juega el partido.

**Por qué importan:**
- Algunos equipos juegan mejor en ciertos meses
- El día de la semana afecta (partidos de miércoles = cansancio)
- La season (temporada) muestra evolución

**Columnas:**
- `Month`: Mes (1-12)
- `DayOfWeek`: Día semana (0=Lunes, 6=Domingo)
- `Season_Year`: Año

---

## 📈 Matriz de Correlación

Una vez creadas todas las features, usamos **correlación** para ver cuáles predicen mejor los goles/resultados:

```
Correlación fuerte (0.8+):     Muy predictivo
Correlación media (0.4-0.7):   Algo predictivo
Correlación débil (0.0-0.3):   Poco predictivo
```

---

## 🎯 Cómo se Usan en ML

```
MODELO ML
│
├─ INPUT: Features (X) - Son el "ojo" del modelo
│  ├─ HomeShots
│  ├─ AwayShots
│  ├─ HomeTeam_Form
│  ├─ HomeAdvantage
│  └─ ... (todas las features)
│
├─ PROCESS: El modelo aprende pesos (importancia) para cada feature
│
└─ OUTPUT: Predicción
   ├─ Resultado (1X2)
   └─ Goles totales
```

---

## 📊 Tabla Resumen

| Feature | Tipo | Rango Típico | Importancia |
|---------|------|--------------|------------|
| HomeShots | Base | 5-20 | Media |
| HomeTeam_Form | Derivada | 0-3 | ALTA |
| H2H_HomeTeamWins | Derivada | 0-1 (%) | ALTA |
| HomeGoalsFor | Derivada | 0.5-3.5 | ALTA |
| HomeAdvantage | Derivada | -1 a +1 | Media |
| Month | Temporal | 1-12 | Baja |
| DayOfWeek | Temporal | 0-6 | Baja |

---

## 🔧 Próximo Paso

Una vez creadas estas features, ejecutamos **modelos ML**:

1. **Random Forest**: Modelo 1 (baseline)
2. **Gradient Boosting**: Modelo 2 (mejor rendimiento)
3. **Comparar**: Cuál predice mejor resultados y goles

---

## 💡 Tips para Entender Features

**Pregunta clave:** "¿Esta variable me ayuda a predecir si el home team gana?"

✅ **SÍ importa:**
- Form reciente (último indicador de calidad)
- H2H (patrón histórico)
- Goles a favor/contra (indicador de capacidad)

❌ **NO importa:**
- Tarjetas rojas (raro ocurra, poco correlaciona)
- Día de la semana (en PL todos juegan cuando se programa)

---

Preguntas? Contesta en el notebook y revisamos juntos 📓
