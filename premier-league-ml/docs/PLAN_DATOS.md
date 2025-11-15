# Plan de Datos - Premier League ML

## 1. Fuentes de Datos Disponibles

### Opción A: APIs Gratuitas (RECOMENDADO)
- **football-data.org**: API con datos históricos de ligas
  - Registración gratuita
  - Limite: 10 requests/minuto
  - Datos: Resultados, equipos, estadísticas

- **RapidAPI - Football**: Múltiples endpoints
  - Más de 1000 requests/día gratis
  - Datos completos de PL

- **ESPN API**: Datos de ESPN (sin documentación oficial)

### Opción B: Web Scraping
- **Wikipedia**: Histórico de temporadas PL
- **BBC Sport / Sky Sports**: (requiere parsing cuidadoso)
- **Understat.com**: Datos avanzados (xG, etc.)

### Opción C: Datasets Públicos
- **Kaggle**: Datasets completos de PL (varias temporadas)
  - Premier League Complete Dataset
  - Historical match data

## 2. Features Clave a Recopilar

### Features de Equipo
- **Form**: Últimos 5 partidos (W/D/L)
- **Posición**: Lugar actual en tabla
- **Puntos**: Total acumulado
- **Goles**: A favor y en contra
- **Diferencia de goles**: GF - GA
- **Casa/Visitante**: Rendimiento en cada categoría

### Features Históricos
- **Head-to-Head**: Últimos 5 enfrentamientos diretos
- **Rachas**: Victorias/derrotas consecutivas
- **Consistencia**: Varianza en resultados recientes

### Features Derivados
- **Fuerza Relativa**: Rating de equipo vs rival
- **Probabilidad Histórica**: % de victorias cuando se enfrentan

## 3. Tipos de Predicciones

### 1. Resultado Final (3 clases)
- Victoria Local (1)
- Empate (X)
- Victoria Visitante (2)

### 2. Goles Totales (binario/multiclase)
- Bajo/Alto (< 2.5 / >= 2.5)

### 3. Ambos Anotan (SÍ/NO)
- Prediction: ¿Ambos equipos meterán al menos 1 gol?

## 4. Métricas de Rendimiento

```
Accuracy: % de predicciones correctas
Precision: De predicciones positivas, cuántas son correctas
Recall: De casos positivos reales, cuántos predijo correctamente
F1-Score: Balance entre Precision y Recall
ROC-AUC: Curva ROC para modelos probabilísticos
```

## 5. Recomendaciones de Odds

Una vez tenemos predicciones probabilísticas:

```
Odd Implícita = 1 / Probabilidad Predicha

Ejemplo:
- Si predecimos 65% de probabilidad de victoria local
- Odd = 1 / 0.65 = 1.54
- Comparar con odds reales del mercado
- Si mercado ofrece 1.80 > 1.54 = Buena apuesta (value bet)
```

## 6. Datos Necesarios por Temporada

Para entrenar bien necesitamos:
- **Mínimo**: 2-3 temporadas completas (760 partidos)
- **Ideal**: 5-10 temporadas (3800-7600 partidos)
- **Mejor**: Últimas 20 temporadas para tendencias

## 7. Plan de Acción Inmediato

### PASO 1: Recopilación (Esta semana)
```
Elegir fuente de datos:
→ Opción A (API) es más fácil para comenzar
→ Usar football-data.org o Kaggle
```

### PASO 2: Exploración
```
Cargar 1-2 temporadas
Entender estructura de datos
Visualizar distribuciones
```

### PASO 3: Limpieza
```
Valores faltantes
Duplicados
Outliers
```

### PASO 4: Features
```
Calcular form, posición, H2H
Crear variables derivadas
```

### PASO 5: Modelado
```
Split train/test
Entrenar 3-4 algoritmos
Comparar rendimiento
```

## Decisiones Pendientes

[ ] ¿Qué temporadas queremos (últimas 2? últimas 5?)?
[ ] ¿Fuente principal de datos?
[ ] ¿Predicción principal: Resultado (1X2) o Goles Totales?
[ ] ¿Incluir datos de apuestas reales o solo predecir?

---

**Próximo paso**: Decidir fuente de datos e iniciar descarga
