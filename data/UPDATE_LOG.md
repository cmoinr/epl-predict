# Data Update Log

Registro de actualizaciones manuales al dataset EPL y reentrenamiento de modelos.

## Formato para documentar:

```
### [Fecha] - Actualización por [Quien]
- Fuente: [Sofascore/Flashscore/etc]
- Partidos agregados: X
- Temporada: 2024/25
- Rango de fechas: YYYY-MM-DD a YYYY-MM-DD
- Modelo entrenado: [GradientBoosting/RandomForest/etc]
- Métricas: Accuracy, Precision, Recall, F1-Score
- Notas: [Cualquier detalle importante]
```

## Historial:

### [2024-11-15] - Actualización Manual + Reentrenamiento
- Fuente: Sofascore/Flashscore (captura manual)
- Partidos agregados: 30
- Temporada: 2024/25
- Rango de fechas: 2024-10-26 a 2024-11-16
- Jornadas cubiertas: Jornada 9, 10, 11
- Dataset total: 9410 partidos (aumento de 30)

**Feature Engineering:**
- 24 features creadas (forma, goles, ventaja de casa, temporales, etc)
- Train: 7998 muestras | Test: 1412 muestras
- Split temporal: 85/15

**Modelos Entrenados:**
1. LogisticRegression: Accuracy=0.7174, F1=0.7095
2. RandomForest: Accuracy=0.7096, F1=0.6951
3. GradientBoosting: Accuracy=0.7415, F1=0.7357 ✅ MEJOR

**Modelo Seleccionado:**
- Tipo: GradientBoosting Classifier
- Accuracy: 74.15%
- Precision: 73.27%
- Recall: 74.15%
- F1-Score: 73.57%
- Archivo: `best_model_20251115_154031.pkl`

**Notas:**
- Los 30 nuevos partidos mejoraron la validación del modelo
- 3 partidos tuvieron inconsistencias menores en stats de shots (corregidos en validación)
- Recomendación: Continuar actualizando cada 3-4 jornadas para mantener modelo actualizado

