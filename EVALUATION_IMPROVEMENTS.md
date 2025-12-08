# EVALUACIÓN DE MEJORAS - PRIORIDAD ACTUALIZADA

## RESULTADO DEL TEST

**ADVERTENCIA**: Las nuevas features empeoraron significativamente el modelo:

| Métrica | Baseline | Con Mejoras | Cambio |
|---------|----------|-------------|--------|
| **Accuracy Global** | 74.03% | 61.85% | **-12.18%** ❌ |
| **Draw Accuracy** | 45.17% | 27.73% | **-17.44%** ❌ |
| **Goals MAE** | 0.837 | 0.940 | **+0.103** ❌ |

## PROBLEMA IDENTIFICADO

Las funciones agregadas generan **demasiados NaNs** en los primeros partidos (cold start):
- `add_h2h_draw_rate()`: Sin historial H2H → NaN
- `add_draw_tendency()`: Sin historial de equipo → NaN
- `add_strength_balance()`: Sin historial → NaN
- `add_weak_defense_flag()` y `add_strong_attack_flag()`: Sin historial → NaN

Al rellenar con 0 o valores default, se introduce **ruido** que confunde al modelo.

## SOLUCIÓN PROPUESTA

### OPCIÓN 1: Simplificar Features (RECOMENDADO)

En lugar de agregar 5 funciones nuevas, agregar **solo 2-3 features selectivas**:

1. **H2H_DrawRate** (pero con mejor manejo de NaN):
   - Si no hay historial H2H, usar DrawRate global del equipo
   - Si no hay historial del equipo, usar liga promedio (0.25)

2. **Strength_Balance** (simplificado):
   - Usar solo GoalsFor - GoalsAgainst de ventana deslizante
   - Ya existe algo similar en el dataset actual

3. **Attack_Defense_Mismatch** (NUEVA - más efectiva):
   - `HomeTeam_StrongAttack` AND `AwayTeam_WeakDefense` = probable goleada
   - Calcular solo cuando hay suficiente historial (>5 partidos)

### OPCIÓN 2: Usar Features Solo con Datos Suficientes

Modificar las funciones para:
```python
def add_feature_with_threshold(df, min_matches=5):
    """Solo calcula feature si hay >= min_matches de historial"""
    if len(historical_data) < min_matches:
        return DEFAULT_VALUE  # No NaN, sino valor razonable
    else:
        return calculated_feature
```

### OPCIÓN 3: Agregar Datos 2025/26 PRIMERO

**MEJOR ESTRATEGIA**: En lugar de mejorar features ahora, agregar datos 2025/26 primero:
- 14 jornadas adicionales = ~140 partidos nuevos
- Incremento de 1.5% en tamaño del dataset
- Datos más recientes pueden mejorar predicciones de draws naturalmente

## RECOMENDACIÓN FINAL

**NO implementar estas mejoras todavía**. Sigue esta secuencia:

1. **PASO 1**: Agregar datos 2025/26 jornadas 1-15
2. **PASO 2**: Re-entrenar con datos actualizados
3. **PASO 3**: Ejecutar `diagnose_models.py` de nuevo
4. **PASO 4**: Si draw accuracy sigue <50%, **entonces** implementar features selectivas (Opción 1)

### Beneficios de esta estrategia:
- Datos recientes son más relevantes que features complejas
- Evitamos overfitting con features que tienen muchos NaNs
- Mantenemos el modelo simple (Occam's Razor)
- 2025/26 puede tener patrones diferentes a 2000-2024

## CÓDIGO A MANTENER

Por ahora, **revertir** las mejoras de `feature_engineering.py` y mantener solo:
- Features base (28 features originales)
- Form, Goals Stats, Home Advantage, Shoot Stats

## PRÓXIMOS PASOS

```bash
# 1. Recolectar datos 2025/26
# (manual - desde FBRef, Understat, ESPN)

# 2. Agregar a epl_final.csv
python add_match_data.py  # (crear este script)

# 3. Re-entrenar
python retrain_models_improved.py

# 4. Comparar
python diagnose_models.py
```

Si después de agregar datos 2025/26 el draw accuracy mejora a 48-50%, ¡éxito!
Si sigue <45%, entonces implementar features selectivas (Opción 1).

---

**Lección aprendida**: Más features ≠ Mejor modelo. Calidad > Cantidad.
