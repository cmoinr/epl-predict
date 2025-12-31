# 💰 Value Betting y Análisis de Odds

## ¿Qué es Value Betting?

**Value Betting** es apostar cuando tienes una **ventaja matemática** sobre el mercado.

```
Tu Modelo:      Predice Home Win 60%
Odds Mercado:   Home @1.80  →  Implica 55.6%

Ventaja (Edge): 60% - 55.6% = 4.4%
                 
¿Es Value Bet?  SÍ ✓ (tenemos edge positivo)
```

> 💡 **Concepto clave**: No es suficiente predecir correctamente. Debes predecir **mejor que el mercado**.

---

## 📊 Matemática de las Odds

### ¿Qué es una Odd?

Una **odd** es el pago por unidad apostada si ganas.

```
Odd @2.00  →  Si apuestas $100 y ganas, recibes $200
              (ganancia: $100)

Odd @1.50  →  Si apuestas $100 y ganas, recibes $150
              (ganancia: $50)

Odd @3.00  →  Si apuestas $100 y ganas, recibes $300
              (ganancia: $200)
```

### Conversión: Odds → Probabilidad

```python
# Probabilidad implícita = 1 / Odd

odd = 2.00
prob = 1 / odd = 0.50 = 50%

odd = 1.80
prob = 1 / odd = 0.556 = 55.6%

odd = 3.50
prob = 1 / odd = 0.286 = 28.6%
```

### Conversión: Probabilidad → Odds

```python
# Odd = 1 / Probabilidad

prob = 0.60 = 60%
odd = 1 / 0.60 = 1.67

prob = 0.40 = 40%
odd = 1 / 0.40 = 2.50

prob = 0.75 = 75%
odd = 1 / 0.75 = 1.33
```

---

## 🎯 Tipos de Mercados

### 1. **Match Result (1X2)**

```
Result    Odd (ejemplo)    Prob Implícita
─────────────────────────────────────────
Home      1.80             55.6%
Draw      3.50             28.6%
Away      4.50             22.2%
                          ─────────
Total Probabilidad:        106.4%
```

### 2. **Over/Under (Total Goles)**

```
Over 2.5   @1.95    Prob: 51.3%
Under 2.5  @1.90    Prob: 52.6%

Interpretación:
- Over 2.5: El partido tendrá 3+ goles
- Under 2.5: El partido tendrá 0-2 goles
```

### 3. **Both Teams To Score (BTTS)**

```
BTTS Yes   @1.85    Prob: 54.1%
BTTS No    @2.05    Prob: 48.8%

Interpretación:
- BTTS Yes: Ambos equipos anotan
- BTTS No: Un equipo no anota
```

### 4. **Handicaps Asiáticos**

```
Home -0.5  @1.90    = Home debe ganar (empate pierde)
Home 0.0   @1.90    = Home gana o empata (empate devolución)
Home +0.5  @2.10    = Home gana, empieza con +0.5

Arsenal -1.5 vs Chelsea:
Si Arsenal gana 1-0: Pierde (1-0-1.5 = -0.5)
Si Arsenal gana 2-0: Gana (2-0-1.5 = 0.5)
```

---

## 🏦 El Overround (Margen de la Casa)

### Problema: Probabilidades que Suman >100%

```
Cuotas 1X2:
Home:  1.80  →  55.6%
Draw:  3.50  →  28.6%
Away:  4.50  →  22.2%
              ──────────
Total:        106.4%

¿Por qué 106.4% y no 100%?
Porque la casa de apuestas se queda con la diferencia (6.4%)
```

### Cálculo del Overround

```python
overround = (prob_home + prob_draw + prob_away) - 1
overround = (0.556 + 0.286 + 0.222) - 1 = 0.064 = 6.4%

# También se llama "vig" (vigorish) o "juice"
```

### Ajuste de Probabilidades

Para comparar con el modelo, debemos **ajustar** las probabilidades:

```python
prob_home = 1 / 1.80 = 0.5556
prob_draw = 1 / 3.50 = 0.2857
prob_away = 1 / 4.50 = 0.2222
total = 1.0635

# Ajustes normalizados
prob_home_adj = 0.5556 / 1.0635 = 0.5225 = 52.25%
prob_draw_adj = 0.2857 / 1.0635 = 0.2687 = 26.87%
prob_away_adj = 0.2222 / 1.0635 = 0.2089 = 20.89%

# Total: 100% ✓
```

### Código en Python

```python
def calculate_implied_probability(odds_dict):
    """
    odds_dict: {'home': 1.80, 'draw': 3.50, 'away': 4.50}
    Devuelve probabilidades ajustadas
    """
    probs = {key: 1/odd for key, odd in odds_dict.items()}
    total = sum(probs.values())
    
    probs_adj = {key: prob/total for key, prob in probs.items()}
    
    return probs_adj, (total - 1)  # overround

# Uso
odds = {'home': 1.80, 'draw': 3.50, 'away': 4.50}
adj_probs, overround = calculate_implied_probability(odds)

print(f"Prob ajustadas: {adj_probs}")
print(f"Overround: {overround:.2%}")
```

---

## 📈 Cálculo de Value Bet

### Fórmula del EV (Expected Value)

```
EV = (Probabilidad Modelo × Ganancia) - (Probabilidad Pérdida × Apuesta)

EV = (P × (Odd - 1)) - ((1 - P) × 1)

EV = P × (Odd - 1) - (1 - P)
```

### Ejemplo Práctico

```
Modelo predice: Home Win 60%
Odd mercado:    1.80
Apuesta:        $100

EV = (0.60 × (1.80 - 1)) - (0.40 × 1)
EV = (0.60 × 0.80) - 0.40
EV = 0.48 - 0.40
EV = 0.08 = 8%

Por cada $100 apostados, esperas ganar $8 a largo plazo
```

### Interpretación

```python
EV > 0:  Value Bet ✓ (apostar es +EV)
EV = 0:  Fair Bet (indiferente)
EV < 0:  Bad Bet ✗ (apostar pierde dinero)

# Ejemplo anterior: EV = 0.08 > 0
# → Esta es una value bet
```

### Código Python

```python
def calculate_ev(prob_model, odd, stake=1.0):
    """
    Calcula el valor esperado de una apuesta
    
    Args:
        prob_model: Probabilidad según el modelo (0-1)
        odd: Cuota de la casa (e.g., 1.80)
        stake: Monto apostado (default: 1)
    
    Returns:
        ev: Valor esperado
        ev_pct: Porcentaje de valor esperado
    """
    ev = (prob_model * (odd - 1)) - ((1 - prob_model) * stake)
    ev_pct = (ev / stake) * 100
    
    return ev, ev_pct

# Uso
prob = 0.60
odd = 1.80
ev, ev_pct = calculate_ev(prob, odd, stake=100)

print(f"EV absoluto: ${ev:.2f}")
print(f"EV porcentaje: {ev_pct:.2f}%")
```

---

## 🎯 Estrategias de Value Betting

### Estrategia 1: Threshold de Confianza Mínima

```python
def find_value_bets(predictions, odds, min_ev=0.05):
    """
    Encuentra value bets con EV mínimo de 5%
    
    Args:
        predictions: {'home': 0.55, 'draw': 0.30, 'away': 0.15}
        odds: {'home': 1.80, 'draw': 3.50, 'away': 4.50}
        min_ev: EV mínimo requerido
    
    Returns:
        List de value bets
    """
    value_bets = []
    
    for outcome in ['home', 'draw', 'away']:
        prob = predictions[outcome]
        odd = odds[outcome]
        
        ev = (prob * (odd - 1)) - ((1 - prob) * 1)
        ev_pct = ev * 100
        
        if ev > min_ev:
            value_bets.append({
                'outcome': outcome,
                'probability': prob * 100,
                'odd': odd,
                'ev_pct': ev_pct
            })
    
    return sorted(value_bets, key=lambda x: x['ev_pct'], reverse=True)

# Uso
predictions = {'home': 0.60, 'draw': 0.25, 'away': 0.15}
odds = {'home': 1.80, 'draw': 3.50, 'away': 4.50}

value_bets = find_value_bets(predictions, odds, min_ev=0.05)

for bet in value_bets:
    print(f"{bet['outcome'].upper():6} | {bet['probability']:5.1f}% | "
          f"@{bet['odd']:.2f} | EV: {bet['ev_pct']:+.2f}%")
```

Output:
```
HOME   |  60.0% | @1.80 | EV: +8.00%
```

### Estrategia 2: Kelly Criterion (Apuesta Óptima)

**Kelly Criterion** nos dice cuánto apostar para maximizar ganancias.

```
Kelly % = (EV / (Odd - 1)) × 100

Ejemplo:
EV = 0.08 (8%)
Odd = 1.80

Kelly % = (0.08 / (1.80 - 1)) × 100
Kelly % = (0.08 / 0.80) × 100
Kelly % = 10%

→ Apostar el 10% del bankroll en esta apuesta
```

```python
def kelly_criterion(prob, odd, bankroll):
    """
    Calcula el monto óptimo a apostar usando Kelly Criterion
    
    Args:
        prob: Probabilidad del modelo
        odd: Cuota de la casa
        bankroll: Capital total disponible
    
    Returns:
        kelly_pct: Porcentaje a apostar
        kelly_stake: Monto a apostar
    """
    kelly_pct = (prob * (odd - 1) - (1 - prob)) / (odd - 1)
    kelly_stake = kelly_pct * bankroll
    
    return kelly_pct, kelly_stake

# Uso
prob = 0.60
odd = 1.80
bankroll = 1000  # $1000

kelly_pct, kelly_stake = kelly_criterion(prob, odd, bankroll)

print(f"Kelly %: {kelly_pct:.1%}")
print(f"Monto a apostar: ${kelly_stake:.2f}")
```

Output:
```
Kelly %: 10.0%
Monto a apostar: $100.00
```

### Nota: Fractional Kelly

Los traders profesionales usan **Fractional Kelly** (50% de Kelly) para menor riesgo:

```python
# En lugar de apostar el 10% (Full Kelly)
# Apostar el 5% (Half Kelly = 50% de Full Kelly)

stake_half_kelly = kelly_stake * 0.5  # $50 en lugar de $100
```

---

## 📊 Análisis de Múltiples Mercados

### Comparación: ML vs Mercado

```python
def compare_ml_vs_market(ml_probs, market_odds):
    """
    Compara predicciones del modelo vs probabilidades implícitas del mercado
    
    ml_probs: {'home': 0.55, 'draw': 0.30, 'away': 0.15}
    market_odds: {'home': 1.80, 'draw': 3.50, 'away': 4.50}
    """
    results = {}
    
    # Calcular probabilidades de mercado
    market_probs = {k: 1/v for k, v in market_odds.items()}
    total_prob = sum(market_probs.values())
    market_probs_adj = {k: v/total_prob for k, v in market_probs.items()}
    
    for outcome in ['home', 'draw', 'away']:
        ml_p = ml_probs[outcome]
        mkt_p = market_probs_adj[outcome]
        odd = market_odds[outcome]
        
        ev = (ml_p * (odd - 1)) - ((1 - ml_p) * 1)
        
        results[outcome] = {
            'ml_probability': ml_p,
            'market_probability': mkt_p,
            'difference': ml_p - mkt_p,
            'odd': odd,
            'ev': ev,
            'is_value': ev > 0
        }
    
    return results

# Uso
ml_probs = {'home': 0.55, 'draw': 0.30, 'away': 0.15}
odds = {'home': 1.80, 'draw': 3.50, 'away': 4.50}

comparison = compare_ml_vs_market(ml_probs, odds)

for outcome, data in comparison.items():
    print(f"\n{outcome.upper()}:")
    print(f"  ML:      {data['ml_probability']:.1%}")
    print(f"  Mercado: {data['market_probability']:.1%}")
    print(f"  Diff:    {data['difference']:+.1%}")
    print(f"  Odd:     {data['odd']:.2f}")
    print(f"  EV:      {data['ev']:+.4f}")
    print(f"  Value:   {'✓ YES' if data['is_value'] else '✗ NO'}")
```

---

## ⚠️ Pitfalls Comunes

### ❌ **Apostar en Cualquier Diferencia**

```python
# ❌ MAL: EV muy pequeño
ML: 55%, Mercado: 54% (diff 1%)
→ Margen de error en predicción > ganancia esperada

# ✅ BIEN: EV significativo (>3%)
ML: 60%, Mercado: 50% (diff 10%)
→ Margen suficiente para ganar a largo plazo
```

### ❌ **Ignorar el Tamaño del Bankroll**

```python
# ❌ MAL: Apostar todo a una bet
stake = bankroll  # Arriesgas todo

# ✅ BIEN: Fraccionar el bankroll
stake = bankroll * kelly_pct  # O kelly_pct * 0.5
```

### ❌ **Overconfidence en Predicciones**

```python
# Incluso con 60% de predicción:
# En 10 apuestas independientes:
# - 4 pueden ser pérdidas
# - Necesitas varias apuestas para ver ganancias

# Usar mínimo de apuestas antes de evaluar:
# n_min = 100 apuestas
```

---

## 📈 Backtesting Value Bets

```python
def backtest_value_bets(predictions, odds, results, min_ev=0.05):
    """
    Backtestea la estrategia de value betting
    
    Args:
        predictions: [{'home': 0.55, ...}, ...]
        odds: [{'home': 1.80, ...}, ...]
        results: [0, 1, 2, ...]  # 0=home, 1=draw, 2=away
    """
    total_bets = 0
    total_profit = 0
    winning_bets = 0
    
    mapping = {0: 'home', 1: 'draw', 2: 'away'}
    
    for i, (pred, odd, result) in enumerate(zip(predictions, odds, results)):
        outcome_name = mapping[result]
        
        # Calcular EV para outcome ganador
        prob = pred[outcome_name]
        ev = (prob * (odd[outcome_name] - 1)) - ((1 - prob) * 1)
        
        if ev > min_ev:
            total_bets += 1
            stake = 100  # Apuesta fija $100
            
            if result == mapping[result]:
                # Gano
                profit = stake * (odd[outcome_name] - 1)
                winning_bets += 1
            else:
                # Pierdo
                profit = -stake
            
            total_profit += profit
    
    win_rate = winning_bets / total_bets if total_bets > 0 else 0
    
    return {
        'total_bets': total_bets,
        'winning_bets': winning_bets,
        'win_rate': win_rate,
        'total_profit': total_profit,
        'avg_profit_per_bet': total_profit / total_bets if total_bets > 0 else 0
    }
```

---

## 💼 Casos de Uso en EPL-Predict

El proyecto usa value betting para:
1. **Identificar apuestas de valor** contra odds de mercado
2. **Filtrar predicciones de confianza baja** (EV < 0)
3. **Optimizar Kelly Criterion** para sizing de apuestas
4. **Backtestear estrategias** históricamente

---

## 📚 Recursos

- "Sports Betting Mathematics" - Edmond Tomaj
- [Pinnacle Sports - Value Betting](https://www.pinnacle.com/en/betting-articles/Betting-Strategy/value-betting)
- Kelly Criterion: [Investopedia](https://www.investopedia.com/terms/k/kellycriterion.asp)

---

## 🚀 Siguiente Paso

Continúa con [08_PIPELINE_COMPLETO.md](08_PIPELINE_COMPLETO.md) para ver cómo todos estos conceptos se integran en el flujo end-to-end.
