# 📥 Cómo Obtener Más Datos de Odds Históricos

## 🎯 Objetivo

Expandir tu dataset de **380 partidos con odds** (4%) a **9,500+ partidos con odds** (100%)

---

## 🌐 Fuente Recomendada: football-data.co.uk

**Ventajas**:
- ✅ GRATIS
- ✅ Datos desde temporada 2000/01 hasta actualidad
- ✅ Mismo formato que `epl_odds.csv`
- ✅ Cuotas de múltiples casas (Bet365, William Hill, etc.)
- ✅ Actualizado semanalmente

**URL**: https://www.football-data.co.uk/englandm.php

---

## 📝 Instrucciones Paso a Paso

### Método 1: Descarga Manual (Más Fácil)

1. **Visita**: https://www.football-data.co.uk/englandm.php

2. **Descarga cada temporada** (formato CSV):
   ```
   Season 2000/01: https://www.football-data.co.uk/mmz4281/0001/E0.csv
   Season 2001/02: https://www.football-data.co.uk/mmz4281/0102/E0.csv
   Season 2002/03: https://www.football-data.co.uk/mmz4281/0203/E0.csv
   ...
   Season 2024/25: https://www.football-data.co.uk/mmz4281/2425/E0.csv
   ```

3. **Guardar archivos** en:
   ```
   data/raw/football-data/
   ├── E0_0001.csv  (2000/01)
   ├── E0_0102.csv  (2001/02)
   ├── E0_0203.csv  (2002/03)
   ...
   └── E0_2425.csv  (2024/25)
   ```

### Método 2: Descarga Automática (Script Python)

Crea el archivo `scripts/download_odds_data.py`:

```python
import requests
import pandas as pd
from pathlib import Path
import time

def download_football_data_odds():
    """Descarga datos históricos de football-data.co.uk"""
    
    base_url = "https://www.football-data.co.uk/mmz4281/{season}/E0.csv"
    output_dir = Path("data/raw/football-data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Temporadas desde 2000/01 hasta 2024/25
    seasons = [
        "0001", "0102", "0203", "0304", "0405",
        "0506", "0607", "0708", "0809", "0910",
        "1011", "1112", "1213", "1314", "1415",
        "1516", "1617", "1718", "1819", "1920",
        "2021", "2122", "2223", "2324", "2425"
    ]
    
    print("📥 Descargando datos de football-data.co.uk...")
    print(f"   Destino: {output_dir}")
    print()
    
    successful = 0
    failed = []
    
    for season in seasons:
        url = base_url.format(season=season)
        output_file = output_dir / f"E0_{season}.csv"
        
        try:
            print(f"   Descargando temporada 20{season[:2]}/20{season[2:]}...", end=" ")
            
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Guardar
            with open(output_file, 'wb') as f:
                f.write(response.content)
            
            # Verificar
            df = pd.read_csv(output_file)
            print(f"✅ {len(df)} partidos")
            successful += 1
            
            time.sleep(1)  # Ser respetuoso con el servidor
            
        except Exception as e:
            print(f"❌ Error: {e}")
            failed.append(season)
    
    print()
    print("="*60)
    print(f"✅ Descargadas: {successful}/{len(seasons)} temporadas")
    if failed:
        print(f"❌ Fallidas: {', '.join(failed)}")
    print("="*60)

if __name__ == '__main__':
    download_football_data_odds()
```

**Ejecutar**:
```bash
python scripts/download_odds_data.py
```

---

## 🔄 Integrar Nuevos Datos

Una vez descargadas todas las temporadas, ejecuta:

### Paso 1: Consolidar archivos CSV

```python
# scripts/consolidate_odds_data.py
import pandas as pd
from pathlib import Path

def consolidate_football_data():
    """Consolida todos los CSVs de football-data en uno solo"""
    
    input_dir = Path("data/raw/football-data")
    output_file = Path("data/raw/epl_odds_complete.csv")
    
    all_files = sorted(input_dir.glob("E0_*.csv"))
    
    print(f"📊 Consolidando {len(all_files)} archivos...")
    
    dfs = []
    for file in all_files:
        try:
            df = pd.read_csv(file)
            
            # Agregar columna de temporada
            season = file.stem.split('_')[1]
            season_label = f"20{season[:2]}/20{season[2:]}"
            df['Season'] = season_label
            
            dfs.append(df)
            print(f"   ✅ {season_label}: {len(df)} partidos")
            
        except Exception as e:
            print(f"   ❌ {file.name}: {e}")
    
    # Consolidar
    df_complete = pd.concat(dfs, ignore_index=True)
    df_complete.to_csv(output_file, index=False)
    
    print()
    print(f"✅ Consolidado: {len(df_complete):,} partidos con odds")
    print(f"   Guardado en: {output_file}")
    
    return df_complete

if __name__ == '__main__':
    consolidate_football_data()
```

### Paso 2: Re-ejecutar pipeline de integración

```bash
# Ahora con TODOS los datos de odds
python scripts/integrate_market_data.py
```

Esto actualizará:
- `epl_enriched_with_odds.csv` (ahora con ~9,500 partidos con odds)
- `epl_with_market_intelligence.csv` (features completas)

---

## 📊 Columnas Disponibles en football-data.co.uk

### Resultados y Estadísticas
```
Div, Date, HomeTeam, AwayTeam
FTHG, FTAG, FTR          # Full Time
HTHG, HTAG, HTR          # Half Time
HS, AS                   # Shots
HST, AST                 # Shots on Target
HC, AC                   # Corners
HF, AF                   # Fouls
HY, AY, HR, AR           # Cards
```

### Cuotas (Odds)
```
B365H, B365D, B365A      # Bet365
BWH, BWD, BWA            # Bet&Win
IWH, IWD, IWA            # Interwetten
PSH, PSD, PSA            # Pinnacle
WHH, WHD, WHA            # William Hill
VCH, VCD, VCA            # VC Bet
```

### Cuotas Asiáticas
```
Bb1X2                    # Betbrain number of BOs
BbMxH, BbAvH             # Max/Average home win odds
BbMxD, BbAvD             # Max/Average draw odds
BbMxA, BbAvA             # Max/Average away win odds
```

### Over/Under 2.5 Goles
```
BbOU                     # Number of BOs
BbMx>2.5, BbAv>2.5       # Max/Average over 2.5 goals
BbMx<2.5, BbAv<2.5       # Max/Average under 2.5 goals
```

---

## 🎯 Ventajas de Datos Completos

Con 9,500+ partidos con odds podrás:

### 1. **Entrenar Modelos Robustos**
```python
# Features de mercado disponibles para TODO el dataset
X_train con 26 features de mercado × 9,500 partidos
```

### 2. **Backtesting Realista**
```python
# Simular 25 temporadas de apuestas
ROI promedio: X%
Win rate: X%
Máximo drawdown: X%
```

### 3. **Análisis Temporal**
```python
# Evolución del mercado 2000-2025
# ¿Mercado más eficiente ahora?
# ¿Cambios en márgenes de casas?
```

### 4. **Especialización por Casa**
```python
# ¿Qué casa tiene mejores cuotas?
# ¿Diferencias entre Bet365, Pinnacle, etc.?
# ¿Cuál es la más "predecible"?
```

### 5. **Value Betting Rentable**
```python
# Con 9,500 partidos:
# - Encontrar nichos rentables
# - Optimizar kelly fraction
# - Validar edge sostenible
```

---

## 🔗 Fuentes Alternativas

### 1. **Kaggle Datasets**
- **Búsqueda**: "Premier League odds historical"
- **Ventaja**: Datasets pre-procesados
- **URL**: https://www.kaggle.com/datasets

### 2. **The Odds API**
- **Ventaja**: Odds en tiempo real
- **Costo**: Gratis hasta 500 requests/mes
- **URL**: https://the-odds-api.com/

### 3. **Betfair Exchange**
- **Ventaja**: Odds de intercambio (más precisas)
- **Requiere**: Cuenta de Betfair
- **URL**: https://www.betfair.com/

### 4. **Repositorios GitHub**
```bash
# Buscar:
git clone https://github.com/search?q=premier+league+odds
```

---

## ⚠️ Notas Importantes

### Compatibilidad de Datos
```python
# football-data.co.uk usa formato similar a epl_odds.csv
# Pero nombres de columnas pueden variar ligeramente

# Mapeo recomendado:
column_mapping = {
    'Date': 'MatchDate',
    'FTHG': 'FullTimeHomeGoals',
    'FTAG': 'FullTimeAwayGoals',
    'FTR': 'FullTimeResult',
    'B365H': 'Bet365_Home',
    'WHH': 'WilliamHill_Home',
    # ... etc
}
```

### Limpieza de Datos
```python
# Algunos archivos tienen inconsistencias
# - Nombres de equipos pueden variar
# - Columnas faltantes en temporadas antiguas
# - Formato de fecha diferente

# Usar scripts de limpieza antes de merge
```

### Valores Faltantes
```python
# No todas las temporadas tienen todas las casas
# Temporadas antiguas tienen menos opciones de odds

# Estrategia:
# - Usar promedios cuando hay múltiples casas
# - Imputar con odds de casas similares
# - Filtrar partidos sin odds mínimas
```

---

## ✅ Checklist de Integración

- [ ] Descargar datos de football-data.co.uk (25 temporadas)
- [ ] Consolidar en `epl_odds_complete.csv`
- [ ] Verificar compatibilidad con `epl_final.csv`
- [ ] Mapear nombres de equipos consistentes
- [ ] Ejecutar `scripts/integrate_market_data.py`
- [ ] Verificar cobertura: ¿9,500+ partidos con odds?
- [ ] Re-entrenar modelos con dataset completo
- [ ] Backtest en 10,000+ partidos
- [ ] Evaluar ROI y win rate
- [ ] Ajustar estrategia de value betting

---

## 🚀 Ejecución Rápida

```bash
# 1. Descargar datos
python scripts/download_odds_data.py

# 2. Consolidar
python scripts/consolidate_odds_data.py

# 3. Integrar con proyecto
python scripts/integrate_market_data.py

# 4. Analizar
python scripts/analyze_market_features.py

# 5. Re-entrenar modelos
python retrain_models_improved.py

# 6. Backtest completo
python scripts/backtest_value_betting.py
```

---

## 💡 Tip Pro

Si quieres odds **actualizadas semanalmente** para predicciones futuras:

```python
# Automatizar descarga de última semana
def update_current_season():
    url = "https://www.football-data.co.uk/mmz4281/2425/E0.csv"
    df = pd.read_csv(url)
    
    # Actualizar sample_odds.csv con próximos partidos
    # ...
```

---

**¡Buena suerte expandiendo tu dataset!** 🚀

Con 9,500+ partidos con odds, tu modelo tendrá la data necesaria para superar al mercado.
