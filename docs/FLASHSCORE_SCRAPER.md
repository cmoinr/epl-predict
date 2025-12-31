# 🌐 Flashscore Web Scraper

## 📖 Descripción

Sistema completo de scraping para extraer datos de partidos futuros de la Premier League desde Flashscore.com, incluyendo todas las cuotas de apuestas (1X2, Over/Under 2.5, y BTTS). **Estado: 100% Operativo**

## ⚠️ Consideraciones Importantes

### Protecciones Anti-Scraping
Flashscore implementa varias medidas de protección:
- **JavaScript Dinámico**: El contenido se carga con JS, no directamente en el HTML
- **Cloudflare**: Posible protección anti-bots
- **Rate Limiting**: Limitaciones en número de requests
- **User-Agent Detection**: Detecta bots por headers

### Legalidad y Ética
⚖️ **IMPORTANTE**:
- Revisa los Términos de Servicio de Flashscore
- El scraping puede estar prohibido en sus términos
- Considera usar APIs oficiales si están disponibles
- Respeta los delays y no sobrecargues el servidor
- Es solo para uso educativo/personal

## 🚀 Instalación

### 1. Instalar Dependencias Python
```bash
pip install -r requirements.txt
```

### 2. Instalar ChromeDriver

#### Opción A: Automática (Recomendado)
El script intentará usar `webdriver-manager` que instala ChromeDriver automáticamente.

#### Opción B: Manual
1. Ve a https://chromedriver.chromium.org/downloads
2. Descarga la versión que coincida con tu Chrome
3. Extrae el ejecutable
4. Añádelo al PATH o colócalo en el directorio del proyecto

**Verificar versión de Chrome**:
- En Chrome: `chrome://settings/help`
- Descarga el ChromeDriver correspondiente

## 📋 Uso

### Ejecución Básica (Recomendado)
```bash
# Extraer 10 partidos (modo headless por defecto)
python scripts/extract_upcoming_odds.py --max 10

# Extraer 5 partidos con navegador visible (para debug)
python scripts/extract_upcoming_odds.py --max 5 --visible

# Extraer en archivo personalizado
python scripts/extract_upcoming_odds.py --max 20 --output data/processed/my_odds.csv
```

### Uso Programático
```python
from scripts.flashscore_scraper import FlashscoreScraper

# Crear scraper con ventana visible (para debug)
scraper = FlashscoreScraper(use_selenium=True, headless=False)

# Extraer partidos futuros con odds completos
matches = scraper.get_upcoming_matches_with_odds(max_matches=10)

# Ver resultados
for match in matches:
    print(f"{match['date']} - {match['home_team']} vs {match['away_team']}")
    print(f"  1X2: {match['home_win_odds']}/{match['draw_odds']}/{match['away_win_odds']}")
    print(f"  O/U 2.5: {match['over_2_5_odds']}/{match['under_2_5_odds']}")
    print(f"  BTTS: {match['both_score_yes']}/{match['both_score_no']}")

# Cerrar navegador
scrapeCompletamente Implementado
- [x] Extracción con Selenium (renderiza JavaScript)
- [x] Headers anti-detección
- [x] Delays aleatorios (simula comportamiento humano)
- [x] Ocultación de propiedades de WebDriver
- [x] Navegación automática entre pestañas de odds
- [x] **Extracción completa de odds 1X2** (Victoria Local/Empate/Victoria Visitante)
- [x] **Extracción de odds Over/Under 2.5 goles**
- [x] **Extracción de odds BTTS** (Both Teams To Score)
- [x] Conversión de formato de fecha (DD.MM.YYYY HH:MM → YYYY-MM-DD)
- [x] Exportación a CSV compatible con `sample_odds_history.csv`
- [x] Procesamiento multi-partido con navegación confiable
- [x] Estadísticas de cobertura de datos (100% en todos los campos)
- [x] Modo headless y visible para debugging
- [x] CLI con argumentos personalizables

### 🎯 Cobertura de Datos: 100%
Todos los campos extraídos con éxito:
- ✓ `date` (formato YYYY-MM-DD)
- ✓ `home_team`
- ✓ `away_team`
- ✓ `home_win_odds` (1)
- ✓ `draw_odds` (X)
- ✓ `away_win_odds` (2)
- ✓ `over_2_5_odds`
- ✓ `under_2_5_odds`
- ✓ `both_score_yes`
- ✓ `both_score_no`

### 🔮 Mejoras Futuras (Opcionales)
- [ ] Rotación de User-Agents
- [ ] Proxy support para mayor anonimato
- [ ] Extracción de más líneas de Over/Under (0.5, 1.5, 3.5)
- [ ] Extracción de estadísticas H2Hupcoming_odds.py`
Script CLI para ejecución práctica:
- Argumentos de línea de comandos
- Generación automática de archivos CSV
- Estadísticas de cobertura de datos
- Manejo de errores robusto

## 🎯 Funcionalidades

### ✅ Implementado (Formato Python)
```python
[
    {
        'date': '2026-01-01',
        'home_team': 'Liverpool',
        'away_team': 'Manchester City',
        'home_win_odds': 2.18,
        'draw_odds': 3.35,
        'away_win_odds': 3.50,
        'over_2_5_odds': 2.10,
        'under_2_5_odds': 1.70,
        'both_score_yes': 1.83,
        'both_score_no': 1.90
    },
    # ... más partidos
]
```

### Formato CSV Generado
El archivo CSV tiene el siguiente formato (compatible con `sample_odds_history.csv`):

```csv
date,home_team,away_team,home_win_odds,draw_odds,away_win_odds,over_2_5_odds,under_2_5_odds,both_score_yes,both_score_no
2026-01-01,Crystal Palace,Fulham,2.18,3.35,3.5,2.1,1.7,1.83,1.9
2026-01-01,Liverpool,Leeds Utd,1.55,4.5,5.75,1.6,2.3,1.7,2.1
2026-01-01,Brentford,Tottenham,2.24,3.55,3.2,1.85,1.9,1.73,2.05
```

### Nombre del Archivo
Los archivos se guardan automáticamente con timestamp:
- **Ubicación**: `data/processed/`
- **Formato**: `upcoming_odds_YYYYMMDD_HHMMSS.csv`
- **Ejemplo**: `upcoming_odds_20251231_181637.csv 📊 Estructura de Datos

### Datos Extraídos
```python
{
    � Detalles Técnicos

### Selectores CSS Utilizados
El scraper usa selectores específicos de Flashscore identificados mediante inspección:

#### 1. Listado de Partidos
- `div.event__match--scheduled` - Partidos futuros programados
- `a.eventRowLink` - Link a detalles del partido

#### 2. Detalles del Partido
- `div.duelParticipant__startTime div` - Fecha y hora
- `a.participant__participantName` - Nombres de equipos (2 elementos)

#### 3. Odds 1X2
- `[data-analytics-element="ODDS_COMPARIONS_ODD_CELL_1"]` - Victoria Local
- `[data-analytics-element="ODDS_COMPARIONS_ODD_CELL_2"]` - Empate
- `[data-analytics-element="ODDS_COMPARIONS_ODD_CELL_3"]` - Victoria Visitante

#### 4. Odds Over/Under
- URL: `/mas-de-menos-de/`
- Busca fila con `span[data-testid="wcl-oddsValue"]` = "2.5"
- Extrae de esa fila: CELL_2 (Over) y CELL_3 (Under)

#### 5. Odds BTTS
- URL: `/ambos-equipos-marcaran/`
- `[data-analytics-element="ODDS_COMPARIONS_ODD_CELL_2"]` - BTTS Yes
- `[data-analytics-element="ODDS_COMPARIONS_ODD_CELL_3"]` - BTTS No

### Flujo de Navegación
```
1. Ir a: https://www.flashscore.com.ve/futbol/inglaterra/premier-league/partidos/
2. Para cada partido:
   a. Click en partido → Ir a página de detalles
   b. Extraer fecha y equipos
   c. Click en pestaña "Cuotas" → Extraer 1X2
   d. Click en pestaña "Más de/Menos de" → Extraer O/U 2.5
   e. Click en pestaña "Ambos equipos marcarán" → Extraer BTTS
   f. Volver a lista: driver.get(URL_PARTIDOS)
3. Compilar todos los datos en lista
4. Exportar a CSV
```

### Conversión de Fecha
```python
# Input de Flashscore: "01.01.2026 13:30"
# Output para dataset: "2026-01-01"

def _convert_date_format(raw_date):
    date_obj = datetime.strptime(raw_date, "%d.%m.%Y %H:%M")
    return date_obj.strftime("%Y-%m-%d")
```

## �'success': True,
    'matches': [
        {
            'home_team': 'Manchester City',
            'away_team': 'Liverpool',
            'score': '2-1',
            'time': '90+3'
        },
        # ...
    ],
    'timestamp': '2025-12-30T10:30:00',
    'url': 'https://www.flashscore.com...',
    'method': 'selenium'
}
```

## 🛠️ Troubleshooting

### Error: ChromeDriver no encontrado
```
❌ Error al inicializar Selenium: chromedriver not found
```

**Solución**:
1. Verifica que ChromeDriver esté en el PATH
2. O usa `webdriver-manager`: ya está en requirements.txt
3. Actualiza Chrome a la última versión

### Error: Timeout esperando elementos
```
⚠️ Timeout esperando elementos
```

**Posibles causas**:
- Flashscore ha cambiado su estructura HTML
- Protección anti-bot está bloqueando
- Internet lento

**Solución**:
1. Ejecuta con `headless=False` para ver qué pasa
2. Revisa el archivo `flashscore_debug.html` generado
3. Actualiza los selectores CSS en `_parse_matches()`

### Error: HTTP 403 o 503
```
❌ Error: código de respuesta 403
```

**Causa**: Flashscore detectó el bot

**Solución**:
1. Aumenta los delays aleatorios
2. Usa proxies rotativos
3. Considera alternativas (API oficial)

### Elementos no se encuentran
```
⚠️ No se encontraron partidos con los selectores conocidos
```

**Causa**: Flashscore cambió su HTML/CSS

**Solución**:
1. Revisa `flashscore_debug.html`
2. Inspecciona la página real en Chrome DevTools
3. AUso con get_value_bets.py
```python
import pandas as pd
from scripts.extract_upcoming_odds import extract_upcoming_with_odds

# 1. Extraer odds de Flashscore
print("🔍 Extrayendo odds de Flashscore...")
matches = extract_upcoming_with_odds(max_matches=10)

# 2. Convertir a DataFrame
df = pd.DataFrame(matches)

# 3. Guardar o usar directamente
df.to_csv('data/processed/live_odds.csv', index=False)

# 4. Usar con el predictor
# El formato es compatible con sample_odds_history.csv
# por lo que puedes usarlo directamente en tus modelos
```

### Automatización Diaria

#### Windows (Task Scheduler)
1. Abrir "Programador de tareas"
2. Crear tarea básica
3. Configurar:
   -� Resultados de Prueba

### Última Ejecución (31/12/2025)
```
✅ Configuración: 10 partidos, modo headless
✅ Total procesado: 10/10 partidos (100%)
✅ Cobertura de datos:
   - home_win_odds    : 10/10 (100.0%)
   - draw_odds        : 10/10 (100.0%)
   - away_win_odds    : 10/10 (100.0%)
   - over_2_5_odds    : 10/10 (100.0%)
   - under_2_5_odds   : 10/10 (100.0%)
   - both_score_yes   : 10/10 (100.0%)
   - both_score_no    : 10/10 (100.0%)

✅ Tiempo aproximado: ~30 segundos por partido
✅ Sin errores ni bloqueos de Cloudflare
```

### Ejemplos de Datos Extraídos
```
2026-01-01, Crystal Palace vs Fulham
  1X2: 2.18 / 3.35 / 3.50
  O/U 2.5: 2.10 / 1.70
  BTTS: 1.83 / 1.90

2026-01-01, Liverpool vs Leeds Utd
  1X2: 1.55 / 4.50 / 5.75
  O/U 2.5: 1.60 / 2.30
  BTTS: 1.70 / 2.10
```

## 📞 Soporte

Si encuentras problemas:
1. Verifica que todas las dependencias estén instaladas (`pip install -r requirements.txt`)
2. Usa `--visible` para ver el navegador en acción
3. Revisa que ChromeDriver coincida con tu versión de Chrome
4. Verifica tu conexión a internet
5. Flashscore puede haber cambiado selectores (revisa con DevTools)
# Editar crontab
crontab -e

# Añadir línea (ejecutar diariamente a las 8 AM)
0 8 * * * cd /path/to/epl-predict && .venv/bin/python scripts/extract_upcoming_odds.py --max 20
```

### Script Wrapper (Recomendado)
Crea `scripts/update_live_data.py`:
```python
#!/usr/bin/env python3
"""
Script para actualizar odds diariamente
"""
from scripts.extract_upcoming_odds import extract_upcoming_with_odds
import pandas as pd
from datetime import datetime

def main():
    print(f"🕐 Actualización iniciada: {datetime.now()}")
    
    try:
        # Extraer 20 partidos
        matches = extract_upcoming_with_odds(max_matches=20)
        
        # Guardar con nombre fijo (sobrescribe el anterior)
        df = pd.DataFrame(matches)
        df.to_csv('data/processed/latest_odds.csv', index=False)
        
        print(f"✅ {len(matches)} partidos actualizados exitosamente")
        
    except Exception as e:
        print(f"❌ Error en actualización: {e}")
        # Aquí puedes enviar una notificación por email

if __name__ == '__main__':
    main()
#### 3. RapidAPI - Football
- API-FOOTBALL: Datos en tiempo real
- LiveScore API: Resultados en vivo

### Scraping Alternativo
Si Flashscore bloquea, prueba:
- **Sofascore.com**: Estructura similar, menos protecciones
- **BetExplorer.com**: Datos históricos de odds
- **Football-Data.co.uk**: CSV descargables (no requiere scraping)

## 💡 Mejores Prácticas

### 1. Respeta los Rate Limits
```python
# Añade delays entre requests
scraper._random_delay(min_seconds=2, max_seconds=5)
```

### 2. Cache los Resultados
```python
# No hagas múltiples requests para los mismos datos
# Guarda en CSV y reutiliza
```

### 3. Usa Headless Solo en Producción
```python
# Para desarrollo/debug: headless=False
scraper = FlashscoreScraper(headless=False)

# Para automatización: headless=True
scraper = FlashscoreScraper(headless=True)
```

### 4. Maneja Errores Gracefully
```python
try:
    data = scraper.get_premier_league_data()
except Exception as e:
    print(f"Error: {e}")
    # Fallback a datos cached o API alternativa
```

## 📈 Integración con el Proyecto

### Actualizar Odds en Tiempo Real
```python
# En get_value_bets.py o predict_match.py
from scripts.flashscore_scraper import FlashscoreScraper

def get_live_odds():
    scraper = FlashscoreScraper()
    data = scraper.get_premier_league_data()
    
    # Convertir a formato del proyecto
    if data['success']:
        # Actualizar sample_odds.csv
        pass
    
    scraper.close()
```

### Automatización Diaria
```bash
# Cron job (Linux/Mac)
0 8 * * * cd /path/to/epl-predict && python scripts/flashscore_scraper.py

# Task Scheduler (Windows)
# Crea una tarea que ejecute el script diariamente
```

## 📞 Soporte

Si encuentras problemas:
1. Verifica que todas las dependencias estén instaladas
2. Revisa los logs y el archivo debug HTML
3. Prueba con `headless=False` para ver qué ocurre
4. Considera usar APIs oficiales como alternativa

## ⚖️ Disclaimer

Este scraper es solo para propósitos educativos. El usuario es responsable de:
- Cumplir con los Términos de Servicio de Flashscore
- Respetar las leyes de scraping de su jurisdicción
- No sobrecargar los servidores de Flashscore
- Usar los datos de forma ética y legal

**Recomendación**: Usa APIs oficiales cuando sea posible.
