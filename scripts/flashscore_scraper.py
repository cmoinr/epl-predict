"""
Flashscore Web Scraper para Premier League
Extrae datos en tiempo real de partidos, resultados y odds

NOTA: Flashscore usa protecciones anti-scraping, por lo que este script
      incluye estrategias para evitar detección.
"""

import time
import random
from datetime import datetime
from pathlib import Path
import pandas as pd
from typing import Dict, List, Optional

try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options
    from selenium.common.exceptions import TimeoutException, NoSuchElementException
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    print("⚠️  Selenium no está instalado. Instala con: pip install selenium")

from bs4 import BeautifulSoup
import requests
import re


class FlashscoreScraper:
    """
    Scraper para Flashscore con múltiples estrategias de extracción
    """
    
    def __init__(self, use_selenium=True, headless=True):
        """
        Inicializa el scraper
        
        Args:
            use_selenium: Si True, usa Selenium para JS rendering
            headless: Si True, ejecuta Chrome en modo headless (sin ventana)
        """
        self.use_selenium = use_selenium and SELENIUM_AVAILABLE
        self.headless = headless
        self.driver = None
        
        # Headers para simular un navegador real
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'es-ES,es;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0'
        }
        
        if self.use_selenium:
            self._init_selenium()
    
    def _init_selenium(self):
        """Inicializa el driver de Selenium con opciones anti-detección"""
        try:
            chrome_options = Options()
            
            if self.headless:
                chrome_options.add_argument('--headless')
            
            # Opciones para evitar detección
            chrome_options.add_argument('--disable-blink-features=AutomationControlled')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--window-size=1920,1080')
            chrome_options.add_argument(f'user-agent={self.headers["User-Agent"]}')
            
            # Excluir la bandera de automatización
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            
            self.driver = webdriver.Chrome(options=chrome_options)
            
            # Ejecutar script para ocultar webdriver property
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            print("✅ Selenium inicializado correctamente")
            
        except Exception as e:
            print(f"❌ Error al inicializar Selenium: {e}")
            print("   Asegúrate de tener ChromeDriver instalado:")
            print("   https://chromedriver.chromium.org/downloads")
            self.use_selenium = False
    
    def _random_delay(self, min_seconds=1, max_seconds=3):
        """Espera aleatoria para simular comportamiento humano"""
        time.sleep(random.uniform(min_seconds, max_seconds))
    
    def _convert_date_format(self, raw_date):
        """
        Convierte formato de fecha de Flashscore a formato del dataset
        Input: "01.01.2026 13:30" 
        Output: "2026-01-01"
        """
        try:
            # Parsear fecha en formato DD.MM.YYYY HH:MM
            date_obj = datetime.strptime(raw_date, "%d.%m.%Y %H:%M")
            # Retornar en formato YYYY-MM-DD
            return date_obj.strftime("%Y-%m-%d")
        except Exception as e:
            print(f"      ⚠️  Error convirtiendo fecha '{raw_date}': {e}")
            # Si falla, retornar la fecha original
            return raw_date
    
    def get_premier_league_data(self, url='https://www.flashscore.com.ve/futbol/inglaterra/premier-league/'):
        """
        Extrae datos de la página principal de Premier League
        
        Returns:
            Dict con información extraída o None si falla
        """
        print(f"\n🔍 Intentando acceder a: {url}")
        
        if self.use_selenium:
            return self._get_data_with_selenium(url)
        else:
            return self._get_data_with_requests(url)
    
    def _get_data_with_selenium(self, url):
        """Extrae datos usando Selenium (recomendado para Flashscore)"""
        try:
            print("🌐 Cargando página con Selenium...")
            self.driver.get(url)
            
            # Esperar a que la página cargue
            self._random_delay(2, 4)
            
            # Esperar a que aparezcan elementos clave
            try:
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "event__match"))
                )
                print("✅ Página cargada correctamente")
            except TimeoutException:
                print("⚠️  Timeout esperando elementos - intentando continuar...")
            
            # Scroll para cargar contenido dinámico
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight/2);")
            self._random_delay(1, 2)
            
            # Obtener HTML
            html = self.driver.page_source
            soup = BeautifulSoup(html, 'html.parser')
            
            # Extraer partidos
            matches = self._parse_matches(soup)
            
            return {
                'success': True,
                'matches': matches,
                'timestamp': datetime.now().isoformat(),
                'url': url,
                'method': 'selenium'
            }
            
        except Exception as e:
            print(f"❌ Error con Selenium: {e}")
            return {
                'success': False,
                'error': str(e),
                'method': 'selenium'
            }
    
    def _get_data_with_requests(self, url):
        """Extrae datos usando requests básico (menos efectivo para Flashscore)"""
        try:
            print("🌐 Intentando con requests básico...")
            
            session = requests.Session()
            response = session.get(url, headers=self.headers, timeout=10)
            
            if response.status_code == 200:
                print(f"✅ Respuesta recibida (código: {response.status_code})")
                
                soup = BeautifulSoup(response.content, 'html.parser')
                matches = self._parse_matches(soup)
                
                return {
                    'success': True,
                    'matches': matches,
                    'timestamp': datetime.now().isoformat(),
                    'url': url,
                    'method': 'requests',
                    'note': 'Puede que no capture todo el contenido dinámico'
                }
            else:
                print(f"❌ Error: código de respuesta {response.status_code}")
                return {
                    'success': False,
                    'error': f'HTTP {response.status_code}',
                    'method': 'requests'
                }
                
        except Exception as e:
            print(f"❌ Error con requests: {e}")
            return {
                'success': False,
                'error': str(e),
                'method': 'requests'
            }
    
    def _parse_matches(self, soup):
        """
        Parsea partidos del HTML de Flashscore
        
        NOTA: Las clases CSS de Flashscore pueden cambiar.
              Este es un punto de partida que necesitará ajustes.
        """
        matches = []
        
        # SIEMPRE guardar HTML para debug (útil para ajustar selectores)
        debug_path = Path(__file__).parent.parent / 'data' / 'flashscore_debug.html'
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        debug_path.write_text(str(soup.prettify()), encoding='utf-8')
        print(f"💾 HTML guardado en: {debug_path}")
        
        # Intentar múltiples selectores (Flashscore cambia frecuentemente)
        possible_selectors = [
            'div.event__match',
            'div[class*="event__match"]',
            'div[id*="g_1_"]',
            'div.sportName',
            '[class*="event"]',
        ]
        
        elements = []
        for selector in possible_selectors:
            elements = soup.select(selector)
            if elements:
                print(f"✅ Encontrados {len(elements)} elementos con selector: {selector}")
                break
        
        if not elements:
            print("⚠️  No se encontraron partidos con los selectores conocidos")
            print("   El HTML de Flashscore puede haber cambiado")
            print(f"   Revisa el HTML en: {debug_path}")
            return []
        
        # Intentar extraer información de cada partido
        print(f"🔍 Procesando {len(elements)} elementos...")
        for i, element in enumerate(elements):
            try:
                match_data = self._extract_match_info(element)
                if match_data:
                    matches.append(match_data)
                    if i < 3:  # Mostrar primeros 3 para debug
                        print(f"   ✓ Partido {i+1}: {match_data.get('home_team', '?')} vs {match_data.get('away_team', '?')}")
            except Exception as e:
                if i < 3:  # Solo mostrar primeros errores
                    print(f"   ✗ Error procesando elemento {i+1}: {e}")
                continue
        
        print(f"\n📊 Total de partidos extraídos: {len(matches)}")
        return matches
    
    def _extract_match_info(self, element):
        """
        Extrae información de un elemento de partido
        
        NOTA: Esta función necesitará ajustes según la estructura actual de Flashscore
        """
        match_data = {}
        
        # Intentar extraer equipo local (múltiples variantes de clases)
        home_selectors = [
            'div.event__participant--home',
            'div[class*="participant"][class*="home"]',
            'span.participant-name--home',
            '.event__homeParticipant',
        ]
        for selector in home_selectors:
            home = element.select_one(selector)
            if home:
                match_data['home_team'] = home.get_text(strip=True)
                break
        
        # Intentar extraer equipo visitante
        away_selectors = [
            'div.event__participant--away',
            'div[class*="participant"][class*="away"]',
            'span.participant-name--away',
            '.event__awayParticipant',
        ]
        for selector in away_selectors:
            away = element.select_one(selector)
            if away:
                match_data['away_team'] = away.get_text(strip=True)
                break
        
        # Si no encontró con selectores específicos, buscar en el texto
        if not match_data.get('home_team') or not match_data.get('away_team'):
            # Buscar todos los divs de participantes
            participants = element.select('div[class*="participant"], span[class*="participant"]')
            if len(participants) >= 2:
                match_data['home_team'] = participants[0].get_text(strip=True)
                match_data['away_team'] = participants[1].get_text(strip=True)
        
        # Intentar extraer marcador
        score_selectors = [
            'div.event__score',
            'span.event__score',
            '[class*="event__score"]',
            'div.event__scores',
        ]
        for selector in score_selectors:
            score = element.select_one(selector)
            if score:
                match_data['score'] = score.get_text(strip=True)
                break
        
        # Intentar extraer tiempo/fecha
        time_selectors = [
            'div.event__time',
            'span.event__time',
            '[class*="event__time"]',
            'div.event__stage',
        ]
        for selector in time_selectors:
            match_time = element.select_one(selector)
            if match_time:
                match_data['time'] = match_time.get_text(strip=True)
                break
        
        # Extraer ID del partido si está disponible
        if element.get('id'):
            match_data['match_id'] = element.get('id')
        
        # Extraer clases para debug
        if element.get('class'):
            match_data['element_classes'] = ' '.join(element.get('class'))
        
        # Solo devolver si al menos tenemos equipos
        if 'home_team' in match_data and 'away_team' in match_data:
            return match_data
        
        return None
    
    def save_to_csv(self, data, output_path='data/raw/flashscore_data.csv'):
        """Guarda los datos extraídos en CSV"""
        if not data.get('success') or not data.get('matches'):
            print("❌ No hay datos para guardar")
            return False
        
        try:
            df = pd.DataFrame(data['matches'])
            
            output_file = Path(__file__).parent.parent / output_path
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"\n💾 Datos guardados en: {output_file}")
            print(f"   {len(df)} partidos exportados")
            
            return True
            
        except Exception as e:
            print(f"❌ Error al guardar CSV: {e}")
            return False
    
    def get_upcoming_matches_with_odds(self, max_matches=20):
        """
        Extrae partidos futuros CON ODDS desde la página de calendario
        
        Args:
            max_matches: Límite de partidos a procesar (para evitar timeouts)
        
        Returns:
            List de dicts con estructura: date, home_team, away_team, odds...
        """
        if not self.use_selenium:
            print("❌ Esta función requiere Selenium")
            return []
        
        url = 'https://www.flashscore.com.ve/futbol/inglaterra/premier-league/partidos/'
        print(f"\n🔍 Accediendo a página de partidos futuros...")
        print(f"   URL: {url}")
        
        try:
            self.driver.get(url)
            self._random_delay(3, 5)
            
            # Esperar que carguen los partidos
            try:
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "event__match"))
                )
            except TimeoutException:
                print("⚠️  Timeout esperando partidos")
            
            # Scroll para cargar más partidos
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight/2);")
            self._random_delay(1, 2)
            
            # Obtener todos los partidos programados (futuros)
            upcoming_elements = self.driver.find_elements(
                By.CSS_SELECTOR, 
                "div.event__match--scheduled"
            )
            
            print(f"✅ Encontrados {len(upcoming_elements)} partidos futuros")
            
            if len(upcoming_elements) == 0:
                print("⚠️  No se encontraron partidos futuros")
                return []
            
            # Limitar cantidad para evitar timeout
            upcoming_elements = upcoming_elements[:max_matches]
            print(f"📊 Procesando primeros {len(upcoming_elements)} partidos...")
            
            matches_data = []
            
            for idx in range(len(upcoming_elements)):
                print(f"\n🔄 Iteración {idx+1}/{len(upcoming_elements)}")
                try:
                    # RE-BUSCAR elementos en cada iteración para evitar stale elements
                    current_elements = self.driver.find_elements(
                        By.CSS_SELECTOR, 
                        "div.event__match--scheduled"
                    )[:max_matches]
                    print(f"   Elementos encontrados en esta iteración: {len(current_elements)}")
                    
                    if idx >= len(current_elements):
                        print(f"   ⚠️  idx {idx} >= len {len(current_elements)}, saliendo del loop")
                        break
                    
                    element = current_elements[idx]
                    
                    print(f"\n[{idx+1}/{len(current_elements)}] Procesando partido...")
                    
                    # Buscar el link dentro del elemento
                    try:
                        link = element.find_element(By.CSS_SELECTOR, "a.eventRowLink")
                        match_url = link.get_attribute('href')
                    except NoSuchElementException:
                        print(f"   ⚠️  No se encontró link para partido {idx+1}")
                        continue
                    
                    # Hacer clic y navegar a detalles
                    self.driver.execute_script("arguments[0].click();", link)
                    self._random_delay(2, 3)
                    
                    # EXTRAER TODOS LOS DATOS DESDE LA PÁGINA DE DETALLES
                    match_details = self._extract_match_details_from_page()
                    
                    if not match_details:
                        print(f"   ⚠️  No se pudieron extraer detalles del partido")
                        self.driver.back()
                        self._random_delay(1, 2)
                        continue
                    
                    home_team = match_details.get('home_team')
                    away_team = match_details.get('away_team')
                    match_date = match_details.get('date')
                    
                    if not home_team or not away_team:
                        print(f"   ⚠️  No se pudieron extraer equipos")
                        self.driver.back()
                        self._random_delay(1, 2)
                        continue
                    
                    print(f"   {home_team} vs {away_team} - {match_date}")
                    
                    # Extraer odds de la página de detalles
                    odds = self._extract_odds_from_match_page()
                    
                    # Compilar datos
                    match_data = {
                        'date': match_date,
                        'home_team': home_team,
                        'away_team': away_team,
                        'home_win_odds': odds.get('home_win', None),
                        'draw_odds': odds.get('draw', None),
                        'away_win_odds': odds.get('away_win', None),
                        'over_2_5_odds': odds.get('over_2_5', None),
                        'under_2_5_odds': odds.get('under_2_5', None),
                        'both_score_yes': odds.get('btts_yes', None),
                        'both_score_no': odds.get('btts_no', None),
                    }
                    
                    matches_data.append(match_data)
                    print(f"   ✓ Datos extraídos (Home: {odds.get('home_win', 'N/A')})")
                    
                    # Volver a la lista de partidos navegando directamente a la URL
                    print(f"   ← Volviendo a lista de partidos...")
                    self.driver.get("https://www.flashscore.com.ve/futbol/inglaterra/premier-league/partidos/")
                    self._random_delay(2, 3)
                    
                    # Esperar que la página de partidos se cargue
                    try:
                        WebDriverWait(self.driver, 10).until(
                            EC.presence_of_element_located((By.CSS_SELECTOR, "div.event__match--scheduled"))
                        )
                        print(f"   ✓ Lista de partidos recargada")
                    except TimeoutException:
                        print(f"   ⚠️  Timeout esperando recarga de lista")
                    
                    # Re-encontrar elementos (el DOM se refresca al volver)
                    upcoming_elements = self.driver.find_elements(
                        By.CSS_SELECTOR, 
                        "div.event__match--scheduled"
                    )[:max_matches]
                    print(f"   → Elementos disponibles después de recargar: {len(upcoming_elements)}")
                    
                except Exception as e:
                    print(f"   ✗ Error procesando partido {idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    # Intentar volver a la lista
                    try:
                        self.driver.get("https://www.flashscore.com.ve/futbol/inglaterra/premier-league/partidos/")
                        self._random_delay(1, 2)
                    except:
                        pass
                    continue
            
            print(f"\n✅ Total procesado: {len(matches_data)} partidos con odds")
            return matches_data
            
        except Exception as e:
            print(f"❌ Error general: {e}")
            return []
    
    def _extract_match_details_from_page(self):
        """
        Extrae detalles del partido desde la página de detalles
        Usa las clases fijas que identificó el usuario:
        - duelParticipant__startTime para fecha
        - participant__participantName para equipos
        """
        details = {}
        
        try:
            # Esperar a que cargue la página
            self._random_delay(1, 2)
            
            # Extraer fecha/hora
            try:
                date_elem = self.driver.find_element(
                    By.CSS_SELECTOR, 
                    "div.duelParticipant__startTime div"
                )
                raw_date = date_elem.text.strip()
                # Convertir formato de "01.01.2026 13:30" a "2026-01-01"
                details['date'] = self._convert_date_format(raw_date)
            except NoSuchElementException:
                print("      ⚠️  No se encontró fecha")
            
            # Extraer equipos (hay 2 elementos con esta clase)
            try:
                team_elements = self.driver.find_elements(
                    By.CSS_SELECTOR,
                    "a.participant__participantName"
                )
                
                if len(team_elements) >= 2:
                    details['home_team'] = team_elements[0].text.strip()
                    details['away_team'] = team_elements[1].text.strip()
                else:
                    print(f"      ⚠️  Solo se encontraron {len(team_elements)} equipos")
            except NoSuchElementException:
                print("      ⚠️  No se encontraron equipos")
            
            if details.get('home_team') and details.get('away_team'):
                print(f"      ✓ Detalles: {details['home_team']} vs {details['away_team']} - {details.get('date', 'N/A')}")
            
        except Exception as e:
            print(f"      ⚠️  Error extrayendo detalles: {e}")
        
        return details
    
    def _extract_team_from_element(self, element, side='home'):
        """Extrae nombre del equipo (home o away) de un elemento de partido"""
        selectors = [
            f'div.event__participant--{side}',
            f'[class*="participant"][class*="{side}"]',
        ]
        
        for selector in selectors:
            try:
                team_elem = element.find_element(By.CSS_SELECTOR, selector)
                return team_elem.text.strip()
            except NoSuchElementException:
                continue
        
        return None
    
    def _extract_date_from_element(self, element):
        """Extrae fecha del partido"""
        selectors = [
            'div.event__time',
            'span.event__time',
            '[class*="event__time"]',
        ]
        
        for selector in selectors:
            try:
                time_elem = element.find_element(By.CSS_SELECTOR, selector)
                return time_elem.text.strip()
            except NoSuchElementException:
                continue
        
        return None
    
    def _extract_odds_from_match_page(self):
        """
        Extrae odds de la página de detalles del partido
        Busca cuotas 1X2, O/U 2.5, BTTS
        """
        odds = {}
        
        try:
            # Esperar que cargue la página
            self._random_delay(1, 2)
            
            # Buscar y hacer clic en la pestaña de Cuotas
            try:
                # Buscar el botón con data-testid="wcl-tab" que contiene "Cuotas"
                odds_button = self.driver.find_element(
                    By.XPATH,
                    "//button[@data-testid='wcl-tab' and contains(., 'Cuotas')]"
                )
                print("      → Clic en pestaña de cuotas...")
                self.driver.execute_script("arguments[0].click();", odds_button)
                self._random_delay(2, 3)
            except NoSuchElementException:
                print("      ⚠️  No se encontró pestaña de cuotas")
                # Intentar con el link directo
                try:
                    odds_link = self.driver.find_element(
                        By.CSS_SELECTOR,
                        "a[href*='/cuotas/']"
                    )
                    self.driver.execute_script("arguments[0].click();", odds_link)
                    self._random_delay(2, 3)
                except:
                    pass
            
            # Extraer odds 1X2 con los selectores específicos
            odds.update(self._find_1x2_odds())
            
            # TODO: Extraer Over/Under 2.5
            odds.update(self._find_over_under_odds())
            
            # TODO: Extraer BTTS
            odds.update(self._find_btts_odds())
            
        except Exception as e:
            print(f"      ⚠️  Error extrayendo odds: {e}")
        
        return odds
    
    def _find_1x2_odds(self):
        """Busca odds de resultado 1X2 usando selectores específicos de Flashscore"""
        odds = {}
        
        try:
            # Esperar a que carguen los odds
            self._random_delay(1, 2)
            
            # Buscar divs con data-analytics-element específicos
            try:
                # Victoria Local (1)
                home_div = self.driver.find_element(
                    By.CSS_SELECTOR,
                    '[data-analytics-element="ODDS_COMPARIONS_ODD_CELL_1"]'
                )
                home_span = home_div.find_element(By.TAG_NAME, 'span')
                odds['home_win'] = float(home_span.text)
                
                # Empate (X)
                draw_div = self.driver.find_element(
                    By.CSS_SELECTOR,
                    '[data-analytics-element="ODDS_COMPARIONS_ODD_CELL_2"]'
                )
                draw_span = draw_div.find_element(By.TAG_NAME, 'span')
                odds['draw'] = float(draw_span.text)
                
                # Victoria Visitante (2)
                away_div = self.driver.find_element(
                    By.CSS_SELECTOR,
                    '[data-analytics-element="ODDS_COMPARIONS_ODD_CELL_3"]'
                )
                away_span = away_div.find_element(By.TAG_NAME, 'span')
                odds['away_win'] = float(away_span.text)
                
                print(f"      ✓ 1X2: {odds['home_win']} / {odds['draw']} / {odds['away_win']}")
                
            except NoSuchElementException as e:
                print(f"      ⚠️  No se encontraron odds 1X2 con selectores específicos")
                
                # Guardar HTML para debug
                debug_path = Path(__file__).parent.parent / 'data' / 'odds_page_debug.html'
                if not debug_path.exists():
                    soup = BeautifulSoup(self.driver.page_source, 'html.parser')
                    debug_path.write_text(str(soup.prettify()), encoding='utf-8')
                    print(f"      💾 HTML de odds guardado en: {debug_path}")
        
        except Exception as e:
            print(f"      ⚠️  Error buscando odds 1X2: {e}")
        
        return odds
    
    def _find_over_under_odds(self):
        """Busca odds Over/Under 2.5 usando selectores específicos de Flashscore"""
        odds = {}
        
        try:
            # Navegar a la pestaña Over/Under (Más de/Menos de)
            try:
                over_under_link = self.driver.find_element(
                    By.CSS_SELECTOR,
                    "a[href*='/mas-de-menos-de/']"
                )
                print("      → Navegando a Over/Under...")
                self.driver.execute_script("arguments[0].click();", over_under_link)
                self._random_delay(2, 3)
            except NoSuchElementException:
                print("      ⚠️  No se encontró pestaña Over/Under")
                return odds
            
            # Buscar la fila que contiene "2.5" en la columna Total
            try:
                # Buscar todas las filas
                rows = self.driver.find_elements(
                    By.CSS_SELECTOR,
                    'div.ui-table__row'
                )
                
                row_with_2_5 = None
                for row in rows:
                    try:
                        # Buscar span con data-testid="wcl-oddsValue" dentro de esta fila
                        value_span = row.find_element(
                            By.CSS_SELECTOR,
                            'span[data-testid="wcl-oddsValue"]'
                        )
                        if value_span.text.strip() == "2.5":
                            row_with_2_5 = row
                            print(f"      ✓ Encontrada fila con Total 2.5")
                            break
                    except NoSuchElementException:
                        continue
                
                if not row_with_2_5:
                    print("      ⚠️  No se encontró fila con 2.5 goles")
                    return odds
                
                # Extraer Over 2.5 de esta fila específica
                try:
                    over_link = row_with_2_5.find_element(
                        By.CSS_SELECTOR,
                        'a[data-analytics-element="ODDS_COMPARIONS_ODD_CELL_2"]'
                    )
                    over_span = over_link.find_element(By.CSS_SELECTOR, 'span:not([class*="icon"])')
                    odds['over_2_5'] = float(over_span.text)
                except (NoSuchElementException, ValueError) as e:
                    print(f"      ⚠️  No se encontró Over 2.5: {e}")
                
                # Extraer Under 2.5 de esta fila específica
                try:
                    under_link = row_with_2_5.find_element(
                        By.CSS_SELECTOR,
                        'a[data-analytics-element="ODDS_COMPARIONS_ODD_CELL_3"]'
                    )
                    under_span = under_link.find_element(By.CSS_SELECTOR, 'span:not([class*="icon"])')
                    odds['under_2_5'] = float(under_span.text)
                except (NoSuchElementException, ValueError) as e:
                    print(f"      ⚠️  No se encontró Under 2.5: {e}")
                
                if 'over_2_5' in odds and 'under_2_5' in odds:
                    print(f"      ✓ O/U 2.5: {odds['over_2_5']} / {odds['under_2_5']}")
                        
            except Exception as e:
                print(f"      ⚠️  Error buscando fila 2.5: {e}")
                return odds
        
        except Exception as e:
            print(f"      ⚠️  Error buscando O/U: {e}")
        
        return odds
    
    def _find_btts_odds(self):
        """Busca odds Both Teams To Score usando selectores específicos de Flashscore"""
        odds = {}
        
        try:
            # Navegar a la pestaña BTTS (Ambos Equipos Marcarán)
            try:
                btts_link = self.driver.find_element(
                    By.CSS_SELECTOR,
                    "a[href*='/ambos-equipos-marcaran/']"
                )
                print("      → Navegando a BTTS...")
                self.driver.execute_script("arguments[0].click();", btts_link)
                self._random_delay(2, 3)
            except NoSuchElementException:
                print("      ⚠️  No se encontró pestaña BTTS")
                return odds
            
            # BTTS solo tiene una fila, extraer directamente
            try:
                # BTTS Yes (Sí) - CELL_2
                btts_yes_link = self.driver.find_element(
                    By.CSS_SELECTOR,
                    'a[data-analytics-element="ODDS_COMPARIONS_ODD_CELL_2"]'
                )
                btts_yes_span = btts_yes_link.find_element(By.CSS_SELECTOR, 'span:not([class*="icon"])')
                odds['btts_yes'] = float(btts_yes_span.text)
            except (NoSuchElementException, ValueError) as e:
                print(f"      ⚠️  No se encontró BTTS Yes: {e}")
            
            # BTTS No - CELL_3
            try:
                btts_no_link = self.driver.find_element(
                    By.CSS_SELECTOR,
                    'a[data-analytics-element="ODDS_COMPARIONS_ODD_CELL_3"]'
                )
                btts_no_span = btts_no_link.find_element(By.CSS_SELECTOR, 'span:not([class*="icon"])')
                odds['btts_no'] = float(btts_no_span.text)
            except (NoSuchElementException, ValueError) as e:
                print(f"      ⚠️  No se encontró BTTS No: {e}")
            
            if 'btts_yes' in odds and 'btts_no' in odds:
                print(f"      ✓ BTTS: {odds['btts_yes']} / {odds['btts_no']}")
        
        except Exception as e:
            print(f"      ⚠️  Error buscando BTTS: {e}")
        
        return odds
    
    def close(self):
        """Cierra el driver de Selenium"""
        if self.driver:
            self.driver.quit()
            print("\n🔒 Driver de Selenium cerrado")


def main():
    """Función principal para pruebas"""
    print("="*70)
    print("🏴󠁧󠁢󠁥󠁮󠁧󠁿  FLASHSCORE SCRAPER - PREMIER LEAGUE")
    print("="*70)
    
    if not SELENIUM_AVAILABLE:
        print("\n❌ Selenium no está disponible")
        print("   Instala con: pip install selenium")
        print("   También necesitas ChromeDriver:")
        print("   https://chromedriver.chromium.org/downloads")
        return
    
    # Crear scraper
    scraper = FlashscoreScraper(use_selenium=True, headless=False)
    
    try:
        # Extraer datos
        url = 'https://www.flashscore.com.ve/futbol/inglaterra/premier-league/'
        data = scraper.get_premier_league_data(url)
        
        # Mostrar resultados
        print("\n" + "="*70)
        print("📊 RESULTADOS")
        print("="*70)
        
        if data.get('success'):
            print(f"✅ Extracción exitosa")
            print(f"   Método: {data.get('method')}")
            print(f"   Partidos encontrados: {len(data.get('matches', []))}")
            
            # Mostrar primeros partidos
            if data.get('matches'):
                print(f"\n📋 Primeros partidos:")
                for i, match in enumerate(data['matches'][:5], 1):
                    print(f"   {i}. {match.get('home_team', 'N/A')} vs {match.get('away_team', 'N/A')}")
                    if 'score' in match:
                        print(f"      Marcador: {match['score']}")
                    if 'time' in match:
                        print(f"      Tiempo: {match['time']}")
            
            # Guardar datos
            scraper.save_to_csv(data)
            
        else:
            print(f"❌ Extracción fallida")
            print(f"   Error: {data.get('error', 'Desconocido')}")
            
            if data.get('method') == 'selenium':
                print("\n💡 SUGERENCIAS:")
                print("   - Verifica que ChromeDriver esté instalado correctamente")
                print("   - Flashscore puede estar bloqueando el acceso")
                print("   - Intenta con headless=False para ver qué pasa")
        
    finally:
        scraper.close()
    
    print("\n" + "="*70)
    print("✅ PROCESO FINALIZADO")
    print("="*70)


if __name__ == '__main__':
    main()
