"""
Script de prueba rápida para verificar acceso a Flashscore
"""

import sys
from pathlib import Path

# Agregar src al path
sys.path.append(str(Path(__file__).parent.parent))

from scripts.flashscore_scraper import FlashscoreScraper, SELENIUM_AVAILABLE

def quick_test():
    """Prueba rápida de acceso a Flashscore"""
    
    print("="*70)
    print("🧪 TEST RÁPIDO - FLASHSCORE SCRAPER")
    print("="*70)
    
    # Verificar disponibilidad de Selenium
    if not SELENIUM_AVAILABLE:
        print("\n❌ Selenium no está disponible")
        print("   Ejecuta: pip install selenium webdriver-manager")
        return
    
    print("\n✅ Selenium disponible")
    
    # Intentar con modo visible primero (para debug)
    print("\n📋 Configuración:")
    print("   - Modo: Ventana visible (para observar)")
    print("   - Método: Selenium + ChromeDriver")
    
    try:
        print("\n🚀 Iniciando scraper...")
        scraper = FlashscoreScraper(use_selenium=True, headless=False)
        
        print("\n🌐 Intentando acceder a Flashscore...")
        url = 'https://www.flashscore.com.ve/futbol/inglaterra/premier-league/'
        
        data = scraper.get_premier_league_data(url)
        
        # Resultados
        print("\n" + "="*70)
        print("📊 RESULTADOS DE LA PRUEBA")
        print("="*70)
        
        if data.get('success'):
            print("\n✅ ¡ÉXITO! Se pudo acceder a Flashscore")
            print(f"\n📈 Estadísticas:")
            print(f"   - Método usado: {data.get('method', 'N/A')}")
            print(f"   - Partidos encontrados: {len(data.get('matches', []))}")
            print(f"   - Timestamp: {data.get('timestamp', 'N/A')}")
            
            if data.get('matches'):
                print(f"\n🏆 Primeros 3 partidos extraídos:")
                for i, match in enumerate(data['matches'][:3], 1):
                    home = match.get('home_team', 'N/A')
                    away = match.get('away_team', 'N/A')
                    score = match.get('score', 'Sin marcador')
                    time = match.get('time', 'Sin tiempo')
                    
                    print(f"\n   {i}. {home} vs {away}")
                    print(f"      Marcador: {score}")
                    print(f"      Estado: {time}")
                
                # Guardar datos
                print(f"\n💾 Guardando datos...")
                if scraper.save_to_csv(data):
                    print("   Datos guardados exitosamente en data/raw/flashscore_data.csv")
                
            else:
                print("\n⚠️  No se encontraron partidos")
                print("   Posibles razones:")
                print("   - No hay partidos activos en este momento")
                print("   - Los selectores CSS necesitan actualización")
                print("   - Revisa el archivo flashscore_debug.html")
            
            print(f"\n💡 CONCLUSIÓN:")
            print("   ✅ El scraping es POSIBLE con Flashscore")
            print("   📝 Puede requerir ajustes en los selectores CSS")
            
        else:
            print("\n❌ No se pudo acceder correctamente")
            print(f"   Error: {data.get('error', 'Desconocido')}")
            
            print(f"\n🔍 ANÁLISIS:")
            if '403' in str(data.get('error', '')):
                print("   - Flashscore detectó el bot (Error 403)")
                print("   - Protección anti-scraping activa")
                print(f"\n💡 SOLUCIONES:")
                print("   1. Usar proxies rotativos")
                print("   2. Aumentar delays aleatorios")
                print("   3. Implementar rotación de User-Agents")
                print("   4. Considerar APIs oficiales")
                
            elif '503' in str(data.get('error', '')):
                print("   - Servicio temporalmente no disponible")
                print("   - Intenta de nuevo más tarde")
                
            elif 'ChromeDriver' in str(data.get('error', '')):
                print("   - Problema con ChromeDriver")
                print(f"\n💡 SOLUCIONES:")
                print("   1. Verifica que Chrome esté actualizado")
                print("   2. Reinstala webdriver-manager")
                print("   3. Descarga ChromeDriver manualmente:")
                print("      https://chromedriver.chromium.org/downloads")
            
            else:
                print("   - Error desconocido")
                print("   - Revisa los logs arriba para más detalles")
            
            print(f"\n⚠️  CONCLUSIÓN:")
            print("   El scraping puede estar bloqueado o requiere ajustes")
        
        scraper.close()
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Prueba interrumpida por el usuario")
        
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*70)
    print("✅ PRUEBA FINALIZADA")
    print("="*70)
    
    print("\n📚 PRÓXIMOS PASOS:")
    print("   1. Si funcionó: Integrar con get_value_bets.py")
    print("   2. Si falló: Revisar docs/FLASHSCORE_SCRAPER.md")
    print("   3. Alternativa: Usar APIs oficiales (The Odds API, etc)")


if __name__ == '__main__':
    quick_test()
