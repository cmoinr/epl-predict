@echo off
REM Script para ejecutar el dashboard Streamlit en Windows
REM Uso: run_streamlit.bat

echo.
echo ======================================
echo   EPL Predictor - Streamlit Dashboard
echo ======================================
echo.
echo 🚀 Iniciando la aplicacion...
echo.
echo La app abrira en tu navegador en: http://localhost:8501
echo.
echo Presiona Ctrl+C para detener el servidor
echo.
echo ======================================
echo.

streamlit run app.py --logger.level=info

pause
