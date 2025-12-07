#!/bin/bash
# Script para ejecutar el dashboard Streamlit
# Uso: bash run_streamlit.sh

echo "🚀 Iniciando EPL Predictor Dashboard..."
echo ""
echo "⏳ La app abrirá en tu navegador en: http://localhost:8501"
echo ""
echo "Presiona Ctrl+C para detener el servidor"
echo "---"
echo ""

streamlit run app.py --logger.level=info
