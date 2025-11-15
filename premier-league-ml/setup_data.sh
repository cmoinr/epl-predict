#!/bin/bash
# Script para descargar y configurar el dataset de Premier League

set -e

echo "🚀 Configuración del Dataset EPL"
echo "=================================="
echo ""

# Crear directorio si no existe
mkdir -p data/raw
mkdir -p data/processed

echo "📁 Directorios creados:"
echo "   data/raw"
echo "   data/processed"
echo ""

# Verificar si el archivo ya existe
if [ -f "data/raw/epl_final.csv" ]; then
    echo "✅ Dataset epl_final.csv ya existe"
    echo ""
    echo "📊 Información del archivo:"
    wc -l data/raw/epl_final.csv | awk '{print "   Líneas: " $1}'
    ls -lh data/raw/epl_final.csv | awk '{print "   Tamaño: " $5}'
    echo ""
else
    echo "❌ Archivo epl_final.csv NO encontrado"
    echo ""
    echo "📥 Opciones para obtenerlo:"
    echo ""
    echo "OPCIÓN 1: Descargar desde Kaggle Web"
    echo "  1. Ir a https://www.kaggle.com/datasets"
    echo "  2. Buscar 'English Premier League EPL Match Data 2000-2025'"
    echo "  3. Click en 'Download'"
    echo "  4. Descomprimir en data/raw/"
    echo "  5. Renombrar a epl_final.csv si es necesario"
    echo ""
    echo "OPCIÓN 2: Usar Kaggle CLI"
    echo "  1. Instalar: pip install kaggle"
    echo "  2. Descargar credenciales: https://www.kaggle.com/account"
    echo "  3. Ejecutar:"
    echo "     kaggle datasets download -d vivovinco/english-premier-league-matches"
    echo "  4. Descomprimir en data/raw/"
    echo ""
    exit 1
fi

echo "🎯 Próximo paso: Ejecutar EDA notebook"
echo "   jupyter notebook notebooks/01_eda_and_modeling.ipynb"
