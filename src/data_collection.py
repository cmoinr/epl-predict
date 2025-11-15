"""
Script para descargar y gestionar datasets de la Premier League
"""

import os
import sys
import json
from pathlib import Path
import pandas as pd
import requests
from typing import Optional


def get_data_paths():
    """Obtiene las rutas de datos del proyecto"""
    root = Path(__file__).parent.parent
    return {
        'raw': root / 'data' / 'raw',
        'processed': root / 'data' / 'processed'
    }


def create_data_dirs():
    """Crea directorios de datos si no existen"""
    paths = get_data_paths()
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def download_kaggle_dataset(dataset_name: str, file_name: str) -> Optional[str]:
    """
    Descarga un dataset de Kaggle usando Kaggle CLI
    
    Requisitos:
    - Instalar: pip install kaggle
    - Crear ~/.kaggle/kaggle.json con credenciales de Kaggle
    
    Args:
        dataset_name: Nombre del dataset (ej: 'vivovinco/english-premier-league-matches')
        file_name: Nombre del archivo CSV a descargar
    
    Returns:
        Ruta del archivo descargado o None si falla
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("❌ Kaggle CLI no instalado.")
        print("   Instala con: pip install kaggle")
        print("   Luego configura en: ~/.kaggle/kaggle.json")
        return None
    
    try:
        paths = create_data_dirs()
        raw_path = paths['raw']
        
        print(f"📥 Descargando dataset: {dataset_name}")
        
        api = KaggleApi()
        api.authenticate()
        
        # Descargar el dataset
        api.dataset_download_files(
            dataset_name,
            path=raw_path,
            unzip=True
        )
        
        file_path = raw_path / file_name
        
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"✅ Descarga exitosa: {file_path.name} ({size_mb:.2f} MB)")
            return str(file_path)
        else:
            print(f"⚠️  Archivo {file_name} no encontrado en la descarga")
            return None
            
    except Exception as e:
        print(f"❌ Error descargando dataset: {e}")
        return None


def load_epl_data(file_path: str) -> Optional[pd.DataFrame]:
    """
    Carga el dataset EPL desde archivo CSV
    
    Args:
        file_path: Ruta del archivo CSV
    
    Returns:
        DataFrame con los datos o None si falla
    """
    try:
        df = pd.read_csv(file_path)
        print(f"✅ Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
        return df
    except Exception as e:
        print(f"❌ Error cargando datos: {e}")
        return None


def inspect_epl_data(df: pd.DataFrame) -> None:
    """
    Inspecciona la estructura del dataset EPL
    
    Args:
        df: DataFrame con los datos
    """
    print("\n" + "="*60)
    print("INSPECCIÓN DEL DATASET EPL")
    print("="*60)
    
    print(f"\n📊 Dimensiones: {df.shape[0]} filas × {df.shape[1]} columnas")
    
    print(f"\n📋 Columnas:")
    for i, col in enumerate(df.columns, 1):
        dtype = df[col].dtype
        nulls = df[col].isnull().sum()
        print(f"   {i}. {col:<20} {str(dtype):<15} (nulls: {nulls})")
    
    print(f"\n📅 Primeras filas:")
    print(df.head())
    
    print(f"\n📊 Resumen estadístico:")
    print(df.describe())
    
    if 'Date' in df.columns or 'date' in df.columns:
        date_col = 'Date' if 'Date' in df.columns else 'date'
        print(f"\n📅 Rango de fechas: {df[date_col].min()} a {df[date_col].max()}")
    
    print(f"\n✓ Valores únicos por columna:")
    for col in df.columns:
        unique_count = df[col].nunique()
        if unique_count <= 20:
            print(f"   {col}: {unique_count} valores únicos")
            print(f"      {df[col].unique()}")
        else:
            print(f"   {col}: {unique_count} valores únicos")


def save_processed_data(df: pd.DataFrame, filename: str = 'epl_processed.csv') -> str:
    """
    Guarda datos procesados
    
    Args:
        df: DataFrame a guardar
        filename: Nombre del archivo
    
    Returns:
        Ruta del archivo guardado
    """
    paths = create_data_dirs()
    file_path = paths['processed'] / filename
    
    df.to_csv(file_path, index=False)
    print(f"✅ Datos guardados en: {file_path}")
    return str(file_path)


if __name__ == '__main__':
    print("🚀 Script de Recopilación de Datos - Premier League ML")
    print("="*60)
    
    # Crear directorios
    paths = create_data_dirs()
    print(f"✅ Directorios creados/verificados")
    print(f"   Raw data: {paths['raw']}")
    print(f"   Processed: {paths['processed']}")
