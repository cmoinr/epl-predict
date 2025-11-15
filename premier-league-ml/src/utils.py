"""
Funciones auxiliares para el proyecto
"""

import os
import json
from pathlib import Path


def get_project_root():
    """Obtiene la ruta raíz del proyecto"""
    return Path(__file__).parent.parent


def create_data_paths():
    """Crea rutas necesarias para datos"""
    root = get_project_root()
    paths = {
        'raw': root / 'data' / 'raw',
        'processed': root / 'data' / 'processed',
        'models': root / 'models'
    }
    
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    
    return paths


def load_config(config_name='config.json'):
    """Carga configuración desde JSON"""
    config_path = get_project_root() / config_name
    if config_path.exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    return {}


if __name__ == '__main__':
    print(f"Project root: {get_project_root()}")
    print(f"Data paths: {create_data_paths()}")
