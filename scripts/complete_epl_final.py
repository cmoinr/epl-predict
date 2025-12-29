"""
Script para completar epl_final.csv con los partidos faltantes de 2003/04 y 2004/05
Usa epl_odds.csv como fuente ya que tiene los datos completos
"""

import pandas as pd
from pathlib import Path

def complete_epl_final():
    """Completa epl_final.csv con partidos faltantes de epl_odds.csv"""
    
    print("="*80)
    print("🔄 COMPLETANDO epl_final.csv CON PARTIDOS FALTANTES")
    print("="*80)
    print()
    
    # Cargar datasets
    df_final = pd.read_csv('data/raw/epl_final.csv')
    df_odds = pd.read_csv('data/raw/epl_odds.csv', low_memory=False)
    
    print("1️⃣ Estado inicial:")
    print(f"   epl_final.csv: {len(df_final)} partidos")
    print(f"   epl_odds.csv:  {len(df_odds)} partidos")
    print()
    
    # Crear backup
    backup_path = Path('data/raw/epl_final_before_completion.csv')
    if not backup_path.exists():
        df_final.to_csv(backup_path, index=False)
        print(f"   ✅ Backup creado: {backup_path.name}")
    print()
    
    # Mapeo de columnas epl_odds → epl_final
    column_mapping = {
        'Div': 'Div',
        'Season': 'Season',
        'Date': 'MatchDate',
        'HomeTeam': 'HomeTeam',
        'AwayTeam': 'AwayTeam',
        'FTHG': 'FullTimeHomeGoals',
        'FTAG': 'FullTimeAwayGoals',
        'FTR': 'FullTimeResult',
        'HTHG': 'HalfTimeHomeGoals',
        'HTAG': 'HalfTimeAwayGoals',
        'HTR': 'HalfTimeResult',
        'Referee': 'Referee',
        'HS': 'HomeShots',
        'AS': 'AwayShots',
        'HST': 'HomeShotsOnTarget',
        'AST': 'AwayShotsOnTarget',
        'HF': 'HomeFouls',
        'AF': 'AwayFouls',
        'HC': 'HomeCorners',
        'AC': 'AwayCorners',
        'HY': 'HomeYellowCards',
        'AY': 'AwayYellowCards',
        'HR': 'HomeRedCards',
        'AR': 'AwayRedCards'
    }
    
    print("2️⃣ Identificando partidos faltantes...")
    print()
    
    missing_records = []
    
    for season in ['2003/04', '2004/05']:
        # Partidos existentes en epl_final
        df_final_season = df_final[df_final['Season'] == season]
        
        # Partidos completos en epl_odds
        df_odds_season = df_odds[df_odds['Season'] == season]
        
        print(f"   {season}:")
        print(f"     En epl_final.csv: {len(df_final_season)} partidos")
        print(f"     En epl_odds.csv:  {len(df_odds_season)} partidos")
        
        # Crear identificador único para cada partido
        df_final_season['match_id'] = (
            df_final_season['Season'] + '_' + 
            df_final_season['HomeTeam'] + '_' + 
            df_final_season['AwayTeam']
        )
        
        df_odds_season['match_id'] = (
            df_odds_season['Season'] + '_' + 
            df_odds_season['HomeTeam'] + '_' + 
            df_odds_season['AwayTeam']
        )
        
        # Identificar partidos faltantes
        existing_ids = set(df_final_season['match_id'])
        all_ids = set(df_odds_season['match_id'])
        missing_ids = all_ids - existing_ids
        
        print(f"     Faltantes:        {len(missing_ids)} partidos")
        
        # Extraer partidos faltantes de epl_odds
        df_missing = df_odds_season[df_odds_season['match_id'].isin(missing_ids)]
        
        # Seleccionar y renombrar columnas
        available_cols = {k: v for k, v in column_mapping.items() if k in df_missing.columns}
        df_missing_mapped = df_missing[list(available_cols.keys())].copy()
        df_missing_mapped = df_missing_mapped.rename(columns=available_cols)
        
        missing_records.append(df_missing_mapped)
        print()
    
    # Combinar partidos faltantes
    if missing_records:
        df_new_records = pd.concat(missing_records, ignore_index=True)
        
        print("3️⃣ Preparando datos para integración...")
        print(f"   Total partidos a agregar: {len(df_new_records)}")
        print()
        
        # Asegurar que las columnas coincidan
        # Añadir columnas faltantes con NaN
        for col in df_final.columns:
            if col not in df_new_records.columns:
                df_new_records[col] = None
        
        # Ordenar columnas igual que df_final
        df_new_records = df_new_records[df_final.columns]
        
        # Convertir tipos de datos
        for col in df_new_records.columns:
            if col in df_final.columns:
                try:
                    df_new_records[col] = df_new_records[col].astype(df_final[col].dtype)
                except:
                    pass
        
        print("4️⃣ Integrando partidos...")
        
        # Combinar
        df_complete = pd.concat([df_final, df_new_records], ignore_index=True)
        
        # Ordenar por Season y MatchDate
        # Convertir fechas con formato flexible (DD/MM/YY o YYYY-MM-DD)
        df_complete['MatchDate'] = pd.to_datetime(
            df_complete['MatchDate'], 
            format='mixed', 
            dayfirst=True,
            errors='coerce'
        )
        df_complete = df_complete.sort_values(['Season', 'MatchDate']).reset_index(drop=True)
        
        print(f"   ✅ {len(df_new_records)} partidos agregados")
        print()
        
        # Guardar
        print("5️⃣ Guardando archivo actualizado...")
        df_complete.to_csv('data/raw/epl_final.csv', index=False)
        print(f"   💾 Guardado: data/raw/epl_final.csv")
        print(f"   Total partidos: {len(df_complete)}")
        print()
        
        # Verificación final
        print("6️⃣ Verificación final:")
        print("-" * 80)
        
        season_counts = df_complete['Season'].value_counts().sort_index()
        
        for season in ['2003/04', '2004/05']:
            count = season_counts.get(season, 0)
            status = '✅' if count == 380 else f'⚠️ ({count})'
            print(f"   {season}: {count} partidos {status}")
        
        print()
        print("="*80)
        print("✅ COMPLETADO CON ÉXITO")
        print("="*80)
        print()
        print(f"Total partidos en epl_final.csv: {len(df_complete):,}")
        print(f"Temporadas con 380 partidos: {(season_counts == 380).sum()}")
        
    else:
        print("❌ No se encontraron partidos faltantes")


if __name__ == '__main__':
    complete_epl_final()
