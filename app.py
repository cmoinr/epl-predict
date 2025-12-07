"""
⚽ EPL Predictor - Dashboard Streamlit
Predictor de resultados Premier League con análisis de Value Betting
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
from datetime import datetime, timedelta

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from predictor import EPLPredictor
from odds_comparison import OddsComparison


# ═════════════════════════════════════════════════════════════════
# CONFIG STREAMLIT
# ═════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="⚽ EPL Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
    <style>
    .metric-card {
        background-color: #161b22;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #30363d;
    }
    .header-title {
        text-align: center;
        margin-bottom: 30px;
    }
    </style>
""", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════
# FUNCIONES CON CACHING
# ═════════════════════════════════════════════════════════════════

@st.cache_resource
def load_predictor():
    """Cargar modelos una sola vez"""
    return EPLPredictor('models')


@st.cache_resource
def load_data():
    """Cargar dataset histórico una sola vez"""
    try:
        df = pd.read_csv('data/raw/epl_final.csv')
        return df
    except FileNotFoundError:
        st.error("❌ Dataset no encontrado en data/raw/epl_final.csv")
        return None


@st.cache_resource
def load_odds_data():
    """Cargar datos de odds de ejemplo"""
    try:
        df_odds = pd.read_csv('data/processed/sample_odds.csv')
        return df_odds
    except FileNotFoundError:
        return None


# ═════════════════════════════════════════════════════════════════
# FUNCIONES AUXILIARES
# ═════════════════════════════════════════════════════════════════

def get_teams_list(df):
    """Obtener lista de equipos del dataset"""
    if df is None:
        return []
    home_teams = df['HomeTeam'].unique()
    away_teams = df['AwayTeam'].unique()
    teams = sorted(set(home_teams) | set(away_teams))
    return teams


def create_probability_gauge(label, value, color='green'):
    """Crear gráfico gauge para probabilidad"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value * 100,
        title={'text': label},
        delta={'reference': 50},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 33], 'color': "#fee5d9"},
                {'range': [33, 66], 'color': "#fcae91"},
                {'range': [66, 100], 'color': "#fb6a4a"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10))
    return fig


def create_odds_comparison_table(prediction, market_odds=None):
    """Crear tabla de comparación modelo vs mercado"""
    data = []
    
    if prediction and 'resultado' in prediction:
        probs_rf = prediction['resultado'].get('random_forest', {}).get('probabilidades', {})
        probs_gb = prediction['resultado'].get('gradient_boosting', {}).get('probabilidades', {})
        
        outcomes = ['Home Win', 'Draw', 'Away Win']
        
        for outcome in outcomes:
            prob_rf = probs_rf.get(outcome, 0) / 100
            prob_gb = probs_gb.get(outcome, 0) / 100
            
            # Probabilidad promedio del modelo
            prob_model = (prob_rf + prob_gb) / 2
            
            # Odds de mercado (ejemplo)
            if market_odds and outcome in market_odds:
                odds = market_odds[outcome]
                implied_prob = 1 / odds
                edge = prob_model - implied_prob
            else:
                odds = "-"
                implied_prob = "-"
                edge = "-"
            
            data.append({
                'Outcome': outcome,
                'Model Prob (RF)': f"{prob_rf:.1%}",
                'Model Prob (GB)': f"{prob_gb:.1%}",
                'Model Avg': f"{prob_model:.1%}",
                'Market Odds': odds,
                'Market Implied': implied_prob if isinstance(implied_prob, str) else f"{implied_prob:.1%}",
                'Edge': edge if isinstance(edge, str) else f"{edge:+.1%}"
            })
    
    return pd.DataFrame(data)


# ═════════════════════════════════════════════════════════════════
# HEADER
# ═════════════════════════════════════════════════════════════════

st.markdown("""
    <div style='text-align: center; margin-bottom: 30px;'>
        <h1>⚽ EPL Predictor</h1>
        <p style='font-size: 18px; color: #8b949e;'>
            Predictor inteligente de resultados Premier League con análisis de Value Betting
        </p>
    </div>
""", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════
# SIDEBAR - INPUTS
# ═════════════════════════════════════════════════════════════════

st.sidebar.markdown("## ⚙️ Configuración de Predicción")

# Cargar datos
df_historical = load_data()
teams = get_teams_list(df_historical)

if not teams:
    st.error("❌ No se pueden cargar los equipos del dataset")
    st.stop()

# Inputs
home_team = st.sidebar.selectbox(
    "🏠 Equipo Local",
    teams,
    index=0
)

away_team = st.sidebar.selectbox(
    "✈️ Equipo Visitante",
    teams,
    index=1
)

# Validar que no sean el mismo equipo
if home_team == away_team:
    st.sidebar.warning("⚠️ Selecciona equipos diferentes")
    st.stop()

# Selector de fecha
match_date = st.sidebar.date_input(
    "📅 Fecha del partido",
    value=datetime.now(),
    min_value=datetime(2000, 1, 1),
    max_value=datetime.now() + timedelta(days=365)
)

match_date_str = match_date.strftime('%Y-%m-%d')

# Separador
st.sidebar.markdown("---")
st.sidebar.markdown("## 💰 Cuotas del Mercado (Manual)")

# Inputs de odds
with st.sidebar.expander("📊 Ingresar Odds", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        home_win_odds = st.number_input(
            "🏠 Victoria Local",
            min_value=1.01,
            value=2.50,
            step=0.01,
            help="Cuota para victoria del equipo local"
        )
        draw_odds = st.number_input(
            "🤝 Empate",
            min_value=1.01,
            value=3.50,
            step=0.01,
            help="Cuota para empate"
        )
        away_win_odds = st.number_input(
            "✈️ Victoria Visitante",
            min_value=1.01,
            value=2.80,
            step=0.01,
            help="Cuota para victoria del equipo visitante"
        )
    
    with col2:
        over_2_5_odds = st.number_input(
            "⚽ Over 2.5 Goles",
            min_value=1.01,
            value=1.85,
            step=0.01,
            help="Cuota para más de 2.5 goles"
        )
        under_2_5_odds = st.number_input(
            "⚽ Under 2.5 Goles",
            min_value=1.01,
            value=1.95,
            step=0.01,
            help="Cuota para 2.5 goles o menos"
        )
        both_score_yes = st.number_input(
            "🎯 Ambos Marcan (Sí)",
            min_value=1.01,
            value=1.75,
            step=0.01,
            help="Cuota para ambos equipos marcan"
        )
        both_score_no = st.number_input(
            "🎯 Ambos Marcan (No)",
            min_value=1.01,
            value=1.90,
            step=0.01,
            help="Cuota para que no ambos equipos marquen"
        )

# Botón de predicción
predict_button = st.sidebar.button(
    "🔮 PREDECIR PARTIDO",
    type="primary",
    use_container_width=True
)

# ═════════════════════════════════════════════════════════════════
# INFORMACIÓN ADICIONAL EN SIDEBAR
# ═════════════════════════════════════════════════════════════════

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Sobre este Dashboard")
st.sidebar.info(
    """
    **Funcionalidades:**
    - 🤖 Predicciones con Random Forest + Gradient Boosting
    - 📊 Análisis de probabilidades (1X2)
    - 💰 Análisis de Value Betting
    - 📈 Comparación modelo vs mercado
    
    **Modelos entrenados con:**
    - ~9,400 partidos históricos EPL
    - 25+ features derivadas
    - Validación cruzada
    """
)

# ═════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ═════════════════════════════════════════════════════════════════

if predict_button:
    try:
        with st.spinner("🔄 Cargando modelos..."):
            predictor = load_predictor()
        
        with st.spinner(f"🔮 Prediciendo {home_team} vs {away_team}..."):
            result = predictor.predict_match(
                df_historical=df_historical,
                home_team=home_team,
                away_team=away_team,
                match_date=match_date_str
            )
        
        # ═══════════════════════════════════════════════════
        # SECCION 1: RESUMEN DEL PARTIDO
        # ═══════════════════════════════════════════════════
        
        st.markdown(f"""
            <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
                <h2 style='text-align: center; margin: 0;'>{home_team} vs {away_team}</h2>
                <p style='text-align: center; margin: 5px 0; font-size: 14px;'>{match_date_str}</p>
            </div>
        """, unsafe_allow_html=True)
        
        # ═══════════════════════════════════════════════════
        # SECCION 2: PROBABILIDADES (GAUGES)
        # ═══════════════════════════════════════════════════
        
        st.markdown("### 📊 Probabilidades Predichas")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if result and 'resultado' in result:
                probs = result['resultado'].get('random_forest', {}).get('probabilidades', {})
                home_win_prob = probs.get('Home Win', 0) / 100
                st.plotly_chart(create_probability_gauge('Home Win', home_win_prob, '#1f77b4'), 
                              use_container_width=True)
        
        with col2:
            if result and 'resultado' in result:
                probs = result['resultado'].get('random_forest', {}).get('probabilidades', {})
                draw_prob = probs.get('Draw', 0) / 100
                st.plotly_chart(create_probability_gauge('Draw', draw_prob, '#ff7f0e'), 
                              use_container_width=True)
        
        with col3:
            if result and 'resultado' in result:
                probs = result['resultado'].get('random_forest', {}).get('probabilidades', {})
                away_win_prob = probs.get('Away Win', 0) / 100
                st.plotly_chart(create_probability_gauge('Away Win', away_win_prob, '#2ca02c'), 
                              use_container_width=True)
        
        # ═══════════════════════════════════════════════════
        # SECCION 3: DETALLES TÉCNICOS
        # ═══════════════════════════════════════════════════
        
        st.markdown("### 🔬 Detalles Técnicos")
        
        tab1, tab2, tab3 = st.tabs(["Random Forest", "Gradient Boosting", "Goles & BTTS"])
        
        with tab1:
            if result and 'resultado' in result:
                rf_result = result['resultado'].get('random_forest', {})
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "Predicción",
                        rf_result.get('prediccion', 'N/A')
                    )
                
                with col2:
                    st.metric(
                        "Confianza",
                        f"{rf_result.get('confianza', 0):.1f}%"
                    )
                
                # Probabilidades
                probs = rf_result.get('probabilidades', {})
                st.bar_chart(pd.DataFrame({
                    'Outcome': list(probs.keys()),
                    'Probability %': list(probs.values())
                }).set_index('Outcome'))
        
        with tab2:
            if result and 'resultado' in result:
                gb_result = result['resultado'].get('gradient_boosting', {})
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric(
                        "Predicción",
                        gb_result.get('prediccion', 'N/A')
                    )
                
                with col2:
                    st.metric(
                        "Confianza",
                        f"{gb_result.get('confianza', 0):.1f}%"
                    )
                
                # Probabilidades
                probs = gb_result.get('probabilidades', {})
                st.bar_chart(pd.DataFrame({
                    'Outcome': list(probs.keys()),
                    'Probability %': list(probs.values())
                }).set_index('Outcome'))
        
        with tab3:
            if result and 'goles_totales' in result:
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Goles Totales (Pred)",
                        f"{result['goles_totales'].get('promedio', 0):.2f}"
                    )
                
                with col2:
                    st.metric(
                        "Over/Under 2.5",
                        f"{result['goles_totales'].get('over_2_5', 0):.1%}"
                    )
                
                with col3:
                    st.metric(
                        "BTTS Prob",
                        f"{result.get('btts', {}).get('probabilidad', 0):.1%}"
                    )
        
        # ═══════════════════════════════════════════════════
        # SECCION 4: COMPARACIÓN MODELO VS MERCADO
        # ═══════════════════════════════════════════════════
        
        st.markdown("### 💰 Análisis de Value Betting")
        
        # Crear datos para análisis
        market_odds = {
            'Home Win': home_win_odds,
            'Draw': draw_odds,
            'Away Win': away_win_odds
        }
        
        # Obtener probabilidades del modelo (promedio de RF y GB)
        if result and 'resultado' in result:
            probs_rf = result['resultado'].get('random_forest', {}).get('probabilidades', {})
            probs_gb = result['resultado'].get('gradient_boosting', {}).get('probabilidades', {})
            
            # Crear tabla de análisis
            analysis_data = []
            
            outcomes = ['Home Win', 'Draw', 'Away Win']
            colors = ['🟢', '🟡', '🔴']  # Green, yellow, red
            
            for outcome, color in zip(outcomes, colors):
                prob_rf = probs_rf.get(outcome, 0) / 100
                prob_gb = probs_gb.get(outcome, 0) / 100
                prob_avg = (prob_rf + prob_gb) / 2
                odds = market_odds.get(outcome, 0)
                
                if odds > 0:
                    implied_prob = 1 / odds
                    edge = prob_avg - implied_prob
                    expected_value = (prob_avg * (odds - 1)) - (1 - prob_avg)
                    
                    # Recomendación basada en edge y EV
                    if edge > 0.05 and expected_value > 0.10:
                        recommendation = "✅ STRONG BET"
                        rec_color = "🟢"
                    elif edge > 0.03 and expected_value > 0.05:
                        recommendation = "👍 BET"
                        rec_color = "🟢"
                    elif edge > 0 and expected_value > 0:
                        recommendation = "⚠️ MAYBE"
                        rec_color = "🟡"
                    else:
                        recommendation = "❌ PASS"
                        rec_color = "🔴"
                    
                    analysis_data.append({
                        'Resultado': outcome,
                        'Modelo RF': f"{prob_rf:.1%}",
                        'Modelo GB': f"{prob_gb:.1%}",
                        'Promedio': f"{prob_avg:.1%}",
                        'Cuota Mercado': f"{odds:.2f}",
                        'Prob. Implícita': f"{implied_prob:.1%}",
                        'Edge': f"{edge:+.2%}",
                        'EV': f"{expected_value:+.2%}",
                        'Recomendación': recommendation
                    })
            
            df_analysis = pd.DataFrame(analysis_data)
            st.dataframe(df_analysis, use_container_width=True, hide_index=True)
            
            # Explicación de términos
            with st.expander("📚 ¿Qué significan estos términos?"):
                st.markdown("""
                **Modelo RF/GB:** Probabilidad predicha por cada modelo (0-100%)
                
                **Promedio:** Probabilidad promedio de ambos modelos
                
                **Cuota Mercado:** Cuota ingresada manualmente (odds del mercado)
                
                **Prob. Implícita:** Probabilidad que el mercado está asignando (1/cuota)
                
                **Edge:** Ventaja del modelo sobre el mercado
                - Positivo = modelo tiene ventaja
                - Negativo = mercado tiene ventaja
                
                **EV (Expected Value):** Ganancia esperada a largo plazo
                - EV = (Prob. Modelo × (Cuota - 1)) - (1 - Prob. Modelo)
                - EV > 10% = Excelente oportunidad
                - EV > 5% = Buena oportunidad
                - EV > 0% = Valor positivo
                
                **Recomendación:**
                - ✅ STRONG BET: Edge >5% y EV >10%
                - 👍 BET: Edge >3% y EV >5%
                - ⚠️ MAYBE: Edge >0% y EV >0%
                - ❌ PASS: No hay ventaja
                """)
        
        # ═══════════════════════════════════════════════════
        # SECCION 5: ANÁLISIS DE GOLES Y BTTS
        # ═══════════════════════════════════════════════════
        
        st.markdown("### ⚽ Análisis Goles y BTTS")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Over/Under 2.5 Goles")
            
            # Obtener probabilidad Over con fallback
            over_prob = 0.5
            if result and 'goles_totales' in result:
                goles_pred = result['goles_totales'].get('promedio', 2.5)
                over_prob = result['goles_totales'].get('over_2_5', 0.5)
            else:
                goles_pred = 2.5
            
            # Análisis OU - SIEMPRE mostrar si hay cuotas
            if over_2_5_odds > 0 and under_2_5_odds > 0:
                ou_analysis = pd.DataFrame({
                    'Mercado': ['Over 2.5', 'Under 2.5'],
                    'Cuota': [over_2_5_odds, under_2_5_odds],
                    'Prob. Implícita': [f"{1/over_2_5_odds:.1%}", f"{1/under_2_5_odds:.1%}"],
                    'Prob. Modelo': [f"{over_prob:.1%}", f"{1-over_prob:.1%}"],
                    'Edge': [f"{over_prob - (1/over_2_5_odds):+.2%}", 
                            f"{(1-over_prob) - (1/under_2_5_odds):+.2%}"]
                })
                st.dataframe(ou_analysis, use_container_width=True, hide_index=True)
            else:
                st.warning("⚠️ Ingresa las cuotas Over/Under en el sidebar para análisis")
            
            st.metric("Goles Predichos", f"{goles_pred:.2f}")
        
        with col2:
            st.subheader("Both Teams to Score (BTTS)")
            
            # Obtener probabilidad BTTS con fallback
            btts_prob = 0.5
            if result and 'btts' in result:
                btts_prob = result['btts'].get('probabilidad', 0.5)
            
            # Análisis BTTS - SIEMPRE mostrar si hay cuotas
            if both_score_yes > 0 and both_score_no > 0:
                btts_analysis = pd.DataFrame({
                    'Mercado': ['Sí', 'No'],
                    'Cuota': [both_score_yes, both_score_no],
                    'Prob. Implícita': [f"{1/both_score_yes:.1%}", f"{1/both_score_no:.1%}"],
                    'Prob. Modelo': [f"{btts_prob:.1%}", f"{1-btts_prob:.1%}"],
                    'Edge': [f"{btts_prob - (1/both_score_yes):+.2%}", 
                            f"{(1-btts_prob) - (1/both_score_no):+.2%}"]
                })
                st.dataframe(btts_analysis, use_container_width=True, hide_index=True)
                st.metric("Probabilidad BTTS", f"{btts_prob:.1%}")
            else:
                st.warning("⚠️ Ingresa las cuotas BTTS (Sí/No) en el sidebar para análisis")
                st.metric("Probabilidad BTTS", f"{btts_prob:.1%}")
        
        # ═══════════════════════════════════════════════════
        # SECCION 6: RESUMEN DE OPORTUNIDADES
        # ═══════════════════════════════════════════════════
        
        st.markdown("### 🎯 Resumen de Oportunidades")
        
        # Contar oportunidades
        strong_bets = df_analysis[df_analysis['Recomendación'] == '✅ STRONG BET'].shape[0]
        good_bets = df_analysis[df_analysis['Recomendación'] == '👍 BET'].shape[0]
        maybe_bets = df_analysis[df_analysis['Recomendación'] == '⚠️ MAYBE'].shape[0]
        pass_bets = df_analysis[df_analysis['Recomendación'] == '❌ PASS'].shape[0]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("🟢 Strong Bet", strong_bets)
        with col2:
            st.metric("👍 Good Bet", good_bets)
        with col3:
            st.metric("🟡 Maybe", maybe_bets)
        with col4:
            st.metric("❌ Pass", pass_bets)
        
        # ═══════════════════════════════════════════════════
        # SECCION 7: DATOS RAW (EXPANDER)
        # ═══════════════════════════════════════════════════
        
        with st.expander("📋 Datos Completos de Predicción (JSON)"):
            st.json(result)
        
        # Success message
        st.success("✅ Predicción completada exitosamente")
        
    except Exception as e:
        st.error(f"❌ Error durante la predicción: {str(e)}")
        st.info("Asegúrate de que:")
        st.write("- El dataset existe en `data/raw/epl_final.csv`")
        st.write("- Los modelos están en `models/`")

else:
    # Pantalla inicial
    st.markdown("""
        <div style='background: #161b22; border: 1px solid #30363d; padding: 30px; border-radius: 10px; text-align: center;'>
            <h3>👋 Bienvenido al EPL Predictor</h3>
            <p style='color: #c9d1d9;'>Selecciona dos equipos, una fecha y las cuotas del mercado en la barra lateral para comenzar.</p>
            <p style='color: #c9d1d9;'><strong>El dashboard mostrará:</strong></p>
            <ul style='text-align: left; display: inline-block; color: #c9d1d9;'>
                <li>📊 Probabilidades de cada resultado (1X2)</li>
                <li>🤖 Predicciones de 2 modelos independientes</li>
                <li>⚽ Goles totales y BTTS</li>
                <li>💰 Análisis completo de VALUE BETTING</li>
                <li>✅ Recomendaciones de apuesta (Edge + EV)</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📋 Cómo Usar el Dashboard")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Paso 1: Selecciona Equipos**
        - 🏠 Equipo local
        - ✈️ Equipo visitante
        - 📅 Fecha del partido
        
        **Paso 2: Ingresa Cuotas**
        - Expande "💰 Ingresar Odds" en sidebar
        - Ingresa las cuotas del mercado:
          - 🏠 Victoria Local
          - 🤝 Empate
          - ✈️ Victoria Visitante
          - ⚽ Over/Under 2.5
          - 🎯 BTTS (Sí/No)
        """)
    
    with col2:
        st.markdown("""
        **Paso 3: Predecir**
        - Click en "🔮 PREDECIR PARTIDO"
        - Espera carga de modelos (~3s)
        
        **Paso 4: Analiza**
        - Ver probabilidades del modelo
        - Analizar Value Betting
        - Revisar recomendaciones
        - Explorar goles y BTTS
        """)
    
    # Información de cuotas
    st.markdown("### 💰 Dónde Obtener Cuotas")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Casas de Apuestas Principales:**
        - Betfair
        - Bet365
        - William Hill
        - Pinnacle
        - Unibet
        """)
    
    with col2:
        st.markdown("""
        **Agregadores de Odds:**
        - OddsPortal
        - BetBrain
        - SofaScore
        - Flashscore
        """)
    
    # Mostrar estadísticas del dataset
    if df_historical is not None:
        st.markdown("### 📊 Dataset Disponible")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 Partidos en BD", len(df_historical))
        
        with col2:
            st.metric("👥 Equipos", len(set(df_historical['HomeTeam'].unique()) | set(df_historical['AwayTeam'].unique())))
        
        with col3:
            st.metric("📅 Años", f"{df_historical['MatchDate'].min()[:4]} - {df_historical['MatchDate'].max()[:4]}")
        
        with col4:
            st.metric("⚙️ Features", 25)
    
    # Información de valores por defecto
    st.info(
        """
        **💡 Valores por Defecto:** Los campos de cuotas tienen valores por defecto para demostración.
        Para análisis real, ingresa las cuotas del mercado actual de tu casa de apuestas preferida.
        """
    )


# ═════════════════════════════════════════════════════════════════
# FOOTER
# ═════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #8b949e; font-size: 12px; margin-top: 20px;'>
        <p>⚽ EPL Predictor v1.0 | ML Models: Random Forest + Gradient Boosting</p>
        <p>Desarrollado con ❤️ usando Streamlit</p>
    </div>
""", unsafe_allow_html=True)
