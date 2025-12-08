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

# CSS personalizado + Google Fonts
st.markdown("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    
    <style>
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }
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
        # SECCION 2: PROBABILIDADES (BARRA HORIZONTAL)
        # ═══════════════════════════════════════════════════
        
        st.markdown("### 📊 Probabilidades Predichas (1X2)")
        
        if result and 'resultado' in result:
            # Obtener probabilidades promedio de RF y GB
            probs_rf = result['resultado'].get('random_forest', {}).get('probabilidades', {})
            probs_gb = result['resultado'].get('gradient_boosting', {}).get('probabilidades', {})
            
            home_prob = (probs_rf.get('Home Win', 0) + probs_gb.get('Home Win', 0)) / 2
            draw_prob = (probs_rf.get('Draw', 0) + probs_gb.get('Draw', 0)) / 2
            away_prob = (probs_rf.get('Away Win', 0) + probs_gb.get('Away Win', 0)) / 2
            
            # Crear gráfico de barra horizontal apilada
            fig = go.Figure()
            
            # Victoria Local (azul)
            fig.add_trace(go.Bar(
                y=['Probabilidad'],
                x=[home_prob],
                name=f'🏠 {home_team}',
                orientation='h',
                marker=dict(color='#667eea'),
                text=f'{home_prob:.1f}%',
                textposition='inside',
                textfont=dict(size=16, color='white', family='Inter'),
                hovertemplate=f'<b>{home_team}</b><br>Probabilidad: %{{x:.1f}}%<extra></extra>'
            ))
            
            # Empate (gris)
            fig.add_trace(go.Bar(
                y=['Probabilidad'],
                x=[draw_prob],
                name='🤝 Empate',
                orientation='h',
                marker=dict(color='#8b949e'),
                text=f'{draw_prob:.1f}%',
                textposition='inside',
                textfont=dict(size=16, color='white', family='Inter'),
                hovertemplate='<b>Empate</b><br>Probabilidad: %{x:.1f}%<extra></extra>'
            ))
            
            # Victoria Visitante (morado)
            fig.add_trace(go.Bar(
                y=['Probabilidad'],
                x=[away_prob],
                name=f'✈️ {away_team}',
                orientation='h',
                marker=dict(color='#764ba2'),
                text=f'{away_prob:.1f}%',
                textposition='inside',
                textfont=dict(size=16, color='white', family='Inter'),
                hovertemplate=f'<b>{away_team}</b><br>Probabilidad: %{{x:.1f}}%<extra></extra>'
            ))
            
            fig.update_layout(
                barmode='stack',
                height=150,
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(
                    showticklabels=False,
                    showgrid=False,
                    range=[0, 100]
                ),
                yaxis=dict(
                    showticklabels=False,
                    showgrid=False
                ),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                showlegend=True,
                legend=dict(
                    orientation='h',
                    yanchor='bottom',
                    y=1.02,
                    xanchor='center',
                    x=0.5,
                    font=dict(size=14, family='Inter')
                ),
                hoverlabel=dict(
                    bgcolor='#161b22',
                    font_size=14,
                    font_family='Inter'
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Gráficos adicionales: Over/Under 2.5 y BTTS
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### ⚽ Over/Under 2.5 Goles")
            
            # Calcular probabilidad Over/Under
            goles_pred = 2.5
            if result and 'goles_totales' in result:
                goles_pred = result['goles_totales'].get('promedio', 2.5)
            
            # Calcular probabilidades
            if goles_pred > 2.5:
                over_prob = 0.5 + min((goles_pred - 2.5) * 0.15, 0.45)
            else:
                over_prob = 0.5 - min((2.5 - goles_pred) * 0.15, 0.45)
            
            over_prob_pct = over_prob * 100
            under_prob_pct = (1 - over_prob) * 100
            
            # Crear gráfico de dona
            fig_ou = go.Figure(data=[go.Pie(
                labels=['Over 2.5', 'Under 2.5'],
                values=[over_prob_pct, under_prob_pct],
                hole=0.6,
                marker=dict(colors=['#667eea', '#764ba2']),
                textinfo='label+percent',
                textfont=dict(size=13, family='Inter', color='white'),
                hovertemplate='<b>%{label}</b><br>Probabilidad: %{value:.1f}%<extra></extra>'
            )])
            
            fig_ou.update_layout(
                height=250,
                margin=dict(l=20, r=20, t=20, b=20),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                showlegend=False,
                annotations=[dict(
                    text=f'{goles_pred:.2f}<br>goles',
                    x=0.5, y=0.5,
                    font=dict(size=20, family='Inter', color='#c9d1d9'),
                    showarrow=False
                )],
                hoverlabel=dict(
                    bgcolor='#161b22',
                    font_size=14,
                    font_family='Inter'
                )
            )
            
            st.plotly_chart(fig_ou, use_container_width=True)
        
        with col2:
            st.markdown("#### 🎯 Both Teams to Score (BTTS)")
            
            # Obtener probabilidad BTTS
            btts_prob = 0.5
            if result and 'ambos_anotan' in result:
                btts_promedio = result['ambos_anotan'].get('promedio', {})
                btts_prob = btts_promedio.get('si', 50) / 100
            
            btts_si_pct = btts_prob * 100
            btts_no_pct = (1 - btts_prob) * 100
            
            # Crear gráfico de dona
            fig_btts = go.Figure(data=[go.Pie(
                labels=['Sí', 'No'],
                values=[btts_si_pct, btts_no_pct],
                hole=0.6,
                marker=dict(colors=['#667eea', '#8b949e']),
                textinfo='label+percent',
                textfont=dict(size=13, family='Inter', color='white'),
                hovertemplate='<b>%{label}</b><br>Probabilidad: %{value:.1f}%<extra></extra>'
            )])
            
            fig_btts.update_layout(
                height=250,
                margin=dict(l=20, r=20, t=20, b=20),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                showlegend=False,
                annotations=[dict(
                    text='BTTS',
                    x=0.5, y=0.5,
                    font=dict(size=20, family='Inter', color='#c9d1d9'),
                    showarrow=False
                )],
                hoverlabel=dict(
                    bgcolor='#161b22',
                    font_size=14,
                    font_family='Inter'
                )
            )
            
            st.plotly_chart(fig_btts, use_container_width=True)
        
        # ═══════════════════════════════════════════════════
        # SECCION 3: COMPARACIÓN MODELO VS MERCADO
        # ═══════════════════════════════════════════════════
        
        st.markdown("### 💰 Análisis de Value Betting")
        
        # TAB 1: Análisis 1X2
        tab1x2, tabou, tabbtts = st.tabs(["1X2", "Over/Under 2.5", "BTTS"])
        
        # ─── TAB 1X2 ───
        with tab1x2:
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
                
                for outcome in outcomes:
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
                        elif edge > 0.03 and expected_value > 0.05:
                            recommendation = "👍 BET"
                        elif edge > 0 and expected_value > 0:
                            recommendation = "⚠️ MAYBE"
                        else:
                            recommendation = "❌ PASS"
                        
                        analysis_data.append({
                            'Resultado': outcome,
                            'Modelo RF': f"{prob_rf:.1%}",
                            'Modelo GB': f"{prob_gb:.1%}",
                            'Promedio': f"{prob_avg:.1%}",
                            'Cuota': f"{odds:.2f}",
                            'Prob. Implícita': f"{implied_prob:.1%}",
                            'Edge': f"{edge:+.2%}",
                            'EV': f"{expected_value:+.2%}",
                            'Recomendación': recommendation
                        })
                
                df_analysis = pd.DataFrame(analysis_data)
                st.dataframe(df_analysis, use_container_width=True, hide_index=True)
        
        # ─── TAB OVER/UNDER ───
        with tabou:
            if result and 'goles_totales' in result:
                goles_pred = result['goles_totales'].get('promedio', 2.5)
                
                # Calcular probabilidades
                if goles_pred > 2.5:
                    over_prob = 0.5 + min((goles_pred - 2.5) * 0.15, 0.45)
                else:
                    over_prob = 0.5 - min((2.5 - goles_pred) * 0.15, 0.45)
                
                ou_data = []
                
                for market, odds in [('Over 2.5', over_2_5_odds), ('Under 2.5', under_2_5_odds)]:
                    prob_model = over_prob if 'Over' in market else (1 - over_prob)
                    
                    if odds > 0:
                        implied_prob = 1 / odds
                        edge = prob_model - implied_prob
                        expected_value = (prob_model * (odds - 1)) - (1 - prob_model)
                        
                        if edge > 0.05 and expected_value > 0.10:
                            recommendation = "✅ STRONG BET"
                        elif edge > 0.03 and expected_value > 0.05:
                            recommendation = "👍 BET"
                        elif edge > 0 and expected_value > 0:
                            recommendation = "⚠️ MAYBE"
                        else:
                            recommendation = "❌ PASS"
                        
                        ou_data.append({
                            'Mercado': market,
                            'Goles Pred': f"{goles_pred:.2f}",
                            'Prob. Modelo': f"{prob_model:.1%}",
                            'Cuota': f"{odds:.2f}",
                            'Prob. Implícita': f"{implied_prob:.1%}",
                            'Edge': f"{edge:+.2%}",
                            'EV': f"{expected_value:+.2%}",
                            'Recomendación': recommendation
                        })
                
                df_ou = pd.DataFrame(ou_data)
                st.dataframe(df_ou, use_container_width=True, hide_index=True)
        
        # ─── TAB BTTS ───
        with tabbtts:
            if result and 'ambos_anotan' in result:
                btts_promedio = result['ambos_anotan'].get('promedio', {})
                btts_prob = btts_promedio.get('si', 50) / 100
                
                btts_data = []
                
                for market, odds in [('BTTS Sí', both_score_yes), ('BTTS No', both_score_no)]:
                    prob_model = btts_prob if 'Sí' in market else (1 - btts_prob)
                    
                    if odds > 0:
                        implied_prob = 1 / odds
                        edge = prob_model - implied_prob
                        expected_value = (prob_model * (odds - 1)) - (1 - prob_model)
                        
                        if edge > 0.05 and expected_value > 0.10:
                            recommendation = "✅ STRONG BET"
                        elif edge > 0.03 and expected_value > 0.05:
                            recommendation = "👍 BET"
                        elif edge > 0 and expected_value > 0:
                            recommendation = "⚠️ MAYBE"
                        else:
                            recommendation = "❌ PASS"
                        
                        btts_data.append({
                            'Mercado': market,
                            'Prob. Modelo': f"{prob_model:.1%}",
                            'Cuota': f"{odds:.2f}",
                            'Prob. Implícita': f"{implied_prob:.1%}",
                            'Edge': f"{edge:+.2%}",
                            'EV': f"{expected_value:+.2%}",
                            'Recomendación': recommendation
                        })
                
                df_btts = pd.DataFrame(btts_data)
                st.dataframe(df_btts, use_container_width=True, hide_index=True)
        
        # Explicación de términos
        with st.expander("📚 ¿Qué significan estos términos?"):
            st.markdown("""
            **Modelo RF/GB:** Probabilidad predicha por cada modelo (0-100%)
            
            **Promedio:** Probabilidad promedio de ambos modelos
            
            **Cuota:** Cuota ingresada manualmente (odds del mercado)
            
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
        # GUARDAR PREDICCIÓN EN CSV
        # ═══════════════════════════════════════════════════
        
        try:
            # Crear registro de la predicción
            prediction_record = {
                'date': match_date_str,
                'home_team': home_team,
                'away_team': away_team,
                'home_win_odds': home_win_odds,
                'draw_odds': draw_odds,
                'away_win_odds': away_win_odds,
                'over_2_5_odds': over_2_5_odds,
                'under_2_5_odds': under_2_5_odds,
                'both_score_yes': both_score_yes,
                'both_score_no': both_score_no
            }
            
            # Leer CSV existente
            csv_path = Path('data/processed/odds_history.csv')
            if csv_path.exists():
                df_history = pd.read_csv(csv_path)
            else:
                df_history = pd.DataFrame()
            
            # Agregar nuevo registro
            df_new_record = pd.DataFrame([prediction_record])
            df_history = pd.concat([df_history, df_new_record], ignore_index=True)
            
            # Guardar CSV
            df_history.to_csv(csv_path, index=False)
            
        except Exception as e:
            st.warning(f"⚠️ No se pudo guardar en histórico: {str(e)}")
        
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
