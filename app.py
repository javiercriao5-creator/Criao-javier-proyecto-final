import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
from sklearn.preprocessing import StandardScaler

# ══════════════════════════════════════════════════════════
# 1. CONFIGURACIÓN DE PÁGINA
# ══════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Oráculo Musical",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════
# 2. CSS CYBERPUNK
# ══════════════════════════════════════════════════════════
CYBERPUNK_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600;700&family=Share+Tech+Mono&display=swap');

.stApp {
    background-color: #07070f;
    background-image:
        radial-gradient(ellipse at 15% 40%, rgba(0,245,255,0.06) 0%, transparent 55%),
        radial-gradient(ellipse at 85% 15%, rgba(180,0,255,0.07) 0%, transparent 55%),
        radial-gradient(ellipse at 50% 90%, rgba(255,0,128,0.04) 0%, transparent 50%);
    font-family: 'Rajdhani', sans-serif !important;
}

h1, h2, h3, h4 {
    font-family: 'Orbitron', monospace !important;
    letter-spacing: 2px;
}
h1 { color: #00f5ff !important; text-shadow: 0 0 20px rgba(0,245,255,0.6); font-size: 2rem !important; }
h2 { color: #bf00ff !important; text-shadow: 0 0 14px rgba(191,0,255,0.5); }
h3 { color: #00f5ff !important; }
h4 { color: #ff0080 !important; font-size: 1rem !important; }

p, span, label, div {
    font-family: 'Rajdhani', sans-serif !important;
    color: #c0d8e8 !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d1a 0%, #0a0a15 100%) !important;
    border-right: 1px solid rgba(0,245,255,0.15);
}

input, textarea {
    background: rgba(0,245,255,0.05) !important;
    border: 1px solid rgba(0,245,255,0.3) !important;
    border-radius: 4px !important;
    color: #ffffff !important;
    font-family: 'Share Tech Mono', monospace !important;
}

.stButton > button {
    background: linear-gradient(135deg, #00f5ff, #bf00ff) !important;
    color: #07070f !important;
    font-family: 'Orbitron', monospace !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    letter-spacing: 2px !important;
    border: none !important;
    border-radius: 4px !important;
    padding: 0.7rem 1.5rem !important;
    width: 100% !important;
    box-shadow: 0 0 20px rgba(0,245,255,0.3), 0 0 40px rgba(191,0,255,0.2) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 0 30px rgba(0,245,255,0.5), 0 0 60px rgba(191,0,255,0.3) !important;
}

hr { border-color: rgba(0,245,255,0.15) !important; }

.stCaption {
    color: rgba(0,245,255,0.4) !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.75rem !important;
}

[data-testid="metric-container"] {
    background: rgba(0,245,255,0.04) !important;
    border: 1px solid rgba(0,245,255,0.15) !important;
    border-radius: 6px !important;
    padding: 1rem !important;
}
[data-testid="stMetricValue"] {
    color: #00f5ff !important;
    font-family: 'Orbitron', monospace !important;
}

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #07070f; }
::-webkit-scrollbar-thumb { background: rgba(0,245,255,0.3); border-radius: 3px; }
</style>
"""
st.markdown(CYBERPUNK_CSS, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# 3. HEADER
# ══════════════════════════════════════════════════════════
st.markdown("""
<div style="text-align:center; padding: 1.5rem 0 0.5rem 0;">
    <h1 style="font-size:2.4rem; margin-bottom:0;">🔮 ORÁCULO MUSICAL</h1>
    <p style="color:rgba(0,245,255,0.6); font-family:'Share Tech Mono',monospace;
              font-size:0.85rem; letter-spacing:3px; margin-top:0.3rem;">
        HIT PREDICTION ENGINE &nbsp;·&nbsp; RANDOM FOREST &nbsp;·&nbsp; 79.9% ACCURACY
    </p>
</div>
""", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# 4. CARGA DE MODELO Y DATOS
# ══════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    return joblib.load("oraculo_musical_modelo.pkl")

@st.cache_data
def load_data():
    df = pd.read_csv("dataset_lastfm_ML_listo.csv")
    X = df.drop(columns=["nombre_cancion","nombre_artista","url",
                          "oyentes","reproducciones","es_hit"]).fillna(0)
    y = df["es_hit"]

    cols_num = ["longitud_nombre_cancion","longitud_nombre_artista","ratio_reproducciones_oyentes"]
    scaler = StandardScaler()
    scaler.fit(X[cols_num])

    tags = [c.replace("tag_","").title() for c in X.columns if c.startswith("tag_")]
    return X.columns, scaler, tags, X, y, cols_num

modelo = load_model()
columnas_modelo, scaler, lista_tags, X_base, y_base, cols_num = load_data()

# ══════════════════════════════════════════════════════════
# 5. SESSION STATE — historial
# ══════════════════════════════════════════════════════════
if "historial" not in st.session_state:
    st.session_state.historial = []

# ══════════════════════════════════════════════════════════
# 6. SIDEBAR
# ══════════════════════════════════════════════════════════
st.sidebar.markdown("""
<div style='text-align:center; padding:0.5rem 0 1rem 0;'>
    <p style='font-family:Orbitron,monospace; font-size:0.8rem;
              letter-spacing:2px; color:#00f5ff;'>
        ▸ PARÁMETROS DE ANÁLISIS
    </p>
</div>
""", unsafe_allow_html=True)

nombre_cancion     = st.sidebar.text_input("🎵 Nombre de la canción", "Bohemian Rhapsody").strip()
nombre_artista     = st.sidebar.text_input("🎤 Nombre del artista", "Queen").strip()
oyentes            = st.sidebar.number_input("👤 Oyentes mensuales estimados", min_value=1, value=500_000, step=10_000)
reproducciones     = st.sidebar.number_input("▶️ Reproducciones estimadas",    min_value=1, value=2_500_000, step=100_000)
tags_seleccionados = st.sidebar.multiselect(
    "🎸 Géneros musicales (1–3):",
    lista_tags,
    default=[lista_tags[0]] if lista_tags else []
)

st.sidebar.markdown("<br>", unsafe_allow_html=True)
predecir = st.sidebar.button("🔮 EJECUTAR ANÁLISIS")

if st.sidebar.button("🗑️ Limpiar historial"):
    st.session_state.historial = []
    st.sidebar.success("Historial borrado.")

# ══════════════════════════════════════════════════════════
# 7. VALIDACIÓN
# ══════════════════════════════════════════════════════════
def validar_inputs():
    errores = []
    if not nombre_cancion:
        errores.append("El nombre de la canción no puede estar vacío.")
    if not nombre_artista:
        errores.append("El nombre del artista no puede estar vacío.")
    if not tags_seleccionados:
        errores.append("Selecciona al menos un género musical.")
    if reproducciones < oyentes:
        errores.append("Las reproducciones no deberían ser menores que los oyentes.")
    return errores

# ══════════════════════════════════════════════════════════
# 8. HELPERS DE VISUALIZACIÓN
# ══════════════════════════════════════════════════════════
NEON_CYAN   = "#00f5ff"
NEON_PURPLE = "#bf00ff"
NEON_PINK   = "#ff0080"
BG_CARD     = "#0d0d1a"

def apply_cyberpunk_style(fig, ax):
    fig.patch.set_facecolor(BG_CARD)
    ax.set_facecolor(BG_CARD)
    ax.tick_params(axis='both', colors='#4a7a9a', labelsize=9)
    ax.xaxis.label.set_color('#4a7a9a')
    ax.yaxis.label.set_color('#4a7a9a')
    ax.title.set_color(NEON_CYAN)
    for spine in ax.spines.values():
        spine.set_edgecolor("#1a2a3a")
    return fig, ax

def plot_gauge(probabilidad, prediccion):
    fig, ax = plt.subplots(figsize=(7, 0.7))
    fig.patch.set_facecolor(BG_CARD)
    ax.set_facecolor(BG_CARD)
    color = NEON_CYAN if prediccion == 1 else NEON_PINK
    ax.barh([""], [probabilidad], color=color, height=0.5, alpha=0.9)
    ax.barh([""], [1 - probabilidad], left=[probabilidad], color="#1a1a2e", height=0.5)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0%","25%","50%","75%","100%"],
                        color="#4a7a9a", fontsize=8, fontfamily="monospace")
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#1a2a3a")
    ax.axvline(0.5, color="#ffffff", linewidth=0.8, linestyle="--", alpha=0.3)
    plt.tight_layout(pad=0.1)
    return fig

def plot_feature_importance(modelo, columnas_modelo):
    importancias = modelo.feature_importances_
    feat_df = pd.DataFrame({"feature": columnas_modelo, "imp": importancias})
    feat_df = feat_df.sort_values("imp", ascending=False).head(10)

    labels = (feat_df["feature"]
              .str.replace("tag_", "🎵 ", regex=False)
              .str.replace("longitud_nombre_cancion", "📝 Long. título", regex=False)
              .str.replace("longitud_nombre_artista", "🎤 Long. artista", regex=False)
              .str.replace("ratio_reproducciones_oyentes", "🔄 Ratio repro/oyentes", regex=False))

    colors = [NEON_CYAN if i == 0 else NEON_PURPLE if i == 1 else "#2a4a6a"
              for i in range(len(feat_df))]

    fig, ax = plt.subplots(figsize=(6, 3.8))
    fig, ax = apply_cyberpunk_style(fig, ax)
    ax.barh(labels[::-1], feat_df["imp"][::-1].values,
            color=colors[::-1], edgecolor="none", alpha=0.85)
    ax.set_xlabel("Importancia relativa", fontsize=8)
    ax.set_title("TOP FEATURES GLOBALES", fontsize=9,
                 fontweight="bold", fontfamily="monospace")
    plt.tight_layout()
    return fig

def plot_explicabilidad(entrada_scaled, columnas_modelo, modelo):
    importancias = modelo.feature_importances_
    vals  = entrada_scaled.values[0]
    contrib = importancias * vals

    feat_df = pd.DataFrame({"feature": columnas_modelo, "contrib": contrib})
    feat_df = feat_df.reindex(feat_df["contrib"].abs().sort_values(ascending=False).index)
    feat_df = feat_df.head(8)

    labels = (feat_df["feature"]
              .str.replace("tag_", "🎵 ", regex=False)
              .str.replace("longitud_nombre_cancion", "📝 Long. título", regex=False)
              .str.replace("longitud_nombre_artista", "🎤 Long. artista", regex=False)
              .str.replace("ratio_reproducciones_oyentes", "🔄 Ratio repro/oyentes", regex=False))

    colors = [NEON_CYAN if v > 0 else NEON_PINK for v in feat_df["contrib"].values]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    fig, ax = apply_cyberpunk_style(fig, ax)
    ax.barh(labels[::-1], feat_df["contrib"][::-1].values,
            color=colors[::-1], edgecolor="none", alpha=0.85)
    ax.axvline(0, color="#ffffff", linewidth=0.8, alpha=0.4)
    ax.set_xlabel("← hacia Nicho  |  hacia Hit →", fontsize=8, color="#4a7a9a")
    ax.set_title("¿POR QUÉ ESTA PREDICCIÓN?", fontsize=9,
                 fontweight="bold", fontfamily="monospace")

    hit_patch = mpatches.Patch(color=NEON_CYAN, label="Empuja hacia HIT")
    no_patch  = mpatches.Patch(color=NEON_PINK, label="Empuja hacia NICHO")
    ax.legend(handles=[hit_patch, no_patch], fontsize=7,
              facecolor=BG_CARD, edgecolor="#1a2a3a", labelcolor="#c0d8e8")
    plt.tight_layout()
    return fig

# ══════════════════════════════════════════════════════════
# 9. LÓGICA DE PREDICCIÓN
# ══════════════════════════════════════════════════════════
if predecir:
    errores = validar_inputs()
    if errores:
        for e in errores:
            st.error(f"⚠️ {e}")
    else:
        with st.spinner("Procesando señal..."):
            try:
                len_cancion = len(nombre_cancion)
                len_artista = len(nombre_artista)
                ratio       = reproducciones / (oyentes + 1)

                entrada = pd.DataFrame(columns=columnas_modelo)
                entrada.loc[0] = 0.0
                entrada.loc[0, "longitud_nombre_cancion"]      = len_cancion
                entrada.loc[0, "longitud_nombre_artista"]      = len_artista
                entrada.loc[0, "ratio_reproducciones_oyentes"] = ratio
                for t in tags_seleccionados:
                    col_name = f"tag_{t.lower()}"
                    if col_name in columnas_modelo:
                        entrada.loc[0, col_name] = 1

                entrada_scaled           = entrada.copy()
                entrada_scaled[cols_num] = scaler.transform(entrada[cols_num])

                prediccion   = modelo.predict(entrada_scaled)[0]
                probabilidad = (modelo.predict_proba(entrada_scaled)[0][1]
                                if hasattr(modelo, "predict_proba") else float(prediccion))

                # Guardar en historial
                st.session_state.historial.append({
                    "🎵 Canción":      nombre_cancion,
                    "🎤 Artista":      nombre_artista,
                    "🔮 Resultado":    "✅ HIT" if prediccion == 1 else "📉 NICHO",
                    "📊 Probabilidad": f"{probabilidad*100:.1f}%",
                    "🕐 Hora":         datetime.now().strftime("%H:%M:%S")
                })

                # ── Resultado + métricas ──────────────────────
                st.markdown("<br>", unsafe_allow_html=True)
                col_res, col_met = st.columns([2, 1])

                with col_res:
                    if prediccion == 1:
                        st.markdown(f"""
                        <div style="
                            background: linear-gradient(135deg,
                                rgba(0,245,255,0.08), rgba(191,0,255,0.08));
                            border: 1px solid {NEON_CYAN};
                            border-radius: 8px; padding: 1.5rem;
                            box-shadow: 0 0 30px rgba(0,245,255,0.15),
                                        inset 0 0 30px rgba(0,245,255,0.03);">
                            <p style="font-family:Orbitron,monospace; font-size:1.8rem;
                                      color:{NEON_CYAN}; margin:0;
                                      text-shadow: 0 0 20px {NEON_CYAN};">
                                🌟 HIT DETECTADO
                            </p>
                            <p style="color:#c0d8e8; margin:0.5rem 0 0 0; font-size:1.1rem;">
                                <b style="color:{NEON_CYAN};">"{nombre_cancion}"</b>
                                de <b>{nombre_artista}</b><br>
                                tiene perfil de éxito mundial según el modelo.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        st.balloons()
                    else:
                        st.markdown(f"""
                        <div style="
                            background: linear-gradient(135deg,
                                rgba(255,0,128,0.06), rgba(191,0,255,0.06));
                            border: 1px solid {NEON_PINK};
                            border-radius: 8px; padding: 1.5rem;
                            box-shadow: 0 0 30px rgba(255,0,128,0.12);">
                            <p style="font-family:Orbitron,monospace; font-size:1.8rem;
                                      color:{NEON_PINK}; margin:0;">
                                📉 CANCIÓN DE NICHO
                            </p>
                            <p style="color:#c0d8e8; margin:0.5rem 0 0 0; font-size:1.1rem;">
                                <b style="color:{NEON_PINK};">"{nombre_cancion}"</b>
                                de <b>{nombre_artista}</b><br>
                                tendrá su público, pero no alcanzará el top global.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                with col_met:
                    st.metric("Probabilidad de Hit", f"{probabilidad*100:.1f}%")
                    st.metric("Ratio repro/oyentes", f"{ratio:.1f}x")
                    st.metric("Géneros analizados",  len(tags_seleccionados))

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("##### 📡 SEÑAL DE HIT")
                st.pyplot(plot_gauge(probabilidad, prediccion))

                st.markdown("<hr>", unsafe_allow_html=True)

                col_imp, col_exp = st.columns(2)
                with col_imp:
                    st.markdown("##### 🧠 IMPORTANCIA GLOBAL DE FEATURES")
                    st.caption("Variables que más usa el modelo en general")
                    st.pyplot(plot_feature_importance(modelo, columnas_modelo))

                with col_exp:
                    st.markdown("##### 🔍 EXPLICACIÓN LOCAL")
                    st.caption("Por qué el modelo decidió así para ESTA canción")
                    st.pyplot(plot_explicabilidad(entrada_scaled, columnas_modelo, modelo))

            except Exception as ex:
                st.error(f"❌ Error durante el análisis: {ex}")
                st.info("Verifica que `oraculo_musical_modelo.pkl` y "
                        "`dataset_lastfm_ML_listo.csv` están en la raíz del proyecto.")

# ══════════════════════════════════════════════════════════
# 10. HISTORIAL
# ══════════════════════════════════════════════════════════
if st.session_state.historial:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### 📋 HISTORIAL DE ANÁLISIS")
    hist_df = pd.DataFrame(st.session_state.historial[::-1])
    st.dataframe(hist_df, use_container_width=True, hide_index=True)
else:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center; padding:2rem;
                border:1px dashed rgba(0,245,255,0.15); border-radius:8px;
                color:rgba(0,245,255,0.3); font-family:'Share Tech Mono',monospace;
                font-size:0.85rem; letter-spacing:2px;">
        INTRODUCE LOS DATOS DE UNA CANCIÓN Y PULSA "EJECUTAR ANÁLISIS"
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════
# 11. FOOTER
# ══════════════════════════════════════════════════════════
st.markdown("<br><hr>", unsafe_allow_html=True)
st.caption("🔮 ORÁCULO MUSICAL · Data Science Final Project · 4Geeks Academy · Gustavo Javier Criao · 2025")
