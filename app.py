import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────
# 1. CONFIGURACIÓN DE LA PÁGINA
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Oráculo Musical",
    page_icon="🔮",
    layout="centered"
)

st.title("🔮 El Oráculo Musical")
st.markdown(
    "¿Tienes una canción en mente? Introduce sus datos y nuestra IA "
    "(*Random Forest* · 79.9% de precisión) te dirá si tiene madera de **Hit Mundial**."
)

# ─────────────────────────────────────────────
# 2. CARGA DE MODELO Y DATOS (con caché)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("oraculo_musical_modelo.pkl")


@st.cache_data
def load_data():
    df = pd.read_csv("dataset_lastfm_ML_listo.csv")
    X = df.drop(
        columns=["nombre_cancion", "nombre_artista", "url", "oyentes", "reproducciones", "es_hit"]
    ).fillna(0)

    cols_num = ["longitud_nombre_cancion", "longitud_nombre_artista", "ratio_reproducciones_oyentes"]
    scaler = StandardScaler()
    scaler.fit(X[cols_num])

    tags = [c.replace("tag_", "").title() for c in X.columns if c.startswith("tag_")]
    return X.columns, scaler, tags, X


modelo = load_model()
columnas_modelo, scaler, lista_tags, X_base = load_data()

# ─────────────────────────────────────────────
# 3. BARRA LATERAL — Inputs del usuario
# ─────────────────────────────────────────────
st.sidebar.header("🎶 Datos de la Canción")

nombre_cancion = st.sidebar.text_input("Nombre de la canción", "Bohemian Rhapsody").strip()
nombre_artista = st.sidebar.text_input("Nombre del artista", "Queen").strip()
oyentes = st.sidebar.number_input("Oyentes mensuales estimados", min_value=1, value=500_000, step=10_000)
reproducciones = st.sidebar.number_input("Reproducciones estimadas", min_value=1, value=2_500_000, step=100_000)
tags_seleccionados = st.sidebar.multiselect(
    "Géneros musicales (1 a 3 recomendados):",
    lista_tags,
    default=[lista_tags[0]] if lista_tags else []
)

predecir = st.sidebar.button("🔮 Predecir Éxito")

# ─────────────────────────────────────────────
# 4. VALIDACIÓN DE INPUTS
# ─────────────────────────────────────────────
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

# ─────────────────────────────────────────────
# 5. PREDICCIÓN
# ─────────────────────────────────────────────
if predecir:
    errores = validar_inputs()

    if errores:
        for e in errores:
            st.error(f"⚠️ {e}")
    else:
        with st.spinner("La IA está analizando la canción..."):
            try:
                # Ingeniería de features
                len_cancion = len(nombre_cancion)
                len_artista = len(nombre_artista)
                ratio = reproducciones / (oyentes + 1)

                # Construimos la fila de entrada
                entrada = pd.DataFrame(columns=columnas_modelo)
                entrada.loc[0] = 0.0
                entrada.loc[0, "longitud_nombre_cancion"] = len_cancion
                entrada.loc[0, "longitud_nombre_artista"] = len_artista
                entrada.loc[0, "ratio_reproducciones_oyentes"] = ratio

                for t in tags_seleccionados:
                    col_name = f"tag_{t.lower()}"
                    if col_name in columnas_modelo:
                        entrada.loc[0, col_name] = 1

                # Normalización
                cols_num = ["longitud_nombre_cancion", "longitud_nombre_artista", "ratio_reproducciones_oyentes"]
                entrada[cols_num] = scaler.transform(entrada[cols_num])

                # Predicción
                prediccion = modelo.predict(entrada)[0]

                if hasattr(modelo, "predict_proba"):
                    probabilidad = modelo.predict_proba(entrada)[0][1]
                else:
                    probabilidad = float(prediccion)

                # ── Resultado principal ──────────────────────
                st.markdown("---")
                if prediccion == 1:
                    st.success("🌟 **¡TENEMOS UN HIT!**")
                    st.write(f"La IA predice que **'{nombre_cancion}'** de **{nombre_artista}** será un éxito rotundo.")
                    st.balloons()
                else:
                    st.warning("📉 **Canción de Nicho.**")
                    st.write(
                        f"La IA cree que **'{nombre_cancion}'** de **{nombre_artista}** "
                        "tendrá su público, pero no alcanzará el estatus de Hit global."
                    )

                st.info(f"Probabilidad matemática de ser un Hit: **{probabilidad * 100:.2f}%**")

                # ── Gauge visual de probabilidad ─────────────
                st.markdown("#### 📊 Indicador de Hit")
                fig_gauge, ax = plt.subplots(figsize=(6, 0.6))
                ax.barh([""], [probabilidad], color="#2ecc71" if prediccion == 1 else "#e74c3c", height=0.4)
                ax.barh([""], [1 - probabilidad], left=[probabilidad], color="#ecf0f1", height=0.4)
                ax.set_xlim(0, 1)
                ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
                ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
                ax.set_yticks([])
                ax.spines[["top", "right", "left"]].set_visible(False)
                st.pyplot(fig_gauge)
                plt.close(fig_gauge)

                # ── Gráfico de importancia de features ───────
                st.markdown("---")
                st.markdown("#### 🧠 ¿Qué factores más influyen en la predicción?")
                importancias = modelo.feature_importances_
                feat_names = columnas_modelo.tolist()
                feat_df = pd.DataFrame({
                    "feature": feat_names,
                    "importancia": importancias
                }).sort_values("importancia", ascending=False).head(10)

                # Nombre más legible para los tags
                feat_df["feature"] = feat_df["feature"].str.replace("tag_", "🎵 ", regex=False)
                feat_df["feature"] = feat_df["feature"].str.replace("longitud_nombre_cancion", "📝 Longitud título", regex=False)
                feat_df["feature"] = feat_df["feature"].str.replace("longitud_nombre_artista", "🎤 Longitud artista", regex=False)
                feat_df["feature"] = feat_df["feature"].str.replace("ratio_reproducciones_oyentes", "🔄 Ratio reproducciones/oyentes", regex=False)

                fig_imp, ax2 = plt.subplots(figsize=(7, 4))
                bars = ax2.barh(feat_df["feature"][::-1], feat_df["importancia"][::-1], color="#6c5ce7")
                ax2.set_xlabel("Importancia relativa")
                ax2.set_title("Top 10 features del modelo")
                ax2.spines[["top", "right"]].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig_imp)
                plt.close(fig_imp)

            except Exception as ex:
                st.error(f"❌ Error durante la predicción: {ex}")
                st.info("Comprueba que el modelo y el dataset están correctamente ubicados en la raíz del proyecto.")

# ─────────────────────────────────────────────
# 6. PIE DE PÁGINA
# ─────────────────────────────────────────────
st.markdown("---")
st.caption("Oráculo Musical · Proyecto Final de Data Science · 4Geeks Academy · Gustavo Javier Criao")
