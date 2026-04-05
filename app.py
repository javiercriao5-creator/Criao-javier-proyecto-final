import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# 1. Configuración de la página
st.set_page_config(page_title="Oráculo Musical", page_icon="🔮", layout="centered")

st.title("🔮 El Oráculo Musical")
st.markdown("¿Tienes una canción en mente? Introduce sus datos y nuestra IA (un modelo de *Random Forest* con 79.9% de precisión) te dirá si tiene madera de **Hit Mundial**.")

# 2. Cargar el modelo y los datos base (usamos caché para que sea súper rápido)
@st.cache_resource
def load_model():
    return joblib.load('oraculo_musical_modelo.pkl')

@st.cache_data
def load_data():
    # Cargamos el dataset para recrear el escalador y obtener las columnas exactas
    df = pd.read_csv('dataset_lastfm_ML_listo.csv')
    X = df.drop(columns=['nombre_cancion', 'nombre_artista', 'url', 'oyentes', 'reproducciones', 'es_hit']).fillna(0)
    
    cols_num = ['longitud_nombre_cancion', 'longitud_nombre_artista', 'ratio_reproducciones_oyentes']
    scaler = StandardScaler()
    scaler.fit(X[cols_num]) # Entrenamos el escalador con los datos originales
    
    # Extraemos los nombres limpios de los tags (quitando 'tag_' y poniendo mayúscula)
    tags = [c.replace('tag_', '').title() for c in X.columns if c.startswith('tag_')]
    return X.columns, scaler, tags

modelo = load_model()
columnas_modelo, scaler, lista_tags = load_data()

# 3. Interfaz de usuario (Barra lateral)
st.sidebar.header("🎶 Datos de la Canción")
nombre_cancion = st.sidebar.text_input("Nombre de la canción", "Bohemian Rhapsody")
nombre_artista = st.sidebar.text_input("Nombre del artista", "Queen")
oyentes = st.sidebar.number_input("Oyentes mensuales estimados", min_value=1, value=500000)
reproducciones = st.sidebar.number_input("Reproducciones estimadas", min_value=1, value=2500000)
tags_seleccionados = st.sidebar.multiselect("Selecciona los géneros (1 a 3 recomendados):", lista_tags, default=[lista_tags[0]])

# 4. Botón de Predicción y Lógica
if st.sidebar.button("🔮 Predecir Éxito"):
    with st.spinner('La IA está analizando la canción...'):
        # Calculamos nuestras variables creadas en el EDA
        len_cancion = len(nombre_cancion)
        len_artista = len(nombre_artista)
        ratio = reproducciones / (oyentes + 1)

        # Creamos una fila vacía con las columnas que espera el modelo
        entrada = pd.DataFrame(columns=columnas_modelo)
        entrada.loc[0] = 0.0 # Llenamos de ceros decimales inicialmente

        # Insertamos los datos numéricos
        entrada.loc[0, 'longitud_nombre_cancion'] = len_cancion
        entrada.loc[0, 'longitud_nombre_artista'] = len_artista
        entrada.loc[0, 'ratio_reproducciones_oyentes'] = ratio

        # Activamos (ponemos a 1) los tags que eligió el usuario
        for t in tags_seleccionados:
            col_name = f"tag_{t.lower()}"
            if col_name in columnas_modelo:
                entrada.loc[0, col_name] = 1

        # IMPORTANTE: Normalizamos los números igual que hicimos al entrenar
        cols_num = ['longitud_nombre_cancion', 'longitud_nombre_artista', 'ratio_reproducciones_oyentes']
        entrada[cols_num] = scaler.transform(entrada[cols_num])

        # Hacemos la predicción
        prediccion = modelo.predict(entrada)[0]
        probabilidad = modelo.predict_proba(entrada)[0][1]

        # Mostramos el resultado de forma bonita
        st.markdown("---")
        if prediccion == 1:
            st.success(f"🌟 **¡TENEMOS UN HIT!**")
            st.write(f"La IA predice que **'{nombre_cancion}'** será un éxito rotundo.")
            st.balloons() # Lanza globos en la pantalla
        else:
            st.warning(f"📉 **Canción de Nicho.**")
            st.write(f"La IA cree que **'{nombre_cancion}'** tendrá su público, pero no alcanzará el estatus de Hit global.")

        st.info(f"Probabilidad matemática de ser un Hit: **{probabilidad * 100:.2f}%**")
