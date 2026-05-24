# 🔮 Oráculo Musical — Predictor de Hits con Machine Learning

> ¿Tiene tu canción lo que se necesita para ser un éxito mundial? La IA lo sabe.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-RandomForest-orange?logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-En%20producción-brightgreen)

---

## 📌 Descripción del Proyecto

**Oráculo Musical** es un modelo predictivo de Machine Learning entrenado para identificar si una canción tiene el perfil de un **Hit mundial** o de un **tema de nicho**.

El modelo fue construido a partir de datos reales de **Last.fm**, analizando patrones de oyentes, reproducciones y etiquetas de géneros musicales para aprender qué distingue a un éxito de una canción que pasa desapercibida.

La aplicación cuenta con una interfaz web interactiva desarrollada en **Streamlit** que permite introducir los datos de cualquier canción y obtener una predicción en tiempo real, junto con la probabilidad matemática calculada por el modelo.

---

## 🎯 Problema que Resuelve

El sector musical mueve miles de millones de euros al año, pero la industria sigue apostando mayoritariamente por intuición y experiencia. Este proyecto demuestra que es posible usar datos y algoritmos para anticipar el potencial comercial de una canción **antes** de invertir en su producción o promoción.

Orientado a:
- Discográficas y scouting de talento
- Agencias de publicidad en búsqueda de música para campañas
- Artistas independientes que quieren validar su trabajo

---

## 🧠 Stack Tecnológico

| Categoría | Tecnología |
|---|---|
| Lenguaje | Python 3.11+ |
| Machine Learning | Scikit-Learn (Random Forest Classifier) |
| Análisis de datos | Pandas, NumPy |
| Interfaz web | Streamlit |
| Datos fuente | Last.fm (via API / dataset propio) |
| Serialización del modelo | Joblib |
| Entorno | GitHub Codespaces / Local |

---

## 📊 Rendimiento del Modelo

| Métrica | Valor |
|---|---|
| Algoritmo | Random Forest Classifier |
| Precisión (Accuracy) | **79.9%** |
| Dataset de entrenamiento | `dataset_lastfm_ML_listo.csv` |

El modelo fue entrenado con features de ingeniería propias:
- `longitud_nombre_cancion` — longitud del título
- `longitud_nombre_artista` — longitud del nombre del artista
- `ratio_reproducciones_oyentes` — ratio de engagement
- `tag_*` — etiquetas de género musical (one-hot encoding)

---

## 🚀 Cómo Ejecutar el Proyecto

### ⚡ Opción 1: GitHub Codespaces (recomendado)

1. Haz clic en **Code → Open with Codespaces**
2. Espera a que el entorno se configure automáticamente
3. Ejecuta la aplicación:

```bash
streamlit run app.py
```

### 💻 Opción 2: Ejecución Local

**Prerrequisitos:** Python 3.11+

```bash
# 1. Clona el repositorio
git clone https://github.com/javiercriao5-creator/Criao-javier-proyecto-final.git
cd Criao-javier-proyecto-final

# 2. Instala las dependencias
pip install -r requirements.txt

# 3. Lanza la aplicación
streamlit run app.py
```

La app se abrirá automáticamente en `http://localhost:8501`

---

## 🗂️ Estructura del Proyecto

```
Criao-javier-proyecto-final/
│
├── app.py                          # App principal de Streamlit
├── oraculo_musical_modelo.pkl      # Modelo entrenado (Random Forest)
├── dataset_lastfm_ML_listo.csv     # Dataset procesado de Last.fm
├── requirements.txt                # Dependencias del proyecto
├── .env.example                    # Variables de entorno de ejemplo
│
├── src/
│   └── explore.ipynb               # Notebook de exploración y EDA
│
├── data/
│   ├── raw/                        # Datos originales sin procesar
│   ├── interim/                    # Datos transformados temporalmente
│   └── processed/                  # Datos listos para modelado
│
└── models/                         # Modelos y artefactos de ML
```

---

## 🖥️ Uso de la Aplicación

1. Introduce el **nombre de la canción** y el **nombre del artista**
2. Estima los **oyentes mensuales** y las **reproducciones**
3. Selecciona hasta 3 **géneros musicales** que describan la canción
4. Pulsa **"🔮 Predecir Éxito"**
5. El Oráculo te revelará si tienes un **Hit mundial** o un **tema de nicho**, junto con la probabilidad matemática calculada por el modelo

---

## 👤 Autor

**Gustavo Javier Criao**
Profesional Eléctrico en transición a Data Science | 4Geeks Academy

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Conectar-0077B5?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/https://www.linkedin.com/in/gustavo-javier-criao-187824222/)
[![GitHub](https://img.shields.io/badge/GitHub-javiercriao5--creator-181717?logo=github&logoColor=white)](https://github.com/javiercriao5-creator)

---

## 📄 Licencia

Este proyecto fue desarrollado como parte del **Data Science and Machine Learning Bootcamp** de [4Geeks Academy](https://4geeksacademy.com).
