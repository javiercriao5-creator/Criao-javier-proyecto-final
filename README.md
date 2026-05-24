# 🔮 Music Oracle — Hit Predictor with Machine Learning

> Does your song have what it takes to be a worldwide hit? AI knows.


![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-RandomForest-orange?logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/Status-En%20producción-brightgreen)

---

## 📌 Project Description

**Musical Oracle** is a predictive Machine Learning model trained to identify whether a song has the profile of a **global hit** or a **song of niche**.

The model was built using real data from **Last.fm**, analyzing listener patterns, plays, and music genre tags to learn what distinguishes a hit song from one that goes unnoticed.

The application features an interactive web interface developed in **Streamlit** that allows users to input data for any song and obtain a real-time prediction, along with the mathematical probability calculated by the model.

---

## 🎯 Problem Solved

The music industry generates billions of euros annually, but the sector still relies primarily on intuition and experience. This project demonstrates that it is possible to use data and algorithms to anticipate the commercial potential of a song **before** investing in its production or promotion.

Target Audiences:
- Record labels and talent scouting
- Advertising agencies seeking music for campaigns
- Independent artists seeking to validate their work

---

## 🧠 Technology Stack

| Category | Technology |

---|---|

| Language | Python 3.11+ |

| Machine Learning | Scikit-Learn (Random Forest Classifier) ​​|

| Data Analysis | Pandas, NumPy |

| Web Interface | Streamlit |

| Source Data | Last.fm (via API / custom dataset) |

| Model Serialization | Joblib |

| Environment | GitHub Codespaces / Local |

---

## 📊 Model Performance

| Metric | Value |

---|---|

| Algorithm | Random Forest Classifier |

| Accuracy | 79.9% |

| Training Dataset | `dataset_lastfm_ML_listo.csv` |

The model was trained using custom engineering features:
- `song_name_length` — title length
- `artist_name_length` — artist name length
- `listener_play_ratio` — engagement rate
- `tag_*` — music genre tags (one-hot encoding)

---

## 🚀 How to Run the Project

### ⚡ Option 1: GitHub Codespaces (recommended)

1. Click on **Code → Open with Codespaces**
2. Wait for the environment to configure automatically
3. Run the application:

```bash
streamlit run app.py
```

### 💻 Option 2: Local Run

**Prerequisites:** Python 3.11+

```bash
# 1. Clone the repository
git clone https://github.com/javiercriao5-creator/Criao-javier-proyecto-final.git
cd Criao-javier-proyecto-final

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the application
streamlit run app.py
```

The app will automatically open at `http://localhost:8501`

---

## 🗂️ Project Structure

```
Criao-javier-proyecto-final/
│
├── app.py # Main Streamlit app
├── oraculo_musical_modelo.pkl # Trained model (Random Forest)
├── dataset_lastfm_ML_listo.csv # Processed Last.fm dataset
├── requirements.txt # Project dependencies
├── .env.example # Example environment variables
│
├── src/
│ └── explore.ipynb # Exploration and EDA notebook
│
├── data/
│ ├── raw/ # Raw data
│ ├── interim/ # Temporarily transformed data
│ └── processed/ # Data ready for modeling
│
└── models/ # ML models and artifacts
```

---

## 🖥️ Using the Application

1. Enter the **song title** and **artist name**
2. Estimate the **monthly listeners** and the **Plays**
3. Select up to 3 **musical genres** that describe the song
4. Tap **"🔮 Predict Success"**
5. The Oracle will reveal whether you have a **global hit** or a **niche track**, along with the mathematical probability calculated by the model

---

## 👤 Author

**Gustavo Javier Criao**
Electrical Engineer transitioning to Data Science | 4Geeks Academy

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Conectar-0077B5?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/https://www.linkedin.com/in/gustavo-javier-criao-187824222/)
[![GitHub](https://img.shields.io/badge/GitHub-javiercriao5--creator-181717?logo=github&logoColor=white)](https://github.com/javiercriao5-creator)

---

## 📄 License

This project was developed as part of the **Data Science and Machine Learning Bootcamp** of [4Geeks Academy](https://4geeksacademy.com).
