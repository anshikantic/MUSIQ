# MUSIQ 🎵
## Intelligent Music Discovery & Personality Engine

MUSIQ is a Machine Learning-based music recommendation system that helps users discover songs based on their musical preferences, listening patterns, and audio characteristics.

Unlike traditional recommendation systems, MUSIQ goes beyond simply recommending similar songs. It explores a user's musical personality, identifies different music moods, and helps users discover songs outside their usual genres.

## ✨ Features

- **Song Clustering:** Uses K-Means clustering to group songs with similar audio characteristics.
- **Music Recommendations:** Recommends similar songs using cosine similarity.
- **Musical DNA:** Builds a musical personality profile based on users' favourite songs.
- **Mood Explorer:** Categorizes music into different moods using audio features.
- **Genre Escape:** Helps users discover music from genres outside their usual preferences.
- **Cluster Visualization:** Uses PCA to visualize song clusters in 2D.
- **Interactive Dashboard:** Provides an easy-to-use interface through Streamlit.

## 🧠 Machine Learning Techniques

- K-Means Clustering
- Principal Component Analysis (PCA)
- Cosine Similarity
- Feature Scaling
- Data Preprocessing

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Plotly
- Matplotlib
- Seaborn

## 📂 Project Structure

```text
MUSIQ/
│
├── app.py
├── dataset/
│   └── Spotify dataset.csv
│
├── models/
│
├── src/
│   ├── clustering
│   ├── dataloader
│   ├── moodanalysis
│   ├── music_personality
│   ├── preprocessing
│   └── recommendation
│
├── requirements.txt
├── README.md
└── .gitignore
```

## 🚀 Installation and Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_LINK
```

### 2. Open the project folder

```bash
cd MUSIQ
```

### 3. Install dependencies

```bash
py -m pip install -r requirements.txt
```

### 4. Run the application

```bash
py -m streamlit run app.py
```

The application will open in your browser.

## 📊 Dataset

MUSIQ uses a Spotify tracks dataset containing song information and audio features such as:

- Danceability
- Energy
- Valence
- Acousticness
- Instrumentalness
- Tempo
- Loudness
- Speechiness
- Popularity

These features are used to analyze songs and generate recommendations.

## 🎯 Project Objective

The main objective of MUSIQ is to make music discovery more personalized and engaging by combining machine learning, data analysis, and interactive visualization.

## 👩‍💻 Project Information

**Project Name:** MUSIQ  
**Project Type:** Machine Learning / Data Science  
**Domain:** Music Recommendation System  
**Interface:** Streamlit

---

*Discover your sound. Explore your mood. Find your MUSIQ.* 🎧
