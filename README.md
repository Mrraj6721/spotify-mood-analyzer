# 🎵 Advanced Spotify Mood Analyzer

An interactive music analysis web app that uses Spotify audio features to cluster tracks into emotional moods like Happy, Calm, Sad, or Energetic. Built using **Streamlit**, **Spotipy**, and **Scikit-learn**, it's perfect for data-driven music exploration.

---

## 🚀 Features

* 🔍 Search tracks by any **artist name**
* 🎧 Fetch audio features: `valence`, `energy`, `danceability`, etc.
* 🧠 Use **K-Means** clustering to group songs into **4 moods**
* 📊 Display mood clusters using **PCA-based 2D plots**
* 🎶 Preview songs directly in the app
* 📅 Export clustered data as **CSV**

---

## 🚫 Prerequisites

You need a **Spotify Developer Account** to get:

* `SPOTIPY_CLIENT_ID`
* `SPOTIPY_CLIENT_SECRET`

Create credentials here: [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)

---

## 🔧 Local Installation Guide

### 1. Clone Repository

```bash
git clone https://github.com/Mrraj6721/spotify-mood-analyzer.git
cd spotify-mood-analyzer
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate     # On Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Create `.streamlit/secrets.toml`

```toml
# Inside .streamlit/secrets.toml
SPOTIPY_CLIENT_ID = "your_client_id"
SPOTIPY_CLIENT_SECRET = "your_client_secret"
```

### 5. Run the App

```bash
streamlit run app.py
```

---

## ☁️ Deploy to Streamlit Cloud

1. Push your project to GitHub
2. Go to [https://streamlit.io/cloud](https://streamlit.io/cloud) and log in
3. Click **New app**, connect your repo, and choose `app.py`
4. Add your secrets in **App Settings > Secrets**

```toml
SPOTIPY_CLIENT_ID = "your_client_id"
SPOTIPY_CLIENT_SECRET = "your_client_secret"
```

5. Click **Deploy**

---

## 📅 Example Use Case

* Enter artist: `Arijit Singh`
* Select number of tracks: `15`
* Click `Analyze`
* View mood clusters
* Preview tracks & download results

---

## 📊 Audio Features Used

| Feature        | Description                              |
| -------------- | ---------------------------------------- |
| `valence`      | Positivity of track (0 = sad, 1 = happy) |
| `energy`       | Intensity and activity                   |
| `danceability` | Suitability for dancing                  |
| `acousticness` | Whether the track is acoustic            |
| `tempo`        | Speed in beats per minute                |

---

## 🌐 Project Structure

```
Spotify_advance/
├── app.py
├── requirements.txt
├── .streamlit/
│   └── secrets.toml
├── assets/  # optional for images
└── README.md
```

---

## 📄 License

MIT License © 2025 [Prashant Raj](https://github.com/Mrraj6721)

---

## 🙌 Credits

* [Spotipy](https://spotipy.readthedocs.io/)
* [Spotify Web API](https://developer.spotify.com/)
* [Streamlit](https://streamlit.io/)
* [Scikit-learn](https://scikit-learn.org/)
