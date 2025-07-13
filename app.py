import os
import time
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

# === Load credentials from Streamlit secrets ===
client_id = st.secrets["SPOTIPY_CLIENT_ID"]
client_secret = st.secrets["SPOTIPY_CLIENT_SECRET"]

# === Authenticate ===
auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(auth_manager=auth_manager)

# === UI CONFIG ===
st.set_page_config(page_title="🎧 Spotify Mood Analyzer", layout="wide")
st.title("🎵 Advanced Spotify Mood Analyzer")
st.caption("Clustering moods using audio features with Streamlit + Spotipy + ML")

# === Sidebar Input ===
artist = st.sidebar.text_input("Enter Artist Name:", value="Arijit Singh")
limit = st.sidebar.slider("Number of Tracks to Analyze", 5, 50, 15)

# === Spotify Helper Functions ===
def fetch_artist_id(artist_name):
    results = sp.search(q=artist_name, type='artist', limit=1)
    items = results.get("artists", {}).get("items", [])
    if items:
        return items[0]["id"]
    return None

@st.cache_data(show_spinner=False)
def fetch_tracks_by_artist_id(artist_id, limit):
    results = sp.artist_top_tracks(artist_id, country="IN")
    return results["tracks"][:limit]

@st.cache_data(show_spinner=False)
def fetch_audio_features(track_ids):
    all_features = []
    for tid in track_ids:
        try:
            features = sp.audio_features(tid)[0]
            if features and all(k in features and features[k] is not None for k in ["valence", "energy", "danceability", "acousticness", "tempo"]):
                all_features.append(features)
            time.sleep(0.1)
        except:
            continue
    return all_features

def create_dataframe(tracks, features):
    data = []
    for track, f in zip(tracks, features):
        if all(col in f and f[col] is not None for col in ["valence", "energy", "danceability", "acousticness", "tempo"]):
            data.append({
                "track_name": track["name"],
                "artist": track["artists"][0]["name"],
                "preview_url": track["preview_url"],
                "album_image": track["album"]["images"][0]["url"],
                "popularity": track["popularity"],
                "valence": f["valence"],
                "energy": f["energy"],
                "danceability": f["danceability"],
                "acousticness": f["acousticness"],
                "tempo": f["tempo"]
            })
    return pd.DataFrame(data)

def cluster_and_label(df):
    # Only keep numeric audio features
    required_cols = ["valence", "energy", "danceability", "acousticness", "tempo"]
    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        st.error(f"Missing required audio feature columns: {missing}")
        return df

    X = df[required_cols]

    if len(X) <= 1:
        df["mood_cluster"] = 0
        df["mood"] = "Undefined"
        return df

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=min(4, len(df)), random_state=42)
    df["mood_cluster"] = kmeans.fit_predict(X_scaled)

    # Mood labeling
    mood_labels = ["Calm", "Happy", "Sad", "Energetic"]
    df["mood"] = df["mood_cluster"].map(lambda x: mood_labels[x % len(mood_labels)])

    # PCA for plot
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    df["PCA1"] = components[:, 0]
    df["PCA2"] = components[:, 1]

    return df

# === Main App Logic ===
if st.button("🎯 Analyze Tracks"):
    with st.spinner("Fetching data from Spotify..."):
        artist_id = fetch_artist_id(artist)
        if not artist_id:
            st.error("❌ Artist not found.")
            st.stop()

        tracks = fetch_tracks_by_artist_id(artist_id, limit)
        if not tracks:
            st.error("No tracks found.")
            st.stop()

        ids = [t["id"] for t in tracks]
        features = fetch_audio_features(ids)

        if not features:
            st.error("❌ No valid audio features found. Try with a different artist or reduce the number of tracks.")
            st.stop()

        df = create_dataframe(tracks, features)
        df = cluster_and_label(df)

        if "mood" not in df.columns:
            st.error("⚠️ Mood clustering failed. Please try with a different artist or fewer tracks.")
            st.stop()

        st.success("Analysis Complete ✅")

        # 📊 Mood Distribution
        st.subheader("📊 Mood Distribution")
        mood_counts = df["mood"].value_counts()
        st.bar_chart(mood_counts)

        # 🖼️ PCA Scatterplot
        st.subheader("🖼️ Mood Clusters (2D PCA)")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df, x="PCA1", y="PCA2", hue="mood", s=100, palette="Set2", ax=ax)
        st.pyplot(fig)

        # 🎶 Track Explorer
        st.subheader("🎶 Track Explorer")
        for _, row in df.iterrows():
            col1, col2 = st.columns([1, 4])
            with col1:
                st.image(row["album_image"], width=100)
            with col2:
                st.markdown(f"**{row['track_name']}** by *{row['artist']}*")
                st.markdown(f"Mood: {row['mood']} | Popularity: {row['popularity']}")
                if row["preview_url"]:
                    st.audio(row["preview_url"], format="audio/mp3")
                else:
                    st.write("⚠️ No preview available.")

        # 💾 Download CSV
        st.download_button(
            label="📥 Download Mood Data",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="spotify_mood_data.csv",
            mime="text/csv"
        )
