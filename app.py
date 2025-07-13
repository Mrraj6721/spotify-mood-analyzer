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
@st.cache_data(show_spinner=False)
def fetch_tracks(artist_query, limit):
    results = sp.search(q=f"artist:{artist_query}", type='track', limit=limit, market="IN")
    return results["tracks"]["items"]

@st.cache_data(show_spinner=False)
def fetch_audio_features(track_ids):
    all_features = []
    for tid in track_ids:
        try:
            features = sp.audio_features(tid)[0]
            if features:
                all_features.append(features)
            time.sleep(0.1)
        except:
            continue
    return all_features

def create_dataframe(tracks, features):
    data = []
    for track, f in zip(tracks, features):
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
    X = df[["valence", "energy", "danceability", "acousticness", "tempo"]]
    X_scaled = StandardScaler().fit_transform(X)

    kmeans = KMeans(n_clusters=4, random_state=42)
    df["cluster"] = kmeans.fit_predict(X_scaled)

    mood_map = {
        0: "Chill 🧘",
        1: "Energetic ⚡",
        2: "Romantic 💖",
        3: "Sad 😢"
    }
    df["mood"] = df["cluster"].map(lambda x: mood_map.get(x, f"Mood {x}"))

    # PCA for plot
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    df["PCA1"] = components[:, 0]
    df["PCA2"] = components[:, 1]

    return df

# === Main App Logic ===
if st.button("🎯 Analyze Tracks"):
    with st.spinner("Fetching data from Spotify..."):
        tracks = fetch_tracks(artist, limit)
        if not tracks:
            st.error("No tracks found.")
            st.stop()

        ids = [t["id"] for t in tracks]
        features = fetch_audio_features(ids)
        df = create_dataframe(tracks, features)
        df = cluster_and_label(df)

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

        # 🎧 Track Explorer
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
