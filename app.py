import os
import time
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seab as sns
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
def fetch_tracks_by_search(artist_name, limit):
    results = sp.search(q=f"artist:{artist_name}", type='track', limit=limit)
    return results['tracks']['items']

@st.cache_data(show_spinner=False)
def fetch_audio_features(track_ids):
    all_features = []
    for tid in track_ids:
        try:
            features = sp.audio_features([tid])
            if features and features[0] is not None:
                f = features[0]
                if all(k in f and f[k] is not None for k in ["valence", "energy", "danceability", "acousticness", "tempo"]):
                    all_features.append(f)
        except Exception as e:
            st.warning(f"⚠️ Failed to get audio features for track ID: {tid}")
        time.sleep(0.1)
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

    mood_labels = ["Calm", "Happy", "Sad", "Energetic"]
    df["mood"] = df["mood_cluster"].map(lambda x: mood_labels[x % len(mood_labels)])

    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    df["PCA1"] = components[:, 0]
    df["PCA2"] = components[:, 1]

    return df

# === Main App Logic ===
if st.button("🎯 Analyze Tracks"):
    with st.spinner("Fetching data from Spotify..."):
        tracks = fetch_tracks_by_search(artist, limit)
        if not tracks:
            st.error("No tracks found.")
            st.stop()

        st.write(f"✅ Fetched {len(tracks)} tracks. Now fetching audio features...")
        ids = [t["id"] for t in tracks]
        features = fetch_audio_features(ids)
        st.write(f"🎧 Fetched audio features for {len(features)} valid tracks.")

        if not features:
            st.error("❌ No valid audio features found. Try with a different artist or reduce the number of tracks.")
            st.stop()

        df = create_dataframe(tracks, features)
        df = cluster_and_label(df)

        st.success("Analysis Complete ✅")

        # 📊 Mood Distribution
        st.subheader("📊 Mood Distribution")
        if "mood" in df.columns:
            mood_counts = df["mood"].value_counts()
            st.bar_chart(mood_counts)
        else:
            st.warning("⚠️ Mood clustering failed. Please try with a different artist or fewer tracks.")

        # 🖼️ PCA Scatterplot
        if "PCA1" in df.columns and "PCA2" in df.columns:
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
                st.markdown(f"Mood: {row.get('mood', 'N/A')} | Popularity: {row['popularity']}")
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
