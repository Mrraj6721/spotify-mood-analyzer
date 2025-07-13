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

# === Load credentials from secrets ===
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
    results = sp.search(q=f"artist:{artist_query}", type='track', limit=50, market="IN")
    items = results["tracks"]["items"]
    filtered = [
        t for t in items
        if t.get("preview_url") and t.get("popularity", 0) > 40 and t.get("album", {}).get("images")
    ]
    return filtered[:limit]

@st.cache_data(show_spinner=False)
def fetch_audio_features(track_ids):
    all_features = []
    for tid in track_ids:
        try:
            features = sp.audio_features(tid)[0]
            if features:
                all_features.append(features)
            else:
                st.warning(f"⚠️ No audio features for track ID: {tid}")
            time.sleep(0.1)
        except:
            st.warning(f"⚠️ Failed to get audio features for track ID: {tid}")
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
        tracks = fetch_tracks(artist, limit)
        if not tracks:
            st.error("❌ No tracks found. Try a different artist.")
            st.stop()

        ids = [t["id"] for t in tracks]
        features = fetch_audio_features(ids)

        if not features:
            st.error("❌ No valid audio features found. Try with a different artist like 'Taylor Swift'.")
            st.stop()

        df = create_dataframe(tracks, features)
        df = cluster_and_label(df)

        st.success("🎧 Analysis Complete!")

        st.subheader("📊 Mood Distribution")
        mood_counts = df["mood"].value_counts()
        st.bar_chart(mood_counts)

        st.subheader("🖼️ Mood Clusters (2D PCA)")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df, x="PCA1", y="PCA2", hue="mood", s=100, palette="Set2", ax=ax)
        st.pyplot(fig)

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

        st.download_button(
            label="📥 Download Mood Data",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="spotify_mood_data.csv",
            mime="text/csv"
        )
