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

# === Load Spotify API credentials from Streamlit secrets ===
client_id = st.secrets["SPOTIPY_CLIENT_ID"]
client_secret = st.secrets["SPOTIPY_CLIENT_SECRET"]

# === Spotify Authentication ===
auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(auth_manager=auth_manager)

# === Page Config ===
st.set_page_config(page_title="🎧 Spotify Mood Analyzer", layout="wide")
st.title("🎵 Advanced Spotify Mood Analyzer")
st.caption("Clustering moods using audio features with Streamlit + Spotipy + ML")

# === Sidebar ===
artist = st.sidebar.text_input("Enter Artist Name:", value="Arijit Singh")
limit = st.sidebar.slider("Number of Tracks to Analyze", 5, 50, 15)

# === Fetch Spotify Tracks ===
@st.cache_data(show_spinner=False)
def fetch_tracks(artist_query, limit):
    try:
        results = sp.search(q=f"artist:{artist_query}", type='track', limit=50, market="IN")
        items = results["tracks"]["items"]
        st.info(f"🔍 Total tracks fetched from Spotify: **{len(items)}**")

        filtered = [
            t for t in items
            if t.get("preview_url") and t.get("popularity", 0) > 40 and t.get("album", {}).get("images")
        ]
        st.success(f"✅ Tracks with valid preview & popularity: **{len(filtered)}**")
        return filtered[:limit]

    except Exception as e:
        st.error(f"⚠️ Error while fetching tracks: {e}")
        return []

@st.cache_data(show_spinner=False)
def fetch_audio_features(track_ids):
    all_features = []
    for tid in track_ids:
        try:
            features = sp.audio_features(tid)[0]
            if features:
                all_features.append(features)
            else:
                st.warning(f"⚠️ Failed to get audio features for track ID: {tid}")
            time.sleep(0.1)
        except:
            st.warning(f"⚠️ Failed to get audio features for track ID: {tid}")
            continue
    return all_features

# === Convert to DataFrame ===
def create_dataframe(tracks, features):
    data = []
    for track, f in zip(tracks, features):
        if not f:
            continue
        data.append({
            "track_name": track["name"],
            "artist": track["artists"][0]["name"],
            "preview_url": track["preview_url"],
            "album_image": track["album"]["images"][0]["url"],
            "popularity": track["popularity"],
            "valence": f.get("valence", 0),
            "energy": f.get("energy", 0),
            "danceability": f.get("danceability", 0),
            "acousticness": f.get("acousticness", 0),
            "tempo": f.get("tempo", 0)
        })
    return pd.DataFrame(data)

# === Cluster and Label Moods ===
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

    # Label moods
    mood_labels = ["Calm 🧘", "Happy 😊", "Sad 😢", "Energetic ⚡"]
    df["mood"] = df["mood_cluster"].map(lambda x: mood_labels[x % len(mood_labels)])

    # PCA for 2D plotting
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    df["PCA1"] = components[:, 0]
    df["PCA2"] = components[:, 1]
    return df

# === Main App ===
if st.button("🎯 Analyze Tracks"):
    with st.spinner("Fetching data from Spotify..."):
        tracks = fetch_tracks(artist, limit)
        if not tracks:
            st.error("❌ No tracks found. Try a different artist.")
            st.stop()

        track_ids = [t["id"] for t in tracks]
        features = fetch_audio_features(track_ids)

        if not features or len(features) < 1:
            st.error("❌ No valid audio features found. Try with a different artist or reduce the number of tracks.")
            st.stop()

        df = create_dataframe(tracks, features)
        if df.empty:
            st.error("❌ Failed to create track data. Please try again.")
            st.stop()

        df = cluster_and_label(df)
        st.success("✅ Mood analysis complete!")

        # Mood Distribution Chart
        st.subheader("📊 Mood Distribution")
        st.bar_chart(df["mood"].value_counts())

        # PCA Plot
        st.subheader("🖼️ Mood Clusters (2D PCA)")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df, x="PCA1", y="PCA2", hue="mood", s=100, palette="Set2", ax=ax)
        st.pyplot(fig)

        # Track Explorer
        st.subheader("🎧 Track Explorer")
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

        # Download CSV
        st.download_button(
            label="📥 Download Mood Data",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="spotify_mood_data.csv",
            mime="text/csv"
        )
