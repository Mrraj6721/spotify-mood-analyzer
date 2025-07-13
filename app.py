import os
import time
import pandas as pd
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

# === Authenticate with Spotify ===
auth_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
sp = spotipy.Spotify(auth_manager=auth_manager)

# === Page Config ===
st.set_page_config(page_title="🎧 Spotify Mood Analyzer", layout="wide")
st.title("🎵 Advanced Spotify Mood Analyzer")
st.caption("Clustering moods using audio features with Streamlit + Spotipy + ML")

# === Sidebar ===
artist = st.sidebar.text_input("Enter Artist Name:", value="Arijit Singh")
limit = st.sidebar.slider("Number of Tracks to Analyze", 5, 50, 15)

# === Relaxed Filter: Fetch Tracks ===
@st.cache_data(show_spinner=False)
def fetch_tracks(artist_query, limit):
    try:
        results = sp.search(q=f"artist:{artist_query}", type='track', limit=50, market="IN")
        items = results["tracks"]["items"]
        st.info(f"🔍 Total tracks fetched from Spotify: **{len(items)}**")

        # ✅ Relaxed: Only needs preview and album image
        filtered = [
            t for t in items
            if t.get("preview_url") and t.get("album", {}).get("images")
        ]
        st.success(f"✅ Tracks with valid preview & image: **{len(filtered)}**")
        return filtered[:limit]

    except Exception as e:
        st.error(f"⚠️ Error while fetching tracks: {e}")
        return []

# === Fetch Audio Features ===
@st.cache_data(show_spinner=False)
def fetch_audio_features(track_ids):
    features = []
    for tid in track_ids:
        try:
            af = sp.audio_features([tid])[0]
            if af:
                features.append(af)
            else:
                st.warning(f"⚠️ Failed to get audio features for track ID: {tid}")
        except:
            st.warning(f"⚠️ Error fetching features for track ID: {tid}")
        time.sleep(0.2)
    return features

# === Build DataFrame ===
def create_dataframe(tracks, features):
    data = []
    for t, f in zip(tracks, features):
        if f:  # Ensure audio features exist
            data.append({
                "track_name": t["name"],
                "artist": t["artists"][0]["name"],
                "preview_url": t["preview_url"],
                "album_image": t["album"]["images"][0]["url"],
                "popularity": t["popularity"],
                "valence": f["valence"],
                "energy": f["energy"],
                "danceability": f["danceability"],
                "acousticness": f["acousticness"],
                "tempo": f["tempo"]
            })
    return pd.DataFrame(data)

# === Cluster and Label ===
def cluster_and_label(df):
    required_cols = ["valence", "energy", "danceability", "acousticness", "tempo"]
    if not all(col in df.columns for col in required_cols):
        st.error("❌ Missing required audio features. Clustering skipped.")
        return df

    if len(df) <= 1:
        df["mood_cluster"] = 0
        df["mood"] = "Undefined"
        return df

    X = df[required_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=min(4, len(df)), random_state=42)
    df["mood_cluster"] = kmeans.fit_predict(X_scaled)

    mood_labels = ["Calm 🧘", "Energetic ⚡", "Romantic 💖", "Sad 😢"]
    df["mood"] = df["mood_cluster"].map(lambda x: mood_labels[x % len(mood_labels)])

    # PCA for 2D Plot
    pca = PCA(n_components=2)
    components = pca.fit_transform(X_scaled)
    df["PCA1"] = components[:, 0]
    df["PCA2"] = components[:, 1]

    return df

# === Main Button Action ===
if st.button("🎯 Analyze Tracks"):
    with st.spinner("Fetching data from Spotify..."):
        tracks = fetch_tracks(artist, limit)

        if not tracks:
            st.error("❌ No tracks found. Try a different artist.")
            st.stop()

        track_ids = [t["id"] for t in tracks]
        features = fetch_audio_features(track_ids)

        if not features:
            st.error("❌ No valid audio features found. Try with a different artist or reduce the number of tracks.")
            st.stop()

        df = create_dataframe(tracks, features)

        if df.empty:
            st.error("❌ No valid tracks to process.")
            st.stop()

        df = cluster_and_label(df)

        st.success("🎧 Mood clustering complete!")

        # === Mood Chart ===
        st.subheader("📊 Mood Distribution")
        st.bar_chart(df["mood"].value_counts())

        # === PCA Scatterplot ===
        st.subheader("🖼️ Mood Clusters (2D PCA)")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df, x="PCA1", y="PCA2", hue="mood", palette="Set2", s=100, ax=ax)
        st.pyplot(fig)

        # === Track Explorer ===
        st.subheader("🎶 Track Explorer")
        for _, row in df.iterrows():
            col1, col2 = st.columns([1, 4])
            with col1:
                st.image(row["album_image"], width=100)
            with col2:
                st.markdown(f"**{row['track_name']}** by *{row['artist']}*")
                st.markdown(f"Mood: {row['mood']} | Popularity: {row['popularity']}")
                if row["preview_url"]:
                    st.audio(row["preview_url"])
                else:
                    st.write("⚠️ No preview available.")

        # === Download Button ===
        st.download_button(
            label="📥 Download Mood Data as CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="spotify_mood_data.csv",
            mime="text/csv"
        )
