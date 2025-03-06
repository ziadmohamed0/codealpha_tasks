import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
import numpy as np

def load_data():
    """Load all datasets and merge necessary information."""
    data = pd.read_csv("data.csv")
    data_by_artist = pd.read_csv("data_by_artist.csv")
    data_by_genres = pd.read_csv("data_by_genres.csv")
    data_by_year = pd.read_csv("data_by_year.csv")
    data_w_genres = pd.read_csv("data_w_genres.csv")
    return data, data_by_artist, data_by_genres, data_by_year, data_w_genres

def prepare_features(data):
    """Extract numerical features and standardize them."""
    features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data[features])
    return data_scaled, features

def train_model(data_scaled):
    """Train Nearest Neighbors model on song features."""
    model = NearestNeighbors(n_neighbors=5, algorithm='auto')
    model.fit(data_scaled)
    return model

def recommend_songs(song_name, data, model, features):
    """Find and recommend similar songs."""
    if song_name not in data['name'].values:
        print("Song not found.")
        return
    
    song_index = data[data['name'] == song_name].index[0]
    song_features = data.iloc[song_index][features].values.reshape(1, -1)
    distances, indices = model.kneighbors(song_features)
    recommendations = data.iloc[indices[0]]['name'].tolist()
    print(f"🎵 Recommendations for '{song_name}':")
    for song in recommendations[1:]:
        print(f"- {song}")

if __name__ == "__main__":
    data, data_by_artist, data_by_genres, data_by_year, data_w_genres = load_data()
    data_scaled, features = prepare_features(data)
    model = train_model(data_scaled)
    
    song_to_recommend = "Dans La Vie Faut Pas S'en Faire"
    recommend_songs(song_to_recommend, data, model, features)
