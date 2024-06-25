import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from ast import literal_eval
import random

def plot_movies_2d():
    df = pd.read_csv('data/movies_embeddings.csv')
    embeddings = np.array(df['embedding'].apply(literal_eval).tolist())

    tsne = TSNE(n_components=2, random_state=42)
    scaler = StandardScaler()
    embeddings_2d = tsne.fit_transform(embeddings)
    embeddings_2d = scaler.fit_transform(embeddings_2d)

    df['x'] = embeddings_2d[:, 0]
    df['y'] = embeddings_2d[:, 1]

    fig = px.scatter(
        df.head(150), x='x', y='y', text='title',
        hover_data={'title': True, 'rank': True, 'x': False, 'y': False},
        title='Movie Similarity based on Storyline Embeddings'
    )
    fig.update_traces(textposition='top center', textfont=dict(size=6))
    fig.show()

def plot_movies_3d():
    df = pd.read_csv('data/movies_embeddings.csv')
    embeddings = np.array(df['embedding'].apply(literal_eval).tolist())

    tsne = TSNE(n_components=3, random_state=42)
    scaler = StandardScaler()
    embeddings_3d = tsne.fit_transform(embeddings)
    embeddings_3d = scaler.fit_transform(embeddings_3d)

    df['x'] = embeddings_3d[:, 0]
    df['y'] = embeddings_3d[:, 1]
    df['z'] = embeddings_3d[:, 2]

    fig = px.scatter_3d(
        df.head(150), x='x', y='y', z='z', text='title',
        hover_data={'title': True, 'rank': True, 'x': False, 'y': False, 'z': False},
        title='Movie Similarity based on Storyline Embeddings'
    )
    fig.update_traces(textposition='top center', textfont=dict(size=6))
    fig.show()


if __name__ == '__main__':
    plot_movies_2d()
