import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from ast import literal_eval

def plot_movies_2d(file='data/movies_embeddings.csv'):
    df = pd.read_csv(file)
    embeddings = np.array(df['embedding'].apply(literal_eval).tolist())

    tsne = TSNE(n_components=2)
    scaler = StandardScaler()
    embeddings_2d = tsne.fit_transform(embeddings)
    embeddings_2d = scaler.fit_transform(embeddings_2d)

    df['x'] = embeddings_2d[:, 0]
    df['y'] = embeddings_2d[:, 1]

    fig = px.scatter(
        df, x='x', y='y', text='title',
        hover_data={'title': True, 'x': False, 'y': False},
        title='Movie Similarity based on Plot Embeddings'
    )
    fig.update_traces(textposition='top center', textfont=dict(size=6))
    fig.show()

def plot_movies_3d(file='data/movies_embeddings.csv'):
    df = pd.read_csv(file)
    embeddings = np.array(df['embedding'].apply(literal_eval).tolist())

    tsne = TSNE(n_components=3)
    scaler = StandardScaler()
    embeddings_3d = tsne.fit_transform(embeddings)
    embeddings_3d = scaler.fit_transform(embeddings_3d)

    df['x'] = embeddings_3d[:, 0]
    df['y'] = embeddings_3d[:, 1]
    df['z'] = embeddings_3d[:, 2]

    fig = px.scatter_3d(
        df, x='x', y='y', z='z', text='title',
        hover_data={'title': True, 'x': False, 'y': False, 'z': False},
        title='Movie Similarity based on Plot Embeddings'
    )
    fig.update_traces(textposition='top center', textfont=dict(size=6))
    fig.show()


if __name__ == '__main__':
    plot_movies_3d()
