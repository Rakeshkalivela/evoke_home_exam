import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

class ItemCF:
    def __init__(self, k: int = 10, metric: str = "cosine"):
        self.k = k
        self.nn = NearestNeighbors(metric=metric, algorithm="brute")

    def fit(self, ratings: pd.DataFrame) -> None:
        """Expects columns: userId, movieId, rating"""
        
        self.uidx = {u: i for i, u in enumerate(ratings.userId.unique())}
        self.midx = {m: i for i, m in enumerate(ratings.movieId.unique())}
        self.ridx = {i: m for m, i in self.midx.items()}

        rows = ratings.movieId.map(self.midx).values
        cols = ratings.userId.map(self.uidx).values
        data = ratings.rating.values
        csr_mat = csr_matrix((data, (rows, cols)),
                         shape=(len(self.midx), len(self.uidx)))
        self.nn.fit(csr_mat)
        self._mat = csr_mat                     

    def most_similar(self, movie_ids, top: int = 10):
        """Return {movie_id: [similar_movie_ids]}"""
        out = {}
        for mid in movie_ids:
            if mid not in self.midx:
                out[mid] = []
                continue
            i = self.midx[mid]
            _, idx = self.nn.kneighbors(self._mat[i], n_neighbors=top + 1)
            out[mid] = [self.ridx[j] for j in idx[0][1:]]   # drop itself
        return out