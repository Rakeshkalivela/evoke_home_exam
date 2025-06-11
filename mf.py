import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

class MFModel(nn.Module):
    def __init__(self, n_users: int, n_items: int, n_latent: int = 20):
        super().__init__()
        self.user_p = nn.Embedding(n_users, n_latent)
        self.item_q = nn.Embedding(n_items, n_latent)
        self.user_b = nn.Embedding(n_users, 1)
        self.item_b = nn.Embedding(n_items, 1)
        self.mu = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, u, i):
        return (
            self.mu +
            self.user_b(u).squeeze() +
            self.item_b(i).squeeze() +
            (self.user_p(u) * self.item_q(i)).sum(1)
        )

    def train_model(self, train_df, test_df, n_epochs=30, lr=0.05, lambda_=0.01):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        users = torch.tensor(train_df.u_idx.values, dtype=torch.long)
        items = torch.tensor(train_df.m_idx.values, dtype=torch.long)
        ratings = torch.tensor(train_df.rating.values, dtype=torch.float32)

        test_users = torch.tensor(test_df.u_idx.values, dtype=torch.long)
        test_items = torch.tensor(test_df.m_idx.values, dtype=torch.long)
        test_ratings = torch.tensor(test_df.rating.values, dtype=torch.float32)

        self.mu.data.fill_(ratings.mean())

        for epoch in range(n_epochs):
            self.train()
            optimizer.zero_grad()
            preds = self(users, items)

            reg = (
                self.user_b(users).pow(2).sum() +
                self.item_b(items).pow(2).sum() +
                self.user_p(users).pow(2).sum() +
                self.item_q(items).pow(2).sum()
            )
            loss = ((ratings - preds) ** 2).mean() + lambda_ * reg / len(users)
            loss.backward()
            optimizer.step()

            # Evaluate on test set
            self.eval()
            with torch.no_grad():
                test_preds = self(test_users, test_items)
                test_rmse = torch.sqrt(((test_preds - test_ratings) ** 2).mean()).item()

            print(f"Epoch {epoch+1}: Loss = {loss.item():.4f} | Test RMSE = {test_rmse:.4f}")

    def recommend_top_k(self, user_ids, k=10):
        self.eval()
        with torch.no_grad():
            all_items = torch.arange(self.item_q.num_embeddings)
            out = {}
            for uid in user_ids:
                u = torch.tensor([uid], dtype=torch.long)
                preds = self(u.repeat(len(all_items)), all_items)
                top = preds.argsort(descending=True)[:k]
                out[int(uid)] = [int(item) for item in top]
            return out

    def display_user_recommendations(self, user_id, ratings_df, movies_df, ridx, k=10):
        uidx_i = int(ratings_df[ratings_df.userId == user_id].u_idx.values[0])
        user_idx_tensor = torch.tensor([uidx_i], dtype=torch.long)

        scores = self.mu + self.user_b(user_idx_tensor).squeeze() + \
         self.item_b.weight.squeeze() + \
         (self.user_p(user_idx_tensor) * self.item_q.weight).sum(1)

        sorted_idxs = scores.argsort(descending=True)
        top_idxs = [i.item() for i in sorted_idxs if i.item() in ridx][:k]
        top_movie_ids = [ridx[i] for i in top_idxs]

        recommended = movies_df[movies_df.movieId.isin(top_movie_ids)][['movieId', 'title']]
        recommended = recommended.reset_index(drop=True)

        # Top 5 user-rated movies
        user_ratings = ratings_df[ratings_df.userId == user_id]
        top_rated = user_ratings.sort_values(by="rating", ascending=False).head(5)
        top_rated_movies = top_rated.merge(movies_df, on="movieId")[['movieId', 'title', 'rating']]
        top_rated_movies = top_rated_movies.reset_index(drop=True)

        print(f"\nTop 5 movies rated by user {user_id}:\n")
        print(top_rated_movies)

        print(f"\nTop {k} movie recommendations for user {user_id}:\n")
        print(recommended)
