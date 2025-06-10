import torch, torch.nn as nn

class MFModel(nn.Module):
    def __init__(self, n_users:int, n_items:int, n_latent:int=32):
        super().__init__()
        self.user_p = nn.Embedding(n_users, n_latent)
        self.item_q = nn.Embedding(n_items, n_latent)
        self.user_b = nn.Embedding(n_users, 1)
        self.item_b = nn.Embedding(n_items, 1)
        self.mu     = nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(self, u, i):
        pred = (self.mu +
                self.user_b(u).squeeze() + self.item_b(i).squeeze() +
                (self.user_p(u) * self.item_q(i)).sum(1))
        return pred


    def train(model, df, n_epochs=10, lr=5e-3, lambda_=1e-4):
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        users = torch.tensor(df.u_idx.values, dtype=torch.long)
        items = torch.tensor(df.m_idx.values, dtype=torch.long)
        ratings = torch.tensor(df.rating.values, dtype=torch.float32)
    
        model.mu.data.fill_(ratings.mean())  # initialize global mean
    
        for epoch in range(n_epochs):
            opt.zero_grad()
            preds = model(users, items)
    
            # Regularization manually
            reg = (
                model.user_b(users).pow(2).sum() +
                model.item_b(items).pow(2).sum() +
                model.user_p(users).pow(2).sum() +
                model.item_q(items).pow(2).sum()
            )
            loss = ((ratings - preds) ** 2).mean() + lambda_ * reg / len(users)
    
            loss.backward()
            opt.step()
            if ((epoch+1)%5)==0:
                print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")


    def recommend_top_k(model, user_ids, k=10):
        with torch.no_grad():
            all_items = torch.arange(model.item_q.num_embeddings)
            out = {}
            for uid in user_ids:
                u = torch.tensor([uid], dtype=torch.long)
                preds = model(u.repeat(len(all_items)), all_items)
                top = preds.argsort(descending=True)[:k]
                out[int(uid)] = [int(item) for item in top]
            return out
