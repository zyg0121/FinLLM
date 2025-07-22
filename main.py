# finllm.py --- FinLLM reference implementation (PyTorch 2.0)
# --------------------------------------------------------------
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gelu(x: torch.Tensor) -> torch.Tensor:
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class BiLSTMBranch(nn.Module):
    def __init__(self, feat_dim: int, hidden_dim: int = 64, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = self.dropout(out[:, -1, :])
        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerBranch(nn.Module):
    def __init__(self, d_model: int = 128, nhead: int = 4, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=256,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pos_encoding = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pos_encoding(x)
        h = self.transformer(x)
        h_cls = self.dropout(h[:, 0, :])
        return h_cls


class CrossModalFusion(nn.Module):
    def __init__(self, dim_tech: int, dim_sent: int, hidden: int = 128):
        super().__init__()
        self.q_tech = nn.Linear(dim_tech, hidden)
        self.kv_sent = nn.Linear(dim_sent, hidden * 2)
        self.q_sent = nn.Linear(dim_sent, hidden)
        self.kv_tech = nn.Linear(dim_tech, hidden * 2)
        self.scale = hidden ** -0.5
        self.act = _gelu
        self.out_dim = hidden * 2

    def _attn(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        weights = torch.softmax(scores, dim=-1)
        return torch.matmul(weights, v).squeeze(1)

    def forward(self, h_tech: torch.Tensor, h_sent: torch.Tensor) -> torch.Tensor:
        q1 = self.q_tech(h_tech).unsqueeze(1)
        k1, v1 = self.kv_sent(h_sent).chunk(2, dim=-1)
        k1 = k1.unsqueeze(1)
        v1 = v1.unsqueeze(1)
        z1 = self._attn(q1, k1, v1)
        q2 = self.q_sent(h_sent).unsqueeze(1)
        k2, v2 = self.kv_tech(h_tech).chunk(2, dim=-1)
        k2 = k2.unsqueeze(1)
        v2 = v2.unsqueeze(1)
        z2 = self._attn(q2, k2, v2)
        return self.act(torch.cat([z1, z2], dim=-1))


class RiskAwareHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(_gelu(self.fc1(x))).squeeze(-1)


class FinLLM(nn.Module):
    def __init__(self, tech_feat_dim: int, sent_embed_dim: int = 128, lstm_hidden: int = 64):
        super().__init__()
        self.branch_tech = BiLSTMBranch(tech_feat_dim, lstm_hidden)
        self.branch_sent = TransformerBranch(d_model=sent_embed_dim)
        self.fusion = CrossModalFusion(dim_tech=lstm_hidden * 2, dim_sent=sent_embed_dim)
        self.head = RiskAwareHead(in_dim=self.fusion.out_dim)

    def forward(self, x_tech: torch.Tensor, x_sent: torch.Tensor) -> torch.Tensor:
        h_tech = self.branch_tech(x_tech)
        h_sent = self.branch_sent(x_sent)
        z = self.fusion(h_tech, h_sent)
        return self.head(z)


def risk_aware_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float = 0.95, lam: float = 0.1):
    mse = F.mse_loss(pred, target)
    diff = target - pred
    var = torch.quantile(diff, alpha)
    mask = diff >= var
    es = diff[mask].mean() if mask.any() else torch.tensor(0.0, device=pred.device, dtype=pred.dtype)
    return mse + lam * es


def train_epoch(model: nn.Module, loader, optim, device="cuda"):
    model.train()
    total_loss = 0.0
    for x_tech, x_sent, y in loader:
        x_tech = x_tech.to(device)
        x_sent = x_sent.to(device)
        y = y.to(device).squeeze(-1)
        optim.zero_grad()
        preds = model(x_tech, x_sent)
        loss = risk_aware_loss(preds, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optim.step()
        total_loss += loss.item() * y.size(0)
    return total_loss / len(loader.dataset)


if __name__ == "__main__":
    class DummyDS(torch.utils.data.Dataset):
        def __len__(self):
            return 1024

        def __getitem__(self, idx):
            x_tech = torch.randn(30, 16)
            x_sent = torch.randn(14, 128)
            y = torch.randn(())
            return x_tech, x_sent, y

    ds = DummyDS()
    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = FinLLM(tech_feat_dim=16).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(5):
        loss = train_epoch(model, loader, optim, device)
        print(f"epoch {epoch:02d} | train loss = {loss:.4f}")

    torch.save(model.state_dict(), "finllm_checkpoint.pth")
