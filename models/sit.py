"""
Signature-based Transformer (SIT) model.

Implements dual self-attention over time (horizon) and assets with a dynamic
cross-asset attention bias derived from cross-signatures.

Inputs:
    x_sigs: (B, H, D, 2)          # path signatures per asset
    cross_sigs: (B, H, D, D, 1)   # cross-signatures
    date_feats: (B, H, 3)         # time features
    future_return_unscaled: (B, H, D)  # provided for loss only (unused here)

Outputs:
    weights: (B, H, D)    # long-only portfolio weights (softmax)
    mu_hat: (B, H, D)     # unnormalized return scores from return_projection
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _causal_mask(horizon: int, device: torch.device) -> torch.Tensor:
    """Upper triangular mask with -inf above diagonal for causal temporal attention."""
    mask = torch.full((horizon, horizon), float("-inf"), device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask


class FeedForward(nn.Module):
    """Simple FFN block used after attention."""

    def __init__(self, d_model: int, ff_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalBlock(nn.Module):
    """Temporal self-attention over horizon steps for each asset independently."""

    def __init__(self, d_model: int, n_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.ff = FeedForward(d_model, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        # x: (B*D, H, d_model)
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x


class AssetBlock(nn.Module):
    """Asset self-attention with dynamic bias from cross-signatures."""

    def __init__(self, d_model: int, n_heads: int, ff_dim: int, dropout: float, bias_embed_dim: int, hidden_c: int):
        super().__init__()
        self.n_heads = n_heads
        self.bias_embed_dim = bias_embed_dim
        self.gamma_param = nn.Parameter(torch.tensor(1.0))

        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.ff = FeedForward(d_model, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # MLP to embed cross-signatures into attention bias space
        self.beta_mlp = nn.Sequential(
            nn.Linear(1, hidden_c),
            nn.ReLU(),
            nn.Linear(hidden_c, n_heads * bias_embed_dim),
        )
        # MLP to embed queries for bias interaction
        self.query_token_mlp = nn.Sequential(
            nn.Linear(d_model, hidden_c),
            nn.ReLU(),
            nn.Linear(hidden_c, n_heads * bias_embed_dim),
        )

    def forward(self, x: torch.Tensor, cross_sigs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, H, D, d_model) encoded tokens
            cross_sigs: (B, H, D, D, 1)
        """
        B, H, D, _ = x.shape

        # Compute dynamic bias
        beta_embed = self.beta_mlp(cross_sigs)  # (B, H, D, D, n_heads*bias_embed_dim)
        beta_embed = beta_embed.view(B, H, D, D, self.n_heads, self.bias_embed_dim)

        query_embed = self.query_token_mlp(x)   # (B, H, D, n_heads*bias_embed_dim)
        query_embed = query_embed.view(B, H, D, self.n_heads, self.bias_embed_dim)

        # dyn_bias: (B, H, n_heads, D, D)
        dyn_bias = torch.einsum("bhink,bhdjnk->bhnij", query_embed, beta_embed)
        gamma = F.softplus(self.gamma_param)
        dyn_bias = dyn_bias * gamma

        # Run attention per horizon step with bias
        out = []
        for h in range(H):
            tokens = x[:, h]  # (B, D, d_model)
            bias = dyn_bias[:, h]  # (B, n_heads, D, D)
            # Average bias across batch and heads for 2D mask
            bias_2d = bias.mean(dim=(0, 1))  # (D, D)
            attn_out, _ = self.attn(tokens, tokens, tokens, attn_mask=bias_2d)
            tokens = self.norm1(tokens + attn_out)
            ff_out = self.ff(tokens)
            tokens = self.norm2(tokens + ff_out)
            out.append(tokens)

        return torch.stack(out, dim=1)  # (B, H, D, d_model)


class SIT(nn.Module):
    """Signature-based Portfolio Transformer."""

    def __init__(
        self,
        num_assets: int,
        d_model: int,
        n_heads: int,
        num_layers: int,
        ff_dim: int,
        hidden_c: int,
        bias_embed_dim: int = 16,
        dropout: float = 0.1,
        max_position: float = 0.1,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.num_assets = num_assets
        self.d_model = d_model
        self.n_heads = n_heads
        self.num_layers = num_layers
        self.ff_dim = ff_dim
        self.hidden_c = hidden_c
        self.bias_embed_dim = bias_embed_dim
        self.max_position = max_position
        self.temperature = temperature

        # Embeddings
        self.sig_embed = nn.Linear(2, d_model)
        self.date_proj = nn.Linear(3, d_model)
        self.asset_embed = nn.Embedding(num_assets, d_model)
        self.input_proj = nn.Linear(3 * d_model, d_model)

        # Transformer layers (paired temporal + asset blocks)
        self.temporal_layers = nn.ModuleList(
            [TemporalBlock(d_model, n_heads, ff_dim, dropout) for _ in range(num_layers)]
        )
        self.asset_layers = nn.ModuleList(
            [AssetBlock(d_model, n_heads, ff_dim, dropout, bias_embed_dim, hidden_c) for _ in range(num_layers)]
        )

        # Output heads
        self.projection = nn.Linear(d_model, 1)
        self.return_projection = nn.Linear(d_model, 1)

    def _build_inputs(self, x_sigs: torch.Tensor, date_feats: torch.Tensor) -> torch.Tensor:
        """
        Build input token embeddings by combining signature, date, and asset identity.
        Args:
            x_sigs: (B, H, D, 2)
            date_feats: (B, H, 3)
        Returns:
            tokens: (B, H, D, d_model)
        """
        B, H, D, _ = x_sigs.shape
        sig_emb = self.sig_embed(x_sigs)                    # (B, H, D, d_model)
        date_emb = self.date_proj(date_feats)               # (B, H, d_model)
        date_emb = date_emb.unsqueeze(2).expand(-1, -1, D, -1)  # (B, H, D, d_model)
        asset_ids = torch.arange(D, device=x_sigs.device)
        asset_emb = self.asset_embed(asset_ids)             # (D, d_model)
        asset_emb = asset_emb.unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1)  # (B, H, D, d_model)

        concat = torch.cat([sig_emb, date_emb, asset_emb], dim=-1)
        tokens = self.input_proj(concat)
        return tokens

    def forward(
        self,
        x_sigs: torch.Tensor,
        cross_sigs: torch.Tensor,
        date_feats: torch.Tensor,
        future_return_unscaled: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            weights: (B, H, D) long-only weights
            mu_hat: (B, H, D) raw scores before softmax
        """
        B, H, D, _ = x_sigs.shape
        tokens = self._build_inputs(x_sigs, date_feats)  # (B, H, D, d_model)

        # Temporal attention per asset (flatten assets into batch dimension)
        causal_mask = _causal_mask(H, x_sigs.device)
        tokens = tokens.view(B * D, H, self.d_model)
        for t_layer, a_layer in zip(self.temporal_layers, self.asset_layers):
            tokens = t_layer(tokens, attn_mask=causal_mask)
            tokens = tokens.view(B, D, H, self.d_model).transpose(1, 2)  # (B, H, D, d_model)
            tokens = a_layer(tokens, cross_sigs)
            tokens = tokens.view(B, H * D, self.d_model)                 # flatten for next temporal layer
            tokens = tokens.view(B * D, H, self.d_model)

        tokens = tokens.view(B, H, D, self.d_model)

        logits_delta = torch.tanh(self.projection(tokens)).squeeze(-1) * self.max_position  # (B, H, D)
        mu_hat = self.return_projection(tokens).squeeze(-1)                                 # (B, H, D)
        weights = torch.softmax(mu_hat / self.temperature, dim=-1)

        # Return weights and mu_hat; delta/logits_delta unused for now
        return weights, mu_hat
