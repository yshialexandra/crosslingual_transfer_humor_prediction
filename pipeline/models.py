from __future__ import annotations

import torch
import torch.nn as nn


def masked_mean_pool(x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    if mask is None:
        return x.mean(dim=1)
    mask = mask.unsqueeze(-1).to(x.dtype)
    return (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)


class MultimodalSegmentClassifier(nn.Module):
    def __init__(
        self,
        text_dim,
        audio_dim,
        hidden_dim=512,
        num_heads=8,
        dropout=0.25,
        mode="concat",
    ):
        super().__init__()
        if mode not in {"text", "concat", "cross_attention"}:
            raise ValueError(f"Unknown mode: {mode}")
        self.mode = mode
        self.text_proj = nn.Sequential(nn.Linear(text_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.audio_proj = nn.Sequential(nn.Linear(audio_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())

        if mode == "cross_attention":
            self.text_to_audio = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.audio_to_text = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.text_attn_norm = nn.LayerNorm(hidden_dim)
            self.audio_attn_norm = nn.LayerNorm(hidden_dim)

        classifier_dim = hidden_dim if mode == "text" else hidden_dim * 2
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_dim),
            nn.Dropout(dropout),
            nn.Linear(classifier_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, text_hidden, audio_hidden, text_mask, audio_mask):
        text_h = self.text_proj(text_hidden)
        audio_h = self.audio_proj(audio_hidden)

        if self.mode == "text":
            fused = masked_mean_pool(text_h, text_mask)
            return self.classifier(fused)

        if self.mode == "concat":
            text_vec = masked_mean_pool(text_h, text_mask)
            audio_vec = masked_mean_pool(audio_h, audio_mask)
        else:
            text_key_padding_mask = ~text_mask.bool()
            audio_key_padding_mask = ~audio_mask.bool()
            text_ctx, _ = self.text_to_audio(
                query=text_h,
                key=audio_h,
                value=audio_h,
                key_padding_mask=audio_key_padding_mask,
                need_weights=False,
            )
            audio_ctx, _ = self.audio_to_text(
                query=audio_h,
                key=text_h,
                value=text_h,
                key_padding_mask=text_key_padding_mask,
                need_weights=False,
            )
            text_ctx = self.text_attn_norm(text_h + text_ctx)
            audio_ctx = self.audio_attn_norm(audio_h + audio_ctx)
            text_vec = masked_mean_pool(text_ctx, text_mask)
            audio_vec = masked_mean_pool(audio_ctx, audio_mask)

        fused = torch.cat([text_vec, audio_vec], dim=-1)
        return self.classifier(fused)
