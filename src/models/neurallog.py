import torch
import torch.nn as nn
import numpy as np


class PositionEmbedding(nn.Module):
    def __init__(self, max_len, embed_dim):
        super(PositionEmbedding, self).__init__()
        self.pos_encoding = self._build_positional_encoding(max_len, embed_dim)

    def _build_positional_encoding(self, position, d_model):
        def get_angles(pos, i, d_model):
            return pos / np.power(10000, (2 * (i // 2)) / np.float32(d_model))

        angle_rads = get_angles(
            np.arange(position)[:, np.newaxis],
            np.arange(d_model)[np.newaxis, :],
            d_model
        )
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])  # sin on even
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])  # cos on odd

        pos_encoding = angle_rads[np.newaxis, ...]  # (1, position, d_model)
        return torch.tensor(pos_encoding, dtype=torch.float32)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pos_encoding[:, :seq_len, :].to(x.device)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.att = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [seq_len, batch_size, embed_dim]
        attn_output, _ = self.att(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x


class NeuralLog(nn.Module):
    def __init__(self, embed_dim=256, ff_dim=512, max_len=128, num_heads=4, dropout=0.1, num_classes=2):
        super(NeuralLog, self).__init__()
        self.input_projection = nn.Linear(768, embed_dim)
        self.pos_embedding = PositionEmbedding(max_len, embed_dim)
        self.transformer = TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.input_projection(x)                          # (batch, seq_len, embed_dim)
        x = self.pos_embedding(x)                             # (batch, seq_len, embed_dim)
        x = x.permute(1, 0, 2)                                # (seq_len, batch, embed_dim)
        x = self.transformer(x)                               # (seq_len, batch, embed_dim)
        x = x.permute(1, 0, 2).mean(dim=1)                    # (batch, embed_dim) - global average pooling
        x = self.classifier(x)                                # (batch, num_classes)
        return x
