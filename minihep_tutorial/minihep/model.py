"""A deliberately small masked four-vector transformer."""

import torch
from torch import nn


class MiniHEP(nn.Module):
    def __init__(self, width: int = 32, heads: int = 4, layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.vector_projection = nn.Linear(4, width)
        self.type_embedding = nn.Embedding(7, width, padding_idx=0)
        self.mask_embedding = nn.Parameter(torch.zeros(width))
        block = nn.TransformerEncoderLayer(
            d_model=width, nhead=heads, dim_feedforward=2 * width,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=layers, enable_nested_tensor=False)
        self.readout = nn.Linear(width, 4)
        # TransformerEncoder clones the initial block; give each layer its own
        # matrix initialization so their attention weights do not start equal.
        for layer in self.encoder.layers:
            for parameter in layer.parameters():
                if parameter.ndim > 1:
                    nn.init.xavier_uniform_(parameter)

    def _states(self, vectors: torch.Tensor, types: torch.Tensor,
                valid: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        if vectors.ndim != 3 or vectors.shape[-1] != 4:
            raise ValueError("vectors must have shape (batch, tokens, 4)")
        if types.shape != vectors.shape[:2] or valid.shape != types.shape or hidden.shape != types.shape:
            raise ValueError("types, valid and hidden must have shape (batch, tokens)")
        if torch.any(hidden & ~valid):
            raise ValueError("Cannot hide padding")
        visible_vectors = vectors.masked_fill((hidden | ~valid).unsqueeze(-1), 0.0)
        state = self.vector_projection(visible_vectors) + self.type_embedding(types)
        state = state + hidden.unsqueeze(-1) * self.mask_embedding
        return self.encoder(state, src_key_padding_mask=~valid)

    def forward(self, vectors: torch.Tensor, types: torch.Tensor,
                valid: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        return self.readout(self._states(vectors, types, valid, hidden))

    def encode_events(self, vectors: torch.Tensor, types: torch.Tensor,
                      valid: torch.Tensor) -> torch.Tensor:
        """Mean-pooled event representations with all four-vectors visible."""
        hidden = torch.zeros_like(valid)
        states = self._states(vectors, types, valid, hidden)
        return (states * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(dim=1, keepdim=True)


def sample_hidden(valid: torch.Tensor, probability: float, generator: torch.Generator) -> torch.Tensor:
    """Hide whole token four-vectors; guarantee one target per event."""
    if not 0 < probability <= 1:
        raise ValueError("probability must be in (0, 1]")
    hidden = (torch.rand(valid.shape, generator=generator, device=valid.device) < probability) & valid
    empty = ~hidden.any(dim=1)
    if empty.any():
        rows = torch.nonzero(empty, as_tuple=True)[0]
        # Random valid token in each otherwise-empty row.
        scores = torch.rand(valid[rows].shape, generator=generator, device=valid.device)
        scores = scores.masked_fill(~valid[rows], -1.0)
        chosen = scores.argmax(dim=1)
        hidden[rows, chosen] = True
    return hidden


def masked_mse(prediction: torch.Tensor, target: torch.Tensor,
               hidden: torch.Tensor) -> torch.Tensor:
    squared = (prediction - target).square()
    return squared[hidden].mean()
