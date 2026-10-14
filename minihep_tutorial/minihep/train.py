"""Reproducible, CPU-sized pretraining loop and held-out evaluation."""

from dataclasses import dataclass

import numpy as np
import torch

from .model import MiniHEP, masked_mse, sample_hidden


@dataclass
class History:
    train: list[float]
    validation: list[float]


def _evaluate(model: MiniHEP, vectors: torch.Tensor, types: torch.Tensor,
              valid: torch.Tensor, indices: np.ndarray, seed: int,
              batch_size: int, mask_probability: float) -> float:
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    total, count = 0.0, 0
    with torch.no_grad():
        for batch_ids in np.array_split(indices, max(1, (len(indices) + batch_size - 1) // batch_size)):
            x, t, v = vectors[batch_ids], types[batch_ids], valid[batch_ids]
            hidden = sample_hidden(v, mask_probability, generator)
            error = (model(x, t, v, hidden) - x).square()[hidden]
            total += error.sum().item()
            count += error.numel()
    return total / count


def pretrain(model: MiniHEP, vectors: np.ndarray, types: np.ndarray, valid: np.ndarray,
             train_indices: np.ndarray, val_indices: np.ndarray, epochs: int = 4,
             batch_size: int = 64, mask_probability: float = 0.25,
             learning_rate: float = 1e-3, seed: int = 42) -> History:
    """Train only on train indices; reuse a fixed validation mask each epoch."""
    torch.manual_seed(seed)
    x = torch.as_tensor(vectors, dtype=torch.float32)
    t = torch.as_tensor(types, dtype=torch.long)
    v = torch.as_tensor(valid, dtype=torch.bool)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed + 1)
    shuffle = np.random.default_rng(seed + 2)
    history = History([], [])
    for epoch in range(epochs):
        model.train()
        order = shuffle.permutation(train_indices)
        total, count = 0.0, 0
        for batch_ids in np.array_split(order, max(1, (len(order) + batch_size - 1) // batch_size)):
            bx, bt, bv = x[batch_ids], t[batch_ids], v[batch_ids]
            hidden = sample_hidden(bv, mask_probability, generator)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(bx, bt, bv, hidden)
            loss = masked_mse(prediction, bx, hidden)
            loss.backward()
            optimizer.step()
            elements = int(hidden.sum()) * 4
            total += loss.item() * elements
            count += elements
        history.train.append(total / count)
        history.validation.append(_evaluate(model, x, t, v, val_indices, seed + 1000,
                                            batch_size, mask_probability))
        print(f"epoch {epoch + 1:2d} | train {history.train[-1]:.4f} | validation {history.validation[-1]:.4f}")
    return history


def heldout_loss(model: MiniHEP, vectors: np.ndarray, types: np.ndarray, valid: np.ndarray,
                 test_indices: np.ndarray, batch_size: int = 64,
                 mask_probability: float = 0.25) -> float:
    """Call once after model and hyperparameters are settled."""
    return _evaluate(model, torch.as_tensor(vectors, dtype=torch.float32),
                     torch.as_tensor(types, dtype=torch.long),
                     torch.as_tensor(valid, dtype=torch.bool),
                     test_indices, seed=2042, batch_size=batch_size,
                     mask_probability=mask_probability)
