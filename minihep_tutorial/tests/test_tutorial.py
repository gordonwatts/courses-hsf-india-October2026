from pathlib import Path

import numpy as np
import pytest
import torch

from minihep.data import JET, MET, VectorScaler, load_workshop, split_events
from minihep.model import MiniHEP, sample_hidden
from minihep.train import heldout_loss, pretrain


ROOT_FILE = Path(__file__).resolve().parents[2] / "dataWW_d1.root"


@pytest.fixture(scope="module")
def events():
    return load_workshop(ROOT_FILE, max_events=128, seed=42)


def test_real_workshop_units_and_padding(events):
    assert events.source == "real workshop ATLAS Open Data"
    assert events.vectors.shape == (128, 5, 4)
    assert np.all(events.valid[:, (0, 1, 4)])
    assert np.all(events.types[:, 4] == MET)
    assert np.all(events.types[:, 2:4][events.valid[:, 2:4]] == JET)
    assert np.any(~events.valid[:, 2]) and np.any(events.valid[:, 3])
    assert np.all(events.vectors[~events.valid] == 0)
    assert len(np.unique(events.event_ids, axis=0)) == len(events.event_ids)
    assert np.allclose(events.vectors[:, 4, 3],
                       np.linalg.norm(events.vectors[:, 4, :2], axis=-1))
    assert np.allclose(events.vectors[:, 0, 3],
                       np.linalg.norm(events.vectors[:, 0, :3], axis=-1), rtol=1e-3)


def test_split_and_train_only_scaler(events):
    train, val, test = split_events(len(events.vectors))
    assert len(set(train) & set(val)) == len(set(train) & set(test)) == len(set(val) & set(test)) == 0
    assert len(np.unique(np.concatenate((train, val, test)))) == len(events.vectors)
    scaler = VectorScaler().fit(events.vectors, events.valid, train)
    changed = events.vectors.copy()
    changed[val, 0, 0] = 1e8
    altered_fit = VectorScaler().fit(changed, events.valid, train)
    assert np.allclose(scaler.mean, altered_fit.mean)
    assert np.all(scaler.transform(events.vectors, events.valid)[~events.valid] == 0)


def test_model_mask_padding_and_cpu_training(events):
    torch.set_num_threads(1)
    train, val, test = split_events(len(events.vectors))
    x = VectorScaler().fit(events.vectors, events.valid, train).transform(events.vectors, events.valid)
    valid = torch.from_numpy(events.valid[:8])
    hidden = sample_hidden(valid, 0.25, torch.Generator().manual_seed(1))
    assert torch.all(hidden.any(dim=1)) and not torch.any(hidden & ~valid)
    torch.manual_seed(1)
    model = MiniHEP(width=16, heads=2, layers=1, dropout=0.0)
    model.eval()
    with torch.no_grad():
        baseline = model(torch.from_numpy(x[:8]), torch.from_numpy(events.types[:8]), valid, hidden)
        altered = torch.from_numpy(x[:8].copy())
        altered[~valid] = 1e6
        changed = model(altered, torch.from_numpy(events.types[:8]), valid, hidden)
    assert torch.allclose(baseline[valid], changed[valid], atol=1e-5)
    history = pretrain(model, x, events.types, events.valid, train, val,
                       epochs=1, batch_size=32, seed=1)
    assert np.isfinite(history.train + history.validation).all()
    assert np.isfinite(heldout_loss(model, x, events.types, events.valid, test))
