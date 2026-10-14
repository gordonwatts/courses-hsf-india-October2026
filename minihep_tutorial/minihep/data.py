"""Convert the workshop's reduced ATLAS Open Data into four-vector tokens.

Its PT columns are GeV, while lep_E and
jet_E are MeV. Missing transverse momentum is a transverse proxy, not a
measured neutrino-system four-vector.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import uproot


PAD, ELECTRON_MINUS, ELECTRON_PLUS, MUON_MINUS, MUON_PLUS, JET, MET = range(7)
TYPE_NAMES = ("pad", "e-", "e+", "mu-", "mu+", "jet", "MET")
FEATURE_NAMES = ("px", "py", "pz", "E")

WORKSHOP_BRANCHES = (
    "eventNumber", "channelNumber", "label", "lep_n", "jet_n",
    "met_et", "met_phi",
    *(f"lep_{field}_{i}" for i in range(2)
      for field in ("pt", "eta", "phi", "E", "charge", "type")),
    *(f"jet_{field}_{i}" for i in range(2)
      for field in ("pt", "eta", "phi", "E")),
)


@dataclass(frozen=True)
class EventTokens:
    vectors: np.ndarray  # (events, 5, 4), GeV
    types: np.ndarray    # (events, 5), PAD=0
    valid: np.ndarray    # (events, 5), True for physical/object proxy tokens
    event_ids: np.ndarray  # (events, 2): channel and event number
    labels: np.ndarray | None  # diagnostics only, never passed to the model
    source: str

    def __post_init__(self) -> None:
        n = len(self.vectors)
        if self.vectors.shape != (n, 5, 4) or self.types.shape != (n, 5) or self.valid.shape != (n, 5):
            raise ValueError("Expected five tokens and four vector components per event")
        if self.event_ids.shape != (n, 2):
            raise ValueError("Expected two event identifier columns")
        if not np.isfinite(self.vectors).all():
            raise ValueError("Non-finite four-vector component")
        if np.any(self.types[~self.valid] != PAD) or np.any(self.vectors[~self.valid] != 0):
            raise ValueError("Padded tokens must have type PAD and zero vector")


def fourvector(pt: np.ndarray, eta: np.ndarray, phi: np.ndarray, energy: np.ndarray) -> np.ndarray:
    """Return (px, py, pz, E), all in GeV."""
    return np.stack((pt * np.cos(phi), pt * np.sin(phi), pt * np.sinh(eta), energy), axis=-1)


def _lepton_type(pdg_abs: np.ndarray, charge: np.ndarray) -> np.ndarray:
    if np.any(~np.isin(pdg_abs, (11, 13))) or np.any(~np.isin(charge, (-1, 1))):
        raise ValueError("Expected lepton type 11/13 and charge -1/+1")
    return np.where(pdg_abs == 11, np.where(charge < 0, ELECTRON_MINUS, ELECTRON_PLUS),
                    np.where(charge < 0, MUON_MINUS, MUON_PLUS)).astype(np.int64)


def load_workshop(path: str | Path, max_events: int | None = 6000, seed: int = 42) -> EventTokens:
    """Read two-lepton events uniformly from the workshop ROOT file.

    `max_events=None` keeps every selected event. No label or MC weight enters
    the four-vectors. The 2-jet cap reflects the *stored columns*, not jet_n.
    """
    path = Path(path)
    with uproot.open(path, handler=uproot.source.file.MemmapSource) as root_file:
        if "tree_event" not in root_file:
            raise ValueError("Expected tree_event in the workshop ROOT file")
        tree = root_file["tree_event"]
        missing = set(WORKSHOP_BRANCHES) - set(tree.keys())
        if missing:
            raise ValueError(f"Missing workshop branches: {sorted(missing)}")
        data = tree.arrays(WORKSHOP_BRANCHES, library="np")

    selected = np.flatnonzero(data["lep_n"] == 2)
    if max_events is not None:
        if max_events <= 0:
            raise ValueError("max_events must be positive or None")
        if len(selected) > max_events:
            selected = np.random.default_rng(seed).choice(selected, size=max_events, replace=False)
    n = len(selected)
    if n == 0:
        raise ValueError("No two-lepton events found")
    vectors = np.zeros((n, 5, 4), dtype=np.float32)
    types = np.zeros((n, 5), dtype=np.int64)
    valid = np.zeros((n, 5), dtype=bool)

    for i in range(2):
        pt = data[f"lep_pt_{i}"][selected]
        energy = data[f"lep_E_{i}"][selected] / 1000.0  # MeV to GeV
        if np.any(pt <= 0) or np.any(energy <= 0):
            raise ValueError("Invalid stored lepton")
        vectors[:, i] = fourvector(pt, data[f"lep_eta_{i}"][selected],
                                   data[f"lep_phi_{i}"][selected], energy)
        types[:, i] = _lepton_type(data[f"lep_type_{i}"][selected],
                                   data[f"lep_charge_{i}"][selected])
        valid[:, i] = True

    for i in range(2):
        present = data["jet_n"][selected] > i
        if np.any(present):
            pt = data[f"jet_pt_{i}"][selected][present]
            energy = data[f"jet_E_{i}"][selected][present] / 1000.0
            if np.any(pt <= 0) or np.any(energy <= 0):
                raise ValueError("Invalid stored jet")
            vectors[present, 2 + i] = fourvector(pt, data[f"jet_eta_{i}"][selected][present],
                                                 data[f"jet_phi_{i}"][selected][present], energy)
            types[present, 2 + i] = JET
            valid[present, 2 + i] = True

    met = data["met_et"][selected]
    phi = data["met_phi"][selected]
    if np.any(met < 0):
        raise ValueError("Invalid missing transverse momentum")
    vectors[:, 4, 0] = met * np.cos(phi)
    vectors[:, 4, 1] = met * np.sin(phi)
    vectors[:, 4, 3] = met
    types[:, 4] = MET
    valid[:, 4] = True

    event_ids = np.stack((data["channelNumber"][selected], data["eventNumber"][selected]), axis=-1)
    labels = data["label"][selected].astype(np.int8)
    return EventTokens(vectors, types, valid, event_ids, labels, "real workshop ATLAS Open Data")


def split_events(n_events: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Disjoint random 70/15/15 event split, before fitting any scaler."""
    if n_events < 20:
        raise ValueError("Need at least 20 events for train/validation/test splits")
    order = np.random.default_rng(seed).permutation(n_events)
    train_end = int(0.70 * n_events)
    val_end = int(0.85 * n_events)
    return order[:train_end], order[train_end:val_end], order[val_end:]


class VectorScaler:
    """asinh compression followed by scaling fitted on train tokens only."""

    def __init__(self, scale_gev: float = 50.0):
        self.scale_gev = scale_gev
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, vectors: np.ndarray, valid: np.ndarray, train_indices: np.ndarray) -> "VectorScaler":
        compressed = np.arcsinh(vectors[train_indices] / self.scale_gev)
        values = compressed[valid[train_indices]]
        self.mean = values.mean(axis=0).astype(np.float32)
        self.std = np.maximum(values.std(axis=0), 1e-6).astype(np.float32)
        return self

    def transform(self, vectors: np.ndarray, valid: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Fit VectorScaler on training events first")
        result = ((np.arcsinh(vectors / self.scale_gev) - self.mean) / self.std).astype(np.float32)
        result[~valid] = 0.0
        return result
