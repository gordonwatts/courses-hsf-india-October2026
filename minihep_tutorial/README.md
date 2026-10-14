# MiniHEP: masked four-vectors on the Delhi workshop data

This is a small CPU-first tutorial in **self-supervised pretraining**. Its input is the course repository's [`dataWW_d1.root`](../dataWW_d1.root), a **real reduced ATLAS Open Data simulation sample**, with H→WW signal and background labels used by the existing [classification exercise](../01-tutorial-exercise-NN-JAX.ipynb). MiniHEP does **not** train on those labels. It hides particle four-vectors and predicts them using the visible event objects. This is a teaching model, not a pretrained general-purpose collider foundation model or a physics result.

## Project map

```text
minihep_tutorial/
  notebooks/01_masked_four_vectors.ipynb  # first exercise and diagnostic plot
  minihep/data.py                         # reduced workshop ROOT schema and scaling
  minihep/model.py                        # type embeddings + masked transformer
  minihep/train.py                        # train/validation/test masked loss
  tests/test_tutorial.py                  # loader and training checks
  SCHEMA.md                               # units and branch contract
  requirements.txt
```

## Run on the course data

From the course repository root, with Python 3.10 or newer:

```bash
python -m venv .venv
# Activate .venv for your shell, then:
python -m pip install -r minihep_tutorial/requirements.txt
jupyter lab minihep_tutorial/notebooks/01_masked_four_vectors.ipynb
```

Run every cell. The notebook finds `dataWW_d1.root` in the course checkout, samples 6,000 two-lepton events uniformly with a fixed seed, makes disjoint 70/15/15 event splits, fits scaling on training events only, trains four epochs with a 17,636-parameter transformer, and plots train/validation masked four-vector error. It reports test error once at the end. The model uses one CPU thread by default for the workshop's one-core Binder environment. `MAX_EVENTS`, epochs, width, and dropout are visible in the notebook.

The checked real-data run produced this [first diagnostic plot](figures/real_workshop_masked_loss.png). Its loss values are a smoke test for this seed, sample, and objective, not a physics benchmark.

Run tests from the tutorial directory:

```bash
cd minihep_tutorial
python -m pytest -q
```

## What becomes a token?

One ROOT entry is one collision event, not one token. The loader keeps entries with `lep_n == 2` and turns each event into a sequence of five positions: first lepton, second lepton, first jet, second jet, and missing transverse momentum (MET). Each present position becomes a token with a discrete type (e−, e+, μ−, μ+, jet, or MET) and four continuous numbers `(px, py, pz, E)` in GeV. For a lepton or jet, the loader computes `px = pT cos φ`, `py = pT sin φ`, and `pz = pT sinh η` from the ROOT columns, and converts the stored energy from MeV to GeV.

For example, an event with two leptons and one jet has tokens `[e+, μ−, jet, PAD, MET]`. The unused second-jet position contains zeros and is excluded by the padding mask; its `-7` ROOT filler values are never used. The file stores only two jet columns even when `jet_n` is larger. MET is represented by `(MET cos φ, MET sin φ, 0, MET)`, a transverse **proxy**, not a measured neutrino-pair four-vector. See [SCHEMA.md](SCHEMA.md) for the branch and unit details.

## The MiniHEP model

The implementation is in [`minihep/model.py`](minihep/model.py). A linear layer maps each standardized four-vector from 4 numbers to a width-32 representation. The model adds a learned object-type embedding (including a padding type) and, for a hidden token, a learned mask embedding. Two transformer encoder layers then let each token attend to the other valid tokens; each layer has four attention heads, a width-64 feed-forward layer, GELU activations, and 0.1 dropout. A padding mask keeps absent jets out of attention.

A final linear layer maps each token's width-32 state back to four predicted vector components. There is no classification head in this notebook. With all vectors visible, `encode_events` averages the valid token states to make one width-32 event embedding for a later task. The default model has 17,636 trainable parameters. It uses no position embedding, so swapping two same-type objects swaps their token outputs while leaving the pooled event embedding unchanged.

## How are the tokens trained?

Events are split into disjoint training, validation, and test sets first. Each four-vector component is compressed with `asinh(component / 50 GeV)` and standardized using only training events. On each training pass, the code randomly hides all four numbers of about 25% of the real tokens, always hiding at least one token per event. It keeps the token's type and adds a learned mask embedding, so the transformer knows *what kind* of object is missing but must infer its momentum from the other visible objects. The loss is mean squared error on the hidden four-vectors only; padded jets contribute nothing. Dropout is active during training, while validation uses a fixed masking sequence. `label`, `mcWeight`, and event identifiers are never model inputs.

## Scope and next steps

This first notebook checks masked reconstruction and exposes an event embedding. A later exercise can freeze that encoder and train a small supervised H→WW versus background probe with labels from the training split, then report validation and held-out test performance. Check the physics content of the sample and generator/systematic splits before claiming any generalization beyond this file.
