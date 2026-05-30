"""Tests for biosae.experts.supervised_encoder (P1-on-ESM).

Offline / CPU. The encoder is exercised on a tiny structured signal where a
class fires on a dedicated feature over a span, so the occurrence-pooled CE
must learn it and the label-free latents must become span-discriminative.
"""

from __future__ import annotations

import pytest
import torch

from biosae.experts import (
    SupervisedEncoder,
    SupervisedEncoderConfig,
    train_supervised_encoder,
)


def _cfg(**kw) -> SupervisedEncoderConfig:
    base = dict(d_in=16, d_latent=32, depth=1, n_heads=4, n_motif_classes=4,
                epochs=0, batch_proteins=8, device="cpu", seed=0)
    base.update(kw)
    return SupervisedEncoderConfig(**base)


def _labelled(n=16, d=16, L=30, n_classes=4, seed=1):
    g = torch.Generator().manual_seed(seed)
    acts, occ = [], []
    for i in range(n):
        x = torch.randn(L, d, generator=g) * 0.3
        c = 1 + (i % (n_classes - 1))
        s = 4 + (i % 8)
        x[s:s + 5, c] += 2.0
        acts.append(x)
        occ.append([(s, s + 5, c)])
    return acts, occ


def test_config_validation():
    with pytest.raises(ValueError, match="divisible"):
        _cfg(d_latent=30, n_heads=4)
    with pytest.raises(ValueError, match="n_motif_classes"):
        _cfg(n_motif_classes=1)
    with pytest.raises(ValueError, match="label_pool"):
        _cfg(label_pool="bogus")


def test_encode_shapes_and_per_protein():
    m = SupervisedEncoder(_cfg())
    z = m.encode(torch.randn(2, 12, 16))
    assert z.shape == (2, 12, 32)
    zl = m.encode_proteins([torch.randn(20, 16), torch.randn(7, 16)])
    assert [t.shape for t in zl] == [(20, 32), (7, 32)]


def test_training_learns_and_latents_become_discriminative():
    acts, occ = _labelled(n=16, n_classes=4)
    cfg = _cfg(n_motif_classes=4, epochs=25, lr=3e-3)
    model, hist = train_supervised_encoder(acts, occ, cfg)
    assert hist["ce"][-1] < hist["ce"][0]
    assert hist["ce_acc"][-1] > 0.8                  # easy toy → near-perfect
    # label-free latents should separate the planted spans from background.
    from biosae.sae.evaluation import score_occurrences
    Z = torch.cat(model.encode_proteins(acts), dim=0)
    lengths = [a.shape[0] for a in acts]
    flat, off = [], 0
    for i, a in enumerate(acts):
        for s, e, c in occ[i]:
            flat.append((str(c), off + s, off + e))
        off += a.shape[0]
    oc = score_occurrences(Z, flat, lengths, n_neg_per_pos=2, n_perm=30, seed=0)
    assert oc["mean_occ_auc"] > oc["mean_null"] + 0.15


def test_rejects_misaligned_occurrences():
    acts, occ = _labelled(n=8)
    with pytest.raises(ValueError, match="align"):
        train_supervised_encoder(acts, occ[:-1], _cfg(epochs=1))
