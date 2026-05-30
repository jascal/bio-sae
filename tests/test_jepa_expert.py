"""Tests for biosae.experts — JEPA world-model experts + ensemble plumbing.

All offline and CPU-only: no ESM-2 forward, no network. The protein-native
JEPA is exercised on small random / structured tensors; the Hugging Face
backbone is tested via its *graceful-degradation* contract (a missing or
unsupported checkpoint must raise a clear error, never crash the import).
"""

from __future__ import annotations

import pytest
import torch

from biosae.experts import (
    CoExtraction,
    ExpertEnsemble,
    IdentityExpert,
    JepaConfig,
    JepaExpert,
    ProteinJEPA,
    Router,
    SupervisedJepaConfig,
    train_label_jepa,
    train_protein_jepa,
)
from biosae.experts.jepa_expert import (
    FlatJepaScorer,
    HFJepaBackbone,
    JepaBackendUnavailable,
    mutation_action,
)


def _cfg(**kw) -> JepaConfig:
    base = dict(d_in=16, d_latent=32, depth=1, predictor_depth=1, n_heads=4,
                epochs=0, batch_proteins=4, device="cpu", seed=0)
    base.update(kw)
    return JepaConfig(**base)


def _proteins(lengths=(20, 13, 27, 8), d=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.randn(L, d, generator=g) for L in lengths]


# ---------------------------------------------------------------------------
# JepaConfig validation
# ---------------------------------------------------------------------------
def test_config_rejects_bad_mask_mode():
    with pytest.raises(ValueError, match="mask_mode"):
        _cfg(mask_mode="nope")


def test_config_rejects_bad_mask_ratio():
    with pytest.raises(ValueError, match="mask_ratio"):
        _cfg(mask_ratio=1.5)


def test_config_rejects_indivisible_latent():
    with pytest.raises(ValueError, match="divisible"):
        _cfg(d_latent=30, n_heads=4)


# ---------------------------------------------------------------------------
# ProteinJEPA forward / encode / predict
# ---------------------------------------------------------------------------
def test_encode_shapes():
    m = ProteinJEPA(_cfg())
    x = torch.randn(3, 20, 16)
    z = m.encode(x)
    assert z.shape == (3, 20, 32)


def test_encode_deterministic_in_eval():
    m = ProteinJEPA(_cfg(dropout=0.0)).eval()
    x = torch.randn(2, 10, 16)
    assert torch.allclose(m.encode(x), m.encode(x), atol=1e-6)


def test_predict_shapes_and_query_mask():
    m = ProteinJEPA(_cfg()).eval()
    ctx = torch.randn(2, 12, 32)
    pred = m.predict(ctx)
    assert pred.shape == (2, 12, 32)
    qmask = torch.zeros(2, 12, dtype=torch.bool)
    qmask[:, -3:] = True
    pred_q = m.predict(ctx, query_mask=qmask)
    assert pred_q.shape == (2, 12, 32)


def test_action_changes_prediction():
    m = ProteinJEPA(_cfg(action_dim=20)).eval()
    ctx = torch.randn(1, 10, 32)
    base = m.predict(ctx)
    act = mutation_action("A", action_dim=20)
    moved = m.predict(ctx, action=act)
    assert not torch.allclose(base, moved, atol=1e-5)


# ---------------------------------------------------------------------------
# EMA target encoder + JEPA loss
# ---------------------------------------------------------------------------
def test_ema_update_moves_target_toward_context():
    m = ProteinJEPA(_cfg(ema_decay=0.5))
    # Perturb the context encoder so it differs from its EMA copy.
    with torch.no_grad():
        for p in m.context_encoder.parameters():
            p.add_(torch.randn_like(p))
    before = [tp.clone() for tp in m.target_encoder.parameters()]
    m.ema_update()
    moved = any(not torch.allclose(b, a)
                for b, a in zip(before, m.target_encoder.parameters()))
    assert moved
    # Target must stay frozen w.r.t. autograd.
    assert all(not p.requires_grad for p in m.target_encoder.parameters())


def test_jepa_loss_nonnegative_and_target_not_collapsed():
    m = ProteinJEPA(_cfg())
    x = torch.randn(4, 18, 16)
    mask = torch.zeros(4, 18, dtype=torch.bool)
    mask[:, 15:] = True                       # last 3 are padding
    loss, info = m.jepa_loss(x, mask, generator=torch.Generator().manual_seed(0))
    assert float(loss) >= 0.0
    assert info["n_masked"] > 0
    assert info["target_var"] > 0.0           # EMA target carries variance


def test_training_reduces_loss_on_structured_signal():
    """A learnable position-structured signal: loss should drop, no collapse."""
    torch.manual_seed(0)
    d = 16
    prots = []
    for _ in range(16):
        L = 24
        pos = torch.arange(L).float()
        base = torch.randn(L, d) * 0.2
        base[:, 0] += (pos / 4).sin()         # predictable structure along position
        prots.append(base)
    cfg = _cfg(epochs=25, batch_proteins=8, lr=3e-3, mask_mode="future")
    model, hist = train_protein_jepa(prots, cfg)
    assert hist["loss"][-1] < hist["loss"][0]
    assert hist["target_var"][-1] > 1e-4      # did not collapse to a constant


# ---------------------------------------------------------------------------
# JepaExpert wrapper
# ---------------------------------------------------------------------------
def test_jepa_expert_encode_flat_and_per_protein():
    model = ProteinJEPA(_cfg())
    ex = JepaExpert(model, name="jepa")
    assert ex.d_latent == 32
    z = ex.encode(torch.randn(20, 16))                 # flat → one protein
    assert z.shape == (20, 32)
    zl = ex.encode_proteins(_proteins())
    assert [t.shape[0] for t in zl] == [20, 13, 27, 8]
    assert all(t.shape[1] == 32 for t in zl)


def test_jepa_expert_predict_roundtrip():
    ex = JepaExpert(ProteinJEPA(_cfg()))
    z = ex.encode(torch.randn(12, 16))
    assert ex.predict(z).shape == z.shape


def test_jepa_expert_rejects_wrong_model_type():
    with pytest.raises(TypeError):
        JepaExpert(torch.nn.Linear(4, 4))


def test_mutation_action_one_hot_and_unknown():
    a = mutation_action("C", action_dim=20)
    assert a.sum() == 1.0 and a.argmax().item() == 1     # C is index 1
    assert mutation_action("Z", action_dim=20).sum() == 0.0   # unknown → no-op


# ---------------------------------------------------------------------------
# Router + ExpertEnsemble
# ---------------------------------------------------------------------------
def test_router_weights_sum_to_one_all_strategies():
    experts = [IdentityExpert(16, "esm2"), JepaExpert(ProteinJEPA(_cfg()))]
    x = torch.randn(30, 16)
    for strat in ("uniform", "input_norm"):
        w = Router(experts, strategy=strat).weights(x)
        assert w.shape == (2,)
        assert pytest.approx(float(w.sum()), abs=1e-5) == 1.0
        assert (w >= 0).all()
    w = Router(experts, strategy="learned", d_in=16).weights(x)
    assert pytest.approx(float(w.sum()), abs=1e-5) == 1.0


def test_router_learned_requires_d_in():
    with pytest.raises(ValueError, match="d_in"):
        Router([IdentityExpert(16)], strategy="learned")


def test_router_unknown_strategy_raises():
    with pytest.raises(ValueError, match="strategy"):
        Router([IdentityExpert(16)], strategy="bogus")


def test_ensemble_concat_and_route_widths():
    esm = IdentityExpert(16, "esm2")
    jepa = JepaExpert(ProteinJEPA(_cfg()))            # d_latent 32
    x = torch.randn(25, 16)
    cat = ExpertEnsemble([esm, jepa], fusion="concat")
    assert cat.d_latent == 16 + 32
    assert cat.encode(x).shape == (25, 48)
    routed = ExpertEnsemble([esm, jepa], fusion="route")
    assert routed.d_latent == 32                       # max width
    assert routed.encode(x).shape == (25, 32)


def test_identity_expert_is_passthrough():
    esm = IdentityExpert(16, "esm2")
    x = torch.randn(7, 16)
    assert torch.equal(esm.encode(x), x)


# ---------------------------------------------------------------------------
# FlatJepaScorer (score_against_ground_truth adapter)
# ---------------------------------------------------------------------------
def test_flat_scorer_shapes_and_length_check():
    ex = JepaExpert(ProteinJEPA(_cfg()))
    prots = _proteins()
    lengths = [p.shape[0] for p in prots]
    scorer = FlatJepaScorer(ex, lengths=lengths)
    X = torch.cat(prots, dim=0)
    xhat, z = scorer(X)
    assert xhat.shape == X.shape                        # readout reconstructs d_in
    assert z.shape == (X.shape[0], 32)
    with pytest.raises(ValueError, match="lengths"):
        scorer(torch.randn(X.shape[0] + 1, 16))


# ---------------------------------------------------------------------------
# Hugging Face backbone — graceful degradation
# ---------------------------------------------------------------------------
def test_hf_metadata_missing_returns_none():
    assert HFJepaBackbone.metadata("definitely/not-a-real-model-xyz") is None


def test_hf_load_missing_raises_actionable():
    with pytest.raises(JepaBackendUnavailable):
        HFJepaBackbone.load("definitely/not-a-real-model-xyz", d_in=8)


def test_hf_available_false_for_missing():
    assert HFJepaBackbone.available("definitely/not-a-real-model-xyz") is False


def test_backend_unavailable_is_runtime_error():
    assert issubclass(JepaBackendUnavailable, RuntimeError)


@pytest.mark.parametrize("model_id", ["facebook/vjepa2-vitl-fpc64-256", "quentinll/lewm-cube"])
def test_hf_cached_checkpoint_metadata_or_skip(model_id):
    """If a real JEPA checkpoint is cached, metadata parses and load degrades
    cleanly; otherwise skip (CI without the cache)."""
    meta = HFJepaBackbone.metadata(model_id)
    if meta is None:
        pytest.skip(f"{model_id} not in local HF cache")
    assert "hidden_size" in meta and meta["hidden_size"] > 0
    # On a stack that can't build it, load must raise the typed error (not crash).
    try:
        HFJepaBackbone.load(model_id, d_in=320)
    except JepaBackendUnavailable:
        pass


# ---------------------------------------------------------------------------
# CoExtraction dataclass
# ---------------------------------------------------------------------------
def test_coextraction_feeds():
    esm_res = torch.randn(40, 16)
    jepa_res = torch.randn(40, 8)
    co = CoExtraction(
        esm_residue=esm_res, esm_pooled=torch.randn(3, 16),
        jepa_residue=jepa_res, jepa_pooled=torch.randn(3, 8),
        lengths=[20, 13, 7],
    )
    assert co.feed("esm").shape == (40, 16)
    assert co.feed("jepa").shape == (40, 8)
    assert co.feed("concat").shape == (40, 24)
    with pytest.raises(ValueError, match="unknown feed"):
        co.feed("bogus")


def test_coextraction_without_jepa_falls_back_to_esm():
    co = CoExtraction(
        esm_residue=torch.randn(10, 16), esm_pooled=torch.randn(2, 16),
        jepa_residue=None, jepa_pooled=None, lengths=[6, 4],
    )
    assert torch.equal(co.concat_residue(), co.esm_residue)
    with pytest.raises(ValueError, match="no JEPA feed"):
        co.feed("jepa")


# ---------------------------------------------------------------------------
# Supervised Label-JEPA (P2)
# ---------------------------------------------------------------------------
def _sup_cfg(**kw) -> SupervisedJepaConfig:
    base = dict(d_in=16, d_latent=32, depth=1, predictor_depth=1, n_heads=4,
                n_motif_classes=4, epochs=0, batch_proteins=8, device="cpu", seed=0)
    base.update(kw)
    return SupervisedJepaConfig(**base)


def _labelled_proteins(n=16, d=16, L=30, n_classes=4, seed=1):
    """Proteins with a class-discriminative span: class c fires on feature c."""
    g = torch.Generator().manual_seed(seed)
    acts, occ = [], []
    for i in range(n):
        x = torch.randn(L, d, generator=g) * 0.3
        c = 1 + (i % (n_classes - 1))          # classes 1..M (0 is background)
        s = 4 + (i % 8)
        x[s:s + 5, c] += 2.0
        acts.append(x)
        occ.append([(s, s + 5, c)])
    return acts, occ


def test_supervised_config_validates():
    with pytest.raises(ValueError, match="n_motif_classes"):
        _sup_cfg(n_motif_classes=1)
    with pytest.raises(ValueError, match="label_pool"):
        _sup_cfg(label_pool="bogus")
    with pytest.raises(ValueError, match="supervision"):
        _sup_cfg(supervision="nope")


def test_label_head_present_only_for_supervised_cfg():
    assert ProteinJEPA(_sup_cfg()).label_head is not None
    assert ProteinJEPA(_cfg()).label_head is None          # plain JepaConfig → no head


def test_predict_label_shape_and_guard():
    m = ProteinJEPA(_sup_cfg(n_motif_classes=5)).eval()
    z_pred = torch.randn(3, 12, 32)
    span = torch.zeros(3, 12, dtype=torch.bool)
    span[:, 4:8] = True
    logits = m.predict_label(z_pred, span)
    assert logits.shape == (3, 5)
    # max-pool over an empty span must not NaN/inf (nan_to_num guard).
    empty = torch.zeros(3, 12, dtype=torch.bool)
    assert torch.isfinite(m.predict_label(z_pred, empty)).all()
    # the plain model has no head
    with pytest.raises(RuntimeError, match="no label head"):
        ProteinJEPA(_cfg()).predict_label(z_pred, span)


def test_label_jepa_learns_and_does_not_collapse():
    acts, occ = _labelled_proteins(n=16, n_classes=4)
    cfg = _sup_cfg(n_motif_classes=4, epochs=40, lr=3e-3, label_weight=1.0)
    model, hist = train_label_jepa(acts, occ, cfg)
    # CE drops below the uniform-prior baseline ln(4) ≈ 1.386, accuracy rises.
    assert hist["ce"][-1] < hist["ce"][0]
    assert hist["ce_acc"][-1] > 0.5
    assert hist["target_var"][-1] > 1e-4                   # EMA target didn't collapse
    # encode stays label-free (one protein, no spans needed)
    z = JepaExpert(model).encode(acts[0])
    assert z.shape == (acts[0].shape[0], cfg.d_latent)


def test_label_jepa_rejects_misaligned_occurrences():
    acts, occ = _labelled_proteins(n=8)
    with pytest.raises(ValueError, match="align"):
        train_label_jepa(acts, occ[:-1], _sup_cfg(epochs=1))


def test_supervised_cfg_inherits_jepa_validation():
    # SupervisedJepaConfig must still enforce the base JepaConfig invariants.
    with pytest.raises(ValueError, match="divisible"):
        _sup_cfg(d_latent=30, n_heads=4)
