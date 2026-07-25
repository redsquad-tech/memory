import torch

from assocmem.baselines import ExactKNNBaseline, FrequencyBaseline
from assocmem.encoding import TernaryQuery
from assocmem.online import OnlineExample, evaluate_online


def test_evaluator_is_prequential():
    query = TernaryQuery(4, torch.tensor([0]), torch.tensor([1.0], dtype=torch.float32))
    stream = [OnlineExample(query, 1), OnlineExample(query, 1)]
    summary = evaluate_online(FrequencyBaseline(2), stream, log_every=0)
    assert summary["examples"] == 2
    assert summary["prequential_nll_bits"] > 0
    # First prediction is uniform/tie-broken to class 0; only the second can
    # benefit from the first observation.
    assert summary["accuracy"] == 0.5


def test_knn_cached_key_norm_matches_stored_key():
    query = TernaryQuery(
        8,
        torch.tensor([1, 3, 6]),
        torch.tensor([1.0, -1.0, 1.0]),
    )
    model = ExactKNNBaseline(8, 2, capacity=3, top_k=1)
    prediction = model.predict(query)
    model.observe(query, 1, prediction)
    assert int(model.key_nnz[0]) == int(model.keys[0].abs().sum()) == query.nnz
    assert torch.isfinite(model.predict(query).probabilities).all()


def test_knn_uses_empirical_prior_and_same_energy():
    first = TernaryQuery(8, torch.tensor([1, 3]), torch.tensor([1.0, -1.0]))
    other = TernaryQuery(8, torch.tensor([2]), torch.tensor([1.0]))
    model = ExactKNNBaseline(8, 2, capacity=4, top_k=1, key_scale=4.0, prior_mass=1.0, key_nnz=2)
    prediction = model.predict(first)
    model.observe(first, 1, prediction)
    model.observe(other, 0, model.predict(other))
    model.observe(other, 0, model.predict(other))
    result = model.predict(first)
    assert torch.isclose(result.probabilities.sum(), torch.tensor(1.0))
    # Exact match has E=0 and therefore vote mass 1. The prior is the
    # smoothed empirical distribution (3/5, 2/5), not a uniform prior.
    expected = torch.tensor([0.3, 0.7])
    assert torch.allclose(result.probabilities, expected, atol=1e-6)
