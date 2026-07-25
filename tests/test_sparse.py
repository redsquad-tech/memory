import torch

from assocmem.encoding import TernaryQuery
from assocmem.sparse import SparseExactIndex


def test_sparse_index_matches_dense_scores_and_ties():
    index = SparseExactIndex(dimension=10, capacity=3, key_nnz=3, key_scale=6)
    keys = torch.zeros((3, 10))
    keys[0, [1, 2]] = torch.tensor([3.0, -3.0])
    keys[1, [1, 4, 8]] = torch.tensor([-2.0, 2.0, 2.0])
    keys[2, [6]] = 6.0
    for slot in range(3):
        index.set_key(slot, keys[slot])
    query = TernaryQuery(10, torch.tensor([1, 4]), torch.tensor([1.0, -1.0], dtype=torch.float32))
    result = index.score(query)
    dense_scores = keys[:, query.indices] @ query.values
    expected = torch.argsort(dense_scores, descending=True, stable=True)
    assert torch.equal(result.slot_ids, expected)
    assert torch.allclose(result.scores, dense_scores[expected])
