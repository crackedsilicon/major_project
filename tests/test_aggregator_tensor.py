import torch
from server.aggregator import aggregate


def make_state_dict(val):
    return {
        'w1': torch.tensor([val], dtype=torch.float32),
        'b1': torch.tensor(val * 2.0, dtype=torch.float32)
    }


def test_tensor_aggregation():
    s1 = make_state_dict(1.0)
    s2 = make_state_dict(3.0)
    models = [s1, s2]
    now = 1_700_000_000.0  # arbitrary constant for timestamps
    metadatas = [
        {'data_size': 10, 'val_score': 0.5, 'timestamp': now},
        {'data_size': 5, 'val_score': 0.2, 'timestamp': now}
    ]

    agg = aggregate(models, metadatas, alpha=1.0, beta=1.0, gamma=0.0)
    # with gamma=0, weights proportional to ds*(1+val_score)
    w1 = 10 * (1.0 + 0.5)
    w2 = 5 * (1.0 + 0.2)
    s = w1 + w2
    expected_w1 = w1 / s
    expected_w2 = w2 / s

    # expected tensor values
    expected_w = expected_w1 * 1.0 + expected_w2 * 3.0
    expected_b = expected_w1 * 2.0 + expected_w2 * 6.0

    assert torch.allclose(agg['w1'], torch.tensor([expected_w], dtype=torch.float32), atol=1e-6)
    assert torch.allclose(agg['b1'], torch.tensor(expected_b, dtype=torch.float32), atol=1e-6)
