import time
from server.aggregator import aggregate, compute_weights


def test_aggregate_simple():
    models = [
        {'w': 1.0},
        {'w': 3.0}
    ]
    now = time.time()
    metadatas = [
        {'data_size': 10, 'val_score': 0.5, 'timestamp': now},
        {'data_size': 5, 'val_score': 0.2, 'timestamp': now}
    ]
    agg = aggregate(models, metadatas)
    assert 'w' in agg
    # weighted average should be between 1 and 3
    assert 1.0 <= agg['w'] <= 3.0


def test_adaptive_weights_use_validation_quality():
    now = time.time()
    metadatas = [
        {'data_size': 10, 'val_score': 0.9, 'timestamp': now},
        {'data_size': 10, 'val_score': 0.1, 'timestamp': now},
    ]
    weights = compute_weights(metadatas, gamma=0.0)

    assert weights[0] > weights[1]
    assert abs(sum(weights) - 1.0) < 1e-9


def test_adaptive_weights_penalize_stale_clients():
    now = time.time()
    metadatas = [
        {'data_size': 10, 'val_score': 0.5, 'timestamp': now},
        {'data_size': 10, 'val_score': 0.5, 'timestamp': now - 100.0},
    ]
    weights = compute_weights(metadatas, gamma=1.0)

    assert weights[0] > weights[1]


def test_fedavg_weights_are_uniform():
    weights = compute_weights([
        {'data_size': 1, 'val_score': 0.1},
        {'data_size': 100, 'val_score': 0.9},
    ], strategy='fedavg')

    assert weights == [0.5, 0.5]


def test_fedprox_uses_data_weighted_server_aggregation():
    weights = compute_weights([
        {'data_size': 1},
        {'data_size': 3},
    ], strategy='fedprox')

    assert weights == [0.25, 0.75]
