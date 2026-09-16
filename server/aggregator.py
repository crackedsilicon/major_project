"""Adaptive federated aggregator.

Provides state_dict-aware aggregation using PyTorch tensors. The primary
aggregation strategy computes a weighted average of parameters where weights
are derived from client metadata (data_size, val_score, staleness).

Function: aggregate(models, metadatas, alpha=1.0, beta=1.0, gamma=1.0)
 - models: list of state_dict-like mappings (tensors or scalars)
 - metadatas: list of dicts with keys 'data_size', 'val_score', 'timestamp'
 - alpha: weight multiplier for data_size
 - beta: weight multiplier for val_score
 - gamma: staleness penalty multiplier
"""
import time
from typing import List, Dict
import torch


def compute_weights(metadatas: List[Dict], alpha=1.0, beta=1.0, gamma=1.0, strategy='adaptive'):
    """Return normalized aggregation weights for client metadata."""
    if not metadatas:
        raise ValueError("No metadata provided for weight computation")
    if strategy == 'fedavg':
        return [1.0 / len(metadatas)] * len(metadatas)
    if strategy in ('weighted', 'fedprox'):
        weights = [max(0.0, float(md.get('data_size', 1.0))) for md in metadatas]
    else:
        weights = _compute_adaptive_weights(metadatas, alpha=alpha, beta=beta, gamma=gamma)
    total = sum(weights)
    if total <= 0.0:
        return [1.0 / len(weights)] * len(weights)
    return [weight / total for weight in weights]


def _compute_adaptive_weights(metadatas: List[Dict], alpha=1.0, beta=1.0, gamma=1.0):
    now = time.time()
    weights = []
    for md in metadatas:
        ds = float(md.get('data_size', 1.0))
        vs = float(md.get('val_score', 0.0))
        staleness = float(now - md.get('timestamp', now))
        # combine terms: larger data_size and val_score increase weight; staleness penalizes
        w = (ds ** alpha) * ((1.0 + vs) ** beta) / (1.0 + gamma * staleness)
        weights.append(w)
    return weights


def aggregate(models: List[Dict], metadatas: List[Dict], alpha=1.0, beta=1.0, gamma=1.0, strategy='adaptive'):
    """Aggregate PyTorch state_dicts using adaptive weights.

    Returns a state_dict suitable for `load_state_dict` on the global model.
    """
    if len(models) == 0:
        raise ValueError("No models provided for aggregation")
    if len(models) != len(metadatas):
        raise ValueError("Number of models and metadatas must match")

    weights = compute_weights(
        metadatas,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        strategy=strategy,
    )

    # initialize accumulator with zeros using the first model's keys
    agg = {}
    for key in models[0].keys():
        val = models[0][key]
        if isinstance(val, torch.Tensor):
            agg[key] = torch.zeros_like(val, dtype=val.dtype)
        else:
            try:
                # attempt numeric zero
                agg[key] = 0.0
            except Exception:
                agg[key] = None

    # accumulate weighted parameters
    for w, state in zip(weights, models):
        for k, v in state.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                if agg.get(k) is None:
                    agg[k] = (v * w).clone()
                else:
                    agg[k] = agg[k] + v * w
            else:
                # numeric fallback
                try:
                    if agg.get(k) is None:
                        agg[k] = v * w
                    else:
                        agg[k] = agg[k] + v * w
                except Exception:
                    # leave as-is
                    pass

    # ensure tensors are detached/clone
    for k, v in list(agg.items()):
        if isinstance(v, torch.Tensor):
            agg[k] = v.clone()

    return agg
