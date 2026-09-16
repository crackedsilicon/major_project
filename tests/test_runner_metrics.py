import torch
from torch.utils.data import DataLoader, TensorDataset

from experiments.runner import filter_loader_by_classes


def test_filter_loader_by_classes_keeps_requested_labels():
    dataset = TensorDataset(
        torch.arange(12, dtype=torch.float32).reshape(6, 2),
        torch.tensor([0, 1, 2, 1, 2, 0]),
    )
    loader = DataLoader(dataset, batch_size=3, shuffle=False)

    filtered = filter_loader_by_classes(loader, [1, 2])
    labels = torch.cat([labels for _, labels in filtered]).tolist()

    assert labels == [1, 2, 1, 2]