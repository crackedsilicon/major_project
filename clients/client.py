"""Client implementation for local training and metadata reporting.

This client supports a simple supervised local training loop using PyTorch.
It returns the updated model state_dict and metadata for aggregation.
"""
import time
from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim


class Client:
    def __init__(self, client_id, model, device='cpu'):
        self.client_id = client_id
        self.device = device
        self.model = deepcopy(model).to(self.device)
        self.last_trained = time.time()
        # rehearsal buffer for Class-Incremental Learning (store (x,y) tensors)
        self.rehearsal = []
        self.rehearsal_capacity = 200

    def local_train(self, train_loader, epochs=1, lr=0.01, teacher_state_dict=None, distill_lambda=0.5, mu=0.0, global_state=None):
        """Train the local model on supplied DataLoader and return state_dict.

        train_loader: iterable of (x,y) tensors on CPU
        """
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        distill = None
        teacher_model = None
        if teacher_state_dict is not None:
            # create teacher model for distillation
            try:
                teacher_model = deepcopy(self.model)
                teacher_model.load_state_dict(teacher_state_dict, strict=False)
                teacher_model.to(self.device)
                teacher_model.eval()
                distill = nn.KLDivLoss(reduction='batchmean')
            except Exception:
                teacher_model = None
                distill = None
        optimizer = optim.SGD(self.model.parameters(), lr=lr)

        # If we have rehearsal samples, mix them with current training data
        from torch.utils.data import ConcatDataset, DataLoader

        rehearsal_loader = self.sample_rehearsal_loader(batch_size=getattr(train_loader, 'batch_size', 32))
        if rehearsal_loader is not None:
            try:
                combined_dataset = ConcatDataset([train_loader.dataset, rehearsal_loader.dataset])
                loader = DataLoader(combined_dataset, batch_size=getattr(train_loader, 'batch_size', 32), shuffle=True)
            except Exception:
                loader = train_loader
        else:
            loader = train_loader

        total = 0
        correct = 0
        for epoch in range(epochs):
            for x, y in loader:
                x = x.to(self.device)
                y = y.to(self.device)
                optimizer.zero_grad()
                out = self.model(x)
                loss = criterion(out, y)
                # distillation loss on rehearsal samples using teacher logits
                if teacher_model is not None and distill is not None:
                    with torch.no_grad():
                        t_out = teacher_model(x)
                    p = torch.log_softmax(out / 1.0, dim=1)
                    q = torch.softmax(t_out / 1.0, dim=1)
                    loss = (1.0 - distill_lambda) * loss + distill_lambda * distill(p, q)
                # FedProx proximal term: mu/2 * ||w - w_global||^2
                if mu is not None and mu > 0.0 and global_state is not None:
                    prox = 0.0
                    for name, param in self.model.named_parameters():
                        if name in global_state:
                            g = global_state[name]
                            if isinstance(g, torch.Tensor):
                                prox += torch.sum((param - g.to(self.device)) ** 2)
                    loss = loss + (mu / 2.0) * prox
                loss.backward()
                optimizer.step()

                preds = out.argmax(dim=1)
                total += y.size(0)
                correct += (preds == y).sum().item()

        self.last_trained = time.time()
        acc = (correct / total) if total > 0 else 0.0
        # after local training, update rehearsal buffer with a sample of current data
        try:
            self._update_rehearsal_from_loader(train_loader)
        except Exception:
            pass
        return self.model.state_dict(), {'accuracy': acc, 'data_size': total, 'timestamp': self.last_trained}

    def evaluate(self, val_loader):
        self.model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                out = self.model(x)
                preds = out.argmax(dim=1)
                total += y.size(0)
                correct += (preds == y).sum().item()
        return {'accuracy': (correct / total) if total > 0 else 0.0}

    def metadata(self, val_loader=None):
        val_score = 0.0
        if val_loader is not None:
            val_score = self.evaluate(val_loader)['accuracy']
        return {
            'client_id': self.client_id,
            'val_score': val_score,
            'data_size': 0,
            'timestamp': self.last_trained
        }

    def _update_rehearsal_from_loader(self, loader, sample_frac=0.1):
        """Add a small fraction of samples from loader into rehearsal buffer."""
        import random
        samples = []
        for x, y in loader:
            for xi, yi in zip(x, y):
                samples.append((xi.cpu(), yi.cpu()))
        k = max(1, int(len(samples) * sample_frac))
        chosen = random.sample(samples, min(k, len(samples)))
        # append and cap
        self.rehearsal.extend(chosen)
        if len(self.rehearsal) > self.rehearsal_capacity:
            # keep most recent
            self.rehearsal = self.rehearsal[-self.rehearsal_capacity:]

    def sample_rehearsal_loader(self, batch_size=32):
        """Return a DataLoader built from the rehearsal buffer."""
        if len(self.rehearsal) == 0:
            return None
        xs = torch.stack([t[0] for t in self.rehearsal])
        ys = torch.tensor([int(t[1].item()) if hasattr(t[1], 'item') else int(t[1]) for t in self.rehearsal])
        ds = torch.utils.data.TensorDataset(xs, ys)
        return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)

    def get_features(self, x):
        """Extract penultimate-layer features from the model for input tensor x."""
        # Assumes model has `net` as nn.Sequential and last layer is classifier
        self.model.eval()
        with torch.no_grad():
            if hasattr(self.model, 'net') and isinstance(self.model.net, torch.nn.Sequential):
                # apply all layers except last
                feats = x.to(self.device)
                for layer in list(self.model.net.children())[:-1]:
                    feats = layer(feats)
                return feats
            else:
                # fallback: use full model outputs as features
                return self.model(x.to(self.device))

    def compute_prototypes(self, loader):
        """Compute class prototypes (mean feature vectors) from loader data."""
        feats_per_class = {}
        counts = {}
        self.model.eval()
        with torch.no_grad():
            for x, y in loader:
                f = self.get_features(x)
                if f.dim() == 1:
                    f = f.unsqueeze(0)
                for fi, yi in zip(f, y):
                    cls = int(yi.item()) if hasattr(yi, 'item') else int(yi)
                    if cls not in feats_per_class:
                        feats_per_class[cls] = fi.cpu().clone()
                        counts[cls] = 1
                    else:
                        feats_per_class[cls] += fi.cpu()
                        counts[cls] += 1

        prototypes = {}
        for cls, s in feats_per_class.items():
            prototypes[cls] = (s / counts[cls]).clone()
        # store locally
        self.prototypes = prototypes
        return prototypes
