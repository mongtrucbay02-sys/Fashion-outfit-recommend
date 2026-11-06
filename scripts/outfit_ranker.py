"""Load a trained MLP for the existing ranking evaluator."""
import numpy as np
import torch


class OutfitRanker:
    def __init__(self, checkpoint, embedding_sha256):
        torch.set_num_threads(4)
        bundle = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if bundle['embedding_sha256'] != embedding_sha256:
            raise ValueError('MLP checkpoint belongs to a different embedding archive')
        c = bundle['config']
        self.model = torch.nn.Sequential(
            torch.nn.Linear(768, c['hidden']), torch.nn.ReLU(), torch.nn.Dropout(c['dropout']),
            torch.nn.Linear(c['hidden'], 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
        self.model.load_state_dict(bundle['state_dict'])
        self.model.eval()

    def predict_proba(self, features):
        # Legacy evaluator prepends categorical features; MLP uses only the visual suffix.
        x = torch.as_tensor(np.asarray(features)[:, -768:], dtype=torch.float32)
        with torch.no_grad():
            p = self.model(x).squeeze(1).sigmoid().numpy()
        if not np.isfinite(p).all():
            raise ValueError('Non-finite MLP predictions')
        return np.column_stack((1-p, p))
