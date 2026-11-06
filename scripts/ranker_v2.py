"""Reusable smaller whole-outfit and pairwise compatibility networks."""
import torch


class RankerV2(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pairwise = config['family'] == 'pairwise'
        width = 384 if self.pairwise else 768
        def head():
            return torch.nn.Sequential(torch.nn.Linear(width, config['hidden']), torch.nn.ReLU(),
                torch.nn.Dropout(config['dropout']), torch.nn.Linear(config['hidden'], 1))
        self.heads = torch.nn.ModuleList([head() for _ in range(3 if self.pairwise else 1)])

    def forward(self, x):
        if not self.pairwise:
            return self.heads[0](x)
        a, b, c, ab, ac, bc = x.split(128, dim=1)
        return (self.heads[0](torch.cat((a,b,ab),1)) + self.heads[1](torch.cat((a,c,ac),1))
                + self.heads[2](torch.cat((b,c,bc),1))) / 3


def load_ensemble(path, embedding_sha256):
    bundle = torch.load(path, weights_only=False, map_location='cpu')
    if bundle['embedding_sha256'] != embedding_sha256:
        raise ValueError('Embedding archive does not match checkpoint')
    models = []
    for state in bundle['states']:
        model = RankerV2(bundle['config'])
        model.load_state_dict(state)
        models.append(model.eval())
    return models, bundle


def predict(models, x):
    with torch.no_grad():
        return torch.cat([torch.stack([m(batch).squeeze(1).sigmoid() for m in models]).mean(0)
                          for batch in x.split(4096)]).numpy()
