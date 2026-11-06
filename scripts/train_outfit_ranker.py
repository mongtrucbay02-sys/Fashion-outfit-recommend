"""CPU pairwise outfit ranker; fixed legacy candidate protocol, validation-only selection."""
import hashlib
import itertools
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from evaluate_recommendation import mmr

ROOT = Path(__file__).absolute().parents[1]


def main():
    torch.set_num_threads(4)
    started = time.perf_counter()
    out = ROOT / 'artifacts/outfit_ranker'
    out.mkdir(parents=True, exist_ok=True)
    data = ROOT / 'data/processed/polyvore'
    items = json.loads((data / 'items.json').read_text())
    rows = {s: json.loads((data / f'{s}_examples.json').read_text()) for s in ('train', 'validation', 'test')}
    path = ROOT / 'artifacts/embedding/catalog_vectors.npz'
    archive = np.load(path, allow_pickle=False)
    ids = archive['ids'].astype(str)
    index = {v: i for i, v in enumerate(ids)}
    vectors = torch.tensor(archive['vectors'], dtype=torch.float32)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    positives = {s: [r for r in rows[s] if r['label'] == 1] for s in rows}
    used = {s: {i for r in rows[s] for i in r['item_ids']} for s in rows}
    assert not (used['train'] & used['validation'] or used['train'] & used['test'] or used['validation'] & used['test'])
    train = np.array([[index[i] for i in r['item_ids']] for r in positives['train']])
    known = set(map(tuple, train))
    pools = {slot: np.array([index[i] for i in ids if items[i]['split'] == 'train' and items[i]['slot'] == slot]) for slot in ('top', 'bottom', 'shoes')}
    neighbors = {}
    for slot, pool in pools.items():
        knn = NearestNeighbors(n_neighbors=min(21, len(pool)), metric='cosine', n_jobs=4).fit(vectors[pool].numpy())
        unique = np.unique(train[:, ('top', 'bottom', 'shoes').index(slot)])
        near = knn.kneighbors(vectors[unique].numpy(), return_distance=False)
        for iid, nn in zip(unique, near):
            neighbors[int(iid)] = pool[nn][pool[nn] != iid]
    rules = json.loads((ROOT / 'artifacts/mining/fpgrowth_rules.json').read_text())

    def features(triples):
        a, b, c = vectors[triples].unbind(1)
        return torch.cat((a, b, c, a*b, a*c, b*c), dim=1)

    def candidates(split):
        # Keep RNG call order identical to evaluate_recommendation.py.
        rs = positives[split].copy()
        rng = random.Random(71)
        rng.shuffle(rs)
        pool = {s: sorted(i for i, v in items.items() if v['slot'] == s and v['split'] == split and i in index) for s in ('bottom', 'shoes')}
        result = []
        for row in rs[:100]:
            truth = row['item_ids']
            options = []
            for slot, iid in zip(('bottom', 'shoes'), truth[1:]):
                others = [i for i in pool[slot] if i != iid]
                choices = [iid] + rng.sample(others, min(9, len(others)))
                rng.shuffle(choices)
                options.append(choices)
            cs = [(truth[0], b, s) for b, s in itertools.product(*options)]
            rule = []
            for c in cs:
                tokens = {items[i]['slot'] + ':' + items[i]['category'] for i in c}
                rule.append(max((r['confidence'] for r in rules if set(r['antecedents']) | set(r['consequents']) <= tokens), default=0))
            result.append((cs, tuple(truth), features(np.array([[index[i] for i in c] for c in cs])), np.array(rule)))
        return result

    validation = candidates('validation')

    def scores(model, prepared):
        model.eval()
        with torch.no_grad():
            return [model(x).squeeze(1).sigmoid().numpy() for _, _, x, _ in prepared]

    def assess(prepared, predictions, alpha=1., lam=1.):
        hits, rr, diversity = [], [], []
        for (cs, truth, _, rule), score in zip(prepared, predictions):
            selected = [cs[i] for i in mmr(cs, alpha*score+(1-alpha)*rule, 5, lam)]
            hits.append(int(truth in selected))
            rr.append(1/(selected.index(truth)+1) if truth in selected else 0)
            sim = [sum(x == y for x, y in zip(a[1:], b[1:]))/2 for a, b in itertools.combinations(selected, 2)]
            diversity.append(1-float(np.mean(sim)))
        return {'hit_at_5': float(np.mean(hits)), 'mrr_at_5': float(np.mean(rr)), 'diversity_at_5': float(np.mean(diversity)), 'queries': len(hits)}

    def make_model(hidden, dropout):
        return torch.nn.Sequential(torch.nn.Linear(768, hidden), torch.nn.ReLU(), torch.nn.Dropout(dropout), torch.nn.Linear(hidden, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))

    experiments, best = [], None
    configs = [{'hidden': 128, 'dropout': .2, 'hard_fraction': 0.}, {'hidden': 128, 'dropout': .2, 'hard_fraction': .5}, {'hidden': 256, 'dropout': .3, 'hard_fraction': .5}]
    for config in configs:
        torch.manual_seed(42)
        rng = np.random.default_rng(42)
        model = make_model(config['hidden'], config['dropout'])
        opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
        for epoch in range(1, 21):
            begin = time.perf_counter()
            model.train()
            losses = []
            for start in range(0, len(train), 256):
                # Resample positives and a mixture of random / visually close same-slot negatives.
                pos = train[rng.integers(len(train), size=min(256, len(train)-start))]
                neg = pos.copy()
                for j, triple in enumerate(pos):
                    for attempt in range(100):
                        slot = int(rng.integers(3))
                        pool = neighbors[int(triple[slot])] if rng.random() < config['hard_fraction'] else pools[('top', 'bottom', 'shoes')[slot]]
                        replacement = int(rng.choice(pool))
                        trial = triple.copy()
                        trial[slot] = replacement
                        if replacement != triple[slot] and tuple(trial) not in known:
                            neg[j] = trial
                            break
                    else:
                        raise RuntimeError('Could not sample a negative')
                opt.zero_grad()
                loss = torch.nn.functional.softplus(model(features(neg))-model(features(pos))).mean()
                loss.backward()
                opt.step()
                losses.append(loss.item())
            metrics = assess(validation, scores(model, validation))
            record = {'config': config, 'epoch': epoch, 'loss': float(np.mean(losses)), 'seconds': time.perf_counter()-begin, 'validation': metrics}
            experiments.append(record)
            key = (metrics['hit_at_5'], metrics['mrr_at_5'])
            if best is None or key > best[0]:
                best = (key, record)
                torch.save({'state_dict': model.state_dict(), 'config': config, 'epoch': epoch, 'embedding_sha256': digest}, out / 'best.pt')
            print(json.dumps(record), flush=True)
    checkpoint = torch.load(out / 'best.pt', weights_only=False, map_location='cpu')
    config = checkpoint['config']
    model = make_model(config['hidden'], config['dropout'])
    model.load_state_dict(checkpoint['state_dict'])
    val_scores = scores(model, validation)
    search = [{'alpha': a, 'lambda': l, **assess(validation, val_scores, a, l)} for a in (1., .75, .5) for l in (1., .9, .7)]
    chosen = max(search, key=lambda m: m['hit_at_5']+.05*m['diversity_at_5'])
    # Test is constructed and scored only after model and reranking selection are frozen.
    test = candidates('test')
    predictions = scores(model, test)
    report = {'protocol': 'Same 100 queries and 100 candidates as legacy ranking; synthetic training negatives may be valid outfits.', 'embedding_sha256': digest, 'selection': 'validation Hit@5 then MRR@5; reranking validation Hit@5 + .05 diversity', 'experiments': experiments, 'best_model': best[1], 'reranking_search': search, 'chosen': chosen, 'test_baseline': assess(test, predictions), 'test_selected': assess(test, predictions, chosen['alpha'], chosen['lambda']), 'previous_test_hit_at_5': .12, 'elapsed_seconds': time.perf_counter()-started}
    checkpoint['reranking'] = chosen
    torch.save(checkpoint, out / 'best.pt')
    (ROOT / 'reports/metrics/ranking_mlp.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in ('experiments', 'reranking_search')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
