"""Rank complete triples; evaluate controlled candidate pools, not human taste.

All queries are held-out positive triples, with the original bottom/shoes plus
same-split distractors. Candidate construction is identical across variants.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import random
import joblib
import numpy as np

ROOT = Path(__file__).absolute().parents[1]


def feature_row(ids, items):
    a,b,c = [items[i]['category'] for i in ids]
    return {'top':a, 'bottom':b, 'shoes':c, 'top_bottom':a+'|'+b, 'top_shoes':a+'|'+c, 'bottom_shoes':b+'|'+c}


def mmr(candidates, relevance, k, lam):
    selected, remaining = [], list(range(len(candidates)))
    while remaining and len(selected) < k:
        def objective(i):
            overlap = max((sum(x == y for x,y in zip(candidates[i][1:], candidates[j][1:]))/2 for j in selected), default=0)
            return lam*relevance[i] - (1-lam)*overlap
        best = max(remaining, key=objective)
        selected.append(best); remaining.remove(best)
    return selected


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-dir', type=Path)
    p.add_argument('--embeddings', type=Path)
    p.add_argument('--mlp-checkpoint', type=Path)
    p.add_argument('--queries', type=int, default=100)
    p.add_argument('--per-slot', type=int, default=10)
    args = p.parse_args()
    data = ROOT / 'data/processed/polyvore'
    items = json.loads((data / 'items.json').read_text())
    expected_version = ('image_' + hashlib.sha256(args.embeddings.read_bytes()).hexdigest()[:12]) if args.embeddings else 'metadata_baseline'
    model_dir = args.model_dir or ROOT / 'artifacts/compatibility' / expected_version
    bundle = joblib.load(model_dir / 'random_forest.joblib')
    if bundle['feature_version'] != expected_version:
        raise ValueError('Embedding version does not match compatibility model')
    if args.mlp_checkpoint:
        if args.embeddings is None:
            raise ValueError('MLP evaluation requires embeddings')
        from outfit_ranker import OutfitRanker
        bundle['model'] = OutfitRanker(args.mlp_checkpoint, hashlib.sha256(args.embeddings.read_bytes()).hexdigest())
    embeddings = None
    if args.embeddings:
        a = np.load(args.embeddings, allow_pickle=False)
        embeddings = dict(zip(a['ids'].astype(str), a['vectors']))
    if bundle['feature_version'].startswith('image_') and embeddings is None:
        raise ValueError('Image model requires its embeddings')
    rules = json.loads((ROOT / 'artifacts/mining/fpgrowth_rules.json').read_text())
    prepared = {}
    for split in ('validation', 'test'):
        rows = [r for r in json.loads((data / f'{split}_examples.json').read_text()) if r['label'] == 1]
        rng = random.Random(71); rng.shuffle(rows)
        # Keep candidate pool within exported embeddings when applicable.
        pools = {s:sorted(i for i,v in items.items() if v['slot']==s and v['split']==split and (embeddings is None or i in embeddings)) for s in ('bottom','shoes')}
        prepared[split] = []
        for row in rows[:args.queries]:
            truth = row['item_ids']
            options = []
            for slot, iid in zip(('bottom','shoes'), truth[1:]):
                others = [i for i in pools[slot] if i != iid]
                choices = [iid] + rng.sample(others, min(args.per_slot-1, len(others)))
                rng.shuffle(choices); options.append(choices)
            candidates = [(truth[0],b,s) for b,s in itertools.product(*options)]
            X = bundle['vectorizer'].transform([feature_row(c,items) for c in candidates])
            if embeddings is not None:
                visual=[]
                for c in candidates:
                    a,b,d = [embeddings[i] for i in c]
                    visual.append(np.concatenate([a,b,d,a*b,a*d,b*d]))
                X=np.hstack([X,visual])
            compatibility = bundle['model'].predict_proba(X)[:,1]
            rule_scores=[]
            for c in candidates:
                tokens = {items[i]['slot']+':'+items[i]['category'] for i in c}
                rule_scores.append(max((r['confidence'] for r in rules if set(r['antecedents'])|set(r['consequents']) <= tokens),default=0))
            prepared[split].append((candidates, tuple(truth), compatibility, np.array(rule_scores)))
    def assess(split, alpha, lam):
        hits, reciprocal, diversity = [],[],[]
        for candidates,truth,comp,rule in prepared[split]:
            selected = mmr(candidates, alpha*comp+(1-alpha)*rule, 5, lam)
            outfits=[candidates[i] for i in selected]
            hits.append(truth in outfits)
            reciprocal.append(1/(outfits.index(truth)+1) if truth in outfits else 0)
            pair_sim=[sum(x==y for x,y in zip(a[1:],b[1:]))/2 for a,b in itertools.combinations(outfits,2)]
            diversity.append(1-np.mean(pair_sim) if pair_sim else 0)
        return {'hit_at_5':float(np.mean(hits)), 'mrr_at_5':float(np.mean(reciprocal)),
                'diversity_at_5':float(np.mean(diversity)), 'queries':len(hits)}
    experiments=[]
    for a in (1.0,0.75,0.5):
        for lam in (1.0,0.9,0.7):
            m=assess('validation',a,lam)
            experiments.append({'alpha':a,'lambda':lam,**m})
    best=max(experiments,key=lambda r:r['hit_at_5']+0.05*r['diversity_at_5'])
    report={'protocol':'original full triple retrieval among same-split controlled candidates; original guaranteed present',
            'candidates_per_query':args.per_slot**2,'feature_version':bundle['feature_version'],
            'validation_selection_objective':'Hit@5 + 0.05*diversity@5', 'validation_search':experiments,
            'chosen':best,'test_baseline':assess('test',1,1), 'test_selected':assess('test',best['alpha'],best['lambda'])}
    out=ROOT/'reports/metrics';out.mkdir(parents=True,exist_ok=True)
    report_name = 'ranking_mlp_verified.json' if args.mlp_checkpoint else f"ranking_{bundle['feature_version']}.json"
    if args.mlp_checkpoint:
        report['model_checkpoint'] = str(args.mlp_checkpoint)
    (out/report_name).write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
