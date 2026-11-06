"""Predeclared validation search; old 100 and fresh 500 query confirmation after freeze.

Confirmation queries are from the same previously used test split, NOT a new independent dataset.
"""
import copy
import hashlib
import itertools
import json
from pathlib import Path
import random
import time

import joblib
import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from evaluate_recommendation import feature_row, mmr
from outfit_ranker import OutfitRanker
from ranker_v2 import RankerV2, load_ensemble, predict

ROOT = Path(__file__).absolute().parents[1]


def main():
    torch.set_num_threads(4)
    start_time = time.perf_counter()
    out = ROOT/'artifacts/outfit_ranker_v2'
    out.mkdir(parents=True, exist_ok=True)
    report_path = ROOT/'reports/metrics/ranking_v2.json'
    configs = [{'family': family, 'hidden': 32 if family == 'pairwise' else 64,
                'dropout': .4, 'weight_decay': .01, 'hard_fraction': hard}
               for family in ('whole', 'pairwise') for hard in (0., .25, .5)]
    plan = {'configs': configs, 'seeds': [17,42,83], 'max_epochs': 30, 'patience': 6,
            'validation_queries': 267, 'confirmation_queries': 500, 'legacy_queries': 100,
            'selection': 'mean best validation Hit@5 over three seeds; tie mean MRR@5; then validation-only ensemble and reranking comparison with LR, RF and previous MLP',
            'note': 'Confirmation uses further test queries, not a new independent dataset; old classifiers previously saw aggregate classification evaluation on this split.'}
    (out/'plan.json').write_text(json.dumps(plan, indent=2))
    path = ROOT/'artifacts/embedding/catalog_vectors.npz'
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    z = np.load(path, allow_pickle=False)
    ids = z['ids'].astype(str)
    index = {v:i for i,v in enumerate(ids)}
    vectors = torch.tensor(z['vectors'], dtype=torch.float32)
    data = ROOT/'data/processed/polyvore'
    items = json.loads((data/'items.json').read_text())
    rows = {s:json.loads((data/f'{s}_examples.json').read_text()) for s in ('train','validation','test')}
    positives = {s:[r for r in rs if r['label']==1] for s,rs in rows.items()}
    used = {s:{i for r in rs for i in r['item_ids']} for s,rs in rows.items()}
    assert all(not used[a]&used[b] for a,b in itertools.combinations(used,2))
    train = np.array([[index[i] for i in r['item_ids']] for r in positives['train']])
    known = set(map(tuple,train))
    rules = json.loads((ROOT/'artifacts/mining/fpgrowth_rules.json').read_text())
    rule_cache = {}

    def features(triples):
        a,b,c = vectors[triples].unbind(1)
        return torch.cat((a,b,c,a*b,a*c,b*c),1)

    def candidates(split, count):
        rs = positives[split].copy()
        rng = random.Random(71)
        rng.shuffle(rs)
        pools = {s:sorted(i for i,v in items.items() if v['slot']==s and v['split']==split and i in index) for s in ('bottom','shoes')}
        groups, truths, rule_scores, truth_indices = [], [], [], []
        for row in rs[:count]:
            truth = tuple(row['item_ids'])
            options=[]
            for slot,iid in zip(('bottom','shoes'),truth[1:]):
                others=[i for i in pools[slot] if i!=iid]
                choices=[iid]+rng.sample(others,min(9,len(others)))
                rng.shuffle(choices)
                options.append(choices)
            cs=[(truth[0],b,s) for b,s in itertools.product(*options)]
            assert len(cs)==100 and len(set(cs))==100
            groups.append(cs); truths.append(truth); truth_indices.append(cs.index(truth))
            rr=[]
            for c in cs:
                cats=tuple(items[i]['slot']+':'+items[i]['category'] for i in c)
                if cats not in rule_cache:
                    tokens=set(cats)
                    rule_cache[cats]=max((r['confidence'] for r in rules if set(r['antecedents'])|set(r['consequents'])<=tokens),default=0)
                rr.append(rule_cache[cats])
            rule_scores.append(rr)
        flat=[c for cs in groups for c in cs]
        return {'groups':groups,'truths':truths,'truth_indices':np.array(truth_indices),
                'x':features(np.array([[index[i] for i in c] for c in flat])), 'rules':np.array(rule_scores),
                'fingerprint':hashlib.sha256(json.dumps(groups).encode()).hexdigest()}

    val=candidates('validation',len(positives['validation']))

    def fast_metrics(prepared,scores):
        order=np.argsort(-scores.reshape(-1,100),axis=1,kind='stable')[:,:5]
        hit=order==prepared['truth_indices'][:,None]
        return {'hit_at_5':float(hit.any(1).mean()), 'mrr_at_5':float((hit/np.arange(1,6)).sum(1).mean())}

    def assess(prepared,scores,alpha=1.,lam=1.,lo=0,hi=None):
        ss=scores.reshape(-1,100)
        n=len(ss) if hi is None else hi
        hits,rr,div=[],[],[]
        for i in range(lo,n):
            cs=prepared['groups'][i]
            selected=mmr(cs,alpha*ss[i]+(1-alpha)*prepared['rules'][i],5,lam)
            truth=int(prepared['truth_indices'][i])
            hits.append(int(truth in selected));rr.append(1/(selected.index(truth)+1) if truth in selected else 0)
            outfits=[cs[j] for j in selected]
            div.append(1-np.mean([sum(x==y for x,y in zip(a[1:],b[1:]))/2 for a,b in itertools.combinations(outfits,2)]))
        return {'hit_at_5':float(np.mean(hits)), 'mrr_at_5':float(np.mean(rr)), 'diversity_at_5':float(np.mean(div)), 'queries':len(hits), 'hits':hits}

    pools={s:np.array([index[i] for i in ids if items[i]['split']=='train' and items[i]['slot']==s]) for s in ('top','bottom','shoes')}
    neighbors={}
    for slot,pool in pools.items():
        knn=NearestNeighbors(n_neighbors=min(21,len(pool)),metric='cosine',n_jobs=4).fit(vectors[pool].numpy())
        unique=np.unique(train[:,('top','bottom','shoes').index(slot)])
        for iid,nn in zip(unique,knn.kneighbors(vectors[unique].numpy(),return_distance=False)):
            neighbors[int(iid)]=pool[nn][pool[nn]!=iid]
    runs=[]; summaries=[]
    for cid,config in enumerate(configs):
        states=[]; selected_runs=[]
        for seed in plan['seeds']:
            torch.manual_seed(seed);rng=np.random.default_rng(seed)
            model=RankerV2(config)
            opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=config['weight_decay'])
            best_key=(-1.,-1.); best_state=None; stale=0; history=[]
            for epoch in range(1,plan['max_epochs']+1):
                model.train(); losses=[]
                permutation=rng.permutation(len(train))
                for begin in range(0,len(train),256):
                    pos=train[permutation[begin:begin+256]];neg=pos.copy()
                    for j,triple in enumerate(pos):
                        for attempt in range(100):
                            slot=int(rng.integers(3))
                            pool=neighbors[int(triple[slot])] if rng.random()<config['hard_fraction'] else pools[('top','bottom','shoes')[slot]]
                            replacement=int(rng.choice(pool));trial=triple.copy();trial[slot]=replacement
                            if replacement!=triple[slot] and tuple(trial) not in known:
                                neg[j]=trial;break
                        else: raise RuntimeError('Negative sampling failed')
                    opt.zero_grad()
                    loss=torch.nn.functional.softplus(model(features(neg))-model(features(pos))).mean()
                    loss.backward();opt.step();losses.append(loss.item())
                model.eval()
                metrics=fast_metrics(val,predict([model],val['x']))
                history.append({'epoch':epoch,'loss':float(np.mean(losses)),**metrics})
                key=(metrics['hit_at_5'],metrics['mrr_at_5'])
                if key>best_key:
                    best_key=key;best_state=copy.deepcopy(model.state_dict());best_epoch=epoch;stale=0
                else: stale+=1
                if epoch>=10 and stale>=plan['patience']:break
            run={'config_id':cid,'seed':seed,'epoch':best_epoch,'hit_at_5':best_key[0],'mrr_at_5':best_key[1],'history':history}
            runs.append(run);selected_runs.append(run);states.append(best_state)
            print(json.dumps({k:v for k,v in run.items() if k!='history'}),flush=True)
        bundle={'config':config,'states':states,'embedding_sha256':digest,'seeds':plan['seeds']}
        torch.save(bundle,out/f'config_{cid}.pt')
        summary={'config_id':cid,'config':config,'mean_hit_at_5':float(np.mean([r['hit_at_5'] for r in selected_runs])),
                 'std_hit_at_5':float(np.std([r['hit_at_5'] for r in selected_runs])), 'mean_mrr_at_5':float(np.mean([r['mrr_at_5'] for r in selected_runs]))}
        summaries.append(summary)
        (out/'progress.json').write_text(json.dumps({'status':'training','runs':runs,'summaries':summaries},indent=2))
    chosen_config=max(summaries,key=lambda r:(r['mean_hit_at_5'],r['mean_mrr_at_5']))
    models,bundle=load_ensemble(out/f"config_{chosen_config['config_id']}.pt",digest)
    torch.save(bundle,out/'ensemble.pt')
    old=OutfitRanker(ROOT/'artifacts/outfit_ranker/best.pt',digest)
    version='image_'+digest[:12]
    baseline={name:joblib.load(ROOT/f'artifacts/compatibility/{version}/{name}.joblib') for name in ('logistic_regression','random_forest')}
    def score_all(prepared):
        result={'ensemble':predict(models,prepared['x']), 'previous_mlp':old.predict_proba(prepared['x'].numpy())[:,1]}
        cats=[feature_row(c,items) for cs in prepared['groups'] for c in cs]
        for name,b in baseline.items():
            assert b['feature_version']==version
            x=np.hstack((b['vectorizer'].transform(cats),prepared['x'].numpy()))
            result[name]=b['model'].predict_proba(x)[:,1]
        return result
    val_scores=score_all(val)
    choices=[]
    for name,score in val_scores.items():
        for a in (1.,.75,.5):
            for lam in (1.,.9,.7):
                m=assess(val,score,a,lam);m.pop('hits')
                choices.append({'model':name,'alpha':a,'lambda':lam,**m})
    selected=max(choices,key=lambda r:r['hit_at_5']+.05*r['diversity_at_5'])
    selection={'chosen_config':chosen_config,'selected':selected,'validation_search':choices,'embedding_sha256':digest}
    (out/'selection.json').write_text(json.dumps(selection,indent=2))
    print('FROZEN_SELECTION '+json.dumps(selected),flush=True)
    # No more selection after this point. Evaluate the old 100 and next 500 queries.
    test=candidates('test',600)
    assert not set(test['truths'][:100])&set(test['truths'][100:])
    test_scores=score_all(test)
    old_choice=json.loads((ROOT/'reports/metrics/ranking_mlp.json').read_text())['chosen']
    settings={'selected':(selected['model'],selected['alpha'],selected['lambda']),
              'previous_mlp':('previous_mlp',old_choice['alpha'],old_choice['lambda']),
              'original_rf':('random_forest',1.,1.)}
    # LR and ensemble use their own validation-optimal reranking, never test selection.
    for name in ('logistic_regression','ensemble'):
        c=max((r for r in choices if r['model']==name),key=lambda r:r['hit_at_5']+.05*r['diversity_at_5'])
        settings[name]=(name,c['alpha'],c['lambda'])
    evaluations={}
    for name,(model,a,lam) in settings.items():
        evaluations[name]={'settings':{'model':model,'alpha':a,'lambda':lam},
            'legacy100':assess(test,test_scores[model],a,lam,0,100),
            'confirmation500':assess(test,test_scores[model],a,lam,100,600)}
    # Confirm exact legacy reproduction before interpreting the additional queries.
    prior=json.loads((ROOT/'reports/metrics/ranking_mlp.json').read_text())['test_selected']
    for metric in ('hit_at_5','mrr_at_5','diversity_at_5'):
        assert np.isclose(evaluations['previous_mlp']['legacy100'][metric],prior[metric])
    delta=np.array(evaluations['selected']['confirmation500']['hits'])-np.array(evaluations['previous_mlp']['confirmation500']['hits'])
    rng=np.random.default_rng(2026)
    boot=np.array([rng.choice(delta,len(delta),replace=True).mean() for _ in range(5000)])
    report={'plan':plan,'runs':runs,'config_summary':summaries,**selection,'test':evaluations,
            'paired_hit_difference_confirmation':{'difference':float(delta.mean()),'bootstrap95':np.quantile(boot,[.025,.975]).tolist(), 'note':'Query bootstrap; shared items may make intervals optimistic.'},
            'candidate_hashes':{'validation':val['fingerprint'],'test600':test['fingerprint']},
            'elapsed_seconds':time.perf_counter()-start_time, 'status':'completed'}
    report_path.write_text(json.dumps(report,indent=2))
    (out/'progress.json').write_text(json.dumps({'status':'completed','report':str(report_path)},indent=2))
    print(json.dumps({'selected':selected,'test':{k:{s:{m:v for m,v in d.items() if m!='hits'} if isinstance(d,dict) else d for s,d in value.items()} for k,value in evaluations.items()},'difference':report['paired_hit_difference_confirmation'],'seconds':report['elapsed_seconds']},indent=2),flush=True)


if __name__=='__main__':main()
