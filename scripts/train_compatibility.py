"""LR/RF experiments with metadata baseline or a versioned image embedding file.

Evaluation uses project's synthetic-negative triple protocol, not official AUC.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import joblib
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).absolute().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embeddings', type=Path)
    args = parser.parse_args()
    data = ROOT / 'data/processed/polyvore'
    items = json.loads((data / 'items.json').read_text())
    rows = {s: json.loads((data / f'{s}_examples.json').read_text()) for s in ('train', 'validation', 'test')}
    def metadata_features(r):
        cats = [items[i]['category'] for i in r['item_ids']]
        return {'top': cats[0], 'bottom': cats[1], 'shoes': cats[2],
                'top_bottom': cats[0] + '|' + cats[1], 'top_shoes': cats[0] + '|' + cats[2],
                'bottom_shoes': cats[1] + '|' + cats[2]}
    vectorizer = DictVectorizer(sparse=False)
    vectorizer.fit([metadata_features(r) for r in rows['train']])
    features = {s: vectorizer.transform([metadata_features(r) for r in rs]) for s, rs in rows.items()}
    version = 'metadata_baseline'
    if args.embeddings:
        archive = np.load(args.embeddings, allow_pickle=False)
        em = dict(zip(archive['ids'].astype(str), archive['vectors']))
        for split, rs in rows.items():
            vectors = []
            for r in rs:
                a, b, c = [em[i] for i in r['item_ids']]
                vectors.append(np.concatenate([a, b, c, a*b, a*c, b*c]))
            features[split] = np.hstack([features[split], vectors])
        version = 'image_' + hashlib.sha256(args.embeddings.read_bytes()).hexdigest()[:12]
    labels = {s: np.array([r['label'] for r in rs]) for s, rs in rows.items()}
    best = {}
    validation_results = []
    for family in ('logistic_regression', 'random_forest'):
        for parameter in ([0.1, 1.0, 10.0] if family == 'logistic_regression' else [8, 16, None]):
            if family == 'logistic_regression':
                model = make_pipeline(StandardScaler(), LogisticRegression(C=parameter, max_iter=1500, random_state=42))
            else:
                model = RandomForestClassifier(n_estimators=200, max_depth=parameter, min_samples_leaf=3,
                                               random_state=42, n_jobs=4)
            start = time.perf_counter()
            model.fit(features['train'], labels['train'])
            score = model.predict_proba(features['validation'])[:, 1]
            auc = float(roc_auc_score(labels['validation'], score))
            thresholds = np.linspace(0.1, 0.9, 33)
            threshold = float(max(thresholds, key=lambda t: f1_score(labels['validation'], score >= t)))
            validation_results.append({'family': family, 'parameter': parameter, 'validation_auc': auc,
                                       'threshold': threshold, 'seconds': time.perf_counter() - start})
            if family not in best or auc > best[family]['auc']:
                best[family] = {'model': model, 'auc': auc, 'threshold': threshold, 'parameter': parameter}
            print(validation_results[-1], flush=True)
    out = ROOT / 'artifacts/compatibility' / version
    out.mkdir(parents=True, exist_ok=True)
    results = {'feature_version': version, 'protocol': 'project triples with synthetic same-slot negatives',
               'counts': {s: len(rs) for s, rs in rows.items()}, 'validation_search': validation_results, 'test': {}}
    for family, value in best.items():
        score = value['model'].predict_proba(features['test'])[:, 1]
        pred = score >= value['threshold']
        results['test'][family] = {'auc': float(roc_auc_score(labels['test'], score)),
                                  'f1': float(f1_score(labels['test'], pred)),
                                  'accuracy': float(accuracy_score(labels['test'], pred)),
                                  'confusion_matrix': confusion_matrix(labels['test'], pred).tolist()}
        joblib.dump({'model': value['model'], 'vectorizer': vectorizer, 'threshold': value['threshold'],
                     'feature_version': version}, out / f'{family}.joblib')
    rp = ROOT / 'reports/metrics'
    rp.mkdir(parents=True, exist_ok=True)
    (rp / f'compatibility_{version}.json').write_text(json.dumps(results, indent=2))
    print(json.dumps(results['test'], indent=2))


if __name__ == '__main__':
    main()
