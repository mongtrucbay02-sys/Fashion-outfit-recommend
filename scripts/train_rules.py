"""Train association rules only on train; compare identical Apriori/FP-Growth settings."""
import json
from pathlib import Path
import time
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

ROOT = Path(__file__).absolute().parents[1]


def main():
    p = ROOT / 'data/processed/polyvore'
    train = json.loads((p / 'train_transactions.json').read_text())
    encoder = TransactionEncoder()
    frame = pd.DataFrame(encoder.fit_transform(train), columns=encoder.columns_)
    out = ROOT / 'artifacts/mining'
    out.mkdir(parents=True, exist_ok=True)
    report, results = {}, {}
    for name, fn in [('apriori', apriori), ('fpgrowth', fpgrowth)]:
        start = time.perf_counter()
        sets = fn(frame, min_support=0.005, use_colnames=True, max_len=3)
        rules = association_rules(sets, metric='confidence', min_threshold=0.15)
        rules = rules[(rules['lift'] > 1) & rules['consequents'].map(lambda s: len(s) == 1)]
        report[name] = {'seconds': time.perf_counter() - start, 'itemsets': len(sets), 'rules': len(rules)}
        results[name] = set(sets['itemsets'])
        serial = [{'antecedents': sorted(r['antecedents']), 'consequents': sorted(r['consequents']),
                   'support': float(r['support']), 'confidence': float(r['confidence']), 'lift': float(r['lift'])}
                  for _, r in rules.iterrows()]
        (out / f'{name}_rules.json').write_text(json.dumps(serial, indent=2), encoding='utf-8')
        for split in ('validation', 'test'):
            ts = json.loads((p / f'{split}_transactions.json').read_text())
            report[name][split + '_antecedent_coverage'] = sum(any(set(r['antecedents']) <= set(t) for r in serial) for t in ts) / len(ts)
    assert results['apriori'] == results['fpgrowth']
    report['same_frequent_itemsets'] = True
    report['train_transactions'] = len(train)
    report['settings'] = {'min_support': 0.005, 'min_confidence': 0.15, 'max_len': 3}
    rp = ROOT / 'reports/metrics'
    rp.mkdir(parents=True, exist_ok=True)
    (rp / 'rules.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
