"""Standalone preprocessing. Preserve official item-disjoint splits.

Three-slot examples use positive outfit projections and explicitly synthetic
same-slot replacements. They are NOT the official compatibility benchmark.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import itertools
from PIL import Image
import pyarrow.parquet as pq

ROOT = Path(__file__).absolute().parents[1]
SLOTS = ('top', 'bottom', 'shoes')
MAP = {'tops': 'top', 'bottoms': 'bottom', 'shoes': 'shoes'}


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    raw = ROOT / 'data/raw/polyvore'
    out = ROOT / 'data/processed/polyvore'
    out.mkdir(parents=True, exist_ok=True)
    metadata = json.loads((raw / 'polyvore_item_metadata.json').read_text(encoding='utf-8'))
    categories = {}
    with (raw / 'categories.csv').open(encoding='utf-8') as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                categories.setdefault(row[0], row[1])
    items, examples, transactions, report, seen = {}, {}, {}, {}, {}
    image_dir = out / 'images'
    image_dir.mkdir(exist_ok=True)
    reserved_ids = set()
    for split, original in [('train', 'train'), ('validation', 'valid'), ('test', 'test')]:
        original_outfits = json.loads((raw / 'disjoint' / f'{original}.json').read_text())
        source_ids = {str(x['item_id']) for o in original_outfits for x in o['items']}
        overlap = source_ids & reserved_ids
        outfits = [o for o in original_outfits if not any(str(x['item_id']) in reserved_ids for x in o['items'])]
        reserved_ids.update(source_ids)
        wanted = {str(x['item_id']) for o in outfits for x in o['items']}
        seen[split] = wanted
        count = invalid = 0
        for batch in pq.ParquetFile(raw / 'data/disjoint' / f'{split}.parquet').iter_batches(batch_size=64):
            for row in batch.to_pylist():
                iid = str(row['item_id'])
                m = metadata.get(iid, {})
                slot = MAP.get(m.get('semantic_category'))
                if iid not in wanted or slot is None:
                    continue
                path = image_dir / f'{iid}.jpg'
                try:
                    if not path.exists():
                        im = Image.open(io.BytesIO(row['image']['bytes'])).convert('RGB')
                        im.save(path, quality=95)
                    with Image.open(path) as im:
                        im.verify()
                except Exception:
                    invalid += 1
                    continue
                items[iid] = {'item_id': iid, 'slot': slot, 'split': split,
                              'category': categories.get(str(m.get('category_id')), 'unknown'),
                              'description': m.get('description') or m.get('title') or m.get('url_name', ''),
                              'image_path': path.relative_to(ROOT).as_posix()}
                count += 1
        positive, seen_triples = [], set()
        for o in outfits:
            groups = {s: [] for s in SLOTS}
            for x in o['items']:
                iid = str(x['item_id'])
                if iid in items:
                    groups[items[iid]['slot']].append(iid)
            if not all(groups.values()):
                continue
            # One deterministic triple per outfit; avoids multiplying near duplicates.
            ids = tuple(sorted(set(groups[s]))[0] for s in SLOTS)
            if ids not in seen_triples:
                seen_triples.add(ids)
                positive.append({'item_ids': list(ids), 'label': 1,
                                 'source_outfit_id': str(o['set_id']),
                                 'label_source': 'projection_of_curated_outfit'})
        pools = {s: sorted(i for i in wanted if i in items and items[i]['slot'] == s) for s in SLOTS}
        rng = random.Random(42)
        negatives, used_negative = [], set()
        for row in positive:
            for _ in range(50):
                ids = list(row['item_ids'])
                slot_idx = rng.randrange(3)
                ids[slot_idx] = rng.choice(pools[SLOTS[slot_idx]])
                key = tuple(ids)
                if key not in seen_triples and key not in used_negative:
                    negatives.append({'item_ids': ids, 'label': 0,
                                      'source_outfit_id': row['source_outfit_id'],
                                      'label_source': 'synthetic_same_slot_replacement'})
                    used_negative.add(key)
                    break
        examples[split] = positive + negatives
        transactions[split] = [[items[i]['slot'] + ':' + items[i]['category'] for i in r['item_ids']] for r in positive]
        report[split] = {'original_outfits': len(original_outfits), 'retained_outfits': len(outfits),
                         'removed_overlap_outfits': len(original_outfits)-len(outfits),
                         'source_overlap_item_ids': len(overlap), 'usable_images': count, 'invalid_images': invalid,
                         'positive_triples': len(positive), 'synthetic_negatives': len(negatives)}
        print(split, report[split], flush=True)
    for a, b in itertools.combinations(seen, 2):
        if seen[a] & seen[b]:
            raise ValueError(f'Item leakage between {a} and {b}')
    if not all(examples.values()):
        raise ValueError('At least one split has no complete triple')
    dump(out / 'items.json', items)
    for split in examples:
        dump(out / f'{split}_examples.json', examples[split])
        dump(out / f'{split}_transactions.json', transactions[split])
    report['protocol'] = '3-slot projected positives and synthetic negatives; NOT official benchmark'
    report['seed'] = 42
    report['split_policy'] = 'Keep train; exclude whole held-out outfits sharing any item with earlier original splits. Original files unchanged.'
    dump(out / 'preprocessing_report.json', report)


if __name__ == '__main__':
    main()
