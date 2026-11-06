"""Export vectors tied to a trained checkpoint; reusable by compatibility training."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from train_embedding import ROOT, Encoder, transform


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--output', type=Path, default=ROOT / 'artifacts/embedding/catalog_vectors.npz')
    args = p.parse_args()
    torch.set_num_threads(4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Encoder(pretrained=False).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=False)['model'])
    model.eval()
    items = json.loads((ROOT / 'data/processed/polyvore/items.json').read_text())
    # All items used by examples, across the official item-disjoint splits.
    required = set()
    for split in ('train', 'validation', 'test'):
        rows = json.loads((ROOT / f'data/processed/polyvore/{split}_examples.json').read_text())
        required.update(i for r in rows for i in r['item_ids'])
    ids = sorted(required)
    vectors = []
    tfm = transform()
    with torch.no_grad():
        for start in range(0, len(ids), args.batch_size):
            batch = []
            for iid in ids[start:start+args.batch_size]:
                with Image.open(ROOT / items[iid]['image_path']) as image:
                    batch.append(tfm(image.convert('RGB')))
            vectors.append(model(torch.stack(batch).to(device)).cpu().numpy())
            if start % (args.batch_size*50) == 0: print(f'Encoded {start+len(batch)}/{len(ids)}', flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, ids=np.array(ids), vectors=np.concatenate(vectors))
    args.output.with_suffix('.json').write_text(json.dumps({'checkpoint': str(args.checkpoint),
        'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(), 'count': len(ids)}, indent=2))


if __name__ == '__main__': main()
