"""ResNet-50 triplet training on the local CPU or CUDA; checkpoint each epoch.

Same-item augmentation positives measure instance retrieval, not human style.
--limit makes an explicit pilot run. Only projection + layer4 are fine-tuned.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image

ROOT = Path(__file__).absolute().parents[1]
os.environ.setdefault('TORCH_HOME', str(ROOT / 'artifacts/torch_cache'))


class Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 128)
        for name, p in self.backbone.named_parameters():
            p.requires_grad = name.startswith(('layer4.', 'fc.'))

    def forward(self, x):
        return F.normalize(self.backbone(x), dim=1)


def transform(augment=False):
    ops = ([transforms.RandomResizedCrop(160, scale=(0.75, 1.0)), transforms.RandomHorizontalFlip()]
           if augment else [transforms.Resize((160, 160))])
    return transforms.Compose(ops + [transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


class Triplets(Dataset):
    def __init__(self, items, split, limit=0):
        self.items = [v for v in items.values() if v['split'] == split]
        random.Random(42).shuffle(self.items)
        if limit:
            self.items = self.items[:limit]
        self.pools = {s: [r for r in self.items if r['slot'] == s] for s in ('top', 'bottom', 'shoes')}
        self.aug = transform(True)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        row = self.items[idx]
        candidates = self.pools[row['slot']]
        if len(candidates) < 2:
            raise ValueError('At least two items per slot required')
        negative = random.choice(candidates)
        while negative['item_id'] == row['item_id']:
            negative = random.choice(candidates)
        with Image.open(ROOT / row['image_path']) as im:
            a, p = self.aug(im.convert('RGB')), self.aug(im.convert('RGB'))
        with Image.open(ROOT / negative['image_path']) as im:
            n = self.aug(im.convert('RGB'))
        return a, p, n


@torch.no_grad()
def evaluate(model, items, split, device, limit=300):
    selected = [v for v in items.values() if v['split'] == split]
    random.Random(123).shuffle(selected)
    selected = selected[:limit]
    gallery, queries = [], []
    model.eval()
    deterministic = transform()
    # Fixed mirror query for a reproducible, explicitly limited instance task.
    for start in range(0, len(selected), 16):
        images, flipped = [], []
        for row in selected[start:start+16]:
            with Image.open(ROOT / row['image_path']) as im:
                image = im.convert('RGB')
                images.append(deterministic(image))
                flipped.append(deterministic(image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)))
        gallery.append(model(torch.stack(images).to(device)).cpu())
        queries.append(model(torch.stack(flipped).to(device)).cpu())
    sims = torch.cat(queries) @ torch.cat(gallery).T
    for i, r in enumerate(selected):
        sims[i, torch.tensor([s['slot'] != r['slot'] for s in selected])] = -float('inf')
    ranks = sims.argsort(dim=1, descending=True)
    truth = torch.arange(len(selected))[:, None]
    return {'recall_at_1': float((ranks[:, :1] == truth).any(1).float().mean()),
            'recall_at_5': float((ranks[:, :5] == truth).any(1).float().mean()),
            'queries': len(selected), 'protocol': 'same-item horizontal-flip retrieval, same-slot gallery'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--max-steps', type=int, default=0)
    p.add_argument('--margin', type=float, default=1.0)
    p.add_argument('--resume', type=Path)
    p.add_argument('--run-name', default='full')
    args = p.parse_args()
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.set_num_threads(4)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    items = json.loads((ROOT / 'data/processed/polyvore/items.json').read_text())
    ds = Triplets(items, 'train', args.limit)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
    model = Encoder(pretrained=not bool(args.resume)).to(device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    start_epoch, best = 0, -1.0
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        start_epoch = state['epoch']; best = state['best']
    out = ROOT / 'artifacts/embedding' / args.run_name
    out.mkdir(parents=True, exist_ok=True)
    report = {'device': str(device), 'train_items': len(ds), 'pilot': bool(args.limit or args.max_steps),
              'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, 'epochs': []}
    loss_fn = nn.TripletMarginLoss(margin=args.margin)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        # Freeze BN running statistics for reliable small CPU batches.
        for module in model.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()
        start, total, steps = time.perf_counter(), 0.0, 0
        for step, (a, pos, neg) in enumerate(loader):
            optimizer.zero_grad()
            encoded = model(torch.cat([a, pos, neg]).to(device))
            loss = loss_fn(*encoded.chunk(3))
            if not torch.isfinite(loss): raise ValueError('Nonfinite loss')
            loss.backward(); optimizer.step()
            total += loss.item(); steps += 1
            if steps == 1 or steps % 20 == 0:
                print(f'epoch={epoch+1} step={steps}/{len(loader)} loss={loss.item():.4f} seconds={time.perf_counter()-start:.1f}', flush=True)
            if steps % 100 == 0:
                torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                            'epoch': epoch, 'best': best, 'config': report['arguments'],
                            'input_size': 160, 'embedding_dim': 128,
                            'partial_epoch_steps': steps, 'resume_policy': 'restart_current_epoch'}, out / 'last.pt')
            if args.max_steps and steps >= args.max_steps: break
        if not steps: raise ValueError('No training steps')
        validation = evaluate(model, items, 'validation', device)
        improved = validation['recall_at_1'] > best
        best = max(best, validation['recall_at_1'])
        state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch+1,
                 'best': best, 'config': report['arguments'], 'input_size': 160, 'embedding_dim': 128}
        torch.save(state, out / 'last.pt')
        if improved: torch.save(state, out / 'best.pt')
        row = {'epoch': epoch+1, 'steps': steps, 'mean_loss': total/steps, 'seconds': time.perf_counter()-start, 'validation': validation}
        report['epochs'].append(row)
        (out / 'training_report.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(row), flush=True)
    if not (out / 'best.pt').exists():
        raise ValueError('No completed epoch checkpoint in this run; resume with a larger epoch target')
    model.load_state_dict(torch.load(out / 'best.pt', map_location=device, weights_only=False)['model'])
    report['test'] = evaluate(model, items, 'test', device)
    (out / 'training_report.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
