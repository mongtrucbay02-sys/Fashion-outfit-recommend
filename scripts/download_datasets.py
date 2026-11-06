"""Download official datasets, resume partial HTTP files and validate assets.

Credentials are read by huggingface_hub from the existing local login only.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
ROOT = Path(__file__).absolute().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def polyvore(reuse):
    from huggingface_hub import HfApi, hf_hub_download
    repo = 'mvasil/polyvore-outfits'
    info = HfApi().dataset_info(repo, files_metadata=True)
    base = ROOT / 'data/raw/polyvore'
    wanted = [s for s in info.siblings if s.rfilename.startswith(('disjoint/', 'data/disjoint/'))
              or s.rfilename in ('categories.csv', 'polyvore_item_metadata.json', 'README.md')]
    records = []
    for s in wanted:
        target = base / s.rfilename
        target.parent.mkdir(parents=True, exist_ok=True)
        old = Path(reuse) / s.rfilename if reuse else None
        if not target.exists() and old and old.is_file() and old.stat().st_size == s.size:
            shutil.copyfile(old, target)
            print(f'Reused {s.rfilename}', flush=True)
        if not target.exists() or target.stat().st_size != s.size:
            hf_hub_download(repo, s.rfilename, repo_type='dataset', revision=info.sha, local_dir=base)
        if target.stat().st_size != s.size:
            raise ValueError(f'Invalid size: {s.rfilename}')
        digest = sha256(target)
        expected_hash = getattr(s.lfs, 'sha256', None) if s.lfs else None
        if expected_hash and digest != expected_hash:
            raise ValueError(f'Checksum mismatch: {s.rfilename}')
        records.append({'path': s.rfilename, 'size': s.size, 'sha256': digest})
        print(f'Validated {s.rfilename}', flush=True)
    (base / 'manifest.json').write_text(json.dumps({'revision': info.sha, 'files': records}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', choices=['polyvore'], required=True)
    p.add_argument('--reuse')
    args = p.parse_args()
    polyvore(args.reuse)
