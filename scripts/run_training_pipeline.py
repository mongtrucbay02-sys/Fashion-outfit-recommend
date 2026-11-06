"""Sequential local CPU workflow. No UI is created by this training pipeline.

Every subprocess writes a separate log, and the status file records real exits.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).absolute().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', choices=['pilot', 'full'], default='full')
    parser.add_argument('--start-at', help='Continue from a named stage; earlier artifacts must already exist')
    parser.add_argument('--resume', type=Path, help='Checkpoint for the first selected image training stage')
    args = parser.parse_args()
    name = args.profile
    limited = name == 'pilot'
    tasks = [
        ('rules', ['scripts/train_rules.py']),
        ('metadata_baseline', ['scripts/train_compatibility.py']),
        ('metadata_ranking', ['scripts/evaluate_recommendation.py']),
        ('embedding', ['scripts/train_embedding.py', '--run-name', name, '--epochs', '1' if limited else '3'] +
                      (['--limit', '256', '--max-steps', '8'] if limited else [])),
    ]
    if not limited:
        tasks.extend([
            ('export_embeddings', ['scripts/export_embeddings.py', '--checkpoint', 'artifacts/embedding/full/best.pt']),
            ('image_compatibility', ['scripts/train_compatibility.py', '--embeddings', 'artifacts/embedding/catalog_vectors.npz']),
            ('image_ranking', ['scripts/evaluate_recommendation.py', '--embeddings', 'artifacts/embedding/catalog_vectors.npz']),
        ])
    if args.start_at:
        names = [name for name, _ in tasks]
        if args.start_at not in names:
            parser.error('Unknown start stage: ' + args.start_at)
        tasks = tasks[names.index(args.start_at):]
    if args.resume:
        if tasks[0][0] != 'embedding' or not args.resume.is_file():
            parser.error('Select an image training stage and an existing checkpoint')
        tasks[0][1].extend(['--resume', str(args.resume.resolve())])
    (ROOT / 'logs').mkdir(exist_ok=True)
    status_path = ROOT / 'reports' / f'pipeline_{name}.json'
    status_path.parent.mkdir(exist_ok=True)
    state = {'profile': name, 'status': 'running', 'steps': [], 'ui_created': False}
    for label, arguments in tasks:
        state['active_step'] = label
        state['updated_at'] = datetime.now(timezone.utc).isoformat()
        status_path.write_text(json.dumps(state, indent=2))
        print(f'START {label}', flush=True)
        mode = 'a' if args.resume and label == tasks[0][0] else 'w'
        with (ROOT / 'logs' / f'{name}_{label}.log').open(mode, encoding='utf-8') as log:
            if mode == 'a':
                print(f'Resuming from {args.resume} at {state["updated_at"]}', file=log, flush=True)
            result = subprocess.run([sys.executable, '-X', 'utf8', '-u', *arguments], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        state['steps'].append({'name': label, 'exit_code': result.returncode})
        if result.returncode:
            state['status'] = 'failed'; status_path.write_text(json.dumps(state, indent=2))
            raise SystemExit(f'Failed {label}; inspect logs/{name}_{label}.log')
        print(f'FINISHED {label}', flush=True)
    state['status'] = 'training_finished_requires_result_review'
    state['active_step'] = None
    state['updated_at'] = datetime.now(timezone.utc).isoformat()
    status_path.write_text(json.dumps(state, indent=2))


if __name__ == '__main__': main()
