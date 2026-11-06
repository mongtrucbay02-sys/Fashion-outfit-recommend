"""Report completion from actual artifacts; pilot outputs never count as full training."""
import json
from pathlib import Path

ROOT = Path(__file__).absolute().parents[1]


def main():
    checks = {}
    for dataset in ('polyvore',):
        path = ROOT / f'data/processed/{dataset}/preprocessing_report.json'
        checks[dataset + '_preprocessed'] = path.is_file()
    for model in ('embedding',):
        path = ROOT / f'artifacts/{model}/full/training_report.json'
        complete = False
        if path.exists():
            report = json.loads(path.read_text())
            complete = not report.get('pilot', True) and bool(report.get('test')) and bool(report.get('epochs'))
        checks[model + '_full_training'] = complete
    checks['association_rules'] = (ROOT / 'artifacts/mining/fpgrowth_rules.json').is_file()
    checks['image_compatibility_evaluated'] = bool(list((ROOT/'reports/metrics').glob('compatibility_image_*.json')))
    checks['image_ranking_evaluated'] = bool(list((ROOT/'reports/metrics').glob('ranking_image_*.json')))
    print(json.dumps({'checks': checks, 'ready_for_result_review': all(checks.values()),
                      'note': 'Review quality and limitations before creating UI; artifact existence is not an accuracy guarantee.'}, indent=2))


if __name__ == '__main__': main()
