"""M3-only tiny-overfit check using the existing shared trainer and data splits.

Run from the team repository root:
    python -m src.modern_gate

This sanity gate is NOT a tuning trial and must not be included in CER model
comparison tables. Its learning rate is intentionally 3e-4: the previous
Transformer gate at 5e-3 failed to memorize 4 examples under seed 403.
No test split or labels are accessed.
"""

import json
from pathlib import Path

from src.train import ROOT, run_experiment, split_fingerprint


def main():
    config = json.loads((ROOT / 'configs' / 'modern.json').read_text(encoding='utf-8'))
    config.update(
        model='modern',
        augmentation=False,
        dropout=0.0,
        learning_rate=0.0003,
        epochs=300,
        patience=300,
        batch_size=4,
        seed=403,
        tiny_overfit_gate=True,
    )
    run_dir = ROOT / 'experiments' / 'modern_results' / 'gates' / 'tiny_overfit_stable_lr_0p0003'
    summary_path = run_dir / 'run_summary.json'
    if summary_path.is_file():
        previous = json.loads((run_dir / 'config.json').read_text(encoding='utf-8'))
        if previous != config:
            raise ValueError('Existing M3 gate has a different config; do not mix results.')
        result = json.loads(summary_path.read_text(encoding='utf-8'))
        if result.get('split_sha256') != split_fingerprint():
            raise ValueError('Frozen data split fingerprint changed.')
        if not Path(result['checkpoint_path']).is_file():
            raise FileNotFoundError('Gate checkpoint is missing; do not treat as a complete run.')
        print('Reusing completed M3 gate evidence:', run_dir)
    else:
        if run_dir.exists():
            raise RuntimeError(f'Incomplete gate run; inspect it before retrying: {run_dir}')
        result = run_experiment(config, run_dir=run_dir, tiny_overfit=True)
    if result['val_exact_match'] != 1.0:
        raise AssertionError('M3 tiny-overfit failed: fewer than 4/4 TRAIN labels memorized')
    print('PASS: ModernCTC memorized all four TRAIN examples.')


if __name__ == '__main__':
    main()
