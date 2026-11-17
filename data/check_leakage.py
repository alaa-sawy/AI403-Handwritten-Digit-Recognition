"""M1 Part 2.4: exact-duplicate checks BEFORE creating train/validation.

Reads TRAIN SHA-256 values from data/audit_outputs/data_audit_images.csv.
Computes SHA-256 of official TEST image BYTES solely for duplicate checking.
Does NOT read test labels, preview test images, train models, or tune settings.

Output: data/audit_outputs/duplicate_groups.csv and data/audit_outputs/leakage_check_summary.txt.
"""
from collections import defaultdict
from pathlib import Path
import csv
import hashlib
import tarfile

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'data/audit_outputs' / 'data_audit_images.csv'
ARCHIVE = ROOT / 'data' / 'raw' / 'ORAND-CAR-2014.tar.gz'
OUT = ROOT / 'data/audit_outputs'

if not AUDIT.is_file():
    raise FileNotFoundError('Missing training audit CSV: ' + str(AUDIT))
if not ARCHIVE.is_file():
    raise FileNotFoundError('Missing ORAND archive: ' + str(ARCHIVE))

# We use the training audit produced in Part 2.2. The label remains a string.
with AUDIT.open('r', encoding='utf-8-sig', newline='') as handle:
    training = list(csv.DictReader(handle))

required = {'source', 'filename', 'label', 'sha256', 'status'}
if not training or not required.issubset(training[0].keys()):
    raise ValueError('Audit CSV is empty or missing columns: ' + repr(required))
if len(training) != 5009:
    raise ValueError(f'Expected 5009 training rows, got {len(training)}')
if any(row['status'].strip().upper() != 'OK' or not row['sha256'].strip() for row in training):
    raise ValueError('Training audit contains an invalid/missing SHA256 row')

training_by_hash = defaultdict(list)
for row in training:
    training_by_hash[row['sha256']].append(row)

# TRAIN-TRAIN exact duplicates and label disagreements.
train_duplicates = {h: rows for h, rows in training_by_hash.items() if len(rows) > 1}
conflicting = {h: rows for h, rows in train_duplicates.items()
               if len({r['label'] for r in rows}) > 1}

# Check image byte hashes only for official TEST. No GT file is opened.
test_images = 0
cross_split_matches = []
with tarfile.open(ARCHIVE, 'r:gz') as archive:
    for member in archive:
        if not member.isfile() or not member.name.lower().endswith(('.png','.jpg','.jpeg')):
            continue
        if '/a_test_images/' not in member.name and '/b_test_images/' not in member.name:
            continue
        stream = archive.extractfile(member)
        if stream is None:
            raise OSError('Unreadable test image in archive: ' + member.name)
        content_hash = hashlib.sha256(stream.read()).hexdigest()
        test_images += 1
        if content_hash in training_by_hash:
            for row in training_by_hash[content_hash]:
                cross_split_matches.append((row['source'], row['filename'], member.name))

# Freeze evidence: record every exact TRAIN duplicate group.
OUT.mkdir(parents=True, exist_ok=True)
duplicate_csv = OUT / 'duplicate_groups.csv'
with duplicate_csv.open('w', newline='', encoding='utf-8-sig') as handle:
    writer = csv.writer(handle)
    writer.writerow(['duplicate_group','source','filename','label','sha256'])
    for i, (digest, rows) in enumerate(sorted(train_duplicates.items()), start=1):
        for row in rows:
            writer.writerow([i, row['source'], row['filename'], row['label'], digest])

summary = [
    'M1 PART 2.4 - EXACT HASH / LEAKAGE AUDIT',
    f'Official TRAIN images: {len(training)}',
    f'Official TEST images hashed only (no labels opened): {test_images}',
    f'Exact duplicate groups inside TRAIN: {len(train_duplicates)}',
    f'Exact TRAIN duplicates with inconsistent labels: {len(conflicting)}',
    f'Exact duplicate image matches across TRAIN and TEST: {len(cross_split_matches)}',
    'WRITER DISJOINTNESS: UNKNOWN (no verified writer_id metadata supplied)',
    'PERCEPTUAL NEAR-DUPLICATES: NOT CHECKED (different bytes may look alike)',
    'ACTION: Keep all identical TRAIN images in the same train/val group; or remove one copy.',
    'ACTION: Keep official TEST sealed for training and hyperparameter selection.',
]
if conflicting:
    summary.append('WARNING: conflicting GT in identical TRAIN images — resolve before splitting.')
if cross_split_matches:
    summary.append('WARNING: TRAIN/TEST exact duplicates detected — investigate before training.')
if not conflicting and not cross_split_matches:
    summary.append('PASS: no conflicting TRAIN labels or exact TRAIN/TEST duplicates.')
summary_path = OUT / 'leakage_check_summary.txt'
summary_path.write_text('\n'.join(summary) + '\n', encoding='utf-8')
print('\n'.join(summary))
print('Detailed duplicate groups:', duplicate_csv)
print('Saved summary:', summary_path)

