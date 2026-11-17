from pathlib import Path
from collections import Counter, defaultdict
from io import BytesIO
from hashlib import sha256
import csv
import tarfile

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'data' / 'raw' / 'ORAND-CAR-2014.tar.gz'
OUTPUT_DIR = ROOT / 'data/audit_outputs'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_FILES = {
    'CAR-A': ('ORAND-CAR-2014/CAR-A/a_train_images/',
              'ORAND-CAR-2014/CAR-A/a_train_gt.txt'),
    'CAR-B': ('ORAND-CAR-2014/CAR-B/b_train_images/',
              'ORAND-CAR-2014/CAR-B/b_train_gt.txt'),
}

image_rows = []
labels = {}

# Read each archive member once, in archive order.
with tarfile.open(ARCHIVE, 'r:gz') as archive:
    for member in archive:
        if not member.isfile():
            continue

        for source, (image_prefix, gt_path) in TRAIN_FILES.items():
            if member.name == gt_path:
                content = archive.extractfile(member).read().decode('utf-8-sig')
                labels[source] = dict(
                    line.strip().split('\t')
                    for line in content.splitlines() if line.strip()
                )
                break

            if not (member.name.startswith(image_prefix) and
                    member.name.lower().endswith('.png')):
                continue

            filename = Path(member.name).name
            data = archive.extractfile(member).read()
            record = {
                'source': source, 'filename': filename,
                'label': '', 'width': '', 'height': '',
                'mode': '', 'sha256': sha256(data).hexdigest(),
                'status': 'OK',
            }
            try:
                with Image.open(BytesIO(data)) as image:
                    image.load()  # Full decode: catches damaged images
                    record['width'], record['height'] = image.size
                    record['mode'] = image.mode
                    if image.width == 0 or image.height == 0:
                        raise ValueError('Zero image dimension')
            except Exception as exc:
                record['status'] = f'ERROR: {type(exc).__name__}: {str(exc)[:80]}'
            image_rows.append(record)
            break

# Join image information with the official TRAIN ground truths.
for row in image_rows:
    row['label'] = labels[row['source']].get(row['filename'], '')

image_rows.sort(key=lambda x: (x['source'], x['filename']))
lengths = Counter(len(row['label']) for row in image_rows if row['label'])
source_counts = Counter(row['source'] for row in image_rows)
unique_labels = len(set(row['label'] for row in image_rows if row['label']))
corrupt = [row for row in image_rows if row['status'] != 'OK']
missing = [row for row in image_rows if not row['label']]

# Same image bytes under two filenames = exact duplicate.
by_hash = defaultdict(list)
for row in image_rows:
    by_hash[row['sha256']].append(row)
duplicate_groups = [g for g in by_hash.values() if len(g) > 1]
conflicting_duplicates = [g for g in duplicate_groups
                          if len({r['label'] for r in g}) > 1]

csv_path = OUTPUT_DIR / 'data_audit_images.csv'
columns = ['source', 'filename', 'label', 'width', 'height', 'mode', 'sha256', 'status']
with csv_path.open('w', encoding='utf-8-sig', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    writer.writerows(image_rows)

summary = [
    'ORAND-CAR: TRAIN IMAGE AUDIT',
    f'Train images: {len(image_rows)}',
    f'CAR-A images: {source_counts["CAR-A"]}',
    f'CAR-B images: {source_counts["CAR-B"]}',
    f'Images with unreadable content: {len(corrupt)}',
    f'Images missing labels: {len(missing)}',
    f'Unique TRAIN labels (A+B combined): {unique_labels}',
    f'Label length distribution: {dict(sorted(lengths.items()))}',
    f'Exact duplicate image groups: {len(duplicate_groups)}',
    f'Identical image groups with conflicting labels: {len(conflicting_duplicates)}',
    'Official TEST data and labels were not opened.',
    f'Results CSV: {csv_path}',
]
summary_path = OUTPUT_DIR / 'image_audit_summary.txt'
summary_path.write_text('\n'.join(summary) + '\n', encoding='utf-8')
print('\n'.join(summary))

if corrupt or missing or conflicting_duplicates:
    print('\nREVIEW REQUIRED: some audit checks failed.')
else:
    print('\nPASS: all TRAIN images open and have labels; no conflicting exact duplicates.')

