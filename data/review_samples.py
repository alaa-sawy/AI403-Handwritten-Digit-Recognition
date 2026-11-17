"""AI403 / M1 Part 2.3: produce a stratified training-only visual review gallery.

Copy to AI403/data/review_samples.py, run from AI403 root:
    python data/review_samples.py

Reads existing data/audit_outputs/data_audit_images.csv and data/raw/ORAND-CAR-2014.tar.gz.
Outputs two PNG review galleries and one editable CSV in data/audit_outputs/.
NO test images or test ground-truth files are read.
"""
from collections import Counter, defaultdict
import csv
from io import BytesIO
from pathlib import Path
import random
import tarfile

try:
    from PIL import Image, ImageDraw, ImageFont, ImageOps
except ImportError as exc:
    raise SystemExit("Please install Pillow first: python -m pip install pillow") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = PROJECT_ROOT / "data" / "raw" / "ORAND-CAR-2014.tar.gz"
AUDIT_CSV = PROJECT_ROOT / "data/audit_outputs" / "data_audit_images.csv"
OUTPUT = PROJECT_ROOT / "data/audit_outputs"
SAMPLE_SIZE = 40
SEED = 403

if not ARCHIVE.is_file():
    raise SystemExit(f"Archive missing: {ARCHIVE}")
if not AUDIT_CSV.is_file():
    raise SystemExit(f"Audit CSV missing: {AUDIT_CSV}")

with AUDIT_CSV.open("r", encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

required = {"source", "filename", "label", "mode", "sha256", "width", "height", "status"}
if not rows or not required.issubset(rows[0]):
    raise SystemExit("Audit CSV is missing required columns; rerun data/audit_images.py")
if any(row["status"] != "OK" for row in rows):
    raise SystemExit("Some image audit statuses are not OK. Resolve these first.")
if any(row["source"] not in ("CAR-A", "CAR-B") for row in rows):
    raise SystemExit("Unknown source in audit CSV. Refusing to read possible test data.")

rng = random.Random(SEED)
selected = {}  # (source, filename) -> row + selection reasons

def choose(row, reason):
    key = (row["source"], row["filename"])
    if key not in selected:
        selected[key] = {**row, "reasons": []}
    if reason not in selected[key]["reasons"]:
        selected[key]["reasons"].append(reason)

# Include both sources and all available sequence lengths in the review.
for source in ("CAR-A", "CAR-B"):
    for length in sorted({len(r["label"]) for r in rows}):
        group = [r for r in rows if r["source"] == source and len(r["label"]) == length]
        for r in rng.sample(group, min(2, len(group))):
            choose(r, f"length_{length}")

# Include any non-RGB image; important for alpha-aware preprocessing.
for r in rows:
    if r["mode"] != "RGB":
        choose(r, "non_RGB_mode")

# Include all exact duplicate images (only 1 group expected in these data).
by_hash = defaultdict(list)
for r in rows:
    by_hash[r["sha256"]].append(r)
for group in by_hash.values():
    if len(group) > 1:
        for r in group:
            choose(r, "exact_duplicate")

# Include extreme widths/heights that might cause aspect-ratio issues.
for field in ("width", "height"):
    for r in (min(rows, key=lambda r: int(r[field])),
              max(rows, key=lambda r: int(r[field]))):
        choose(r, f"extreme_{field}")

# Fill to a fixed size with a deterministic, balanced random sample.
if len(selected) > SAMPLE_SIZE:
    raise SystemExit(f"Special cases ({len(selected)}) exceed SAMPLE_SIZE={SAMPLE_SIZE}")
for source in ("CAR-A", "CAR-B"):
    candidates = [r for r in rows if r["source"] == source and
                  (r["source"], r["filename"]) not in selected]
    rng.shuffle(candidates)
    target = SAMPLE_SIZE // 2
    already = sum(item["source"] == source for item in selected.values())
    for r in candidates[:max(0, target - already)]:
        choose(r, "random_coverage")
remaining = [r for r in rows if (r["source"], r["filename"]) not in selected]
rng.shuffle(remaining)
for r in remaining[:SAMPLE_SIZE - len(selected)]:
    choose(r, "random_coverage")

sample = sorted(selected.values(), key=lambda r: (r["source"], len(r["label"]), r["filename"]))
assert len(sample) == SAMPLE_SIZE, (len(sample), SAMPLE_SIZE)

# Build lookup of ONLY TRAIN images; leave official TEST untouched.
request_paths = {}
for i, r in enumerate(sample):
    prefix = "a" if r["source"] == "CAR-A" else "b"
    path = f"ORAND-CAR-2014/{r['source']}/{prefix}_train_images/{r['filename']}"
    request_paths[path] = i

blobs = {}
with tarfile.open(ARCHIVE, "r:gz") as archive:
    for member in archive:
        if member.name in request_paths and member.isfile():
            src = archive.extractfile(member)
            if src is None:
                raise RuntimeError(f"Cannot read {member.name}")
            blobs[request_paths[member.name]] = src.read()
        if len(blobs) == len(sample):
            break

missing = [sample[i]["filename"] for i in range(len(sample)) if i not in blobs]
if missing:
    raise RuntimeError(f"Images missing from archive: {missing}")

OUTPUT.mkdir(parents=True, exist_ok=True)
review_csv = OUTPUT / "sample_review.csv"
# Re-running should never erase notes already written by the reviewer.
prior_reviews = {}
if review_csv.is_file():
    with review_csv.open("r", encoding="utf-8-sig", newline="") as f:
        for item in csv.DictReader(f):
            prior_reviews[(item.get("source", ""), item.get("filename", ""))] = (
                item.get("review_status", ""), item.get("notes", ""))
with review_csv.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["source", "filename", "label", "label_length", "reason", "review_status", "notes"])
    for r in sample:
        status, notes = prior_reviews.get((r["source"], r["filename"]), ("", ""))
        writer.writerow([r["source"], r["filename"], r["label"], len(r["label"]),
                         ", ".join(r["reasons"]), status, notes])

FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
WIN_FONT_PATH = Path("C:/Windows/Fonts/arial.ttf")
font_file = str(FONT_PATH if FONT_PATH.exists() else WIN_FONT_PATH)
try:
    font = ImageFont.truetype(font_file, 17)
    small_font = ImageFont.truetype(font_file, 14)
    title_font = ImageFont.truetype(font_file, 23)
except OSError:
    font = ImageFont.load_default()
    small_font = ImageFont.load_default()
    title_font = ImageFont.load_default()

# 2 contact sheets of 20 photos each, preserving all image pixels and labels.
CELL_W, CELL_H, COLS, ROWS = 350, 172, 4, 5
for page in range(2):
    canvas = Image.new("RGB", (COLS * CELL_W, 54 + ROWS * CELL_H), "#f4f7fa")
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 15), f"M1 / Training visual audit - page {page+1}/2", fill="#13293d", font=title_font)
    for local_index, r in enumerate(sample[page * 20:(page + 1) * 20]):
        i = page * 20 + local_index
        x = (local_index % COLS) * CELL_W
        y = 54 + (local_index // COLS) * CELL_H
        draw.rounded_rectangle((x + 5, y + 4, x + CELL_W - 5, y + CELL_H - 5),
                               radius=12, fill="white", outline="#d9e5ef", width=1)
        # Preserve transparent strokes against white background (do not composite on black).
        with Image.open(BytesIO(blobs[i])) as original:
            rgba = original.convert("RGBA")
            base = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
            base.alpha_composite(rgba)
            rgb = base.convert("RGB")
            thumb = ImageOps.contain(rgb, (CELL_W - 30, 114), method=Image.Resampling.LANCZOS)
        canvas.paste(thumb, (x + (CELL_W - thumb.width) // 2, y + 9 + (114 - thumb.height) // 2))
        draw.text((x + 14, y + 125), f"Label: {r['label']}", fill="#172b41", font=font)
        draw.text((x + 14, y + 149), f"{r['source']}  |  {r['filename']}", fill="#64748b", font=small_font)
    dest = OUTPUT / f"sample_review_{page+1:02d}.png"
    canvas.save(dest)
    print("Saved gallery:", dest)

print("Saved review checklist:", review_csv)
print("Total selected:", len(sample))
print("By source:", dict(Counter(x["source"] for x in sample)))
print("By label length:", dict(sorted(Counter(len(x["label"]) for x in sample).items())))
print("Special mode images:", sum(x["mode"] != "RGB" for x in sample))
print("Test set untouched. Manual image/label review still required.")

