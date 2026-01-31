import json
import sys
from pathlib import Path

pred_dir = Path(sys.argv[1]) / "json"
align_path = Path(sys.argv[2])
out_dir = Path(sys.argv[3])

out_dir.mkdir(parents=True, exist_ok=True)

# ---------- LOAD ALIGNMENT ----------
with open(align_path) as f:
    alignment = json.load(f)

# ---------- FIXED DOMAINS ----------
src = "coco"
tgt = "bdd"

print(f"Source domain: {src}")
print(f"Target domain: {tgt}")

# ---------- BUILD ALLOWED MODEL IDS ----------
allowed_model_ids = set()

for cls in alignment.values():
    raw_id = int(cls[src]["id"])   # obj365 raw id
    model_id = raw_id - 1         # SHIFT BY ONE
    allowed_model_ids.add(model_id)

print("Allowed MODEL IDs:", allowed_model_ids)

# ---------- FILTER ----------
files = list(pred_dir.glob("*.json"))
print("Files found:", len(files))

saved = 0
for jf in files:
    with open(jf) as f:
        data = json.load(f)

    kept = [
        p for p in data["predictions"]
        if int(p["label_id"]) in allowed_model_ids
    ]

    if not kept:
        continue

    data["predictions"] = kept

    with open(out_dir / jf.name, "w") as f:
        json.dump(data, f, indent=2)

    saved += 1

print(f"✅ Saved {saved} filtered files")
print("DONE")
