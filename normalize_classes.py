import json
import re
import sys

def normalize(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())

def normalize_classes(in_json, out_json):
    with open(in_json) as f:
        classes = json.load(f)

    normalized = {}
    for cid, name in classes.items():
        normalized[cid] = {
            "raw_name": name,
            "norm_name": normalize(name),
        }

    with open(out_json, "w") as f:
        json.dump(normalized, f, indent=2)

if __name__ == "__main__":
    normalize_classes(sys.argv[1], sys.argv[2])

