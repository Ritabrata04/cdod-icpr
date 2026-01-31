import json
from itertools import combinations

DATASETS = {
    "cityscapes": "cityscapes_norm.json",
    "coco": "coco_norm.json",
    "objects365": "objects365_norm.json",
    "bdd":"bdd_class.json"
}

def load_norm(path):
    with open(path) as f:
        return json.load(f)

norms = {k: load_norm(v) for k, v in DATASETS.items()}

def invert(norm_dict):
    return {v["norm_name"]: (cid, v["raw_name"])
            for cid, v in norm_dict.items()}

inv = {k: invert(v) for k, v in norms.items()}

for a, b in combinations(inv.keys(), 2):
    common = sorted(set(inv[a]) & set(inv[b]))

    alignment = {}
    for cname in common:
        alignment[cname] = {
            a: {"id": inv[a][cname][0], "name": inv[a][cname][1]},
            b: {"id": inv[b][cname][0], "name": inv[b][cname][1]},
        }

    out = f"{a}_to_{b}_alignment.json"
    with open(out, "w") as f:
        json.dump(alignment, f, indent=2)

    print(f"{a} ↔ {b}: {len(common)} common classes → {out}")
