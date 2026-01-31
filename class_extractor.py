# import json
# import sys
# from pathlib import Path

# def extract_categories(gt_json, out_json):
#     with open(gt_json, "r") as f:
#         data = json.load(f)

#     assert "categories" in data, "No categories field found"

#     categories = {
#         int(cat["id"]): cat["name"].strip()
#         for cat in data["categories"]
#     }

#     with open(out_json, "w") as f:
#         json.dump(categories, f, indent=2)

#     print(f"Saved {len(categories)} categories to {out_json}")

# if __name__ == "__main__":
#     gt_json = sys.argv[1]
#     out_json = sys.argv[2]

#     extract_categories(gt_json, out_json)


import json
import sys

def extract_categories(gt_json, out_json):
    with open(gt_json, "r") as f:
        data = json.load(f)

    categories = set()

    for img in data:
        for lbl in img.get("labels", []):
            categories.add(lbl["category"].strip())

    # Convert to ID mapping
    categories = {i: name for i, name in enumerate(sorted(categories))}

    with open(out_json, "w") as f:
        json.dump(categories, f, indent=2)

    print(f"Saved {len(categories)} categories to {out_json}")

if __name__ == "__main__":
    gt_json = sys.argv[1]
    out_json = sys.argv[2]
    extract_categories(gt_json, out_json)
