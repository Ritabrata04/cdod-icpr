import json
from pathlib import Path
import argparse
from collections import Counter
import random

# ---------------------------- Utilities ----------------------------

def xyxy_to_xywh(box):
    """Convert [x1, y1, x2, y2] → [x, y, w, h]"""
    x1, y1, x2, y2 = box
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

def normalize_label(label):
    """Lowercase, remove spaces, strip"""
    return label.lower().replace(" ", "").strip()

# ---------------------------- Alignment Loader ----------------------------

def load_alignment(alignment_json, target_domain="bdd"):
    """Return normalized detector label → target category_id mapping"""
    raw = json.load(open(alignment_json))
    mapping = {}
    for key, val in raw.items():
        norm_key = normalize_label(key)
        if target_domain in val:
            mapping[norm_key] = int(val[target_domain]["id"])
    return mapping

# ---------------------------- Main ----------------------------

def main(pred_dir, gt_json, alignment_json, out_json, target_domain="bdd"):

    pred_dir = Path(pred_dir)
    pred_files = list(pred_dir.glob("*.json"))
    print(f"[PRED] Found {len(pred_files)} prediction files")

    # Load GT BDD images
    bdd = json.load(open(gt_json))
    gt_image_map = {Path(img["name"]).stem: Path(img["name"]).name for img in bdd}
    print(f"[GT] Loaded {len(gt_image_map)} BDD images")

    # Alignment mapping
    alignment_map = load_alignment(alignment_json, target_domain)
    print(f"[ALIGN] Loaded {len(alignment_map)} labels for target domain '{target_domain}'")

    coco_dets = []
    skipped_log = []
    label_counter = Counter()
    skipped_images = 0
    skipped_labels = 0

    # Optional: sample subset (set to 1.0 = all images)
    sample_size = max(1, int(1.0 * len(gt_image_map)))
    sampled_stems = set(random.sample(list(gt_image_map.keys()), sample_size))
    print(f"[SUBSET] Processing {len(sampled_stems)} images")

    for jf in pred_files:
        stem = jf.stem
        if stem not in sampled_stems:
            continue
        if stem not in gt_image_map:
            skipped_images += 1
            continue
        image_id = gt_image_map[stem]  # filename

        pred = json.load(open(jf))
        for p in pred.get("predictions", []):
            raw_label = p.get("label_name", "")
            norm_label = normalize_label(raw_label)

            if norm_label not in alignment_map:
                skipped_labels += 1
                skipped_log.append({
                    "image": stem,
                    "raw": raw_label,
                    "norm": norm_label,
                    "reason": "not_in_alignment"
                })
                continue

            category_id = alignment_map[norm_label]

            coco_dets.append({
                "image_id": image_id,
                "category_id": category_id,
                "bbox": xyxy_to_xywh(p["bbox_xyxy"]),
                "score": float(p.get("score", 1.0))
            })

            label_counter[norm_label] += 1

    # ---------------- Save Outputs ----------------
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    json.dump(coco_dets, open(out_json, "w"))
    skip_file = out_json.replace(".json", "_skipped.json")
    json.dump(skipped_log, open(skip_file, "w"), indent=2)

    # ---------------- Summary ----------------
    print("\n========== SUMMARY ==========")
    print(f"Saved dets: {len(coco_dets)}")
    print(f"Skipped images: {skipped_images}")
    print(f"Skipped labels: {skipped_labels}")
    print("\n[SKIP REASONS]")
    for k, v in Counter(x["reason"] for x in skipped_log).items():
        print(f"{k}: {v}")
    print("\n[TOP KEPT LABELS]")
    for k, v in label_counter.most_common(10):
        print(f"{k}: {v}")
    print("\n[FILES]")
    print(f"Dets → {out_json}")
    print(f"Skipped → {skip_file}")

# ---------------- Entry Point ----------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pred_dir")
    ap.add_argument("gt_json")
    ap.add_argument("alignment_json")
    ap.add_argument("out_json")
    ap.add_argument("--target_domain", default="bdd")
    args = ap.parse_args()

    main(
        args.pred_dir,
        args.gt_json,
        args.alignment_json,
        args.out_json,
        args.target_domain
    )




# import json
# from pathlib import Path
# import argparse
# from collections import Counter
# import random

# # ---------------------------- Utilities ----------------------------

# def xyxy_to_xywh(box):
#     """Convert [x1, y1, x2, y2] → [x, y, w, h]"""
#     x1, y1, x2, y2 = box
#     return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]

# def normalize_label(label):
#     """Case-insensitive, space-insensitive normalization"""
#     return label.lower().replace(" ", "").strip()

# # ---------------------------- Alignment Loader ----------------------------

# def load_alignment(alignment_json, target_domain):
#     """
#     Loads alignment JSON.

#     Cross-domain mode:
#         label → {domain: {id, name}}
#     Same-domain mode:
#         normalized_label → category_id
#     """
#     with open(alignment_json, "r") as f:
#         raw = json.load(f)

#     first_val = next(iter(raw.values()))

#     if isinstance(first_val, dict) and target_domain in first_val:
#         print(f"[ALIGN] Cross-domain mode → {target_domain}")
#         return raw, "cross"

#     print("[ALIGN] Same-domain mode (normalized labels)")
#     alignment = {}
#     for cid, entry in raw.items():
#         raw_name = normalize_label(
#             entry.get("raw_name", entry.get("norm_name", ""))
#         )
#         alignment[raw_name] = int(cid)

#     return alignment, "same"

# # ---------------------------- Main ----------------------------

# def main(pred_dir, gt_json, alignment_json, out_json, target_domain="coco"):

#     # ---------------------------- Load GT ----------------------------
#     with open(gt_json, "r") as f:
#         gt = json.load(f)

#     gt_image_map = {
#         Path(img["file_name"]).stem: img["id"]
#         for img in gt["images"]
#     }

#     print(f"[GT] Loaded {len(gt_image_map)} GT images")

#     # ---------------------------- Subsample----------------------------
#     all_stems = list(gt_image_map.keys())
#     sample_size = max(1, int(1* len(all_stems)))
#     sampled_stems = set(random.sample(all_stems, sample_size))

#     print(f"[SUBSET] Using {sample_size}/{len(all_stems)} images")

#     # ---------------------------- Load Alignment ----------------------------
#     alignment, mode = load_alignment(alignment_json, target_domain)

#     # ---------------------------- Iterate Predictions ----------------------------
#     pred_dir = Path(pred_dir)
#     pred_files = list(pred_dir.glob("*.json"))

#     print(f"[PRED] Found {len(pred_files)} prediction files")

#     coco_dets = []
#     skipped_images = 0
#     skipped_labels = 0

#     skipped_due_to_alignment = []
#     label_counter = Counter()

#     for jf in pred_files:
#         stem = jf.stem

#         # limit to 5% subset
#         if stem not in sampled_stems:
#             continue

#         if stem not in gt_image_map:
#             skipped_images += 1
#             continue

#         image_id = gt_image_map[stem]

#         with open(jf, "r") as f:
#             pred = json.load(f)

#         for p in pred.get("predictions", []):
#             pred_label_raw = p["label_name"]
#             pred_label = normalize_label(pred_label_raw)

#             # ---------------------------- Alignment ----------------------------
#             if mode == "cross":
#                 if pred_label not in alignment:
#                     skipped_labels += 1
#                     skipped_due_to_alignment.append({
#                         "image": stem,
#                         "raw_label": pred_label_raw,
#                         "normalized": pred_label,
#                         "reason": "label_not_in_alignment"
#                     })
#                     continue

#                 if target_domain not in alignment[pred_label]:
#                     skipped_labels += 1
#                     skipped_due_to_alignment.append({
#                         "image": stem,
#                         "raw_label": pred_label_raw,
#                         "normalized": pred_label,
#                         "reason": f"no_mapping_to_{target_domain}"
#                     })
#                     continue

#                 category_id = int(
#                     alignment[pred_label][target_domain]["id"]
#                 )

#             else:
#                 if pred_label not in alignment:
#                     skipped_labels += 1
#                     skipped_due_to_alignment.append({
#                         "image": stem,
#                         "raw_label": pred_label_raw,
#                         "normalized": pred_label,
#                         "reason": "label_not_in_alignment"
#                     })
#                     continue

#                 category_id = alignment[pred_label]

#             # ---------------------------- Store Detection ----------------------------
#             coco_dets.append({
#                 "image_id": image_id,
#                 "category_id": category_id,
#                 "bbox": xyxy_to_xywh(p["bbox_xyxy"]),
#                 "score": float(p["score"])
#             })

#             label_counter[pred_label] += 1

#     # ---------------------------- Save Outputs ----------------------------
#     with open(out_json, "w") as f:
#         json.dump(coco_dets, f)

#     skipped_json = out_json.replace(".json", "_skipped_alignment.json")
#     with open(skipped_json, "w") as f:
#         json.dump(skipped_due_to_alignment, f, indent=2)

#     # ---------------------------- Summary ----------------------------
#     print("\n========== SUMMARY ==========")
#     print(f"Saved detections           : {len(coco_dets)}")
#     print(f"Skipped images (no GT)     : {skipped_images}")
#     print(f"Skipped predictions (cls) : {skipped_labels}")

#     print("\n[SKIP REASONS]")
#     for k, v in Counter(x["reason"] for x in skipped_due_to_alignment).items():
#         print(f"  {k}: {v}")

#     print("\n[TOP SKIPPED LABELS]")
#     for lbl, cnt in Counter(
#         x["normalized"] for x in skipped_due_to_alignment
#     ).most_common(10):
#         print(f"  {lbl}: {cnt}")

#     print("\n[TOP KEPT LABELS]")
#     for lbl, cnt in label_counter.most_common(10):
#         print(f"  {lbl}: {cnt}")

#     print("\n[OK] Outputs written:")
#     print(f"  Detections → {out_json}")
#     print(f"  Skipped    → {skipped_json}")

# # ---------------------------- Entry ----------------------------

# if __name__ == "__main__":
#     ap = argparse.ArgumentParser()
#     ap.add_argument("pred_dir")
#     ap.add_argument("gt_json")
#     ap.add_argument("alignment_json")
#     ap.add_argument("out_json")
#     ap.add_argument("--target_domain", default="coco")
#     args = ap.parse_args()

#     main(
#         args.pred_dir,
#         args.gt_json,
#         args.alignment_json,
#         args.out_json,
#         args.target_domain
#     )


# import json
# from pathlib import Path
# import argparse
# from collections import Counter


# def xyxy_to_xywh(box):
#     x1, y1, x2, y2 = box
#     return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


# def normalize_label(name: str):
#     return name.strip().lower()


# def main(pred_dir, gt_json, alignment_json, out_json, target_domain="coco", max_print=5):
#     # ----------------------------
#     # Load ground truth
#     # ----------------------------
#     with open(gt_json, "r") as f:
#         gt = json.load(f)

#     gt_images = gt["images"]

#     # Map: image key (stem) -> image_id
#     gt_key_to_id = {}
#     for img in gt_images:
#         key = Path(img["file_name"]).stem
#         gt_key_to_id[key] = img["id"]

#     print(f"[GT] Loaded {len(gt_key_to_id)} images")
#     print("[GT] Sample keys:")
#     for k in list(gt_key_to_id.keys())[:max_print]:
#         print("   ", k)

#     # ----------------------------
#     # Load alignment (case-insensitive)
#     # ----------------------------
#     with open(alignment_json, "r") as f:
#         alignment_raw = json.load(f)

#     alignment = {normalize_label(k): v for k, v in alignment_raw.items()}

#     print(f"[ALIGN] Loaded {len(alignment)} aligned classes")
#     print("[ALIGN] Sample entries:")
#     for k in list(alignment.keys())[:max_print]:
#         if target_domain in alignment[k]:
#             print(
#                 f"   {k} -> {target_domain} "
#                 f"(id={alignment[k][target_domain]['id']})"
#             )

#     # ----------------------------
#     # Iterate predictions
#     # ----------------------------
#     coco_dets = []
#     skipped_images = 0
#     skipped_labels = 0
#     label_counter = Counter()

#     pred_files = sorted(Path(pred_dir).glob("*.json"))
#     print(f"[PRED] Found {len(pred_files)} prediction JSONs")
#     print("[PRED] Sample filenames:")
#     for p in pred_files[:max_print]:
#         print("   ", p.name)

#     matched_images = set()

#     for jf in pred_files:
#         image_key = jf.stem

#         if image_key not in gt_key_to_id:
#             skipped_images += 1
#             continue

#         image_id = gt_key_to_id[image_key]
#         matched_images.add(image_key)

#         with open(jf, "r") as f:
#             pred = json.load(f)

#         for p in pred.get("predictions", []):
#             raw_label = p.get("label_name", "")
#             label = normalize_label(raw_label)
#             label_counter[raw_label] += 1

#             if label not in alignment:
#                 skipped_labels += 1
#                 continue

#             if target_domain not in alignment[label]:
#                 skipped_labels += 1
#                 continue

#             category_id = int(alignment[label][target_domain]["id"])

#             coco_dets.append({
#                 "image_id": image_id,
#                 "category_id": category_id,
#                 "bbox": xyxy_to_xywh(p["bbox_xyxy"]),
#                 "score": float(p["score"])
#             })

#     # ----------------------------
#     # Save
#     # ----------------------------
#     with open(out_json, "w") as f:
#         json.dump(coco_dets, f)

#     # ----------------------------
#     # Debug summary
#     # ----------------------------
#     print("\n========== SUMMARY ==========")
#     print(f"Saved detections           : {len(coco_dets)}")
#     print(f"Skipped images (no GT)     : {skipped_images}")
#     print(f"Skipped predictions (cls) : {skipped_labels}")

#     print("\n[OVERLAP]")
#     print(f"GT images        : {len(gt_key_to_id)}")
#     print(f"Prediction files: {len(pred_files)}")
#     print(f"Matched images  : {len(matched_images)}")

#     print("\n[DEBUG] Top predicted labels:")
#     for lbl, cnt in label_counter.most_common(10):
#         print(f"   {lbl}: {cnt}")

#     print("\n[OK] COCO-format detections written to:")
#     print(f"    {out_json}")


# if __name__ == "__main__":
#     ap = argparse.ArgumentParser()
#     ap.add_argument("pred_dir")
#     ap.add_argument("gt_json")
#     ap.add_argument("alignment_json")
#     ap.add_argument("out_json")
#     ap.add_argument("--target_domain", default="coco")
#     ap.add_argument("--max_print", type=int, default=5)
#     args = ap.parse_args()

#     main(
#         args.pred_dir,
#         args.gt_json,
#         args.alignment_json,
#         args.out_json,
#         args.target_domain,
#         args.max_print
#     )
