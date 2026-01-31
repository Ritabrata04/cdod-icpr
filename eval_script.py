import json
from pathlib import Path
import cv2
from tqdm import tqdm

from mmdet.apis import DetInferencer

# ==========================================================
# PATHS
# ==========================================================

BDD_VAL_DIR = Path(
    r"C:\Users\ISI_UTS\Ritabrata_Hrishit\PolarNeXt\datasets\bdd\val"
)

RESULTS_ROOT = Path("results_bdd_val")

DEVICE = "cuda:0"
BATCH_SIZE = 64
SAVE_IMAGES = False  # keep False for speed and RAM safety

# ==========================================================
# TRAINED MODELS
# ==========================================================

TRAINED_MODELS = {
    "coco": {
        "config": r"C:\Users\ISI_UTS\Ritabrata_Hrishit\PolarNeXt\projects\PolarNeXt\configs\polarnext_r50-torch_fpn_1x_coco.py",
        "weights": r"C:\Users\ISI_UTS\Ritabrata_Hrishit\PolarNeXt\polarnext_r50_epoch_36.pth"
    },
    "cityscapes": {
        "config": r"C:\Users\ISI_UTS\Ritabrata_Hrishit\PolarNeXt\configs\cityscapes\faster-rcnn_r50_fpn_1x_cityscapes.py",
        "weights": r"C:\Users\ISI_UTS\Ritabrata_Hrishit\PolarNeXt\faster_rcnn_r50_fpn_1x_cityscapes_20200502-829424c0.pth"
    },
    "objects365": {
        "config": r"C:\Users\ISI_UTS\Ritabrata_Hrishit\PolarNeXt\configs\objects365\faster-rcnn_r50_fpn_16xb4-1x_objects365v2.py",
        "weights": r"C:\Users\ISI_UTS\Ritabrata_Hrishit\PolarNeXt\faster_rcnn_r50_fpn_16x4_1x_obj365v2_20221220_175040-5910b015.pth"
    }
}

# ==========================================================
# UTILS
# ==========================================================

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def draw_boxes(img, preds, class_names):
    vis = img.copy()
    for p in preds:
        x1, y1, x2, y2 = map(int, p["bbox_xyxy"])
        cls = p["label_id"]
        score = p["score"]
        name = class_names[cls] if cls < len(class_names) else str(cls)

        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            vis,
            f"{name}:{score:.2f}",
            (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )
    return vis


# ==========================================================
# MAIN
# ==========================================================

def run_bdd_val():

    images = sorted(BDD_VAL_DIR.glob("*.jpg"))
    print(f"BDD val images found: {len(images)}")

    for train_name, train_cfg in TRAINED_MODELS.items():

        print(f"\n==============================")
        print(f" Train domain: {train_name}")
        print(f" Test domain : bdd_val")
        print(f"==============================")

        inferencer = DetInferencer(
            model=train_cfg["config"],
            weights=train_cfg["weights"],
            device=DEVICE
        )

        # Handle missing dataset_meta safely
        class_names = (
            inferencer.model.dataset_meta["classes"]
            if inferencer.model.dataset_meta is not None
            else []
        )

        out_dir = RESULTS_ROOT / f"train_{train_name}__test_bdd_val"
        json_out_dir = out_dir / "json"
        img_out_dir = out_dir / "images"

        ensure_dir(json_out_dir)
        if SAVE_IMAGES:
            ensure_dir(img_out_dir)

        # --------------------------------------------------
        # Batch inference
        # --------------------------------------------------
        for i in tqdm(range(0, len(images), BATCH_SIZE)):
            batch_paths = images[i:i + BATCH_SIZE]
            batch_str = [str(p) for p in batch_paths]

            results = inferencer(batch_str)

            for img_path, res in zip(batch_paths, results["predictions"]):

                img = cv2.imread(str(img_path))
                h, w = img.shape[:2]

                records = []
                for bbox, score, cls in zip(
                    res["bboxes"], res["scores"], res["labels"]
                ):
                    cls = int(cls)
                    records.append({
                        "label_id": cls,
                        "label_name": (
                            class_names[cls]
                            if cls < len(class_names) else "UNKNOWN"
                        ),
                        "score": float(score),
                        "bbox_xyxy": list(map(float, bbox))
                    })

                json_data = {
                    "image": img_path.name,
                    "width": w,
                    "height": h,
                    "train_domain": train_name,
                    "test_domain": "bdd_val",
                    "predictions": records
                }

                json_path = json_out_dir / f"{img_path.stem}.json"
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(json_data, f)

                if SAVE_IMAGES:
                    vis_img = draw_boxes(img, records, class_names)
                    cv2.imwrite(str(img_out_dir / img_path.name), vis_img)

        print(f"Saved results → {out_dir}")


if __name__ == "__main__":
    run_bdd_val()
    print("\n✅ All inference runs on BDD val completed.")



# import json
# from pathlib import Path
# from tqdm import tqdm
# import torch
# import cv2
# import numpy as np

# from mmengine import Config
# from mmdet.apis import init_detector, inference_detector

# # --------------------------------------------------
# # Torch / CUDA settings
# # --------------------------------------------------
# torch.set_num_threads(1)
# torch.backends.cudnn.benchmark = True
# torch.set_grad_enabled(False)

# DEVICE = "cuda:0"
# BATCH_SIZE = 1  # Process one image at a time for simplicity
# RESULTS_ROOT = Path("results_cross_dataset")

# # --------------------------------------------------
# # DATASETS
# # --------------------------------------------------
# DATASETS = {
#     "coco": {"image_dir": "datasets/coco/val2017"},
#     "cityscapes": {"image_dir": "datasets/cityscapes/leftImg8bit/val"},
#     "objects365": {"image_dir": "datasets/objects365/images/val/val"},
#     "bdd100k": {"image_dir": "datasets/bdd/val"},
# }

# # --------------------------------------------------
# # TRAIN CONFIGS
# # --------------------------------------------------
# TRAIN_CONFIGS = {
#     "objects365": {
#         "config": r"C:\Users\ISI_UTS\Ritabrata_Hrishit\PolarNeXt\configs\objects365\faster-rcnn_r50_fpn_16xb4-1x_objects365v2.py",
#         "weights": r"C:\Users\ISI_UTS\Ritabrata_Hrishit\PolarNeXt\faster_rcnn_r50_fpn_16x4_1x_obj365v2_20221220_175040-5910b015.pth",
#     },
#     "coco": {
#         "config": r"C:\Users\ISI_UTS\Ritabrata_Hrishit\PolarNeXt\projects\PolarNeXt\configs\polarnext_r50-torch_fpn_1x_coco.py",
#         "weights": r"C:\Users\ISI_UTS\Ritabrata_Hrishit\PolarNeXt\polarnext_r50_epoch_36.pth",
#     },
#     "cityscapes": {
#         "config": r"C:\Users\ISI_UTS\Ritabrata_Hrishit\PolarNeXt\configs\cityscapes\faster_rcnn_r50_fpn_1x_cityscapes.py",
#         "weights": r"C:\Users\ISI_UTS\Ritabrata_Hrishit\PolarNeXt\faster_rcnn_r50_fpn_1x_cityscapes_20200502-829424c0.pth",
#     },
#     "bdd100k": {
#         "config": r"C:\Users\ISI_UTS\Ritabrata_Hrishit\PolarNeXt\bdd100k-models\det\configs\det\faster_rcnn_r50_fpn_1x_det_bdd100k.py",
#         "weights": r"C:\Users\ISI_UTS\Ritabrata_Hrishit\PolarNeXt\faster_rcnn_r50_fpn_1x_det_bdd100k.pth",
#     },
# }

# # --------------------------------------------------
# # TRAIN → TEST PAIRS
# # --------------------------------------------------
# RUN_PAIRS = [
#     ("bdd100k", "bdd100k"),
#     ("bdd100k", "objects365"),
#     ("bdd100k", "cityscapes"),
#     ("bdd100k", "coco"),
#     ("coco", "bdd100k"),
#     ("cityscapes", "bdd100k"),
#     ("objects365", "bdd100k"),
# ]

# # --------------------------------------------------
# def ensure_dir(p):
#     Path(p).mkdir(parents=True, exist_ok=True)

# # --------------------------------------------------
# def run_minimal_inference():
#     """
#     Minimal inference using init_detector + inference_detector directly.
#     Bypasses DetInferencer and all its config requirements.
#     """
    
#     for train_name, test_name in RUN_PAIRS:
#         print("\n======================================")
#         print(f" Train: {train_name}  →  Test: {test_name}")
#         print("======================================")

#         # Initialize model directly - no pipeline/meta patching needed
#         config_file = TRAIN_CONFIGS[train_name]["config"]
#         checkpoint_file = TRAIN_CONFIGS[train_name]["weights"]
        
#         print(f" Loading model from {train_name}...")
#         model = init_detector(config_file, checkpoint_file, device=DEVICE)
        
#         # Get test images
#         img_dir = Path(DATASETS[test_name]["image_dir"])
#         out_dir = RESULTS_ROOT / f"train_{train_name}__test_{test_name}" / "json"
#         ensure_dir(out_dir)

#         images = sorted(
#             p for p in img_dir.rglob("*")
#             if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
#         )

#         print(f" Images found: {len(images)}")

#         # Run inference on each image
#         for img_path in tqdm(images, desc=f"Inference ({train_name}→{test_name})"):
#             # Read image to get dimensions
#             img = cv2.imread(str(img_path))
#             h, w = img.shape[:2]
            
#             # Run inference
#             result = inference_detector(model, str(img_path))
            
#             # Extract predictions from result
#             records = []
            
#             # Handle different result formats (DataSample or list)
#             if hasattr(result, 'pred_instances'):
#                 # MMDet 3.x format
#                 pred_instances = result.pred_instances
#                 bboxes = pred_instances.bboxes.cpu().numpy()
#                 scores = pred_instances.scores.cpu().numpy()
#                 labels = pred_instances.labels.cpu().numpy()
#             else:
#                 # MMDet 2.x format or other
#                 # Assuming result is a list of arrays per class
#                 bboxes = []
#                 scores = []
#                 labels = []
#                 for cls_id, cls_result in enumerate(result):
#                     if len(cls_result) > 0:
#                         for det in cls_result:
#                             bboxes.append(det[:4])
#                             scores.append(det[4])
#                             labels.append(cls_id)
#                 bboxes = np.array(bboxes) if bboxes else np.empty((0, 4))
#                 scores = np.array(scores) if scores else np.empty(0)
#                 labels = np.array(labels) if labels else np.empty(0)
            
#             # Convert to JSON format
#             for bbox, score, cls in zip(bboxes, scores, labels):
#                 records.append({
#                     "label_id": int(cls),
#                     "score": float(score),
#                     "bbox_xyxy": list(map(float, bbox)),
#                 })

#             # Save JSON
#             with open(out_dir / f"{img_path.stem}.json", "w") as f:
#                 json.dump({
#                     "image": img_path.name,
#                     "width": w,
#                     "height": h,
#                     "train_domain": train_name,
#                     "test_domain": test_name,
#                     "predictions": records,
#                 }, f, indent=2)

#         print(f" ✔ JSONs saved → {out_dir}")
        
#         # Clean up model to free memory
#         del model
#         torch.cuda.empty_cache()

# # --------------------------------------------------
# if __name__ == "__main__":
#     run_minimal_inference()
#     print("\nCross-dataset inference completed.")


# ###########ONLY FOR BDD RUNS
# import json
# from pathlib import Path
# from tqdm import tqdm
# import torch
# import cv2

# from mmengine import Config
# from mmdet.apis import DetInferencer

# # --------------------------------------------------
# # Torch / CUDA settings
# # --------------------------------------------------
# torch.set_num_threads(1)
# torch.backends.cudnn.benchmark = True
# torch.set_grad_enabled(False)

# DEVICE = "cuda:0"
# BATCH_SIZE = 16
# RESULTS_ROOT = Path("results_cross_dataset")

# # --------------------------------------------------
# # DATASETS
# # --------------------------------------------------
# DATASETS = {
#     "coco": {"image_dir": "datasets/coco/val2017"},
#     "cityscapes": {"image_dir": "datasets/cityscapes/leftImg8bit/val"},
#     "objects365": {"image_dir": "datasets/objects365/images/val/val"},
#     "bdd100k": {"image_dir": "datasets/bdd/val"},
# }

# # --------------------------------------------------
# # TRAIN CONFIGS
# # --------------------------------------------------
# TRAIN_CONFIGS = {
#     "objects365": {
#         "config": r"C:\Users\ISI_UTS\Ritabrata_Hrishit\PolarNeXt\configs\objects365\faster-rcnn_r50_fpn_16xb4-1x_objects365v2.py",
#         "weights": r"C:\Users\ISI_UTS\Ritabrata_Hrishit\PolarNeXt\faster_rcnn_r50_fpn_16x4_1x_obj365v2_20221220_175040-5910b015.pth",
#     },
#     "coco": {
#         "config": r"C:\Users\ISI_UTS\Ritabrata_Hrishit\PolarNeXt\projects\PolarNeXt\configs\polarnext_r50-torch_fpn_1x_coco.py",
#         "weights": r"C:\Users\ISI_UTS\Ritabrata_Hrishit\PolarNeXt\polarnext_r50_epoch_36.pth",
#     },
#     "cityscapes": {
#         "config": r"C:\Users\ISI_UTS\Ritabrata_Hrishit\PolarNeXt\configs\cityscapes\faster_rcnn_r50_fpn_1x_cityscapes.py",
#         "weights": r"C:\Users\ISI_UTS\Ritabrata_Hrishit\PolarNeXt\faster_rcnn_r50_fpn_1x_cityscapes_20200502-829424c0.pth",
#     },
#     "bdd100k": {
#         "config": r"C:\Users\ISI_UTS\Ritabrata_Hrishit\PolarNeXt\bdd100k-models\det\configs\det\faster_rcnn_r50_fpn_1x_det_bdd100k.py",
#         "weights": r"C:\Users\ISI_UTS\Ritabrata_Hrishit\PolarNeXt\faster_rcnn_r50_fpn_1x_det_bdd100k.pth",
#     },
# }

# # --------------------------------------------------
# # TRAIN → TEST PAIRS
# # --------------------------------------------------
# RUN_PAIRS = [
#     ("bdd100k", "bdd100k"),
#     ("bdd100k", "objects365"),
#     ("bdd100k", "cityscapes"),
#     ("bdd100k", "coco"),
#     ("coco", "bdd100k"),
#     ("cityscapes", "bdd100k"),
#     ("objects365", "bdd100k"),
# ]

# # --------------------------------------------------
# def ensure_dir(p):
#     Path(p).mkdir(parents=True, exist_ok=True)

# def chunked(lst, n):
#     for i in range(0, len(lst), n):
#         yield lst[i:i + n]

# # --------------------------------------------------
# def load_and_patch_config(train_name):
#     """
#     Patch legacy configs so they work with DetInferencer:
#     - Ensure test_dataloader.dataset.pipeline exists
#     - Ensure model.dataset_meta exists
#     """
#     cfg = Config.fromfile(TRAIN_CONFIGS[train_name]["config"])
#     ref_cfg = Config.fromfile(TRAIN_CONFIGS["coco"]["config"])

#     # --------------------------------------------------
#     # Patch test pipeline (legacy BDD configs)
#     # --------------------------------------------------
#     if not hasattr(cfg, "test_dataloader") or \
#        not hasattr(cfg.test_dataloader.dataset, "pipeline"):
#         print(" [INFO] Borrowing test pipeline from COCO PolarNeXt config")
#         cfg.test_dataloader = ref_cfg.test_dataloader

#     # --------------------------------------------------
#     # Patch dataset_meta (CRITICAL)
#     # --------------------------------------------------
#     if not hasattr(cfg, "model") or \
#        not hasattr(cfg.model, "dataset_meta") or \
#        cfg.model.dataset_meta is None:
#         print(" [INFO] Borrowing model.dataset_meta from COCO PolarNeXt config")
#         cfg.model.dataset_meta = ref_cfg.model.dataset_meta

#     return cfg

# # --------------------------------------------------
# def run_selected_cross_dataset():

#     for train_name, test_name in RUN_PAIRS:

#         print("\n======================================")
#         print(f" Train: {train_name}  →  Test: {test_name}")
#         print("======================================")

#         cfg = load_and_patch_config(train_name)

#         inferencer = DetInferencer(
#             model=cfg,
#             weights=TRAIN_CONFIGS[train_name]["weights"],
#             device=DEVICE,
#         )

#         img_dir = Path(DATASETS[test_name]["image_dir"])
#         out_dir = RESULTS_ROOT / f"train_{train_name}__test_{test_name}" / "json"
#         ensure_dir(out_dir)

#         images = sorted(
#             p for p in img_dir.rglob("*")
#             if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
#         )

#         print(f" Images found: {len(images)}")

#         for batch_paths in tqdm(
#             list(chunked(images, BATCH_SIZE)),
#             desc=f"Inference ({train_name}→{test_name})"
#         ):
#             outputs = inferencer([str(p) for p in batch_paths])["predictions"]

#             for img_path, preds in zip(batch_paths, outputs):
#                 img = cv2.imread(str(img_path))
#                 h, w = img.shape[:2]

#                 records = []
#                 for bbox, score, cls in zip(
#                     preds["bboxes"],
#                     preds["scores"],
#                     preds["labels"]
#                 ):
#                     records.append({
#                         "label_id": int(cls),
#                         "score": float(score),
#                         "bbox_xyxy": list(map(float, bbox)),
#                     })

#                 with open(out_dir / f"{img_path.stem}.json", "w") as f:
#                     json.dump({
#                         "image": img_path.name,
#                         "width": w,
#                         "height": h,
#                         "train_domain": train_name,
#                         "test_domain": test_name,
#                         "predictions": records,
#                     }, f, indent=2)

#         print(f" ✔ JSONs saved → {out_dir}")

# # --------------------------------------------------
# if __name__ == "__main__":
#     run_selected_cross_dataset()
#     print("\nSelected cross-dataset inference completed.")

################ ONLY FOR COCO VAL RUNS
# import json
# from pathlib import Path
# import cv2
# from tqdm import tqdm

# from mmdet.apis import DetInferencer

# # ==========================================================
# # PATHS
# # ==========================================================

# BDD_VAL_DIR = Path(
#     r"C:\Users\ISI_UTS\Ritabrata_Hrishit\PolarNeXt\datasets\bdd\val"
# )

# RESULTS_ROOT = Path("results_coco_val")

# DEVICE = "cuda:0"
# BATCH_SIZE = 64
# SAVE_IMAGES = True  # set False if you want speed

# # ==========================================================
# # TRAINED MODELS (3 runs)
# # ==========================================================

# TRAINED_MODELS = {
#     "coco": {
#         "config": r"C:\Users\ISI_UTS\Ritabrata_Hrishit\PolarNeXt\projects\PolarNeXt\configs\polarnext_r50-torch_fpn_1x_coco.py",
#         "weights": r"C:\Users\ISI_UTS\Ritabrata_Hrishit\PolarNeXt\polarnext_r50_epoch_36.pth"
#     },
#     "cityscapes": {
#         "config": r"C:\Users\ISI_UTS\Ritabrata_Hrishit\PolarNeXt\configs\cityscapes\faster-rcnn_r50_fpn_1x_cityscapes.py",
#         "weights": r"C:\Users\ISI_UTS\Ritabrata_Hrishit\PolarNeXt\faster_rcnn_r50_fpn_1x_cityscapes_20200502-829424c0.pth"
#     },
#     "objects365": {
#         "config": r"C:\Users\ISI_UTS\Ritabrata_Hrishit\PolarNeXt\configs\objects365\faster-rcnn_r50_fpn_16xb4-1x_objects365v2.py",
#         "weights": r"C:\Users\ISI_UTS\Ritabrata_Hrishit\PolarNeXt\faster_rcnn_r50_fpn_16x4_1x_obj365v2_20221220_175040-5910b015.pth"
#     },
#     "bdd": {
#         "config": r"C:\Users\ISI_UTS\Ritabrata_Hrishit\PolarNeXt\bdd100k-models\det\configs\det\faster_rcnn_r50_fpn_1x_det_bdd100k.py",
#         "weights": r"C:\Users\ISI_UTS\Ritabrata_Hrishit\PolarNeXt\faster_rcnn_r50_fpn_1x_det_bdd100k.pth"
#     }
# }

# # ==========================================================
# # UTILS
# # ==========================================================

# def ensure_dir(p):
#     Path(p).mkdir(parents=True, exist_ok=True)


# def draw_boxes(img, preds, class_names):
#     vis = img.copy()
#     for p in preds:
#         x1, y1, x2, y2 = map(int, p["bbox_xyxy"])
#         cls = p["label_id"]
#         score = p["score"]
#         name = class_names[cls] if cls < len(class_names) else str(cls)

#         cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(
#             vis,
#             f"{name}:{score:.2f}",
#             (x1, max(0, y1 - 5)),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.5,
#             (0, 255, 0),
#             1,
#         )
#     return vis


# # ==========================================================
# # MAIN
# # ==========================================================

# def run_coco_val_only():

#     images = sorted(COCO_VAL_DIR.glob("*.jpg"))
#     print(f"COCO val images found: {len(images)}")

#     for train_name, train_cfg in TRAINED_MODELS.items():

#         print(f"\n==============================")
#         print(f" Train domain: {train_name}")
#         print(f" Test domain : coco_val")
#         print(f"==============================")

#         inferencer = DetInferencer(
#             model=train_cfg["config"],
#             weights=train_cfg["weights"],
#             device=DEVICE
#         )

#         class_names = inferencer.model.dataset_meta["classes"]

#         out_dir = RESULTS_ROOT / f"train_{train_name}__test_coco_val"
#         img_out_dir = out_dir / "images"
#         json_out_dir = out_dir / "json"

#         ensure_dir(json_out_dir)
#         if SAVE_IMAGES:
#             ensure_dir(img_out_dir)

#         # --------------------------------------------------
#         # Batch inference
#         # --------------------------------------------------
#         for i in tqdm(range(0, len(images), BATCH_SIZE)):
#             batch_paths = images[i:i + BATCH_SIZE]
#             batch_str = [str(p) for p in batch_paths]

#             results = inferencer(batch_str)

#             for img_path, res in zip(batch_paths, results["predictions"]):
#                 img = cv2.imread(str(img_path))
#                 h, w = img.shape[:2]

#                 records = []
#                 for bbox, score, cls in zip(
#                     res["bboxes"], res["scores"], res["labels"]
#                 ):
#                     records.append({
#                         "label_id": int(cls),
#                         "label_name": class_names[int(cls)]
#                             if int(cls) < len(class_names) else "UNKNOWN",
#                         "score": float(score),
#                         "bbox_xyxy": list(map(float, bbox))
#                     })

#                 json_data = {
#                     "image": img_path.name,
#                     "width": w,
#                     "height": h,
#                     "train_domain": train_name,
#                     "test_domain": "coco_val",
#                     "predictions": records
#                 }

#                 json_path = json_out_dir / f"{img_path.stem}.json"
#                 with open(json_path, "w", encoding="utf-8") as f:
#                     json.dump(json_data, f)

#                 if SAVE_IMAGES:
#                     vis_img = draw_boxes(img, records, class_names)
#                     cv2.imwrite(str(img_out_dir / img_path.name), vis_img)

#         print(f"Saved results → {out_dir}")


# if __name__ == "__main__":
#     run_coco_val_only()
#     print("\n✅ All inference runs on COCO val completed.")




#ONLY THE OBJECTS365 TRAINED MODEL RUNNING INFERENCE ON THE 3 DATASETS, BUT RELEASING JSONS ONLY
#BATCHES OF 16
# import json
# from pathlib import Path
# import cv2
# from tqdm import tqdm
# import torch

# from mmdet.apis import DetInferencer

# torch.set_num_threads(1)
# torch.backends.cudnn.benchmark = True

# TEST_DATASETS = {
#     "coco": {
#         "image_dir": "datasets/coco/test2017"
#     },
#     "cityscapes": {
#         "image_dir": "datasets/cityscapes/leftImg8bit/test"
#     },
#     "objects365": {
#         "image_dir": "datasets/objects365/images/val"
#     }
# }

# TRAIN_NAME = "objects365"
# TRAIN_CFG = {
#     "config": r"C:\Users\ISI_UTS\Ritabrata_Hrishit\PolarNeXt\configs\objects365\faster-rcnn_r50_fpn_16xb4-1x_objects365v2.py",
#     "weights": r"C:\Users\ISI_UTS\Ritabrata_Hrishit\PolarNeXt\faster_rcnn_r50_fpn_16x4_1x_obj365v2_20221220_175040-5910b015.pth"
# }

# DEVICE = "cuda:0"
# BATCH_SIZE = 16  # reduce if OOM
# RESULTS_ROOT = Path("results_cross_dataset")

# def ensure_dir(p):
#     Path(p).mkdir(parents=True, exist_ok=True)

# def chunked(lst, n):
#     for i in range(0, len(lst), n):
#         yield lst[i:i + n]

# def run_cross_dataset():

#     print(f"\n==============================")
#     print(f" Train domain: {TRAIN_NAME}")
#     print(f"==============================")

#     inferencer = DetInferencer(
#         model=TRAIN_CFG["config"],
#         weights=TRAIN_CFG["weights"],
#         device=DEVICE
#     )

#     class_names = inferencer.model.dataset_meta["classes"]

#     for test_name, test_cfg in tqdm(
#         TEST_DATASETS.items(),
#         desc=f"Test domains ({TRAIN_NAME})"
#     ):
#         img_dir = Path(test_cfg["image_dir"])
#         out_dir = RESULTS_ROOT / f"train_{TRAIN_NAME}__test_{test_name}"
#         json_out_dir = out_dir / "json"
#         ensure_dir(json_out_dir)

#         images = sorted(
#             p for p in img_dir.rglob("*")
#             if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
#         )

#         image_paths = [str(p) for p in images]

#         print(f"\n→ {TRAIN_NAME} → {test_name}")
#         print(f"  Images found: {len(images)}")

#         img_idx = 0

#         for batch_paths in tqdm(
#             list(chunked(image_paths, BATCH_SIZE)),
#             desc=f"Inference",
#             leave=False
#         ):
#             batch_results = inferencer(batch_paths)
#             batch_preds = batch_results["predictions"]

#             for preds in batch_preds:
#                 img_path = images[img_idx]

#                 img = cv2.imread(str(img_path))
#                 h, w = img.shape[:2]

#                 records = []
#                 for bbox, score, cls in zip(
#                     preds["bboxes"],
#                     preds["scores"],
#                     preds["labels"]
#                 ):
#                     cls = int(cls)
#                     records.append({
#                         "label_id": cls,
#                         "label_name": class_names[cls]
#                         if cls < len(class_names) else "UNKNOWN",
#                         "score": float(score),
#                         "bbox_xyxy": list(map(float, bbox))
#                     })

#                 json_data = {
#                     "image": img_path.name,
#                     "width": w,
#                     "height": h,
#                     "train_domain": TRAIN_NAME,
#                     "test_domain": test_name,
#                     "predictions": records
#                 }

#                 with open(json_out_dir / f"{img_path.stem}.json", "w") as f:
#                     json.dump(json_data, f, indent=2)

#                 img_idx += 1

#         print(f"   ✔ JSONs saved → {json_out_dir}")


# if __name__ == "__main__":
#     run_cross_dataset()
#     print("\n Objects365 → (COCO, Cityscapes, Objects365) inference completed.")


##ALL TO ALL, WITH IMAGES AND JSONS BOTH, NO BATCHES
# import os
# import json
# from pathlib import Path
# import cv2

# from mmdet.apis import DetInferencer


# TEST_DATASETS = {
#     "coco": {
#         "image_dir": "datasets/coco/test2017"
#     },
#     "cityscapes": {
#         "image_dir": "datasets/cityscapes/leftImg8bit/test"
#     },
#     "objects365": {
#         "image_dir": "datasets/objects365/images/val"
#     }
# }

# # ==========================================================
# # TRAINED MODELS
# # ==========================================================

# TRAINED_MODELS = {
#     "coco": {
#         "config": r"C:\Users\ISI_UTS\Ritabrata_Hrishit\PolarNeXt\projects\PolarNeXt\configs\polarnext_r50-torch_fpn_1x_coco.py",
#         "weights": r"C:\Users\ISI_UTS\Ritabrata_Hrishit\PolarNeXt\polarnext_r50_epoch_36.pth"
#     },
#     "cityscapes": {
#         "config": r"C:\Users\ISI_UTS\Ritabrata_Hrishit\PolarNeXt\configs\cityscapes\faster-rcnn_r50_fpn_1x_cityscapes.py",
#         "weights": r"C:\Users\ISI_UTS\Ritabrata_Hrishit\PolarNeXt\faster_rcnn_r50_fpn_1x_cityscapes_20200502-829424c0.pth"
#     },
#     "objects365": {
#         "config": r"C:\Users\ISI_UTS\Ritabrata_Hrishit\PolarNeXt\configs\objects365\faster-rcnn_r50_fpn_16xb4-1x_objects365v2.py",
#         "weights": r"C:\Users\ISI_UTS\Ritabrata_Hrishit\PolarNeXt\faster_rcnn_r50_fpn_16x4_1x_obj365v2_20221220_175040-5910b015.pth"
#     }
# }

# DEVICE = "cuda:0"
# RESULTS_ROOT = Path("results_cross_dataset")

# def ensure_dir(p):
#     Path(p).mkdir(parents=True, exist_ok=True)


# def draw_boxes(img, preds, class_names):
#     vis = img.copy()
#     for p in preds:
#         x1, y1, x2, y2 = map(int, p["bbox_xyxy"])
#         cls = p["label_id"]
#         score = p["score"]
#         name = class_names[cls] if cls < len(class_names) else str(cls)

#         cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.putText(
#             vis,
#             f"{name}:{score:.2f}",
#             (x1, max(0, y1 - 5)),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.5,
#             (0, 255, 0),
#             1,
#         )
#     return vis


# #MAIN 3*3 LOOP OF INFERENCER

# def run_cross_dataset():
#     for train_name, train_cfg in TRAINED_MODELS.items():
      
#         print(f" Train domain: {train_name}")
#         print(f"==============================")

#         inferencer = DetInferencer(
#             model=train_cfg["config"],
#             weights=train_cfg["weights"],
#             device=DEVICE
#         )

#         class_names = inferencer.model.dataset_meta["classes"]

#         for test_name, test_cfg in TEST_DATASETS.items():
#             print(f"\n   ➜ Test domain: {test_name}")

#             img_dir = Path(test_cfg["image_dir"])
#             out_dir = RESULTS_ROOT / f"train_{train_name}__test_{test_name}"
#             img_out_dir = out_dir / "images"
#             json_out_dir = out_dir / "json"

#             ensure_dir(img_out_dir)
#             ensure_dir(json_out_dir)

#             images = sorted(
#                 [p for p in img_dir.rglob("*")
#                  if p.suffix.lower() in [".png", ".jpg", ".jpeg"]]
#             )

#             print(f"     Images: {len(images)}")

#             for img_path in images:
#                 img = cv2.imread(str(img_path))
#                 h, w = img.shape[:2]

#                 result = inferencer(str(img_path))
#                 preds = result["predictions"][0]

#                 records = []
#                 for bbox, score, cls in zip(preds["bboxes"], preds["scores"], preds["labels"]):
#                     records.append({
#                         "label_id": int(cls),
#                         "label_name": class_names[int(cls)] if int(cls) < len(class_names) else "UNKNOWN",
#                         "score": float(score),
#                         "bbox_xyxy": list(map(float, bbox))
#                     })

#                 # Save JSON
#                 json_data = {
#                     "image": img_path.name,
#                     "width": w,
#                     "height": h,
#                     "train_domain": train_name,
#                     "test_domain": test_name,
#                     "predictions": records
#                 }

#                 json_path = json_out_dir / f"{img_path.stem}.json"
#                 with open(json_path, "w", encoding="utf-8") as f:
#                     json.dump(json_data, f, indent=2)

#                 # Save visualization
#                 vis_img = draw_boxes(img, records, class_names)
#                 out_img_path = img_out_dir / img_path.name
#                 cv2.imwrite(str(out_img_path), vis_img)

#             print(f"Saved → {out_dir}")


# if __name__ == "__main__":
#     run_cross_dataset()
#     print("\n All 3×3 cross-dataset inference runs completed.")
