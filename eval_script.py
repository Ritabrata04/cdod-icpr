import json
from pathlib import Path
import cv2
from tqdm import tqdm
import torch

from mmdet.apis import DetInferencer

torch.set_num_threads(1)
torch.backends.cudnn.benchmark = True

DEVICE = "cuda:0"  
BATCH_SIZE = 16     
RESULTS_ROOT = Path("results_cross_dataset_4x4")

DATASETS = {
    "coco": {
        "image_dir": r"C:\Users\ISI_UTS\Ritabrata_Hrishit\PolarNeXt\datasets\coco\val2017"
    },
    "cityscapes": {
        "image_dir": r"C:\Users\ISI_UTS\Ritabrata_Hrishit\PolarNeXt\datasets\cityscapes\leftImg8bit\val"
    },
    "objects365": {
        "image_dir": r"C:\Users\ISI_UTS\Ritabrata_Hrishit\PolarNeXt\datasets\objects365\images\val\val"
    },
    "bdd100k": {
        "image_dir": r"C:\Users\ISI_UTS\Ritabrata_Hrishit\PolarNeXt\datasets\bdd\val"
    },
}


TRAINED_MODELS = {
    "coco": {
        "config": r"C:\Users\ISI_UTS\Ritabrata_Hrishit\PolarNeXt\projects\PolarNeXt\configs\polarnext_r50-torch_fpn_1x_coco.py",
        "weights": r"C:\Users\ISI_UTS\Ritabrata_Hrishit\PolarNeXt\polarnext_r50_epoch_36.pth",
    },
    "cityscapes": {
        "config": r"C:\Users\ISI_UTS\Ritabrata_Hrishit\PolarNeXt\configs\cityscapes\faster-rcnn_r50_fpn_1x_cityscapes.py",
        "weights": r"C:\Users\ISI_UTS\Ritabrata_Hrishit\PolarNeXt\faster_rcnn_r50_fpn_1x_cityscapes_20200502-829424c0.pth",
    },
    "objects365": {
        "config": r"C:\Users\ISI_UTS\Ritabrata_Hrishit\PolarNeXt\configs\objects365\faster-rcnn_r50_fpn_16xb4-1x_objects365v2.py",
        "weights": r"C:\Users\ISI_UTS\Ritabrata_Hrishit\PolarNeXt\faster_rcnn_r50_fpn_16x4_1x_obj365v2_20221220_175040-5910b015.pth",
    },
    "bdd100k": {
        "config": r"C:\Users\ISI_UTS\Ritabrata_Hrishit\PolarNeXt\bdd100k-models\det\configs\det\faster_rcnn_r50_fpn_1x_det_bdd100k.py",
        "weights": r"C:\Users\ISI_UTS\Ritabrata_Hrishit\PolarNeXt\faster_rcnn_r50_fpn_1x_det_bdd100k.pth",
    },
}

# ==========================================================

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# ==========================================================

def run_cross_dataset_4x4():

    for train_name, train_cfg in TRAINED_MODELS.items():

        print(f"\n==============================")
        print(f" Train domain: {train_name}")
        print(f"==============================")

        inferencer = DetInferencer(
            model=train_cfg["config"],
            weights=train_cfg["weights"],
            device=DEVICE
        )

        class_names = (
            inferencer.model.dataset_meta["classes"]
            if inferencer.model.dataset_meta is not None
            else []
        )

        for test_name, test_cfg in DATASETS.items():

            print(f" → Test domain: {test_name}")

            img_dir = Path(test_cfg["image_dir"])
            out_dir = RESULTS_ROOT / f"train_{train_name}__test_{test_name}" / "json"
            ensure_dir(out_dir)

            images = sorted(
                p for p in img_dir.rglob("*")
                if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
            )

            image_paths = [str(p) for p in images]

            print(f"   Images: {len(images)}")

            img_idx = 0

            for batch in tqdm(
                list(chunked(image_paths, BATCH_SIZE)),
                desc=f"{train_name} → {test_name}",
                leave=False
            ):
                outputs = inferencer(batch)["predictions"]

                for preds in outputs:

                    img_path = images[img_idx]
                    img = cv2.imread(str(img_path))
                    h, w = img.shape[:2]

                    records = []
                    for bbox, score, cls in zip(
                        preds["bboxes"],
                        preds["scores"],
                        preds["labels"]
                    ):
                        cls = int(cls)
                        records.append({
                            "label_id": cls,
                            "label_name": class_names[cls] if cls < len(class_names) else "UNKNOWN",
                            "score": float(score),
                            "bbox_xyxy": list(map(float, bbox))
                        })

                    with open(out_dir / f"{img_path.stem}.json", "w") as f:
                        json.dump({
                            "image": img_path.name,
                            "width": w,
                            "height": h,
                            "train_domain": train_name,
                            "test_domain": test_name,
                            "predictions": records,
                        }, f, indent=2)

                    img_idx += 1

            print(f"   ✔ Saved → {out_dir}")

    print("\nAll 4×4 cross-dataset inference completed.")


if __name__ == "__main__":
    run_cross_dataset_4x4()
