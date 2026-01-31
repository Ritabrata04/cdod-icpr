import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pathlib import Path

def evaluate_coco(gt_json, pred_json, iou_type="bbox"):
    """
    Evaluate predictions against COCO-format ground truth.

    Args:
        gt_json (str): path to ground truth COCO JSON
        pred_json (str): path to predictions COCO JSON
        iou_type (str): 'bbox' for box AP, 'segm' for mask AP
    """
    print(f"\n[INFO] Loading GT: {gt_json}")
    coco_gt = COCO(gt_json)

    print(f"[INFO] Loading Predictions: {pred_json}")
    coco_dt = coco_gt.loadRes(pred_json)

    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate COCO-format predictions")
    parser.add_argument("--gt", type=str, required=True, help="Path to COCO-format ground truth JSON")
    parser.add_argument("--pred", type=str, required=True, help="Path to COCO-format predictions JSON")
    args = parser.parse_args()

    evaluate_coco(args.gt, args.pred, iou_type="bbox")
