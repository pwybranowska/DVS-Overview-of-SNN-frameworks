"""
evaluate_gen1.py
----------------
Evaluator for GEN1 object detection predictions.

Evaluates prediction JSON files produced by:
  - Norse detector
  - snnTorch detector
  - SpikingJelly detector

Metrics:
  - COCO-style mAP
  - AP50
  - AP75
  - mAR@100
  - Precision@IoU=0.5
  - Recall@IoU=0.5
  - F1@IoU=0.5

Expected prediction JSON format:
{
  "sample_name": {
    "boxes":  [[x1, y1, x2, y2], ...],
    "scores": [0.91, ...],
    "labels": [1, ...]
  }
}

Class mapping:
  Prophesee class_id 0 -> 1 = car
  Prophesee class_id 1 -> 2 = pedestrian
"""

import json
from pathlib import Path

import numpy as np
import torch

try:
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
except ImportError as e:
    raise ImportError(
        "Missing TorchMetrics. Install with:\n"
        "pip install torchmetrics pycocotools"
    ) from e


# ============================================================
# CONFIG
# ============================================================

CONFIG = {
    # Dataset root containing train/val/test folders
    # If evaluate_gen1.py is inside e.g. spikingjelly/, and dataset is one folder above:
    # "data_root": "..",
    #
    # If dataset is in ../download/Gen1EventCubeDataset:
    "data_root": "../download/Gen1EventCubeDataset",

    # Split to evaluate
    "split": "val",

    # IMPORTANT:
    # Must match training code.
    # Use one timestamp, not all bbox timestamps from the .npy file.
    # Options: "first", "middle", "last", "all"
    "bbox_timestamp_mode": "last",

    # Prediction files
    "prediction_files": {
        "Norse": "norse_gen1_predictions.json",
        "snnTorch": "snntorch_gen1_predictions.json",
        "SpikingJelly": "spikingjelly/spikingjelly_gen1_predictions.json",
    },

    # Image size
    "image_w": 304,
    "image_h": 240,

    # F1 matching threshold
    "f1_iou_thresh": 0.5,

    # Output files
    "save_csv": "evaluation_results.csv",
    "save_json": "evaluation_results.json",
}


# Prophesee GEN1 raw class ids -> training/evaluation ids
# 0 = car, 1 = pedestrian
CLASS_MAP = {0: 1, 1: 2}


# ============================================================
# TIMESTAMP FILTERING
# ============================================================

def select_bbox_timestamp(raw, mode="last"):
    """
    Select bounding boxes from one annotation timestamp.

    GEN1 bbox files may contain boxes for many timestamps.
    The detector outputs one prediction set per event cube, so evaluation
    must use the same timestamp selection as training.
    """
    if raw.dtype.names is None:
        return raw

    if "t" in raw.dtype.names:
        time_field = "t"
    elif "ts" in raw.dtype.names:
        time_field = "ts"
    else:
        return raw

    unique_ts = np.unique(raw[time_field])

    if len(unique_ts) == 0:
        return raw

    if mode == "last":
        selected_ts = unique_ts[-1]
    elif mode == "middle":
        selected_ts = unique_ts[len(unique_ts) // 2]
    elif mode == "first":
        selected_ts = unique_ts[0]
    elif mode == "all":
        return raw
    else:
        raise ValueError(
            f"Unknown bbox_timestamp_mode: {mode}. "
            "Use 'first', 'middle', 'last', or 'all'."
        )

    return raw[raw[time_field] == selected_ts]


# ============================================================
# BASIC BOX UTILS
# ============================================================

def box_iou_single_to_many(box, boxes):
    """
    IoU between one box [4] and many boxes [N, 4].
    Format: xyxy.
    """
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.float32)

    x1 = torch.maximum(box[0], boxes[:, 0])
    y1 = torch.maximum(box[1], boxes[:, 1])
    x2 = torch.minimum(box[2], boxes[:, 2])
    y2 = torch.minimum(box[3], boxes[:, 3])

    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)
    inter = inter_w * inter_h

    area_box = (box[2] - box[0]).clamp(min=0) * (box[3] - box[1]).clamp(min=0)
    area_boxes = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * \
                 (boxes[:, 3] - boxes[:, 1]).clamp(min=0)

    union = area_box + area_boxes - inter
    return inter / union.clamp(min=1e-6)


def clip_boxes_xyxy(boxes, image_w, image_h):
    """
    Clip boxes to image boundaries.
    """
    if boxes.numel() == 0:
        return boxes

    boxes[:, 0] = boxes[:, 0].clamp(0, image_w - 1)
    boxes[:, 2] = boxes[:, 2].clamp(0, image_w - 1)
    boxes[:, 1] = boxes[:, 1].clamp(0, image_h - 1)
    boxes[:, 3] = boxes[:, 3].clamp(0, image_h - 1)

    return boxes


def remove_invalid_boxes(boxes, scores=None, labels=None):
    """
    Remove boxes with non-positive width/height or NaN/Inf.
    """
    if boxes.numel() == 0:
        return boxes, scores, labels

    valid = torch.isfinite(boxes).all(dim=1)
    valid &= (boxes[:, 2] > boxes[:, 0])
    valid &= (boxes[:, 3] > boxes[:, 1])

    boxes = boxes[valid]

    if scores is not None:
        scores = scores[valid]
    if labels is not None:
        labels = labels[valid]

    return boxes, scores, labels


# ============================================================
# LOADING PREDICTIONS AND GROUND TRUTH
# ============================================================

def load_predictions(pred_path):
    """
    Load prediction JSON.
    """
    pred_path = Path(pred_path)

    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_path}")

    with pred_path.open("r") as f:
        return json.load(f)


def load_gt_boxes(npy_path):
    """
    Load GEN1 ground truth boxes from *_bbox.npy.

    Returns:
      boxes:  Tensor[M, 4] in xyxy format
      labels: Tensor[M]
    """
    npy_path = Path(npy_path)

    if not npy_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {npy_path}")

    raw = np.load(npy_path, allow_pickle=True)

    # CRITICAL FIX:
    # Select only one annotation timestamp, not all timestamps.
    raw = select_bbox_timestamp(
        raw,
        mode=CONFIG["bbox_timestamp_mode"]
    )

    boxes = []
    labels = []

    if raw.dtype.names is not None:
        # Structured Prophesee format:
        # fields usually include: ts/t, x, y, w, h, class_id, ...
        for row in raw:
            cls = int(row["class_id"])
            if cls not in CLASS_MAP:
                continue

            x1 = float(row["x"])
            y1 = float(row["y"])
            x2 = x1 + float(row["w"])
            y2 = y1 + float(row["h"])

            boxes.append([x1, y1, x2, y2])
            labels.append(CLASS_MAP[cls])

    else:
        # Fallback for plain arrays:
        # [M, 5] = x1, y1, x2, y2, cls
        # [M, 6] = t, x1, y1, x2, y2, cls
        raw = raw.reshape(-1, raw.shape[-1]) if raw.ndim == 1 else raw
        offset = 1 if raw.shape[1] == 6 else 0

        for row in raw:
            x1, y1, x2, y2 = row[offset:offset + 4]
            cls = int(row[offset + 4])

            if cls not in CLASS_MAP:
                continue

            boxes.append([float(x1), float(y1), float(x2), float(y2)])
            labels.append(CLASS_MAP[cls])

    if len(boxes) == 0:
        return (
            torch.zeros((0, 4), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.int64),
        )

    boxes = torch.tensor(boxes, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.int64)

    boxes = clip_boxes_xyxy(boxes, CONFIG["image_w"], CONFIG["image_h"])
    boxes, _, labels = remove_invalid_boxes(boxes, None, labels)

    return boxes, labels


def prediction_to_torchmetrics(sample_pred):
    """
    Convert one sample prediction dict to TorchMetrics format.
    """
    boxes = torch.tensor(sample_pred.get("boxes", []), dtype=torch.float32)
    scores = torch.tensor(sample_pred.get("scores", []), dtype=torch.float32)
    labels = torch.tensor(sample_pred.get("labels", []), dtype=torch.int64)

    if boxes.numel() == 0:
        boxes = torch.zeros((0, 4), dtype=torch.float32)
        scores = torch.zeros((0,), dtype=torch.float32)
        labels = torch.zeros((0,), dtype=torch.int64)
    else:
        boxes = boxes.reshape(-1, 4)
        scores = scores.reshape(-1)
        labels = labels.reshape(-1)

        boxes = clip_boxes_xyxy(boxes, CONFIG["image_w"], CONFIG["image_h"])
        boxes, scores, labels = remove_invalid_boxes(boxes, scores, labels)

    return {
        "boxes": boxes,
        "scores": scores,
        "labels": labels,
    }


def make_torchmetrics_lists(pred_json, data_root, split):
    """
    Builds preds and targets lists for TorchMetrics.

    Uses .h5 files in the split folder as the reference sample list.
    This ensures the evaluator checks all validation samples.
    """
    split_dir = Path(data_root) / split

    if not split_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {split_dir}")

    stems = sorted([
        p.stem for p in split_dir.glob("*.h5")
        if not p.stem.endswith("_bbox")
    ])

    if len(stems) == 0:
        raise FileNotFoundError(f"No .h5 files found in {split_dir}")

    preds = []
    targets = []

    missing_predictions = 0

    for stem in stems:
        # Prediction
        if stem in pred_json:
            pred = prediction_to_torchmetrics(pred_json[stem])
        else:
            missing_predictions += 1
            pred = {
                "boxes": torch.zeros((0, 4), dtype=torch.float32),
                "scores": torch.zeros((0,), dtype=torch.float32),
                "labels": torch.zeros((0,), dtype=torch.int64),
            }

        # Ground truth
        gt_path = split_dir / f"{stem}_bbox.npy"
        gt_boxes, gt_labels = load_gt_boxes(gt_path)

        target = {
            "boxes": gt_boxes,
            "labels": gt_labels,
        }

        preds.append(pred)
        targets.append(target)

    if missing_predictions > 0:
        print(
            f"[Warning] Missing predictions for {missing_predictions} samples. "
            f"They were treated as empty predictions."
        )

    return preds, targets


# ============================================================
# TORCHMETRICS mAP
# ============================================================

def evaluate_map(preds, targets):
    """
    COCO-style mAP evaluation.
    """
    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        class_metrics=True,
    )

    metric.update(preds, targets)
    result = metric.compute()

    # Convert tensors to Python floats/lists
    clean = {}
    for key, value in result.items():
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                clean[key] = float(value.item())
            else:
                clean[key] = value.detach().cpu().tolist()
        else:
            clean[key] = value

    return clean


# ============================================================
# PRECISION / RECALL / F1 @ IoU
# ============================================================

def evaluate_f1_at_iou(preds, targets, iou_thresh=0.5):
    """
    Simple global Precision/Recall/F1 at a fixed IoU threshold.

    Matching rules:
      - class label must match
      - IoU must be >= threshold
      - one GT box can be matched only once
      - predictions are processed in descending score order
    """
    total_tp = 0
    total_fp = 0
    total_fn = 0

    per_class = {
        1: {"tp": 0, "fp": 0, "fn": 0},  # car
        2: {"tp": 0, "fp": 0, "fn": 0},  # pedestrian
    }

    for pred, target in zip(preds, targets):
        p_boxes = pred["boxes"]
        p_scores = pred["scores"]
        p_labels = pred["labels"]

        t_boxes = target["boxes"]
        t_labels = target["labels"]

        # Sort predictions by confidence descending
        if p_scores.numel() > 0:
            order = torch.argsort(p_scores, descending=True)
            p_boxes = p_boxes[order]
            p_scores = p_scores[order]
            p_labels = p_labels[order]

        matched_gt = torch.zeros((t_boxes.shape[0],), dtype=torch.bool)

        sample_tp = 0
        sample_fp = 0

        for i in range(p_boxes.shape[0]):
            pred_box = p_boxes[i]
            pred_label = int(p_labels[i].item())

            same_class = (t_labels == pred_label)
            available = ~matched_gt
            candidates = same_class & available

            if candidates.sum() == 0:
                sample_fp += 1
                if pred_label in per_class:
                    per_class[pred_label]["fp"] += 1
                continue

            candidate_indices = torch.where(candidates)[0]
            candidate_boxes = t_boxes[candidate_indices]

            ious = box_iou_single_to_many(pred_box, candidate_boxes)
            best_iou, best_local_idx = ious.max(dim=0)

            if best_iou.item() >= iou_thresh:
                matched_index = candidate_indices[best_local_idx]
                matched_gt[matched_index] = True

                sample_tp += 1
                if pred_label in per_class:
                    per_class[pred_label]["tp"] += 1
            else:
                sample_fp += 1
                if pred_label in per_class:
                    per_class[pred_label]["fp"] += 1

        sample_fn = int((~matched_gt).sum().item())

        # Count FN per class
        unmatched_indices = torch.where(~matched_gt)[0]
        for idx in unmatched_indices:
            cls = int(t_labels[idx].item())
            if cls in per_class:
                per_class[cls]["fn"] += 1

        total_tp += sample_tp
        total_fp += sample_fp
        total_fn += sample_fn

    precision = total_tp / max(total_tp + total_fp, 1)
    recall = total_tp / max(total_tp + total_fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    per_class_metrics = {}
    for cls, vals in per_class.items():
        tp = vals["tp"]
        fp = vals["fp"]
        fn = vals["fn"]

        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        f = 2 * p * r / max(p + r, 1e-12)

        name = "car" if cls == 1 else "pedestrian"
        per_class_metrics[name] = {
            "precision": p,
            "recall": r,
            "f1": f,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "per_class": per_class_metrics,
    }


# ============================================================
# EVALUATE ONE MODEL
# ============================================================

def evaluate_model(model_name, pred_path):
    print(f"\nEvaluating: {model_name}")
    print(f"Predictions: {pred_path}")

    pred_json = load_predictions(pred_path)

    preds, targets = make_torchmetrics_lists(
        pred_json=pred_json,
        data_root=CONFIG["data_root"],
        split=CONFIG["split"],
    )

    # Debug stats — very important for checking timestamp filtering.
    print("Total samples:", len(targets))
    print("Total predictions:", sum(len(p["boxes"]) for p in preds))
    print("Total GT boxes:", sum(len(t["boxes"]) for t in targets))

    map_result = evaluate_map(preds, targets)

    f1_result = evaluate_f1_at_iou(
        preds,
        targets,
        iou_thresh=CONFIG["f1_iou_thresh"],
    )

    row = {
        "model": model_name,

        # TorchMetrics COCO-style metrics
        "mAP": map_result.get("map", -1.0),
        "AP50": map_result.get("map_50", -1.0),
        "AP75": map_result.get("map_75", -1.0),
        "mAR@100": map_result.get("mar_100", -1.0),

        # Simple global metrics at IoU=0.5
        "Precision@0.5": f1_result["precision"],
        "Recall@0.5": f1_result["recall"],
        "F1@0.5": f1_result["f1"],

        "TP": f1_result["tp"],
        "FP": f1_result["fp"],
        "FN": f1_result["fn"],

        # Optional per-class metrics
        "car_F1@0.5": f1_result["per_class"]["car"]["f1"],
        "pedestrian_F1@0.5": f1_result["per_class"]["pedestrian"]["f1"],
    }

    return row


# ============================================================
# OUTPUT
# ============================================================

def print_results_table(rows):
    print("\n" + "=" * 120)
    print("GEN1 OBJECT DETECTION EVALUATION")
    print("=" * 120)

    print(
        f"{'Model':<15} "
        f"{'mAP':>8} "
        f"{'AP50':>8} "
        f"{'AP75':>8} "
        f"{'mAR@100':>9} "
        f"{'Prec@.5':>9} "
        f"{'Rec@.5':>9} "
        f"{'F1@.5':>9} "
        f"{'CarF1':>9} "
        f"{'PedF1':>9} "
        f"{'TP':>6} "
        f"{'FP':>6} "
        f"{'FN':>6}"
    )

    print("-" * 120)

    for r in rows:
        print(
            f"{r['model']:<15} "
            f"{r['mAP']:>8.4f} "
            f"{r['AP50']:>8.4f} "
            f"{r['AP75']:>8.4f} "
            f"{r['mAR@100']:>9.4f} "
            f"{r['Precision@0.5']:>9.4f} "
            f"{r['Recall@0.5']:>9.4f} "
            f"{r['F1@0.5']:>9.4f} "
            f"{r['car_F1@0.5']:>9.4f} "
            f"{r['pedestrian_F1@0.5']:>9.4f} "
            f"{r['TP']:>6} "
            f"{r['FP']:>6} "
            f"{r['FN']:>6}"
        )

    print("=" * 120)


def save_results(rows):
    # Save JSON
    json_path = Path(CONFIG["save_json"])
    json_path.write_text(json.dumps(rows, indent=2))
    print(f"Saved JSON results -> {json_path}")

    # Save CSV
    csv_path = Path(CONFIG["save_csv"])

    if len(rows) == 0:
        return

    headers = list(rows[0].keys())

    with csv_path.open("w", encoding="utf-8") as f:
        f.write(",".join(headers) + "\n")

        for row in rows:
            values = []
            for h in headers:
                v = row[h]
                if isinstance(v, float):
                    values.append(f"{v:.8f}")
                else:
                    values.append(str(v))
            f.write(",".join(values) + "\n")

    print(f"Saved CSV results -> {csv_path}")


# ============================================================
# MAIN
# ============================================================

def main():
    rows = []

    for model_name, pred_path in CONFIG["prediction_files"].items():
        pred_path = Path(pred_path)

        if not pred_path.exists():
            print(f"[Skipping] {model_name}: missing file {pred_path}")
            continue

        row = evaluate_model(model_name, pred_path)
        rows.append(row)

    if len(rows) == 0:
        print("No prediction files were found. Nothing to evaluate.")
        return

    print_results_table(rows)
    save_results(rows)


if __name__ == "__main__":
    main()