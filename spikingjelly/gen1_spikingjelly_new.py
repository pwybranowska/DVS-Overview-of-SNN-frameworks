"""
gen1_spikingjelly_snn.py
------------------------
SpikingJelly convolutional SNN object detector for the GEN1 EventCube dataset.

This version is intentionally matched as closely as possible to the provided
Norse implementation:
  - same dataset strategy
  - same bbox loading policy: all bbox timestamps are used
  - same anchor settings
  - same SSD-style detection head
  - same focal loss
  - same post-processing
  - same logit-accumulation readout over time

Only the framework-specific spiking blocks are changed from Norse to SpikingJelly.

Expected dataset layout:
  download/Gen1EventCubeDataset/
    train/   <name>.h5  +  <name>_bbox.npy
    val/     <name>.h5  +  <name>_bbox.npy
    test/    <name>.h5  +  <name>_bbox.npy
"""

import math
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.ops import batched_nms

from spikingjelly.activation_based import neuron, surrogate, functional


# ============================================================
#  CONFIG
# ============================================================
CONFIG = {
    # ---- paths ----
    "data_root":     "download/Gen1EventCubeDataset",
    # "data_root":  ".",
    "ckpt_path":     "spikingjelly_gen1_final.pt",
    "pred_path":     "spikingjelly_gen1_final_predictions.json",

    # ---- mode: "train" | "eval" | "infer" ----
    "mode":          "train",

    # ---- data ----
    "image_h":       240,
    "image_w":       304,
    "in_channels":   10,
    "num_bins":      10,
    "num_workers":   4,
    "train_fraction": 1.0,

    # ---- model ----
    "base_channels": 64,
    "num_classes":   3,       # 0=background  1=car  2=pedestrian
    "num_anchors":   9,
    "dropout":       0.1,

    # ---- SpikingJelly LIF parameters ----
    # Matched qualitatively to the provided Norse version with low threshold.
    # Norse uses v_th=0.3. SpikingJelly uses v_threshold directly.
    "lif_tau_per_block": [2.07, 2.44, 1.95, 2.17],
    "lif_decay_input": False,
    "lif_threshold": 0.3,
    "lif_reset":     0.0,
    "lif_detach_reset": False,

    # ---- training ----
    "epochs":        50,
    "batch_size":    8,
    "lr":            1e-3,
    "weight_decay":  5e-4,
    "grad_clip":     1.0,
    "seed":          42,

    # ---- anchor matching ----
    "iou_pos":       0.5,
    "iou_neg":       0.3,
    "anchor_scales": (2.0, 4.0, 8.0),
    "anchor_ratios": (0.5, 1.0, 2.0),

    # ---- focal loss ----
    "focal_gamma":   2.0,
    "focal_alpha":   0.25,

    # ---- post-processing ----
    "score_thresh":  0.05,
    "nms_thresh":    0.5,
    "top_k":         100,
}

# Prophesee GEN1: raw class_id 0=car -> training 1, raw 1=pedestrian -> training 2
CLASS_MAP = {0: 1, 1: 2}


# ============================================================
#  DATASET
# ============================================================
class GEN1Dataset(Dataset):
    """
    Reads pre-voxelised .h5 cubes + Prophesee _bbox.npy files.

    This is matched to the provided Norse implementation:
      - reads only num_bins frames using h5py list indexing
      - pads 6-channel samples to 10 channels
      - loads all bbox entries from *_bbox.npy, without timestamp filtering
      - supports horizontal flip augmentation for training
    """

    def __init__(self, split: str, augment: bool = False):
        self.dir = Path(CONFIG["data_root"]) / split
        self.num_bins = CONFIG["num_bins"]
        self.augment = augment

        if not self.dir.exists():
            raise FileNotFoundError(f"Split not found: {self.dir}")

        all_h5 = sorted([
            p.stem for p in self.dir.glob("*.h5")
            if not p.stem.endswith("_bbox")
        ])

        self.samples = [
            s for s in all_h5
            if (self.dir / f"{s}_bbox.npy").exists()
        ]

        skipped = len(all_h5) - len(self.samples)
        if skipped:
            print(f"[Dataset] {split}: skipped {skipped} samples missing _bbox.npy")

        if split == "train":
            frac = CONFIG.get("train_fraction", 1.0)
            if frac < 1.0:
                n = max(1, int(len(self.samples) * frac))
                rng = np.random.default_rng(CONFIG["seed"])
                idx = rng.choice(len(self.samples), n, replace=False)
                self.samples = [self.samples[i] for i in sorted(idx)]
                print(f"[Dataset] {split}: using {n} samples ({frac * 100:.0f}%)")

        if not self.samples:
            raise FileNotFoundError(f"No samples in {self.dir}")

        print(f"[Dataset] {split}: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def _load_cube(self, stem: str) -> np.ndarray:
        with h5py.File(self.dir / f"{stem}.h5", "r") as f:
            key = list(f.keys())[0]
            T_full = f[key].shape[0]
            C = f[key].shape[1]
            idx = np.linspace(0, T_full - 1, self.num_bins, dtype=int).tolist()
            out = f[key][idx].astype(np.float32)  # [T, C, H, W]

        if C < CONFIG["in_channels"]:
            T, _, H, W = out.shape
            pad = np.zeros((T, CONFIG["in_channels"] - C, H, W), dtype=np.float32)
            out = np.concatenate([out, pad], axis=1)

        if out.shape[1] > CONFIG["in_channels"]:
            out = out[:, :CONFIG["in_channels"], :, :]

        mx = out.max()
        if mx > 0:
            out /= mx

        return out

    def _load_boxes(self, stem: str) -> torch.Tensor:
        raw = np.load(self.dir / f"{stem}_bbox.npy", allow_pickle=True)
        boxes = []

        if raw.dtype.names is not None:
            for row in raw:
                cls = int(row["class_id"])
                if cls not in CLASS_MAP:
                    continue

                x1 = float(row["x"])
                y1 = float(row["y"])
                x2 = x1 + float(row["w"])
                y2 = y1 + float(row["h"])

                boxes.append([x1, y1, x2, y2, float(CLASS_MAP[cls])])
        else:
            raw = raw.reshape(-1, raw.shape[-1]) if raw.ndim == 1 else raw
            offset = 1 if raw.shape[1] == 6 else 0

            for row in raw:
                x1, y1, x2, y2 = row[offset:offset + 4]
                cls = int(row[offset + 4])
                if cls not in CLASS_MAP:
                    continue

                boxes.append([
                    float(x1), float(y1), float(x2), float(y2),
                    float(CLASS_MAP[cls])
                ])

        if boxes:
            return torch.tensor(boxes, dtype=torch.float32)
        return torch.zeros((0, 5), dtype=torch.float32)

    def __getitem__(self, idx):
        stem = self.samples[idx]
        cube = self._load_cube(stem)
        target = self._load_boxes(stem)

        # Horizontal flip augmentation, matched to Norse version.
        if self.augment and np.random.rand() > 0.5:
            cube = cube[:, :, :, ::-1].copy()
            if target.shape[0] > 0:
                W = CONFIG["image_w"]
                flipped = target.clone()
                flipped[:, 0] = W - target[:, 2]
                flipped[:, 2] = W - target[:, 0]
                target = flipped

        return torch.from_numpy(cube), target, stem


def collate_fn(batch):
    cubes, targets, stems = zip(*batch)
    return torch.stack(cubes, 0), list(targets), list(stems)


# ============================================================
#  MODEL
# ============================================================
class ConvSpikeBlock(nn.Module):
    """
    Conv2d -> BatchNorm2d -> SpikingJelly LIFNode -> Dropout
    Input/output: [T, B, C, H, W]
    """

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dropout: float = 0.0, name: str = "", tau: float = 2.0):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

        self.lif = neuron.LIFNode(
            tau=tau,
            decay_input=CONFIG["lif_decay_input"],
            v_threshold=CONFIG["lif_threshold"],
            v_reset=CONFIG["lif_reset"],
            surrogate_function=surrogate.ATan(),
            detach_reset=CONFIG["lif_detach_reset"],
            step_mode="m",
        )

        self.drop = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        T, B, C, H, W = x.shape

        x = self.bn(self.conv(x.reshape(T * B, C, H, W)))
        _, Co, Hh, Ww = x.shape
        x = x.reshape(T, B, Co, Hh, Ww)

        x = self.lif(x)

        if not isinstance(self.drop, nn.Identity):
            x = self.drop(x.reshape(T * B, Co, Hh, Ww)).reshape(T, B, Co, Hh, Ww)

        return x


class SpikingJellySSD(nn.Module):
    """
    4-block SNN backbone + SSD head with logit accumulation readout.

    This mirrors the provided Norse readout:
      - backbone produces spiking features for all timesteps
      - detection heads are applied at every timestep
      - continuous logits are averaged over time
    """

    def __init__(self):
        super().__init__()

        B = CONFIG["base_channels"]
        A = CONFIG["num_anchors"]
        C = CONFIG["num_classes"]
        ic = CONFIG["in_channels"]
        dr = CONFIG["dropout"]

        taus = CONFIG["lif_tau_per_block"]

        self.block1 = ConvSpikeBlock(ic,  B,    stride=2, dropout=dr, name="block1", tau=taus[0])
        self.block2 = ConvSpikeBlock(B,   B*2,  stride=2, dropout=dr, name="block2", tau=taus[1])
        self.block3 = ConvSpikeBlock(B*2, B*4,  stride=2, dropout=dr, name="block3", tau=taus[2])
        self.block4 = ConvSpikeBlock(B*4, B*4,  stride=1, dropout=dr, name="block4", tau=taus[3])

        self.loc_head = nn.Conv2d(B * 4, A * 4, 3, padding=1)
        self.cls_head = nn.Conv2d(B * 4, A * C, 3, padding=1)

        self._A = A
        self._C = C

    def forward(self, x):
        # [B, T, C, H, W] -> [T, B, C, H, W]
        x = x.permute(1, 0, 2, 3, 4).contiguous()

        functional.reset_net(self)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        T, B2, C, Hf, Wf = x.shape

        # Logit accumulation readout, matched to Norse implementation.
        x_flat = x.reshape(T * B2, C, Hf, Wf)

        loc_flat = self.loc_head(x_flat)
        cls_flat = self.cls_head(x_flat)

        loc_t = loc_flat.reshape(T, B2, self._A * 4, Hf, Wf)
        cls_t = cls_flat.reshape(T, B2, self._A * self._C, Hf, Wf)

        loc_mean = loc_t.mean(dim=0)
        cls_mean = cls_t.mean(dim=0)

        loc = loc_mean.reshape(B2, self._A, 4, Hf, Wf)
        cls = cls_mean.reshape(B2, self._A, self._C, Hf, Wf)

        return {"loc": loc, "cls": cls}


# ============================================================
#  ANCHORS
# ============================================================
def generate_anchors(feat_h, feat_w):
    img_h = CONFIG["image_h"]
    img_w = CONFIG["image_w"]
    scales = CONFIG["anchor_scales"]
    ratios = CONFIG["anchor_ratios"]
    sy = img_h / feat_h
    sx = img_w / feat_w

    anchors = []
    for iy in range(feat_h):
        for ix in range(feat_w):
            cx = (ix + 0.5) * sx
            cy = (iy + 0.5) * sy
            for s in scales:
                for r in ratios:
                    w = s * math.sqrt(r) * sx
                    h = s / math.sqrt(r) * sy
                    anchors.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    return torch.tensor(anchors, dtype=torch.float32)


def build_anchors(model, device):
    model.eval()
    with torch.no_grad():
        functional.reset_net(model)
        dummy = torch.zeros(
            1,
            CONFIG["num_bins"],
            CONFIG["in_channels"],
            CONFIG["image_h"],
            CONFIG["image_w"],
            device=device,
        )
        out = model(dummy)
        functional.reset_net(model)

    _, A, _, Hf, Wf = out["loc"].shape
    anchors = generate_anchors(Hf, Wf).to(device)
    expected = A * Hf * Wf
    if anchors.shape[0] < expected:
        anchors = anchors.repeat(math.ceil(expected / anchors.shape[0]), 1)
    return anchors[:expected]


# ============================================================
#  BOX UTILITIES
# ============================================================
def xyxy_to_cxcywh(b):
    x1, y1, x2, y2 = b.unbind(-1)
    return torch.stack([
        (x1 + x2) / 2,
        (y1 + y2) / 2,
        (x2 - x1).clamp(1e-6),
        (y2 - y1).clamp(1e-6),
    ], -1)


def cxcywh_to_xyxy(b):
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([
        cx - w / 2,
        cy - h / 2,
        cx + w / 2,
        cy + h / 2,
    ], -1)


def box_iou(a, b):
    if a.numel() == 0 or b.numel() == 0:
        return torch.zeros(a.shape[0], b.shape[0], device=a.device)

    area_a = (a[:, 2] - a[:, 0]).clamp(0) * (a[:, 3] - a[:, 1]).clamp(0)
    area_b = (b[:, 2] - b[:, 0]).clamp(0) * (b[:, 3] - b[:, 1]).clamp(0)

    lt = torch.max(a[:, None, :2], b[:, :2])
    rb = torch.min(a[:, None, 2:], b[:, 2:])
    inter = (rb - lt).clamp(0).prod(-1)

    return inter / (area_a[:, None] + area_b - inter).clamp(1e-6)


# ============================================================
#  ANCHOR MATCHING & ENCODING
# ============================================================
def match_and_encode(anchors, gt):
    A = anchors.shape[0]
    device = anchors.device

    cls_t = torch.full((A,), -1, dtype=torch.long, device=device)
    box_t = torch.zeros((A, 4), dtype=torch.float32, device=device)

    if gt.shape[0] == 0:
        cls_t[:] = 0
        return cls_t, box_t

    gt_boxes = gt[:, :4]
    gt_labels = gt[:, 4].long()

    ious = box_iou(anchors, gt_boxes)
    best_iou, best_g = ious.max(dim=1)

    cls_t[best_iou < CONFIG["iou_neg"]] = 0
    cls_t[best_iou >= CONFIG["iou_pos"]] = gt_labels[best_g[best_iou >= CONFIG["iou_pos"]]]

    # Force-match each GT.
    for g in range(gt_boxes.shape[0]):
        cls_t[ious[:, g].argmax()] = gt_labels[g]

    anc_c = xyxy_to_cxcywh(anchors)
    gt_c = xyxy_to_cxcywh(gt_boxes[best_g])

    box_t[:, 0] = (gt_c[:, 0] - anc_c[:, 0]) / anc_c[:, 2].clamp(1e-6)
    box_t[:, 1] = (gt_c[:, 1] - anc_c[:, 1]) / anc_c[:, 3].clamp(1e-6)
    box_t[:, 2] = torch.log((gt_c[:, 2] / anc_c[:, 2].clamp(1e-6)).clamp(1e-6)).clamp(-10, 10)
    box_t[:, 3] = torch.log((gt_c[:, 3] / anc_c[:, 3].clamp(1e-6)).clamp(1e-6)).clamp(-10, 10)

    return cls_t, box_t


def decode_boxes(anchors, deltas):
    anc = xyxy_to_cxcywh(anchors)

    cx = deltas[:, 0] * anc[:, 2] + anc[:, 0]
    cy = deltas[:, 1] * anc[:, 3] + anc[:, 1]
    w = torch.exp(deltas[:, 2]).clamp(max=1e3) * anc[:, 2]
    h = torch.exp(deltas[:, 3]).clamp(max=1e3) * anc[:, 3]

    return cxcywh_to_xyxy(torch.stack([cx, cy, w, h], -1))


# ============================================================
#  FOCAL LOSS
# ============================================================
class DetectionLoss(nn.Module):
    """
    Focal loss for classification + Smooth-L1 for box regression.
    Matched to the provided Norse implementation.
    """

    def __init__(self):
        super().__init__()
        self.gamma = CONFIG["focal_gamma"]
        self.alpha = CONFIG["focal_alpha"]

    def forward(self, pred_loc, pred_cls, anchors, targets):
        B, A, _, H, W = pred_loc.shape
        N = A * H * W
        C = CONFIG["num_classes"]
        device = pred_loc.device

        loc_flat = pred_loc.permute(0, 3, 4, 1, 2).contiguous().reshape(B, N, 4)
        cls_flat = pred_cls.permute(0, 3, 4, 1, 2).contiguous().reshape(B, N, C)

        tot_cls = torch.tensor(0.0, device=device)
        tot_box = torch.tensor(0.0, device=device)
        num_pos = 0

        for b in range(B):
            cls_t, box_t = match_and_encode(anchors, targets[b].to(device))
            valid = cls_t >= 0
            pos = cls_t > 0
            num_pos += int(pos.sum())

            if valid.any():
                ce = F.cross_entropy(cls_flat[b][valid], cls_t[valid], reduction="none")
                pt = torch.exp(-ce)
                a_w = torch.where(
                    cls_t[valid] > 0,
                    torch.tensor(self.alpha, device=device),
                    torch.tensor(1.0 - self.alpha, device=device),
                )
                tot_cls = tot_cls + (a_w * (1 - pt) ** self.gamma * ce).mean()

            if pos.any():
                tot_box = tot_box + F.smooth_l1_loss(
                    loc_flat[b][pos], box_t[pos], reduction="mean"
                )

        cls_loss = tot_cls / max(B, 1)
        box_loss = tot_box / max(B, 1)
        loss = cls_loss + box_loss

        return loss, {
            "loss": float(loss.detach()),
            "cls_loss": float(cls_loss.detach()),
            "box_loss": float(box_loss.detach()),
            "num_pos": num_pos,
        }


# ============================================================
#  POST-PROCESSING
# ============================================================
@torch.no_grad()
def post_process(pred_loc, pred_cls, anchors):
    B, A, _, H, W = pred_loc.shape
    N = A * H * W
    C = CONFIG["num_classes"]

    loc = pred_loc.permute(0, 3, 4, 1, 2).contiguous().reshape(B, N, 4)
    cls = pred_cls.permute(0, 3, 4, 1, 2).contiguous().reshape(B, N, C)

    results = []
    for b in range(B):
        probs = F.softmax(cls[b], dim=-1)
        scores, labels = probs[:, 1:].max(dim=-1)
        labels = labels + 1

        keep = scores >= CONFIG["score_thresh"]
        if not keep.any():
            results.append({"boxes": [], "scores": [], "labels": []})
            continue

        s_ = scores[keep]
        l_ = labels[keep]
        b_ = decode_boxes(anchors[keep], loc[b][keep])

        b_[:, [0, 2]] = b_[:, [0, 2]].clamp(0, CONFIG["image_w"])
        b_[:, [1, 3]] = b_[:, [1, 3]].clamp(0, CONFIG["image_h"])

        idx = batched_nms(b_, s_, l_, CONFIG["nms_thresh"])[:CONFIG["top_k"]]

        results.append({
            "boxes": [b_[i].tolist() for i in idx],
            "scores": [float(s_[i]) for i in idx],
            "labels": [int(l_[i]) for i in idx],
        })

    return results


# ============================================================
#  TRAINING / EVAL LOOPS
# ============================================================
def train_one_epoch(model, loader, anchors, criterion, optimizer, epoch):
    model.train()
    device = next(model.parameters()).device

    total = {"loss": 0.0, "cls_loss": 0.0, "box_loss": 0.0, "num_pos": 0}
    n = 0

    for i, (x, targets, _) in enumerate(loader):
        x = x.to(device, non_blocking=True)

        functional.reset_net(model)
        optimizer.zero_grad(set_to_none=True)

        out = model(x)
        loss, stats = criterion(out["loc"], out["cls"], anchors, targets)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
        optimizer.step()
        functional.reset_net(model)

        for k in total:
            total[k] += stats[k]
        n += 1

        if i % 20 == 0:
            print(
                f"  epoch {epoch:03d}  step {i:04d}/{len(loader)}  "
                f"loss={stats['loss']:.4f}  cls={stats['cls_loss']:.4f}  "
                f"box={stats['box_loss']:.4f}  pos={stats['num_pos']}"
            )

    return {k: v / max(n, 1) if k != "num_pos" else v for k, v in total.items()}


@torch.no_grad()
def eval_one_epoch(model, loader, anchors, criterion):
    model.eval()
    device = next(model.parameters()).device

    total = {"loss": 0.0, "cls_loss": 0.0, "box_loss": 0.0}
    n = 0

    for x, targets, _ in loader:
        x = x.to(device, non_blocking=True)

        functional.reset_net(model)
        out = model(x)
        _, stats = criterion(out["loc"], out["cls"], anchors, targets)
        functional.reset_net(model)

        for k in total:
            total[k] += stats[k]
        n += 1

    return {k: v / max(n, 1) for k, v in total.items()}


@torch.no_grad()
def run_inference(model, loader, anchors):
    model.eval()
    device = next(model.parameters()).device
    all_preds = {}

    for x, _, stems in loader:
        x = x.to(device, non_blocking=True)

        functional.reset_net(model)
        out = model(x)
        dets = post_process(out["loc"], out["cls"], anchors)
        functional.reset_net(model)

        for stem, det in zip(stems, dets):
            all_preds[stem] = det

    return all_preds


def save_predictions_json(preds, path):
    import json
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(preds, indent=2))
    print(f"Predictions saved -> {path}")


# ============================================================
#  MAIN
# ============================================================
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    set_seed(CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device : {device}")
    print(f"Mode   : {CONFIG['mode']}")

    model = SpikingJellySSD().to(device)
    anchors = build_anchors(model, device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Anchors: {anchors.shape[0]}   Params: {n_params:,}")

    criterion = DetectionLoss()

    def make_loader(split, shuffle, augment=False):
        ds = GEN1Dataset(split, augment=augment)
        return DataLoader(
            ds,
            batch_size=CONFIG["batch_size"],
            shuffle=shuffle,
            num_workers=CONFIG["num_workers"],
            collate_fn=collate_fn,
            pin_memory=(device.type == "cuda"),
        )

    if CONFIG["mode"] == "train":
        train_loader = make_loader("train", shuffle=True, augment=True)
        val_loader = make_loader("val", shuffle=False, augment=False)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=CONFIG["lr"],
            weight_decay=CONFIG["weight_decay"],
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=CONFIG["epochs"],
        )

        best_val = float("inf")

        for epoch in range(1, CONFIG["epochs"] + 1):
            tr = train_one_epoch(model, train_loader, anchors, criterion, optimizer, epoch)
            va = eval_one_epoch(model, val_loader, anchors, criterion)
            scheduler.step()

            print(
                f"Epoch {epoch:03d}/{CONFIG['epochs']}  |  "
                f"train loss={tr['loss']:.4f} cls={tr['cls_loss']:.4f} "
                f"box={tr['box_loss']:.4f} pos={int(tr['num_pos'])}  |  "
                f"val loss={va['loss']:.4f} cls={va['cls_loss']:.4f} "
                f"box={va['box_loss']:.4f}"
            )

            if va["loss"] < best_val:
                best_val = va["loss"]
                torch.save(
                    {
                        "epoch": epoch,
                        "model": model.state_dict(),
                        "anchors": anchors.cpu(),
                        "config": CONFIG,
                    },
                    CONFIG["ckpt_path"],
                )
                print(f"  Checkpoint saved -> {CONFIG['ckpt_path']}")

    elif CONFIG["mode"] == "eval":
        ckpt = torch.load(CONFIG["ckpt_path"], map_location=device)
        model.load_state_dict(ckpt["model"])
        anchors = ckpt["anchors"].to(device)
        val_loader = make_loader("val", shuffle=False, augment=False)
        print("Validation:", eval_one_epoch(model, val_loader, anchors, criterion))

    elif CONFIG["mode"] == "infer":
        ckpt = torch.load(CONFIG["ckpt_path"], map_location=device)
        model.load_state_dict(ckpt["model"])
        anchors = ckpt["anchors"].to(device)
        val_loader = make_loader("val", shuffle=False, augment=False)
        save_predictions_json(
            run_inference(model, val_loader, anchors),
            CONFIG["pred_path"],
        )

    else:
        raise ValueError(f"Unknown mode: {CONFIG['mode']}")


if __name__ == "__main__":
    main()
