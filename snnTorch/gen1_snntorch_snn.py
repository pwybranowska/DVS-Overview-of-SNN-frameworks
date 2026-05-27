"""
------------------------------------------------------------------------
SnnTorch convolutional SNN object detector for the GEN1 EventCube dataset.

Dataset:
    download/Gen1EventCubeDataset/
    train/   <name>.h5  +  <name>_bbox.npy
    val/     ...
    test/    ...
    
"""

import math
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import snntorch as snn          
from snntorch import surrogate
from torchvision.ops import batched_nms

# -------------------------------------------------------------------------------------------------------------------------------------
CONFIG = {
    # ---- paths ----
    "data_root":  "download/Gen1EventCubeDataset",
    # "ckpt_path":  "snntorch_gen1_best.pt",
    "ckpt_path":  "snntorch_gen1_best_v2.pt",
    "pred_path":  "snntorch_gen1_predictions.json",

    # ---- mode ----
    "mode":       "train",   # "train" | "eval" | "infer"

    # ---- data ----
    "image_h":    240,
    "image_w":    304,
    "in_channels": 10,
    "num_bins":   5,
    "num_workers": 2,   

    # ---- model ----
    "base_channels": 32,
    "num_classes":   3,    # 0=background  1=car  2=pedestrian
    "num_anchors":   6,
    "lif_beta":      0.75,   # membrane decay init value

    # ---- training ----
    "epochs":        2,
    "batch_size":    4,    # each sample is ~9 MB after subsampling -> fine
    "lr":            1e-3,
    "weight_decay":  1e-4,
    "grad_clip":     1.0,
    "seed":          42,

    # ---- anchor matching ----
    "iou_pos":       0.5,
    "iou_neg":       0.3,
    "anchor_scales": (1.0, 1.5),
    "anchor_ratios": (0.5, 1.0, 2.0),

    # ---- post-processing ----
    "score_thresh":  0.3,
    "nms_thresh":    0.5,
    "top_k":         100,
}

# Prophesee GEN1 raw class ids -> training ids  (background = 0)
# 0 = car,  1 = pedestrian
CLASS_MAP = {0: 1, 1: 2}


# -------------------------------------------------------------------------------------------------------------------------
#  Dataset loading

class GEN1Dataset(Dataset):
    """
    Loads [1200, 6, 240, 304] uint8 cubes from .h5 files.

    Memory-safe loading:
        1. Read cube as uint8 (keeps it at 525 MB, never converted fully)
        2. Subsample 1200 -> num_bins by reshape + mean over axis 0 groups
        3. Cast to float32 and normalise only the small [T, 6, H, W] result
    """

    def __init__(self, split: str):
        self.dir      = Path(CONFIG["data_root"]) / split
        self.num_bins = CONFIG["num_bins"]

        if not self.dir.exists():
            raise FileNotFoundError(f"Split directory not found: {self.dir}")

        self.samples = sorted([
            p.stem for p in self.dir.glob("*.h5")
            if not p.stem.endswith("_bbox")
        ])
        if not self.samples:
            raise FileNotFoundError(f"No .h5 files in {self.dir}")

        print(f"[Dataset] split={split:5s}  samples={len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def _load_cube(self, stem: str) -> np.ndarray:
        """
        Load .h5, subsample T -> num_bins, pad C -> 10, return float32 [T, 10, H, W].

        Dataset has mixed shapes:
            [~1200, 10, 240, 304]  -- majority (87% of files)
            [~1200,  6, 240, 304]  -- minority (padded to 10 with zeros)
            T also varies (593-1200), so we use evenly-spaced index sampling
            instead of reshape (which requires exact divisibility).
        """
        with h5py.File(self.dir / f"{stem}.h5", "r") as f:
            key  = list(f.keys())[0]
            cube = f[key][:]          # uint8 [T_full, C, H, W]

        T_full, C, H, W = cube.shape
        T = self.num_bins

        # Evenly-spaced temporal subsampling (works for any T_full)
        indices = np.linspace(0, T_full - 1, T, dtype=int)
        out = cube[indices].astype(np.float32)   # [T, C, H, W]

        # Pad channel dim to 10 so all samples have the same shape
        if C < 10:
            pad = np.zeros((T, 10 - C, H, W), dtype=np.float32)
            out = np.concatenate([out, pad], axis=1)   # [T, 10, H, W]

        # Normalise to [0, 1]
        mx = out.max()
        if mx > 0:
            out /= mx
        return out  # [T, 10, H, W]  float32

    def _load_boxes(self, stem: str) -> torch.Tensor:
        """
        Load _bbox.npy (Prophesee structured array).
        Returns [G, 5] tensor: x1 y1 x2 y2 class_training_id
        """
        raw = np.load(self.dir / f"{stem}_bbox.npy", allow_pickle=True)
        boxes = []

        if raw.dtype.names is not None:
            # Structured array: fields ts, x, y, w, h, class_id, confidence, track_id
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
            # Plain array [M, 5] or [M, 6]  (t,) x1, y1, x2, y2, cls
            raw = raw.reshape(-1, raw.shape[-1]) if raw.ndim == 1 else raw
            offset = 1 if raw.shape[1] == 6 else 0
            for row in raw:
                x1, y1, x2, y2 = row[offset:offset + 4]
                cls = int(row[offset + 4])
                if cls not in CLASS_MAP:
                    continue
                boxes.append([float(x1), float(y1), float(x2), float(y2), float(CLASS_MAP[cls])])

        if boxes:
            return torch.tensor(boxes, dtype=torch.float32)
        return torch.zeros((0, 5), dtype=torch.float32)

    def __getitem__(self, idx):
        stem   = self.samples[idx]
        cube   = self._load_cube(stem)          # [T, 6, H, W] float32
        target = self._load_boxes(stem)         # [G, 5] float32
        return torch.from_numpy(cube), target, stem


def collate_fn(batch):
    cubes, targets, stems = zip(*batch)
    return torch.stack(cubes, 0), list(targets), list(stems)


# -------------------------------------------------------------------------------------------------------------------------
#  MODEL


class ConvSpikeBlock(nn.Module):
    """
    Conv2d (applied per-timestep) -> BatchNorm2d -> LIF
    Input / output shape: [T, B, C, H, W]

    Membrane potential (mem) initialised with lif.init_leaky()
    at the start of each sequence and manually stepped for each t.
    Loop over T inside forward() instead of passing the whole [T, B, C, H, W] tensor to the neuron at once.
    
    """
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid(slope=25)

        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)

        # snn.Leaky wraps the LIF dynamics
        #  beta = membrane decay constant
        self.lif  = snn.Leaky(
            beta=CONFIG["lif_beta"],
            spike_grad=spike_grad,
            init_hidden=False, 
            # init_hidden=True,
            learn_beta=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [T, B, C, H, W]
        T, B, C, H, W = x.shape

        # Initialise membrane potential to zero for this batch
        # init_leaky() returns a zero tensor shaped to match the first input
        mem = self.lif.init_leaky()     # scalar 0 – will broadcast on first step
        # self.lif.init_hidden = True

        spike_seq = []
        for t in range(T):
            # --- spatial convolution for one timestep ---
            frame = x[t]                        # [B, C, H, W]
            cur   = self.bn(self.conv(frame))   # [B, C_out, H_out, W_out]

            # --- LIF dynamics: returns (spike, updated_mem) ---
            spk, mem = self.lif(cur, mem)       # spk: [B, C_out, H_out, W_out]
            spike_seq.append(spk)

        return torch.stack(spike_seq, dim=0)    # [T, B, C_out, H_out, W_out]


class SNNTorchSpikingSSD(nn.Module):
    """
    4-block SNN backbone + SSD detection head.

    Input:  [B, T, C, H, W]    (C = in_channels from CONFIG)
    Output: {"loc": [B,A,4,Hf,Wf],  "cls": [B,A,C,Hf,Wf]}
    """

    def __init__(self):
        super().__init__()
        B_ch = CONFIG["base_channels"]
        A    = CONFIG["num_anchors"]
        C    = CONFIG["num_classes"]
        in_c = CONFIG["in_channels"]

        # Backbone: strides 2-2-2-1  ->  1/8 resolution feature map
        self.block1 = ConvSpikeBlock(in_c,   B_ch,     stride=2)
        self.block2 = ConvSpikeBlock(B_ch,   B_ch * 2, stride=2)
        self.block3 = ConvSpikeBlock(B_ch*2, B_ch * 4, stride=2)
        self.block4 = ConvSpikeBlock(B_ch*4, B_ch * 4, stride=1)

        # Detection head (standard Conv2d on temporally-pooled 2-D features)
        self.loc_head = nn.Conv2d(B_ch * 4, A * 4, kernel_size=3, padding=1)
        self.cls_head = nn.Conv2d(B_ch * 4, A * C, kernel_size=3, padding=1)

        self._A = A
        self._C = C

    def forward(self, x: torch.Tensor) -> dict:
        # [B, T, C, H, W] -> [T, B, C, H, W]  (time-first for the SNN blocks)
        x = x.permute(1, 0, 2, 3, 4).contiguous()

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        # Temporal mean-pool spike trains -> rate-coded feature map [B, C, Hf, Wf]
        feats = x.mean(dim=0)

        B2, _, Hf, Wf = feats.shape
        loc = self.loc_head(feats).reshape(B2, self._A, 4,       Hf, Wf)
        cls = self.cls_head(feats).reshape(B2, self._A, self._C, Hf, Wf)
        return {"loc": loc, "cls": cls}

# -------------------------------------------------------------------------------------------------------------------------
#  ANCHORS


def generate_anchors(feat_h, feat_w):
    img_h  = CONFIG["image_h"]
    img_w  = CONFIG["image_w"]
    scales = CONFIG["anchor_scales"]
    ratios = CONFIG["anchor_ratios"]

    stride_y = img_h / feat_h
    stride_x = img_w / feat_w
    anchors  = []

    for iy in range(feat_h):
        for ix in range(feat_w):
            cx = (ix + 0.5) * stride_x
            cy = (iy + 0.5) * stride_y
            for s in scales:
                for r in ratios:
                    w = s * math.sqrt(r) * stride_x
                    h = s / math.sqrt(r) * stride_y
                    anchors.append([cx - w/2, cy - h/2,
                                    cx + w/2, cy + h/2])

    return torch.tensor(anchors, dtype=torch.float32)


def build_anchors(model, device):
    model.eval()
    T = CONFIG["num_bins"]
    C = CONFIG["in_channels"]
    H = CONFIG["image_h"]
    W = CONFIG["image_w"]

    with torch.no_grad():
        dummy = torch.zeros(1, T, C, H, W, device=device)
        out   = model(dummy)

    _, A, _, Hf, Wf = out["loc"].shape
    anchors  = generate_anchors(Hf, Wf).to(device)

    expected = A * Hf * Wf
    if anchors.shape[0] < expected:
        reps    = math.ceil(expected / anchors.shape[0])
        anchors = anchors.repeat(reps, 1)
    return anchors[:expected]


# -------------------------------------------------------------------------------------------------------------------------
#  BOX UTILITIES


def xyxy_to_cxcywh(b):
    x1, y1, x2, y2 = b.unbind(-1)
    return torch.stack([(x1+x2)/2, (y1+y2)/2,
                        (x2-x1).clamp(1e-6), (y2-y1).clamp(1e-6)], -1)


def cxcywh_to_xyxy(b):
    cx, cy, w, h = b.unbind(-1)
    return torch.stack([cx-w/2, cy-h/2, cx+w/2, cy+h/2], -1)


def box_iou(a, b):
    if a.numel() == 0 or b.numel() == 0:
        return torch.zeros(a.shape[0], b.shape[0], device=a.device)
    area_a = (a[:,2]-a[:,0]).clamp(0) * (a[:,3]-a[:,1]).clamp(0)
    area_b = (b[:,2]-b[:,0]).clamp(0) * (b[:,3]-b[:,1]).clamp(0)
    lt     = torch.max(a[:,None,:2], b[:,:2])
    rb     = torch.min(a[:,None,2:], b[:,2:])
    inter  = (rb - lt).clamp(0).prod(-1)
    return inter / (area_a[:,None] + area_b - inter).clamp(1e-6)



# -------------------------------------------------------------------------------------------------------------------------
#  ANCHOR MATCHING & BOX ENCODING / DECODING


def match_and_encode(anchors, gt):
    A       = anchors.shape[0]
    device  = anchors.device
    iou_pos = CONFIG["iou_pos"]
    iou_neg = CONFIG["iou_neg"]

    cls_t = torch.full((A,), -1, dtype=torch.long,   device=device)
    box_t = torch.zeros((A, 4),  dtype=torch.float32, device=device)

    if gt.shape[0] == 0:
        cls_t[:] = 0
        return cls_t, box_t

    gt_boxes  = gt[:, :4]
    gt_labels = gt[:, 4].long()

    ious             = box_iou(anchors, gt_boxes)
    best_iou, best_g = ious.max(dim=1)

    cls_t[best_iou < iou_neg]  = 0
    pos_mask = best_iou >= iou_pos
    cls_t[pos_mask] = gt_labels[best_g[pos_mask]]

    for g in range(gt_boxes.shape[0]):
        cls_t[ious[:, g].argmax()] = gt_labels[g]

    anc_c = xyxy_to_cxcywh(anchors)
    gt_c  = xyxy_to_cxcywh(gt_boxes[best_g])

    box_t[:, 0] = (gt_c[:, 0] - anc_c[:, 0]) / anc_c[:, 2].clamp(1e-6)
    box_t[:, 1] = (gt_c[:, 1] - anc_c[:, 1]) / anc_c[:, 3].clamp(1e-6)
    box_t[:, 2] = torch.log((gt_c[:, 2] / anc_c[:, 2].clamp(1e-6)).clamp(1e-6)).clamp(-10, 10)
    box_t[:, 3] = torch.log((gt_c[:, 3] / anc_c[:, 3].clamp(1e-6)).clamp(1e-6)).clamp(-10, 10)

    return cls_t, box_t


def decode_boxes(anchors, deltas):
    anc = xyxy_to_cxcywh(anchors)
    cx  = deltas[:, 0] * anc[:, 2] + anc[:, 0]
    cy  = deltas[:, 1] * anc[:, 3] + anc[:, 1]
    w   = torch.exp(deltas[:, 2]).clamp(max=1e3) * anc[:, 2]
    h   = torch.exp(deltas[:, 3]).clamp(max=1e3) * anc[:, 3]
    return cxcywh_to_xyxy(torch.stack([cx, cy, w, h], -1))


# -------------------------------------------------------------------------------------------------------------------------
#  LOSS


class DetectionLoss(nn.Module):

    def forward(self, pred_loc, pred_cls, anchors, targets):
        B, A, _, H, W = pred_loc.shape
        N      = A * H * W
        device = pred_loc.device
        C      = CONFIG["num_classes"]

        loc_flat = pred_loc.permute(0,3,4,1,2).contiguous().reshape(B, N, 4)
        cls_flat = pred_cls.permute(0,3,4,1,2).contiguous().reshape(B, N, C)

        tot_cls = torch.tensor(0.0, device=device)
        tot_box = torch.tensor(0.0, device=device)
        num_pos = 0

        for b in range(B):
            cls_t, box_t = match_and_encode(anchors, targets[b].to(device))
            valid = cls_t >= 0
            pos   = cls_t > 0

            if valid.any():
                tot_cls = tot_cls + F.cross_entropy(
                    cls_flat[b][valid], cls_t[valid], reduction="mean")
            if pos.any():
                tot_box = tot_box + F.smooth_l1_loss(
                    loc_flat[b][pos], box_t[pos], reduction="mean")
                num_pos += int(pos.sum())

        loss  = (tot_cls + tot_box) / max(B, 1)
        stats = {
            "loss":     float(loss.detach()),
            "cls_loss": float(tot_cls.detach() / max(B, 1)),
            "box_loss": float(tot_box.detach() / max(B, 1)),
            "num_pos":  num_pos,
        }
        return loss, stats


# -------------------------------------------------------------------------------------------------------------------------
#  POST-PROCESSING


@torch.no_grad()
def post_process(pred_loc, pred_cls, anchors):
    B, A, _, H, W = pred_loc.shape
    N   = A * H * W
    C   = CONFIG["num_classes"]
    loc = pred_loc.permute(0,3,4,1,2).contiguous().reshape(B, N, 4)
    cls = pred_cls.permute(0,3,4,1,2).contiguous().reshape(B, N, C)

    results = []
    for b in range(B):
        probs          = F.softmax(cls[b], dim=-1)
        scores, labels = probs[:, 1:].max(dim=-1)
        labels         = labels + 1

        keep = scores >= CONFIG["score_thresh"]
        if not keep.any():
            results.append({"boxes": [], "scores": [], "labels": []})
            continue

        scores_ = scores[keep]
        labels_ = labels[keep]
        boxes_  = decode_boxes(anchors[keep], loc[b][keep])
        idx     = batched_nms(boxes_, scores_, labels_, CONFIG["nms_thresh"])[:CONFIG["top_k"]]
        results.append({
            "boxes":  [boxes_[i].tolist() for i in idx],
            "scores": [float(scores_[i])  for i in idx],
            "labels": [int(labels_[i])    for i in idx],
        })
    return results


# -------------------------------------------------------------------------------------------------------------------------
#  TRAINING / EVAL LOOPS


def train_one_epoch(model, loader, anchors, criterion, optimizer, epoch):
    model.train()
    device = next(model.parameters()).device
    total  = {"loss": 0., "cls_loss": 0., "box_loss": 0., "num_pos": 0}
    n      = 0

    for i, (x, targets, _stems) in enumerate(loader):
        x = x.to(device)
        optimizer.zero_grad(set_to_none=True)

        out         = model(x)
        loss, stats = criterion(out["loc"], out["cls"], anchors, targets)
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
        optimizer.step()

        for k in total:
            total[k] += stats[k]
        n += 1

        if i % 20 == 0:
            print(f"  epoch {epoch:03d}  step {i:04d}/{len(loader)}  "
                f"loss={stats['loss']:.4f}  "
                f"cls={stats['cls_loss']:.4f}  "
                f"box={stats['box_loss']:.4f}  "
                f"pos={stats['num_pos']}")

    return {k: v / max(n, 1) if k != "num_pos" else v
            for k, v in total.items()}


@torch.no_grad()
def eval_one_epoch(model, loader, anchors, criterion):
    model.eval()
    device = next(model.parameters()).device
    total  = {"loss": 0., "cls_loss": 0., "box_loss": 0.}
    n      = 0

    for x, targets, _stems in loader:
        x        = x.to(device)
        out      = model(x)
        _, stats = criterion(out["loc"], out["cls"], anchors, targets)
        for k in total:
            total[k] += stats[k]
        n += 1

    return {k: v / max(n, 1) for k, v in total.items()}


@torch.no_grad()
def run_inference(model, loader, anchors):
    model.eval()
    device    = next(model.parameters()).device
    all_preds = {}

    for x, _targets, stems in loader:
        x    = x.to(device)
        out  = model(x)
        dets = post_process(out["loc"], out["cls"], anchors)
        for stem, det in zip(stems, dets):
            all_preds[stem] = det

    return all_preds


def save_predictions_json(preds, path):
    import json
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(preds, indent=2))
    print(f"Predictions saved -> {path}")


# -------------------------------------------------------------------------------------------------------------------------


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    set_seed(CONFIG["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Mode   : {CONFIG['mode']}")

    model   = SNNTorchSpikingSSD().to(device)
    anchors = build_anchors(model, device)
    print(f"Anchors: {anchors.shape[0]}")

    criterion = DetectionLoss()

    def make_loader(split, shuffle):
        ds = GEN1Dataset(split)
        return DataLoader(
            ds,
            batch_size  = CONFIG["batch_size"],
            shuffle     = shuffle,
            num_workers = CONFIG["num_workers"],
            collate_fn  = collate_fn,
            pin_memory  = (device.type == "cuda"),
        )

    # ---- TRAIN ----
    if CONFIG["mode"] == "train":
        train_loader = make_loader("train", shuffle=True)
        val_loader   = make_loader("val",   shuffle=False)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=CONFIG["lr"], weight_decay=CONFIG["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=CONFIG["epochs"])

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
                torch.save({"epoch": epoch, "model": model.state_dict(),
                            "anchors": anchors.cpu(), "config": CONFIG}, CONFIG["ckpt_path"])
                print(f"  Checkpoint saved -> {CONFIG['ckpt_path']}")

    # ---- EVAL ----
    elif CONFIG["mode"] == "eval":
        ckpt = torch.load(CONFIG["ckpt_path"], map_location=device)
        model.load_state_dict(ckpt["model"])
        val_loader = make_loader("val", shuffle=False)
        print("Validation:", eval_one_epoch(model, val_loader, anchors, criterion))

    # ---- INFER ----
    elif CONFIG["mode"] == "infer":
        ckpt = torch.load(CONFIG["ckpt_path"], map_location=device)
        model.load_state_dict(ckpt["model"])
        val_loader = make_loader("val", shuffle=False)
        save_predictions_json(
            run_inference(model, val_loader, anchors),
            CONFIG["pred_path"])