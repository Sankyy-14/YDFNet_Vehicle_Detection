"""
Generates a small synthetic "traffic scene" dataset that mimics UA-DETRAC's
label format (car / van / bus / others, normalized cxcywh boxes), so the
rest of the pipeline (dataset loading, training loop, loss, evaluation,
inference/visualization) can be built, run, and demoed *right now* without
waiting on the real UA-DETRAC download (which requires a manual access
request at https://detrac-db.rit.albany.edu/).

This is NOT a substitute for real training data — it's scaffolding /
plumbing verification. Swap `ToyVehicleDataset` for `UADetracDataset`
(data/ua_detrac_dataset.py) once Member 1 has the real files downloaded;
every other file (model, loss, train.py, evaluate.py) is written against the
same (image, target) interface, so nothing else needs to change.

Each synthetic image is a plain background with 0-6 randomly placed,
randomly colored/sized rectangles standing in for vehicles, plus a class
label per rectangle sampled from the same 4 classes UA-DETRAC uses.
"""
from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import Dataset

from data.ua_detrac_dataset import CLASS_NAMES, NUM_CLASSES

# Rough per-class size/color ranges, just so classes are visually distinguishable
# in demo outputs (not meant to be realistic vehicle imagery).
CLASS_STYLE = {
    0: {"size_range": (30, 60), "color": (200, 60, 60)},    # car   - small, red
    1: {"size_range": (40, 70), "color": (60, 140, 200)},   # van   - medium, blue
    2: {"size_range": (70, 110), "color": (60, 180, 90)},   # bus   - large, green
    3: {"size_range": (25, 90), "color": (180, 160, 60)},   # others - variable, yellow
}


class ToyVehicleDataset(Dataset):
    def __init__(self, num_samples: int = 200, img_size: int = 640, seed: int = 0, min_boxes=1, max_boxes=6):
        self.num_samples = num_samples
        self.img_size = img_size
        self.min_boxes = min_boxes
        self.max_boxes = max_boxes
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        rng = random.Random(idx * 9973 + 17)  # deterministic per-index for reproducibility
        s = self.img_size

        # Flat road-colored background with a bit of noise
        img = np.full((s, s, 3), 90, dtype=np.uint8)
        noise = np.random.RandomState(idx).randint(-8, 8, size=(s, s, 1))
        img = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)

        n_boxes = rng.randint(self.min_boxes, self.max_boxes)
        boxes, labels = [], []

        for _ in range(n_boxes):
            cls = rng.randint(0, NUM_CLASSES - 1)
            style = CLASS_STYLE[cls]
            lo, hi = style["size_range"]
            w = rng.randint(lo, hi)
            h = int(w * rng.uniform(0.6, 1.0))  # vehicles are wider than tall, roughly

            x1 = rng.randint(0, max(1, s - w))
            y1 = rng.randint(0, max(1, s - h))
            x2, y2 = x1 + w, y1 + h

            color = style["color"]
            # occasionally jitter color a bit so it's not perfectly uniform per class
            jitter = rng.randint(-15, 15)
            color = tuple(max(0, min(255, c + jitter)) for c in color)
            img[y1:y2, x1:x2] = color

            cx, cy = (x1 + x2) / 2 / s, (y1 + y2) / 2 / s
            nw, nh = w / s, h / s
            boxes.append([cx, cy, nw, nh])
            labels.append(cls)

        image_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            "labels": torch.tensor(labels, dtype=torch.long) if labels else torch.zeros((0,), dtype=torch.long),
            "image_path": f"toy_{idx}",
        }
        return image_t, target


if __name__ == "__main__":
    # Quick smoke test + save a couple of sample images: python -m data.make_toy_dataset
    from PIL import Image, ImageDraw
    from pathlib import Path

    out_dir = Path("samples")
    out_dir.mkdir(exist_ok=True)

    ds = ToyVehicleDataset(num_samples=5, img_size=640)
    for i in range(3):
        img_t, target = ds[i]
        arr = (img_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(arr)
        draw = ImageDraw.Draw(pil_img)
        s = ds.img_size
        for box, label in zip(target["boxes"], target["labels"]):
            cx, cy, w, h = (box * s).tolist()
            x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
            draw.rectangle([x1, y1, x2, y2], outline="white", width=2)
            draw.text((x1, max(0, y1 - 12)), CLASS_NAMES[label], fill="white")
        pil_img.save(out_dir / f"toy_sample_{i}.png")
        print(f"Saved samples/toy_sample_{i}.png with {len(target['labels'])} boxes: "
              f"{[CLASS_NAMES[l] for l in target['labels'].tolist()]}")