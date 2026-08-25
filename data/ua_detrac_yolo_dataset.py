"""
UA-DETRAC dataset loader for the pre-converted YOLO-format export
(Kaggle: dtrnngc/ua-detrac-dataset), matching this exact folder layout:

    UA-DETRAC/DETRAC_Upload/
        images/
            train/
                MVI_20011_img00004.jpg
                ...
            val/
                ...
        labels/
            train/
                MVI_20011_img00004.txt   (one line per box: class cx cy w h, normalized)
                ...
            val/
                ...

Verified class mapping (confirmed visually against real boxes on real images):
    0 = bus
    1 = car
    2 = others
    3 = van

This is a *different* loader from data/ua_detrac_dataset.py (which parses the
official XML-annotation release). Use THIS one for the Kaggle YOLO-format
export. Both produce the exact same (image_tensor, target_dict) interface
expected by train.py, so nothing else in the pipeline needs to change.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

CLASS_NAMES = ["bus", "car", "others", "van"]
NUM_CLASSES = len(CLASS_NAMES)


class UADetracYoloDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        img_size: int = 640,
        max_samples: int | None = None,
    ):
        """Args:
            root: path to the DETRAC_Upload folder (the one containing images/ and labels/).
                  e.g. r"C:\\Users\\A8IN\\Downloads\\archive\\content\\UA-DETRAC\\DETRAC_Upload"
            split: 'train' or 'val'
            img_size: images resized to img_size x img_size
            max_samples: if set, caps the dataset to this many samples, evenly
                spaced across the full sorted file list (not just the first N)
                so a capped run still spans multiple video sequences rather
                than just one. Use this to keep training time-boxed on a
                laptop GPU instead of training on the full ~82k image set.
        """
        self.root = Path(root)
        self.img_dir = self.root / "images" / split
        self.label_dir = self.root / "labels" / split
        self.img_size = img_size

        if not self.img_dir.exists():
            raise FileNotFoundError(f"Image dir not found: {self.img_dir}")
        if not self.label_dir.exists():
            raise FileNotFoundError(f"Label dir not found: {self.label_dir}")

        # Only keep images that actually have a matching label file.
        self.samples: list[tuple[Path, Path]] = []
        for img_path in sorted(self.img_dir.glob("*.jpg")):
            label_path = self.label_dir / (img_path.stem + ".txt")
            if label_path.exists():
                self.samples.append((img_path, label_path))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No matching image/label pairs found in {self.img_dir} / {self.label_dir}. "
                "Check that filenames correspond (e.g. img00004.jpg <-> img00004.txt)."
            )

        # Optionally cap dataset size (useful for fast iteration / time-limited runs).
        # Sampled evenly across the sorted list rather than just the first N, so we
        # don't accidentally train only on one video sequence.
        if max_samples is not None and len(self.samples) > max_samples:
            step = len(self.samples) / max_samples
            self.samples = [self.samples[int(i * step)] for i in range(max_samples)]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        img_path, label_path = self.samples[idx]

        image = Image.open(img_path).convert("RGB")
        image = image.resize((self.img_size, self.img_size))
        image_t = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

        boxes, labels = [], []
        with open(label_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                cls, cx, cy, w, h = line.split()
                # Already normalized (0-1) in this export, and already (cx,cy,w,h) —
                # exactly the format our loss/model expect, no conversion needed.
                boxes.append([float(cx), float(cy), float(w), float(h)])
                labels.append(int(cls))

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4)),
            "labels": torch.tensor(labels, dtype=torch.long) if labels else torch.zeros((0,), dtype=torch.long),
            "image_path": str(img_path),
        }
        return image_t, target


def collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    targets = [b[1] for b in batch]
    return images, targets


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print(r'Usage: python -m data.ua_detrac_yolo_dataset "C:\path\to\DETRAC_Upload"')
        sys.exit(0)

    ds = UADetracYoloDataset(sys.argv[1], split="train", max_samples=20)
    print(f"Indexed {len(ds)} image/label pairs (capped at 20 for this test)")
    img, target = ds[0]
    print("image shape:", tuple(img.shape))
    print("boxes:", target["boxes"].shape, "labels:", target["labels"].tolist())
    print("class names:", [CLASS_NAMES[l] for l in target["labels"].tolist()])