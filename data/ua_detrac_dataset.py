"""
PyTorch Dataset for UA-DETRAC.

Expected directory layout (this is UA-DETRAC's official released structure —
point `root` at the folder containing these):

    UA-DETRAC/
      Insight-MVT_Annotation_Train/
        MVI_20011/
          img00001.jpg
          img00002.jpg
          ...
        MVI_20012/
          ...
      DETRAC-Train-Annotations-XML/
        MVI_20011.xml
        MVI_20012.xml
        ...
      Insight-MVT_Annotation_Test/          (optional, for the test split)
      DETRAC-Test-Annotations-XML/          (optional)

Each XML file (one per video sequence) contains <frame> elements, each with
a <target_list> of <target> boxes with a <box> (left, top, width, height)
and an <attribute> tag giving the vehicle_type string (e.g. "car", "bus",
"van", "others").

NOTE ON DATASET AVAILABILITY: this loader expects the real UA-DETRAC files
on disk — it does not download them (the dataset requires a manual request
form at https://detrac-db.rit.albany.edu/, standard for this benchmark).
Point ROOT at wherever your team downloads it. Data Engineering (Member 1)
owns getting the raw files into this layout; everyone else can develop
against `data/make_toy_dataset.py`'s synthetic stand-in in the meantime.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

# UA-DETRAC's raw vehicle_type strings, collapsed to our 4 training classes.
# "others" is a real UA-DETRAC label for ambiguous/unclassified vehicles.
CLASS_MAP = {"car": 0, "van": 1, "bus": 2, "others": 3}
CLASS_NAMES = ["car", "van", "bus", "others"]
NUM_CLASSES = len(CLASS_NAMES)


class UADetracDataset(Dataset):
    def __init__(self, root: str | Path, split: str = "train", img_size: int = 640, transforms=None):
        """Args:
            root: path to the UA-DETRAC root folder (see layout above).
            split: 'train' or 'test'.
            img_size: images are resized (with letterboxing) to img_size x img_size.
            transforms: optional albumentations-style transform applied to
                (image, boxes, labels) — see data/transforms.py.
        """
        self.root = Path(root)
        self.img_size = img_size
        self.transforms = transforms

        img_dir_name = "Insight-MVT_Annotation_Train" if split == "train" else "Insight-MVT_Annotation_Test"
        xml_dir_name = "DETRAC-Train-Annotations-XML" if split == "train" else "DETRAC-Test-Annotations-XML"
        self.img_root = self.root / img_dir_name
        self.xml_root = self.root / xml_dir_name

        self.samples: list[tuple[Path, list[dict]]] = []
        self._index()

    def _index(self):
        """Parses every sequence's XML file once and builds a flat list of
        (image_path, [box_dicts]) samples across all sequences/frames."""
        if not self.xml_root.exists():
            raise FileNotFoundError(
                f"UA-DETRAC annotation dir not found at {self.xml_root}. "
                "Download the dataset from https://detrac-db.rit.albany.edu/ "
                "and point `root` at the extracted folder. For development "
                "without the real dataset, use data/make_toy_dataset.py instead."
            )

        for xml_path in sorted(self.xml_root.glob("*.xml")):
            seq_name = xml_path.stem  # e.g. "MVI_20011"
            seq_img_dir = self.img_root / seq_name
            if not seq_img_dir.exists():
                continue

            tree = ET.parse(xml_path)
            root_el = tree.getroot()

            for frame_el in root_el.findall("frame"):
                frame_num = int(frame_el.get("num"))
                img_path = seq_img_dir / f"img{frame_num:05d}.jpg"
                if not img_path.exists():
                    continue

                boxes = []
                target_list = frame_el.find("target_list")
                if target_list is None:
                    continue
                for target_el in target_list.findall("target"):
                    box_el = target_el.find("box")
                    attr_el = target_el.find("attribute")
                    if box_el is None:
                        continue
                    left = float(box_el.get("left"))
                    top = float(box_el.get("top"))
                    w = float(box_el.get("width"))
                    h = float(box_el.get("height"))

                    vtype = attr_el.get("vehicle_type", "others") if attr_el is not None else "others"
                    cls_id = CLASS_MAP.get(vtype, CLASS_MAP["others"])

                    boxes.append({"xywh": (left, top, w, h), "label": cls_id})

                if boxes:  # skip frames with zero annotated vehicles
                    self.samples.append((img_path, boxes))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict]:
        img_path, boxes = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        orig_w, orig_h = image.size

        image = image.resize((self.img_size, self.img_size))
        image_t = torch.from_numpy(
            __import__("numpy").array(image)
        ).permute(2, 0, 1).float() / 255.0

        # Convert (left, top, w, h) in pixel coords -> normalized (cx, cy, w, h)
        norm_boxes, labels = [], []
        for b in boxes:
            left, top, w, h = b["xywh"]
            cx = (left + w / 2) / orig_w
            cy = (top + h / 2) / orig_h
            nw = w / orig_w
            nh = h / orig_h
            norm_boxes.append([cx, cy, nw, nh])
            labels.append(b["label"])

        target = {
            "boxes": torch.tensor(norm_boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
            "image_path": str(img_path),
        }

        if self.transforms is not None:
            image_t, target = self.transforms(image_t, target)

        return image_t, target


def collate_fn(batch: list[tuple[torch.Tensor, dict]]):
    """Images stack cleanly (all resized to the same img_size), but targets
    have a variable number of boxes per image, so they stay a list of dicts
    rather than being stacked into a single tensor."""
    images = torch.stack([b[0] for b in batch])
    targets = [b[1] for b in batch]
    return images, targets


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m data.ua_detrac_dataset /path/to/UA-DETRAC")
        sys.exit(0)

    ds = UADetracDataset(sys.argv[1], split="train")
    print(f"Indexed {len(ds)} annotated frames")
    img, target = ds[0]
    print("image shape:", tuple(img.shape))
    print("boxes:", target["boxes"].shape, "labels:", target["labels"])