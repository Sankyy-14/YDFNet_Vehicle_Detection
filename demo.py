"""
Runs inference with a trained (or freshly initialized) YDFNet checkpoint on
a sample image and saves a visualization with predicted boxes drawn.

Usage:
    python demo.py --checkpoint checkpoints/ydfnet_epoch15.pt --toy-sample 3
    python demo.py --checkpoint checkpoints/ydfnet_epoch15.pt --image path/to/real.jpg
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from data.make_toy_dataset import ToyVehicleDataset
from data.ua_detrac_yolo_dataset import CLASS_NAMES, NUM_CLASSES
from models.ydfnet import YDFNet

BOX_COLORS = ["#e74c3c", "#3498db", "#2ecc71", "#f1c40f"]  # matches CLASS_STYLE hues in make_toy_dataset.py


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None, help="Path to a .pt checkpoint from train.py. If omitted, uses a freshly initialized (untrained) model.")
    p.add_argument("--image", type=str, default=None, help="Path to a real image to run inference on")
    p.add_argument("--toy-sample", type=int, default=None, help="Index into ToyVehicleDataset to use as input instead of a real image")
    p.add_argument("--img-size", type=int, default=320)
    p.add_argument("--num-queries", type=int, default=30)
    p.add_argument("--score-threshold", type=float, default=0.3)
    p.add_argument("--out", type=str, default="outputs/demo_result.png")
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def load_model(args, device):
    model = YDFNet(num_classes=NUM_CLASSES, num_queries=args.num_queries)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"[model] Loaded weights from {args.checkpoint} (epoch {ckpt.get('epoch', '?')})")
    else:
        print("[model] WARNING: no --checkpoint given, using randomly-initialized weights (predictions will be meaningless, this only verifies the pipeline runs)")
    model.to(device).eval()
    return model


def load_input_image(args) -> tuple[torch.Tensor, Image.Image]:
    """Returns (model_input_tensor[1,3,H,W], PIL image for drawing on)."""
    if args.image:
        pil_img = Image.open(args.image).convert("RGB").resize((args.img_size, args.img_size))
    elif args.toy_sample is not None:
        ds = ToyVehicleDataset(num_samples=args.toy_sample + 1, img_size=args.img_size)
        img_t, _ = ds[args.toy_sample]
        arr = (img_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        pil_img = Image.fromarray(arr)
    else:
        raise ValueError("Provide either --image or --toy-sample")

    arr = np.array(pil_img).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor, pil_img


def draw_predictions(pil_img: Image.Image, boxes, scores, labels, img_size: int) -> Image.Image:
    img = pil_img.copy()
    draw = ImageDraw.Draw(img)
    for box, score, label in zip(boxes, scores, labels):
        cx, cy, w, h = (box * img_size).tolist()
        x1, y1, x2, y2 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        color = BOX_COLORS[int(label) % len(BOX_COLORS)]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label_text = f"{CLASS_NAMES[int(label)]} {score:.2f}"
        draw.text((x1 + 2, max(0, y1 - 14)), label_text, fill=color)
    return img


def main():
    args = parse_args()
    device = torch.device(args.device)

    model = load_model(args, device)
    input_tensor, pil_img = load_input_image(args)
    input_tensor = input_tensor.to(device)

    preds = model.predict(input_tensor, score_threshold=args.score_threshold)[0]
    n_det = len(preds["boxes"])
    print(f"[inference] {n_det} detections above score threshold {args.score_threshold}")
    for box, score, label in zip(preds["boxes"], preds["scores"], preds["labels"]):
        print(f"  {CLASS_NAMES[int(label)]:8s} score={score:.3f} box(cx,cy,w,h)={[round(x,3) for x in box.tolist()]}")

    result_img = draw_predictions(pil_img, preds["boxes"], preds["scores"], preds["labels"], args.img_size)

    from pathlib import Path
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    result_img.save(args.out)
    print(f"[output] Saved visualization to {args.out}")


if __name__ == "__main__":
    main()