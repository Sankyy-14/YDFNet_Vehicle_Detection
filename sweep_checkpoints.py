"""
Runs demo.py's inference logic across ALL saved checkpoints on the SAME image,
saving one output picture per epoch, so you can quickly eyeball which
checkpoint actually detects best (useful when training isn't monotonically
improving, e.g. due to query collapse in early DETR-style training).

Usage:
    python sweep_checkpoints.py --image "C:\\path\\to\\some_real_image.jpg"

Outputs land in outputs/sweep/epoch_N.png -- open the outputs/sweep folder
in VS Code/File Explorer afterward and just look through them.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data.ua_detrac_yolo_dataset import CLASS_NAMES, NUM_CLASSES
from models.ydfnet import YDFNet
from demo import load_input_image, draw_predictions


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image", type=str, required=True)
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--num-queries", type=int, default=100)
    p.add_argument("--score-threshold", type=float, default=0.3)
    p.add_argument("--out-dir", type=str, default="outputs/sweep")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    class DummyArgs:
        pass

    dummy = DummyArgs()
    dummy.image = args.image
    dummy.toy_sample = None
    dummy.img_size = args.img_size
    input_tensor, pil_img = load_input_image(dummy)
    input_tensor = input_tensor.to(device)

    ckpts = sorted(
        Path(args.checkpoint_dir).glob("ydfnet_epoch*.pt"),
        key=lambda p: int(p.stem.split("epoch")[-1]),
    )
    print(f"Found {len(ckpts)} checkpoints: {[c.name for c in ckpts]}")

    for ckpt_path in ckpts:
        epoch_num = int(ckpt_path.stem.split("epoch")[-1])
        model = YDFNet(num_classes=NUM_CLASSES, num_queries=args.num_queries)
        ckpt = torch.load(ckpt_path, map_location=device)
        try:
            model.load_state_dict(ckpt["model_state"])
        except RuntimeError as e:
            print(f"  epoch {epoch_num}: SKIPPED (architecture mismatch — likely a stale checkpoint from an earlier run): {e}")
            continue
        model.to(device).eval()

        preds = model.predict(input_tensor, score_threshold=args.score_threshold)[0]
        n_det = len(preds["boxes"])

        result_img = draw_predictions(pil_img, preds["boxes"], preds["scores"], preds["labels"], args.img_size)
        out_path = out_dir / f"epoch_{epoch_num:02d}.png"
        result_img.save(out_path)
        print(f"  epoch {epoch_num:2d}: {n_det:3d} detections -> saved {out_path}")

    print(f"\nDone. Open the '{out_dir}' folder and look through epoch_XX.png files to pick the best one.")


if __name__ == "__main__":
    main()