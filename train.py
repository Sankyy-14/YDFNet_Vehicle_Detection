"""
Training entry point for YDFNet.

Usage (toy data, for pipeline verification — runs on CPU in a couple minutes):
    python train.py --toy --epochs 5 --batch-size 4

Usage (real UA-DETRAC data, once downloaded):
    python train.py --data-root /path/to/UA-DETRAC --epochs 50 --batch-size 8 --device cuda

This is intentionally a plain, readable training loop (no Trainer abstraction)
so every team member can see exactly what's happening at each step —
useful when you're explaining the training process to your supervisor.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data.ua_detrac_yolo_dataset import UADetracYoloDataset, collate_fn, NUM_CLASSES
from data.make_toy_dataset import ToyVehicleDataset
from models.ydfnet import YDFNet
from models.loss import HungarianMatcher, SetCriterion


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default=None, help="Path to UA-DETRAC root (DETRAC_Upload folder for the YOLO-format export)")
    p.add_argument("--max-samples", type=int, default=None, help="Cap dataset size for faster/time-boxed runs (e.g. 6000)")
    p.add_argument("--toy", action="store_true", help="Train on synthetic toy data instead of real UA-DETRAC (for pipeline testing)")
    p.add_argument("--toy-samples", type=int, default=200)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-backbone", type=float, default=1e-5, help="Lower LR for pretrained backbone weights")
    p.add_argument("--img-size", type=int, default=640)
    p.add_argument("--num-queries", type=int, default=150)
    p.add_argument("--yolo-variant", type=str, default="yolov10n.pt")
    p.add_argument("--freeze-backbone", action="store_true")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    p.add_argument("--log-every", type=int, default=5)
    return p.parse_args()


def build_dataset(args):
    if args.toy or args.data_root is None:
        print("[data] Using synthetic ToyVehicleDataset (pass --data-root for real UA-DETRAC)")
        return ToyVehicleDataset(num_samples=args.toy_samples, img_size=args.img_size)
    print(f"[data] Loading UA-DETRAC from {args.data_root}")
    return UADetracYoloDataset(args.data_root, split="train", img_size=args.img_size, max_samples=args.max_samples)


def main():
    args = parse_args()
    device = torch.device(args.device)
    Path(args.checkpoint_dir).mkdir(exist_ok=True)

    dataset = build_dataset(args)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0
    )
    print(f"[data] {len(dataset)} samples, {len(loader)} batches/epoch")

    model = YDFNet(
        num_classes=NUM_CLASSES,
        yolo_variant=args.yolo_variant,
        freeze_backbone=args.freeze_backbone,
        num_queries=args.num_queries,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] YDFNet ready — {n_params:,} trainable parameters")

    matcher = HungarianMatcher()
    criterion = SetCriterion(num_classes=NUM_CLASSES, matcher=matcher).to(device)

    # Discriminative learning rates: pretrained backbone gets a much smaller
    # LR than the randomly-initialized BiFPN + DETR head, standard practice
    # when fine-tuning on top of pretrained weights.
    backbone_params = list(model.backbone.parameters())
    other_params = list(model.bifpn.parameters()) + list(model.head.parameters())
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in backbone_params if p.requires_grad], "lr": args.lr_backbone},
            {"params": other_params, "lr": args.lr},
        ],
        weight_decay=1e-4,
    )

    model.train()
    history = []
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        running = {"loss_total": 0.0, "loss_class": 0.0, "loss_bbox": 0.0, "loss_giou": 0.0}

        for step, (images, targets) in enumerate(loader, start=1):
            images = images.to(device)
            targets = [{"boxes": t["boxes"].to(device), "labels": t["labels"].to(device)} for t in targets]

            outputs = model(images)
            losses = criterion(outputs, targets)

            optimizer.zero_grad()
            losses["loss_total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()

            for k in running:
                running[k] += losses[k].item()

            if step % args.log_every == 0 or step == len(loader):
                avg = {k: v / step for k, v in running.items()}
                print(
                    f"epoch {epoch}/{args.epochs} step {step}/{len(loader)} | "
                    f"total {avg['loss_total']:.3f} | class {avg['loss_class']:.3f} | "
                    f"bbox {avg['loss_bbox']:.3f} | giou {avg['loss_giou']:.3f}"
                )

        epoch_avg = {k: v / len(loader) for k, v in running.items()}
        epoch_time = time.time() - epoch_start
        print(f"== epoch {epoch} done in {epoch_time:.1f}s | avg loss {epoch_avg['loss_total']:.3f} ==")
        history.append(epoch_avg)

        ckpt_path = Path(args.checkpoint_dir) / f"ydfnet_epoch{epoch}.pt"
        torch.save(
            {"model_state": model.state_dict(), "args": vars(args), "epoch": epoch, "loss": epoch_avg},
            ckpt_path,
        )
        print(f"[checkpoint] saved {ckpt_path}")

    print("\nTraining complete. Loss history:")
    for i, h in enumerate(history, start=1):
        print(f"  epoch {i}: {h['loss_total']:.4f}")


if __name__ == "__main__":
    main()