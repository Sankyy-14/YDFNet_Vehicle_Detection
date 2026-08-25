"""
YDFNet: YOLOv10 backbone + BiFPN neck + DETR-style head.

End-to-end assembly of the three pieces built in backbone.py, bifpn.py, and
detr_head.py. This mirrors the architecture described in the research paper
your team is using to justify the Vehicle Detection module:

    Image -> YOLOv10 backbone (P3/P4/P5) -> BiFPN (multi-scale fusion)
          -> DETR head (set prediction, no NMS) -> class + bbox predictions

Design choice: the DETR head in this prototype attends over the BiFPN P4
scale only (stride 16, e.g. 40x40 tokens @ 640 input = 1600 tokens). This is
a deliberate simplification — a full multi-scale deformable-attention DETR
(as in Deformable DETR) would attend over P3+P4+P5 jointly for better small
object recall, at higher compute cost. Attending over P3 alone would be very
expensive (80x80 = 6400 tokens, transformer self-attention is O(n^2)) and
attending over P5 alone loses too much spatial detail for small/distant
vehicles. P4 is the standard tradeoff point in DETR-style detectors. See
README "Known Limitations" for how to extend this to full multi-scale
attention if your team wants a closer match to the paper.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from models.backbone import YOLOv10Backbone
from models.bifpn import BiFPN
from models.detr_head import DETRHead


class YDFNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 4,
        yolo_variant: str = "yolov10n.pt",
        freeze_backbone: bool = False,
        bifpn_channels: int = 128,
        bifpn_layers: int = 3,
        num_queries: int = 150,
        detr_layers: int = 4,
        detr_heads: int = 8,
        head_scale: str = "p4",
    ):
        super().__init__()
        assert head_scale in ("p3", "p4", "p5"), "head_scale must be one of p3/p4/p5"
        self.head_scale = head_scale

        self.backbone = YOLOv10Backbone(yolo_variant, freeze=freeze_backbone)
        self.bifpn = BiFPN(
            in_channels=self.backbone.out_channels,
            out_channels=bifpn_channels,
            num_layers=bifpn_layers,
        )
        self.head = DETRHead(
            in_channels=bifpn_channels,
            num_classes=num_classes,
            num_queries=num_queries,
            d_model=bifpn_channels,
            nhead=detr_heads,
            num_decoder_layers=detr_layers,
        )
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Args:
            x: image batch, (B, 3, H, W), H and W multiples of 32.

        Returns:
            dict with 'pred_logits' (B, num_queries, num_classes+1) and
            'pred_boxes' (B, num_queries, 4) in normalized (cx, cy, w, h).
        """
        backbone_feats = self.backbone(x)
        fused_feats = self.bifpn(backbone_feats)
        return self.head(fused_feats[self.head_scale])

    @torch.no_grad()
    def predict(
        self, x: torch.Tensor, score_threshold: float = 0.5
    ) -> list[dict[str, torch.Tensor]]:
        """Convenience inference wrapper: runs forward pass, applies softmax
        to class logits, and filters out low-confidence / "no object" queries.
        No NMS step is needed — that's the whole point of the DETR head.

        Returns a list (one entry per image in the batch) of dicts with keys
        'boxes' (N, 4) in normalized (cx, cy, w, h), 'scores' (N,), and
        'labels' (N,) (0-indexed into your class list, background excluded).
        """
        self.eval()
        out = self.forward(x)
        probs = out["pred_logits"].softmax(-1)  # (B, Q, num_classes+1)
        scores, labels = probs[..., :-1].max(-1)  # exclude background class

        results = []
        for b in range(x.shape[0]):
            keep = scores[b] > score_threshold
            results.append(
                {
                    "boxes": out["pred_boxes"][b][keep],
                    "scores": scores[b][keep],
                    "labels": labels[b][keep],
                }
            )
        return results


if __name__ == "__main__":
    # Quick smoke test: python -m models.ydfnet
    model = YDFNet(num_classes=4)
    model.eval()
    dummy = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        out = model(dummy)
    print("pred_logits:", tuple(out["pred_logits"].shape))
    print("pred_boxes:", tuple(out["pred_boxes"].shape))

    preds = model.predict(dummy, score_threshold=0.0)  # threshold=0 just to see shapes
    print("predict() output keys:", preds[0].keys())
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {n_params:,} | Trainable: {n_trainable:,}")