"""
YOLOv10 backbone wrapper for YDFNet.

Loads a pretrained YOLOv10 detection model (via ultralytics) and strips off its
native PAN neck + v10Detect head, exposing only the raw multi-scale backbone
features (P3, P4, P5). These are the feature maps that BiFPN (models/bifpn.py)
and the DETR-style head (models/detr_head.py) consume instead.

Why bypass YOLOv10's own neck?
YOLOv10 already ships with its own FPN+PAN neck (layers 11-22) feeding into a
v10Detect head. YDFNet's whole point is to *replace* that neck+head combo with
BiFPN (better multi-scale fusion) + a DETR-style head (better occlusion
handling), so we only keep the backbone (layers 0-10) and discard the rest.

Layer indices below were determined empirically for yolov10n (see
inspect scripts / README) and correspond to:
    layer 4  -> P3, stride 8,  64 channels,  80x80 @ 640 input
    layer 6  -> P4, stride 16, 128 channels, 40x40 @ 640 input
    layer 10 -> P5, stride 32, 256 channels, 20x20 @ 640 input

If you swap to yolov10s/m/l/x, re-run utils/inspect_backbone.py to confirm
these indices and channel counts still hold (channel counts scale with model
size; layer indices should stay the same since the backbone topology doesn't
change across yolov10 variants).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from ultralytics import YOLO

# Layer indices (in the underlying nn.Sequential) whose outputs are P3/P4/P5.
P3_LAYER_IDX = 4
P4_LAYER_IDX = 6
P5_LAYER_IDX = 10


class YOLOv10Backbone(nn.Module):
    """Extracts P3/P4/P5 feature maps from a pretrained YOLOv10 backbone.

    Args:
        variant: which pretrained YOLOv10 checkpoint to start from, e.g.
            'yolov10n.pt', 'yolov10s.pt', 'yolov10m.pt'. Downloaded
            automatically by ultralytics on first use.
        freeze: if True, backbone weights are frozen (useful for the first
            few epochs of fine-tuning so the randomly-initialized BiFPN/DETR
            head doesn't immediately destroy pretrained backbone features).
    """

    # Output channel counts per scale, indexed by yolov10 variant prefix.
    # (n=nano, s=small, m=medium — extend this dict if you use l/x variants)
    CHANNELS_BY_VARIANT = {
        "yolov10n": (64, 128, 256),
        "yolov10s": (128, 256, 512),
        "yolov10m": (192, 384, 576),
    }

    def __init__(self, variant: str = "yolov10n.pt", freeze: bool = False):
        super().__init__()
        yolo = YOLO(variant)
        full_model = yolo.model.model  # nn.Sequential of all layers

        # Keep only the backbone layers (everything up to and including P5).
        # We run these manually in forward() rather than as a Sequential,
        # since a couple of the kept blocks reference earlier saved outputs
        # is NOT the case here (0-10 is a clean, non-branching chain for
        # yolov10n/s/m — confirmed via utils/inspect_backbone.py).
        self.layers = nn.ModuleList([full_model[i] for i in range(P5_LAYER_IDX + 1)])

        self.p3_idx = P3_LAYER_IDX
        self.p4_idx = P4_LAYER_IDX
        self.p5_idx = P5_LAYER_IDX

        key = variant.replace(".pt", "")
        self.out_channels = self.CHANNELS_BY_VARIANT.get(key, (64, 128, 256))

        if freeze:
            for p in self.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Args:
            x: input image batch, shape (B, 3, H, W). H and W should be
               multiples of 32 (YOLO's total backbone stride).

        Returns:
            dict with keys 'p3', 'p4', 'p5' mapping to feature maps of shape
            (B, C3, H/8, W/8), (B, C4, H/16, W/16), (B, C5, H/32, W/32).
        """
        feats = {}
        out = x
        for i, layer in enumerate(self.layers):
            out = layer(out)
            if i == self.p3_idx:
                feats["p3"] = out
            elif i == self.p4_idx:
                feats["p4"] = out
            elif i == self.p5_idx:
                feats["p5"] = out
        return feats


if __name__ == "__main__":
    # Quick smoke test: python -m models.backbone
    backbone = YOLOv10Backbone("yolov10n.pt")
    backbone.eval()
    dummy = torch.randn(2, 3, 640, 640)
    with torch.no_grad():
        feats = backbone(dummy)
    for k, v in feats.items():
        print(f"{k}: {tuple(v.shape)}")