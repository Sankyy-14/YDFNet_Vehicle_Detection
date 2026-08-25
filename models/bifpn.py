"""
Bidirectional Feature Pyramid Network (BiFPN), as used in EfficientDet and in
the YDFNet paper's neck.

Why BiFPN instead of a plain FPN/PAN:
A standard top-down FPN only lets high-level (semantic, low-resolution)
features flow down to help low-level (fine, high-resolution) features.
BiFPN adds a second, bottom-up pass so fine-grained detail also flows back
up into the semantic features, and repeats this bidirectional fusion across
several stacked layers. It also uses *learned, per-input weights* when fusing
feature maps at each node, so the network can decide how much to trust each
scale rather than treating them equally. This is exactly what the YDFNet
paper argues improves detection of both tiny, distant vehicles (which need
fine detail) and large, close vehicles (which need semantic context) in the
same frame.

This module implements a simplified single-cell BiFPN (3 input levels: P3,
P4, P5) that can be stacked N times (num_layers) as in the original paper.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    """Depthwise-separable conv: cheaper than a full conv, standard in BiFPN."""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.depthwise = nn.Conv2d(
            channels, channels, kernel_size, padding=pad, groups=channels, bias=False
        )
        self.pointwise = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(channels, eps=1e-3, momentum=0.01)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class WeightedFeatureFusion(nn.Module):
    """Fast normalized weighted fusion of 2 or 3 input feature maps (Eq. from
    the EfficientDet paper): out = sum(w_i * relu(w_i) * in_i) / (sum(w_i) + eps)
    """

    def __init__(self, num_inputs: int, eps: float = 1e-4):
        super().__init__()
        self.eps = eps
        self.weights = nn.Parameter(torch.ones(num_inputs, dtype=torch.float32))

    def forward(self, inputs: list[torch.Tensor]) -> torch.Tensor:
        w = F.relu(self.weights)
        w = w / (w.sum() + self.eps)
        out = sum(w[i] * inputs[i] for i in range(len(inputs)))
        return out


class BiFPNLayer(nn.Module):
    """A single BiFPN layer: one top-down pass then one bottom-up pass over
    P3, P4, P5, with weighted fusion + depthwise-separable conv at each node.
    """

    def __init__(self, channels: int):
        super().__init__()
        # Top-down pathway (P5 -> P4 -> P3)
        self.fuse_p4_td = WeightedFeatureFusion(2)
        self.fuse_p3_out = WeightedFeatureFusion(2)
        self.conv_p4_td = DepthwiseSeparableConv(channels)
        self.conv_p3_out = DepthwiseSeparableConv(channels)

        # Bottom-up pathway (P3 -> P4 -> P5)
        self.fuse_p4_out = WeightedFeatureFusion(3)  # p4_in + p4_td + p3_out(down)
        self.fuse_p5_out = WeightedFeatureFusion(2)  # p5_in + p4_out(down)
        self.conv_p4_out = DepthwiseSeparableConv(channels)
        self.conv_p5_out = DepthwiseSeparableConv(channels)

    def forward(self, feats: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        p3_in, p4_in, p5_in = feats["p3"], feats["p4"], feats["p5"]

        # --- Top-down pass ---
        p5_up = F.interpolate(p5_in, size=p4_in.shape[-2:], mode="nearest")
        p4_td = self.conv_p4_td(self.fuse_p4_td([p4_in, p5_up]))

        p4_up = F.interpolate(p4_td, size=p3_in.shape[-2:], mode="nearest")
        p3_out = self.conv_p3_out(self.fuse_p3_out([p3_in, p4_up]))

        # --- Bottom-up pass ---
        p3_down = F.max_pool2d(p3_out, kernel_size=2)
        p4_out = self.conv_p4_out(self.fuse_p4_out([p4_in, p4_td, p3_down]))

        p4_down = F.max_pool2d(p4_out, kernel_size=2)
        p5_out = self.conv_p5_out(self.fuse_p5_out([p5_in, p4_down]))

        return {"p3": p3_out, "p4": p4_out, "p5": p5_out}


class BiFPN(nn.Module):
    """Stack of BiFPNLayers. Projects backbone channels (which differ per
    scale) to a common `out_channels` before the first layer, since BiFPN
    fusion requires matching channel counts across scales.
    """

    def __init__(
        self,
        in_channels: tuple[int, int, int],
        out_channels: int = 128,
        num_layers: int = 3,
    ):
        super().__init__()
        c3, c4, c5 = in_channels
        self.proj_p3 = nn.Conv2d(c3, out_channels, 1)
        self.proj_p4 = nn.Conv2d(c4, out_channels, 1)
        self.proj_p5 = nn.Conv2d(c5, out_channels, 1)

        self.layers = nn.ModuleList(
            [BiFPNLayer(out_channels) for _ in range(num_layers)]
        )
        self.out_channels = out_channels

    def forward(self, feats: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        x = {
            "p3": self.proj_p3(feats["p3"]),
            "p4": self.proj_p4(feats["p4"]),
            "p5": self.proj_p5(feats["p5"]),
        }
        for layer in self.layers:
            x = layer(x)
        return x


if __name__ == "__main__":
    # Quick smoke test: python -m models.bifpn
    bifpn = BiFPN(in_channels=(64, 128, 256), out_channels=128, num_layers=3)
    dummy = {
        "p3": torch.randn(2, 64, 80, 80),
        "p4": torch.randn(2, 128, 40, 40),
        "p5": torch.randn(2, 256, 20, 20),
    }
    out = bifpn(dummy)
    for k, v in out.items():
        print(f"{k}: {tuple(v.shape)}")