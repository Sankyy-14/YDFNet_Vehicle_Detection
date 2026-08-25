"""
DETR-style detection head for YDFNet.

Why a DETR head instead of YOLO's native anchor-based / anchor-free head:
YOLO-style heads predict boxes independently per grid cell/anchor, then rely
on NMS (non-max suppression) to remove duplicate detections. NMS is exactly
where dense, overlapping traffic scenes break down: two vehicles that are
genuinely close together (e.g. bumper-to-bumper at a red light) produce boxes
that NMS can't tell apart from duplicate detections of the *same* vehicle, so
one of them often gets incorrectly suppressed.

DETR reframes detection as direct *set prediction*: a fixed number of learned
"object queries" attend over the whole image (via a transformer decoder) and
each query predicts at most one object, with a bipartite (Hungarian) matching
loss during training that assigns each query to a distinct ground-truth
object. This removes NMS entirely and lets the model reason about all
detections jointly, which is what gives it an edge on occluded/overlapping
vehicles.

This implementation is a simplified single-scale DETR head: it takes BiFPN's
fused P4 map (a reasonable resolution/semantics tradeoff — P3 is large and
slow to attend over, P5 loses too much fine detail), flattens it into a
sequence of tokens, and decodes `num_queries` object predictions.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class PositionalEncoding2D(nn.Module):
    """Fixed sine/cosine 2D positional encoding, added to flattened feature
    tokens so the transformer knows each token's spatial location (attention
    itself is permutation-invariant and has no notion of position otherwise).
    """

    def __init__(self, channels: int):
        super().__init__()
        assert channels % 4 == 0, "channels must be divisible by 4 for 2D sin/cos encoding"
        self.channels = channels

    def forward(self, h: int, w: int, device) -> torch.Tensor:
        c = self.channels // 2
        y_pos = torch.arange(h, device=device).unsqueeze(1).repeat(1, w)
        x_pos = torch.arange(w, device=device).unsqueeze(0).repeat(h, 1)

        div_term = torch.exp(
            torch.arange(0, c, 2, device=device).float() * (-torch.log(torch.tensor(10000.0)) / c)
        )

        pe = torch.zeros(h, w, self.channels, device=device)
        pe[:, :, 0:c:2] = torch.sin(x_pos.unsqueeze(-1) * div_term)
        pe[:, :, 1:c:2] = torch.cos(x_pos.unsqueeze(-1) * div_term)
        pe[:, :, c::2] = torch.sin(y_pos.unsqueeze(-1) * div_term)
        pe[:, :, c + 1 :: 2] = torch.cos(y_pos.unsqueeze(-1) * div_term)
        return pe.flatten(0, 1)  # (H*W, channels)


class MLP(nn.Module):
    """Simple N-layer feed-forward network, used for the bbox regression head
    (standard in the original DETR paper)."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int = 3):
        super().__init__()
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.layers = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = torch.relu(layer(x)) if i < len(self.layers) - 1 else layer(x)
        return x


class DETRHead(nn.Module):
    """DETR-style transformer decoder detection head.

    Args:
        in_channels: channel count of the incoming BiFPN feature map.
        num_classes: number of vehicle classes to predict (NOT including the
            "no object" background class — that's added automatically as an
            extra logit).
        num_queries: number of object slots. Should comfortably exceed the
            max number of vehicles you expect in one frame. UA-DETRAC scenes
            can be dense (busy intersections), so we default higher than
            vanilla DETR's COCO-tuned 100.
        d_model: transformer embedding dimension.
        nhead: number of attention heads.
        num_decoder_layers: depth of the transformer decoder.
    """

    def __init__(
        self,
        in_channels: int = 128,
        num_classes: int = 4,  # e.g. car, bus, truck, van (adjust to your label map)
        num_queries: int = 150,
        d_model: int = 128,
        nhead: int = 8,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 512,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_queries = num_queries

        self.input_proj = nn.Conv2d(in_channels, d_model, kernel_size=1)
        self.pos_encoding = PositionalEncoding2D(d_model)

        self.query_embed = nn.Embedding(num_queries, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # +1 class for "no object" / background, as in the original DETR.
        self.class_head = nn.Linear(d_model, num_classes + 1)
        self.bbox_head = MLP(d_model, d_model, 4, num_layers=3)  # (cx, cy, w, h) normalized [0,1]

    def forward(self, feat_map: torch.Tensor) -> dict[str, torch.Tensor]:
        """Args:
            feat_map: BiFPN feature map, shape (B, C, H, W).

        Returns:
            dict with:
                'pred_logits': (B, num_queries, num_classes + 1)
                'pred_boxes':  (B, num_queries, 4), normalized (cx, cy, w, h)
        """
        b, c, h, w = feat_map.shape
        x = self.input_proj(feat_map)  # (B, d_model, H, W)

        tokens = x.flatten(2).permute(0, 2, 1)  # (B, H*W, d_model)
        pos = self.pos_encoding(h, w, feat_map.device).unsqueeze(0)  # (1, H*W, d_model)
        memory = tokens + pos

        queries = self.query_embed.weight.unsqueeze(0).repeat(b, 1, 1)  # (B, num_queries, d_model)

        decoded = self.decoder(tgt=queries, memory=memory)  # (B, num_queries, d_model)

        pred_logits = self.class_head(decoded)
        pred_boxes = self.bbox_head(decoded).sigmoid()  # squash to [0,1] normalized coords

        return {"pred_logits": pred_logits, "pred_boxes": pred_boxes}


if __name__ == "__main__":
    # Quick smoke test: python -m models.detr_head
    head = DETRHead(in_channels=128, num_classes=4, num_queries=150)
    dummy = torch.randn(2, 128, 40, 40)  # BiFPN P4-scale output
    out = head(dummy)
    print("pred_logits:", tuple(out["pred_logits"].shape))
    print("pred_boxes:", tuple(out["pred_boxes"].shape))