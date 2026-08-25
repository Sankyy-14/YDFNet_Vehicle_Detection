"""
Bipartite matching + set-prediction loss for the DETR-style head.

Why this is needed (and why it's the trickiest part of DETR to get right):
The model outputs a fixed set of `num_queries` predictions per image, but
each image has a variable, usually much smaller, number of ground-truth
vehicles. Before we can compute a loss, we first need to decide *which*
prediction should be compared against *which* ground-truth box (and which
predictions should be pushed towards the "no object" class). This is solved
optimally via the Hungarian algorithm on a pairwise cost matrix that
combines classification cost, L1 box distance, and GIoU — exactly as in the
original DETR paper (Carion et al., 2020).

Once the assignment is fixed, the loss itself is just:
  - cross-entropy over classes for every query (matched -> its GT class,
    unmatched -> background)
  - L1 loss between matched boxes
  - GIoU loss between matched boxes (helps with scale-invariance, since L1
    alone penalizes errors on large boxes more than small ones)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    cx, cy, w, h = boxes.unbind(-1)
    x1, y1 = cx - w / 2, cy - h / 2
    x2, y2 = cx + w / 2, cy + h / 2
    return torch.stack([x1, y1, x2, y2], dim=-1)


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """GIoU between every pair of boxes in boxes1 (N,4) and boxes2 (M,4),
    both in (x1,y1,x2,y2) format. Returns (N, M) matrix."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2[None, :] - inter
    iou = inter / union.clamp(min=1e-6)

    lt_c = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    rb_c = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh_c = (rb_c - lt_c).clamp(min=0)
    area_c = wh_c[:, :, 0] * wh_c[:, :, 1]

    return iou - (area_c - union) / area_c.clamp(min=1e-6)


class HungarianMatcher(nn.Module):
    """Computes an optimal bipartite matching between predictions and targets."""

    def __init__(self, cost_class: float = 1.0, cost_bbox: float = 5.0, cost_giou: float = 2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(self, outputs: dict, targets: list[dict]) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Args:
            outputs: dict with 'pred_logits' (B,Q,C+1) and 'pred_boxes' (B,Q,4)
            targets: list of dicts (len B), each with 'labels' (n_i,) and
                'boxes' (n_i, 4) in normalized (cx,cy,w,h)

        Returns:
            list of (pred_idx, target_idx) index tensors, one pair per image.
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # (B*Q, C+1)
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # (B*Q, 4)

        tgt_ids = torch.cat([t["labels"] for t in targets])
        tgt_bbox = torch.cat([t["boxes"] for t in targets])

        if tgt_ids.numel() == 0:
            # No ground truth at all in this batch; every query matches nothing.
            return [(torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)) for _ in targets]

        cost_class = -out_prob[:, tgt_ids]
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        cost_giou = -generalized_box_iou(
            box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox)
        )

        cost = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        cost = cost.view(bs, num_queries, -1).cpu()

        sizes = [len(t["boxes"]) for t in targets]
        indices = []
        for i, c in enumerate(cost.split(sizes, dim=-1)):
            if sizes[i] == 0:
                indices.append((torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)))
                continue
            row, col = linear_sum_assignment(c[i].numpy())
            indices.append((torch.as_tensor(row, dtype=torch.long), torch.as_tensor(col, dtype=torch.long)))
        return indices


class SetCriterion(nn.Module):
    """Combines the Hungarian matcher with the actual loss computation."""

    def __init__(
        self,
        num_classes: int,
        matcher: HungarianMatcher,
        eos_coef: float = 0.1,
        weight_class: float = 1.0,
        weight_bbox: float = 5.0,
        weight_giou: float = 2.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_class = weight_class
        self.weight_bbox = weight_bbox
        self.weight_giou = weight_giou

        # Down-weight the background/"no object" class in the CE loss, since
        # most queries in any given image are background — without this the
        # model just learns to always predict "no object".
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def forward(self, outputs: dict, targets: list[dict]) -> dict[str, torch.Tensor]:
        indices = self.matcher(outputs, targets)

        # --- Classification loss ---
        pred_logits = outputs["pred_logits"]  # (B, Q, C+1)
        bg_class = self.num_classes
        target_classes = torch.full(
            pred_logits.shape[:2], bg_class, dtype=torch.long, device=pred_logits.device
        )
        for b, (pred_idx, tgt_idx) in enumerate(indices):
            if len(pred_idx) > 0:
                target_classes[b, pred_idx] = targets[b]["labels"][tgt_idx]

        loss_class = F.cross_entropy(
            pred_logits.transpose(1, 2), target_classes, weight=self.empty_weight
        )

        # --- Box losses (only over matched pairs) ---
        pred_boxes_matched, tgt_boxes_matched = [], []
        for b, (pred_idx, tgt_idx) in enumerate(indices):
            if len(pred_idx) > 0:
                pred_boxes_matched.append(outputs["pred_boxes"][b, pred_idx])
                tgt_boxes_matched.append(targets[b]["boxes"][tgt_idx])

        if len(pred_boxes_matched) > 0:
            pred_boxes_matched = torch.cat(pred_boxes_matched)
            tgt_boxes_matched = torch.cat(tgt_boxes_matched)

            loss_bbox = F.l1_loss(pred_boxes_matched, tgt_boxes_matched, reduction="mean")

            giou = generalized_box_iou(
                box_cxcywh_to_xyxy(pred_boxes_matched), box_cxcywh_to_xyxy(tgt_boxes_matched)
            ).diag()
            loss_giou = (1 - giou).mean()
        else:
            loss_bbox = torch.tensor(0.0, device=pred_logits.device)
            loss_giou = torch.tensor(0.0, device=pred_logits.device)

        total = (
            self.weight_class * loss_class
            + self.weight_bbox * loss_bbox
            + self.weight_giou * loss_giou
        )
        return {
            "loss_total": total,
            "loss_class": loss_class.detach(),
            "loss_bbox": loss_bbox.detach() if torch.is_tensor(loss_bbox) else loss_bbox,
            "loss_giou": loss_giou.detach() if torch.is_tensor(loss_giou) else loss_giou,
        }


if __name__ == "__main__":
    # Quick smoke test with random predictions/targets: python -m models.loss
    torch.manual_seed(0)
    B, Q, C = 2, 150, 4
    outputs = {
        "pred_logits": torch.randn(B, Q, C + 1, requires_grad=True),
        "pred_boxes": torch.rand(B, Q, 4, requires_grad=True),
    }
    targets = [
        {"labels": torch.tensor([0, 2]), "boxes": torch.rand(2, 4)},
        {"labels": torch.tensor([1]), "boxes": torch.rand(1, 4)},
    ]
    matcher = HungarianMatcher()
    criterion = SetCriterion(num_classes=C, matcher=matcher)
    losses = criterion(outputs, targets)
    for k, v in losses.items():
        print(f"{k}: {v.item():.4f}")
    losses["loss_total"].backward()
    print("Backward pass OK — gradients flow correctly through matching + loss.")