"""BiSeNetV2 variant exposing pre-BGA branch features for decorrelation training."""
from __future__ import annotations

from typing import Dict, List, Union

import torch
import torch.nn.functional as F

from bisenetv2 import BiSeNetV2


class BiSeNetV2Orth(BiSeNetV2):
    """BiSeNetV2 with an optional cross-branch orthogonality training output."""

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Dict[str, object]]:
        output_size = x.shape[-2:]
        detail = self.detail(x)
        feat2, feat3, feat4, feat5, semantic = self.semantic(x)
        fused = self.bga(detail, semantic)
        logits = self.head(fused, output_size)

        if self.training:
            output: Dict[str, object] = {"out": logits}
            if self.auxiliary:
                aux_features = (feat2, feat3, feat4, feat5)
                output["aux"] = [
                    head(feature, output_size)
                    for head, feature in zip(self.aux_heads, aux_features)
                ]
            output["detail_feature"] = detail
            output["semantic_feature"] = semantic
            return output
        return logits


def build_bisenetv2_orth(**kwargs) -> BiSeNetV2Orth:
    """Build the decorrelation variant with the same options as BiSeNetV2."""
    return BiSeNetV2Orth(**kwargs)


def branch_orthogonality_loss(
    detail: torch.Tensor,
    semantic: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Penalize cross-branch channel correlation on aligned coarse locations.

    Detail features are pooled to the semantic feature resolution. Channels are
    standardized over batch and spatial samples before forming a cross-correlation
    matrix, avoiding a loss that can be minimized merely by shrinking activations.
    """
    detail = F.adaptive_avg_pool2d(detail, output_size=semantic.shape[-2:])
    detail_flat = detail.permute(1, 0, 2, 3).reshape(detail.shape[1], -1)
    semantic_flat = semantic.permute(1, 0, 2, 3).reshape(semantic.shape[1], -1)

    detail_flat = detail_flat - detail_flat.mean(dim=1, keepdim=True)
    semantic_flat = semantic_flat - semantic_flat.mean(dim=1, keepdim=True)
    detail_flat = detail_flat / (detail_flat.std(dim=1, keepdim=True, unbiased=False) + eps)
    semantic_flat = semantic_flat / (semantic_flat.std(dim=1, keepdim=True, unbiased=False) + eps)

    sample_count = max(detail_flat.shape[1], 1)
    cross_correlation = (detail_flat @ semantic_flat.transpose(0, 1)) / sample_count
    channel_pairs = max(detail_flat.shape[0] * semantic_flat.shape[0], 1)
    return cross_correlation.square().sum() * (sample_count / channel_pairs)
