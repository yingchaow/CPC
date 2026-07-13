from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .center import PrototypeCenterLoss
from .cmp import cmp_margin_loss
from .pairwise import pairwise_logistic_loss
from .regularization import quantization_loss


@dataclass
class CompositeLossOutput:
    total: torch.Tensor
    components: dict


class CompositeHashLoss(nn.Module):
    def __init__(self, config, num_classes, hash_bits):
        super().__init__()
        self.config = config
        self.center_loss = PrototypeCenterLoss(
            num_classes=num_classes,
            hash_bits=hash_bits,
            momentum=config.loss.center.momentum,
            temperature=config.loss.center.temperature,
            rgce_r=config.loss.center.rgce_r,
            update=config.loss.center.update,
            dual_center_enabled=config.loss.center.dual_center.enabled,
            dual_center_hard_weight=(
                config.loss.center.dual_center.hard_weight
            ),
            dual_center_separation_weight=(
                config.loss.center.dual_center.separation_weight
            ),
            dual_center_margin=config.loss.center.dual_center.margin,
            dual_center_warmup_epochs=(
                config.loss.center.dual_center.warmup_epochs
            ),
            dual_center_top_k=config.loss.center.dual_center.top_k,
            dual_center_reliability_enabled=(
                config.loss.center.dual_center.reliability_enabled
            ),
            dual_center_negative_centers=(
                config.loss.center.dual_center.negative_centers
            ),
            dual_center_diversity_weight=(
                config.loss.center.dual_center.diversity_weight
            ),
            dual_center_hash_quantization_weight=(
                config.loss.center.dual_center.hash_quantization_weight
            ),
        )

    @staticmethod
    def _zero(reference):
        return reference.new_zeros(())

    @staticmethod
    def _check(name, value):
        if not torch.isfinite(value).all():
            raise FloatingPointError(f"{name} contains NaN or Inf")

    def forward(
        self,
        image_hash,
        text_hash,
        labels,
        selected,
        image_logits=None,
        text_logits=None,
        classification_weights=None,
        classification_targets=None,
        current_epoch=None,
    ):
        selected = selected.bool()
        reliability = None
        if classification_weights is not None:
            reliability = classification_weights.detach().to(
                device=image_hash.device,
                dtype=image_hash.dtype,
            )
            if reliability.shape != selected.shape:
                raise ValueError(
                    "classification_weights must have one value per sample"
                )
            if (reliability < 0).any():
                raise ValueError(
                    "classification_weights must be nonnegative"
                )
        selected_reliability = (
            reliability[selected] if reliability is not None else None
        )
        components = {
            name: self._zero(image_hash)
            for name in (
                "pairwise",
                "center",
                "quantization",
                "classification",
                "cmp",
            )
        }
        if self.config.loss.pairwise.enabled and selected.any():
            components["pairwise"] = pairwise_logistic_loss(
                image_hash[selected],
                text_hash[selected],
                labels[selected],
                mode=self.config.loss.pairwise.mode,
                margin=self.config.loss.pairwise.margin,
                shift=self.config.loss.pairwise.shift,
                temperature=self.config.loss.pairwise.temperature,
                sample_weights=selected_reliability,
            ).mean
        if self.config.loss.center.enabled and selected.any():
            center_output = self.center_loss(
                image_hash[selected],
                text_hash[selected],
                labels[selected],
                reliability=selected_reliability,
                current_epoch=current_epoch,
            )
            components["center"] = self._weighted_mean(
                center_output.per_sample,
                selected_reliability,
            )
        if self.config.loss.quantization.enabled:
            components["quantization"] = quantization_loss(
                image_hash, text_hash
            )
        if self.config.loss.classification.enabled:
            if image_logits is None or text_logits is None:
                raise ValueError(
                    "classification loss requires image/text logits"
                )
            targets = (
                labels.float()
                if classification_targets is None
                else classification_targets.to(
                    device=labels.device,
                    dtype=labels.dtype,
                )
            )
            if targets.shape != labels.shape:
                raise ValueError(
                    "classification targets must match label shape"
                )
            image_classification = F.binary_cross_entropy_with_logits(
                image_logits,
                targets,
                reduction="none",
            ).mean(dim=1)
            text_classification = F.binary_cross_entropy_with_logits(
                text_logits,
                targets,
                reduction="none",
            ).mean(dim=1)
            per_sample_classification = (
                image_classification + text_classification
            ) / 2.0
            if reliability is None:
                components["classification"] = (
                    per_sample_classification.mean()
                )
            else:
                components["classification"] = (
                    per_sample_classification * reliability
                ).mean()
        if self.config.loss.cmp.enabled and selected.any():
            components["cmp"] = cmp_margin_loss(
                image_hash[selected],
                text_hash[selected],
                labels[selected],
                margin=self.config.loss.cmp.margin,
                sample_weights=selected_reliability,
            ).mean
        total = self._zero(image_hash)
        for name in components:
            value = components[name]
            self._check(name, value)
            total = total + getattr(self.config.loss, name).weight * value
        self._check("total", total)
        return CompositeLossOutput(total, components)

    @staticmethod
    def _weighted_mean(values, weights):
        if weights is None:
            return values.mean()
        return (values * weights).sum() / weights.sum().clamp_min(1e-8)
