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
            num_classes,
            hash_bits,
            config.loss.center.momentum,
            config.loss.center.temperature,
            config.loss.center.rgce_r,
            config.loss.center.update,
            config.loss.center.hard_negative.enabled,
            config.loss.center.hard_negative.alpha,
            config.loss.center.hard_negative.margin,
            config.loss.center.hard_negative.label_threshold,
            config.loss.center.dual_center.enabled,
            config.loss.center.dual_center.hard_weight,
            config.loss.center.dual_center.separation_weight,
            config.loss.center.dual_center.margin,
            config.loss.center.dual_center.warmup_epochs,
            config.loss.center.dual_center.top_k,
            config.loss.center.dual_center.reliability_enabled,
            config.loss.center.dual_center.negative_centers,
            config.loss.center.dual_center.diversity_weight,
            config.loss.center.dual_center.hash_quantization_weight,
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
                similarity=self.config.loss.pairwise.similarity,
                hard_negative_enabled=(
                    self.config.loss.pairwise.hard_negative.enabled
                ),
                hard_negative_alpha=(
                    self.config.loss.pairwise.hard_negative.alpha
                ),
                hard_negative_margin=(
                    self.config.loss.pairwise.hard_negative.margin
                ),
                hard_negative_label_threshold=(
                    self.config.loss.pairwise.hard_negative.label_threshold
                ),
            ).mean
        if self.config.loss.center.enabled and selected.any():
            center_reliability = self._center_reliability(
                labels[selected],
                (
                    classification_targets[selected]
                    if classification_targets is not None
                    else None
                ),
            )
            center_output = self.center_loss(
                image_hash[selected],
                text_hash[selected],
                labels[selected],
                reliability=center_reliability,
                current_epoch=current_epoch,
            )
            components["center"] = center_output.mean
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
            if classification_weights is None:
                components["classification"] = (
                    per_sample_classification.mean()
                )
            else:
                classification_weights = classification_weights.to(
                    device=per_sample_classification.device,
                    dtype=per_sample_classification.dtype,
                )
                if classification_weights.shape != (
                    per_sample_classification.shape
                ):
                    raise ValueError(
                        "classification weights must have one value "
                        "per sample"
                    )
                components["classification"] = (
                    per_sample_classification * classification_weights
                ).mean()
        if self.config.loss.cmp.enabled and selected.any():
            components["cmp"] = cmp_margin_loss(
                image_hash[selected],
                text_hash[selected],
                margin=self.config.loss.cmp.margin,
            ).mean
        total = self._zero(image_hash)
        for name in components:
            value = components[name]
            self._check(name, value)
            total = total + getattr(self.config.loss, name).weight * value
        self._check("total", total)
        return CompositeLossOutput(total, components)

    @staticmethod
    def _center_reliability(labels, classification_targets):
        if classification_targets is None:
            return None
        targets = classification_targets.to(
            device=labels.device,
            dtype=labels.dtype,
        )
        positive_count = labels.float().sum(dim=1).clamp_min(1.0)
        reliability = (
            labels.float() * targets.float()
        ).sum(dim=1) / positive_count
        return reliability.clamp(0.0, 1.0).detach()
