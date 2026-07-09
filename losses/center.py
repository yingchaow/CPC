import torch
import torch.nn as nn
import torch.nn.functional as F

from .pairwise import LossOutput


class PrototypeCenterLoss(nn.Module):
    def __init__(
        self,
        num_classes,
        hash_bits,
        momentum,
        temperature,
        rgce_r,
        update="ema",
        dual_center_enabled=False,
        dual_center_hard_weight=0.1,
        dual_center_separation_weight=0.1,
        dual_center_margin=0.2,
        dual_center_warmup_epochs=0,
        dual_center_top_k=0,
        dual_center_reliability_enabled=False,
        dual_center_negative_centers=1,
        dual_center_diversity_weight=0.0,
        dual_center_hash_quantization_weight=0.0,
    ):
        super().__init__()
        self.update_mode = update
        self.dual_center_enabled = bool(dual_center_enabled)
        if self.dual_center_enabled and update != "learnable":
            raise ValueError("dual center requires learnable update")
        prototypes = F.normalize(
            torch.randn(num_classes, hash_bits), p=2, dim=1
        )
        if update == "learnable":
            self.prototypes = nn.Parameter(prototypes)
        elif update == "ema":
            self.register_buffer("prototypes", prototypes)
        else:
            raise ValueError("center update must be ema or learnable")
        self.dual_center_negative_centers = int(
            dual_center_negative_centers
        )
        if self.dual_center_negative_centers < 1:
            raise ValueError("dual center negative centers must be positive")
        if self.dual_center_enabled:
            if self.dual_center_negative_centers == 1:
                hard_negative_prototypes = torch.randn(
                    num_classes, hash_bits
                )
                normalize_dim = 1
            else:
                hard_negative_prototypes = torch.randn(
                    num_classes,
                    self.dual_center_negative_centers,
                    hash_bits,
                )
                normalize_dim = 2
            hard_negative_prototypes = F.normalize(
                hard_negative_prototypes, p=2, dim=normalize_dim
            )
            self.hard_negative_prototypes = nn.Parameter(
                hard_negative_prototypes
            )
        self.momentum = float(momentum)
        self.temperature = float(temperature)
        self.rgce_r = float(rgce_r)
        self.dual_center_hard_weight = float(dual_center_hard_weight)
        self.dual_center_separation_weight = float(
            dual_center_separation_weight
        )
        self.dual_center_margin = float(dual_center_margin)
        self.dual_center_warmup_epochs = int(dual_center_warmup_epochs)
        self.dual_center_top_k = int(dual_center_top_k)
        self.dual_center_reliability_enabled = bool(
            dual_center_reliability_enabled
        )
        self.dual_center_diversity_weight = float(
            dual_center_diversity_weight
        )
        self.dual_center_hash_quantization_weight = float(
            dual_center_hash_quantization_weight
        )

    @torch.no_grad()
    def update(self, image_hash, text_hash, labels, selected):
        if self.update_mode != "ema":
            return
        labels = labels.detach().float()
        selected = selected.detach().bool()
        image_hash = image_hash.detach()
        text_hash = text_hash.detach()
        for class_index in range(labels.shape[1]):
            mask = selected & (labels[:, class_index] > 0)
            if not mask.any():
                continue
            observed = torch.cat(
                [image_hash[mask], text_hash[mask]], dim=0
            ).mean(dim=0)
            updated = (
                self.momentum * self.prototypes[class_index]
                + (1.0 - self.momentum) * observed
            )
            self.prototypes[class_index] = F.normalize(
                updated.unsqueeze(0), p=2, dim=1
            )[0]

    def _robust_loss(self, confidence):
        confidence = confidence.clamp(1e-6, 1.0)
        r = self.rgce_r
        return (
            (1.0 - r) * (1.0 - confidence.pow(r)) / r
            + r * (1.0 - confidence)
        )

    @staticmethod
    def _class_similarity(hash_values, prototypes):
        hash_values = F.normalize(hash_values, p=2, dim=1)
        return hash_values @ prototypes.t()

    def _dual_center_loss(
        self,
        hash_values,
        labels,
        positive_similarity,
        hard_negative_prototypes,
        reliability=None,
    ):
        hash_values = F.normalize(hash_values, p=2, dim=1)
        if hard_negative_prototypes.ndim == 2:
            hard_similarity = hash_values @ hard_negative_prototypes.t()
        elif hard_negative_prototypes.ndim == 3:
            hard_subcenter_similarity = torch.einsum(
                "nb,cmb->ncm", hash_values, hard_negative_prototypes
            )
            hard_similarity = hard_subcenter_similarity.max(dim=2).values
        else:
            raise ValueError(
                "hard negative prototypes must be [C, B] or [C, M, B]"
            )
        labels = labels.float()
        positive_count = labels.sum(dim=1).clamp_min(1.0)
        positive_score = (
            positive_similarity * labels
        ).sum(dim=1, keepdim=True) / positive_count.unsqueeze(1)

        negative_mask = labels <= 0.0
        push_away = F.relu(
            positive_similarity
            - positive_score.detach()
            + self.dual_center_margin
        )
        push_away = push_away * negative_mask.float()
        if self.dual_center_top_k > 0:
            k = min(self.dual_center_top_k, push_away.shape[1])
            _, indices = torch.topk(push_away, k=k, dim=1)
            top_mask = torch.zeros_like(push_away).scatter(
                1, indices, 1.0
            )
            push_away = push_away * top_mask
        if self.dual_center_reliability_enabled and reliability is not None:
            reliability = reliability.to(
                device=push_away.device,
                dtype=push_away.dtype,
            ).view(-1, 1)
            push_away = push_away * reliability.clamp(0.0, 1.0)
        hard_weight = push_away.detach()
        attraction = hard_weight * (1.0 - hard_similarity)
        denominator = hard_weight.sum(dim=1).clamp_min(1e-6)
        hard_attraction = attraction.sum(dim=1) / denominator
        positive_repulsion = push_away.sum(dim=1) / (
            negative_mask.sum(dim=1).clamp_min(1)
        )
        return (
            self.dual_center_hard_weight * hard_attraction
            + self.dual_center_separation_weight * positive_repulsion
        )

    def _negative_center_regularization(self, hard_negative_prototypes):
        if hard_negative_prototypes.ndim != 3:
            return hard_negative_prototypes.new_zeros(())
        regularization = hard_negative_prototypes.new_zeros(())
        if self.dual_center_diversity_weight > 0:
            similarity = torch.einsum(
                "cmb,cnb->cmn",
                hard_negative_prototypes,
                hard_negative_prototypes,
            )
            center_count = similarity.shape[1]
            if center_count > 1:
                off_diagonal = ~torch.eye(
                    center_count,
                    dtype=torch.bool,
                    device=similarity.device,
                ).unsqueeze(0)
                diversity = similarity[off_diagonal.expand_as(similarity)]
                regularization = regularization + (
                    self.dual_center_diversity_weight
                    * diversity.pow(2).mean()
                )
        if self.dual_center_hash_quantization_weight > 0:
            target = hard_negative_prototypes.shape[-1] ** -0.5
            quantization = (
                hard_negative_prototypes.abs() - target
            ).pow(2).mean()
            regularization = regularization + (
                self.dual_center_hash_quantization_weight * quantization
            )
        return regularization

    def forward(
        self,
        image_hash,
        text_hash,
        labels,
        reliability=None,
        current_epoch=None,
    ):
        prototypes = F.normalize(self.prototypes, p=2, dim=1)
        image_similarity = self._class_similarity(image_hash, prototypes)
        text_similarity = self._class_similarity(text_hash, prototypes)
        image_probability = F.softmax(
            image_similarity / self.temperature,
            dim=1,
        )
        text_probability = F.softmax(
            text_similarity / self.temperature,
            dim=1,
        )
        labels = labels.float()
        positive_count = labels.sum(dim=1).clamp_min(1.0)
        image_confidence = (
            labels * image_probability
        ).sum(dim=1) / positive_count
        text_confidence = (
            labels * text_probability
        ).sum(dim=1) / positive_count
        per_sample = (
            self._robust_loss(image_confidence)
            + self._robust_loss(text_confidence)
        ) / 2.0
        dual_center_active = self.dual_center_enabled and (
            current_epoch is None
            or current_epoch >= self.dual_center_warmup_epochs
        )
        if dual_center_active:
            normalize_dim = 1 if self.hard_negative_prototypes.ndim == 2 else 2
            hard_negative_prototypes = F.normalize(
                self.hard_negative_prototypes, p=2, dim=normalize_dim
            )
            dual_center = (
                self._dual_center_loss(
                    image_hash,
                    labels,
                    image_similarity,
                    hard_negative_prototypes,
                    reliability,
                )
                + self._dual_center_loss(
                    text_hash,
                    labels,
                    text_similarity,
                    hard_negative_prototypes,
                    reliability,
                )
            ) / 2.0
            dual_center = dual_center + self._negative_center_regularization(
                hard_negative_prototypes
            )
            per_sample = per_sample + dual_center
        return LossOutput(per_sample.mean(), per_sample)
