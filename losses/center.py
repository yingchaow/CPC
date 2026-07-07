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
        hard_negative_enabled=False,
        hard_negative_alpha=1.0,
        hard_negative_margin=0.2,
        hard_negative_label_threshold=0.0,
        dual_center_enabled=False,
        dual_center_hard_weight=0.1,
        dual_center_separation_weight=0.1,
        dual_center_margin=0.2,
        dual_center_warmup_epochs=0,
        dual_center_top_k=0,
        dual_center_reliability_enabled=False,
        dual_center_positive_pull_weight=0.0,
        dual_center_positive_centers=1,
        dual_center_positive_diversity_weight=0.0,
        dual_center_negative_centers=1,
        dual_center_diversity_weight=0.0,
        dual_center_prototype_separation_weight=0.0,
        dual_center_label_graph_weight=0.0,
        dual_center_label_graph_top_k=0,
        dual_center_hash_quantization_weight=0.0,
        semantic_multi_center_enabled=False,
        semantic_multi_center_centers_per_class=1,
        semantic_multi_center_positive_weight=1.0,
        semantic_multi_center_negative_weight=0.0,
        semantic_multi_center_negative_margin=0.2,
        semantic_multi_center_negative_top_k=0,
        semantic_multi_center_intra_weight=0.0,
        semantic_multi_center_intra_target_similarity=0.4,
        semantic_multi_center_label_graph_weight=0.0,
        semantic_multi_center_label_graph_top_k=0,
    ):
        super().__init__()
        self.update_mode = update
        self.dual_center_enabled = bool(dual_center_enabled)
        self.semantic_multi_center_enabled = bool(
            semantic_multi_center_enabled
        )
        if self.dual_center_enabled and update != "learnable":
            raise ValueError("dual center requires learnable update")
        if dual_center_positive_centers > 1 and update != "learnable":
            raise ValueError("positive multi-center requires learnable update")
        if self.semantic_multi_center_enabled and update != "learnable":
            raise ValueError("semantic multi-center requires learnable update")
        if self.dual_center_enabled and self.semantic_multi_center_enabled:
            raise ValueError(
                "semantic multi-center cannot be combined with dual center"
            )
        self.semantic_multi_center_centers_per_class = int(
            semantic_multi_center_centers_per_class
        )
        self.dual_center_positive_centers = int(
            dual_center_positive_centers
        )
        if self.semantic_multi_center_centers_per_class < 1:
            raise ValueError(
                "semantic multi-center centers per class must be positive"
            )
        if self.dual_center_positive_centers < 1:
            raise ValueError("dual center positive centers must be positive")
        if self.semantic_multi_center_enabled:
            prototypes = F.normalize(
                torch.randn(
                    num_classes,
                    self.semantic_multi_center_centers_per_class,
                    hash_bits,
                ),
                p=2,
                dim=2,
            )
        elif self.dual_center_positive_centers > 1:
            prototypes = F.normalize(
                torch.randn(
                    num_classes,
                    self.dual_center_positive_centers,
                    hash_bits,
                ),
                p=2,
                dim=2,
            )
        else:
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
        self.hard_negative_enabled = bool(hard_negative_enabled)
        self.hard_negative_alpha = float(hard_negative_alpha)
        self.hard_negative_margin = float(hard_negative_margin)
        self.hard_negative_label_threshold = float(
            hard_negative_label_threshold
        )
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
        self.dual_center_positive_pull_weight = float(
            dual_center_positive_pull_weight
        )
        self.dual_center_positive_diversity_weight = float(
            dual_center_positive_diversity_weight
        )
        self.dual_center_diversity_weight = float(
            dual_center_diversity_weight
        )
        self.dual_center_prototype_separation_weight = float(
            dual_center_prototype_separation_weight
        )
        self.dual_center_label_graph_weight = float(
            dual_center_label_graph_weight
        )
        self.dual_center_label_graph_top_k = int(
            dual_center_label_graph_top_k
        )
        self.dual_center_hash_quantization_weight = float(
            dual_center_hash_quantization_weight
        )
        self.semantic_multi_center_positive_weight = float(
            semantic_multi_center_positive_weight
        )
        self.semantic_multi_center_negative_weight = float(
            semantic_multi_center_negative_weight
        )
        self.semantic_multi_center_negative_margin = float(
            semantic_multi_center_negative_margin
        )
        self.semantic_multi_center_negative_top_k = int(
            semantic_multi_center_negative_top_k
        )
        self.semantic_multi_center_intra_weight = float(
            semantic_multi_center_intra_weight
        )
        self.semantic_multi_center_intra_target_similarity = float(
            semantic_multi_center_intra_target_similarity
        )
        self.semantic_multi_center_label_graph_weight = float(
            semantic_multi_center_label_graph_weight
        )
        self.semantic_multi_center_label_graph_top_k = int(
            semantic_multi_center_label_graph_top_k
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
    def _subcenter_similarity(hash_values, prototypes):
        hash_values = F.normalize(hash_values, p=2, dim=1)
        if prototypes.ndim == 2:
            return (hash_values @ prototypes.t()).unsqueeze(2)
        if prototypes.ndim == 3:
            return torch.einsum(
                "nb,cmb->ncm", hash_values, prototypes
            )
        raise ValueError("prototypes must be [C, B] or [C, M, B]")

    def _class_similarity(self, hash_values, prototypes):
        return self._subcenter_similarity(hash_values, prototypes).max(
            dim=2
        ).values

    @staticmethod
    def _positive_pull_loss(class_similarity, labels):
        labels = labels.float()
        positive_count = labels.sum(dim=1).clamp_min(1.0)
        return (
            labels * (1.0 - class_similarity)
        ).sum(dim=1) / positive_count

    def _hard_negative_loss(self, positive_similarity, labels):
        labels = labels.float()
        positive_count = labels.sum(dim=1).clamp_min(1.0)
        positive_score = (
            positive_similarity * labels
        ).sum(dim=1, keepdim=True) / positive_count.unsqueeze(1)
        negative_mask = labels <= self.hard_negative_label_threshold
        penalty = F.relu(
            positive_similarity - positive_score + self.hard_negative_margin
        )
        penalty = penalty * negative_mask.float()
        negative_count = negative_mask.sum(dim=1).clamp_min(1)
        return penalty.sum(dim=1) / negative_count

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

        negative_mask = labels <= self.hard_negative_label_threshold
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

    @staticmethod
    def _as_subcenters(prototypes):
        if prototypes.ndim == 2:
            return prototypes.unsqueeze(1)
        if prototypes.ndim == 3:
            return prototypes
        raise ValueError("prototypes must be [C, B] or [C, M, B]")

    def _prototype_separation_regularization(
        self,
        positive_prototypes,
        hard_negative_prototypes,
    ):
        if self.dual_center_prototype_separation_weight <= 0:
            return positive_prototypes.new_zeros(())
        positive_prototypes = self._as_subcenters(positive_prototypes)
        hard_negative_prototypes = self._as_subcenters(
            hard_negative_prototypes
        )
        similarity = torch.einsum(
            "cpb,cmb->cpm",
            positive_prototypes,
            hard_negative_prototypes,
        )
        separation = F.relu(similarity - self.dual_center_margin).mean()
        return self.dual_center_prototype_separation_weight * separation

    def _label_graph_regularization(self, prototypes, labels):
        if self.dual_center_label_graph_weight <= 0:
            return prototypes.new_zeros(())
        prototypes = self._as_subcenters(prototypes)
        label_similarity = self._batch_label_similarity(labels)
        if self.dual_center_label_graph_top_k > 0:
            k = min(
                self.dual_center_label_graph_top_k,
                max(1, label_similarity.shape[1] - 1),
            )
            _, indices = torch.topk(label_similarity, k=k, dim=1)
            top_mask = torch.zeros_like(label_similarity).scatter(
                1, indices, 1.0
            )
            label_similarity = label_similarity * top_mask
        if label_similarity.sum() <= 0:
            return prototypes.new_zeros(())
        center_similarity = torch.einsum(
            "cpb,dmb->cdpm", prototypes, prototypes
        ).amax(dim=(2, 3))
        graph_pull = (
            label_similarity * (1.0 - center_similarity)
        ).sum() / label_similarity.sum().clamp_min(1e-6)
        return self.dual_center_label_graph_weight * graph_pull

    def _positive_center_regularization(self, prototypes):
        if prototypes.ndim != 3:
            return prototypes.new_zeros(())
        if self.dual_center_positive_diversity_weight <= 0:
            return prototypes.new_zeros(())
        similarity = torch.einsum("cmb,cnb->cmn", prototypes, prototypes)
        center_count = similarity.shape[1]
        if center_count <= 1:
            return prototypes.new_zeros(())
        off_diagonal = ~torch.eye(
            center_count,
            dtype=torch.bool,
            device=similarity.device,
        ).unsqueeze(0)
        diversity = similarity[off_diagonal.expand_as(similarity)]
        return (
            self.dual_center_positive_diversity_weight
            * diversity.pow(2).mean()
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

    def _semantic_class_similarity(self, hash_values, prototypes):
        hash_values = F.normalize(hash_values, p=2, dim=1)
        prototypes = F.normalize(prototypes, p=2, dim=2)
        subcenter_similarity = torch.einsum(
            "nb,cmb->ncm", hash_values, prototypes
        )
        return subcenter_similarity.max(dim=2).values

    def _semantic_sample_loss(self, hash_values, labels, prototypes):
        class_similarity = self._semantic_class_similarity(
            hash_values, prototypes
        )
        probability = F.softmax(
            class_similarity / self.temperature,
            dim=1,
        )
        labels = labels.float()
        positive_count = labels.sum(dim=1).clamp_min(1.0)
        confidence = (
            labels * probability
        ).sum(dim=1) / positive_count
        robust = self._robust_loss(confidence)
        positive_pull = (
            labels * (1.0 - class_similarity)
        ).sum(dim=1) / positive_count
        positive_score = (
            class_similarity * labels
        ).sum(dim=1, keepdim=True) / positive_count.unsqueeze(1)
        negative_mask = labels <= self.hard_negative_label_threshold
        negative_push = F.relu(
            class_similarity
            - positive_score.detach()
            + self.semantic_multi_center_negative_margin
        )
        negative_push = negative_push * negative_mask.float()
        if self.semantic_multi_center_negative_top_k > 0:
            k = min(
                self.semantic_multi_center_negative_top_k,
                negative_push.shape[1],
            )
            _, indices = torch.topk(negative_push, k=k, dim=1)
            top_mask = torch.zeros_like(negative_push).scatter(
                1, indices, 1.0
            )
            negative_push = negative_push * top_mask
        negative_count = negative_mask.sum(dim=1).clamp_min(1)
        negative = negative_push.sum(dim=1) / negative_count
        return (
            robust
            + self.semantic_multi_center_positive_weight * positive_pull
            + self.semantic_multi_center_negative_weight * negative
        )

    def _semantic_center_regularization(self, labels):
        if not self.semantic_multi_center_enabled:
            return self.prototypes.new_zeros(())
        prototypes = F.normalize(self.prototypes, p=2, dim=2)
        regularization = prototypes.new_zeros(())
        center_count = prototypes.shape[1]
        if (
            self.semantic_multi_center_intra_weight > 0
            and center_count > 1
        ):
            intra_similarity = torch.einsum(
                "cmb,cnb->cmn", prototypes, prototypes
            )
            off_diagonal = ~torch.eye(
                center_count,
                dtype=torch.bool,
                device=prototypes.device,
            ).unsqueeze(0)
            intra = intra_similarity[off_diagonal.expand_as(intra_similarity)]
            target = self.semantic_multi_center_intra_target_similarity
            regularization = regularization + (
                self.semantic_multi_center_intra_weight
                * (intra - target).pow(2).mean()
            )
        if self.semantic_multi_center_label_graph_weight > 0:
            label_similarity = self._batch_label_similarity(labels)
            if self.semantic_multi_center_label_graph_top_k > 0:
                k = min(
                    self.semantic_multi_center_label_graph_top_k,
                    max(1, label_similarity.shape[1] - 1),
                )
                _, indices = torch.topk(label_similarity, k=k, dim=1)
                top_mask = torch.zeros_like(label_similarity).scatter(
                    1, indices, 1.0
                )
                label_similarity = label_similarity * top_mask
            if label_similarity.sum() > 0:
                center_similarity = torch.einsum(
                    "cmb,dnb->cdmn", prototypes, prototypes
                ).amax(dim=(2, 3))
                graph_pull = (
                    label_similarity * (1.0 - center_similarity)
                ).sum() / label_similarity.sum().clamp_min(1e-6)
                regularization = regularization + (
                    self.semantic_multi_center_label_graph_weight
                    * graph_pull
                )
        return regularization

    def _batch_label_similarity(self, labels):
        labels = labels.float().to(self.prototypes.device)
        cooccurrence = labels.t() @ labels
        frequency = labels.sum(dim=0, keepdim=True)
        union = frequency.t() + frequency - cooccurrence
        similarity = cooccurrence / union.clamp_min(1e-6)
        similarity.fill_diagonal_(0.0)
        return similarity.clamp(0.0, 1.0)

    def forward(
        self,
        image_hash,
        text_hash,
        labels,
        reliability=None,
        current_epoch=None,
    ):
        if self.semantic_multi_center_enabled:
            prototypes = F.normalize(self.prototypes, p=2, dim=2)
            per_sample = (
                self._semantic_sample_loss(
                    image_hash,
                    labels,
                    prototypes,
                )
                + self._semantic_sample_loss(
                    text_hash,
                    labels,
                    prototypes,
                )
            ) / 2.0
            per_sample = per_sample + self._semantic_center_regularization(
                labels
            )
            return LossOutput(per_sample.mean(), per_sample)
        normalize_dim = 1 if self.prototypes.ndim == 2 else 2
        prototypes = F.normalize(self.prototypes, p=2, dim=normalize_dim)
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
        if self.dual_center_positive_pull_weight > 0:
            positive_pull = (
                self._positive_pull_loss(image_similarity, labels)
                + self._positive_pull_loss(text_similarity, labels)
            ) / 2.0
            per_sample = (
                per_sample
                + self.dual_center_positive_pull_weight * positive_pull
            )
        per_sample = per_sample + self._positive_center_regularization(
            prototypes
        )
        per_sample = per_sample + self._label_graph_regularization(
            prototypes, labels
        )
        if self.hard_negative_enabled:
            hard_negative = (
                self._hard_negative_loss(image_similarity, labels)
                + self._hard_negative_loss(text_similarity, labels)
            ) / 2.0
            per_sample = (
                per_sample
                + self.hard_negative_alpha * hard_negative
            )
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
            dual_center = (
                dual_center
                + self._prototype_separation_regularization(
                    prototypes,
                    hard_negative_prototypes,
                )
            )
            per_sample = per_sample + dual_center
        return LossOutput(per_sample.mean(), per_sample)
