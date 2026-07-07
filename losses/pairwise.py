from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class LossOutput:
    mean: torch.Tensor
    per_sample: torch.Tensor


def jaccard_similarity(labels):
    return label_similarity(labels, mode="jaccard")


def soft_jaccard_similarity(y1, y2, eps=1e-8):
    y1 = y1.float().clamp(0.0, 1.0)
    y2 = y2.float().to(device=y1.device, dtype=y1.dtype).clamp(0.0, 1.0)
    intersection = torch.minimum(
        y1.unsqueeze(1),
        y2.unsqueeze(0),
    ).sum(dim=2)
    union = torch.maximum(
        y1.unsqueeze(1),
        y2.unsqueeze(0),
    ).sum(dim=2)
    return intersection / union.clamp_min(eps)


def soft_label_similarity(y1, y2, mode="jaccard", eps=1e-8):
    y1 = y1.float().clamp(0.0, 1.0)
    y2 = y2.float().to(device=y1.device, dtype=y1.dtype).clamp(0.0, 1.0)
    intersection = torch.minimum(
        y1.unsqueeze(1),
        y2.unsqueeze(0),
    ).sum(dim=2)
    left_cardinality = y1.sum(dim=1, keepdim=True)
    right_cardinality = y2.sum(dim=1, keepdim=True).t()
    if mode == "jaccard":
        union = torch.maximum(
            y1.unsqueeze(1),
            y2.unsqueeze(0),
        ).sum(dim=2)
        return intersection / union.clamp_min(eps)
    if mode == "dice":
        denominator = left_cardinality + right_cardinality
        return 2.0 * intersection / denominator.clamp_min(eps)
    if mode == "cosine":
        denominator = torch.linalg.vector_norm(
            y1, dim=1, keepdim=True
        ) @ torch.linalg.vector_norm(y2, dim=1, keepdim=True).t()
        return (y1 @ y2.t()) / denominator.clamp_min(eps)
    if mode == "overlap":
        denominator = torch.minimum(left_cardinality, right_cardinality)
        return intersection / denominator.clamp_min(eps)
    raise ValueError(f"unsupported soft label similarity mode: {mode}")


def _estimate_confidence(y_hat, top_k=3):
    y_hat = y_hat.float().clamp(0.0, 1.0)
    if y_hat.ndim != 2:
        raise ValueError("y_hat must be a [N, C] tensor")
    k = min(max(int(top_k), 1), y_hat.shape[1])
    return torch.topk(y_hat, k=k, dim=1).values.mean(dim=1)


def pair_supervision_weight(y_hat, confidence=None, confidence_top_k=3):
    y_hat = y_hat.float().clamp(0.0, 1.0)
    if confidence is None:
        confidence = _estimate_confidence(y_hat, confidence_top_k)
    confidence = confidence.to(
        device=y_hat.device,
        dtype=y_hat.dtype,
    ).view(-1).clamp(0.0, 1.0)
    if confidence.shape[0] != y_hat.shape[0]:
        raise ValueError("confidence must have one value per sample")
    return torch.outer(confidence, confidence)


def label_similarity(labels, mode="jaccard"):
    labels = labels.float()
    intersection = labels @ labels.t()
    cardinality = labels.sum(dim=1, keepdim=True)
    if mode == "jaccard":
        union = cardinality + cardinality.t() - intersection
        return intersection / union.clamp_min(1e-8)
    if mode == "dice":
        denominator = cardinality + cardinality.t()
        return 2.0 * intersection / denominator.clamp_min(1e-8)
    if mode == "cosine":
        denominator = torch.sqrt(cardinality @ cardinality.t())
        return intersection / denominator.clamp_min(1e-8)
    if mode == "overlap":
        denominator = torch.minimum(cardinality, cardinality.t())
        return intersection / denominator.clamp_min(1e-8)
    if mode == "binary":
        return (intersection > 0).float()
    raise ValueError(f"unsupported label similarity mode: {mode}")


def pair_target_similarity(
    labels,
    similarity="jaccard",
    similarity_type="hard_jaccard",
    labels_soft=None,
    eps=1e-8,
    blend_lambda=0.5,
    hard_similarity="jaccard",
    second_similarity="dice",
    soft_similarity="jaccard",
):
    if similarity_type == "hard_jaccard":
        return label_similarity(labels, mode=similarity)
    if similarity_type == "hard_blend":
        first_target = label_similarity(labels, mode=hard_similarity)
        second_target = label_similarity(labels, mode=second_similarity)
        return (
            blend_lambda * first_target
            + (1.0 - blend_lambda) * second_target
        )
    if similarity_type in ("soft_jaccard", "blend_jaccard", "blend_soft"):
        y_hat = labels if labels_soft is None else labels_soft
        y_hat = y_hat.to(device=labels.device).float().clamp(0.0, 1.0)
        soft_mode = (
            soft_similarity
            if similarity_type == "blend_soft"
            else "jaccard"
        )
        soft_target = soft_label_similarity(y_hat, y_hat, soft_mode, eps)
        if similarity_type == "soft_jaccard":
            return soft_target
        hard_mode = (
            hard_similarity
            if similarity_type == "blend_soft"
            else "jaccard"
        )
        hard_target = label_similarity(labels, mode=hard_mode)
        return (
            blend_lambda * hard_target
            + (1.0 - blend_lambda) * soft_target
        )
    raise ValueError(f"unsupported pairwise similarity type: {similarity_type}")


def _pair_loss_weight(
    labels,
    similarity_type="hard_jaccard",
    labels_soft=None,
    confidence=None,
    confidence_top_k=3,
):
    if similarity_type in ("hard_jaccard", "hard_blend"):
        return labels.new_ones((labels.shape[0], labels.shape[0]))
    y_hat = labels if labels_soft is None else labels_soft
    y_hat = y_hat.to(device=labels.device).float().clamp(0.0, 1.0)
    if confidence is not None:
        confidence = confidence.to(device=labels.device)
    return pair_supervision_weight(
        y_hat,
        confidence=confidence,
        confidence_top_k=confidence_top_k,
    )


def jaccard_contrast_loss(
    image_hash,
    text_hash,
    labels,
    margin=0.15,
    shift=0.8,
    temperature=1.0,
    hard_negative_enabled=False,
    hard_negative_alpha=1.0,
    hard_negative_margin=0.2,
    hard_negative_label_threshold=0.0,
    similarity="jaccard",
    similarity_type="hard_jaccard",
    labels_soft=None,
    confidence=None,
    confidence_top_k=3,
    blend_lambda=0.5,
    hard_similarity="jaccard",
    second_similarity="dice",
    soft_similarity="jaccard",
):
    target = pair_target_similarity(
        labels.float(),
        similarity=similarity,
        similarity_type=similarity_type,
        labels_soft=labels_soft,
        blend_lambda=blend_lambda,
        hard_similarity=hard_similarity,
        second_similarity=second_similarity,
        soft_similarity=soft_similarity,
    )
    pair_weight = _pair_loss_weight(
        labels.float(),
        similarity_type=similarity_type,
        labels_soft=labels_soft,
        confidence=confidence,
        confidence_top_k=confidence_top_k,
    )
    hash_similarity = image_hash @ text_hash.t()
    diagonal = hash_similarity.diag()
    text_threshold = diagonal.unsqueeze(1) - margin
    image_threshold = diagonal.unsqueeze(0) - margin
    text_cost = torch.where(
        hash_similarity >= text_threshold,
        hash_similarity,
        hash_similarity - shift,
    )
    image_cost = torch.where(
        hash_similarity >= image_threshold,
        hash_similarity,
        hash_similarity - shift,
    )
    eye = torch.eye(
        hash_similarity.shape[0],
        device=hash_similarity.device,
        dtype=hash_similarity.dtype,
    )
    text_cost = text_cost * (1.0 - eye) + torch.diag_embed(
        text_cost.diag().clamp_min(0)
    )
    image_cost = image_cost * (1.0 - eye) + torch.diag_embed(
        image_cost.diag().clamp_min(0)
    )
    negative_weight = 1.0 - target
    if hard_negative_enabled:
        eye_mask = torch.eye(
            hash_similarity.shape[0],
            device=hash_similarity.device,
            dtype=torch.bool,
        )
        negative_mask = (
            target <= hard_negative_label_threshold
        ) & ~eye_mask
        hard_negative_weight = 1.0 + hard_negative_alpha * F.relu(
            hash_similarity.detach() - hard_negative_margin
        )
        negative_weight = torch.where(
            negative_mask,
            negative_weight * hard_negative_weight,
            negative_weight,
        )
    text_pair_loss = (
        temperature
        * torch.exp(
            text_cost / temperature * negative_weight
        )
    )
    image_pair_loss = (
        temperature
        * torch.exp(
            image_cost / temperature * negative_weight
        )
    )
    text_term = (pair_weight * text_pair_loss).mean(dim=1)
    image_term = (pair_weight * image_pair_loss).mean(dim=0)
    positive_mask = ((target - eye) > 0).float()
    separation = (
        pair_weight * positive_mask * torch.exp(shift - hash_similarity)
    ).mean(dim=1)
    diagonal_weight = pair_weight.diag()
    per_sample = (
        text_term + image_term + separation - diagonal_weight * diagonal
    )
    return LossOutput(per_sample.mean(), per_sample)


def _binary_loss(logits, target, mode, gce_q, reverse_weight):
    if mode == "bce":
        return F.binary_cross_entropy_with_logits(
            logits, target, reduction="none"
        )
    probability = torch.sigmoid(logits).clamp(1e-6, 1.0 - 1e-6)
    target_probability = torch.where(
        target > 0.5, probability, 1.0 - probability
    )
    if mode == "gce":
        return (1.0 - target_probability.pow(gce_q)) / gce_q
    if mode == "symmetric_bce":
        forward = F.binary_cross_entropy_with_logits(
            logits, target, reduction="none"
        )
        clipped_target = target.clamp(1e-4, 1.0 - 1e-4)
        reverse = -(
            probability * clipped_target.log()
            + (1.0 - probability) * (1.0 - clipped_target).log()
        )
        return forward + reverse_weight * reverse
    raise ValueError(f"unsupported pairwise mode: {mode}")


def pairwise_logistic_loss(
    image_hash,
    text_hash,
    labels,
    mode="bce",
    gce_q=0.7,
    reverse_weight=0.1,
    margin=0.15,
    shift=0.8,
    temperature=1.0,
    hard_negative_enabled=False,
    hard_negative_alpha=1.0,
    hard_negative_margin=0.2,
    hard_negative_label_threshold=0.0,
    similarity="jaccard",
    similarity_type="hard_jaccard",
    labels_soft=None,
    confidence=None,
    confidence_top_k=3,
    blend_lambda=0.5,
    hard_similarity="jaccard",
    second_similarity="dice",
    soft_similarity="jaccard",
):
    if mode == "jaccard_contrast":
        return jaccard_contrast_loss(
            image_hash,
            text_hash,
            labels,
            margin=margin,
            shift=shift,
            temperature=temperature,
            hard_negative_enabled=hard_negative_enabled,
            hard_negative_alpha=hard_negative_alpha,
            hard_negative_margin=hard_negative_margin,
            hard_negative_label_threshold=(
                hard_negative_label_threshold
            ),
            similarity=similarity,
            similarity_type=similarity_type,
            labels_soft=labels_soft,
            confidence=confidence,
            confidence_top_k=confidence_top_k,
            blend_lambda=blend_lambda,
            hard_similarity=hard_similarity,
            second_similarity=second_similarity,
            soft_similarity=soft_similarity,
        )
    target = (labels.float() @ labels.float().t() > 0).float()
    logits = 0.5 * (image_hash @ text_hash.t())
    image_to_text = _binary_loss(
        logits, target, mode, gce_q, reverse_weight
    ).mean(dim=1)
    text_to_image = _binary_loss(
        logits.t(), target.t(), mode, gce_q, reverse_weight
    ).mean(dim=1)
    per_sample = (image_to_text + text_to_image) / 2.0
    return LossOutput(per_sample.mean(), per_sample)
