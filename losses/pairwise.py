from dataclasses import dataclass

import torch


@dataclass
class LossOutput:
    mean: torch.Tensor
    per_sample: torch.Tensor


def jaccard_similarity(labels):
    labels = labels.float()
    intersection = labels @ labels.t()
    cardinality = labels.sum(dim=1, keepdim=True)
    union = cardinality + cardinality.t() - intersection
    return intersection / union.clamp_min(1e-8)


def jaccard_contrast_loss(
    image_hash,
    text_hash,
    labels,
    margin=0.15,
    shift=0.8,
    temperature=1.0,
):
    target = jaccard_similarity(labels)
    pair_weight = labels.new_ones((labels.shape[0], labels.shape[0]))
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
    text_pair_loss = temperature * torch.exp(
        text_cost / temperature * negative_weight
    )
    image_pair_loss = temperature * torch.exp(
        image_cost / temperature * negative_weight
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


def pairwise_logistic_loss(
    image_hash,
    text_hash,
    labels,
    mode="jaccard_contrast",
    margin=0.15,
    shift=0.8,
    temperature=1.0,
):
    if mode != "jaccard_contrast":
        raise ValueError("pairwise loss only supports jaccard_contrast")
    return jaccard_contrast_loss(
        image_hash,
        text_hash,
        labels,
        margin=margin,
        shift=shift,
        temperature=temperature,
    )
