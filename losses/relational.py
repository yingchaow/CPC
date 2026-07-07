import torch
import torch.nn.functional as F

from .pairwise import LossOutput


def fuzzy_jaccard_similarity(labels):
    labels = labels.float().clamp_min(0.0)
    left = labels.unsqueeze(1)
    right = labels.unsqueeze(0)
    intersection = torch.minimum(left, right).sum(dim=-1)
    union = torch.maximum(left, right).sum(dim=-1)
    return intersection / union.clamp_min(1e-8)


def relational_kd_loss(
    image_hash,
    text_hash,
    teacher_labels,
    mode="huber",
    huber_delta=0.2,
):
    teacher_relation = fuzzy_jaccard_similarity(
        teacher_labels.detach()
    )
    teacher_relation = teacher_relation * 2.0 - 1.0
    image_to_text = image_hash @ text_hash.t()
    text_to_image = text_hash @ image_hash.t()
    if mode == "huber":
        image_loss = F.huber_loss(
            image_to_text,
            teacher_relation,
            delta=huber_delta,
            reduction="none",
        )
        text_loss = F.huber_loss(
            text_to_image,
            teacher_relation.t(),
            delta=huber_delta,
            reduction="none",
        )
    elif mode == "mse":
        image_loss = F.mse_loss(
            image_to_text,
            teacher_relation,
            reduction="none",
        )
        text_loss = F.mse_loss(
            text_to_image,
            teacher_relation.t(),
            reduction="none",
        )
    else:
        raise ValueError(f"unsupported relational KD mode: {mode}")
    per_sample = (image_loss.mean(dim=1) + text_loss.mean(dim=1)) / 2.0
    return LossOutput(per_sample.mean(), per_sample)
