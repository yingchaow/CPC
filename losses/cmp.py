import torch
import torch.nn.functional as F

from .pairwise import LossOutput, reliability_pair_weights


def cmp_margin_loss(
    image_hash,
    text_hash,
    labels,
    margin=0.3,
    sample_weights=None,
):
    batch_size = image_hash.shape[0]
    if batch_size != text_hash.shape[0]:
        raise ValueError("image/text batch sizes must match")
    if labels.ndim != 2 or labels.shape[0] != batch_size:
        raise ValueError("labels must be [batch_size, num_classes]")
    if batch_size <= 1:
        zero = image_hash.new_zeros(())
        return LossOutput(zero, image_hash.new_zeros(batch_size))

    similarity = image_hash @ text_hash.t()
    positive = similarity.diag()
    semantic_positive = labels.float() @ labels.float().t() > 0
    diagonal = torch.eye(
        batch_size,
        device=similarity.device,
        dtype=torch.bool,
    )
    negative_mask = ~(semantic_positive | diagonal)
    pair_weight = reliability_pair_weights(labels, sample_weights)
    image_to_text = F.relu(
        similarity - positive.unsqueeze(1) + margin
    ) * pair_weight
    image_to_text = image_to_text.masked_fill(~negative_mask, 0.0)
    text_to_image = F.relu(
        similarity - positive.unsqueeze(0) + margin
    ) * pair_weight
    text_to_image = text_to_image.masked_fill(~negative_mask, 0.0)
    image_negative_count = negative_mask.sum(dim=1).clamp_min(1)
    text_negative_count = negative_mask.sum(dim=0).clamp_min(1)
    per_sample = (
        image_to_text.sum(dim=1) / image_negative_count
        + text_to_image.sum(dim=0) / text_negative_count
    ) / 2.0
    return LossOutput(per_sample.mean(), per_sample)
