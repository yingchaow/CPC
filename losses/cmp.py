import torch
import torch.nn.functional as F

from .pairwise import LossOutput


def cmp_margin_loss(image_hash, text_hash, margin=0.3):
    batch_size = image_hash.shape[0]
    if batch_size != text_hash.shape[0]:
        raise ValueError("image/text batch sizes must match")
    if batch_size <= 1:
        zero = image_hash.new_zeros(())
        return LossOutput(zero, image_hash.new_zeros(batch_size))

    similarity = image_hash @ text_hash.t()
    positive = similarity.diag()
    negative_mask = ~torch.eye(
        batch_size,
        device=similarity.device,
        dtype=torch.bool,
    )
    image_to_text = F.relu(
        similarity - positive.unsqueeze(1) + margin
    ).masked_fill(~negative_mask, 0.0)
    text_to_image = F.relu(
        similarity - positive.unsqueeze(0) + margin
    ).masked_fill(~negative_mask, 0.0)
    denominator = batch_size - 1
    per_sample = (
        image_to_text.sum(dim=1) / denominator
        + text_to_image.sum(dim=0) / denominator
    ) / 2.0
    return LossOutput(per_sample.mean(), per_sample)
