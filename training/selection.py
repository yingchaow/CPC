import torch
import torch.nn.functional as F

from classic_hashing.models.encoders import unpack_model_outputs


@torch.no_grad()
def knn_classification_supervision(
    image_hash,
    text_hash,
    labels,
    k=20,
    gamma=0.5,
    chunk_size=1024,
    device=None,
):
    compute_device = torch.device(device or image_hash.device)
    image_hash = F.normalize(
        image_hash.float().to(compute_device), dim=1
    )
    text_hash = F.normalize(
        text_hash.float().to(compute_device), dim=1
    )
    labels = labels.float().to(compute_device)
    sample_count = len(labels)
    if sample_count < 2:
        return torch.ones(sample_count), labels.cpu()
    k = min(int(k), sample_count - 1)
    weights = torch.empty(sample_count, device=compute_device)
    soft_targets = torch.empty_like(labels)
    for start in range(0, sample_count, chunk_size):
        end = min(start + chunk_size, sample_count)
        similarity = (
            image_hash[start:end] @ image_hash.t()
            + text_hash[start:end] @ text_hash.t()
        ) / 2.0
        row = torch.arange(end - start, device=compute_device)
        column = torch.arange(start, end, device=compute_device)
        similarity[row, column] = -torch.inf
        top_similarity, neighbor_index = torch.topk(
            similarity, k=k, dim=1
        )
        top_similarity = top_similarity.clamp_min(0.0)
        normalization = top_similarity.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-8)
        neighbor_weight = top_similarity / normalization
        neighbor_labels = labels[neighbor_index]
        aggregate = (
            neighbor_labels * neighbor_weight.unsqueeze(-1)
        ).sum(dim=1)
        consistency = F.cosine_similarity(
            labels[start:end], aggregate, dim=1
        ).clamp(0.0, 1.0)
        batch_weights = gamma + (1.0 - gamma) * consistency
        weights[start:end] = batch_weights
        soft_targets[start:end] = (
            batch_weights.unsqueeze(1) * labels[start:end]
            + (1.0 - batch_weights).unsqueeze(1) * aggregate
        )
    return (
        weights.clamp(gamma, 1.0).cpu(),
        soft_targets.clamp(0.0, 1.0).cpu(),
    )


@torch.no_grad()
def collect_train_representations(model, loader, device, sample_count):
    model.eval()
    image_hashes = None
    text_hashes = None
    restored_labels = None
    seen = torch.zeros(sample_count, dtype=torch.bool)
    for image, text, labels, index in loader:
        image_hash, text_hash, _, _ = unpack_model_outputs(
            model(image.to(device), text.to(device))
        )
        image_hash = image_hash.detach().float().cpu()
        text_hash = text_hash.detach().float().cpu()
        labels = labels.detach().float().cpu()
        if image_hashes is None:
            image_hashes = torch.empty(
                sample_count, image_hash.shape[1], dtype=image_hash.dtype
            )
            text_hashes = torch.empty(
                sample_count, text_hash.shape[1], dtype=text_hash.dtype
            )
            restored_labels = torch.empty(
                sample_count, labels.shape[1], dtype=labels.dtype
            )
        index = index.long()
        image_hashes[index] = image_hash
        text_hashes[index] = text_hash
        restored_labels[index] = labels
        seen[index] = True
    if not seen.all():
        raise RuntimeError(
            "representation collection did not visit every sample"
        )
    return image_hashes, text_hashes, restored_labels
