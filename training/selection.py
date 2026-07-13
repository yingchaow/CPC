import torch
import torch.nn.functional as F

from classic_hashing.models.encoders import unpack_semantic_features


@torch.no_grad()
def knn_classification_supervision(
    image_features,
    text_features,
    labels,
    k=20,
    gamma=0.5,
    chunk_size=1024,
    device=None,
):
    compute_device = torch.device(device or image_features.device)
    image_features = F.normalize(
        image_features.float().to(compute_device), dim=1
    )
    text_features = F.normalize(
        text_features.float().to(compute_device), dim=1
    )
    labels = labels.float().to(compute_device)
    sample_count = len(labels)
    if sample_count < 2:
        return torch.ones(sample_count), labels.cpu()
    k = min(int(k), sample_count - 1)
    weights = torch.empty(sample_count, device=compute_device)
    neighbor_targets = torch.empty_like(labels)
    for start in range(0, sample_count, chunk_size):
        end = min(start + chunk_size, sample_count)
        similarity = (
            image_features[start:end] @ image_features.t()
            + text_features[start:end] @ text_features.t()
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
        neighbor_targets[start:end] = aggregate
    return (
        weights.clamp(gamma, 1.0).cpu(),
        neighbor_targets.clamp(0.0, 1.0).cpu(),
    )


def conservative_reliability(raw_weights, previous_weights, rise_momentum):
    raw_weights = raw_weights.detach().float().cpu()
    if previous_weights is None:
        return raw_weights
    previous_weights = previous_weights.detach().float().cpu()
    if previous_weights.shape != raw_weights.shape:
        raise ValueError(
            "previous reliability shape must match current weights"
        )
    smoothed_rise = (
        rise_momentum * previous_weights
        + (1.0 - rise_momentum) * raw_weights
    )
    return torch.where(
        raw_weights < previous_weights,
        raw_weights,
        smoothed_rise,
    )


@torch.no_grad()
def collect_train_representations(model, loader, device, sample_count):
    model.eval()
    image_features = None
    text_features = None
    restored_labels = None
    seen = torch.zeros(sample_count, dtype=torch.bool)
    for image, text, labels, index in loader:
        batch_image_features, batch_text_features = unpack_semantic_features(
            model(image.to(device), text.to(device))
        )
        batch_image_features = batch_image_features.detach().float().cpu()
        batch_text_features = batch_text_features.detach().float().cpu()
        labels = labels.detach().float().cpu()
        if image_features is None:
            image_features = torch.empty(
                sample_count,
                batch_image_features.shape[1],
                dtype=batch_image_features.dtype,
            )
            text_features = torch.empty(
                sample_count,
                batch_text_features.shape[1],
                dtype=batch_text_features.dtype,
            )
            restored_labels = torch.empty(
                sample_count, labels.shape[1], dtype=labels.dtype
            )
        index = index.long()
        image_features[index] = batch_image_features
        text_features[index] = batch_text_features
        restored_labels[index] = labels
        seen[index] = True
    if not seen.all():
        raise RuntimeError(
            "representation collection did not visit every sample"
        )
    return image_features, text_features, restored_labels
