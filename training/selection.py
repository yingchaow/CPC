import numpy as np
import torch
import torch.nn.functional as F

from classic_hashing.models.encoders import unpack_model_outputs


def remember_rate(epoch, total_epochs, noise_rate, warmup_epochs):
    if epoch < warmup_epochs:
        return 1.0
    forget = noise_rate * (epoch + 1) / total_epochs
    return max(0.0, 1.0 - min(noise_rate, forget))


def build_selected_mask(losses, remember):
    losses = losses.detach().float().cpu()
    count = max(1, int(remember * len(losses)))
    selected_indices = torch.argsort(losses)[:count]
    mask = torch.zeros(len(losses), dtype=torch.bool)
    mask[selected_indices] = True
    return mask


def selection_metrics(selected_mask, noise_mask):
    selected = np.asarray(selected_mask, dtype=bool)
    clean = ~np.asarray(noise_mask, dtype=bool)
    true_positive = np.logical_and(selected, clean).sum()
    return {
        "precision": float(true_positive / max(1, selected.sum())),
        "recall": float(true_positive / max(1, clean.sum())),
        "coverage": float(selected.mean()),
    }


@torch.no_grad()
def _modality_knn_soft_labels(
    features,
    labels,
    k=20,
    chunk_size=1024,
    device=None,
):
    compute_device = torch.device(device or features.device)
    features = F.normalize(features.float().to(compute_device), dim=1)
    labels = labels.float().to(compute_device)
    sample_count = len(labels)
    if sample_count < 2:
        return labels.cpu()
    k = min(int(k), sample_count - 1)
    soft_targets = torch.empty_like(labels)
    for start in range(0, sample_count, chunk_size):
        end = min(start + chunk_size, sample_count)
        similarity = features[start:end] @ features.t()
        row = torch.arange(end - start, device=compute_device)
        column = torch.arange(start, end, device=compute_device)
        similarity[row, column] = -torch.inf
        _, neighbor_index = torch.topk(similarity, k=k, dim=1)
        neighbor_labels = labels[neighbor_index]
        soft_targets[start:end] = neighbor_labels.float().mean(dim=1)
    return soft_targets.clamp(0.0, 1.0).cpu()


@torch.no_grad()
def neighbor_refining_from_soft_labels(
    labels,
    image_soft_labels,
    text_soft_labels,
    pure_weight=1.0,
    hard_weight=0.5,
    noisy_weight=0.2,
):
    labels = labels.float().cpu()
    image_soft_labels = image_soft_labels.float().cpu().clamp(0.0, 1.0)
    text_soft_labels = text_soft_labels.float().cpu().clamp(0.0, 1.0)
    image_top = image_soft_labels.argmax(dim=1, keepdim=True)
    text_top = text_soft_labels.argmax(dim=1, keepdim=True)
    image_consistent = labels.gather(1, image_top).squeeze(1) > 0
    text_consistent = labels.gather(1, text_top).squeeze(1) > 0
    pure_mask = image_consistent & text_consistent
    hard_mask = image_consistent ^ text_consistent
    noisy_mask = ~(image_consistent | text_consistent)
    weights = torch.empty(len(labels), dtype=torch.float32)
    weights[pure_mask] = float(pure_weight)
    weights[hard_mask] = float(hard_weight)
    weights[noisy_mask] = float(noisy_weight)
    fused_soft = 1.0 - (
        1.0 - image_soft_labels
    ) * (1.0 - text_soft_labels)
    targets = labels.clone()
    targets[noisy_mask] = fused_soft[noisy_mask]
    stats = {
        "pure_mask": pure_mask.cpu(),
        "hard_mask": hard_mask.cpu(),
        "noisy_mask": noisy_mask.cpu(),
        "pure_count": int(pure_mask.sum().item()),
        "hard_count": int(hard_mask.sum().item()),
        "noisy_count": int(noisy_mask.sum().item()),
    }
    return weights.clamp(0.0, 1.0), targets.clamp(0.0, 1.0), stats


@torch.no_grad()
def neighbor_refining_supervision(
    image_hash,
    text_hash,
    labels,
    k=20,
    chunk_size=1024,
    pure_weight=1.0,
    hard_weight=0.5,
    noisy_weight=0.2,
    device=None,
):
    image_soft_labels = _modality_knn_soft_labels(
        image_hash,
        labels,
        k=k,
        chunk_size=chunk_size,
        device=device,
    )
    text_soft_labels = _modality_knn_soft_labels(
        text_hash,
        labels,
        k=k,
        chunk_size=chunk_size,
        device=device,
    )
    return neighbor_refining_from_soft_labels(
        labels,
        image_soft_labels,
        text_soft_labels,
        pure_weight=pure_weight,
        hard_weight=hard_weight,
        noisy_weight=noisy_weight,
    )


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
def nsh_guided_soft_label_correction(labels, knn_targets, nsh_weight):
    labels = labels.float().cpu()
    knn_targets = knn_targets.float().cpu().clamp(0.0, 1.0)
    nsh_weight = nsh_weight.float().cpu().clamp(0.0, 1.0).view(-1, 1)
    if labels.shape != knn_targets.shape:
        raise ValueError("labels and kNN targets must have the same shape")
    if nsh_weight.shape[0] != labels.shape[0]:
        raise ValueError("NSH weights must have one value per sample")
    return (
        nsh_weight * labels + (1.0 - nsh_weight) * knn_targets
    ).clamp(0.0, 1.0)


@torch.no_grad()
def knn_classification_weights(
    image_hash,
    text_hash,
    labels,
    k=20,
    gamma=0.5,
    chunk_size=1024,
    device=None,
):
    weights, _ = knn_classification_supervision(
        image_hash,
        text_hash,
        labels,
        k=k,
        gamma=gamma,
        chunk_size=chunk_size,
        device=device,
    )
    return weights


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


@torch.no_grad()
def collect_global_losses(model, criterion, loader, device, sample_count):
    model.eval()
    losses = torch.empty(sample_count, dtype=torch.float32)
    seen = torch.zeros(sample_count, dtype=torch.bool)
    for image, text, labels, index in loader:
        image_hash, text_hash, _, _ = unpack_model_outputs(
            model(image.to(device), text.to(device))
        )
        score = criterion.selection_score(
            image_hash, text_hash, labels.to(device)
        ).cpu()
        index = index.long()
        losses[index] = score
        seen[index] = True
    if not seen.all():
        raise RuntimeError("global selection did not visit every sample")
    return losses
