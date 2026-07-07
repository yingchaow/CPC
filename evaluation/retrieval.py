import numpy as np
import torch

from classic_hashing.models.encoders import unpack_model_outputs


def hamming_distance(query_codes, database_codes):
    bits = query_codes.shape[1]
    return 0.5 * (
        bits - query_codes.float() @ database_codes.float().t()
    )


def relevance(query_labels, database_labels):
    return query_labels.float() @ database_labels.float().t() > 0


def mean_average_precision(
    query_codes, database_codes, query_labels, database_labels
):
    average_precision = []
    database_codes = database_codes.float()
    database_labels = database_labels.float()
    for query_code, query_label in zip(query_codes, query_labels):
        target = relevance(
            query_label.unsqueeze(0), database_labels
        )[0]
        positive_count = int(target.sum().item())
        if positive_count == 0:
            continue
        distance = hamming_distance(
            query_code.unsqueeze(0), database_codes
        )[0]
        ranked = target[torch.argsort(distance)].float()
        precision = ranked.cumsum(0) / torch.arange(
            1, len(ranked) + 1, dtype=torch.float32
        )
        average_precision.append(
            float((precision * ranked).sum().item() / positive_count)
        )
    return float(np.mean(average_precision)) if average_precision else 0.0


def precision_recall_curve(
    query_codes,
    database_codes,
    query_labels,
    database_labels,
    recall_grid,
):
    recall_grid = np.asarray(recall_grid)
    query_precision = []
    database_codes = database_codes.float()
    database_labels = database_labels.float()
    for query_code, query_label in zip(query_codes, query_labels):
        target = relevance(
            query_label.unsqueeze(0), database_labels
        )[0]
        positives = int(target.sum().item())
        if positives == 0:
            continue
        distance = hamming_distance(
            query_code.unsqueeze(0), database_codes
        )[0]
        ranked = target[torch.argsort(distance)].float()
        cumulative = ranked.cumsum(0)
        precision = (
            cumulative
            / torch.arange(1, len(ranked) + 1, dtype=torch.float32)
        ).numpy()
        recall = (cumulative / positives).numpy()
        query_precision.append(
            np.array(
                [
                    precision[recall >= point].max()
                    if np.any(recall >= point)
                    else 0.0
                    for point in recall_grid
                ]
            )
        )
    if not query_precision:
        return np.zeros_like(recall_grid, dtype=np.float32)
    return np.mean(query_precision, axis=0)


@torch.no_grad()
def extract_codes(model, loader, device):
    model.eval()
    image_codes = [None] * len(loader.dataset)
    text_codes = [None] * len(loader.dataset)
    labels = [None] * len(loader.dataset)
    for image, text, batch_labels, index in loader:
        image_hash, text_hash, _, _ = unpack_model_outputs(
            model(image.to(device), text.to(device))
        )
        image_binary = torch.where(
            image_hash >= 0,
            torch.ones_like(image_hash),
            -torch.ones_like(image_hash),
        ).cpu()
        text_binary = torch.where(
            text_hash >= 0,
            torch.ones_like(text_hash),
            -torch.ones_like(text_hash),
        ).cpu()
        for offset, sample_index in enumerate(index.tolist()):
            image_codes[sample_index] = image_binary[offset]
            text_codes[sample_index] = text_binary[offset]
            labels[sample_index] = batch_labels[offset]
    if any(value is None for value in image_codes + text_codes + labels):
        raise RuntimeError("evaluation loader missed dataset samples")
    return (
        torch.stack(image_codes),
        torch.stack(text_codes),
        torch.stack(labels),
    )
