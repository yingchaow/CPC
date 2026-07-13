from collections import defaultdict

import torch

from classic_hashing.models.encoders import unpack_model_outputs


def train_epoch(
    student,
    criterion,
    loader,
    optimizer,
    selected_mask,
    config,
    device,
    classification_weights=None,
    classification_targets=None,
    epoch=None,
):
    student.train()
    criterion.train()
    totals = defaultdict(float)
    batches = 0
    selected_mask = selected_mask.bool().cpu()
    if classification_weights is not None:
        classification_weights = classification_weights.float().cpu()
    if classification_targets is not None:
        classification_targets = classification_targets.float().cpu()
    for image, text, labels, index in loader:
        image = image.to(device)
        text = text.to(device)
        labels = labels.to(device)
        index = index.long()
        selected = selected_mask[index].to(device)
        batch_classification_weights = (
            classification_weights[index].to(device)
            if classification_weights is not None
            else None
        )
        batch_classification_targets = (
            classification_targets[index].to(device)
            if classification_targets is not None
            else None
        )
        (
            image_hash,
            text_hash,
            image_logits,
            text_logits,
        ) = unpack_model_outputs(student(image, text))
        output = criterion(
            image_hash,
            text_hash,
            labels,
            selected,
            image_logits=image_logits,
            text_logits=text_logits,
            classification_weights=batch_classification_weights,
            classification_targets=batch_classification_targets,
            current_epoch=epoch,
        )
        optimizer.zero_grad(set_to_none=True)
        output.total.backward()
        optimizer.step()
        if (
            config.loss.center.enabled
            and config.loss.center.update == "ema"
        ):
            criterion.center_loss.update(
                image_hash, text_hash, labels, selected
            )
        totals["total"] += float(output.total.detach().item())
        for name, value in output.components.items():
            totals[name] += float(value.detach().item())
        batches += 1
    if batches == 0:
        raise RuntimeError("training loader produced no batches")
    return {name: value / batches for name, value in totals.items()}


def format_epoch_log(epoch, epochs, metrics, selection=None):
    fields = [f"Epoch {epoch + 1:03d}/{epochs:03d}"]
    for name in (
        "total",
        "pairwise",
        "center",
        "quantization",
        "classification",
        "cmp",
        "raw_class_weight_mean",
        "class_weight_mean",
        "nr_pure",
        "nr_hard",
        "nr_noisy",
    ):
        value = metrics.get(name)
        if value is not None and (name == "total" or value != 0.0):
            fields.append(f"{name}={value:.6f}")
    if selection is not None:
        fields.extend(
            [
                f"sel_P={selection['precision']:.4f}",
                f"sel_R={selection['recall']:.4f}",
                f"sel_C={selection['coverage']:.4f}",
            ]
        )
    return " | ".join(fields)


def checkpoint_state(
    epoch,
    student,
    criterion,
    optimizer,
    best_metrics,
    config,
):
    return {
        "epoch": epoch,
        "student": student.state_dict(),
        "criterion": criterion.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_metrics": dict(best_metrics),
        "config": dict(config),
        "data_h5": config.dataset.h5_path,
        "noise_h5": config.dataset.noise_path,
        "experiment_identity": {
            "dataset": config.dataset.name,
            "method": config.experiment.name,
            "pairwise_mode": config.loss.pairwise.mode,
            "classification": config.loss.classification.enabled,
            "noise_rate": config.dataset.noise_rate,
            "hash_bits": config.model.hash_bits,
            "seed": config.experiment.seed,
        },
    }


def restore_checkpoint(path, student, criterion, optimizer=None, device="cpu"):
    state = torch.load(path, map_location=device, weights_only=False)
    student.load_state_dict(state["student"])
    criterion.load_state_dict(state["criterion"])
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    return state
