import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from classic_hashing.config import (
    compose_config,
    load_config,
    validate_config,
)
from classic_hashing.data.dataset import create_datasets
from classic_hashing.evaluation.plot_pr_curve import save_pr_outputs
from classic_hashing.evaluation.retrieval import (
    extract_codes,
    mean_average_precision,
    precision_recall_curve,
)
from classic_hashing.losses.composite import CompositeHashLoss
from classic_hashing.models.encoders import DualHashModel
from classic_hashing.training.selection import (
    collect_train_representations,
    knn_classification_supervision,
)
from classic_hashing.training.trainer import (
    checkpoint_state,
    format_epoch_log,
    restore_checkpoint,
    train_epoch,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_loaders(config):
    expected = {
        "dataset": config.dataset.name,
        "image_dim": config.dataset.image_dim,
        "text_dim": config.dataset.text_dim,
        "num_classes": config.dataset.num_classes,
        "query_size": config.dataset.query_size,
        "train_size": config.dataset.train_size,
        "database_start": config.dataset.database_start,
    }
    train, query, database = create_datasets(
        config.dataset.h5_path,
        config.dataset.noise_path,
        expected=expected,
    )
    generator = torch.Generator().manual_seed(config.experiment.seed)
    common = {
        "batch_size": config.train.batch_size,
        "num_workers": config.train.num_workers,
        "worker_init_fn": _seed_worker,
        "drop_last": False,
    }
    train_loader = DataLoader(
        train, shuffle=True, generator=generator, **common
    )
    selection_loader = DataLoader(train, shuffle=False, **common)
    query_loader = DataLoader(query, shuffle=False, **common)
    database_loader = DataLoader(database, shuffle=False, **common)
    return (
        train,
        train_loader,
        selection_loader,
        query_loader,
        database_loader,
    )


def build_components(config, device):
    student = DualHashModel(
        config.dataset.image_dim,
        config.dataset.text_dim,
        config.model.hash_bits,
        config.model.image_hidden_dims,
        config.model.text_hidden_dims,
        config.model.l2_normalize,
        num_classes=config.dataset.num_classes,
        classification_enabled=config.loss.classification.enabled,
    ).to(device)
    criterion = CompositeHashLoss(
        config, config.dataset.num_classes, config.model.hash_bits
    ).to(device)
    optimizer = torch.optim.Adam(
        list(student.parameters()) + list(criterion.parameters()),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    return student, criterion, optimizer


def build_epoch_classification_supervision_with_metrics(
    config,
    epoch,
    student,
    selection_loader,
    device,
    sample_count,
):
    knn = config.robust_training.knn_classification_weight
    if not knn.enabled:
        return None, None, None
    if epoch < knn.warmup_epochs:
        return torch.ones(sample_count, dtype=torch.float32), None, None
    image_hash, text_hash, labels = collect_train_representations(
        student,
        selection_loader,
        device,
        sample_count,
    )
    weights, soft_targets = knn_classification_supervision(
        image_hash,
        text_hash,
        labels,
        k=knn.k,
        gamma=knn.gamma,
        chunk_size=knn.chunk_size,
        device=device,
    )
    return (
        weights,
        soft_targets if knn.soft_label_enabled else None,
        None,
    )


def evaluate_retrieval(student, query_loader, database_loader, device):
    query_image, query_text, query_labels = extract_codes(
        student, query_loader, device
    )
    database_image, database_text, database_labels = extract_codes(
        student, database_loader, device
    )
    image_to_text = mean_average_precision(
        query_image, database_text, query_labels, database_labels
    )
    text_to_image = mean_average_precision(
        query_text, database_image, query_labels, database_labels
    )
    return {
        "i2t": image_to_text,
        "t2i": text_to_image,
        "average": (image_to_text + text_to_image) / 2.0,
        "codes": (
            query_image,
            query_text,
            query_labels,
            database_image,
            database_text,
            database_labels,
        ),
    }


def generate_pr_outputs(config, metrics, output_dir):
    (
        query_image,
        query_text,
        query_labels,
        database_image,
        database_text,
        database_labels,
    ) = metrics["codes"]
    recall = np.linspace(0.0, 1.0, config.evaluation.pr_curve_points)
    image_to_text = precision_recall_curve(
        query_image,
        database_text,
        query_labels,
        database_labels,
        recall,
    )
    text_to_image = precision_recall_curve(
        query_text,
        database_image,
        query_labels,
        database_labels,
        recall,
    )
    return save_pr_outputs(
        True,
        output_dir,
        {
            "recall": recall,
            "i2t": image_to_text,
            "t2i": text_to_image,
            "average": (image_to_text + text_to_image) / 2.0,
        },
        {
            "dataset": config.dataset.name,
            "noise_rate": config.dataset.noise_rate,
            "hash_bits": config.model.hash_bits,
            "seed": config.experiment.seed,
        },
    )


def _write_log(path, line):
    print(line)
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def experiment_output_dir(config):
    return (
        Path(config.experiment.output_dir)
        / config.dataset.name
        / config.experiment.name
        / f"noise_{config.dataset.noise_rate:.1f}"
        / f"{config.model.hash_bits}bit"
        / f"seed{config.experiment.seed}"
    )


def run_experiment(config, device=None, protocol_overrides=None):
    validate_config(config, protocol_overrides=protocol_overrides)
    set_seed(config.experiment.seed)
    device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    output_dir = experiment_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best.pth"
    log_path = output_dir / "train.log"
    log_path.write_text("", encoding="utf-8")
    _write_log(
        log_path,
        (
            f"Experiment={config.experiment.name} | "
            f"dataset={config.dataset.name} | "
            f"pair={config.loss.pairwise.mode} | "
            f"noise={config.dataset.noise_rate} | "
            f"bits={config.model.hash_bits} | "
            f"seed={config.experiment.seed}"
        ),
    )
    (
        train_dataset,
        train_loader,
        selection_loader,
        query_loader,
        database_loader,
    ) = build_loaders(config)
    student, criterion, optimizer = build_components(config, device)
    best_metrics = {"average": float("-inf")}
    best_epoch = 0
    for epoch in range(config.train.epochs):
        selected = torch.ones(len(train_dataset), dtype=torch.bool)
        (
            classification_weights,
            classification_targets,
            _,
        ) = build_epoch_classification_supervision_with_metrics(
            config,
            epoch,
            student,
            selection_loader,
            device,
            len(train_dataset),
        )
        losses = train_epoch(
            student,
            criterion,
            train_loader,
            optimizer,
            selected,
            config,
            device,
            classification_weights=classification_weights,
            classification_targets=classification_targets,
            epoch=epoch,
        )
        if classification_weights is not None:
            losses["class_weight_mean"] = float(
                classification_weights.mean().item()
            )
        _write_log(
            log_path,
            format_epoch_log(epoch, config.train.epochs, losses),
        )
        evaluate_now = (
            (epoch + 1) % config.evaluation.interval == 0
            or epoch + 1 == config.train.epochs
        )
        if not evaluate_now:
            continue
        metrics = evaluate_retrieval(
            student, query_loader, database_loader, device
        )
        _write_log(
            log_path,
            f"Eval {epoch + 1:03d} | I2T={metrics['i2t']:.4f} | "
            f"T2I={metrics['t2i']:.4f} | "
            f"Avg={metrics['average']:.4f}",
        )
        if metrics["average"] > best_metrics["average"]:
            best_metrics = {
                key: metrics[key] for key in ("i2t", "t2i", "average")
            }
            best_epoch = epoch + 1
            torch.save(
                checkpoint_state(
                    best_epoch,
                    student,
                    criterion,
                    optimizer,
                    best_metrics,
                    config,
                ),
                checkpoint_path,
            )
    restore_checkpoint(
        checkpoint_path,
        student,
        criterion,
        optimizer,
        device,
    )
    final_metrics = evaluate_retrieval(
        student, query_loader, database_loader, device
    )
    result = {
        "best_epoch": best_epoch,
        "checkpoint": str(checkpoint_path),
        "log": str(log_path),
        **{
            key: final_metrics[key]
            for key in ("i2t", "t2i", "average")
        },
    }
    if config.evaluation.plot_pr_curve:
        result["pr_output_dir"] = str(
            generate_pr_outputs(config, final_metrics, output_dir / "pr")
        )
    _write_log(
        log_path,
        (
            f"Best epoch={best_epoch:03d} | "
            f"I2T={result['i2t']:.4f} | T2I={result['t2i']:.4f} | "
            f"Avg={result['average']:.4f}"
        ),
    )
    return result


def build_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--dataset-config")
    parser.add_argument("--method-config")
    parser.add_argument("--noise-rate", type=float)
    parser.add_argument("--noise-path")
    parser.add_argument("--hash-bits", type=int)
    parser.add_argument("--experiment-name")
    parser.add_argument(
        "--pairwise-mode",
        choices=("jaccard_contrast",),
    )
    parser.add_argument("--device", default=None)
    return parser


def config_from_args(args):
    single = args.config is not None
    composed = (
        args.dataset_config is not None
        or args.method_config is not None
    )
    if single == composed:
        raise ValueError(
            "use either --config or both "
            "--dataset-config/--method-config"
        )
    if single:
        return load_config(args.config)
    if not args.dataset_config or not args.method_config:
        raise ValueError(
            "--dataset-config and --method-config are both required"
        )
    overrides = {}
    for key, value in (
        ("dataset.noise_rate", args.noise_rate),
        ("dataset.noise_path", args.noise_path),
        ("model.hash_bits", args.hash_bits),
        ("experiment.name", args.experiment_name),
        ("loss.pairwise.mode", args.pairwise_mode),
    ):
        if value is not None:
            overrides[key] = value
    return compose_config(
        args.dataset_config, args.method_config, overrides
    )


def main():
    parser = build_argument_parser()
    args = parser.parse_args()
    print(run_experiment(config_from_args(args), args.device))


if __name__ == "__main__":
    main()
