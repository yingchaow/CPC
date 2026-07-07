from pathlib import Path
import re

import numpy as np


LOSS_KEYS = (
    "total",
    "pairwise",
    "center",
    "cmp",
    "classification",
    "quantization",
    "balance",
    "ema_consistency",
)


def parse_training_losses(text):
    rows = []
    keys = set()
    epoch_pattern = re.compile(r"Epoch\s+(\d+)(?:/\d+)?")
    metric_pattern = re.compile(
        r"\b([A-Za-z_][A-Za-z0-9_]*)=([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)"
    )
    legacy_pattern = re.compile(
        r"Epoch\s+(\d+)\s+Training Loss:\s*"
        r"([-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?)"
    )
    for line in text.splitlines():
        legacy = legacy_pattern.search(line)
        if legacy:
            rows.append(
                {"epoch": int(legacy.group(1)), "total": float(legacy.group(2))}
            )
            keys.add("total")
            continue
        if "Epoch" not in line or "=" not in line:
            continue
        epoch_match = epoch_pattern.search(line)
        if not epoch_match:
            continue
        values = {
            key: float(value)
            for key, value in metric_pattern.findall(line)
            if key in LOSS_KEYS
        }
        if not values:
            continue
        values["epoch"] = int(epoch_match.group(1))
        rows.append(values)
        keys.update(values.keys() - {"epoch"})
    result = {"epoch": [row["epoch"] for row in rows]}
    for key in LOSS_KEYS:
        if key in keys:
            result[key] = [row.get(key, np.nan) for row in rows]
    return result


def save_loss_outputs(log_path, output_dir, metadata):
    log_path = Path(log_path)
    if not log_path.exists():
        return None
    losses = parse_training_losses(log_path.read_text(encoding="utf-8"))
    if not losses.get("epoch"):
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = np.asarray(losses["epoch"], dtype=float)
    styles = {
        "total": ("#000000", "Total"),
        "pairwise": ("#1f77b4", "Pairwise"),
        "center": ("#2ca02c", "Center"),
        "cmp": ("#9467bd", "CMP"),
        "classification": ("#ff7f0e", "Classification"),
        "quantization": ("#8c564b", "Quantization"),
        "balance": ("#e377c2", "Balance"),
        "ema_consistency": ("#7f7f7f", "EMA Consistency"),
    }
    figure = plt.figure(figsize=(7, 4.5))
    for key, (color, label) in styles.items():
        if key not in losses:
            continue
        values = np.asarray(losses[key], dtype=float)
        if np.isnan(values).all():
            continue
        plt.plot(epochs, values, color=color, linewidth=2, label=label)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.legend()
    plt.title(
        f"{metadata['dataset']} Loss Convergence "
        f"noise={metadata['noise_rate']} "
        f"bits={metadata['hash_bits']} seed={metadata['seed']}"
    )
    plt.tight_layout()
    figure.savefig(output_dir / "loss_curves.png", dpi=200)
    plt.close(figure)
    np.savez(
        output_dir / "loss_data.npz",
        **{key: np.asarray(value, dtype=float) for key, value in losses.items()},
        **metadata,
    )
    return output_dir


def save_pr_outputs(enabled, output_dir, curves, metadata):
    if not enabled:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    recall = curves["recall"]
    figure = plt.figure(figsize=(6, 4.5))
    styles = {
        "i2t": ("#1f77b4", "Image→Text"),
        "t2i": ("#ff7f0e", "Text→Image"),
        "average": ("#2ca02c", "Average"),
    }
    for name, (color, label) in styles.items():
        plt.plot(recall, curves[name], color=color, linewidth=2, label=label)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.legend()
    plt.title(
        f"{metadata['dataset']} PR Curves "
        f"noise={metadata['noise_rate']} "
        f"bits={metadata['hash_bits']} seed={metadata['seed']}"
    )
    plt.tight_layout()
    figure.savefig(output_dir / "pr_all.png", dpi=200)
    plt.close(figure)
    for name in ("i2t", "t2i", "average"):
        figure = plt.figure()
        plt.plot(recall, curves[name], color=styles[name][0], linewidth=2)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.ylim(0, 1)
        plt.xlim(0, 1)
        plt.title(
            f"{metadata['dataset']} {name.upper()} "
            f"noise={metadata['noise_rate']} "
            f"bits={metadata['hash_bits']} seed={metadata['seed']}"
        )
        plt.tight_layout()
        figure.savefig(output_dir / f"pr_{name}.png", dpi=200)
        plt.close(figure)
    np.savez(
        output_dir / "pr_data.npz",
        recall=recall,
        precision_i2t=curves["i2t"],
        precision_t2i=curves["t2i"],
        precision_average=curves["average"],
        **metadata,
    )
    return output_dir


def main():
    import torch

    from classic_hashing.train import (
        build_argument_parser,
        build_components,
        build_loaders,
        config_from_args,
        evaluate_retrieval,
        generate_pr_outputs,
    )
    from classic_hashing.training.trainer import restore_checkpoint

    parser = build_argument_parser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--log-path")
    args = parser.parse_args()
    config = config_from_args(args)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    _, _, _, query_loader, database_loader = build_loaders(config)
    student, teacher, criterion, optimizer = build_components(
        config, device
    )
    restore_checkpoint(
        args.checkpoint,
        student,
        teacher,
        criterion,
        optimizer,
        device,
    )
    metrics = evaluate_retrieval(
        student, query_loader, database_loader, device
    )
    output_dir = Path(args.checkpoint).parent / "pr"
    generate_pr_outputs(config, metrics, output_dir)
    log_path = (
        Path(args.log_path)
        if args.log_path
        else Path(args.checkpoint).parent / "train.log"
    )
    save_loss_outputs(
        log_path,
        output_dir,
        {
            "dataset": config.dataset.name,
            "noise_rate": config.dataset.noise_rate,
            "hash_bits": config.model.hash_bits,
            "seed": config.experiment.seed,
        },
    )


if __name__ == "__main__":
    main()
