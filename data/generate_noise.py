import argparse
from pathlib import Path

import numpy as np


def apply_mgsh_label_noise(labels, noise_rate, seed=42):
    labels = np.asarray(labels, dtype=np.float32)
    if labels.ndim != 2:
        raise ValueError("labels must be a 2-D matrix")
    if not 0.0 <= noise_rate <= 1.0:
        raise ValueError("noise_rate must be in [0, 1]")
    eligible = np.flatnonzero(
        np.logical_and(labels.sum(1) > 0, labels.sum(1) < labels.shape[1])
    )
    corrupt_count = int(len(labels) * noise_rate)
    if corrupt_count > len(eligible):
        raise ValueError("not enough eligible rows for requested noise rate")
    rng = np.random.default_rng(seed)
    corrupted = (
        rng.choice(eligible, size=corrupt_count, replace=False)
        if corrupt_count
        else np.empty(0, dtype=np.int64)
    )
    noisy = labels.copy()
    noise_mask = np.zeros(len(labels), dtype=bool)
    for index in corrupted:
        positive = np.flatnonzero(noisy[index] > 0)
        negative = np.flatnonzero(noisy[index] <= 0)
        noisy[index, rng.choice(positive)] = 0.0
        noisy[index, rng.choice(negative)] = 1.0
        noise_mask[index] = True
    return noisy, noise_mask


def generate_noise_file(data_path, output_path, noise_rate, seed=42):
    import h5py

    with h5py.File(data_path, "r") as handle:
        clean = handle["LabTrain"][:].astype(np.float32)
        dataset_name = handle.attrs.get("dataset", "unknown")
    noisy, noise_mask = apply_mgsh_label_noise(clean, noise_rate, seed)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as handle:
        handle.create_dataset("clean_labels", data=clean)
        handle.create_dataset("noisy_labels", data=noisy)
        handle.create_dataset("noise_mask", data=noise_mask)
        handle.attrs["noise_rate"] = float(noise_rate)
        handle.attrs["seed"] = int(seed)
        handle.attrs["dataset"] = dataset_name
    return {
        "output": str(output_path),
        "samples": len(clean),
        "corrupted": int(noise_mask.sum()),
        "actual_rate": float(noise_mask.mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-h5", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--noise-rate", required=True, type=float)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(
        generate_noise_file(
            args.data_h5, args.output, args.noise_rate, args.seed
        )
    )


if __name__ == "__main__":
    main()
