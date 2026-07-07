import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from classic_hashing.config import load_dataset_config
from classic_hashing.data.registry import get_dataset_spec


def _orient(array, feature_dim, name):
    array = np.asarray(array)
    if array.ndim != 2:
        raise ValueError(f"{name} must be two-dimensional")
    row_match = array.shape[1] == feature_dim
    column_match = array.shape[0] == feature_dim
    if row_match and column_match:
        raise ValueError(f"{name} has ambiguous sample axis: {array.shape}")
    if row_match:
        return np.asarray(array, dtype=np.float32)
    if column_match:
        return np.asarray(array.T, dtype=np.float32)
    raise ValueError(
        f"{name} shape {array.shape} does not contain "
        f"feature dimension {feature_dim}"
    )


def _read_mat(path, keys):
    from scipy.io import loadmat

    payload = loadmat(path)
    for key in keys:
        if key in payload:
            return payload[key]
    raise KeyError(f"{path} is missing all keys {keys}")


def _read_hdf5(path, keys):
    import h5py

    with h5py.File(path, "r") as handle:
        for key in keys:
            if key in handle:
                return handle[key][:]
    raise KeyError(f"{path} is missing all keys {keys}")


def load_raw_arrays(dataset, spec, expected_dims=None):
    image_dim, text_dim, num_classes = expected_dims or (
        spec.image_dim,
        spec.text_dim,
        spec.num_classes,
    )
    if spec.source_type == "separate_mat":
        images = _read_mat(dataset["image_path"], spec.image_keys)
        texts = _read_mat(dataset["text_path"], spec.text_keys)
        labels = _read_mat(dataset["label_path"], spec.label_keys)
        images = _orient(images, image_dim, "images")
        texts = _orient(texts, text_dim, "texts")
        labels = _orient(labels, num_classes, "labels")
    elif spec.source_type == "single_hdf5":
        images = _orient(
            _read_hdf5(dataset["source_path"], spec.image_keys),
            image_dim,
            "images",
        )
        texts = _orient(
            _read_hdf5(dataset["source_path"], spec.text_keys),
            text_dim,
            "texts",
        )
        labels = _orient(
            _read_hdf5(dataset["source_path"], spec.label_keys),
            num_classes,
            "labels",
        )
    elif spec.source_type == "iapr_mat":
        images = np.concatenate(
            [
                _orient(
                    _read_mat(dataset["source_path"], ("VDatabase",)),
                    image_dim,
                    "VDatabase",
                ),
                _orient(
                    _read_mat(dataset["source_path"], ("VTest",)),
                    image_dim,
                    "VTest",
                ),
            ],
            axis=0,
        )
        texts = np.concatenate(
            [
                _orient(
                    _read_mat(dataset["source_path"], ("YDatabase",)),
                    text_dim,
                    "YDatabase",
                ),
                _orient(
                    _read_mat(dataset["source_path"], ("YTest",)),
                    text_dim,
                    "YTest",
                ),
            ],
            axis=0,
        )
        labels = np.concatenate(
            [
                _orient(
                    _read_mat(dataset["source_path"], ("databaseL",)),
                    num_classes,
                    "databaseL",
                ),
                _orient(
                    _read_mat(dataset["source_path"], ("testL",)),
                    num_classes,
                    "testL",
                ),
            ],
            axis=0,
        )
    else:
        raise ValueError(f"unsupported source type: {spec.source_type}")
    if not (len(images) == len(texts) == len(labels)):
        raise ValueError("image, text, and labels have different sample counts")
    return images, texts, (labels > 0).astype(np.float32)


def build_split(spec, sample_count):
    train_end = spec.query_size + spec.train_size
    if sample_count <= max(train_end - 1, spec.database_start):
        raise ValueError(
            f"{spec.name} has {sample_count} samples, insufficient for "
            f"query={spec.query_size}, train={spec.train_size}, "
            f"database_start={spec.database_start}"
        )
    return {
        "query": np.arange(0, spec.query_size, dtype=np.int64),
        "train": np.arange(
            spec.query_size, train_end, dtype=np.int64
        ),
        "database": np.arange(
            spec.database_start, sample_count, dtype=np.int64
        ),
    }


def convert_dataset(config, protocol_overrides=None):
    import h5py

    spec = get_dataset_spec(config.dataset.name)
    effective_spec = replace(
        spec,
        image_dim=config.dataset.image_dim,
        text_dim=config.dataset.text_dim,
        num_classes=config.dataset.num_classes,
        query_size=(protocol_overrides or {}).get(
            "query_size", spec.query_size
        ),
        train_size=(protocol_overrides or {}).get(
            "train_size", spec.train_size
        ),
        database_start=(protocol_overrides or {}).get(
            "database_start", spec.database_start
        ),
    )
    images, texts, labels = load_raw_arrays(
        config.dataset, effective_spec
    )
    split = build_split(effective_spec, len(images))
    output_path = Path(config.dataset.h5_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_path, "w") as handle:
        for prefix, indices in (
            ("Train", split["train"]),
            ("Query", split["query"]),
            ("DataBase", split["database"]),
        ):
            handle.create_dataset(f"Img{prefix}", data=images[indices])
            handle.create_dataset(f"Tag{prefix}", data=texts[indices])
            handle.create_dataset(f"Lab{prefix}", data=labels[indices])
        metadata = {
            "dataset": effective_spec.name,
            "image_dim": effective_spec.image_dim,
            "text_dim": effective_spec.text_dim,
            "num_classes": effective_spec.num_classes,
            "query_size": effective_spec.query_size,
            "train_size": effective_spec.train_size,
            "database_start": effective_spec.database_start,
            "database_size": len(split["database"]),
        }
        for key, value in metadata.items():
            handle.attrs[key] = value
    split_path = Path(config.dataset.split_path)
    split_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(split_path, **split)
    return {
        "dataset": effective_spec.name,
        "samples": len(images),
        "query": len(split["query"]),
        "train": len(split["train"]),
        "database": len(split["database"]),
        "output": str(output_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(convert_dataset(load_dataset_config(args.config)))


if __name__ == "__main__":
    main()

