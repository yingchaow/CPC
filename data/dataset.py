from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class NoiseDiagnostics:
    clean_labels: np.ndarray
    noise_mask: np.ndarray


class FeatureDataset(Dataset):
    def __init__(self, images, texts, labels, diagnostics=None):
        self.images = torch.from_numpy(np.asarray(images, dtype=np.float32))
        self.texts = torch.from_numpy(np.asarray(texts, dtype=np.float32))
        self.labels = torch.from_numpy(np.asarray(labels, dtype=np.float32))
        self.diagnostics = diagnostics
        if not (
            len(self.images) == len(self.texts) == len(self.labels)
        ):
            raise ValueError("dataset arrays must align")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return (
            self.images[index],
            self.texts[index],
            self.labels[index],
            index,
        )


def _read_split(handle, prefix):
    return (
        handle[f"Img{prefix}"][:],
        handle[f"Tag{prefix}"][:],
        handle[f"Lab{prefix}"][:],
    )


def create_datasets(data_path, noise_path, expected=None):
    import h5py

    with h5py.File(data_path, "r") as data:
        required = {
            f"{kind}{split}"
            for kind in ("Img", "Tag", "Lab")
            for split in ("Train", "Query", "DataBase")
        }
        missing = required.difference(data.keys())
        if missing:
            raise KeyError(f"data HDF5 is missing keys: {sorted(missing)}")
        if expected is not None:
            for field, configured in expected.items():
                actual = data.attrs.get(field)
                if actual != configured:
                    raise ValueError(
                        f"HDF5 {field}={actual} does not match "
                        f"configuration value {configured}"
                    )
        train_image, train_text, clean_train = _read_split(data, "Train")
        query = _read_split(data, "Query")
        database = _read_split(data, "DataBase")
    with h5py.File(noise_path, "r") as noise:
        required_noise = {
            "noisy_labels", "clean_labels", "noise_mask"
        }
        missing_noise = required_noise.difference(noise.keys())
        if missing_noise:
            raise KeyError(
                f"noise HDF5 is missing keys: {sorted(missing_noise)}"
            )
        noisy_labels = noise["noisy_labels"][:]
        clean_labels = noise["clean_labels"][:]
        diagnostics = NoiseDiagnostics(
            clean_labels=clean_labels,
            noise_mask=noise["noise_mask"][:].astype(bool),
        )
        if expected is not None:
            noise_dataset = noise.attrs.get("dataset", expected["dataset"])
            if noise_dataset != expected["dataset"]:
                raise ValueError(
                    "noise dataset metadata does not match data HDF5"
                )
    if not np.array_equal(clean_train, clean_labels):
        raise ValueError("noise file clean labels do not match LabTrain")
    if len(noisy_labels) != len(train_image):
        raise ValueError("noise labels do not match Train length")
    if diagnostics.noise_mask.shape != (len(train_image),):
        raise ValueError(
            "noise_mask must contain one value per Train sample"
        )
    for split_name, arrays in (
        ("Train", (train_image, train_text, clean_train)),
        ("Query", query),
        ("DataBase", database),
    ):
        if len({len(array) for array in arrays}) != 1:
            raise ValueError(f"{split_name} arrays must align")
    if expected is not None:
        for split_name, arrays in (
            ("Train", (train_image, train_text, clean_train)),
            ("Query", query),
            ("DataBase", database),
        ):
            if arrays[0].shape[1] != expected["image_dim"]:
                raise ValueError(f"{split_name} image_dim mismatch")
            if arrays[1].shape[1] != expected["text_dim"]:
                raise ValueError(f"{split_name} text_dim mismatch")
            if arrays[2].shape[1] != expected["num_classes"]:
                raise ValueError(f"{split_name} num_classes mismatch")
    train = FeatureDataset(
        train_image, train_text, noisy_labels, diagnostics
    )
    return train, FeatureDataset(*query), FeatureDataset(*database)
