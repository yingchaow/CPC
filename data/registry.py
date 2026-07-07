from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    aliases: tuple[str, ...]
    source_type: str
    image_dim: int
    text_dim: int
    num_classes: int
    query_size: int
    train_size: int
    database_start: int
    image_keys: tuple[str, ...]
    text_keys: tuple[str, ...]
    label_keys: tuple[str, ...]


DATASET_SPECS = {
    "mirflickr": DatasetSpec(
        "mirflickr", ("flickr", "mirflickr25k"), "separate_mat",
        4096, 1386, 24, 1900, 9500, 1900,
        ("XAll",), ("YAll",), ("LAll",),
    ),
    "mirflickr_scbch": DatasetSpec(
        "mirflickr_scbch",
        ("flickr_scbch", "mirflickr-scbch"),
        "separate_mat",
        4096,
        1386,
        24,
        2000,
        10000,
        2000,
        ("XAll",),
        ("YAll",),
        ("LAll",),
    ),
    "nuswide10": DatasetSpec(
        "nuswide10", ("nuswide", "nus-wide-tc10"), "separate_mat",
        4096, 1000, 10, 2100, 10500, 2100,
        ("XAll", "IAll"), ("YAll",), ("LAll",),
    ),
    "mscoco": DatasetSpec(
        "mscoco", ("coco", "ms-coco"), "single_hdf5",
        4096, 300, 80, 5000, 10000, 15000,
        ("XAll",), ("YAll",), ("LAll",),
    ),
    "iapr": DatasetSpec(
        "iapr", ("iapr-tc12",), "iapr_mat",
        4096, 2912, 255, 2000, 10000, 2000,
        ("VDatabase", "VTest"),
        ("YDatabase", "YTest"),
        ("databaseL", "testL"),
    ),
}

_ALIASES = {
    alias: name
    for name, spec in DATASET_SPECS.items()
    for alias in (name, *spec.aliases)
}


def canonical_dataset_name(name):
    normalized = str(name).strip().lower()
    if normalized not in _ALIASES:
        raise ValueError(f"unsupported dataset: {name}")
    return _ALIASES[normalized]


def get_dataset_spec(name):
    return DATASET_SPECS[canonical_dataset_name(name)]
