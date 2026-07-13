from copy import deepcopy
from pathlib import Path

import yaml

from classic_hashing.data.registry import get_dataset_spec


DEFAULT_CONFIG = {
    "experiment": {
        "name": "experiment",
        "seed": 42,
        "output_dir": "classic_hashing/outputs",
    },
    "model": {
        "hash_bits": 64,
        "image_hidden_dims": [8192, 8192],
        "text_hidden_dims": [8192],
        "l2_normalize": True,
    },
    "train": {
        "epochs": 50,
        "batch_size": 128,
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "num_workers": 4,
    },
    "robust_training": {
        "knn_classification_weight": {
            "enabled": True,
            "warmup_epochs": 1,
            "k": 20,
            "gamma": 0.5,
            "rise_momentum": 0.9,
            "chunk_size": 1024,
            "soft_label_enabled": True,
        },
    },
    "loss": {
        "pairwise": {
            "enabled": True,
            "mode": "jaccard_contrast",
            "weight": 0.7,
            "margin": 0.15,
            "shift": 0.8,
            "temperature": 1.0,
        },
        "center": {
            "enabled": True,
            "weight": 0.5,
            "update": "learnable",
            "momentum": 0.95,
            "temperature": 0.1,
            "rgce_r": 0.7,
            "dual_center": {
                "enabled": True,
                "hard_weight": 0.03,
                "separation_weight": 0.03,
                "margin": 0.2,
                "warmup_epochs": 5,
                "top_k": 2,
                "reliability_enabled": True,
                "negative_centers": 3,
                "diversity_weight": 0.01,
                "hash_quantization_weight": 0.01,
            },
        },
        "cmp": {"enabled": True, "weight": 0.1, "margin": 0.3},
        "classification": {"enabled": True, "weight": 1.0},
        "quantization": {"enabled": True, "weight": 2.5},
    },
    "evaluation": {
        "interval": 1,
        "plot_pr_curve": False,
        "pr_curve_points": 100,
    },
}


class ConfigNode(dict):
    def __getattribute__(self, name):
        if not name.startswith("__") and dict.__contains__(self, name):
            value = dict.__getitem__(self, name)
        else:
            return dict.__getattribute__(self, name)
        if isinstance(value, dict) and not isinstance(value, ConfigNode):
            value = ConfigNode(value)
            self[name] = value
        return value

    def __setattr__(self, name, value):
        self[name] = value


def _to_node(value):
    if isinstance(value, dict):
        return ConfigNode({key: _to_node(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_to_node(item) for item in value]
    return value


def _load_yaml(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return raw


def load_config(path):
    raw = _load_yaml(path)
    config = _to_node(raw)
    validate_config(config)
    return config


def load_dataset_config(path):
    raw = _load_yaml(path)
    if set(raw) != {"dataset"}:
        raise ValueError("dataset config must contain only dataset")
    config = _to_node(raw)
    validate_dataset_section(config.dataset)
    return config


def _deep_merge(left, right):
    merged = deepcopy(left)
    for key, value in right.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _apply_defaults(config, defaults):
    for key, value in defaults.items():
        if key not in config:
            config[key] = _to_node(deepcopy(value))
            continue
        if isinstance(config[key], dict) and isinstance(value, dict):
            _apply_defaults(config[key], value)


def _prune_unknown(config, schema):
    for key in list(config.keys()):
        if key not in schema:
            del config[key]
            continue
        if isinstance(config[key], dict) and isinstance(schema[key], dict):
            _prune_unknown(config[key], schema[key])


def _apply_override(raw, dotted_key, value):
    parts = dotted_key.split(".")
    target = raw
    for part in parts[:-1]:
        if part not in target or not isinstance(target[part], dict):
            raise KeyError(f"cannot override unknown path {dotted_key}")
        target = target[part]
    if parts[-1] not in target:
        raise KeyError(f"cannot override unknown path {dotted_key}")
    target[parts[-1]] = value


def compose_config(dataset_path, method_path, overrides=None):
    raw = _deep_merge(_load_yaml(method_path), _load_yaml(dataset_path))
    for key, value in (overrides or {}).items():
        _apply_override(raw, key, value)
    config = _to_node(raw)
    validate_config(config)
    return config


def clone_config(config):
    return _to_node(deepcopy(dict(config)))


def validate_dataset_section(dataset, protocol_overrides=None):
    spec = get_dataset_spec(dataset.name)
    dataset.name = spec.name
    if "source_type" not in dataset:
        dataset.source_type = spec.source_type
    expected_protocol = {
        "source_type": spec.source_type,
        "image_dim": spec.image_dim,
        "text_dim": spec.text_dim,
        "num_classes": spec.num_classes,
        "query_size": spec.query_size,
        "train_size": spec.train_size,
        "database_start": spec.database_start,
    }
    expected_protocol.update(protocol_overrides or {})
    for field, expected in expected_protocol.items():
        actual = getattr(dataset, field)
        if actual != expected:
            raise ValueError(
                f"dataset.{field}={actual} does not match "
                f"{spec.name} protocol value {expected}"
            )
    return spec


def _require_positive(name, value):
    if value < 1:
        raise ValueError(f"{name} must be positive")


def _require_nonnegative(name, value):
    if value < 0:
        raise ValueError(f"{name} must be nonnegative")


def _require_between(name, value, low, high):
    if value < low or value > high:
        raise ValueError(f"{name} must be between {low} and {high}")


def _require_choice(name, value, choices):
    if value not in choices:
        allowed = ", ".join(str(choice) for choice in choices)
        raise ValueError(f"{name} must be one of {allowed}")


def _validate_loss_weights(loss):
    for name in ("pairwise", "center", "quantization", "classification", "cmp"):
        _require_nonnegative(f"loss.{name}.weight", loss[name].weight)


def _validate_pairwise(pairwise):
    _require_choice("loss.pairwise.mode", pairwise.mode, ("jaccard_contrast",))


def _validate_center(center):
    _require_choice("loss.center.update", center.update, ("ema", "learnable"))

    dual = center.dual_center
    _require_nonnegative("loss.center.dual_center.hard_weight", dual.hard_weight)
    _require_nonnegative("loss.center.dual_center.separation_weight", dual.separation_weight)
    _require_between("loss.center.dual_center.margin", dual.margin, 0.0, 2.0)
    _require_nonnegative("loss.center.dual_center.warmup_epochs", dual.warmup_epochs)
    _require_nonnegative("loss.center.dual_center.top_k", dual.top_k)
    _require_positive("loss.center.dual_center.negative_centers", dual.negative_centers)
    _require_nonnegative("loss.center.dual_center.diversity_weight", dual.diversity_weight)
    _require_nonnegative(
        "loss.center.dual_center.hash_quantization_weight",
        dual.hash_quantization_weight,
    )

    if dual.enabled and center.update != "learnable":
        raise ValueError("dual_center requires center.update=learnable")


def _validate_knn(knn, classification_enabled):
    if knn.enabled and not classification_enabled:
        raise ValueError("kNN classification weight requires classification loss")
    if knn.soft_label_enabled and not knn.enabled:
        raise ValueError("soft labels require kNN classification weighting")
    _require_nonnegative("knn_classification_weight.warmup_epochs", knn.warmup_epochs)
    _require_positive("knn_classification_weight.k", knn.k)
    _require_between("knn_classification_weight.gamma", knn.gamma, 0.0, 1.0)
    _require_between(
        "knn_classification_weight.rise_momentum",
        knn.rise_momentum,
        0.0,
        1.0,
    )
    _require_positive("knn_classification_weight.chunk_size", knn.chunk_size)


def validate_config(config, protocol_overrides=None):
    _apply_defaults(config, DEFAULT_CONFIG)
    for section, schema in DEFAULT_CONFIG.items():
        _prune_unknown(config[section], schema)
    validate_dataset_section(config.dataset, protocol_overrides)

    _require_positive("dataset.query_size", config.dataset.query_size)
    _require_positive("dataset.train_size", config.dataset.train_size)
    _require_choice("dataset.noise_rate", config.dataset.noise_rate, (0.2, 0.5, 0.8))

    _require_positive("train.epochs", config.train.epochs)
    _require_positive("train.batch_size", config.train.batch_size)
    _require_positive("model.hash_bits", config.model.hash_bits)
    _require_choice("model.hash_bits", config.model.hash_bits, (16, 32, 64, 128))

    loss = config.loss
    _validate_loss_weights(loss)
    _validate_pairwise(loss.pairwise)
    _validate_center(loss.center)
    _require_between("loss.cmp.margin", loss.cmp.margin, 0.0, 2.0)
    _validate_knn(
        config.robust_training.knn_classification_weight,
        loss.classification.enabled,
    )

    if not any(loss[name].enabled for name in DEFAULT_CONFIG["loss"]):
        raise ValueError("at least one training loss must be enabled")
    _require_positive("evaluation.interval", config.evaluation.interval)
    if config.evaluation.pr_curve_points < 2:
        raise ValueError("evaluation.pr_curve_points must be >= 2")
    return config
