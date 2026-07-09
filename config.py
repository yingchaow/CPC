from copy import deepcopy
from pathlib import Path

import yaml

from classic_hashing.data.registry import get_dataset_spec


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


def validate_config(config, protocol_overrides=None):
    validate_dataset_section(config.dataset, protocol_overrides)
    pairwise_defaults = {
        "mode": "jaccard_contrast",
        "margin": 0.15,
        "shift": 0.8,
        "temperature": 1.0,
        "similarity": "jaccard",
    }
    for key, value in pairwise_defaults.items():
        if key not in config.loss.pairwise:
            config.loss.pairwise[key] = value
    hard_negative_defaults = {
        "enabled": False,
        "alpha": 1.0,
        "margin": 0.2,
        "label_threshold": 0.0,
    }
    if "hard_negative" not in config.loss.pairwise:
        config.loss.pairwise.hard_negative = ConfigNode()
    hard_negative = config.loss.pairwise.hard_negative
    for key, value in hard_negative_defaults.items():
        if key not in hard_negative:
            hard_negative[key] = value
    if "hard_negative" not in config.loss.center:
        config.loss.center.hard_negative = ConfigNode()
    center_hard_negative = config.loss.center.hard_negative
    for key, value in hard_negative_defaults.items():
        if key not in center_hard_negative:
            center_hard_negative[key] = value
    dual_center_defaults = {
        "enabled": False,
        "hard_weight": 0.1,
        "separation_weight": 0.1,
        "margin": 0.2,
        "warmup_epochs": 0,
        "top_k": 0,
        "reliability_enabled": False,
        "negative_centers": 1,
        "diversity_weight": 0.0,
        "hash_quantization_weight": 0.0,
    }
    if "dual_center" not in config.loss.center:
        config.loss.center.dual_center = ConfigNode()
    dual_center = config.loss.center.dual_center
    for key, value in dual_center_defaults.items():
        if key not in dual_center:
            dual_center[key] = value
    self_paced_defaults = {
        "enabled": False,
        "warmup_epochs": 5,
        "gamma_start": 0.3,
        "gamma_end": 1.2,
        "soft_label_correction": False,
    }
    if "self_paced" not in config.loss.center:
        config.loss.center.self_paced = ConfigNode()
    center_self_paced = config.loss.center.self_paced
    for key, value in self_paced_defaults.items():
        if key not in center_self_paced:
            center_self_paced[key] = value
    if "classification" not in config.loss:
        config.loss.classification = ConfigNode(
            {"enabled": False, "weight": 1.0}
        )
    if "balance" not in config.loss:
        config.loss.balance = ConfigNode({"enabled": False, "weight": 0.01})
    if "ema_consistency" not in config.loss:
        config.loss.ema_consistency = ConfigNode(
            {"enabled": False, "weight": 0.2}
        )
    if "cmp" not in config.loss:
        config.loss.cmp = ConfigNode(
            {"enabled": False, "weight": 0.1, "margin": 0.3}
        )
    knn_defaults = {
        "enabled": False,
        "warmup_epochs": 5,
        "k": 20,
        "gamma": 0.5,
        "chunk_size": 1024,
        "soft_label_enabled": False,
    }
    if "knn_classification_weight" not in config.robust_training:
        config.robust_training.knn_classification_weight = ConfigNode()
    knn_weight = config.robust_training.knn_classification_weight
    for key, value in knn_defaults.items():
        if key not in knn_weight:
            knn_weight[key] = value
    neighbor_defaults = {
        "enabled": False,
        "warmup_epochs": 5,
        "k": 20,
        "chunk_size": 1024,
        "pure_weight": 1.0,
        "hard_weight": 0.5,
        "noisy_weight": 0.2,
    }
    if "neighbor_refining" not in config.robust_training:
        config.robust_training.neighbor_refining = ConfigNode()
    neighbor_refining = config.robust_training.neighbor_refining
    for key, value in neighbor_defaults.items():
        if key not in neighbor_refining:
            neighbor_refining[key] = value
    if config.dataset.query_size < 1:
        raise ValueError("dataset.query_size must be positive")
    if config.dataset.train_size < 1:
        raise ValueError("dataset.train_size must be positive")
    if config.train.epochs < 1:
        raise ValueError("train.epochs must be positive")
    if config.train.batch_size < 1:
        raise ValueError("train.batch_size must be positive")
    if config.model.hash_bits not in (16, 32, 64, 128):
        raise ValueError("model.hash_bits must be one of 16, 32, 64, 128")
    if config.dataset.noise_rate not in (0.2, 0.5, 0.8):
        raise ValueError("dataset.noise_rate must be one of 0.2, 0.5, 0.8")
    if config.loss.pairwise.mode != "jaccard_contrast":
        raise ValueError("loss.pairwise.mode must be jaccard_contrast")
    if config.loss.pairwise.similarity not in (
        "jaccard",
        "dice",
        "cosine",
        "overlap",
        "binary",
    ):
        raise ValueError(
            "loss.pairwise.similarity must be jaccard, dice, cosine, "
            "overlap, or binary"
        )
    if hard_negative.alpha < 0:
        raise ValueError("hard_negative.alpha must be nonnegative")
    if hard_negative.margin < 0.0 or hard_negative.margin > 1.0:
        raise ValueError("hard_negative.margin must be between 0 and 1")
    if (
        hard_negative.label_threshold < 0.0
        or hard_negative.label_threshold > 1.0
    ):
        raise ValueError(
            "hard_negative.label_threshold must be between 0 and 1"
        )
    if center_hard_negative.alpha < 0:
        raise ValueError("center.hard_negative.alpha must be nonnegative")
    if (
        center_hard_negative.margin < 0.0
        or center_hard_negative.margin > 2.0
    ):
        raise ValueError(
            "center.hard_negative.margin must be between 0 and 2"
        )
    if (
        center_hard_negative.label_threshold < 0.0
        or center_hard_negative.label_threshold > 1.0
    ):
        raise ValueError(
            "center.hard_negative.label_threshold must be between 0 and 1"
        )
    if dual_center.hard_weight < 0:
        raise ValueError(
            "center.dual_center.hard_weight must be nonnegative"
        )
    if dual_center.separation_weight < 0:
        raise ValueError(
            "center.dual_center.separation_weight must be nonnegative"
        )
    if dual_center.margin < 0.0 or dual_center.margin > 2.0:
        raise ValueError(
            "center.dual_center.margin must be between 0 and 2"
        )
    if dual_center.warmup_epochs < 0:
        raise ValueError(
            "center.dual_center.warmup_epochs must be nonnegative"
        )
    if dual_center.top_k < 0:
        raise ValueError("center.dual_center.top_k must be nonnegative")
    if dual_center.negative_centers < 1:
        raise ValueError(
            "center.dual_center.negative_centers must be positive"
        )
    if dual_center.diversity_weight < 0:
        raise ValueError(
            "center.dual_center.diversity_weight must be nonnegative"
        )
    if dual_center.hash_quantization_weight < 0:
        raise ValueError(
            "center.dual_center.hash_quantization_weight must be nonnegative"
        )
    if center_self_paced.warmup_epochs < 0:
        raise ValueError(
            "center.self_paced.warmup_epochs must be nonnegative"
        )
    if center_self_paced.gamma_start <= 0:
        raise ValueError(
            "center.self_paced.gamma_start must be positive"
        )
    if center_self_paced.gamma_end <= 0:
        raise ValueError("center.self_paced.gamma_end must be positive")
    if config.loss.cmp.weight < 0:
        raise ValueError("loss.cmp.weight must be nonnegative")
    if config.loss.cmp.margin < 0.0 or config.loss.cmp.margin > 2.0:
        raise ValueError("loss.cmp.margin must be between 0 and 2")
    if config.robust_training.small_loss.schedule != "mgsh_linear":
        raise ValueError("small_loss.schedule must be mgsh_linear")
    if (
        config.loss.ema_consistency.enabled
        and not config.robust_training.ema_teacher.enabled
    ):
        raise ValueError("EMA consistency requires EMA Teacher")
    if config.loss.balance.enabled:
        raise ValueError("balance loss has been removed from this build")
    if config.loss.ema_consistency.enabled:
        raise ValueError(
            "EMA consistency loss has been removed from this build"
        )
    if (
        "semantic_multi_center" in config.loss.center
        and config.loss.center.semantic_multi_center.enabled
    ):
        raise ValueError(
            "semantic_multi_center has been removed from this build"
        )
    if (
        config.robust_training.small_loss.enabled
        and not config.loss.pairwise.enabled
        and not config.loss.center.enabled
    ):
        raise ValueError("Small-loss requires a supervised ranking signal")
    if knn_weight.enabled and not config.loss.classification.enabled:
        raise ValueError(
            "kNN classification weight requires classification loss"
        )
    if knn_weight.soft_label_enabled and not knn_weight.enabled:
        raise ValueError("soft labels require kNN classification weighting")
    if knn_weight.warmup_epochs < 0:
        raise ValueError(
            "knn_classification_weight.warmup_epochs must be nonnegative"
        )
    if knn_weight.k < 1:
        raise ValueError("knn_classification_weight.k must be positive")
    if not 0.0 <= knn_weight.gamma <= 1.0:
        raise ValueError(
            "knn_classification_weight.gamma must be between 0 and 1"
        )
    if knn_weight.chunk_size < 1:
        raise ValueError(
            "knn_classification_weight.chunk_size must be positive"
        )
    if neighbor_refining.enabled and not config.loss.classification.enabled:
        raise ValueError("neighbor refining requires classification loss")
    if neighbor_refining.enabled and not knn_weight.enabled:
        raise ValueError("neighbor refining requires kNN classification")
    if neighbor_refining.warmup_epochs < 0:
        raise ValueError(
            "neighbor_refining.warmup_epochs must be nonnegative"
        )
    if neighbor_refining.k < 1:
        raise ValueError("neighbor_refining.k must be positive")
    if neighbor_refining.chunk_size < 1:
        raise ValueError("neighbor_refining.chunk_size must be positive")
    for name in ("pure_weight", "hard_weight", "noisy_weight"):
        value = getattr(neighbor_refining, name)
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"neighbor_refining.{name} must be between 0 and 1"
            )
    if config.loss.center.update not in ("ema", "learnable"):
        raise ValueError("center.update must be ema or learnable")
    if (
        center_hard_negative.enabled
        and config.loss.center.update != "learnable"
    ):
        raise ValueError(
            "center hard_negative requires center.update=learnable"
        )
    if dual_center.enabled and config.loss.center.update != "learnable":
        raise ValueError(
            "dual_center requires center.update=learnable"
        )
    if not any(
        (
            config.loss.pairwise.enabled,
            config.loss.center.enabled,
            config.loss.quantization.enabled,
            config.loss.classification.enabled,
            config.loss.cmp.enabled,
        )
    ):
        raise ValueError("at least one training loss must be enabled")
    if config.evaluation.interval < 1:
        raise ValueError("evaluation.interval must be positive")
    if config.evaluation.pr_curve_points < 2:
        raise ValueError("evaluation.pr_curve_points must be >= 2")
    return config
