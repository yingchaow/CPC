from typing import NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DualHashOutput(NamedTuple):
    image_hash: torch.Tensor
    text_hash: torch.Tensor
    image_logits: Optional[torch.Tensor]
    text_logits: Optional[torch.Tensor]
    image_semantic: torch.Tensor
    text_semantic: torch.Tensor


def unpack_model_outputs(outputs):
    if isinstance(outputs, DualHashOutput):
        return (
            outputs.image_hash,
            outputs.text_hash,
            outputs.image_logits,
            outputs.text_logits,
        )
    if len(outputs) == 2:
        image_hash, text_hash = outputs
        return image_hash, text_hash, None, None
    if len(outputs) == 4:
        return outputs
    raise ValueError(
        f"model must return 2 or 4 tensors, received {len(outputs)}"
    )


def unpack_semantic_features(outputs):
    if isinstance(outputs, DualHashOutput):
        return outputs.image_semantic, outputs.text_semantic
    image_hash, text_hash, _, _ = unpack_model_outputs(outputs)
    return image_hash, text_hash


class HashEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, hash_bits, l2_normalize=True):
        super().__init__()
        dimensions = [input_dim, *hidden_dims]
        layers = []
        for index in range(len(dimensions) - 1):
            layers.append(nn.Linear(dimensions[index], dimensions[index + 1]))
            layers.append(nn.ReLU(inplace=True))
        self.semantic_network = nn.Sequential(*layers)
        self.semantic_dim = dimensions[-1]
        self.hash_projection = nn.Linear(self.semantic_dim, hash_bits)
        self.l2_normalize = bool(l2_normalize)

    def forward(self, features):
        semantic = self.semantic_network(features)
        code = torch.tanh(self.hash_projection(semantic))
        if self.l2_normalize:
            code = F.normalize(code, p=2, dim=1)
        return code, semantic


class DualHashModel(nn.Module):
    def __init__(
        self,
        image_dim,
        text_dim,
        hash_bits,
        image_hidden_dims,
        text_hidden_dims,
        l2_normalize=True,
        num_classes=None,
        classification_enabled=False,
    ):
        super().__init__()
        self.image_encoder = HashEncoder(
            image_dim, image_hidden_dims, hash_bits, l2_normalize
        )
        self.text_encoder = HashEncoder(
            text_dim, text_hidden_dims, hash_bits, l2_normalize
        )
        self.classification_enabled = bool(classification_enabled)
        if self.classification_enabled:
            if not num_classes:
                raise ValueError(
                    "num_classes is required for classification heads"
                )
            self.image_classifier = nn.Linear(
                hash_bits, num_classes, bias=False
            )
            self.text_classifier = nn.Linear(
                hash_bits, num_classes, bias=False
            )
            nn.init.orthogonal_(self.image_classifier.weight)
            nn.init.orthogonal_(self.text_classifier.weight)

    def forward(self, image, text):
        image_hash, image_semantic = self.image_encoder(image)
        text_hash, text_semantic = self.text_encoder(text)
        image_logits = None
        text_logits = None
        if self.classification_enabled:
            image_logits = self.image_classifier(image_hash)
            text_logits = self.text_classifier(text_hash)
        return DualHashOutput(
            image_hash=image_hash,
            text_hash=text_hash,
            image_logits=image_logits,
            text_logits=text_logits,
            image_semantic=image_semantic,
            text_semantic=text_semantic,
        )
