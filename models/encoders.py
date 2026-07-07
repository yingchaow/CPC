import torch
import torch.nn as nn
import torch.nn.functional as F


def unpack_model_outputs(outputs):
    if len(outputs) == 2:
        image_hash, text_hash = outputs
        return image_hash, text_hash, None, None
    if len(outputs) == 4:
        return outputs
    raise ValueError(
        f"model must return 2 or 4 tensors, received {len(outputs)}"
    )


class HashEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, hash_bits, l2_normalize=True):
        super().__init__()
        dimensions = [input_dim, *hidden_dims, hash_bits]
        layers = []
        for index in range(len(dimensions) - 1):
            layers.append(nn.Linear(dimensions[index], dimensions[index + 1]))
            if index < len(dimensions) - 2:
                layers.append(nn.ReLU(inplace=True))
        self.network = nn.Sequential(*layers)
        self.l2_normalize = bool(l2_normalize)

    def forward(self, features):
        code = torch.tanh(self.network(features))
        if self.l2_normalize:
            code = F.normalize(code, p=2, dim=1)
        return code


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
        image_hash = self.image_encoder(image)
        text_hash = self.text_encoder(text)
        if not self.classification_enabled:
            return image_hash, text_hash
        return (
            image_hash,
            text_hash,
            self.image_classifier(image_hash),
            self.text_classifier(text_hash),
        )
