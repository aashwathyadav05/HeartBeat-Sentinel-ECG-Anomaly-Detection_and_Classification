"""Unified shared-encoder ECG model for PTB-XL.

One 1D convolutional encoder feeds two task-specific heads:

1. a reconstruction decoder used for anomaly scoring via reconstruction error;
2. a multilabel classifier producing five logits (NORM, MI, STTC, CD, HYP).

The categories are independent (not mutually exclusive), so inference uses
sigmoid scores and training uses ``BCEWithLogitsLoss``. No softmax is used.

Input contract: ``(batch, 12, 1000)`` at 100 Hz, leads in the fixed PTB-XL
order. Signals must already be normalized with the saved training
normalization statistics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .ptbxl_data import LEAD_ORDER, TARGET_CATEGORIES

MODEL_ARCHITECTURE = "shared_encoder_1d_cnn_v1"


@dataclass
class UnifiedModelConfig:
    """Configurable architecture hyperparameters.

    Parameters
    ----------
    input_channels:
        Number of ECG leads (12).
    sequence_length:
        Samples per lead at the training sampling rate (1000 for 10 s @ 100 Hz).
    encoder_channels:
        Output channels of each encoder convolution stage. Strides are 2 each,
        so the temporal resolution is reduced by ``2 ** len(encoder_channels)``.
    decoder_channels:
        Optional explicit decoder channel widths (defaults to reversed encoder).
    classifier_hidden:
        Hidden units in the multilabel head.
    dropout:
        Dropout probability in the classifier head.
    """

    input_channels: int = len(LEAD_ORDER)
    sequence_length: int = 1000
    encoder_channels: Tuple[int, ...] = (32, 64, 128, 256)
    decoder_channels: Optional[Tuple[int, ...]] = None
    classifier_hidden: int = 128
    dropout: float = 0.2

    def __post_init__(self) -> None:
        if self.input_channels != len(LEAD_ORDER):
            raise ValueError(f"input_channels must be {len(LEAD_ORDER)} for PTB-XL")
        if self.sequence_length < 16:
            raise ValueError("sequence_length must be at least 16")
        if not self.encoder_channels:
            raise ValueError("encoder_channels must not be empty")
        if any(channel <= 0 for channel in self.encoder_channels):
            raise ValueError("encoder_channels must be positive")
        if self.classifier_hidden <= 0:
            raise ValueError("classifier_hidden must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.decoder_channels is not None:
            if len(self.decoder_channels) != len(self.encoder_channels):
                raise ValueError("decoder_channels must have the same length as encoder_channels")
            if any(channel <= 0 for channel in self.decoder_channels):
                raise ValueError("decoder_channels must be positive")

    @property
    def feature_dim(self) -> int:
        # Encoder output is summarized with concatenated mean and max pooling.
        return 2 * int(self.encoder_channels[-1])

    def to_dict(self) -> Dict[str, object]:
        data = asdict(self)
        data["encoder_channels"] = list(self.encoder_channels)
        data["decoder_channels"] = (
            list(self.decoder_channels) if self.decoder_channels is not None else None
        )
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "UnifiedModelConfig":
        decoder = data.get("decoder_channels")
        return cls(
            input_channels=int(data.get("input_channels", len(LEAD_ORDER))),
            sequence_length=int(data.get("sequence_length", 1000)),
            encoder_channels=tuple(int(v) for v in data.get("encoder_channels", (32, 64, 128, 256))),
            decoder_channels=tuple(int(v) for v in decoder) if decoder else None,
            classifier_hidden=int(data.get("classifier_hidden", 128)),
            dropout=float(data.get("dropout", 0.2)),
        )


@dataclass
class UnifiedModelOutput:
    """Structured forward-pass result."""

    reconstruction: torch.Tensor
    classification_logits: torch.Tensor
    features: torch.Tensor
    pooled_features: torch.Tensor

    @property
    def classification_scores(self) -> torch.Tensor:
        """Independent sigmoid scores; not calibrated probabilities."""
        return torch.sigmoid(self.classification_logits)

    def to_dict(self) -> Dict[str, torch.Tensor]:
        return {
            "reconstruction": self.reconstruction,
            "classification_logits": self.classification_logits,
            "classification_scores": self.classification_scores,
            "features": self.features,
            "pooled_features": self.pooled_features,
        }


class SharedECGEncoder(nn.Module):
    """1D convolutional ECG encoder shared by both tasks."""

    def __init__(self, config: UnifiedModelConfig) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        in_channels = config.input_channels
        for index, out_channels in enumerate(config.encoder_channels):
            kernel = 7 if index == 0 else 5 if index == 1 else 3
            padding = kernel // 2
            layers.extend(
                [
                    nn.Conv1d(in_channels, out_channels, kernel_size=kernel, stride=2, padding=padding),
                    nn.BatchNorm1d(out_channels),
                    nn.GELU(),
                ]
            )
            in_channels = out_channels
        # One extra stride-1 convolution strengthens the shared representation.
        layers.extend(
            [
                nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(in_channels),
                nn.GELU(),
            ]
        )
        self.network = nn.Sequential(*layers)
        self.output_channels = in_channels
        self.sequence_length = config.sequence_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != len(LEAD_ORDER):
            raise ValueError(f"Encoder expects (batch, 12, samples), got {tuple(x.shape)}")
        return self.network(x)


class ReconstructionDecoder(nn.Module):
    """Transposed-convolution decoder producing ``(batch, 12, 1000)``."""

    def __init__(self, config: UnifiedModelConfig) -> None:
        super().__init__()
        encoder_channels = list(config.encoder_channels)
        decoder_channels = list(config.decoder_channels or list(reversed(encoder_channels)))
        layers: List[nn.Module] = []
        in_channels = encoder_channels[-1]
        for out_channels in decoder_channels:
            layers.extend(
                [
                    nn.ConvTranspose1d(
                        in_channels,
                        out_channels,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.GELU(),
                ]
            )
            in_channels = out_channels
        layers.append(nn.Conv1d(in_channels, config.input_channels, kernel_size=3, padding=1))
        self.network = nn.Sequential(*layers)
        self.sequence_length = config.sequence_length

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        out = self.network(features)
        if out.shape[2] != self.sequence_length:
            out = torch.nn.functional.interpolate(
                out, size=self.sequence_length, mode="linear", align_corners=False
            )
        return out


class MultilabelClassifier(nn.Module):
    """Five independent condition logits from pooled shared features."""

    def __init__(self, config: UnifiedModelConfig) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(config.feature_dim, config.classifier_hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.classifier_hidden, len(TARGET_CATEGORIES)),
        )

    def forward(self, pooled_features: torch.Tensor) -> torch.Tensor:
        return self.network(pooled_features)


class UnifiedECGModel(nn.Module):
    """Shared encoder with reconstruction and multilabel classification heads."""

    architecture = MODEL_ARCHITECTURE

    def __init__(self, config: Optional[UnifiedModelConfig] = None) -> None:
        super().__init__()
        self.config = config or UnifiedModelConfig()
        self.encoder = SharedECGEncoder(self.config)
        self.decoder = ReconstructionDecoder(self.config)
        self.classifier = MultilabelClassifier(self.config)
        self.label_order = list(TARGET_CATEGORIES)

    def forward(self, x: torch.Tensor) -> UnifiedModelOutput:
        if x.ndim != 3 or x.shape[1] != self.config.input_channels:
            raise ValueError(
                f"Expected input shape (batch, {self.config.input_channels}, samples), got {tuple(x.shape)}"
            )
        features = self.encoder(x)
        reconstruction = self.decoder(features)
        pooled = torch.cat(
            [
                torch.mean(features, dim=2),
                torch.amax(features, dim=2),
            ],
            dim=1,
        )
        logits = self.classifier(pooled)
        return UnifiedModelOutput(
            reconstruction=reconstruction,
            classification_logits=logits,
            features=features,
            pooled_features=pooled,
        )

    def predict_scores(self, x: torch.Tensor) -> torch.Tensor:
        """Return five independent sigmoid scores."""
        return torch.sigmoid(self.forward(x).classification_logits)

    def reconstruction_error(self, x: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
        """Per-record mean squared reconstruction error in normalized units."""
        output = self.forward(x)
        squared = (output.reconstruction - x) ** 2
        if reduction == "mean":
            return squared.mean(dim=(1, 2))
        if reduction == "none":
            return squared
        raise ValueError("reduction must be 'mean' or 'none'")

    def task_parameters(self) -> Dict[str, List[nn.Parameter]]:
        return {
            "encoder": list(self.encoder.parameters()),
            "decoder": list(self.decoder.parameters()),
            "classifier": list(self.classifier.parameters()),
        }

    def freeze_encoder(self, freeze: bool = True) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad = not freeze


def create_model(config: Optional[UnifiedModelConfig] = None) -> UnifiedECGModel:
    return UnifiedECGModel(config)


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if (parameter.requires_grad or not trainable_only)
    )


def model_config_from_checkpoint(checkpoint: Dict[str, object]) -> UnifiedModelConfig:
    raw = checkpoint.get("model_config")
    if not isinstance(raw, dict):
        raise ValueError("Checkpoint has no model_config; it is not a unified PTB-XL checkpoint")
    return UnifiedModelConfig.from_dict(raw)
