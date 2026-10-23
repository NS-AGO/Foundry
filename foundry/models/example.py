import torch
import torch.nn as nn
from torch_brain.data import Data

import numpy as np

from foundry.models.readout import build_readout_router
from torch_brain.batching import chain, pad8, pad2d

class ExampleMLP(nn.Module):
    """
    A simple MLP model for neural signal decoding.

    The model first maps the input neural activity into a hidden
    representation, then uses task-specific readouts to produce
    predictions.
    """

    def __init__(
        self,
        task_configs,
        num_channels=64,
        hidden_dim=128,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.task_configs = task_configs

        self.num_channels = num_channels

        self.encoder = nn.Sequential(
            nn.Linear(num_channels, hidden_dim),
            nn.ReLU(),
        )

        self.router = build_readout_router(
            task_configs,
            hidden_dim,
        )

    def forward(
        self,
        *,
        input_values: torch.Tensor,
        task_index: torch.Tensor,
        **kwargs,
    ):
        """
        Compute predictions from neural activity.

        Args:
            input_values:
                Neural input with shape (..., num_channels).

            task_index:
                Identifies which task-specific readout should be used.

            **kwargs:
                Additional inputs passed by the training framework.

        Returns:
            Task-specific model predictions.
        """

        # Compute neural representation.
        x = self.encoder(input_values)

        return self._route_readout(x, task_index)

    def tokenize(self, data: Data) -> dict[str, torch.Tensor]:
        """
        Converts a TemporalData EEG/ECoG sample to model-ready tensors and multitask readout targets.

        When channel_strategy is set, applies prepare_pretokenize to handle heterogeneous electrode counts.

        Args:
            data (torch_brain.data.Data): Input data structure containing an "eeg" or "ecog" field
                along with "channels" and task-specific label fields.

        Returns:
            dict: {
                "input_values" (torch.Tensor): Model input of shape (T, C) or padded/projected,
                "task_index" (torch.Tensor): Target output decoder indices,
                "target_values" (dict[str, torch.Tensor] or similar): Multitask target values,
                "target_weights" (dict[str, torch.Tensor] or similar): Multitask target weights,
                "session_id" (Any): Session identifier,
                "absolute_start" (Any): Absolute segment start time,
                "input_mask" (torch.Tensor, optional): Mask for real channels (if strategy set),
                "input_session_ids" (str, optional): Session ID for strategy (if strategy set),
                "input_channel_counts" (torch.Tensor, optional): Actual channel count per sample (if strategy set),
                "input_seq_len" (torch.Tensor, optional): Actual sequence length (if strategy set),
            }

        Note:
            The tokenized data will retain the same tensor dimensions and layout as present in `data`.
            Input shape normalization and conversion (e.g., unsqueezing or channel placement for Conv1d/Conv2d)
            is handled by the forward model methods, not at tokenization time.

        """
        has_eeg = hasattr(data, "eeg") and data.eeg is not None
        has_ecog = hasattr(data, "ecog") and data.ecog is not None
        has_seeg = hasattr(data, "seeg") and data.seeg is not None

        if not has_eeg and not has_ecog and not has_seeg:
            raise ValueError(
                "Data must have an 'eeg', 'ecog', or 'seeg' channel type"
            )

        if has_eeg:
            signal = data.eeg.signal
            default_type = "EEG"
        elif has_ecog:
            signal = data.ecog.signal
            default_type = "ECOG"
        elif has_seeg:
            signal = data.seeg.signal
            default_type = "SEEG"
        else:
            raise ValueError(f"Data must have an '{default_type}' channel type")

        modality_field = (
            data.channels.type.astype(str)
            if hasattr(data.channels, "type")
            else np.array([default_type] * len(data.channels)).astype(str)
        )
        modality_mask = np.isin(
            np.char.lower(modality_field), list(self.SUPPORTED_MODALITIES)
        )

        signal = np.asarray(signal, dtype=np.float32)
        signal_filtered = signal[:, modality_mask]

        output_values, output_task_index, output_weights = (
            self._extract_targets(data)
        )

        result = {
            "task_index": pad8(output_task_index),
            "target_values": chain(output_values, allow_missing_keys=True),
            "target_weights": chain(output_weights, allow_missing_keys=True),
            "session_id": data.session.id,
            "absolute_start": float(data.absolute_start),
        }

        # Apply channel strategy if present
        if self.channel_strategy is not None:
            # Get sampling rate (assume it exists on the signal object)
            if hasattr(
                data.eeg if has_eeg else (data.ecog if has_ecog else data.seeg),
                "sampling_rate",
            ):
                sampling_rate = (
                    data.eeg
                    if has_eeg
                    else (data.ecog if has_ecog else data.seeg)
                ).sampling_rate
            else:
                sampling_rate = 1.0  # fallback

            # Create dummy channel tokens (baselines don't use channel vocab)
            dummy_channel_tokens = np.arange(
                signal_filtered.shape[1], dtype=np.int64
            )

            # Apply strategy preparation (padding/projection setup)
            strategy_result = self.channel_strategy.prepare_pretokenize(
                signal_filtered,
                dummy_channel_tokens,
                sampling_rate,
            )
            result.update(strategy_result)
            result["input_values"] = pad2d(result["input_values"])
            # SessionSpatialProjector looks up per-session layers by this key.
            # Training pops ``session_id`` before the forward pass, so the
            # projector id must live on its own collated field (same as POYO).
            result["input_session_ids"] = str(data.session.id)
        else:
            # Standard path: no channel strategy
            result["input_values"] = pad2d(torch.from_numpy(signal_filtered))

        return result


    def _route_readout(self, x, task_index):
        batch_size, n_out, dim = x.shape

        flat_embs = x.reshape(batch_size * n_out, dim)
        flat_task_index = task_index.reshape(batch_size * n_out)

        valid = flat_task_index > 0

        return self.router(
            flat_embs[valid],
            (flat_task_index[valid] - 1).long(),
        )