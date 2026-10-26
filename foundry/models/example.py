import torch
import torch.nn as nn
from torch_brain.batching import chain, pad8
from torch_brain.data import Data

import numpy as np

from foundry.models.readout import build_readout_router
from foundry.tasks.config import TaskConfig
from foundry.tasks.targets import extract_multitask_targets

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
        num_channels=32,
        hidden_dim=128,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.task_configs = TaskConfig.normalize_task_configs(task_configs)

        self.num_channels = num_channels

        self.encoder = nn.Sequential(
            nn.Linear(num_channels, hidden_dim),
            nn.ReLU(),
        )

        # Build prediction heads from the task configs so the model does not
        # need to hard-code the task's output dimension.
        self.router = build_readout_router(
            self.task_configs,
            hidden_dim,
        )

    # Foundry attaches model.tokenize to the dataset's transform pipeline.
    # It converts each sampled Data object into the inputs, targets, and
    # metadata expected by the training module.
    def tokenize(self, data: Data) -> dict:
        """
        Prepare channel means, labels, and metadata for batching.
        """

        signal = np.asarray(data.ecog.data, dtype=np.float32)
        if signal.ndim != 2 or signal.shape[1] != self.num_channels:
            raise ValueError(
                f"Expected signal shape (time, {self.num_channels}) instead"
                f"got {signal.shape}."
            )

        if signal.shape[0] == 0:
            raise ValueError("Cannot tokenize an empty window.")

        _, values, task_index, weights = extract_multitask_targets(self.task_configs, data)

        # Foundry's training module separates this dictionary into model inputs,
        # loss targets, and metadata. These keys form its expected batch interface.
        return {
            "input_values": torch.from_numpy(signal.mean(axis=0)),
            "task_index": pad8(task_index),
            "target_values": chain(values, allow_missing_keys=True),
            "target_weights": chain(weights, allow_missing_keys=True),
            "session_id": data.session.id,
            "absolute_start": float(data.absolute_start),
        }
    
    def forward(
        self,
        *,
        input_values: torch.Tensor,
        task_index: torch.Tensor,
        **kwargs,
    ):
        """
        Compute predictions from neural activity.
        """

        # Compute neural representation.
        x = self.encoder(input_values)

        # Reuse the window representation for each target in that window.
        # Shape: (batch, hidden_dim) -> (batch, num_targets, hidden_dim).   
        x = x.unsqueeze(1).expand(-1, task_index.shape[1], -1)
        flat_tasks = task_index.reshape(-1)
        valid = flat_tasks > 0

        # Foundry expects predictions keyed by task name so it can apply each
        # task's configured loss and metrics.
        return self.router(
            x.reshape(-1, x.shape[-1])[valid], 
            (flat_tasks[valid] - 1).long(),
            )