from typing import Callable, Optional, Type

from torch_brain.transforms import Compose

from foundry.data.datamodules.base import NeuralDataModule
from foundry.data.datasets import ExampleDataset
from foundry.data.transforms import (
    ExampleTransform,
)


class ExampleDataModule(NeuralDataModule):

    def __init__(
        self,
        root: str,
        dataset_class,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = False,
        sequence_length: Optional[float] = None,
        transforms: Optional[list[Callable]] = None,
        tokenizer: Optional[Callable] = None,
        seed: int = 42,
        recording_ids: Optional[list[str]] = None,
        **kwargs,
    ):
        # dataset_kwargs = {
        #     **(dataset_kwargs or {}),
        #     "recording_ids": recording_ids,
        # }

        super().__init__(
            dataset_class=dataset_class,
            root=root,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            sequence_length=sequence_length,
            transforms=transforms,
            tokenizer=tokenizer,
            seed=seed,
            recording_ids=recording_ids,
            **kwargs,
        )

    def setup(self, stage=None):
        super().setup(stage)

        exampletransform = ExampleTransform(
            field="ecog",
        )

        # Fit transforms that require training data.
        exampletransform.fit(
            self.dataset,
            split="train",
        )

        existing = (
            list(self.dataset.transform.transforms)
            if self.dataset.transform is not None
            else []
        )

        self.dataset.transform = Compose(
            [
                exampletransform,
            ]
            + existing
        )

    def get_recording_ids(self) -> list[str]:
        return sorted(self.dataset.recording_ids)

    def get_channel_ids(self) -> list[str]:
        return sorted(self.dataset.get_channel_ids())