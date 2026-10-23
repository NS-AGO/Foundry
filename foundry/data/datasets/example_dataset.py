from pathlib import Path
from typing import Callable, Literal, Optional

from torch_brain.datasets import Dataset


class ExampleDataset(Dataset):

    def __init__(
        self,
        root: str,
        recording_ids: Optional[list[str]] = None,
        transform: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__(
            dataset_dir=Path(root) / "neurosoft_monkeys_2026",
            recording_ids=recording_ids,
            transform=transform,
            namespace_attributes=["session.id", "subject.id"],
            **kwargs,
        )

    def get_sampling_intervals(
        self,
        split: Optional[Literal["train", "valid", "test"]] = None,
    ):
        # No split: the full recording. Otherwise: the train/valid/test
        # intervals that the pipeline stored on each Data object.
        if split not in (None, "train", "valid", "test"):
            raise ValueError("split must be 'train', 'valid', 'test', or None.")

        key = "domain" if split is None else f"{split}_domain"
        return {rid: getattr(self.get_recording(rid), key) for rid in self.recording_ids}