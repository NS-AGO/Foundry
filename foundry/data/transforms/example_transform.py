import numpy as np
from torch_brain.data import Data


class ExampleTransform:

    def __init__(self, field: str = "ecog"):
        self.field = field

        # Parameters learned during fit()
        self.parameter = None

    def fit(self, dataset, split="train"):
        """Learn parameters from the dataset."""


    def __call__(self, data: Data) -> Data:
        """Apply the transform to a Data object."""

        return data


