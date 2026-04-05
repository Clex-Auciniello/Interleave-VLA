"""TFDS entry point for ``tfds build`` — wraps VIMADataset with the class name
that TFDS expects based on the directory name ``vima_interleave``."""

from .vima_dataset_builder import VIMADataset


class VIMAInterleave(VIMADataset):
    """Thin subclass so TFDS discovers the builder by directory name."""
    pass
