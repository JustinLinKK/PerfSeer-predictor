"""Temporal convolutional network factory."""
from .module_api import build
def build_model(**kwargs): return build("temporal_convolutional_network", **kwargs)
