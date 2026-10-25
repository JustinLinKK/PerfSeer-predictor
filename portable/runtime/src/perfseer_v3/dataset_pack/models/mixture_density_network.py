"""Mixture density network factory."""
from .module_api import build
def build_model(**kwargs): return build("mixture_density_network", **kwargs)
