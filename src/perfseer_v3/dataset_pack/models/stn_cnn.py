"""Spatial-transformer CNN family factory."""
from .module_api import build
def build_model(**kwargs): return build("stn_cnn", **kwargs)
