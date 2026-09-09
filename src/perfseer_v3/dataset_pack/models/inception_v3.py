"""Inception-v3 family factory."""
from .module_api import build
def build_model(**kwargs): return build("inception_v3", **kwargs)
