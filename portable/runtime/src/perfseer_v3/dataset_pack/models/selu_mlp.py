"""SELU self-normalizing MLP factory."""
from .module_api import build
def build_model(**kwargs): return build("selu_mlp", **kwargs)
