"""Mish-ResNet family factory."""
from .module_api import build
def build_model(**kwargs): return build("mish_resnet", **kwargs)
