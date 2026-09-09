"""ResNet-50 family factory."""
from .module_api import build
def build_model(**kwargs): return build("resnet50", **kwargs)
