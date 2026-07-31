"""EfficientNet B0/B4 family factory."""
from .module_api import build
def build_model(**kwargs): return build("efficientnet_b0_b4", **kwargs)
