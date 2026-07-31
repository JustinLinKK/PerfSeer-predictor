"""MobileNetV3-Large family factory."""
from .module_api import build
def build_model(**kwargs): return build("mobilenet_v3_large", **kwargs)
