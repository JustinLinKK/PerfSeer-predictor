"""ViT-S/16 family factory."""
from .module_api import build
def build_model(**kwargs): return build("vit_s16", **kwargs)
