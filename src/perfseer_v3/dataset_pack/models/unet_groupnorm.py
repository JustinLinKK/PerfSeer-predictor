"""GroupNorm U-Net family factory."""
from .module_api import build
def build_model(**kwargs): return build("unet_groupnorm", **kwargs)
