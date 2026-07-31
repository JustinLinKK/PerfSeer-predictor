"""pix2pix joint generator/discriminator factory."""
from .module_api import build
def build_model(**kwargs): return build("pix2pix", **kwargs)
