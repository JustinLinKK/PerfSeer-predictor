"""Multi-head latent-attention mini-transformer factory."""
from .module_api import build
def build_model(**kwargs): return build("mla_mini_transformer", **kwargs)
