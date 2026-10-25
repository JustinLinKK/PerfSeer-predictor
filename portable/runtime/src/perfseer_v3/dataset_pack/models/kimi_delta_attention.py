"""Kimi-style delta-attention block factory."""
from .module_api import build
def build_model(**kwargs): return build("kimi_delta_attention", **kwargs)
