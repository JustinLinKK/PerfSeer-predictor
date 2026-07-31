"""Llama-style small decoder factory."""
from .module_api import build
def build_model(**kwargs): return build("llama_small", **kwargs)
