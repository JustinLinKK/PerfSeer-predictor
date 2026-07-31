"""MLP-Mixer-S family factory."""
from .module_api import build
def build_model(**kwargs): return build("mlp_mixer_s", **kwargs)
