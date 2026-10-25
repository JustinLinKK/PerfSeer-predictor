"""PANNs CNN14 log-mel-style audio factory."""
from .module_api import build
def build_model(**kwargs): return build("panns_cnn14", **kwargs)
