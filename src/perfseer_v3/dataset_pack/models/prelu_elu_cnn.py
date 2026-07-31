"""PReLU/ELU CNN family factory."""
from .module_api import build
def build_model(**kwargs): return build("prelu_elu_cnn", **kwargs)
