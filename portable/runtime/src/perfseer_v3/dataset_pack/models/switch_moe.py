"""Switch-style mixture-of-experts factory."""
from .module_api import build
def build_model(**kwargs): return build("switch_moe", **kwargs)
