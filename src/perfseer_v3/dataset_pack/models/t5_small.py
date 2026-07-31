"""T5-small teacher-forced factory."""
from .module_api import build
def build_model(**kwargs): return build("t5_small", **kwargs)
