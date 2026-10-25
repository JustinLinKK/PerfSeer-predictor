"""GPT-2-small fine-tuning factory."""
from .module_api import build
def build_model(**kwargs): return build("gpt2_small", **kwargs)
