"""fastText-style EmbeddingBag classifier factory."""
from .module_api import build
def build_model(**kwargs): return build("fasttext_embeddingbag", **kwargs)
