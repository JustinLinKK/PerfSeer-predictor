"""BERT-base fine-tuning factory."""
from .module_api import build
def build_model(**kwargs): return build("bert_base", **kwargs)
