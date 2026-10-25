"""DistilBERT teacher/student training-step factory."""
from .module_api import build
def build_model(**kwargs): return build("distilbert_distillation", **kwargs)
