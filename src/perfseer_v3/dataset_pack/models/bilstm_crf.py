"""BiLSTM-CRF family factory."""
from .module_api import build
def build_model(**kwargs): return build("bilstm_crf", **kwargs)
