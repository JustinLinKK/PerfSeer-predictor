"""M5 raw-waveform CNN factory."""
from .module_api import build
def build_model(**kwargs): return build("m5_waveform_cnn", **kwargs)
