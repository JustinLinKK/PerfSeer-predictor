"""GRU/RNN teacher-forced seq2seq factory."""
from .module_api import build
def build_model(**kwargs): return build("gru_rnn_seq2seq", **kwargs)
