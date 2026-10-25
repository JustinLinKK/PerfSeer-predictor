"""Sequential/materialization composite block factory."""

from .base import _build


def build(block, candidate, generators):
    return _build(block, candidate, generators)
