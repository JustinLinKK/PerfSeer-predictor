"""dense_matrix operation benchmark routing."""

from ..operation_sampler import OperationCandidate, OperationGeneratorSpec
from .base import BenchmarkRecipe
from .recipes import build_family_benchmark


def build(generator: OperationGeneratorSpec, candidate: OperationCandidate) -> BenchmarkRecipe:
    return build_family_benchmark("dense_matrix", generator, candidate)
