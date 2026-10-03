"""Training-only features with the unchanged v3.2 numeric layout."""

from dataclasses import dataclass

from perfseer_v3.features import apply_normalization
from perfseer_v32.features import _build_features

from .capture import graph_from_design
from .version import FEATURE_VERSION, NUMERIC_FEATURE_VERSION


@dataclass(frozen=True)
class TrainingFeatures:
    training: object
    version: str = FEATURE_VERSION

    @property
    def layout(self):
        return self.training.layout

    def validate(self):
        self.training.validate()
        if self.version != FEATURE_VERSION or self.layout.feature_schema_version != NUMERIC_FEATURE_VERSION:
            raise ValueError("v4 training feature contract differs")


def build_features(design):
    result = TrainingFeatures(_build_features(graph_from_design(design)))
    result.validate()
    return result


def normalize_features(features, normalization):
    features.validate()
    result = TrainingFeatures(apply_normalization(features.training, normalization.training))
    result.validate()
    return result
