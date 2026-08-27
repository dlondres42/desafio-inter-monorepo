"""Load a serialized model and score payloads against it.

The entire import surface of the inference server. Deliberately separate from
:mod:`dolores.experiment`: a different consumer with a different lifecycle, and
keeping it standalone is what lets the server's import stay one name wide.

Never talks to MLflow. It reads a portable artifact and predicts.

A bundle is a zip of two members — ``estimator.joblib`` and ``manifest.json`` —
kept as siblings so the manifest can be read, and an unusable bundle rejected,
without executing the pickle.

The manifest guarantees *structural* compatibility: feature names, their order,
their dtypes, and the classes the estimator can emit. It guarantees nothing
semantic. It cannot tell millimetres from centimetres, and nothing at the serving
boundary can — that is a data-quality concern and explicitly out of scope.
"""

from dolores.inference._bundle import (
    LoadedModel,
    Prediction,
    Predictor,
    load_model,
    read_manifest,
    save_bundle,
)
from dolores.inference._errors import (
    BundleError,
    FeatureValidationError,
    IncompatibleBundleError,
    RuntimeMismatchError,
    SchemaMismatchError,
)
from dolores.inference._manifest import (
    ESTIMATOR_NAME,
    FORMAT_VERSION,
    MANIFEST_NAME,
    Feature,
    Manifest,
    Target,
)

__all__ = [
    "ESTIMATOR_NAME",
    "FORMAT_VERSION",
    "MANIFEST_NAME",
    "BundleError",
    "Feature",
    "FeatureValidationError",
    "IncompatibleBundleError",
    "LoadedModel",
    "Manifest",
    "Predictor",
    "Prediction",
    "RuntimeMismatchError",
    "SchemaMismatchError",
    "Target",
    "load_model",
    "read_manifest",
    "save_bundle",
]
