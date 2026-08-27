"""The failure vocabulary of the serving path.

Two families, and the split is what the inference server maps onto HTTP status
codes: a :class:`BundleError` means the artifact itself is unusable and the
process should never have started, while a :class:`FeatureValidationError` means
one request was malformed and the next one may be fine.
"""


class BundleError(Exception):
    """A model bundle cannot be used. Raised at load time, never per request."""


class IncompatibleBundleError(BundleError):
    """The manifest is malformed, or written in a format version we cannot read."""


class RuntimeMismatchError(BundleError):
    """The bundle was written under library versions this environment does not have."""


class SchemaMismatchError(BundleError):
    """An estimator and the frame it was fit on disagree. Raised when writing."""


class FeatureValidationError(ValueError):
    """A prediction payload does not match the manifest's feature schema.

    The API turns this into a 422: the fields named in the message are exactly
    what the caller needs to fix.
    """
