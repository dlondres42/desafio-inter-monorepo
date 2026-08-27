"""The bundle manifest: what a served model says about itself.

The manifest is written beside the pickle, never inside it, so it can be read
without executing anything. That ordering is the point — a runtime-compatibility
check that required unpickling first would be circular.

Structural validation is pydantic's: required fields, their types, and which
element of which field is wrong. The checks that carry domain meaning — the format
version, the non-empty feature list, and the runtime comparison in
:mod:`dolores.inference._bundle` — stay hand-written, because their value is in the
message they produce, not in the fact that they fire.

Note that pydantic validates this manifest, a fixed eight-field header. It cannot
validate a *payload* against the manifest: the fields there are known only once a
bundle is loaded. That check lives in ``LoadedModel._validate_fields``.
"""

from __future__ import annotations

from typing import Any, Final, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from dolores.inference._errors import IncompatibleBundleError

FORMAT_VERSION: Final = 1
MANIFEST_NAME: Final = "manifest.json"
ESTIMATOR_NAME: Final = "estimator.joblib"

#: Compared between the bundle and the environment at load time. ``dolores-lib``
#: is recorded for provenance but deliberately not enforced: an older library
#: writing a format this one still reads is fine.
CHECKED_PACKAGES: Final = ("python", "scikit-learn", "numpy")


def normalise_dtype(dtype: Any) -> str:
    """Reduce a pandas dtype to one of four stable names.

    Raw dtype strings are not portable across pandas majors — a string column is
    ``object`` under pandas 2 and ``str`` under pandas 3 — so recording them
    verbatim would make a bundle written under one unloadable under the other.
    """
    name = str(dtype)
    if name.startswith("float"):
        return "float64"
    if name.startswith(("int", "uint", "Int", "UInt")):
        return "int64"
    if name.startswith(("bool", "Bool")):
        return "bool"
    return "string"


class Feature(BaseModel):
    """One input column: what it is called and what it holds."""

    model_config = ConfigDict(frozen=True)

    name: str
    dtype: str


class Target(BaseModel):
    """The predicted column, and every label the estimator can emit."""

    model_config = ConfigDict(frozen=True)

    name: str
    classes: tuple[str, ...]


class Manifest(BaseModel):
    """Everything needed to serve a model except the model.

    Unknown fields are ignored rather than rejected, so a newer writer can add
    one without breaking older readers; ``format_version`` is what gates real
    incompatibility.

    Attributes:
        format_version: Bumped only on a breaking change to this layout.
        model_id: Stable identifier, logged as an MLflow run tag so a served
            artifact can be traced back to the run that produced it.
        created_at: UTC ISO-8601 timestamp.
        estimator: Fully qualified class path of the fitted estimator.
        features: Input columns **in the order the estimator expects them**.
        target: The predicted column and its classes.
        runtime: Versions the bundle was written under.
        metrics: Optional held-out scores, carried for observability.
    """

    # `protected_namespaces=()` because `model_id` collides with pydantic's
    # reserved `model_` prefix. Renaming the field to appease the library would
    # be the tail wagging the dog — it is the model's id.
    model_config = ConfigDict(frozen=True, protected_namespaces=())

    format_version: int
    model_id: str
    created_at: str = ""
    estimator: str
    features: tuple[Feature, ...]
    target: Target
    runtime: dict[str, str]
    metrics: dict[str, float] = Field(default_factory=dict)

    @field_validator("format_version")
    @classmethod
    def _reject_unreadable_format(cls, value: int) -> int:
        if value != FORMAT_VERSION:
            raise ValueError(
                f"bundle manifest declares format_version {value}; "
                f"this library reads version {FORMAT_VERSION}"
            )
        return value

    @model_validator(mode="after")
    def _reject_empty_features(self) -> Self:
        # An "after" validator rather than Field(min_length=1): when an
        # individual feature is malformed, pydantic drops it and a length
        # constraint then fires a second, misleading "no features" error on top
        # of the real one. Running after item validation reports only the truth.
        if not self.features:
            raise ValueError("bundle manifest declares no features")
        return self

    @property
    def feature_names(self) -> list[str]:
        """Feature names in canonical order — how a payload must be laid out."""
        return [feature.name for feature in self.features]

    def to_dict(self) -> dict[str, Any]:
        """Render as JSON-serializable builtins."""
        return self.model_dump(mode="json")

    @classmethod
    def from_dict(cls, raw: Any) -> Self:
        """Parse and validate a manifest read off disk.

        Args:
            raw: Decoded ``manifest.json`` contents.

        Returns:
            The validated manifest.

        Raises:
            IncompatibleBundleError: If a required field is absent or ill-typed,
                the format version is not one this library reads, or the feature
                list is empty.
        """
        try:
            return cls.model_validate(raw)
        except ValidationError as exc:
            raise IncompatibleBundleError(_describe(exc)) from exc


def _describe(exc: ValidationError) -> str:
    """Flatten a pydantic ValidationError into one line naming every bad field.

    A caller holding a broken bundle wants the field names, not a traceback.
    """
    problems = [
        f"{'.'.join(str(part) for part in error['loc']) or 'manifest'}: {error['msg']}"
        for error in exc.errors()
    ]
    return f"bundle manifest is invalid — {'; '.join(problems)}"
