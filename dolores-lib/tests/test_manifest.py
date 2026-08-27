from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from dolores.inference import (
    FORMAT_VERSION,
    IncompatibleBundleError,
    Manifest,
    read_manifest,
)


def test_manifest_roundtrips_through_json(manifest_dict: dict[str, Any]) -> None:
    manifest = Manifest.from_dict(manifest_dict)

    revived = Manifest.from_dict(json.loads(json.dumps(manifest.to_dict())))

    assert revived == manifest


def test_manifest_to_dict_is_json_serializable(manifest_dict: dict[str, Any]) -> None:
    json.dumps(Manifest.from_dict(manifest_dict).to_dict())


def test_manifest_preserves_feature_order(manifest_dict: dict[str, Any]) -> None:
    manifest = Manifest.from_dict(manifest_dict)

    assert manifest.feature_names == ["f0", "f1", "f2"]


def test_manifest_rejects_unknown_format_version(
    manifest_dict: dict[str, Any],
) -> None:
    manifest_dict["format_version"] = 99

    with pytest.raises(IncompatibleBundleError) as excinfo:
        Manifest.from_dict(manifest_dict)

    message = str(excinfo.value)
    assert "99" in message
    assert str(FORMAT_VERSION) in message


@pytest.mark.parametrize("field", ["features", "target", "runtime", "format_version"])
def test_manifest_rejects_missing_required_field(
    manifest_dict: dict[str, Any], field: str
) -> None:
    del manifest_dict[field]

    with pytest.raises(IncompatibleBundleError, match=field):
        Manifest.from_dict(manifest_dict)


def test_manifest_rejects_empty_features(manifest_dict: dict[str, Any]) -> None:
    manifest_dict["features"] = []

    with pytest.raises(IncompatibleBundleError, match="features"):
        Manifest.from_dict(manifest_dict)


def test_manifest_created_at_is_utc(manifest_dict: dict[str, Any]) -> None:
    manifest = Manifest.from_dict(manifest_dict)

    created_at = datetime.fromisoformat(manifest.created_at)
    assert created_at.utcoffset() is not None
    assert created_at.utcoffset().total_seconds() == 0  # type: ignore[union-attr]


def test_read_manifest_returns_the_manifest_save_bundle_wrote(
    bundle_path: Path, manifest_dict: dict[str, Any]
) -> None:
    assert read_manifest(bundle_path) == Manifest.from_dict(manifest_dict)
