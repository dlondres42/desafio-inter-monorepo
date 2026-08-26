from importlib.metadata import version

import dolores


def test_distribution_is_installed():
    assert version("dolores-lib")


def test_version_is_exported():
    assert dolores.__version__ == version("dolores-lib")
