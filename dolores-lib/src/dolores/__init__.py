"""Training and inference toolkit.

The version is recorded in every model bundle's manifest, so a served artifact
can always be traced back to the library that wrote it.
"""

from importlib.metadata import version

__version__ = version("dolores-lib")

__all__ = ["__version__"]
