"""Optional MLflow integration.

Ships behind the ``dolores-lib[tracking]`` extra and imports MLflow lazily, so
the library — and the inference server's container image — works without it.
Mandatory for the data scientist, absent from the serving path.

The tracking URI is always read from the environment. It is never hardcoded here.
"""
