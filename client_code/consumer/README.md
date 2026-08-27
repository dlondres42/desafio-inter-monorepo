# consumer

The same workflow as `../baseline`, run through `dolores-lib`.

`dolores-lib` is installed here as the **published wheel from TestPyPI**, never as a
path dependency. `uv.lock` pins it to
`source = { registry = "https://test.pypi.org/simple/" }`, which makes running
`iris_pipeline.ipynb` an end-to-end assertion that the release pipeline worked: if
the published wheel is broken, this notebook fails.

The routing is done by `explicit = true` on the TestPyPI index in `pyproject.toml`,
so that index is consulted *only* for `dolores-lib`. pandas, scikit-learn, mlflow and
every transitive dependency still resolve from the default PyPI — verifiable in
`uv.lock`. The alternative, `extra-index-url`, makes resolution order-dependent
across every package and is the dependency-confusion footgun that `explicit` avoids.

```bash
uv sync
uv run jupyter lab
```

Cells still marked **"Still hand-rolled"** are the ones that collapse as the library
grows — training at `v0.4.0`, serialization and serving at `v0.2.0`. The shrinking
diff against `../baseline/iris_eda.ipynb` is the point of this project.

## Iterating on an unreleased library

To point this at your working tree instead of the published wheel:

```bash
uv pip install -e ../../dolores-lib     # venv only; pyproject and uv.lock untouched
```

Any later `uv sync` prunes that editable install and restores the published wheel.
That is the intended behaviour — the declared state of this project is always the
released artifact.
