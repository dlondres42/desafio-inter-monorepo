# client_code

Two notebook projects over the same dataset, kept separate on purpose.

| Project | `dolores-lib` | What it is for |
| --- | --- | --- |
| [`baseline/`](baseline) | not installed | The control. Everything by hand: `read_csv`, `train_test_split`, a scoring loop, `joblib.dump`. This is what the work looks like without the library. |
| [`consumer/`](consumer) | published wheel from TestPyPI | The consumer proof. Same dataset, same result, progressively fewer hand-rolled lines as the library grows. |

Each is an independent uv project with its own lockfile. Neither is a package.

The split exists so the library's value is demonstrable rather than asserted — the
two notebooks are meant to be read side by side — and so that `consumer/` stays an
honest end-to-end test of the release pipeline. It installs `dolores-lib` from
TestPyPI, never as a path dependency, so a broken wheel breaks the notebook.

```bash
make mlflow-start                       # from the repo root, both notebooks need it
cd client_code/consumer && uv sync      # resolves dolores-lib from TestPyPI
uv run jupyter lab
```
