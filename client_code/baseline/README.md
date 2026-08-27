# baseline

The control notebook. No `dolores-lib` — `iris_eda.ipynb` does everything by hand:
`pd.read_csv` and a manual column drop, `train_test_split`, a loop that fits three
sklearn candidates and computes four metrics per run, then `joblib.dump` of the
winner and a cell that pretends to be the inference server.

It is kept working, and kept free of the library, so that `../consumer` has
something to be compared against. Read the two side by side.

```bash
uv sync && uv run jupyter lab
```

The MLflow server comes from the repo root: `make mlflow-start`.
